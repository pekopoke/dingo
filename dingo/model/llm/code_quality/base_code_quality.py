"""Shared input handling, response parsing and validation for code evaluators.

Concrete evaluators supply prompts and metadata, following BaseTextQuality's
separation of judging policy from result processing. API calls remain in BaseOpenAI.
"""

import hashlib
import json

from pydantic import Field, ValidationError

from dingo.io.input import Data, RequiredField
from dingo.io.output.eval_detail import EvalDetail
from dingo.model.llm.base_openai import BaseOpenAI
from dingo.model.llm.code_quality.schema import RULE_NAMES, Classification, CodeQualityResponse
from dingo.utils.exception import ConvertJsonError


def parse_response(response, schema):
    """Reject incomplete, contradictory or invented outputs without logging payloads."""
    try:
        text = response.strip()
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        return schema.model_validate(json.loads(text))
    except (AttributeError, TypeError, ValueError, ValidationError):
        # Pydantic/JSON exceptions may contain the complete model response, including secrets.
        raise ConvertJsonError('Invalid code-quality response: JSON/schema consistency check failed.') from None


def execution_error(metric, code):
    return CodeQualityDetail(
        metric=metric, status=False, applicable=False, not_applicable_kind='execution_error',
        score=None, label=[f'REVIEW_EXECUTION_ERROR.{code}'], reason=[code],
    )


class CodeQualityDetail(EvalDetail):
    """Expose all findings to Executor; retain the primary decision in details."""

    details: dict = Field(default_factory=dict)


class BaseCodeEvaluation(BaseOpenAI):
    _required_fields = [RequiredField.CONTENT]

    @classmethod
    def build_messages(cls, input_data: Data):
        if not isinstance(getattr(input_data, 'content', None), str):
            raise ValueError('Code evaluation requires string content; input is not coerced or rewritten.')
        return [
            {'role': 'system', 'content': cls.prompt},
            {'role': 'user', 'content': json.dumps({'content': input_data.content}, ensure_ascii=False)},
        ]

    @classmethod
    def eval(cls, input_data: Data):
        if not isinstance(getattr(input_data, 'content', None), str):
            return execution_error(cls.__name__, 'MissingOrNonStringContent')
        try:
            result = super().eval(input_data)
        except Exception as exc:
            # Client construction/message building happen outside BaseOpenAI's
            # retry block. Do not expose exception text that may contain inputs.
            result = execution_error(cls.__name__, type(exc).__name__)
        if not result.applicable:
            result.reason = ['Code evaluation failed; see the execution-error label.']
            result.rubric_version = cls.rubric_version()
        return result

    @classmethod
    def rubric_version(cls):
        return f'{cls.__name__}:sha256:{hashlib.sha256(cls.prompt.encode("utf-8")).hexdigest()}'


class BaseCodeClassification(BaseCodeEvaluation):
    """Convert independent 0-5 relevance scores to standard evaluation results."""

    @classmethod
    def process_response(cls, response):
        result = parse_response(response, Classification)
        return CodeQualityDetail(
            metric=cls.__name__, status=result.score <= 2, score=result.score,
            label=['Effectiveness.Low_Code_Content'] if result.score <= 2 else ['QUALITY_GOOD'],
            reason=[result.reason],
            details={**result.model_dump(), 'positive': result.score >= 4,
                     'review_priority': 'high' if result.score <= 2 else 'normal',
                     'review_required': result.score <= 2},
            rubric_version=cls.rubric_version(),
        )


class BaseCodeQuality(BaseCodeEvaluation):
    """Parse quality decisions and validate multi-label evidence and rule reviews."""

    @classmethod
    def build_messages(cls, input_data: Data):
        messages = super().build_messages(input_data)
        candidates = getattr(input_data, 'rule_candidates', [])
        if not isinstance(candidates, list):
            raise ValueError('rule_candidates must be a list')
        metrics = [item.get('metric') for item in candidates if isinstance(item, dict)]
        if len(metrics) != len(candidates) or len(set(metrics)) != len(metrics) or not set(metrics) <= set(RULE_NAMES):
            raise ValueError('rule_candidates contains unknown or duplicate rules')
        messages[1]['content'] = json.dumps(
            {'content': input_data.content, 'rule_candidates': candidates}, ensure_ascii=False,
        )
        return messages

    @classmethod
    def process_response(cls, response):
        parsed = parse_response(response, CodeQualityResponse)
        return CodeQualityDetail(
            metric=cls.__name__, status=bool(parsed.findings), score=parsed.score,
            label=[f'{item.type}.{item.name}' for item in parsed.findings] if parsed.findings else ['QUALITY_GOOD'],
            reason=[item.reason for item in parsed.findings] if parsed.findings else [parsed.reason],
            details={**parsed.model_dump(), 'review_required': bool(parsed.findings),
                     'all_labels': [f'{item.type}.{item.name}' for item in parsed.findings],
                     'classification_positive': parsed.classification.score >= 4,
                     'classification_review_priority': 'high' if parsed.classification.score <= 2 else 'normal'},
            rubric_version=cls.rubric_version(),
        )

    @classmethod
    def eval(cls, input_data: Data):
        # Executor supplies just content. Run preliminary rules here so a normal
        # registered evaluator includes contextual review without executor coupling.
        from dingo.model.llm.code_quality.workflow import rule_candidates, run_code_rules

        if not isinstance(getattr(input_data, 'content', None), str):
            return execution_error(cls.__name__, 'MissingOrNonStringContent')
        rule_results = None
        if not hasattr(input_data, 'rule_candidates'):
            rule_results = run_code_rules(input_data)
            if any(not item.applicable for item in rule_results):
                error = execution_error(cls.__name__, 'RuleFailed')
                error.details['rules'] = [
                    {'metric': item.metric, 'status': item.status, 'applicable': item.applicable,
                     'label': item.label} for item in rule_results
                ]
                return error
            input_data = input_data.model_copy(update={'rule_candidates': rule_candidates(rule_results)})
        # Validate before API calls; malformed supplied candidates are execution errors.
        try:
            cls.build_messages(input_data)
        except (ValueError, TypeError):
            return execution_error(cls.__name__, 'InvalidRuleCandidates')
        result = super().eval(input_data)
        if not result.applicable:
            return result
        payload = result.details
        if rule_results is not None:
            payload['rules'] = [
                {'metric': item.metric, 'status': item.status, 'applicable': item.applicable,
                 'label': item.label} for item in rule_results
            ]
        expected = {item['metric'] for item in getattr(input_data, 'rule_candidates', [])}
        if {item['metric'] for item in payload['rule_reviews']} != expected:
            error = execution_error(cls.__name__, 'IncompleteRuleReview')
            error.usage = result.usage
            return error
        line_count = len(input_data.content.split('\n'))
        if any(item['line_end'] is not None and item['line_end'] > line_count for item in payload['findings']):
            error = execution_error(cls.__name__, 'EvidenceLineOutOfRange')
            error.usage = result.usage
            return error
        return result
