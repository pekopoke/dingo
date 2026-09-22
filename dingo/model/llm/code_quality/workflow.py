"""Small SDK helpers for deterministic candidates and isolated model configs."""

from dingo.config.input_args import EvaluatorLLMArgs
from dingo.io.output.eval_detail import EvalDetail
from dingo.model.llm.code_quality.schema import RULE_NAMES

DEFAULT_QUALITY_MODEL = 'bailian/deepseek-v4.1-flash'
DEFAULT_CLASSIFICATION_MODELS = ('glm-5.3-flash', DEFAULT_QUALITY_MODEL)


def configured_evaluator(evaluator, config):
    """BaseOpenAI uses classmethods: a subclass isolates each model/client safely."""
    if isinstance(config, dict):
        config = EvaluatorLLMArgs(**config)
    return type(evaluator.__name__, (evaluator,), {
        'dynamic_config': config.model_copy(deep=True), 'client': None,
    })


def run_code_rules(input_data):
    """Run the existing rules on a copy; failures stay separate from hits."""
    from dingo.model.rule import rule_common

    results = []
    for name in RULE_NAMES:
        rule = getattr(rule_common, name)
        try:
            results.append(rule.eval(input_data.model_copy(deep=True)))
        except Exception:
            results.append(EvalDetail(
                metric=name, status=False, applicable=False, not_applicable_kind='execution_error',
                label=['REVIEW_EXECUTION_ERROR.RuleFailed'], reason=['Rule execution failed'],
            ))
    return results


def rule_candidates(results):
    # Raw detector reasons may include document snippets. The reviewing LLM sees
    # the source already; transmit only detector identity/labels as candidate hints.
    return [{'metric': result.metric, 'label': result.label}
            for result in results if result.applicable and result.status]


def classification_consensus(results):
    """Both classifiers must succeed; their unrounded mean <=2 indicates low content."""
    scores = [item.score if item.applicable else None for item in results]
    if any(score is not None and (isinstance(score, bool) or score not in range(6)) for score in scores):
        raise ValueError('Classification scores must be integers from 0 to 5')
    complete = len(scores) == 2 and all(score is not None for score in scores)
    if not complete:
        return {'positive': None, 'low_code_content': None, 'scores': scores, 'average_score': None,
                'review_required': True, 'execution_error': True, 'threshold_disagreement': None}
    average = sum(scores) / 2
    low = average <= 2
    positive = all(score >= 4 for score in scores)
    return {
        'positive': positive, 'low_code_content': low, 'scores': scores, 'average_score': average,
        'threshold_disagreement': (scores[0] >= 4) != (scores[1] >= 4),
        'review_priority': 'high' if low else 'normal',
        'review_required': not positive, 'execution_error': False,
    }
