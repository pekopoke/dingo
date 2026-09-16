"""Strict model-output contract; invalid results are execution errors, not passes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Keep existing IDs for retained categories, but restrict their allowed scope.
CODE_COMPONENTS = (
    'code_fence_block_boundary_corruption',
    'truncated_or_missing_code',
    'invalid_code_syntax_or_semantics',
)
MIXED = 'mixed_multiple_corruptions'
SYNTAX_SUBTYPES = (
    'syntax_delimiter_parser_error',
    'cross_language_transpilation_artifact',
)
RULE_NAMES = (
    'RuleContentNull', 'RuleContentShort', 'RuleSpecialCharacter', 'RuleAbnormalChar',
    'RuleSpaceMore', 'RuleOnlyUrl', 'RuleLoremIpsum', 'RuleDocRepeat', 'RulePIIDetection',
    'RuleHtmlEntity', 'RuleHtmlTag',
)
# Public two-level issue taxonomy. Code-error fields below are auxiliary details.
LABELS = {
    'Effectiveness': {'Empty_Content', 'Insufficient_Content', 'Special_Characters',
                      'Abnormal_Characters', 'HTML_Markup', 'Only_URL',
                      'Placeholder_Content', 'Code_Whitespace', 'Redundant_Language_Label',
                      'Fence_Language_Mismatch', 'Syntax_Error', 'Cross_Language_Mixing', 'Low_Code_Content'},
    'Completeness': {'Code_Truncation'},
    'Similarity': {'Document_Repetition'},
    'Security': {'PII_Exposure', 'Secret_Credentials', 'Internal_Endpoint_Exposure',
                 'Porn', 'Violent', 'Gamble', 'Drug', 'Politics'},
}
FENCE_LABELS = {('Effectiveness', 'Redundant_Language_Label'), ('Effectiveness', 'Fence_Language_Mismatch')}
SYNTAX_LABELS = {
    'syntax_delimiter_parser_error': {('Effectiveness', 'Syntax_Error'), ('Effectiveness', 'Code_Whitespace')},
    'cross_language_transpilation_artifact': {('Effectiveness', 'Cross_Language_Mixing')},
}
RULE_LABELS = {
    'RuleHtmlEntity': {('Effectiveness', 'HTML_Markup')},
    'RuleHtmlTag': {('Effectiveness', 'HTML_Markup')},
    'RuleContentNull': {('Effectiveness', 'Empty_Content')},
    'RuleContentShort': {('Effectiveness', 'Insufficient_Content')},
    'RuleSpecialCharacter': {('Effectiveness', 'Special_Characters'), ('Effectiveness', 'Syntax_Error')},
    # RuleAbnormalChar includes RuleSpecialCharacter, so one finding may explain both hits.
    'RuleAbnormalChar': {('Effectiveness', 'Special_Characters'), ('Effectiveness', 'Abnormal_Characters'),
                         ('Effectiveness', 'Syntax_Error')},
    'RuleSpaceMore': {('Effectiveness', 'Code_Whitespace')},
    'RuleOnlyUrl': {('Effectiveness', 'Only_URL')},
    'RuleLoremIpsum': {('Effectiveness', 'Placeholder_Content')},
    'RuleDocRepeat': {('Similarity', 'Document_Repetition')},
    'RulePIIDetection': {('Security', 'PII_Exposure'), ('Security', 'Internal_Endpoint_Exposure'),
                         ('Security', 'Secret_Credentials')},
}


class StrictResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)


class Classification(StrictResponse):
    score: int = Field(ge=0, le=5)
    contains_code: bool
    reason: str = Field(min_length=1)


class Finding(StrictResponse):
    type: str
    name: str
    reason: str = Field(min_length=1)
    line_start: int | None = Field(ge=1)
    line_end: int | None = Field(ge=1)

    @model_validator(mode='after')
    def validate_finding(self):
        if self.name not in LABELS.get(self.type, set()):
            raise ValueError('Unknown finding type/name pair')
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError('Evidence must give both line bounds or neither')
        if self.line_start is not None and self.line_end < self.line_start:
            raise ValueError('Evidence line range is reversed')
        return self


class CodeError(StrictResponse):
    primary: str | None
    tags: list[str]
    syntax_subtypes: list[str]

    @model_validator(mode='after')
    def validate_taxonomy(self):
        if len(self.tags) != len(set(self.tags)) or not set(self.tags) <= set(CODE_COMPONENTS):
            raise ValueError('Invalid or duplicate code component tags')
        if len(self.syntax_subtypes) != len(set(self.syntax_subtypes)) or not set(self.syntax_subtypes) <= set(SYNTAX_SUBTYPES):
            raise ValueError('Invalid or duplicate syntax subtypes')
        if not self.tags:
            if self.primary is not None or self.syntax_subtypes:
                raise ValueError('No code defects must have null primary and empty subtypes')
        elif self.primary == MIXED:
            if len(self.tags) < 2:
                raise ValueError('Mixed corruption requires at least two components')
        elif self.primary not in self.tags:
            raise ValueError('Code primary must be a supported component')
        has_syntax = 'invalid_code_syntax_or_semantics' in self.tags
        if has_syntax != bool(self.syntax_subtypes):
            raise ValueError('Syntax component and subtype presence must agree')
        return self


class Politics(StrictResponse):
    hurt_socialist_regime_system: Literal['neg', 'pos', 'none']
    hurt_china_interests_image: Literal['neg', 'pos', 'none']
    hurt_china_unity_stability: Literal['neg', 'pos', 'none']
    terrorism_and_extremism: Literal['neg', 'pos', 'none']
    ethnic_hate_and_discrimination: Literal['neg', 'pos', 'none']


class RuleReview(StrictResponse):
    metric: Literal[
        'RuleContentNull', 'RuleContentShort', 'RuleSpecialCharacter', 'RuleAbnormalChar',
        'RuleSpaceMore', 'RuleOnlyUrl', 'RuleLoremIpsum', 'RuleDocRepeat', 'RulePIIDetection',
        'RuleHtmlEntity', 'RuleHtmlTag',
    ]
    confirmed: bool
    reason: str = Field(min_length=1)


class CodeQualityResponse(StrictResponse):
    score: int = Field(ge=0, le=1)
    type: str
    name: str
    reason: str = Field(min_length=1)
    classification: Classification
    findings: list[Finding]
    code_error: CodeError
    politics: Politics
    rule_reviews: list[RuleReview]

    @model_validator(mode='after')
    def validate_consistency(self):
        labels = {(item.type, item.name) for item in self.findings}
        if len(labels) != len(self.findings):
            raise ValueError('Duplicate finding labels')
        if self.findings:
            if self.score != 0 or (self.type, self.name) not in labels:
                raise ValueError('Defect primary must match an actual finding with score zero')
        elif (self.score, self.type, self.name) != (1, 'Good', 'None'):
            raise ValueError('Passing result must be 1 / Good / None')
        if (self.classification.score <= 2) != (('Effectiveness', 'Low_Code_Content') in labels):
            raise ValueError('Code relevance finding disagrees with classification score')
        if bool(labels & FENCE_LABELS) != ('code_fence_block_boundary_corruption' in self.code_error.tags):
            raise ValueError('Fence component and language-label findings disagree')
        if (('Completeness', 'Code_Truncation') in labels) != ('truncated_or_missing_code' in self.code_error.tags):
            raise ValueError('Truncation component and finding disagree')
        for subtype, allowed_labels in SYNTAX_LABELS.items():
            matching = labels & allowed_labels
            if subtype == 'syntax_delimiter_parser_error' and matching == {('Effectiveness', 'Code_Whitespace')}:
                # Whitespace may harm readability without causing a parser error.
                continue
            if bool(matching) != (subtype in self.code_error.syntax_subtypes):
                raise ValueError('Syntax subtype and finding disagree')
        if ('neg' in self.politics.model_dump().values()) != (('Security', 'Politics') in labels):
            raise ValueError('Politics finding disagrees with aspect judgments')
        metrics = [item.metric for item in self.rule_reviews]
        if len(metrics) != len(set(metrics)):
            raise ValueError('Duplicate rule review')
        if any(item.confirmed and not labels.intersection(RULE_LABELS[item.metric]) for item in self.rule_reviews):
            raise ValueError('Confirmed rule review must retain a corresponding finding')
        return self
