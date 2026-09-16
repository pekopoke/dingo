"""Version-one code evaluation policies; processing is inherited from shared bases."""

from dingo.model import Model
from dingo.model.llm.code_quality.base_code_quality import BaseCodeClassification, BaseCodeQuality
from dingo.model.llm.code_quality.prompts import CODE_CLASSIFICATION_PROMPT, CODE_QUALITY_PROMPT


@Model.llm_register('LLMCodeClassificationV1')
class LLMCodeClassificationV1(BaseCodeClassification):
    """Standalone 0-5 scoring for separately configured dual-model classification."""

    prompt = CODE_CLASSIFICATION_PROMPT
    _metric_info = {
        'category': 'Classification Metrics', 'metric_name': 'LLMCodeClassificationV1',
        'description': 'Precision-first code-training relevance (0-5) and independent code presence, adapted from calibrated Prompt v5.',
        'paper_title': 'Internal Implementation',
        'examples': 'examples/code_quality/evaluate_code_executor.py',
    }


@Model.llm_register('LLMCodeQualityV1')
class LLMCodeQualityV1(BaseCodeQuality):
    """Multi-label Executor output plus a primary decision and validated evidence."""

    prompt = CODE_QUALITY_PROMPT
    _metric_info = {
        'category': 'Pretrain Text Quality Assessment Metrics', 'metric_name': 'LLMCodeQualityV1',
        'description': 'Effectiveness (including low code content), completeness, repetition and security with contextual rule review and restricted code checks.',
        'paper_title': 'Internal Implementation (adapted from LLMTextQualityV6)',
        'examples': 'examples/code_quality/evaluate_code_executor.py',
        'evaluation_results': 'docs/code_quality_v1.md',
    }
