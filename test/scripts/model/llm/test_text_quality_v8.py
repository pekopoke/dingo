import inspect
import json

from dingo.model import Model
from dingo.model.llm.text_quality.llm_text_quality_v8 import LLMTextQualityV8


def test_v8_is_registered_with_a_self_contained_prompt():
    source = inspect.getsource(inspect.getmodule(LLMTextQualityV8))

    assert Model.get_llm_name_map()["LLMTextQualityV8"] is LLMTextQualityV8
    assert "LLMTextQualityV6" not in source
    assert "LLMTextQualityV7" not in source
    assert "# Role" in LLMTextQualityV8.prompt
    assert "# Input content to evaluate:" in LLMTextQualityV8.prompt


def test_v8_aggregates_multiple_defects():
    response = json.dumps([
        {"score": 0, "type": "Effectiveness", "name": "Words_Stuck", "reason": "Missing spaces"},
        {"score": 0, "type": "Similarity", "name": "Duplication", "reason": "Repeated text"},
    ])

    result = LLMTextQualityV8.process_response(response)

    assert result.metric == "LLMTextQualityV8"
    assert result.status is True
    assert result.score == 0
    assert result.label == ["Effectiveness.Words_Stuck", "Similarity.Duplication"]
