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


def test_v8_prompt_uses_detailed_formula_labels():
    formula_labels = {
        "Formula_Missing",
        "Formula_Partial_Loss",
        "Formula_Token_Corruption",
        "Formula_Unparseable",
        "Formula_Structure_Corruption",
        "Formula_Layout_Corruption",
        "Formula_Extra_Content",
    }

    assert "Formula_Corruption" not in LLMTextQualityV8.prompt
    for label in formula_labels:
        assert f"**{label}**" in LLMTextQualityV8.prompt
        assert f"`0 / Completeness / {label}`" in LLMTextQualityV8.prompt


def test_v8_prompt_uses_detailed_table_labels_in_expected_order():
    table_labels = [
        "Table_Missing",
        "Table_Unparseable",
        "Table_Partial_Loss",
        "Table_Cell_Corruption",
        "Table_Header_Corruption",
        "Table_Structure_Corruption",
        "Table_Layout_Corruption",
        "Table_Extra_Content",
    ]

    assert "Table_Corruption" not in LLMTextQualityV8.prompt
    assert "Table_Data_Inconsistency" not in LLMTextQualityV8.prompt
    label_positions = []
    for label in table_labels:
        label_positions.append(LLMTextQualityV8.prompt.index(f"**{label}**"))
        assert f"`0 / Completeness / {label}`" in LLMTextQualityV8.prompt
    assert label_positions == sorted(label_positions)


def test_v8_prompt_uses_detailed_code_labels_in_expected_order():
    code_labels = [
        "Code_Missing",
        "Code_Unparseable",
        "Code_Partial_Loss",
        "Code_Token_Corruption",
        "Code_Layout_Corruption",
        "Code_Extra_Content",
    ]

    assert "**Code_Corruption**" not in LLMTextQualityV8.prompt
    assert "`0 / Completeness / Code_Corruption`" not in LLMTextQualityV8.prompt
    assert "`0 / Completeness / Code_Indentation_Corruption`" not in LLMTextQualityV8.prompt
    assert "`0 / Completeness / Code_Duplication`" not in LLMTextQualityV8.prompt
    label_positions = []
    for label in code_labels:
        label_positions.append(LLMTextQualityV8.prompt.index(f"**{label}**"))
        assert f"`0 / Completeness / {label}`" in LLMTextQualityV8.prompt
    assert label_positions == sorted(label_positions)


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
