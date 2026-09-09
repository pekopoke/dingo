from dingo.retrieval.sciverse_quality import chunk_consistency_score, normalize_source_text


def test_chunk_consistency_compares_normalized_first_100_characters():
    prefix = "石墨相氮化碳可以用于二氧化碳吸附。" * 8
    chunk = prefix + "chunk suffix"
    source = prefix + "source suffix"

    assert chunk_consistency_score(chunk, source, prefix_length=100) == 1.0


def test_chunk_consistency_marks_material_difference():
    assert chunk_consistency_score("完全不同的检索片段", "原文内容没有对应文本", 100) < 1.0


def test_source_normalization_ignores_whitespace_and_width():
    assert normalize_source_text("Ａ  B\nC") == "A B C"


def test_chunk_consistency_rejects_markup_and_offset_drift():
    chunk = "# 标题\n\n作者1，作者2。正文讨论石墨相氮化碳吸附二氧化碳。"
    source = (
        "前置内容" * 100
        + "<h1>标题</h1><p>作者<sup>1</sup>，作者2。"
        + "正文讨论石墨相氮化碳吸附二氧化碳。</p>"
    )

    assert chunk_consistency_score(chunk, source, prefix_length=30) == 0.0


def test_strict_prefix_preserves_punctuation_and_only_normalizes_line_endings():
    assert chunk_consistency_score("a\r\nb", "a\nb") == 1
    assert chunk_consistency_score("a b", "ab") == 0
    assert chunk_consistency_score("a,b", "a.b") == 0
    assert chunk_consistency_score("ab", "prefix ab") == 0
    assert chunk_consistency_score("x" * 50 + "a", "x" * 50 + "b") == 1


def test_source_request_retains_original_evidence(monkeypatch):
    from types import SimpleNamespace

    from dingo.retrieval.sciverse_quality import SciverseQualityEnricher

    enricher = SciverseQualityEnricher(api_url="https://example.invalid/agentic-search", api_token="test")
    calls = []

    def get_content(doc_id, offset, limit):
        calls.append((doc_id, offset, limit))
        return SimpleNamespace(status_code=200, json=lambda: {"text": "original text"})

    monkeypatch.setattr(enricher, "_get_content", get_content)
    result = enricher.verify_source({"doc_id": "doc", "offset": 17, "chunk": "different"})
    assert calls == [("doc", 17, 200)]
    assert result.source_quality == 0.3
    assert result.to_result_fields()["_source_text"] == "original text"
    assert result.http_status == 200
    assert not result.offset_adjusted


def test_empty_window_is_not_replaced_by_offset_zero(monkeypatch):
    from types import SimpleNamespace

    from dingo.retrieval.sciverse_quality import SciverseQualityEnricher

    enricher = SciverseQualityEnricher(api_url="https://example.invalid", api_token="test")
    calls = []

    def get_content(doc_id, offset, limit):
        calls.append(offset)
        return SimpleNamespace(status_code=200, json=lambda: {"text": "", "chars_returned": 10})
    monkeypatch.setattr(enricher, "_get_content", get_content)
    assert enricher.verify_source({"doc_id": "doc", "offset": 999, "chunk": "text"}).issue == "source_empty"
    assert calls == [999]
