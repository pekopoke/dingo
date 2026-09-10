import json
from pathlib import Path

from dingo.io.input import Data
from dingo.model.rule.scibase.rule_quanliang import RuleQuanliangFieldValidation


class TestRuleQuanliangFieldValidation:
    def test_author_quality_labels(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["author"]

        result = model.eval(Data(author=[]))
        assert result.label == ["author.empty"]

        result = model.eval(
            Data(
                author=[
                    {"name": "   ", "orcid": ""},
                    {"name": "John  Smith", "orcid": ""},
                    {"name": " john smith ", "orcid": ""},
                    {"name": "Alice||Bob", "orcid": "https://orcid.org/0000-0002-1825-0098"},
                ]
            )
        )
        assert result.label == [
            "author.empty_name",
            "author.duplicated_name",
            "author.multiple_names",
            "author.invalid_separator",
            "author.invalid_orcid",
        ]

    def test_author_valid_orcid_checksum(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["author"]

        result = model.eval(
            Data(
                author=[
                    {
                        "name": "John Smith",
                        "orcid": "https://orcid.org/0000-0002-1825-0097",
                    }
                ]
            )
        )

        assert result.status is False
        assert result.label == ["QUALITY_GOOD"]

    def test_author_invalid_separator_patterns(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["author"]

        invalid_names = (
            "Alice|Bob",
            "Alice;Bob",
            "Alice；Bob",
            "Alice,,Bob",
            "Alice，，Bob",
        )
        for name in invalid_names:
            result = model.eval(Data(author=[{"name": name, "orcid": ""}]))
            assert result.label == [
                "author.multiple_names",
                "author.invalid_separator",
            ]

    def test_author_multiple_names_patterns(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["author"]

        candidate_names = (
            "John Smith and Mary Lee",
            "Alice & Bob",
            "张三、李四",
            "John Smith et al.",
            "张三，李四",
            "John Smith, Mary Lee",
            "Smith, John, Lee, Mary",
            "张三 李四",
        )
        for name in candidate_names:
            result = model.eval(Data(author=[{"name": name, "orcid": ""}]))
            assert "author.multiple_names" in result.label

        non_candidate_names = ("John Smith", "Smith, John", "R&D", "AT&T Research")
        for name in non_candidate_names:
            result = model.eval(Data(author=[{"name": name, "orcid": ""}]))
            assert result.label == ["QUALITY_GOOD"]

    def test_doi_empty_format_and_test_prefix_labels(self):
        cases = [
            ("   ", "doi.empty"),
            ("https://doi.org/10.1234/abc", "doi.format_invalid"),
            ("10.1234/abc def", "doi.format_invalid"),
            ("10.1234/abc\tdef", "doi.format_invalid"),
            ("10.0000/example", "doi.error_prefix"),
            ("10.0001/example", "doi.error_prefix"),
            ("10.5555/example", "doi.error_prefix"),
        ]

        for doi, expected_label in cases:
            model = RuleQuanliangFieldValidation()
            model.dynamic_config = model.dynamic_config.model_copy(deep=True)
            model.dynamic_config.key_list = ["doi"]
            result = model.eval(Data(metadata_type="paper", doi=doi))
            assert result.status is True
            assert result.label == [expected_label]

    def test_invalid_test_prefix_doi_only_reports_format(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["doi"]

        result = model.eval(Data(metadata_type="paper", doi="10.0000/"))

        assert result.label == ["doi.format_invalid"]

    def test_rule_quanliang_cases_from_jsonl(self):
        data_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "scibase" / "rule_quanliang_cases.jsonl"
        )
        assert data_path.exists(), f"missing test data file: {data_path}"

        original_key_list = RuleQuanliangFieldValidation.dynamic_config.key_list
        try:
            with data_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    model = RuleQuanliangFieldValidation()
                    model.dynamic_config = model.dynamic_config.model_copy(deep=True)
                    model.dynamic_config.key_list = row["key_list"]
                    result = model.eval(Data(**row["input"]))

                    assert result.metric == "RuleQuanliangFieldValidation"
                    assert result.status is row["expected_status"], row["case"]
                    assert result.label == row["expected_labels"], row["case"]

                    expected_reasons = row["expected_reasons"]
                    if expected_reasons:
                        assert result.reason == expected_reasons, row["case"]
                    else:
                        assert result.reason in (None, []), row["case"]
        finally:
            RuleQuanliangFieldValidation.dynamic_config.key_list = original_key_list

    def test_title_and_abstract_return_hierarchical_multi_labels(self):
        value = (
            "<i>layout</i> <mml:math>x</mml:math> <!--note--> <![CDATA[x]]> "
            "&amp; &#39; &#x0D; \u200b \ufffd \x08 [!sub]"
        )
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["title", "abstract"]

        result = model.eval(Data(title=value, abstract=value))

        expected_title_labels = [
            "title.html_tag.formatting",
            "title.html_tag.xml_comment",
            "title.html_tag.cdata",
            "title.html_tag.math",
            "title.html_entity.named",
            "title.html_entity.decimal",
            "title.html_entity.hex",
            "title.markup_tag.formatting",
            "title.special_char.replacement",
            "title.special_char.control",
            "title.invisible_char.zero_width_space",
            "title.encoding_error",
        ]
        expected_abstract_error_labels = [
            "html_tag_layout",
            "html_tag_math",
            "html_tag_xml_comment",
            "html_tag_cdata",
            "html_entity_named",
            "html_entity_decimal",
            "html_entity_hex",
            "special_char_invisible",
            "special_char_replacement",
            "special_char_control",
            "special_char_markup",
        ]
        assert result.status is True
        assert result.label == [
            *expected_title_labels,
            *(f"abstract.{error_label}" for error_label in expected_abstract_error_labels),
            "abstract.encoding_error",
            "abstract.same_title",
        ]

    def test_title_expanded_markup_and_unicode_labels(self):
        cases = [
            ("A study of <i>formatted</i> text", ["title.html_tag.formatting"]),
            ("<p>A structured title</p>", ["title.html_tag.structure"]),
            ("<a href='https://example.com'>Linked title</a>", ["title.html_tag.link"]),
            ("Image <inline-graphic href='x'/> in title", ["title.html_tag.media"]),
            (
                "<jats:custom>Namespaced title</jats:custom>",
                ["title.html_tag.namespaced"],
            ),
            ("A title <!-- note --> with comment", ["title.html_tag.xml_comment"]),
            ("A title <![CDATA[with data]]> section", ["title.html_tag.cdata"]),
            ("Water H<sub>2</sub>O analysis", ["title.html_tag.sub_sup"]),
            (
                "<mml:math><mml:mi>x</mml:mi></mml:math> equation",
                ["title.html_tag.math"],
            ),
            ("A title with <strong", ["title.html_tag.incomplete"]),
            ("A title with /i> missing opener", ["title.html_tag.incomplete"]),
            ("A title with <!-- broken comment", ["title.html_tag.incomplete"]),
            ("A title with <![CDATA[broken data", ["title.html_tag.incomplete"]),
            (
                "A title with <i>one-sided markup",
                ["title.html_tag.formatting", "title.html_tag.mismatched"],
            ),
            (
                "A title with <i>broken</b>",
                ["title.html_tag.formatting", "title.html_tag.mismatched"],
            ),
            ("Named entity &amp; in title", ["title.html_entity.named"]),
            ("Decimal entity &#160; in title", ["title.html_entity.decimal"]),
            ("Hex entity &#xA0; in title", ["title.html_entity.hex"]),
            ("A [!i]formatted[!/i] title", ["title.markup_tag.formatting"]),
            ("A title with [○!R] residue", ["title.markup_tag.crawler"]),
            (
                "A title with � replacement",
                ["title.special_char.replacement", "title.encoding_error"],
            ),
            ("A title with \x08 control", ["title.special_char.control"]),
            ("A title with \ue000 private use", ["title.special_char.private_use_area"]),
            ("A title with \u200b zero width", ["title.invisible_char.zero_width_space"]),
            ("A title with \ufeff BOM", ["title.invisible_char.bom"]),
            ("A title with \u200c ZWNJ", ["title.invisible_char.zwnj"]),
            ("A title with \u200d ZWJ", ["title.invisible_char.zwj"]),
            ("A title with \u202e bidi control", ["title.invisible_char.bidi_control"]),
            ("A title with \u00a0 NBSP", ["title.space_char.nbsp"]),
            ("A title with \u2003 typographic space", ["title.space_char.typographic"]),
        ]

        for title, expected_labels in cases:
            model = RuleQuanliangFieldValidation()
            model.dynamic_config = model.dynamic_config.model_copy(deep=True)
            model.dynamic_config.key_list = ["title"]

            result = model.eval(Data(title=title))

            assert result.status is True, title
            assert result.label == expected_labels, title

    def test_title_namespaced_link_reports_both_relevant_labels(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["title"]

        result = model.eval(Data(title="<jats:ext-link>Linked title</jats:ext-link>"))

        assert result.label == [
            "title.html_tag.link",
            "title.html_tag.namespaced",
        ]

    def test_title_bibliographic_angle_brackets_are_not_html(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["title"]

        for title in (
            "Collected <Articles> from the archive",
            "A concise <Introduction> to metadata",
            "<The> collected scientific works",
            "Symphony number one <sound recording>",
        ):
            result = model.eval(Data(title=title))
            assert result.status is False, title
            assert result.label == ["QUALITY_GOOD"], title

    def test_title_quality_labels(self):
        cases = [
            (None, ["title.null"]),
            ("   ", ["title.empty"]),
            ("\u00a0", ["title.space_char.nbsp", "title.empty"]),
            ("Test", ["title.too_short"]),
            ("A" * 1001, ["title.too_long"]),
            ("N/A", ["title.too_short", "title.likely_placeholder"]),
            ("This title contains 锟斤拷 encoding noise", ["title.encoding_error"]),
            (
                "2024 IEEE International Conference on Big Data",
                ["title.likely_conference"],
            ),
            ("https://example.com/paper", ["title.likely_identifier"]),
        ]

        for title, expected_labels in cases:
            model = RuleQuanliangFieldValidation()
            model.dynamic_config = model.dynamic_config.model_copy(deep=True)
            model.dynamic_config.key_list = ["title"]
            result = model.eval(Data(title=title))
            assert result.status is True
            assert result.label == expected_labels

    def test_title_rules_avoid_conference_and_url_false_positives(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["title"]

        for title in (
            "IEEE Transactions on Knowledge and Data Engineering",
            "Lessons from the Conference on Machine Learning",
            "https://example.com A Study of Machine Learning",
            "Research at https://example.com/i> remains valid",
        ):
            result = model.eval(Data(title=title))
            assert result.status is False
            assert result.label == ["QUALITY_GOOD"]

    def test_abstract_quality_labels(self):
        cases = [
            ("", "Different title", ["abstract.empty"]),
            ("short abstract", "Different title", ["abstract.too_short"]),
            (
                "No abstract available.",
                "Different title",
                ["abstract.likely_placeholder"],
            ),
            (
                "This text contains the mojibake sequence 锟斤拷 and is long enough.",
                "Different title",
                ["abstract.encoding_error"],
            ),
            (
                "The Same Abstract Title With More Than Thirty Characters",
                "the same abstract title with more than thirty characters",
                ["abstract.same_title"],
            ),
            (
                "https://example.com/paper",
                "Different title",
                ["abstract.likely_identifier"],
            ),
        ]

        for abstract, title, expected_labels in cases:
            model = RuleQuanliangFieldValidation()
            model.dynamic_config = model.dynamic_config.model_copy(deep=True)
            model.dynamic_config.key_list = ["abstract"]
            result = model.eval(Data(title=title, abstract=abstract))
            assert result.status is True
            assert result.label == expected_labels

    def test_abstract_empty_matches_after_trimming(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["abstract"]

        result = model.eval(Data(abstract="   "))

        assert result.status is True
        assert result.label == ["abstract.empty"]

    def test_abstract_placeholder_and_url_rules_avoid_substring_false_positives(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["abstract"]
        abstract = (
            "Some experimental data were unavailable. https://example.com provides "
            "supporting material for the complete study."
        )

        result = model.eval(Data(abstract=abstract))

        assert result.status is False
        assert result.label == ["QUALITY_GOOD"]

    def test_high_false_positive_patterns_are_not_enabled(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["title", "abstract"]

        result = model.eval(
            Data(
                title="Comparison of <candidate> values with amp and gt proteins",
                abstract="A legitimate [sic!] quotation with lt as an abbreviation.",
            )
        )

        assert result.status is False
        assert result.label == ["QUALITY_GOOD"]

    def test_reference_title_propagates_multiple_error_labels(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["references"]

        result = model.eval(
            Data(
                references=[
                    {
                        "id_type": "other",
                        "id": "source-1",
                        "title": "<i>formatted</i> &amp;",
                    }
                ]
            )
        )

        assert result.label == [
            "references.title_html_tag_layout",
            "references.title_html_entity_named",
        ]

    def test_text_issue_reason_contains_all_distinct_matches(self):
        model = RuleQuanliangFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["title"]

        result = model.eval(Data(title="<i>A</i> <i>B</i> &amp; &amp; \u200b"))

        assert result.reason == [
            'title: contains HTML formatting tag: ["<i>", "</i>"]',
            'title: contains named HTML entity: ["&amp;"]',
            'title: contains zero-width space: ["\\u200b"]',
        ]
