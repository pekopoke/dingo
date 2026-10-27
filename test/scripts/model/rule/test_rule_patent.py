import json

import pytest

from dingo.io.input import Data
from dingo.model.rule.scibase.rule_patent import MAX_CLAIMS_LENGTH, RulePatentFieldValidation, check_document_number_unified, check_ipc_unified, check_prior_art_references


def _claims_with_string_length(length: int) -> list[dict]:
    empty_claims = [{"text": ""}]
    return [{"text": "x" * (length - len(str(empty_claims)))}]


class TestRulePatentFieldValidation:
    def test_claims_at_maximum_length_is_valid(self):
        claims = _claims_with_string_length(MAX_CLAIMS_LENGTH)
        assert len(str(claims)) == MAX_CLAIMS_LENGTH

        result = RulePatentFieldValidation().eval(
            Data(
                claims=claims,
                ipc_unified=[],
                document_number="JP2013022164A",
                document_number_unified="JP2013022164A",
                prior_art_references=[],
            )
        )

        assert result.status is False
        assert result.label == ["QUALITY_GOOD"]

    def test_claims_over_maximum_length_is_invalid(self):
        claims = _claims_with_string_length(MAX_CLAIMS_LENGTH + 1)
        assert len(str(claims)) == MAX_CLAIMS_LENGTH + 1

        result = RulePatentFieldValidation().eval(
            Data(
                claims=claims,
                ipc_unified=[],
                document_number="JP2013022164A",
                document_number_unified="JP2013022164A",
                prior_art_references=[],
            )
        )

        assert result.status is True
        assert result.label == ["claims.too_long"]
        assert result.reason == [
            "claims: string length 1000001 exceeds the maximum of 1000000"
        ]

    def test_missing_claims_is_invalid(self):
        result = RulePatentFieldValidation().eval(
            Data(
                ipc_unified=[],
                document_number="JP2013022164A",
                document_number_unified="JP2013022164A",
                prior_art_references=[],
            )
        )

        assert result.status is True
        assert result.label == ["claims.missing_field"]

    @pytest.mark.parametrize(
        "ipc_unified",
        [
            [],
            ["F03G"],
            ["A01A 1/00"],
            ["B60T 8/48", "H99Z 999/999999"],
            '["B60T 8/48", "B60T 8/58"]',
        ],
    )
    def test_valid_ipc_unified(self, ipc_unified):
        assert check_ipc_unified(ipc_unified) == (False, [], [])

    @pytest.mark.parametrize(
        "ipc_code",
        [
            "I01A 1/00",       # Section must be A-H.
            "A00A 1/00",       # Class must be 01-99.
            "A1A 1/00",        # Class must contain exactly two digits.
            "A01a 1/00",       # Subclass must be uppercase A-Z.
            "A01A\u00a01/00",  # Separator must be one ASCII space.
            "A01A  1/00",      # Only one separator space is allowed.
            "A01A 01/00",      # Main group must not be zero-padded.
            "A01A 1000/00",    # Main group must be at most 999.
            "A01A 1/0",        # Subgroup must contain at least two digits.
            "A01A 1/000",      # A zero subgroup must be written as 00.
            "A01A 1/1234567",  # Subgroup must contain at most six digits.
            "A01A 1-00",       # Groups must be separated by a slash.
            "A01A 1/00 A",     # Extra suffixes are not allowed.
            "F03G 0/00",       # Zero groups must be reduced to "F03G".
        ],
    )
    def test_invalid_ipc_format(self, ipc_code):
        invalid, labels, reasons = check_ipc_unified([ipc_code])

        assert invalid is True
        assert labels == ["invalid_ipc_format"]
        assert reasons[0].startswith(f"item[0] {ipc_code!r}")

    def test_reports_every_invalid_item_with_its_value(self):
        invalid, labels, reasons = check_ipc_unified(
            ["B01J 29/40", "C00C 5/27", "B01J 29/18", "C00C 15/08"]
        )

        assert invalid is True
        assert labels == ["invalid_ipc_format"]
        assert reasons == [
            "item[1] 'C00C 5/27' must match IPC format "
            "'<A-H><01-99><A-Z>' or "
            "'<A-H><01-99><A-Z> <1-999>/<2-6 digits>'",
            "item[3] 'C00C 15/08' must match IPC format "
            "'<A-H><01-99><A-Z>' or "
            "'<A-H><01-99><A-Z> <1-999>/<2-6 digits>'",
        ]

    def test_wrong_type_reason_does_not_include_value(self):
        invalid, labels, reasons = check_ipc_unified([123])

        assert invalid is True
        assert labels == ["item_not_string"]
        assert reasons == ["item[0] must be a string"]

    @pytest.mark.parametrize(
        ("ipc_unified", "expected_label"),
        [
            (None, "null"),
            ({"code": "A01A 1/00"}, "not_array"),
            ([1], "item_not_string"),
            ("not-json", "invalid_json"),
        ],
    )
    def test_invalid_ipc_unified_value(self, ipc_unified, expected_label):
        invalid, labels, reasons = check_ipc_unified(ipc_unified)

        assert invalid is True
        assert labels == [expected_label]
        assert reasons

    def test_rule_reports_invalid_ipc_unified(self):
        result = RulePatentFieldValidation().eval(
            Data(
                claims=[],
                ipc_unified=["A01A 01/00"],
                document_number="JP2013022164A",
                document_number_unified="JP2013022164A",
                prior_art_references=[],
            )
        )

        assert result.status is True
        assert result.label == ["ipc_unified.invalid_ipc_format"]
        assert result.reason[0].startswith("ipc_unified: item[0]")

    @pytest.mark.parametrize(
        ("ipc", "ipc_unified"),
        [
            (["3C 06B 21/00 A"], ["C06B 21/00"]),
            (["7H 04L 012/56 B"], ["H04L 12/56"]),
            (["a61k 31/00"], ["A61K 31/00"]),
            ([" C 25D 021/12 "], ["C25D 21/12"]),
            (["C25D21/12", "C08F214/18"], ["C25D 21/12", "C08F 214/18"]),
            (["G01B009/00"], ["G01B 9/00"]),
            (["H01R01970", "D21C00712"], ["H01R 19/70", "D21C 7/12"]),
            (["F03G 0/00"], ["F03G"]),
            (["F03G00000"], ["F03G"]),
            (
                ["C08F214/18", "216/14", "218/16"],
                ["C08F 214/18", "C08F 216/14", "C08F 218/16"],
            ),
            (["A61K", "31/00"], ["A61K 31/00"]),
            (["A61K"], []),
            ([], []),
        ],
    )
    def test_valid_ipc_conversion(self, ipc, ipc_unified):
        original = list(ipc)
        unified = list(ipc_unified)

        assert check_ipc_unified(ipc_unified, ipc) == (False, [], [])
        assert check_ipc_unified(json.dumps(ipc_unified), json.dumps(ipc)) == (
            False,
            [],
            [],
        )
        assert ipc == original
        assert ipc_unified == unified

    def test_conversion_comparison_ignores_order(self):
        assert check_ipc_unified(
            ["G01B 9/00", "C06B 21/00"],
            ["3C 06B 21/00 A", "G01B009/00"],
        ) == (False, [], [])

    def test_wrong_conversion_reports_missing_and_extra_values(self):
        invalid, labels, reasons = check_ipc_unified(
            ["C06B 22/00"],
            ["3C 06B 21/00 A"],
        )

        assert invalid is True
        assert labels == ["missing.value", "extra.value"]
        assert any("expected 'C06B 21/00'" in reason for reason in reasons)
        assert any("ipc_unified[0] 'C06B 22/00'" in reason for reason in reasons)

    @pytest.mark.parametrize("ipc_unified", [[], "[]"])
    def test_all_expected_values_missing(self, ipc_unified):
        invalid, labels, reasons = check_ipc_unified(
            ipc_unified,
            ["C08F214/18", "216/14"],
        )

        assert invalid is True
        assert labels == ["missing.all"]
        assert len(reasons) == 2

    def test_missing_value_and_missing_count_are_distinct(self):
        invalid, labels, reasons = check_ipc_unified(
            ["A61K 31/00"],
            ["A61K 31/00", "A61K 31/00", "G01B009/00"],
        )

        assert invalid is True
        assert labels == ["missing.count", "missing.value"]
        assert any("missing 1" in reason for reason in reasons)
        assert any("'G01B 9/00'" in reason for reason in reasons)

    @pytest.mark.parametrize("ipc", [[], "[]"])
    def test_extra_values_with_empty_source_have_specific_label(self, ipc):
        assert check_ipc_unified(["A45F 5/02"], ipc) == (
            True,
            ["extra.empty_source"],
            ["ipc_unified[0] 'A45F 5/02': source ipc is empty"],
        )

    def test_extra_value_and_extra_count_are_distinct(self):
        invalid, labels, reasons = check_ipc_unified(
            ["A61K 31/00", "A61K 31/00", "G01B 9/00"],
            ["A61K 31/00"],
        )

        assert invalid is True
        assert labels == ["extra.count", "extra.value"]
        assert any("extra at ipc_unified[1]" in reason for reason in reasons)
        assert any("ipc_unified[2] 'G01B 9/00'" in reason for reason in reasons)

    def test_bare_subclass_is_an_explicit_removal_not_an_unknown_form(self):
        assert check_ipc_unified([], ["A61K"]) == (False, [], [])
        assert check_ipc_unified(["A61K"], ["A61K"]) == (
            True,
            ["extra.value"],
            ["ipc_unified[0] 'A61K': no source conversion produces this value"],
        )

    @pytest.mark.parametrize(
        ("ipc", "expected_label"),
        [
            (["H01R 1200"], "conversion_failed.ambiguous"),
            (["G05F001455"], "conversion_failed.unsupported_format"),
            (["A61"], "conversion_failed.unsupported_format"),
            (["unrecognized"], "conversion_failed.unsupported_format"),
            (["216/14"], "conversion_failed.missing_context"),
        ],
    )
    def test_conversion_failures_are_not_treated_as_removals(
        self, ipc, expected_label
    ):
        invalid, labels, reasons = check_ipc_unified([], ipc)

        assert invalid is True
        assert labels == [expected_label]
        assert reasons

    @pytest.mark.parametrize(
        ("ipc", "expected_label"),
        [
            (None, "source_invalid.null"),
            ("not-json", "source_invalid.invalid_json"),
            ('{"code": "A61K 31/00"}', "source_invalid.not_array"),
            (123, "source_invalid.not_array"),
            ([None], "source_invalid.item_not_string"),
        ],
    )
    def test_invalid_source_ipc_is_reported(self, ipc, expected_label):
        invalid, labels, reasons = check_ipc_unified([], ipc)

        assert invalid is True
        assert labels == [expected_label]
        assert reasons

    def test_unresolved_source_suppresses_extra_but_not_known_missing(self):
        invalid, labels, reasons = check_ipc_unified(
            ["C06B 22/00", "A45F 5/02"],
            ["H01R 1200", "3C 06B 21/00 A"],
        )

        assert invalid is True
        assert labels == ["conversion_failed.ambiguous", "missing.value"]
        assert not any(label.startswith("extra.") for label in labels)
        assert any("'C06B 21/00'" in reason for reason in reasons)

    def test_unsupported_source_format_suppresses_extra_but_not_known_missing(self):
        invalid, labels, _ = check_ipc_unified(
            ["A45F 5/02"],
            ["unrecognized", "G01B009/00"],
        )

        assert invalid is True
        assert labels == ["conversion_failed.unsupported_format", "missing.value"]

    def test_invalid_output_item_is_not_also_reported_as_extra(self):
        invalid, labels, reasons = check_ipc_unified(
            ["not-an-ipc"],
            ["G01B009/00"],
        )

        assert invalid is True
        assert labels == [
            "invalid_ipc_format",
            "missing.all",
        ]
        assert not any(label.startswith("extra.") for label in labels)
        assert len(reasons) == 2

    def test_rule_prefixes_detailed_conversion_labels(self):
        model = RulePatentFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["ipc_unified"]

        result = model.eval(
            Data(ipc=["3C 06B 21/00 A"], ipc_unified=["C06B 22/00"])
        )

        assert result.status is True
        assert result.label == [
            "ipc_unified.missing.value",
            "ipc_unified.extra.value",
        ]
        assert all(reason.startswith("ipc_unified: ") for reason in result.reason)

    def test_rule_reports_null_source_but_missing_source_keeps_format_only_mode(self):
        model = RulePatentFieldValidation()
        model.dynamic_config = model.dynamic_config.model_copy(deep=True)
        model.dynamic_config.key_list = ["ipc_unified"]

        null_result = model.eval(Data(ipc=None, ipc_unified=[]))
        assert null_result.label == ["ipc_unified.source_invalid.null"]
        assert model.eval(Data(ipc_unified=[])).status is False

    @pytest.mark.parametrize(
        ("document_number", "document_number_unified"),
        [
            ("JP特開平11-73838A", "JPH11073838A"),
            ("JP特開平5-263266A", "JPH05263266A"),
            ("JP特開昭63-123456A", "JPS63123456A"),
            ("JP特開昭63-123456", "JPS63123456A"),
            ("JP特開2003-75095(P2003-75095A)A", "JP2003075095A"),
            # Parenthesized duplicate data is ignored when it conflicts with
            # the year, serial or kind code outside the parentheses.
            ("JP特開2003-75095(P2004-123456B2)A", "JP2003075095A"),
            ("JP特公平7-46852B2", "JPH07046852B2"),
            ("JP特公昭48-1234B2", "JPS48001234B2"),
            ("JP特表平8-500910A", "JPH08500910A"),
            ("JP特開令6-123A", "JPR06000123A"),
            ("JP特許第3405959号(P3405959)B2", "JP3405959B2"),
            ("JP実用新案登録第3082161号(U3082161)U", "JP3082161U"),
            ("JP登録実用新案第3082161号(U3082161)U", "JP3082161U"),
            ("JP実開平6-10396U", "JPH06010396U"),
            ("JP実公平7-15070Y2", "JPH07015070Y2"),
            ("JP第2891579号B2", "JP2891579B2"),
            ("JPWO96/34063A1", "JPWO1996034063A1"),
            ("JPWO01/088243A1", "JPWO2001088243A1"),
            ("JPWO26/12345A1", "JPWO2026012345A1"),
            ("JPWO30/12345A1", "JPWO1930012345A1"),
            ("JPH1083076A", "JPH10083076A"),
            ("JPH0517383U", "JPH05017383U"),
            ("JPS5918043B2", "JPS59018043B2"),
        ],
    )
    def test_valid_document_number_conversion(
        self,
        document_number,
        document_number_unified,
    ):
        assert check_document_number_unified(
            document_number_unified,
            document_number,
        ) == (False, [], [])

    @pytest.mark.parametrize(
        "document_number_unified",
        [
            "",
            "JP特開平11-73838A",
            "JP2013-022164A",
            "JP2013022164 A",
        ],
    )
    def test_document_number_unified_must_be_ascii_alphanumeric(
        self,
        document_number_unified,
    ):
        invalid, labels, reasons = check_document_number_unified(
            document_number_unified
        )

        assert invalid is True
        assert labels == ["invalid_format"]
        assert reasons == ["value must contain only ASCII letters and digits"]

    @pytest.mark.parametrize("document_number_unified", [None, 123, []])
    def test_document_number_unified_must_be_string(
        self,
        document_number_unified,
    ):
        assert check_document_number_unified(document_number_unified) == (
            True,
            ["wrong_type"],
            ["value must be a string"],
        )

    def test_reports_incorrect_document_number_conversion(self):
        invalid, labels, reasons = check_document_number_unified(
            "JPH11073839A",
            "JP特開平11-73838A",
        )

        assert invalid is True
        assert labels == ["incorrect_conversion"]
        assert reasons == [
            "conversion from document_number 'JP特開平11-73838A' is incorrect: "
            "expected 'JPH11073838A', got 'JPH11073839A'"
        ]

    @pytest.mark.parametrize("year", ["27", "28", "29"])
    def test_jpwo_undefined_two_digit_year_is_not_guessed(self, year):
        original = f"JPWO{year}/12345A1"

        # The document does not define a century for 27-29, so the checker
        # validates the unified field's format without inventing a conversion.
        assert check_document_number_unified(original, original) == (
            True,
            ["invalid_format"],
            ["value must contain only ASCII letters and digits"],
        )

    def test_rule_reports_invalid_document_number_unified(self):
        result = RulePatentFieldValidation().eval(
            Data(
                claims=[],
                ipc_unified=[],
                document_number="JP特開平11-73838A",
                document_number_unified="JPH11073839A",
                prior_art_references=[],
            )
        )

        assert result.status is True
        assert result.label == ["document_number_unified.incorrect_conversion"]
        assert result.reason == [
            "document_number_unified: conversion from document_number "
            "'JP特開平11-73838A' is incorrect: expected 'JPH11073838A', "
            "got 'JPH11073839A'"
        ]

    def test_missing_document_number_unified_is_invalid(self):
        result = RulePatentFieldValidation().eval(
            Data(
                claims=[],
                ipc_unified=[],
                document_number="JP2013022164A",
                prior_art_references=[],
            )
        )

        assert result.status is True
        assert result.label == ["document_number_unified.missing_field"]
        assert result.reason == ["document_number_unified: missing field"]

    def test_valid_prior_art_references(self):
        prior_art_references = [
            {
                "sequence": "1",
                "srep_phase": "SEA",
                "document_number": "JP特開2002-15639(JP,A)A",
                "document_number_unified": "JP2002015639A",
            },
            {
                "sequence": "2",
                "srep_phase": "SEA",
                "document_number": "JP特開平7-302526(JP,A)A",
                "document_number_unified": "JPH07302526A",
            },
        ]

        assert check_prior_art_references(prior_art_references) == (
            False,
            [],
            [],
        )

        assert check_prior_art_references(
            json.dumps(prior_art_references, ensure_ascii=False)
        ) == (False, [], [])

    def test_prior_art_references_reports_item_index(self):
        invalid, labels, reasons = check_prior_art_references(
            [
                {
                    "sequence": "1",
                    "srep_phase": "SEA",
                    "document_number": "JP特開2002-15639(JP,A)A",
                    "document_number_unified": "JP2002015639A",
                },
                {
                    "sequence": "2",
                    "srep_phase": "SEA",
                    "document_number": "JP特開平7-302526(JP,A)A",
                    "document_number_unified": "JPH07302527A",
                },
            ]
        )

        assert invalid is True
        assert labels == ["incorrect_conversion"]
        assert reasons == [
            "item[1].document_number_unified: conversion from document_number "
            "'JP特開平7-302526(JP,A)A' is incorrect: expected "
            "'JPH07302526A', got 'JPH07302527A'"
        ]

    @pytest.mark.parametrize(
        ("prior_art_references", "expected_label", "expected_reason"),
        [
            (None, "wrong_type", "value must be a list"),
            ("not-json", "invalid_json", "value must be a JSON array"),
            ('{"sequence": "1"}', "wrong_type", "value must be a list"),
            (["not-an-object"], "wrong_type", "item[0] must be an object"),
            (
                [{"sequence": "1"}],
                "missing_field",
                "item[0].document_number: missing field",
            ),
        ],
    )
    def test_invalid_prior_art_references_structure(
        self,
        prior_art_references,
        expected_label,
        expected_reason,
    ):
        invalid, labels, reasons = check_prior_art_references(prior_art_references)

        assert invalid is True
        assert labels == [expected_label]
        assert expected_reason in reasons

    def test_rule_reports_invalid_prior_art_reference(self):
        result = RulePatentFieldValidation().eval(
            Data(
                claims=[],
                ipc_unified=[],
                document_number="JP2013022164A",
                document_number_unified="JP2013022164A",
                prior_art_references=[
                    {
                        "sequence": "1",
                        "srep_phase": "SEA",
                        "document_number": "JP特開平7-302526(JP,A)A",
                        "document_number_unified": "JPH07302527A",
                    }
                ],
            )
        )

        assert result.status is True
        assert result.label == ["prior_art_references.incorrect_conversion"]
        assert result.reason == [
            "prior_art_references: item[0].document_number_unified: conversion "
            "from document_number 'JP特開平7-302526(JP,A)A' is incorrect: "
            "expected 'JPH07302526A', got 'JPH07302527A'"
        ]
