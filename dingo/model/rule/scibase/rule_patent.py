import json
import re
from collections import Counter
from typing import Any, List

from dingo.config.input_args import EvaluatorRuleArgs
from dingo.io.input import Data
from dingo.io.output.eval_detail import EvalDetail, QualityLabel
from dingo.model.model import Model
from dingo.model.rule.base import BaseRule

MAX_CLAIMS_LENGTH = 1_000_000
IPC_SUBCLASS_RE = re.compile(r"^[A-H](?:0[1-9]|[1-9][0-9])[A-Z]$")
IPC_CODE_RE = re.compile(
    r"^(?P<section>[A-H])"
    r"(?P<class>0[1-9]|[1-9][0-9])"
    r"(?P<subclass>[A-Z]) "
    r"(?P<main_group>[1-9][0-9]{0,2})/"
    r"(?P<subgroup>[0-9]{2,6})$"
)
IPC_SOURCE_CODE_RE = re.compile(
    r"^(?:[1-7]\s*)?"
    r"(?P<subclass>[A-H]\s*(?:0[1-9]|[1-9][0-9])\s*[A-Z])\s*"
    r"(?P<main_group>[0-9]{1,3})\s*/\s*"
    r"(?P<subgroup>[0-9]{2,6})(?:\s+[AB])?$"
)
IPC_FIXED_WIDTH_RE = re.compile(
    r"^(?P<subclass>[A-H](?:0[1-9]|[1-9][0-9])[A-Z])"
    r"(?P<main_group>[0-9]{3})(?P<subgroup>[0-9]{2})$"
)
IPC_FRAGMENT_RE = re.compile(
    r"^(?P<main_group>[0-9]{1,3})\s*/\s*"
    r"(?P<subgroup>[0-9]{2,6})(?:\s+[AB])?$"
)
IPC_INCOMPLETE_RE = re.compile(r"^[A-H](?:(?:0[1-9]|[1-9][0-9])[A-Z]?)?$")
IPC_AMBIGUOUS_RE = re.compile(
    r"^(?P<subclass>[A-H]\s*(?:0[1-9]|[1-9][0-9])\s*[A-Z])"
    r"(?P<separator>\s*)(?P<digits>[0-9]+)$"
)
ASCII_ALNUM_RE = re.compile(r"^[A-Za-z0-9]+$")
JP_ERA_CITATION_RE = re.compile(
    r"^JP(?P<publication>公表実用新案|実用新案公表|特開|特表|特公|実開|実表|実公)"
    r"(?P<era>[平昭令])"
    r"(?P<era_year>\d+)[\-－‐―]"
    r"(?P<serial>\d+)"
    r"号?"
    r"(?:\([^)]*\))?"
    r"(?P<kind_code>[A-Z]\d?)?$"
)
JP_GREGORIAN_CITATION_RE = re.compile(
    r"^JP(?P<publication>公表実用新案|実用新案公表|特開|特表|特公|実開|実表|実公)"
    r"(?P<year>\d{4})-"
    r"(?P<serial>\d+)"
    r"(?:\([^)]*\))?"
    r"(?P<kind_code>[A-Z]\d?)?$"
)
JP_REGISTRATION_CITATION_RE = re.compile(
    r"^JP(?:特許第|実用新案登録第|登録実用新案第|第)"
    r"(?P<registration_number>\d+)号"
    r"(?:\([^)]*\))?"
    r"(?P<kind_code>[A-Z]\d?)$"
)
JPWO_TWO_DIGIT_YEAR_RE = re.compile(
    r"^JPWO(?P<year>\d{2})/"
    r"(?P<serial>\d{5,6})"
    r"(?P<kind_code>[A-Z]\d?)$"
)
JP_ERA_SHORT_SERIAL_RE = re.compile(
    r"^JP(?P<era>[HSR])"
    r"(?P<era_year>\d{2})"
    r"(?P<serial>\d{1,5})"
    r"(?P<kind_code>[A-Z]\d?)$"
)
JP_ERA_CODES = {"平": "H", "昭": "S", "令": "R"}
JP_PUBLICATION_DEFAULT_KIND_CODES = {
    "特開": "A",
    "特表": "A",
    "特公": "B2",
    "実開": "U",
    "実表": "U",
    "実公": "Y2",
    "公表実用新案": "U",
    "実用新案公表": "U",
}
ValidationResult = tuple[bool, List[str], List[str]]
IPC_SOURCE_NOT_PROVIDED = object()


def check_claims(claims: Any) -> ValidationResult:
    claims_length = len(str(claims))
    if claims_length > MAX_CLAIMS_LENGTH:
        return (
            True,
            ["too_long"],
            [
                f"string length {claims_length} exceeds the maximum "
                f"of {MAX_CLAIMS_LENGTH}"
            ],
        )
    return False, [], []


def _convert_ipc_code(
    ipc_code: str,
    previous_subclass: str | None,
) -> tuple[str, str | None, str | None, str | None]:
    """Classify one source item and return its conversion and context.

    The first value is ``success``, ``removed`` or ``conversion_failed``.
    A bare four-character subclass is the only explicitly removable source
    form; unknown or ambiguous forms are never silently treated as removals.
    """

    source = ipc_code.strip().upper()
    match = IPC_SOURCE_CODE_RE.fullmatch(source)
    if match is None:
        # Only the original, contiguous nine-character form is fixed-width.
        # Removing spaces first would wrongly decode ambiguous US values.
        match = IPC_FIXED_WIDTH_RE.fullmatch(source)
    if match is not None:
        subclass = re.sub(r"\s+", "", match.group("subclass"))
    else:
        compact = re.sub(r"\s+", "", source)
        if IPC_SUBCLASS_RE.fullmatch(compact):
            return "removed", None, compact, None
        if IPC_INCOMPLETE_RE.fullmatch(compact):
            return "conversion_failed", None, None, "unsupported_format"
        ambiguous = IPC_AMBIGUOUS_RE.fullmatch(source)
        if ambiguous is not None:
            subclass = re.sub(r"\s+", "", ambiguous.group("subclass"))
            error_code = (
                "ambiguous"
                if ambiguous.group("separator") or len(ambiguous.group("digits")) < 6
                else "unsupported_format"
            )
            return "conversion_failed", None, subclass, error_code
        match = IPC_FRAGMENT_RE.fullmatch(source)
        if match is None:
            return "conversion_failed", None, None, "unsupported_format"
        if previous_subclass is None:
            return "conversion_failed", None, None, "missing_context"
        subclass = previous_subclass

    main_group = str(int(match.group("main_group")))
    subgroup = match.group("subgroup")
    # Preserve the existing zero-group convention accepted by this checker.
    if main_group == "0" and int(subgroup) == 0:
        return "success", subclass, subclass, None
    expected = f"{subclass} {main_group}/{subgroup}"
    if IPC_CODE_RE.fullmatch(expected) is None:
        return "conversion_failed", None, None, "unsupported_format"
    if subgroup != "00" and int(subgroup) == 0:
        return "conversion_failed", None, None, "unsupported_format"
    return "success", expected, subclass, None


def check_ipc_unified(
    ipc_unified: Any,
    ipc: Any = IPC_SOURCE_NOT_PROVIDED,
) -> ValidationResult:
    """Validate output format and, when available, deterministic IPC conversions.

    Lists and JSON arrays are accepted for both fields. Without a source this
    retains format-only validation. Conversion results are compared as
    multisets, so order is ignored while duplicate counts are preserved.
    """
    if ipc_unified is None:
        return True, ["null"], ["value is null"]

    if isinstance(ipc_unified, str):
        try:
            ipc_unified = json.loads(ipc_unified, strict=False)
        except json.JSONDecodeError:
            return True, ["invalid_json"], ["value must be a JSON array"]

    if not isinstance(ipc_unified, list):
        return True, ["not_array"], ["value must be a list"]

    error_labels: List[str] = []
    reasons: List[str] = []

    valid_output_indices: dict[str, List[int]] = {}
    for index, ipc_code in enumerate(ipc_unified):
        if not isinstance(ipc_code, str):
            if "item_not_string" not in error_labels:
                error_labels.append("item_not_string")
            reasons.append(f"item[{index}] must be a string")
            continue

        # IPC entries whose main group and subgroup are both zero are
        # normalized to the four-character subclass code (for example,
        # ``F03G 0/00`` becomes ``F03G``).
        if IPC_SUBCLASS_RE.fullmatch(ipc_code):
            valid_output_indices.setdefault(ipc_code, []).append(index)
            continue

        match = IPC_CODE_RE.fullmatch(ipc_code)
        if match is None:
            if "invalid_ipc_format" not in error_labels:
                error_labels.append("invalid_ipc_format")
            reasons.append(
                f"item[{index}] {ipc_code!r} must match IPC format "
                "'<A-H><01-99><A-Z>' or "
                "'<A-H><01-99><A-Z> <1-999>/<2-6 digits>'"
            )
            continue

        subgroup = match.group("subgroup")
        if subgroup != "00" and int(subgroup) == 0:
            if "invalid_ipc_format" not in error_labels:
                error_labels.append("invalid_ipc_format")
            reasons.append(
                f"item[{index}] {ipc_code!r} subgroup may be zero only "
                "when written as '00'"
            )
            continue
        valid_output_indices.setdefault(ipc_code, []).append(index)

    if ipc is IPC_SOURCE_NOT_PROVIDED:
        return bool(error_labels), error_labels, reasons
    if ipc is None:
        error_labels.append("source_invalid.null")
        reasons.append("ipc is null")
        return True, error_labels, reasons
    if isinstance(ipc, str):
        try:
            ipc = json.loads(ipc, strict=False)
        except json.JSONDecodeError:
            error_labels.append("source_invalid.invalid_json")
            reasons.append("ipc must be a JSON array")
            return True, error_labels, reasons
    if not isinstance(ipc, list):
        error_labels.append("source_invalid.not_array")
        reasons.append("ipc must be a list")
        return True, error_labels, reasons

    expected_sources: dict[str, List[int]] = {}
    comparison_is_resolved = True
    previous_subclass = None
    for index, source_code in enumerate(ipc):
        if not isinstance(source_code, str):
            if "source_invalid.item_not_string" not in error_labels:
                error_labels.append("source_invalid.item_not_string")
            reasons.append(f"ipc item[{index}] must be a string")
            comparison_is_resolved = False
            previous_subclass = None
            continue
        status, expected, previous_subclass, detail = _convert_ipc_code(
            source_code, previous_subclass
        )
        if status == "success":
            expected_sources.setdefault(expected, []).append(index)
        elif status == "conversion_failed":
            label = f"conversion_failed.{detail}"
            if label not in error_labels:
                error_labels.append(label)
            reason_detail = {
                "ambiguous": "has more than one possible slashless group split",
                "missing_context": "is a group fragment without preceding subclass context",
                "unsupported_format": "uses a compact format that conversion does not support",
            }[detail]
            reasons.append(f"ipc[{index}] {source_code!r} {reason_detail}")
            comparison_is_resolved = False

    expected_counts = Counter(
        {value: len(source_indices) for value, source_indices in expected_sources.items()}
    )
    actual_counts = Counter(
        {value: len(indices) for value, indices in valid_output_indices.items()}
    )
    total_valid_outputs = sum(actual_counts.values())

    def add_label(label: str) -> None:
        if label not in error_labels:
            error_labels.append(label)

    missing_values = [
        value
        for value, expected_count in expected_counts.items()
        if actual_counts[value] < expected_count
    ]
    if missing_values and total_valid_outputs == 0:
        add_label("missing.all")
        for value in missing_values:
            sources = ", ".join(f"ipc[{index}]" for index in expected_sources[value])
            reasons.append(
                f"expected {value!r} {expected_counts[value]} time(s) from {sources}, "
                "but ipc_unified contains no valid IPC items"
            )
    else:
        for value in missing_values:
            expected_count = expected_counts[value]
            actual_count = actual_counts[value]
            sources = ", ".join(f"ipc[{index}]" for index in expected_sources[value])
            if actual_count == 0:
                add_label("missing.value")
                reasons.append(
                    f"expected {value!r} {expected_count} time(s) from {sources}, "
                    "but it is absent from ipc_unified"
                )
            else:
                add_label("missing.count")
                reasons.append(
                    f"{value!r} expected {expected_count} time(s) from {sources}, "
                    f"found {actual_count}; missing {expected_count - actual_count}"
                )

    if comparison_is_resolved:
        if not ipc and total_valid_outputs:
            add_label("extra.empty_source")
            for value, indices in valid_output_indices.items():
                for index in indices:
                    reasons.append(f"ipc_unified[{index}] {value!r}: source ipc is empty")
        else:
            for value, actual_count in actual_counts.items():
                expected_count = expected_counts[value]
                if expected_count == 0:
                    add_label("extra.value")
                    for index in valid_output_indices[value]:
                        reasons.append(
                            f"ipc_unified[{index}] {value!r}: no source conversion produces this value"
                        )
                elif actual_count > expected_count:
                    add_label("extra.count")
                    extra_indices = valid_output_indices[value][expected_count:]
                    reasons.append(
                        f"{value!r} expected {expected_count} time(s), found {actual_count}; "
                        f"extra at {', '.join(f'ipc_unified[{index}]' for index in extra_indices)}"
                    )

    return bool(error_labels), error_labels, reasons


def _expected_jp_document_number(document_number: str) -> str | None:
    """Return the expected normalized value for supported JP citation forms."""
    match = JP_ERA_CITATION_RE.fullmatch(document_number)
    if match is not None:
        if len(match.group("era_year")) > 2 or len(match.group("serial")) > 6:
            return None
        kind_code = match.group("kind_code") or JP_PUBLICATION_DEFAULT_KIND_CODES[
            match.group("publication")
        ]
        return (
            "JP"
            f"{JP_ERA_CODES[match.group('era')]}"
            f"{match.group('era_year').zfill(2)}"
            f"{match.group('serial').zfill(6)}"
            f"{kind_code}"
        )

    match = JP_GREGORIAN_CITATION_RE.fullmatch(document_number)
    if match is not None:
        if len(match.group("serial")) > 6:
            return None
        # Parenthesized text is a duplicate display marker.  The normalized
        # value deliberately uses only the year, serial and kind code outside
        # the parentheses, even when the two representations disagree.
        kind_code = match.group("kind_code") or JP_PUBLICATION_DEFAULT_KIND_CODES[
            match.group("publication")
        ]
        return (
            "JP"
            f"{match.group('year')}"
            f"{match.group('serial').zfill(6)}"
            f"{kind_code}"
        )

    match = JP_REGISTRATION_CITATION_RE.fullmatch(document_number)
    if match is not None:
        if len(match.group("registration_number")) != 7:
            return None
        return (
            "JP"
            f"{match.group('registration_number')}"
            f"{match.group('kind_code')}"
        )

    match = JPWO_TWO_DIGIT_YEAR_RE.fullmatch(document_number)
    if match is not None:
        short_year = int(match.group("year"))
        if 0 <= short_year <= 26:
            year = 2000 + short_year
        elif 30 <= short_year <= 99:
            year = 1900 + short_year
        else:
            # The agreed rule does not define a century for 27-29.
            return None
        return (
            "JPWO"
            f"{year}"
            f"{match.group('serial').zfill(6)}"
            f"{match.group('kind_code')}"
        )

    match = JP_ERA_SHORT_SERIAL_RE.fullmatch(document_number)
    if match is not None:
        return (
            "JP"
            f"{match.group('era')}"
            f"{match.group('era_year')}"
            f"{match.group('serial').zfill(6)}"
            f"{match.group('kind_code')}"
        )

    return None


def check_document_number_unified(
    document_number_unified: Any,
    document_number: Any = None,
) -> ValidationResult:
    error_labels: List[str] = []
    reasons: List[str] = []

    if not isinstance(document_number_unified, str):
        return True, ["wrong_type"], ["value must be a string"]

    if ASCII_ALNUM_RE.fullmatch(document_number_unified) is None:
        error_labels.append("invalid_format")
        reasons.append("value must contain only ASCII letters and digits")

    if isinstance(document_number, str):
        expected = _expected_jp_document_number(document_number)
        if expected is not None and document_number_unified != expected:
            error_labels.append("incorrect_conversion")
            reasons.append(
                f"conversion from document_number {document_number!r} is incorrect: "
                f"expected {expected!r}, got {document_number_unified!r}"
            )

    return bool(error_labels), error_labels, reasons


def check_prior_art_references(prior_art_references: Any) -> ValidationResult:
    if isinstance(prior_art_references, str):
        try:
            prior_art_references = json.loads(prior_art_references, strict=False)
        except json.JSONDecodeError:
            return True, ["invalid_json"], ["value must be a JSON array"]

    if not isinstance(prior_art_references, list):
        return True, ["wrong_type"], ["value must be a list"]

    required_fields = {
        "sequence",
        "srep_phase",
        "document_number",
        "document_number_unified",
    }
    error_labels: List[str] = []
    reasons: List[str] = []

    def add_error_label(error_label: str) -> None:
        if error_label not in error_labels:
            error_labels.append(error_label)

    for index, reference in enumerate(prior_art_references):
        if not isinstance(reference, dict):
            add_error_label("wrong_type")
            reasons.append(f"item[{index}] must be an object")
            continue

        missing_fields = sorted(required_fields - reference.keys())
        if missing_fields:
            add_error_label("missing_field")
            reasons.extend(
                f"item[{index}].{field}: missing field" for field in missing_fields
            )
            continue

        invalid, nested_labels, nested_reasons = check_document_number_unified(
            reference["document_number_unified"],
            reference["document_number"],
        )
        if invalid:
            for error_label in nested_labels:
                add_error_label(error_label)
            reasons.extend(
                f"item[{index}].document_number_unified: {reason}"
                for reason in nested_reasons
            )

    return bool(error_labels), error_labels, reasons


FIELD_VALIDATORS = {
    "claims": lambda record: check_claims(record.get("claims")),
    "ipc_unified": lambda record: check_ipc_unified(
        record.get("ipc_unified"),
        record["ipc"] if "ipc" in record else IPC_SOURCE_NOT_PROVIDED,
    ),
    "document_number_unified": lambda record: check_document_number_unified(
        record.get("document_number_unified"),
        record.get("document_number"),
    ),
    "prior_art_references": lambda record: check_prior_art_references(
        record.get("prior_art_references")
    ),
}


@Model.rule_register("QUALITY_BAD_EFFECTIVENESS", ["xinghe", "quanliang"])
class RulePatentFieldValidation(BaseRule):
    _metric_info = {
        "category": "Rule-Based Metadata Quality Metrics",
        "quality_dimension": "EFFECTIVENESS",
        "metric_name": "RulePatentFieldValidation",
        "description": "Validate patent metadata fields and report invalid fields",
        "paper_title": "",
        "paper_url": "",
        "paper_authors": "",
        "evaluation_results": "",
    }

    _required_fields = []
    dynamic_config = EvaluatorRuleArgs(key_list=list(FIELD_VALIDATORS.keys()))

    def eval(self, input_data: Data) -> EvalDetail:
        res = EvalDetail(metric=self.__class__.__name__)
        record = input_data.to_dict()
        selected_fields = self.dynamic_config.key_list or []
        bad_fields: List[str] = []
        reasons: List[str] = []

        for field in selected_fields:
            if field not in FIELD_VALIDATORS:
                bad_fields.append(f"{field}.unsupported_field")
                reasons.append(f"{field}: unsupported field")
                continue
            if field not in record:
                bad_fields.append(f"{field}.missing_field")
                reasons.append(f"{field}: missing field")
                continue
            invalid, error_labels, detail_reasons = FIELD_VALIDATORS[field](record)
            if invalid:
                bad_fields.extend(f"{field}.{error_label}" for error_label in error_labels)
                reasons.extend(f"{field}: {reason}" for reason in detail_reasons)

        if bad_fields:
            res.status = True
            res.label = bad_fields
            res.reason = reasons
        else:
            res.label = [QualityLabel.QUALITY_GOOD]
        return res
