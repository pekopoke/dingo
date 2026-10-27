from collections import Counter
from copy import deepcopy


def refresh_ms_swift_statistics(record):
    """Refresh standard counters after a test mutates ``messages``."""
    calls = [
        call
        for message in record["messages"]
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]
    record["total_rounds"] = sum(
        message.get("role") == "assistant" for message in record["messages"]
    )
    record["total_tool_calls"] = len(calls)
    record["tool_counts"] = dict(Counter(call["function"]["name"] for call in calls))
    record["tool_call_check"]["total_calls"] = len(calls)
    return record


def standard_ms_swift_record(*, messages=None, tools=None, **extra):
    """Build a complete standardized record for converter tests."""
    record = {
        "messages": deepcopy(messages or []),
        "tools": deepcopy(tools or []),
        "termination": "",
        "source": "test.jsonl",
        "harness": "test",
        "model": "test-model",
        "total_rounds": 0,
        "total_tool_calls": 0,
        "tool_counts": {},
        "reasoning_total_tokens": 0,
        "tool_defs_tag": "complete",
        "missing_tool_defs": [],
        "tool_call_tag": "consistent",
        "tool_call_check": {
            "total_calls": 0,
            "hallucinated_calls": 0,
            "undefined_tool_calls": 0,
            "unverifiable_calls": 0,
            "checked_calls": 0,
            "extra_arg_calls": 0,
            "missing_required_calls": 0,
            "type_mismatch_calls": 0,
            "defaulted_missing_calls": 0,
            "mismatch_calls": 0,
            "mismatches": [],
            "mismatches_truncated": 0,
        },
        "server_tool_calls": [],
        "metadata": {"source_file": "test.jsonl", "line_no": 0, "created_at": ""},
        "completeness_tag": "complete_main_no_sub",
        "completeness": {
            "is_subagent": False,
            "spawn_calls": 0,
            "mounted_subs": 0,
            "subtree_complete": True,
            "unkeyed_mounts": 0,
            "relay_mounts": 0,
            "trailing_unanswered_call": False,
            "no_final_assistant_turn": False,
        },
    }
    record.update(extra)
    return refresh_ms_swift_statistics(record)
