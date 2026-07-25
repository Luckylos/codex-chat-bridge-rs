#!/usr/bin/env python3
"""Generate the Python↔Rust parity golden fixture.

Drives the authoritative Python bridge's deterministic pure transforms over a
curated set of edge-case input vectors and writes `{fn, input, output}` records
to `golden.json`. The Rust integration test `tests/parity_golden.rs` loads the
same fixture and asserts the Rust port produces byte-identical output.

Run from the Python bridge venv:
    uv run --project /opt/codex-chat-bridge python \
        /opt/codex-chat-bridge-rs/tests/parity/generate_golden.py

Only deterministic, side-effect-free functions belong here. Anything touching
IO, time, randomness, or global config is out of scope for byte-diff parity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# The Python bridge package must be importable (run under its venv/project).
from codex_chat_bridge.text_utils import sanitize_string
from codex_chat_bridge.tool_arguments import canonicalize_tool_arguments
from codex_chat_bridge.bridge_context.naming import (
    flatten_namespace_tool_name,
    short_sha256_hex,
)
from codex_chat_bridge.bridge_context.custom_tools import (
    custom_tool_input_from_chat_arguments,
    partial_custom_tool_input_from_chat_arguments,
)

OUT = Path(__file__).with_name("golden.json")


def sanitize_vectors() -> list[str]:
    return [
        "",
        "plain text",
        "tab\tnewline\ncarriage\rreturn",
        "nul\x00byte",
        "bell\x07and\x1bescape",
        "unicode ✓ 世界 🌍",
        "mix\x01of\x02control\x1fchars",
        "\x00\x01\x02\x03",
        "trailing\x7f",  # DEL is >= 0x20, preserved
    ]


def flatten_vectors() -> list[tuple[str, str]]:
    return [
        ("fs", "read"),
        ("db", "query"),
        # Exactly at and around the 64-char boundary.
        ("n" * 30, "m" * 30),  # 30+2+30 = 62
        ("n" * 31, "m" * 31),  # 64 exactly
        ("n" * 32, "m" * 32),  # 66 → hashed
        ("very_long_namespace_name", "very_long_action_name_that_pushes_over_the_limit"),
        # Multi-byte unicode: Python counts code points, Rust must match.
        ("世界" * 20, "动作" * 20),
        ("emoji🌍", "action🚀name"),
    ]


def short_sha_vectors() -> list[str]:
    return ["", "abc", "the quick brown fox", "世界🌍", "a" * 200]


def custom_input_vectors() -> list[str]:
    return [
        "",
        "   ",
        '{"input": "echo hello"}',
        '{"input": "multi\\nline"}',
        '{"other": "no input field"}',
        '{"input": 42}',  # non-string value → return raw
        "not json at all",
        '{"input": "unicode ✓"}',
    ]


def partial_input_vectors() -> list[str]:
    return [
        "",
        '{"input": "ec',  # mid-string
        '{"input": "echo hello"}',
        '{"input": "esc\\n',  # escape mid-stream
        '{"input": "uni\\u00e9',  # \u escape
        '{"input": "bad\\',  # dangling backslash
        '{"other": "x"}',  # no input key
        '{"input"',  # no colon yet
        '{"input":  "  spaced',  # whitespace before value
    ]


def canonicalize_vectors() -> list[object]:
    return [
        None,
        "",
        "   ",
        '{"b": 1, "a": 2}',  # sorted-key re-serialization
        '{"z": {"y": 1, "x": 2}, "a": 3}',  # nested sort
        "not json",  # unparseable → returned as-is
        {"b": 1, "a": 2},  # dict input
        [3, 1, 2],  # list input
        '{"unicode": "世界"}',
    ]


def main() -> int:
    records: list[dict] = []

    for v in sanitize_vectors():
        records.append({"fn": "sanitize_string", "input": v, "output": sanitize_string(v)})

    for ns, name in flatten_vectors():
        records.append(
            {
                "fn": "flatten_namespace_tool_name",
                "input": [ns, name],
                "output": flatten_namespace_tool_name(ns, name),
            }
        )

    for v in short_sha_vectors():
        records.append(
            {"fn": "short_sha256_hex", "input": v, "output": short_sha256_hex(v.encode())}
        )

    for v in custom_input_vectors():
        records.append(
            {
                "fn": "custom_tool_input_from_chat_arguments",
                "input": v,
                "output": custom_tool_input_from_chat_arguments(v),
            }
        )

    for v in partial_input_vectors():
        records.append(
            {
                "fn": "partial_custom_tool_input_from_chat_arguments",
                "input": v,
                "output": partial_custom_tool_input_from_chat_arguments(v),
            }
        )

    for v in canonicalize_vectors():
        records.append(
            {
                "fn": "canonicalize_tool_arguments",
                # Preserve the input type distinction for the Rust side.
                "input": v,
                "output": canonicalize_tool_arguments(v),
            }
        )

    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} golden records to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
