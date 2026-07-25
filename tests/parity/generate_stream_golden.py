#!/usr/bin/env python3
"""Generate the Python↔Rust *streaming* SSE parity golden fixture.

This is the counterpart to `generate_golden.py`, targeting the highest-risk
surface in the whole bridge: the streaming state machine that turns an upstream
Chat Completions SSE byte stream into Responses SSE event bytes.

For each curated upstream scenario we drive the authoritative Python converter
`create_responses_sse_stream_from_chat_stream` (or the buffered-response variant
`create_responses_sse_from_chat_response`), capture the exact output bytes,
normalize the single non-deterministic field (`created_at`), and record
`{name, mode, response_id, frames_b64, output}`. The Rust integration test
`src/parity_stream_golden.rs` loads the same fixture, drives
`create_responses_sse_stream` / `sse_events_from_buffered_chat`, applies the
identical normalization, and asserts byte-identical output.

Determinism contract:
    * `response_id` is always supplied explicitly (never minted).
    * every tool call carries an explicit `id` (no `call_auto_<uuid>`).
    * `created_at` is the only remaining wall-clock field; both sides normalize
      it to 0 before comparison.

Run from the Python bridge venv:
    uv run --project /opt/codex-chat-bridge python \\
        /opt/codex-chat-bridge-rs/tests/parity/generate_stream_golden.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
from pathlib import Path

from codex_chat_bridge.stream_chat_to_responses import (
    create_responses_sse_from_chat_response,
    create_responses_sse_stream_from_chat_stream,
)

OUT = Path(__file__).with_name("stream_golden.json")
FIXED_ID = "resp_bridge_FIXEDID0001"

# The single non-deterministic field. Both sides normalize it before diffing.
_CREATED_AT = re.compile(rb'"created_at": ?\d+')


def _normalize(raw: bytes) -> bytes:
    return _CREATED_AT.sub(b'"created_at": 0', raw)


def sse(obj: dict) -> bytes:
    """Frame a chat-chunk dict as one upstream SSE `data:` block."""
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def delta_chunk(delta: dict, finish_reason: str | None = None) -> bytes:
    choice: dict = {"delta": delta}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return sse({"choices": [choice]})


DONE = b"data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Streaming scenarios: each is a list of raw upstream byte frames.
# --------------------------------------------------------------------------- #
def stream_scenarios() -> list[tuple[str, list[bytes]]]:
    scenarios: list[tuple[str, list[bytes]]] = []

    scenarios.append(
        (
            "simple_text",
            [
                delta_chunk({"role": "assistant", "content": "Hi"}),
                delta_chunk({"content": " there"}, finish_reason="stop"),
                DONE,
            ],
        )
    )

    scenarios.append(
        (
            "reasoning_then_text",
            [
                delta_chunk({"role": "assistant", "reasoning_content": "thinking..."}),
                delta_chunk({"reasoning_content": " more"}),
                delta_chunk({"content": "Answer"}, finish_reason="stop"),
                DONE,
            ],
        )
    )

    scenarios.append(
        (
            "single_tool_call",
            [
                delta_chunk({"role": "assistant"}),
                delta_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_fixed_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    }
                ),
                delta_chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"city":'}}
                        ]
                    }
                ),
                delta_chunk(
                    {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"SF"}'}}
                        ]
                    },
                    finish_reason="tool_calls",
                ),
                DONE,
            ],
        )
    )

    scenarios.append(
        (
            "parallel_tool_calls",
            [
                delta_chunk({"role": "assistant"}),
                delta_chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_fixed_a",
                                "type": "function",
                                "function": {"name": "fn_a", "arguments": "{}"},
                            },
                            {
                                "index": 1,
                                "id": "call_fixed_b",
                                "type": "function",
                                "function": {"name": "fn_b", "arguments": "{}"},
                            },
                        ]
                    },
                    finish_reason="tool_calls",
                ),
                DONE,
            ],
        )
    )

    scenarios.append(
        (
            "structured_content_parts",
            [
                delta_chunk(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "part one"},
                            {"type": "text", "text": "part two"},
                        ],
                    },
                    finish_reason="stop",
                ),
                DONE,
            ],
        )
    )

    scenarios.append(
        (
            "refusal",
            [
                delta_chunk({"role": "assistant", "refusal": "I cannot help"}, finish_reason="stop"),
                DONE,
            ],
        )
    )

    scenarios.append(
        (
            "error_event",
            [
                delta_chunk({"role": "assistant", "content": "partial"}),
                b'event: error\ndata: {"error": {"message": "upstream boom", "type": "server_error"}}\n\n',
            ],
        )
    )

    scenarios.append(
        (
            "inline_error_field",
            [
                b'data: {"error": {"message": "inline failure", "type": "rate_limit"}}\n\n',
            ],
        )
    )

    scenarios.append(
        (
            "finish_length",
            [
                delta_chunk({"role": "assistant", "content": "truncated"}, finish_reason="length"),
                DONE,
            ],
        )
    )

    # A multibyte character (世 = e4 b8 96) split across two byte frames mid-JSON
    # string. The incremental decoder must hold the partial sequence.
    world = "世界".encode()
    split_point = 3  # after first full char, mid-second
    frame_a = b'data: {"choices":[{"delta":{"role":"assistant","content":"' + world[:split_point]
    frame_b = world[split_point:] + b'"}}]}\n\n'
    scenarios.append(
        (
            "multibyte_split_across_frames",
            [
                frame_a,
                frame_b,
                delta_chunk({}, finish_reason="stop"),
                DONE,
            ],
        )
    )

    scenarios.append(
        (
            "malformed_json_skipped",
            [
                b'data: {this is not valid json}\n\n',
                delta_chunk({"role": "assistant", "content": "recovered"}, finish_reason="stop"),
                DONE,
            ],
        )
    )

    # Residual frame delivered without the trailing blank-line delimiter and no
    # [DONE]: the converter must still parse the dangling frame at stream end.
    scenarios.append(
        (
            "residual_frame_no_trailing_blank",
            [
                delta_chunk({"role": "assistant", "content": "first"}),
                b'data: {"choices":[{"delta":{"content":" last"},"finish_reason":"stop"}]}',
            ],
        )
    )

    # Stream ends without [DONE]: finalize with stream_ended_cleanly=false, which
    # marks the turn incomplete rather than completed.
    scenarios.append(
        (
            "no_done_truncated",
            [
                delta_chunk({"role": "assistant", "content": "unfinished"}),
            ],
        )
    )

    # Multiple small content deltas exercising repeated push_content_delta.
    scenarios.append(
        (
            "many_small_deltas",
            [
                delta_chunk({"role": "assistant", "content": "a"}),
                delta_chunk({"content": "b"}),
                delta_chunk({"content": "c"}),
                delta_chunk({"content": "d"}, finish_reason="stop"),
                DONE,
            ],
        )
    )

    # Control characters in content — must be stripped by sanitize_string.
    scenarios.append(
        (
            "content_with_control_chars",
            [
                delta_chunk(
                    {"role": "assistant", "content": "clean\x01dirty\x1f end"},
                    finish_reason="stop",
                ),
                DONE,
            ],
        )
    )

    return scenarios


# --------------------------------------------------------------------------- #
# Buffered (non-streaming) scenarios: a full Chat Completions body.
# --------------------------------------------------------------------------- #
def buffered_scenarios() -> list[tuple[str, dict]]:
    return [
        (
            "buffered_text",
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello world"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
        (
            "buffered_parallel_tool_calls",
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_buf_a",
                                    "type": "function",
                                    "function": {"name": "fn_a", "arguments": "{}"},
                                },
                                {
                                    "id": "call_buf_b",
                                    "type": "function",
                                    "function": {"name": "fn_b", "arguments": '{"x":1}'},
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ),
        (
            "buffered_reasoning",
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "final",
                            "reasoning_content": "because reasons",
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
    ]


async def _drive_stream(frames: list[bytes]) -> bytes:
    async def _chunks():
        for f in frames:
            yield f

    out: list[bytes] = []
    async for ev in create_responses_sse_stream_from_chat_stream(
        _chunks(), tool_context=None, response_id=FIXED_ID
    ):
        out.append(ev)
    return b"".join(out)


async def _drive_buffered(chat_body: dict) -> bytes:
    out: list[bytes] = []
    async for ev in create_responses_sse_from_chat_response(
        chat_body, tool_context=None, response_id=FIXED_ID
    ):
        out.append(ev)
    return b"".join(out)


async def main() -> int:
    records: list[dict] = []

    for name, frames in stream_scenarios():
        raw = _normalize(await _drive_stream(frames))
        records.append(
            {
                "name": name,
                "mode": "stream",
                "response_id": FIXED_ID,
                "frames_b64": [base64.b64encode(f).decode() for f in frames],
                "output": raw.decode("utf-8"),
            }
        )

    for name, chat_body in buffered_scenarios():
        raw = _normalize(await _drive_buffered(chat_body))
        records.append(
            {
                "name": name,
                "mode": "buffered",
                "response_id": FIXED_ID,
                "chat_body": chat_body,
                "output": raw.decode("utf-8"),
            }
        )

    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} stream golden records to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
