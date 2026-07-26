//! Conversion-layer microbenchmarks.
//!
//! Purpose: turn "the JSON conversion is negligible next to the upstream LLM
//! round-trip" from an assertion into a measurement. These bench the two pure,
//! CPU-bound conversion entrypoints in isolation — no network, no upstream —
//! so the numbers are the bridge's own per-turn overhead. Compare the reported
//! times (tens of microseconds expected) against a real upstream round-trip
//! (hundreds of ms to seconds): if conversion is ~4 orders of magnitude
//! smaller, chasing zero-copy/simd here buys nothing end-to-end.
//!
//! Run with: `cargo bench --features bench`

use codex_chat_bridge::bench_api::{
    build_tool_context_from_request, chat_to_responses, responses_to_chat_with_session,
    ResponsesRequest,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use serde_json::{json, Value};

/// A realistic inbound Responses request: a system instruction, a couple of
/// prior turns, a tool definition, and a fresh user message — the shape a real
/// Codex CLI turn carries.
fn sample_responses_request() -> ResponsesRequest {
    let payload = json!({
        "model": "gpt-4o",
        "instructions": "You are a helpful coding assistant. Be concise.",
        "input": [
            { "type": "message", "role": "user",
              "content": [{ "type": "input_text", "text": "List the files in the current directory." }] },
            { "type": "function_call", "call_id": "call_1", "name": "fs__list",
              "arguments": "{\"path\":\".\"}" },
            { "type": "function_call_output", "call_id": "call_1",
              "output": "Cargo.toml\nsrc\nbenches\ntests" },
            { "type": "message", "role": "user",
              "content": [{ "type": "input_text", "text": "Now read Cargo.toml and summarize the dependencies." }] }
        ],
        "tools": [
            { "type": "function", "name": "fs__list",
              "description": "List directory entries",
              "parameters": { "type": "object", "properties": { "path": { "type": "string" } } } },
            { "type": "function", "name": "fs__read",
              "description": "Read a file",
              "parameters": { "type": "object", "properties": { "path": { "type": "string" } } } }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": false
    });
    serde_json::from_value(payload).expect("valid ResponsesRequest fixture")
}

/// A realistic upstream Chat Completions body carrying an assistant message
/// plus a tool call — the response side the bridge must convert back.
fn sample_chat_body() -> Value {
    json!({
        "id": "chatcmpl-xyz",
        "model": "gpt-4o",
        "created": 1_700_000_000,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here are the dependencies grouped by purpose. The web layer uses axum and tokio; serialization is serde and serde_json; the HTTP client is reqwest over rustls.",
                "tool_calls": [{
                    "id": "call_9",
                    "type": "function",
                    "function": { "name": "fs__read", "arguments": "{\"path\":\"Cargo.toml\"}" }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 320, "completion_tokens": 88, "total_tokens": 408 }
    })
}

fn bench_responses_to_chat(c: &mut Criterion) {
    let payload = sample_responses_request();
    let ctx = build_tool_context_from_request(&payload);
    c.bench_function("responses_to_chat_with_session", |b| {
        b.iter(|| {
            let out = responses_to_chat_with_session(
                black_box(&payload),
                black_box("gpt-4o"),
                black_box(None),
                black_box(&ctx),
            );
            black_box(out);
        });
    });
}

fn bench_chat_to_responses(c: &mut Criterion) {
    let chat_body = sample_chat_body();
    let payload = sample_responses_request();
    let ctx = build_tool_context_from_request(&payload);
    let echo = json!({ "temperature": 0.2, "top_p": 0.9 });
    let echo_map = echo.as_object().cloned();
    c.bench_function("chat_to_responses", |b| {
        b.iter(|| {
            let out = chat_to_responses(
                black_box(&chat_body),
                black_box("gpt-4o"),
                black_box(echo_map.as_ref()),
                black_box("resp_bench"),
                black_box(&ctx),
            );
            black_box(out);
        });
    });
}

/// Scaling check: how conversion cost grows with conversation length. If it is
/// linear and the per-item slope is tiny, a long session is still negligible.
fn bench_responses_to_chat_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("responses_to_chat_by_turns");
    for turns in [1usize, 8, 32, 128] {
        let mut input = Vec::new();
        for i in 0..turns {
            input.push(json!({
                "type": "message", "role": "user",
                "content": [{ "type": "input_text", "text": format!("message number {i}") }]
            }));
        }
        let payload: ResponsesRequest = serde_json::from_value(json!({
            "model": "gpt-4o",
            "input": input,
            "stream": false
        }))
        .expect("valid scaling fixture");
        let ctx = build_tool_context_from_request(&payload);
        group.bench_with_input(BenchmarkId::from_parameter(turns), &turns, |b, _| {
            b.iter(|| {
                let out = responses_to_chat_with_session(
                    black_box(&payload),
                    black_box("gpt-4o"),
                    black_box(None),
                    black_box(&ctx),
                );
                black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_responses_to_chat,
    bench_chat_to_responses,
    bench_responses_to_chat_scaling
);
criterion_main!(benches);
