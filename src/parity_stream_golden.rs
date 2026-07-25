//! Python↔Rust *frame-level semantic* streaming SSE parity test.
//!
//! Counterpart to `parity_golden.rs`, targeting the highest-risk surface in the
//! bridge: the streaming state machine that turns an upstream Chat Completions
//! SSE byte stream into Responses SSE event bytes. Loads the fixture generated
//! by `tests/parity/generate_stream_golden.py` and asserts the Rust port
//! (`create_responses_sse_stream` for the live path, `sse_events_from_buffered_chat`
//! for the buffered path) produces the same SSE event stream.
//!
//! ## Why frame-level semantic, not byte-level
//!
//! Both sides emit a sequence of SSE frames whose `data:` payload is JSON. The
//! wire contract that clients actually parse is: *the ordered sequence of
//! `(event_name, json_value)` frames*. JSON object key order and inter-token
//! whitespace are semantically irrelevant — Python's `json.dumps` preserves
//! insertion order and pads with `", "`/`": "`, while serde_json emits compact,
//! key-sorted output. Comparing raw bytes would couple the Rust port to
//! Python's serializer cosmetics and flag non-differences. Instead we parse both
//! streams into frames and compare event-name sequences plus JSON *values*
//! (`serde_json::Value` compares objects as maps, order-independent). This still
//! catches every real divergence: wrong event, wrong/missing field, wrong value,
//! wrong frame ordering, or a dropped/extra frame.
//!
//! The single non-deterministic field, `created_at`, is zeroed recursively in
//! each parsed value before comparison (the Python generator normalizes it too).
//! Regenerate after any intentional behavior change:
//!
//! ```text
//! uv run --project /opt/codex-chat-bridge python \
//!     tests/parity/generate_stream_golden.py
//! ```

#![cfg(test)]

use base64::Engine;
use futures::{stream, StreamExt};
use serde_json::Value;

use crate::context::BridgeToolContext;
use crate::sse::{extract_block, parse_sse_block};
use crate::stream_chat_to_responses::{create_responses_sse_stream, sse_events_from_buffered_chat};

/// Embedded at compile time: no runtime IO, and a removed fixture is a build
/// error rather than a silent skip.
const STREAM_GOLDEN: &str = include_str!("../tests/parity/stream_golden.json");

/// One decoded SSE frame: an optional `event:` name and its parsed JSON `data:`
/// payload. The terminal `[DONE]` sentinel is represented as `Value::Null` data
/// so it participates in ordering comparison without special-casing.
#[derive(Debug, PartialEq)]
struct Frame {
    event: Option<String>,
    data: Value,
}

/// Recursively replace every `created_at` value with `0` so the one wall-clock
/// field is ignored structurally (no string/regex fragility).
fn zero_created_at(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if key == "created_at" {
                    *child = Value::from(0);
                } else {
                    zero_created_at(child);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(zero_created_at),
        _ => {}
    }
}

/// Parse a full SSE byte stream into ordered `(event, json)` frames, applying
/// the `created_at` normalization. Shared by both the expected (fixture) and
/// actual (Rust output) sides so they are compared through the identical lens.
fn parse_frames(raw: &[u8]) -> Vec<Frame> {
    let text = String::from_utf8(raw.to_vec()).expect("SSE output is valid UTF-8");
    let mut frames = Vec::new();
    let mut rest = text;
    while let Some((block, remainder)) = extract_block(&rest) {
        rest = remainder;
        if block.trim().is_empty() {
            continue;
        }
        let (event, data) = parse_sse_block(&block);
        let Some(data) = data else { continue };
        let mut value = if data.trim() == "[DONE]" {
            Value::Null
        } else {
            serde_json::from_str(&data).expect("SSE data payload is valid JSON")
        };
        zero_created_at(&mut value);
        frames.push(Frame { event, data: value });
    }
    frames
}

fn response_id_of(record: &Value) -> String {
    record["response_id"]
        .as_str()
        .expect("record has response_id")
        .to_owned()
}

/// Drive the live streaming path: decode the base64 upstream byte frames, feed
/// them through `create_responses_sse_stream`, and concatenate the output.
async fn run_stream_record(record: &Value) -> Vec<u8> {
    let frames: Vec<Vec<u8>> = record["frames_b64"]
        .as_array()
        .expect("stream record has frames_b64")
        .iter()
        .map(|f| {
            base64::engine::general_purpose::STANDARD
                .decode(f.as_str().expect("frame is base64 string"))
                .expect("frame decodes")
        })
        .collect();

    let out = create_responses_sse_stream(
        stream::iter(frames),
        BridgeToolContext::new(),
        Some(response_id_of(record)),
        None,
        None,
    );
    let events: Vec<Vec<u8>> = out.collect().await;
    events.concat()
}

/// Drive the buffered (non-streaming) path through `sse_events_from_buffered_chat`.
fn run_buffered_record(record: &Value) -> Vec<u8> {
    let chat_body = &record["chat_body"];
    let response_id = response_id_of(record);
    sse_events_from_buffered_chat(
        chat_body,
        BridgeToolContext::new(),
        Some(&response_id),
        None,
    )
    .concat()
}

/// Render a frame sequence as a readable diff aid: one `event → compact_json`
/// line per frame.
fn describe(frames: &[Frame]) -> String {
    frames
        .iter()
        .map(|f| {
            format!(
                "{} → {}",
                f.event.as_deref().unwrap_or("(no event)"),
                serde_json::to_string(&f.data).unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test]
async fn rust_stream_output_matches_python_golden() {
    let records: Vec<Value> =
        serde_json::from_str(STREAM_GOLDEN).expect("stream_golden.json parses");
    assert!(!records.is_empty(), "stream golden fixture is empty");

    let mut mismatches = Vec::new();
    for record in &records {
        let name = record["name"].as_str().expect("record has name");
        let mode = record["mode"].as_str().expect("record has mode");
        let expected = parse_frames(
            record["output"]
                .as_str()
                .expect("record has output")
                .as_bytes(),
        );

        let actual_raw = match mode {
            "stream" => run_stream_record(record).await,
            "buffered" => run_buffered_record(record),
            other => panic!("unknown parity mode in fixture: {other}"),
        };
        let actual = parse_frames(&actual_raw);

        if actual != expected {
            mismatches.push(format!(
                "--- MISMATCH [{name}] (mode={mode}) ---\nEXPECTED ({} frames):\n{}\n\nACTUAL ({} frames):\n{}",
                expected.len(),
                describe(&expected),
                actual.len(),
                describe(&actual),
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "Rust streaming output diverged from the Python golden fixture in {} case(s):\n\n{}",
        mismatches.len(),
        mismatches.join("\n\n")
    );
}
