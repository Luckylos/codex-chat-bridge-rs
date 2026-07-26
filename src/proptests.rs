//! Property-based invariants for the pure streaming primitives.
//!
//! These complement the golden/parity fixtures (which pin exact bytes for known
//! inputs) with universally-quantified laws that must hold for *any* input:
//!
//! * the incremental UTF-8 decoder is **chunk-boundary independent** — the text
//!   it yields depends only on the byte stream, never on where that stream was
//!   split into chunks; and it agrees with `String::from_utf8_lossy`;
//! * the SSE codec **round-trips** — serializing an event then extracting and
//!   parsing the frame recovers the event name and the JSON data value;
//! * the two conversion entrypoints are **total** — for *any* upstream JSON,
//!   however malformed or deeply nested, they neither panic nor produce a
//!   structurally-invalid result. This matters because the converters consume
//!   untrusted upstream bytes on the hot path, where a panic is a DoS.
//!
//! Invariants are deliberately self-referential (decoder-vs-itself, codec
//! round-trip) rather than pinned to a second implementation, so they keep
//! holding across future refactors without over-constraining the output.
#![cfg(test)]

use proptest::prelude::*;
use serde_json::{json, Value};

use crate::context::BridgeToolContext;
use crate::convert::{chat_to_responses, responses_to_chat_with_session};
use crate::sse::{extract_block, parse_sse_block, serialize_event};
use crate::stream_chat_to_responses::Utf8StreamDecoder;
use crate::types::ResponsesRequest;

/// A recursive strategy for arbitrary JSON, bounded in depth and breadth so the
/// runs stay fast while still generating the shapes the converters branch on:
/// nulls, every scalar, and nested objects/arrays. Object keys are drawn from a
/// small alphabet plus a few protocol-significant names (`type`, `role`,
/// `content`, `tool_calls`, ...) so the fuzzer actually lands on the dispatch
/// arms rather than only on unrecognized keys.
fn arb_json() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::from),
        any::<i64>().prop_map(Value::from),
        prop_oneof![
            "[a-z]{1,6}",
            Just("type".to_owned()),
            Just("role".to_owned()),
            Just("content".to_owned()),
            Just("text".to_owned()),
            Just("tool_calls".to_owned()),
            Just("input_text".to_owned()),
            Just("function_call".to_owned()),
            Just("input_image".to_owned()),
        ]
        .prop_map(Value::from),
    ];
    leaf.prop_recursive(4, 32, 6, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..6).prop_map(Value::Array),
            proptest::collection::hash_map(
                prop_oneof![
                    "[a-z]{1,6}",
                    Just("type".to_owned()),
                    Just("role".to_owned()),
                    Just("content".to_owned()),
                    Just("tool_calls".to_owned()),
                    Just("arguments".to_owned()),
                ],
                inner,
                0..6,
            )
            .prop_map(|m| Value::Object(m.into_iter().collect())),
        ]
    })
}

/// Structural validity of a `chat_to_responses` result: it is always an object
/// carrying the fixed Responses envelope, whatever the upstream body was.
fn assert_valid_response_object(v: &Value) {
    let obj = v.as_object().expect("response is a JSON object");
    assert_eq!(obj.get("object"), Some(&json!("response")));
    assert!(
        obj.get("output").is_some_and(Value::is_array),
        "output is an array"
    );
    assert!(
        obj.get("status").is_some_and(Value::is_string),
        "status is a string"
    );
}

/// Feed `data` to a fresh decoder in the given `chunk_sizes`, then finalize.
/// Sizes that overrun the remaining bytes are clamped, and any leftover tail is
/// fed as a final chunk, so an arbitrary size vector always consumes all bytes.
fn decode_in_chunks(data: &[u8], chunk_sizes: &[usize]) -> String {
    let mut decoder = Utf8StreamDecoder::new();
    let mut out = String::new();
    let mut cursor = 0;
    for &size in chunk_sizes {
        if cursor >= data.len() {
            break;
        }
        let end = (cursor + size.max(1)).min(data.len());
        out.push_str(&decoder.decode(&data[cursor..end]));
        cursor = end;
    }
    if cursor < data.len() {
        out.push_str(&decoder.decode(&data[cursor..]));
    }
    out.push_str(&decoder.finalize());
    out
}

proptest! {
    /// However the same byte stream is chunked, the decoded text is identical:
    /// the split points are invisible to the output.
    #[test]
    fn decoder_is_chunk_boundary_independent(
        data in proptest::collection::vec(any::<u8>(), 0..256),
        splits_a in proptest::collection::vec(1usize..8, 0..64),
        splits_b in proptest::collection::vec(1usize..8, 0..64),
    ) {
        let whole = decode_in_chunks(&data, &[data.len().max(1)]);
        let chunked_a = decode_in_chunks(&data, &splits_a);
        let chunked_b = decode_in_chunks(&data, &splits_b);
        prop_assert_eq!(&whole, &chunked_a);
        prop_assert_eq!(&whole, &chunked_b);
    }

    /// The decoder agrees with the standard lossy decode for the full stream.
    #[test]
    fn decoder_matches_from_utf8_lossy(
        data in proptest::collection::vec(any::<u8>(), 0..256),
        splits in proptest::collection::vec(1usize..8, 0..64),
    ) {
        let expected = String::from_utf8_lossy(&data).into_owned();
        prop_assert_eq!(decode_in_chunks(&data, &splits), expected);
    }

    /// Valid UTF-8 always decodes verbatim with no replacements, regardless of
    /// how it is chunked.
    #[test]
    fn decoder_preserves_valid_utf8(
        text in ".*",
        splits in proptest::collection::vec(1usize..8, 0..64),
    ) {
        prop_assert_eq!(decode_in_chunks(text.as_bytes(), &splits), text);
    }

    /// Serializing an event and then extracting + parsing the frame recovers the
    /// original event name and JSON data value. `data:`-only frames (no event
    /// line) are covered by the `None` name case.
    #[test]
    fn sse_event_round_trips(
        name in proptest::option::of("[a-zA-Z][a-zA-Z0-9._]{0,40}"),
        s in ".*",
        n in any::<i64>(),
        b in any::<bool>(),
    ) {
        let data = json!({ "s": s, "n": n, "b": b });
        let bytes = serialize_event(name.as_deref(), &data);
        let text = String::from_utf8(bytes).unwrap();

        let (frame, rest) = extract_block(&text).expect("a serialized event is one complete frame");
        prop_assert_eq!(rest, "");

        let (got_name, got_data) = parse_sse_block(&frame);
        prop_assert_eq!(got_name, name);
        let parsed: Value = serde_json::from_str(&got_data.expect("data payload present")).unwrap();
        prop_assert_eq!(parsed, data);
    }

    /// `extract_block` splits at the first blank line: the returned frame never
    /// contains the delimiter, and re-joining with it reconstructs the
    /// (newline-normalized) buffer.
    #[test]
    fn extract_block_splits_at_first_blank_line(
        head in "[^\r\n]{0,40}",
        tail in ".{0,40}",
    ) {
        let buffer = format!("{head}\n\n{tail}");
        let (frame, rest) = extract_block(&buffer).expect("delimiter present");
        prop_assert_eq!(&frame, &head);
        prop_assert_eq!(rest, tail.replace("\r\n", "\n"));
    }

    /// `chat_to_responses` is total over upstream bodies: any JSON — a bare
    /// scalar, an array, an object with garbage `choices`/`message` shapes —
    /// converts without panicking and yields a structurally-valid Responses
    /// object. This machine-proves the invariant the hand audit of the
    /// `.expect`/`.unwrap` sites can only argue informally.
    #[test]
    fn chat_to_responses_never_panics_on_arbitrary_upstream(chat in arb_json()) {
        let ctx = BridgeToolContext::new();
        let out = chat_to_responses(&chat, "fallback", None, "resp_prop", &ctx);
        assert_valid_response_object(&out);
    }

    /// The same totality guarantee with the request-echo path engaged: an
    /// arbitrary `original_request` map must not change the no-panic /
    /// valid-shape outcome.
    #[test]
    fn chat_to_responses_never_panics_with_echo(
        chat in arb_json(),
        echo in arb_json(),
    ) {
        let ctx = BridgeToolContext::new();
        let echo_map = echo.as_object().cloned();
        let out = chat_to_responses(&chat, "fallback", echo_map.as_ref(), "resp_prop", &ctx);
        assert_valid_response_object(&out);
    }

    /// `responses_to_chat_with_session` is total over inbound requests: any
    /// JSON that deserializes into a `ResponsesRequest` (all fields default, so
    /// most shapes do) converts without panicking and yields a Chat body whose
    /// `messages` is an array. Undeserializable inputs are out of scope — the
    /// HTTP layer rejects those as 400 before this function is reached.
    #[test]
    fn responses_to_chat_never_panics_on_arbitrary_input(input in arb_json()) {
        let Ok(payload) = serde_json::from_value::<ResponsesRequest>(input) else {
            return Ok(());
        };
        let ctx = BridgeToolContext::new();
        let chat = responses_to_chat_with_session(&payload, "m", None, &ctx);
        prop_assert!(
            chat.body.get("messages").is_some_and(Value::is_array),
            "converted chat body always carries a messages array"
        );
    }
}
