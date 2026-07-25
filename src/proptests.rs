//! Property-based invariants for the pure streaming primitives.
//!
//! These complement the golden/parity fixtures (which pin exact bytes for known
//! inputs) with universally-quantified laws that must hold for *any* input:
//!
//! * the incremental UTF-8 decoder is **chunk-boundary independent** — the text
//!   it yields depends only on the byte stream, never on where that stream was
//!   split into chunks; and it agrees with `String::from_utf8_lossy`;
//! * the SSE codec **round-trips** — serializing an event then extracting and
//!   parsing the frame recovers the event name and the JSON data value.
//!
//! Invariants are deliberately self-referential (decoder-vs-itself, codec
//! round-trip) rather than pinned to a second implementation, so they keep
//! holding across future refactors without over-constraining the output.
#![cfg(test)]

use proptest::prelude::*;
use serde_json::{json, Value};

use crate::sse::{extract_block, parse_sse_block, serialize_event};
use crate::stream_chat_to_responses::Utf8StreamDecoder;

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
}
