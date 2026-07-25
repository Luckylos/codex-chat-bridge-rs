//! Server-Sent Events frame codec — pure functions, no async or I/O.
//!

use serde_json::Value;

/// SSE frame delimiter: a blank line (two consecutive LFs).
const FRAME_DELIMITER: &str = "\n\n";

/// Extract the first complete SSE frame from `buffer`.
///
/// Returns `(frame, remaining)` or `None` when no complete frame is present
/// yet. CRLF line endings are normalized to LF first, so `\r\n\r\n`-delimited
/// frames split correctly and the returned remainder is already normalized.
pub fn extract_block(buffer: &str) -> Option<(String, String)> {
    let normalized = buffer.replace("\r\n", "\n");
    let idx = normalized.find(FRAME_DELIMITER)?;
    let block = normalized[..idx].to_owned();
    let rest = normalized[idx + FRAME_DELIMITER.len()..].to_owned();
    Some((block, rest))
}

/// Split an SSE frame into its event name and reassembled data payload.
///
/// Multiple `data:` lines are joined with LF (per the SSE spec). Exactly one
/// leading space after `data:` is stripped when present — intentional
/// multi-space content is preserved. Returns `(None, None)` for a frame that
/// carries neither field.
pub fn parse_sse_block(block: &str) -> (Option<String>, Option<String>) {
    let mut event_name: Option<String> = None;
    let mut data_parts: Vec<String> = Vec::new();

    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = Some(rest.trim().to_owned());
        } else if let Some(rest) = line.strip_prefix("data:") {
            // Strip exactly one leading space, not all whitespace.
            let value = rest.strip_prefix(' ').unwrap_or(rest);
            data_parts.push(value.to_owned());
        }
    }

    let data = if data_parts.is_empty() {
        None
    } else {
        Some(data_parts.join("\n"))
    };
    (event_name, data)
}

/// Serialize a single SSE event to bytes.
///
/// When `event` is `None` no `event:` line is written. `data` is rendered as
/// JSON with non-ASCII preserved (serde_json's default), matching the Python
/// `ensure_ascii=False`.
pub fn serialize_event(event: Option<&str>, data: &Value) -> Vec<u8> {
    let mut out = String::new();
    if let Some(name) = event {
        out.push_str("event: ");
        out.push_str(name);
        out.push('\n');
    }
    out.push_str("data: ");
    out.push_str(&serde_json::to_string(data).unwrap_or_else(|_| "null".to_owned()));
    out.push('\n');
    out.push('\n');
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_block_returns_frame_and_remainder() {
        let (block, rest) = extract_block("event: x\ndata: 1\n\nleftover").unwrap();
        assert_eq!(block, "event: x\ndata: 1");
        assert_eq!(rest, "leftover");
    }

    #[test]
    fn extract_block_none_without_delimiter() {
        assert!(extract_block("event: x\ndata: 1\n").is_none());
    }

    #[test]
    fn extract_block_normalizes_crlf() {
        let (block, rest) = extract_block("data: a\r\n\r\ndata: b\r\n\r\n").unwrap();
        assert_eq!(block, "data: a");
        // Remainder is already LF-normalized for the next scan.
        assert_eq!(rest, "data: b\n\n");
    }

    #[test]
    fn parse_block_splits_event_and_data() {
        let (event, data) = parse_sse_block("event: response.created\ndata: {\"a\":1}");
        assert_eq!(event.as_deref(), Some("response.created"));
        assert_eq!(data.as_deref(), Some("{\"a\":1}"));
    }

    #[test]
    fn parse_block_joins_multiple_data_lines() {
        let (_, data) = parse_sse_block("data: line1\ndata: line2");
        assert_eq!(data.as_deref(), Some("line1\nline2"));
    }

    #[test]
    fn parse_block_strips_only_one_leading_space() {
        let (_, data) = parse_sse_block("data:  two-space");
        // One space consumed, one preserved.
        assert_eq!(data.as_deref(), Some(" two-space"));
    }

    #[test]
    fn parse_block_without_fields_is_empty() {
        let (event, data) = parse_sse_block(": comment only");
        assert!(event.is_none());
        assert!(data.is_none());
    }

    #[test]
    fn serialize_event_with_name() {
        let bytes = serialize_event(Some("response.created"), &json!({ "id": "r1" }));
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "event: response.created\ndata: {\"id\":\"r1\"}\n\n"
        );
    }

    #[test]
    fn serialize_event_without_name_omits_event_line() {
        let bytes = serialize_event(None, &json!({ "x": 1 }));
        assert_eq!(String::from_utf8(bytes).unwrap(), "data: {\"x\":1}\n\n");
    }

    #[test]
    fn serialize_event_preserves_non_ascii() {
        let bytes = serialize_event(None, &json!({ "msg": "café" }));
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "data: {\"msg\":\"café\"}\n\n"
        );
    }

    #[test]
    fn round_trip_extract_then_parse() {
        let frame = serialize_event(Some("response.completed"), &json!({ "status": "ok" }));
        let buffer = String::from_utf8(frame).unwrap();
        let (block, rest) = extract_block(&buffer).unwrap();
        assert!(rest.is_empty());
        let (event, data) = parse_sse_block(&block);
        assert_eq!(event.as_deref(), Some("response.completed"));
        assert_eq!(data.as_deref(), Some("{\"status\":\"ok\"}"));
    }
}
