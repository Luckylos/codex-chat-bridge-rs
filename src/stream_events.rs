//! Responses SSE event constructors — thin builders over [`crate::sse`].
//!
//! Mirrors the Python bridge's `stream_state/tool_events.py`. Every function
//! renders one Responses streaming event: it wraps the typed payload in the
//! `{ "type": <event>, ... }` envelope and serializes it as a single SSE frame
//! whose `event:` name matches the payload `type`.
//!
//! These are shared by the reasoning / message / tool increment state machines
//! that land in the following layers, so they read as dead until those wire in.

use serde_json::{json, Value};

use crate::sse::serialize_event;

pub fn output_item_added(output_index: i64, item: Value) -> Vec<u8> {
    let payload = json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": item,
    });
    serialize_event(Some("response.output_item.added"), &payload)
}

pub fn output_item_done(output_index: i64, item: Value) -> Vec<u8> {
    let payload = json!({
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": item,
    });
    serialize_event(Some("response.output_item.done"), &payload)
}

pub fn content_part_added(
    item_id: &str,
    output_index: i64,
    content_index: usize,
    part: Value,
) -> Vec<u8> {
    let payload = json!({
        "type": "response.content_part.added",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": part,
    });
    serialize_event(Some("response.content_part.added"), &payload)
}

pub fn content_part_done(
    item_id: &str,
    output_index: i64,
    content_index: usize,
    part: Value,
) -> Vec<u8> {
    let payload = json!({
        "type": "response.content_part.done",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "part": part,
    });
    serialize_event(Some("response.content_part.done"), &payload)
}

pub fn output_text_delta(
    item_id: &str,
    output_index: i64,
    content_index: usize,
    delta: &str,
) -> Vec<u8> {
    let payload = json!({
        "type": "response.output_text.delta",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "delta": delta,
    });
    serialize_event(Some("response.output_text.delta"), &payload)
}

pub fn output_text_done(
    item_id: &str,
    output_index: i64,
    content_index: usize,
    text: &str,
) -> Vec<u8> {
    let payload = json!({
        "type": "response.output_text.done",
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "text": text,
    });
    serialize_event(Some("response.output_text.done"), &payload)
}

pub fn reasoning_summary_part_added(
    item_id: &str,
    output_index: i64,
    summary_index: usize,
    part: Value,
) -> Vec<u8> {
    let payload = json!({
        "type": "response.reasoning_summary_part.added",
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "part": part,
    });
    serialize_event(Some("response.reasoning_summary_part.added"), &payload)
}

pub fn reasoning_summary_part_done(
    item_id: &str,
    output_index: i64,
    summary_index: usize,
    part: Value,
) -> Vec<u8> {
    let payload = json!({
        "type": "response.reasoning_summary_part.done",
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "part": part,
    });
    serialize_event(Some("response.reasoning_summary_part.done"), &payload)
}

pub fn reasoning_summary_text_delta(
    item_id: &str,
    output_index: i64,
    summary_index: usize,
    delta: &str,
) -> Vec<u8> {
    let payload = json!({
        "type": "response.reasoning_summary_text.delta",
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "delta": delta,
    });
    serialize_event(Some("response.reasoning_summary_text.delta"), &payload)
}

pub fn reasoning_summary_text_done(
    item_id: &str,
    output_index: i64,
    summary_index: usize,
    text: &str,
) -> Vec<u8> {
    let payload = json!({
        "type": "response.reasoning_summary_text.done",
        "item_id": item_id,
        "output_index": output_index,
        "summary_index": summary_index,
        "text": text,
    });
    serialize_event(Some("response.reasoning_summary_text.done"), &payload)
}

pub fn function_arguments_delta(item_id: &str, output_index: i64, delta: &str) -> Vec<u8> {
    let payload = json!({
        "type": "response.function_call_arguments.delta",
        "item_id": item_id,
        "output_index": output_index,
        "delta": delta,
    });
    serialize_event(Some("response.function_call_arguments.delta"), &payload)
}

pub fn function_arguments_done(item_id: &str, output_index: i64, arguments: &str) -> Vec<u8> {
    let payload = json!({
        "type": "response.function_call_arguments.done",
        "item_id": item_id,
        "output_index": output_index,
        "arguments": arguments,
    });
    serialize_event(Some("response.function_call_arguments.done"), &payload)
}

pub fn custom_input_delta(item_id: &str, output_index: i64, delta: &str) -> Vec<u8> {
    let payload = json!({
        "type": "response.custom_tool_call_input.delta",
        "item_id": item_id,
        "output_index": output_index,
        "delta": delta,
    });
    serialize_event(Some("response.custom_tool_call_input.delta"), &payload)
}

pub fn custom_input_done(item_id: &str, output_index: i64, input_text: &str) -> Vec<u8> {
    let payload = json!({
        "type": "response.custom_tool_call_input.done",
        "item_id": item_id,
        "output_index": output_index,
        "input": input_text,
    });
    serialize_event(Some("response.custom_tool_call_input.done"), &payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::{extract_block, parse_sse_block};

    /// Parse a serialized event frame back into (event_name, payload).
    fn parse(bytes: &[u8]) -> (String, Value) {
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let (block, rest) = extract_block(&text).expect("one complete frame");
        assert_eq!(rest, "", "exactly one frame per event");
        let (event, data) = parse_sse_block(&block);
        let payload: Value = serde_json::from_str(&data.expect("data line")).unwrap();
        (event.expect("event line"), payload)
    }

    #[test]
    fn event_name_matches_payload_type() {
        let (event, payload) = parse(&output_item_added(0, json!({ "id": "x" })));
        assert_eq!(event, "response.output_item.added");
        assert_eq!(payload["type"], json!("response.output_item.added"));
        assert_eq!(payload["output_index"], json!(0));
        assert_eq!(payload["item"], json!({ "id": "x" }));
    }

    #[test]
    fn reasoning_summary_events_carry_indices() {
        let (event, payload) = parse(&reasoning_summary_text_delta("rs_1", 2, 0, "think"));
        assert_eq!(event, "response.reasoning_summary_text.delta");
        assert_eq!(payload["item_id"], json!("rs_1"));
        assert_eq!(payload["output_index"], json!(2));
        assert_eq!(payload["summary_index"], json!(0));
        assert_eq!(payload["delta"], json!("think"));
    }

    #[test]
    fn text_events_carry_content_index() {
        let (event, payload) = parse(&output_text_delta("msg_1", 1, 0, "hi"));
        assert_eq!(event, "response.output_text.delta");
        assert_eq!(payload["content_index"], json!(0));
        assert_eq!(payload["delta"], json!("hi"));
    }

    #[test]
    fn function_and_custom_input_events_render() {
        let (event, payload) = parse(&function_arguments_done("fc_1", 3, "{}"));
        assert_eq!(event, "response.function_call_arguments.done");
        assert_eq!(payload["arguments"], json!("{}"));

        let (event, payload) = parse(&custom_input_done("ctc_1", 4, "raw"));
        assert_eq!(event, "response.custom_tool_call_input.done");
        assert_eq!(payload["input"], json!("raw"));
    }
}
