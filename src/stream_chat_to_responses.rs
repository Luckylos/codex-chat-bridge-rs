//! Chat Completions → Responses SSE consumer.
//!
//! Drives a [`ResponsesStreamState`] from an upstream Chat Completions stream
//! (or a fully buffered Chat body), emitting Responses SSE event bytes. Mirrors
//! the Python bridge's `stream_chat_to_responses`.
//!
//! Two entry points share one chunk processor ([`process_chat_chunk`]):
//!
//! * [`sse_events_from_buffered_chat`] — a non-streaming upstream response is
//!   replayed as a single synthetic delta per choice, so the buffered path
//!   produces byte-identical SSE output to the streamed path.
//! * [`create_responses_sse_stream`] — a live upstream byte stream is decoded,
//!   framed, and processed chunk-by-chunk.
//!
//! The incremental [`Utf8StreamDecoder`] preserves multibyte characters split
//! across chunk boundaries and counts U+FFFD replacements so silent stream
//! corruption is observable via metrics rather than vanishing.

use async_stream::stream;
use futures::{Stream, StreamExt};
use serde_json::{Map, Value};

use crate::context::BridgeToolContext;
use crate::reasoning::extract_reasoning_field;
use crate::sanitize::sanitize_string;
use crate::sse::{extract_block, parse_sse_block};
use crate::stream_responses_state::ResponsesStreamState;

/// Build a fresh stream state seeded with the request echo.
fn new_stream_state(
    tool_context: BridgeToolContext,
    response_id: Option<&str>,
    original_request: Option<&Map<String, Value>>,
) -> ResponsesStreamState {
    let mut state = ResponsesStreamState::new(tool_context, response_id);
    state.set_request_echo(original_request);
    state
}

/// An incremental UTF-8 decoder that holds partial multibyte sequences across
/// chunk boundaries and replaces invalid bytes with U+FFFD.
///
/// Mirrors Python's `codecs.getincrementaldecoder("utf-8")(errors="replace")`:
/// a trailing incomplete sequence is buffered until the next chunk (or replaced
/// on the final flush), while genuinely invalid bytes are replaced immediately.
#[derive(Default)]
pub struct Utf8StreamDecoder {
    leftover: Vec<u8>,
    replacements: u64,
}

impl Utf8StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total U+FFFD replacements emitted so far.
    pub fn replacements(&self) -> u64 {
        self.replacements
    }

    /// Decode a chunk, returning the text that can be emitted now. Any trailing
    /// incomplete multibyte sequence is retained for the next call.
    pub fn decode(&mut self, chunk: &[u8]) -> String {
        let mut bytes = std::mem::take(&mut self.leftover);
        bytes.extend_from_slice(chunk);
        let mut out = String::new();
        let mut cursor = 0usize;

        loop {
            match std::str::from_utf8(&bytes[cursor..]) {
                Ok(valid) => {
                    out.push_str(valid);
                    cursor = bytes.len();
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    // Safe: valid_up_to marks a valid UTF-8 boundary.
                    out.push_str(
                        std::str::from_utf8(&bytes[cursor..cursor + valid_up_to]).unwrap(),
                    );
                    cursor += valid_up_to;
                    match err.error_len() {
                        Some(bad) => {
                            // A genuinely invalid sequence — replace and skip.
                            out.push('\u{FFFD}');
                            self.replacements += 1;
                            cursor += bad;
                        }
                        None => {
                            // Incomplete trailing sequence — hold for next chunk.
                            self.leftover = bytes[cursor..].to_vec();
                            cursor = bytes.len();
                            break;
                        }
                    }
                }
            }
        }
        debug_assert_eq!(cursor, bytes.len());
        out
    }

    /// Flush any held bytes at stream end, replacing an incomplete sequence.
    pub fn finalize(&mut self) -> String {
        if self.leftover.is_empty() {
            return String::new();
        }
        self.leftover.clear();
        self.replacements += 1;
        "\u{FFFD}".to_owned()
    }
}

/// Drain every complete SSE frame from `buffer`, returning parsed
/// `(event_name, data)` messages and the unconsumed remainder.
fn drain_sse_blocks(buffer: &str) -> (Vec<(Option<String>, String)>, String) {
    let mut messages = Vec::new();
    let mut rest = buffer.to_owned();
    while let Some((block, remainder)) = extract_block(&rest) {
        rest = remainder;
        if block.trim().is_empty() {
            continue;
        }
        let (event_name, data) = parse_sse_block(&block);
        match data {
            Some(data) if !data.is_empty() => messages.push((event_name, data)),
            _ => {}
        }
    }
    (messages, rest)
}

/// Process one already-parsed SSE message. `[DONE]` finalizes the stream;
/// otherwise the JSON payload is fed through [`process_chat_chunk`].
fn process_sse_message(
    event_name: Option<&str>,
    data: &str,
    state: &mut ResponsesStreamState,
) -> Vec<Vec<u8>> {
    if data.trim() == "[DONE]" {
        return state.finalize(true);
    }
    match serde_json::from_str::<Value>(data) {
        Ok(Value::Object(payload)) => process_chat_chunk(&payload, event_name, state),
        _ => {
            tracing::warn!(
                "Malformed SSE JSON object, skipping: {:?}",
                &data[..data.len().min(200)]
            );
            Vec::new()
        }
    }
}

/// Emit terminal `response.failed` events when a chunk carries an error, either
/// via an `error` SSE event or an inline `error` field. Returns `None` when the
/// payload is not an error.
fn error_events(
    payload: &Map<String, Value>,
    event_name: Option<&str>,
    state: &mut ResponsesStreamState,
) -> Option<Vec<Vec<u8>>> {
    if event_name != Some("error") && !payload.contains_key("error") {
        return None;
    }
    let err = payload.get("error").unwrap_or(&Value::Null);
    let (message, error_type) = match err {
        Value::Object(obj) => (
            obj.get("message").and_then(Value::as_str),
            obj.get("type")
                .and_then(Value::as_str)
                .unwrap_or("stream_error"),
        ),
        Value::Null => (None, "stream_error"),
        other => (other.as_str(), "stream_error"),
    };
    Some(state.fail(message.unwrap_or("upstream stream error"), error_type))
}

/// Route `tool_calls[]` deltas, attaching the active reasoning text (or the
/// per-chunk reasoning hint) to the first tool item.
fn tool_call_events(
    state: &mut ResponsesStreamState,
    tool_calls: &[Value],
    reasoning_hint: &str,
) -> Vec<Vec<u8>> {
    let mut events = Vec::new();
    let active_reasoning = state.active_reasoning_text_for_tools();
    let reasoning_text = if active_reasoning.is_empty() {
        reasoning_hint.to_owned()
    } else {
        active_reasoning.clone()
    };
    if !active_reasoning.is_empty() {
        events.extend(state.finalize_reasoning_if_open());
    }
    events.extend(state.force_inline_think_to_text());
    for tool_call in tool_calls {
        if tool_call.is_object() {
            let sanitized = if reasoning_text.is_empty() {
                None
            } else {
                Some(sanitize_string(&reasoning_text))
            };
            events.extend(state.push_tool_call_delta(tool_call, sanitized.as_deref()));
        }
    }
    events
}

/// Route a string `content` delta. When reasoning arrived in the same chunk the
/// reasoning item is closed first and the text bypasses inline-think detection
/// (the model already emitted structured reasoning).
fn string_content_events(
    state: &mut ResponsesStreamState,
    content: &str,
    reasoning_delta: &str,
) -> Vec<Vec<u8>> {
    if content.is_empty() {
        return Vec::new();
    }
    if !reasoning_delta.is_empty() {
        let mut events = state.finalize_reasoning_if_open();
        events.extend(state.push_text_delta(&sanitize_string(content)));
        return events;
    }
    state.push_content_delta(&sanitize_string(content))
}

/// Route a structured `content` array of `text` / `output_text` / `refusal`
/// parts, flushing reasoning and inline-think first.
fn structured_content_events(state: &mut ResponsesStreamState, content: &[Value]) -> Vec<Vec<u8>> {
    let mut events = state.finalize_reasoning_if_open();
    events.extend(state.force_inline_think_to_text());
    for (index, part) in content.iter().enumerate() {
        let Some(obj) = part.as_object() else {
            continue;
        };
        if index > 0 {
            events.extend(state.flush_open_text_part());
        }
        match obj.get("type").and_then(Value::as_str) {
            Some("text") | Some("output_text") => {
                if let Some(text) = obj
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    state.add_annotations(obj.get("annotations"));
                    events.extend(state.push_text_delta(&sanitize_string(text)));
                }
            }
            Some("refusal") => {
                if let Some(refusal) = obj
                    .get("refusal")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    events.extend(state.push_refusal_part(&sanitize_string(refusal)));
                }
            }
            other => {
                tracing::debug!(
                    "Skipping unhandled structured content part type: {:?}",
                    other
                );
            }
        }
    }
    events
}

/// Process a single Chat Completions chunk through the state machine, returning
/// the SSE event bytes it produced. Shared by the streamed and buffered paths.
pub fn process_chat_chunk(
    payload: &Map<String, Value>,
    event_name: Option<&str>,
    state: &mut ResponsesStreamState,
) -> Vec<Vec<u8>> {
    if let Some(events) = error_events(payload, event_name, state) {
        return events;
    }

    state.apply_chunk_metadata(&Value::Object(payload.clone()));
    let mut events = state.ensure_started();

    let choice = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .cloned()
        .unwrap_or(Value::Null);
    let delta = choice
        .get("delta")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let reasoning_delta = extract_reasoning_field(&delta).unwrap_or_default();
    if !reasoning_delta.is_empty() {
        events.extend(state.push_reasoning_delta(&sanitize_string(&reasoning_delta)));
    }

    state.add_annotations(delta.get("annotations"));

    match delta.get("content") {
        Some(Value::String(content)) => {
            events.extend(string_content_events(state, content, &reasoning_delta));
        }
        Some(Value::Array(content)) => {
            events.extend(structured_content_events(state, content));
        }
        _ => {}
    }

    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        if !tool_calls.is_empty() {
            events.extend(tool_call_events(state, tool_calls, &reasoning_delta));
        }
    }

    if let Some(refusal) = delta
        .get("refusal")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        events.extend(state.finalize_reasoning_if_open());
        events.extend(state.flush_open_text_part());
        events.extend(state.push_refusal_part(&sanitize_string(refusal)));
    }

    if let Some(finish_reason) = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        state.set_finish_reason(finish_reason);
    }
    events
}

/// Convert a non-streaming Chat choice into a synthetic streaming delta so the
/// buffered path produces identical SSE events. Injects a per-call `index` into
/// each tool call because the streaming protocol distinguishes parallel calls
/// by index, which the non-streaming message shape omits.
fn chat_message_to_fake_delta(choice: &Value) -> Value {
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut delta = Map::new();
    delta.insert("role".to_owned(), Value::from("assistant"));
    delta.insert(
        "content".to_owned(),
        message.get("content").cloned().unwrap_or(Value::Null),
    );
    if let Some(refusal) = message.get("refusal") {
        delta.insert("refusal".to_owned(), refusal.clone());
    }
    if let Some(annotations) = message.get("annotations") {
        delta.insert("annotations".to_owned(), annotations.clone());
    }
    if let Some(reasoning) = extract_reasoning_field(&message) {
        delta.insert("reasoning_content".to_owned(), Value::from(reasoning));
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        let indexed: Vec<Value> = tool_calls
            .iter()
            .filter(|tc| tc.is_object())
            .enumerate()
            .map(|(i, tc)| {
                let mut obj = tc.as_object().cloned().unwrap_or_default();
                obj.insert("index".to_owned(), Value::from(i as i64));
                Value::Object(obj)
            })
            .collect();
        delta.insert("tool_calls".to_owned(), Value::Array(indexed));
    }
    Value::Object(delta)
}

/// Render a fully buffered Chat Completions response as Responses SSE events.
///
/// The buffered body holds the entire turn, so the stream is never truncated: a
/// missing `finish_reason` is a gateway omission, not a drop, hence
/// `stream_ended_cleanly = true`.
pub fn sse_events_from_buffered_chat(
    chat_body: &Value,
    tool_context: BridgeToolContext,
    response_id: Option<&str>,
    original_request: Option<&Map<String, Value>>,
) -> Vec<Vec<u8>> {
    let mut state = new_stream_state(tool_context, response_id, original_request);
    state.apply_chunk_metadata(chat_body);

    let mut events = Vec::new();
    if let Some(choices) = chat_body.get("choices").and_then(Value::as_array) {
        for choice in choices {
            let delta = chat_message_to_fake_delta(choice);
            let mut chunk = Map::new();
            let mut inner = Map::new();
            inner.insert("delta".to_owned(), delta);
            inner.insert(
                "finish_reason".to_owned(),
                choice.get("finish_reason").cloned().unwrap_or(Value::Null),
            );
            chunk.insert(
                "choices".to_owned(),
                Value::Array(vec![Value::Object(inner)]),
            );
            events.extend(process_chat_chunk(&chunk, None, &mut state));
        }
    }
    events.extend(state.finalize(true));
    events
}

/// Consume a live upstream byte stream, yielding Responses SSE event bytes.
///
/// Decodes bytes incrementally (preserving split multibyte chars), frames SSE
/// blocks, and drives the state machine chunk-by-chunk. A residual frame left
/// without a trailing blank line by an abnormal close is still parsed. Reaching
/// the end of the byte stream without a `[DONE]` marker finalizes with
/// `stream_ended_cleanly = false` so a truncated turn is not persisted as
/// complete.
pub fn create_responses_sse_stream<S>(
    upstream: S,
    tool_context: BridgeToolContext,
    response_id: Option<String>,
    original_request: Option<Map<String, Value>>,
    persist: Option<StreamPersist>,
) -> impl Stream<Item = Vec<u8>>
where
    S: Stream<Item = Vec<u8>>,
{
    stream! {
        let mut state = new_stream_state(tool_context, response_id.as_deref(), original_request.as_ref());
        let mut decoder = Utf8StreamDecoder::new();
        let mut buffer = String::new();
        futures::pin_mut!(upstream);

        while let Some(chunk) = upstream.next().await {
            buffer.push_str(&decoder.decode(&chunk));
            let (messages, rest) = drain_sse_blocks(&buffer);
            buffer = rest;
            for (event_name, data) in messages {
                for event in process_sse_message(event_name.as_deref(), &data, &mut state) {
                    yield event;
                }
            }
        }

        buffer.push_str(&decoder.finalize());
        crate::metrics::record_stream_decode_replacements(decoder.replacements());

        // A residual frame without the trailing blank-line delimiter (and no
        // [DONE]) can survive an abnormal close — parse it so nothing is lost.
        if !buffer.trim().is_empty() {
            let (event_name, data) = parse_sse_block(&buffer);
            if let Some(data) = data.filter(|d| !d.is_empty()) {
                for event in process_sse_message(event_name.as_deref(), &data, &mut state) {
                    yield event;
                }
            }
        }

        for event in state.finalize(false) {
            yield event;
        }

        // Persist the completed turn for `previous_response_id` continuation.
        // Runs after finalize so the envelope status and reconstructed
        // assistant message are terminal. Only persistable statuses are saved.
        if let Some(persist) = persist {
            if state.should_persist() {
                persist.save(state.build_assistant_message());
            }
        }
    }
}

/// A captured persistence closure for the streaming path: everything needed to
/// snapshot the finalized turn into the session store, minus the assistant
/// message (reconstructed from stream state at finalize time). Mirrors the
/// Python bridge's `_save_turn` closure over `_persist_turn`.
pub struct StreamPersist {
    pub response_id: String,
    pub messages: Vec<Value>,
    pub tool_context: BridgeToolContext,
    pub model: String,
}

impl StreamPersist {
    fn save(self, assistant_message: Option<Value>) {
        crate::session_bridge::save_session(
            &self.response_id,
            &self.messages,
            &self.tool_context,
            &self.model,
            assistant_message,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use serde_json::json;

    fn parse_event(bytes: &[u8]) -> (Option<String>, Value) {
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let (event, data) = parse_sse_block(&text);
        let value = data
            .filter(|d| d != "[DONE]")
            .map(|d| serde_json::from_str::<Value>(&d).unwrap())
            .unwrap_or(Value::Null);
        (event, value)
    }

    fn event_names(events: &[Vec<u8>]) -> Vec<String> {
        events.iter().filter_map(|e| parse_event(e).0).collect()
    }

    fn last_response(events: &[Vec<u8>]) -> Value {
        parse_event(events.last().unwrap()).1["response"].clone()
    }

    #[test]
    fn decoder_holds_split_multibyte_across_chunks() {
        let mut d = Utf8StreamDecoder::new();
        // "€" is E2 82 AC; split after the first byte.
        assert_eq!(d.decode(&[0xE2]), "");
        assert_eq!(d.decode(&[0x82, 0xAC]), "€");
        assert_eq!(d.replacements(), 0);
    }

    #[test]
    fn decoder_replaces_invalid_bytes() {
        let mut d = Utf8StreamDecoder::new();
        // 0xFF is never valid UTF-8.
        let out = d.decode(&[b'a', 0xFF, b'b']);
        assert_eq!(out, "a\u{FFFD}b");
        assert_eq!(d.replacements(), 1);
    }

    #[test]
    fn decoder_finalize_replaces_incomplete_tail() {
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.decode(&[0xE2]), "");
        assert_eq!(d.finalize(), "\u{FFFD}");
        assert_eq!(d.replacements(), 1);
    }

    #[test]
    fn buffered_text_turn_completes() {
        let chat = json!({
            "model": "gpt-x",
            "choices": [{
                "message": { "role": "assistant", "content": "Hello" },
                "finish_reason": "stop"
            }]
        });
        let events =
            sse_events_from_buffered_chat(&chat, BridgeToolContext::new(), Some("resp_1"), None);
        let names = event_names(&events);
        assert_eq!(names.first().map(String::as_str), Some("response.created"));
        assert_eq!(names.last().map(String::as_str), Some("response.completed"));
        let resp = last_response(&events);
        assert_eq!(resp["status"], json!("completed"));
        assert_eq!(resp["output"][0]["content"][0]["text"], json!("Hello"));
    }

    #[test]
    fn buffered_parallel_tool_calls_keep_distinct_indices() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        { "id": "a", "type": "function", "function": { "name": "f", "arguments": "{}" } },
                        { "id": "b", "type": "function", "function": { "name": "g", "arguments": "{}" } }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let events =
            sse_events_from_buffered_chat(&chat, BridgeToolContext::new(), Some("resp_2"), None);
        let resp = last_response(&events);
        let output = resp["output"].as_array().unwrap();
        let fn_calls: Vec<&Value> = output
            .iter()
            .filter(|i| i["type"] == json!("function_call"))
            .collect();
        assert_eq!(fn_calls.len(), 2, "both parallel tool calls survive");
    }

    #[tokio::test]
    async fn streamed_chunks_produce_completed_turn() {
        let chunks: Vec<Vec<u8>> = vec![
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n".to_vec(),
            b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n".to_vec(),
            b"data: [DONE]\n\n".to_vec(),
        ];
        let out = create_responses_sse_stream(
            stream::iter(chunks),
            BridgeToolContext::new(),
            Some("resp_3".to_owned()),
            None,
            None,
        );
        let events: Vec<Vec<u8>> = out.collect().await;
        let resp = last_response(&events);
        assert_eq!(resp["status"], json!("completed"));
        assert_eq!(resp["output"][0]["content"][0]["text"], json!("Hello"));
    }

    #[tokio::test]
    async fn stream_split_across_frame_boundary_reassembles() {
        // A single SSE frame split across two byte chunks mid-JSON.
        let chunks: Vec<Vec<u8>> = vec![
            b"data: {\"choices\":[{\"delta\":{\"con".to_vec(),
            b"tent\":\"Hi\"},\"finish_reason\":\"stop\"}]}\n\n".to_vec(),
        ];
        let out = create_responses_sse_stream(
            stream::iter(chunks),
            BridgeToolContext::new(),
            None,
            None,
            None,
        );
        let events: Vec<Vec<u8>> = out.collect().await;
        let resp = last_response(&events);
        assert_eq!(resp["output"][0]["content"][0]["text"], json!("Hi"));
    }

    #[tokio::test]
    async fn stream_without_done_finalizes_incomplete() {
        // Content but no finish_reason and no [DONE] — a mid-stream drop.
        let chunks: Vec<Vec<u8>> =
            vec![b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n".to_vec()];
        let out = create_responses_sse_stream(
            stream::iter(chunks),
            BridgeToolContext::new(),
            None,
            None,
            None,
        );
        let events: Vec<Vec<u8>> = out.collect().await;
        let resp = last_response(&events);
        assert_eq!(resp["status"], json!("incomplete"));
        assert_eq!(
            resp["incomplete_details"]["reason"],
            json!("stream_truncated")
        );
    }

    #[tokio::test]
    async fn stream_error_event_fails() {
        let chunks: Vec<Vec<u8>> = vec![
            b"event: error\ndata: {\"error\":{\"message\":\"boom\",\"type\":\"upstream_oops\"}}\n\n"
                .to_vec(),
        ];
        let out = create_responses_sse_stream(
            stream::iter(chunks),
            BridgeToolContext::new(),
            None,
            None,
            None,
        );
        let events: Vec<Vec<u8>> = out.collect().await;
        let (last_event, last) = parse_event(events.last().unwrap());
        assert_eq!(last_event.as_deref(), Some("response.failed"));
        assert_eq!(last["response"]["error"]["type"], json!("upstream_oops"));
    }

    #[test]
    fn inline_think_routes_through_buffered_path() {
        let chat = json!({
            "choices": [{
                "message": { "role": "assistant", "content": "<think>hmm</think>done" },
                "finish_reason": "stop"
            }]
        });
        let events = sse_events_from_buffered_chat(&chat, BridgeToolContext::new(), None, None);
        let names = event_names(&events);
        assert!(names
            .iter()
            .any(|n| n == "response.reasoning_summary_text.delta"));
        let resp = last_response(&events);
        let output = resp["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], json!("reasoning"));
        assert_eq!(output[1]["content"][0]["text"], json!("done"));
    }
}
