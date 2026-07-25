//! Top-level streaming orchestrator: `ResponsesStreamState`.
//!
//! Composes the four increment state machines (envelope, reasoning, message,
//! tools) plus the inline-`<think>` detector, and drives them from upstream
//! Chat Completions SSE deltas. Mirrors the Python bridge's
//! `stream_responses_state.ResponsesStreamState`.
//!
//! Borrow model: `inline_think` and the three sub-states are all disjoint
//! fields of `self`, so routing a content delta borrows each field
//! independently (`&mut self.inline_think` alongside `&mut self.envelope` etc.)
//! rather than through a `self`-consuming method.
//!
//! `build_assistant_message` (session persistence) is intentionally omitted
//! here; it needs the `ChatMessage` type that lands with the session layer.

use serde_json::{Map, Value};

use crate::context::BridgeToolContext;
use crate::stream_envelope::ResponseEnvelopeState;
use crate::stream_inline_think::InlineThinkStateMachine;
use crate::stream_message::MessageState;
use crate::stream_reasoning::ReasoningState;
use crate::stream_tools::ToolStateStore;

/// Lifecycle + component aggregate for one streamed Responses turn.
///
// Constructed by the SSE consumer loop that lands in the next layer, so the
// type and its methods read as dead until that wires in. Tests exercise the
// full surface now.
#[allow(dead_code)]
pub struct ResponsesStreamState {
    envelope: ResponseEnvelopeState,
    reasoning: ReasoningState,
    message: MessageState,
    tools: ToolStateStore,
    inline_think: InlineThinkStateMachine,
}

#[allow(dead_code)]
impl ResponsesStreamState {
    pub fn new(tool_context: BridgeToolContext, response_id: Option<&str>) -> Self {
        Self {
            envelope: ResponseEnvelopeState::new(response_id),
            reasoning: ReasoningState::new(),
            message: MessageState::new(),
            tools: ToolStateStore::new(tool_context),
            inline_think: InlineThinkStateMachine::new(),
        }
    }

    /// Absorb `model` / `created` / `usage` metadata from a Chat chunk.
    pub fn apply_chunk_metadata(&mut self, payload: &Value) {
        self.envelope.apply_metadata(payload);
    }

    /// Seed the request-echo fields rendered into every lifecycle event.
    pub fn set_request_echo(&mut self, original: Option<&Map<String, Value>>) {
        self.envelope.set_request_echo(original);
    }

    /// Accumulate assistant-message annotations for the current/next segment.
    pub fn add_annotations(&mut self, annotations: Option<&Value>) {
        self.message.add_annotations(annotations);
    }

    /// Force the inline-`<think>` detector to text, flushing buffered reasoning.
    pub fn force_inline_think_to_text(&mut self) -> Vec<Vec<u8>> {
        self.inline_think
            .force_to_text(&mut self.envelope, &mut self.reasoning, &mut self.message)
    }

    /// Emit the `response.created` + `response.in_progress` pair, once.
    pub fn ensure_started(&mut self) -> Vec<Vec<u8>> {
        self.envelope.ensure_started()
    }

    /// Push an explicit reasoning-field delta (not inline-`<think>`).
    pub fn push_reasoning_delta(&mut self, delta: &str) -> Vec<Vec<u8>> {
        self.reasoning.push_delta(&mut self.envelope, delta)
    }

    /// The still-open reasoning text to attach to a tool call.
    pub fn active_reasoning_text_for_tools(&self) -> String {
        self.reasoning.active_text_for_tools()
    }

    /// Close the reasoning item if it is open.
    pub fn finalize_reasoning_if_open(&mut self) -> Vec<Vec<u8>> {
        self.reasoning.finalize(&mut self.envelope)
    }

    /// Accumulate a `tool_calls[]` delta.
    ///
    /// A tool call arriving mid-`<think>` means the model is done thinking, so
    /// force the inline-think detector to text first (flushing buffered
    /// reasoning) before the tool item opens. Mirrors the Python driver's
    /// `force_to_text` call ahead of `push_tool_call_delta`.
    pub fn push_tool_call_delta(
        &mut self,
        tool_call: &Value,
        reasoning: Option<&str>,
    ) -> Vec<Vec<u8>> {
        let mut events = self.inline_think.force_to_text(
            &mut self.envelope,
            &mut self.reasoning,
            &mut self.message,
        );
        events.extend(
            self.tools
                .push_delta(&mut self.envelope, tool_call, reasoning),
        );
        events
    }

    /// Push a plain assistant text delta (bypassing inline-think detection).
    pub fn push_text_delta(&mut self, delta: &str) -> Vec<Vec<u8>> {
        self.message.push_text_delta(&mut self.envelope, delta)
    }

    /// Route an assistant `content` delta through inline-`<think>` detection.
    pub fn push_content_delta(&mut self, delta: &str) -> Vec<Vec<u8>> {
        self.inline_think.push_content_delta(
            delta,
            &mut self.envelope,
            &mut self.reasoning,
            &mut self.message,
        )
    }

    /// Push a refusal content part.
    pub fn push_refusal_part(&mut self, refusal: &str) -> Vec<Vec<u8>> {
        self.message.push_refusal_part(&mut self.envelope, refusal)
    }

    /// Close the currently-open text part, if any.
    pub fn flush_open_text_part(&mut self) -> Vec<Vec<u8>> {
        self.message.flush_open_text_part(&mut self.envelope)
    }

    /// Record the upstream `finish_reason` for terminal-status mapping.
    pub fn set_finish_reason(&mut self, finish_reason: &str) {
        self.envelope.finish_reason = Some(finish_reason.to_owned());
    }

    /// Flush every open item in dependency order: start the envelope, drain the
    /// inline-think detector, then finalize reasoning → message → tools.
    fn flush_open_items(&mut self) -> Vec<Vec<u8>> {
        let mut events = self.envelope.ensure_started();
        events.extend(self.inline_think.flush_on_finalize(
            &mut self.envelope,
            &mut self.reasoning,
            &mut self.message,
        ));
        events.extend(self.reasoning.finalize(&mut self.envelope));
        events.extend(self.message.finalize(&mut self.envelope));
        events.extend(self.tools.finalize(&mut self.envelope));
        events
    }

    /// Emit terminal events for the stream.
    ///
    /// `stream_ended_cleanly` marks an explicit upstream `[DONE]` (or a fully
    /// buffered body): the stream ended normally, so a missing `finish_reason`
    /// is a gateway omission, not a truncation. Without that signal a missing
    /// `finish_reason` stays `incomplete` so a partial turn is not persisted
    /// for `previous_response_id` continuation.
    pub fn finalize(&mut self, stream_ended_cleanly: bool) -> Vec<Vec<u8>> {
        if self.envelope.completed {
            return Vec::new();
        }
        self.envelope.completed = true;
        let mut events = self.flush_open_items();
        events.push(self.terminal_event(stream_ended_cleanly));
        events
    }

    /// Choose the terminal event for the finalized stream. Open items are
    /// already flushed by the caller, so this returns the single event
    /// directly rather than re-entering the guarded `fail`.
    fn terminal_event(&mut self, stream_ended_cleanly: bool) -> Vec<u8> {
        let output = self.envelope.completed_output_items();
        // A known finish_reason is authoritative.
        if self.envelope.finish_reason.is_some() {
            return self.envelope.completed_event(output);
        }
        // No finish_reason: a clean end means the gateway merely omitted it.
        if !output.is_empty() && stream_ended_cleanly {
            return self.envelope.completed_event(output);
        }
        // Otherwise the stream was cut short.
        if !output.is_empty() {
            return self.envelope.truncated_event(output);
        }
        self.envelope.failed_event(
            "Stream truncated before any output was produced",
            "stream_truncated",
        )
    }

    /// Build an assistant Chat message for session persistence, or `None` when
    /// the turn produced nothing worth saving.
    ///
    /// Reconstructs the message from the finalized sub-states: tool calls from
    /// the tool store, visible content from the message segments (falling back
    /// to the concatenated text), and reasoning from the reasoning state. Text
    /// fields are sanitized; reasoning is additionally stripped because it is
    /// structural separator content, not user-visible text. Mirrors
    /// `ResponsesStreamState.build_assistant_message`.
    pub fn build_assistant_message(&self) -> Option<Value> {
        let tool_calls = self.tools.persisted_tool_calls();

        let content_parts = self.message.content_parts();
        let content: Value = if !content_parts.is_empty() {
            let sanitized: Vec<Value> = content_parts
                .iter()
                .map(|part| {
                    let mut obj = part.as_object().cloned().unwrap_or_default();
                    if let Some(text) = obj.get("text").and_then(Value::as_str) {
                        obj.insert(
                            "text".to_owned(),
                            Value::from(crate::sanitize::sanitize_string(text)),
                        );
                    }
                    Value::Object(obj)
                })
                .collect();
            crate::convert::chat_message_content_from_response_content(Some(&Value::Array(
                sanitized,
            )))
        } else {
            let text = self.message.text();
            if text.is_empty() {
                Value::Null
            } else {
                Value::from(crate::sanitize::sanitize_string(&text))
            }
        };
        let has_visible_content = !content.is_null();

        let reasoning = crate::sanitize::sanitize_string(self.reasoning.text());
        let reasoning = reasoning.trim();

        if !has_visible_content && tool_calls.is_empty() && reasoning.is_empty() {
            return None;
        }

        let mut msg = Map::new();
        msg.insert("role".to_owned(), Value::from("assistant"));
        msg.insert("content".to_owned(), content);
        if !tool_calls.is_empty() {
            msg.insert("tool_calls".to_owned(), Value::Array(tool_calls));
        }
        if !reasoning.is_empty() {
            msg.insert("reasoning_content".to_owned(), Value::from(reasoning));
        }
        Some(Value::Object(msg))
    }

    /// Whether this finalized turn is safe to persist for `previous_response_id`
    /// continuation: the envelope reached a terminal state and that state maps
    /// to a persistable status (`completed` / `in_progress`). Mirrors the
    /// streaming path's `state.envelope.completed and
    /// should_persist_response_status(state.envelope.status)` guard.
    pub fn should_persist(&self) -> bool {
        self.envelope.completed
            && crate::convert::should_persist_response_status(self.envelope.status)
    }

    /// Flush open items and emit a terminal `response.failed` event.
    pub fn fail(&mut self, message: &str, error_type: &str) -> Vec<Vec<u8>> {
        if self.envelope.completed {
            return Vec::new();
        }
        self.envelope.completed = true;
        let mut events = self.flush_open_items();
        events.push(self.envelope.failed_event(message, error_type));
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn state() -> ResponsesStreamState {
        ResponsesStreamState::new(BridgeToolContext::new(), Some("resp_test"))
    }

    fn parse_event(bytes: &[u8]) -> (String, Value) {
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let mut event = String::new();
        let mut data = String::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                event = rest.to_owned();
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data = rest.to_owned();
            }
        }
        let value = if data.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&data).unwrap()
        };
        (event, value)
    }

    fn event_types(events: &[Vec<u8>]) -> Vec<String> {
        events.iter().map(|e| parse_event(e).0).collect()
    }

    #[test]
    fn ensure_started_emits_created_then_in_progress() {
        let mut s = state();
        let types = event_types(&s.ensure_started());
        assert_eq!(types, ["response.created", "response.in_progress"]);
        // Idempotent.
        assert!(s.ensure_started().is_empty());
    }

    #[test]
    fn text_only_turn_finalizes_completed() {
        let mut s = state();
        s.ensure_started();
        s.set_finish_reason("stop");
        let deltas = s.push_content_delta("Hello world");
        assert!(!deltas.is_empty());
        let events = s.finalize(true);
        let (last_event, last) = parse_event(events.last().unwrap());
        assert_eq!(last_event, "response.completed");
        assert_eq!(last["response"]["status"], json!("completed"));
        // The message text made it into the output.
        let output = &last["response"]["output"];
        assert_eq!(output[0]["content"][0]["text"], json!("Hello world"));
    }

    #[test]
    fn inline_think_routes_reasoning_then_text() {
        let mut s = state();
        s.ensure_started();
        s.set_finish_reason("stop");
        let mut events = s.push_content_delta("<think>pondering</think>answer");
        events.extend(s.finalize(true));
        let types = event_types(&events);
        // Reasoning item opened and closed, then message text, then completed.
        assert!(types
            .iter()
            .any(|t| t == "response.reasoning_summary_text.delta"));
        assert!(types.iter().any(|t| t == "response.output_text.delta"));
        let (_, last) = parse_event(events.last().unwrap());
        let output = &last["response"]["output"];
        // First output item is reasoning, second is the message.
        assert_eq!(output[0]["type"], json!("reasoning"));
        assert_eq!(output[1]["type"], json!("message"));
        assert_eq!(output[1]["content"][0]["text"], json!("answer"));
    }

    #[test]
    fn tool_call_forces_pending_think_to_text() {
        let mut s = state();
        s.ensure_started();
        s.set_finish_reason("tool_calls");
        // A think block opens but never closes before the tool call arrives.
        s.push_content_delta("<think>deciding");
        let tool_call = json!({
            "index": 0,
            "id": "call_x",
            "function": { "name": "do_it", "arguments": "{}" }
        });
        let events = s.push_tool_call_delta(&tool_call, None);
        let types = event_types(&events);
        // force_to_text finalized reasoning, then the tool item opened.
        assert!(types
            .iter()
            .any(|t| t == "response.reasoning_summary_text.done"));
        assert!(types.iter().any(|t| t == "response.output_item.added"));

        let events = s.finalize(true);
        let (_, last) = parse_event(events.last().unwrap());
        // finish_reason "tool_calls" maps to in_progress, not completed.
        assert_eq!(last["response"]["status"], json!("in_progress"));
        let output = &last["response"]["output"];
        // reasoning item + function_call item both present.
        let has_fn = output
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["type"] == json!("function_call"));
        assert!(has_fn);
    }

    #[test]
    fn missing_finish_reason_without_clean_end_is_incomplete() {
        let mut s = state();
        s.ensure_started();
        s.push_content_delta("partial");
        // No finish_reason, mid-stream drop (not clean). truncated_event is
        // rendered as response.completed carrying status=incomplete.
        let events = s.finalize(false);
        let (last_event, last) = parse_event(events.last().unwrap());
        assert_eq!(last_event, "response.completed");
        assert_eq!(last["response"]["status"], json!("incomplete"));
        assert_eq!(
            last["response"]["incomplete_details"]["reason"],
            json!("stream_truncated")
        );
    }

    #[test]
    fn missing_finish_reason_with_clean_end_is_completed() {
        let mut s = state();
        s.ensure_started();
        s.push_content_delta("all good");
        // No finish_reason but the stream ended cleanly ([DONE]).
        let events = s.finalize(true);
        let (last_event, _) = parse_event(events.last().unwrap());
        assert_eq!(last_event, "response.completed");
    }

    #[test]
    fn no_output_at_all_fails_truncated() {
        let mut s = state();
        s.ensure_started();
        // Nothing pushed, no finish_reason, not clean.
        let events = s.finalize(false);
        let (last_event, last) = parse_event(events.last().unwrap());
        assert_eq!(last_event, "response.failed");
        assert_eq!(last["response"]["error"]["type"], json!("stream_truncated"));
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut s = state();
        s.ensure_started();
        s.set_finish_reason("stop");
        s.push_content_delta("hi");
        assert!(!s.finalize(true).is_empty());
        assert!(s.finalize(true).is_empty());
    }

    #[test]
    fn fail_flushes_and_emits_failed() {
        let mut s = state();
        s.ensure_started();
        s.push_content_delta("interrupted");
        let events = s.fail("boom", "stream_error");
        let (last_event, last) = parse_event(events.last().unwrap());
        assert_eq!(last_event, "response.failed");
        assert_eq!(last["response"]["error"]["type"], json!("stream_error"));
        // Guarded after completion.
        assert!(s.fail("again", "stream_error").is_empty());
    }

    #[test]
    fn metadata_freezes_model_into_response() {
        let mut s = state();
        s.apply_chunk_metadata(&json!({ "model": "gpt-x", "created": 123 }));
        s.ensure_started();
        s.set_finish_reason("stop");
        s.push_content_delta("ok");
        let events = s.finalize(true);
        let (_, last) = parse_event(events.last().unwrap());
        assert_eq!(last["response"]["model"], json!("gpt-x"));
    }

    #[test]
    fn active_reasoning_text_for_tools_reflects_open_block() {
        let mut s = state();
        s.ensure_started();
        s.push_reasoning_delta("thinking hard");
        assert_eq!(s.active_reasoning_text_for_tools(), "thinking hard");
        s.finalize_reasoning_if_open();
        // Once finalized, no active text.
        assert_eq!(s.active_reasoning_text_for_tools(), "");
    }
}
