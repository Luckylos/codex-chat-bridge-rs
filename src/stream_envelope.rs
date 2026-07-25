//! Response-envelope state for the streaming (Phase 2) SSE path.
//!
//! Mirrors the Python bridge's `stream_state/envelope.py`. The envelope owns
//! the lifecycle events that wrap every streamed turn — `response.created`,
//! `response.in_progress`, and the terminal `response.completed` /
//! `response.failed` — plus the shared response object those events carry
//! (id, model, usage, request-echo fields) and the output-index allocator the
//! increment state machines draw from.
//!
//! It sits directly above the SSE codec: it builds `Value` response objects and
//! renders them through [`crate::sse::serialize_event`]. The reasoning / message
//! / tool increment machines (next layers) call into it to allocate indices and
//! append completed output items; the top-level orchestrator drives
//! `ensure_started` / `finalize`. Until those land the envelope reads as dead,
//! so the module carries `allow(dead_code)`; the tests lock in the event
//! contract now.
#![allow(dead_code)]

use serde_json::{json, Map, Value};

use crate::convert::{
    incomplete_reason_from_finish_reason, map_chat_usage, response_status_from_finish_reason,
    ResponseStatus, REQUEST_ECHO_FIELDS,
};
use crate::id_gen;
use crate::sse::serialize_event;

/// Wrap a response object in the `{type, response}` envelope and render it as a
/// single SSE event. Mirrors `tool_events.response_event`.
pub(crate) fn response_event(event_name: &str, response: &Value) -> Vec<u8> {
    let payload = json!({ "type": event_name, "response": response });
    serialize_event(Some(event_name), &payload)
}

/// Lifecycle + shared-response state for one streamed turn.
pub(crate) struct ResponseEnvelopeState {
    pub response_started: bool,
    pub completed: bool,
    pub response_id: String,
    pub model: String,
    pub created_at: i64,
    pub status: Option<crate::convert::ResponseStatus>,
    pub usage: Option<Value>,
    pub finish_reason: Option<String>,
    next_output_index: i64,
    completed_items: Vec<(i64, Value)>,
    request_echo: Option<Map<String, Value>>,
}

impl ResponseEnvelopeState {
    pub(crate) fn new(response_id: Option<&str>) -> Self {
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            response_started: false,
            completed: false,
            response_id: response_id.unwrap_or("resp_bridge").to_owned(),
            model: String::new(),
            created_at,
            status: None,
            usage: None,
            finish_reason: None,
            next_output_index: 0,
            completed_items: Vec::new(),
            request_echo: None,
        }
    }

    /// Output item id for the assistant message item of this response.
    pub(crate) fn message_item_id(&self) -> String {
        id_gen::message_item_id(&self.response_id)
    }

    /// Output item id for the reasoning summary item of this response.
    pub(crate) fn reasoning_item_id(&self) -> String {
        id_gen::reasoning_item_id(&self.response_id)
    }

    /// Store the original Responses request for echo-back in the final response.
    pub(crate) fn set_request_echo(&mut self, original: Option<&Map<String, Value>>) {
        self.request_echo = original.cloned();
    }

    /// Allocate the next monotonic output index for a new output item.
    pub(crate) fn allocate_output_index(&mut self) -> i64 {
        let idx = self.next_output_index;
        self.next_output_index += 1;
        idx
    }

    /// The next output index that would be allocated, without consuming it.
    /// The tool store needs this to compute a stable per-call base so parallel
    /// tool calls keep their upstream `index` ordering.
    pub(crate) fn peek_next_output_index(&self) -> i64 {
        self.next_output_index
    }

    /// Advance the output-index cursor so the next allocation is at least
    /// `at_least`. A no-op if the cursor is already past that point. The tool
    /// store uses this after claiming a base+index slot so later items don't
    /// collide with tool-call indices.
    pub(crate) fn advance_output_index_to(&mut self, at_least: i64) {
        if self.next_output_index < at_least {
            self.next_output_index = at_least;
        }
    }

    pub(crate) fn append_completed_item(&mut self, output_index: i64, item: Value) {
        self.completed_items.push((output_index, item));
    }

    /// The completed output items, ordered by their allocated output index.
    pub(crate) fn completed_output_items(&self) -> Vec<Value> {
        let mut items = self.completed_items.clone();
        items.sort_by_key(|(idx, _)| *idx);
        items.into_iter().map(|(_, item)| item).collect()
    }

    /// Build the shared response object at `status` carrying `output`.
    ///
    /// Only request-echo fields the caller actually sent are added (matching
    /// the Python streaming envelope, which — unlike the non-streaming path —
    /// does not seed absent fields as null).
    fn base_response(
        &mut self,
        status: crate::convert::ResponseStatus,
        output: Vec<Value>,
    ) -> Value {
        self.status = Some(status);
        let mut response = Map::new();
        response.insert("id".to_owned(), json!(self.response_id));
        response.insert("object".to_owned(), json!("response"));
        response.insert("created_at".to_owned(), json!(self.created_at));
        response.insert("status".to_owned(), json!(status.as_str()));
        response.insert("model".to_owned(), json!(self.model));
        response.insert("output".to_owned(), Value::Array(output));
        response.insert(
            "usage".to_owned(),
            self.usage.clone().unwrap_or_else(
                || json!({ "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 }),
            ),
        );
        if let Some(echo) = &self.request_echo {
            for &key in REQUEST_ECHO_FIELDS {
                if let Some(value) = echo.get(key) {
                    if !value.is_null() {
                        response.insert(key.to_owned(), value.clone());
                    }
                }
            }
        }
        Value::Object(response)
    }

    /// Emit `response.created` + `response.in_progress` exactly once.
    pub(crate) fn ensure_started(&mut self) -> Vec<Vec<u8>> {
        if self.response_started {
            return Vec::new();
        }
        self.response_started = true;
        let response = self.base_response(ResponseStatus::InProgress, Vec::new());
        vec![
            response_event("response.created", &response),
            response_event("response.in_progress", &response),
        ]
    }

    /// Fold upstream chunk metadata (model / created / usage) into the envelope.
    ///
    /// The first non-empty model is frozen so later chunks repeating a different
    /// alias cannot make the lifecycle events drift.
    pub(crate) fn apply_metadata(&mut self, payload: &Value) {
        if let Some(model) = payload.get("model").and_then(Value::as_str) {
            if !model.is_empty() && self.model.is_empty() {
                self.model = model.to_owned();
            }
        }
        if let Some(created) = payload.get("created").and_then(Value::as_i64) {
            if created != 0 {
                self.created_at = created;
            }
        }
        if let Some(usage) = payload.get("usage") {
            if usage.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
                self.usage = Some(map_chat_usage(Some(usage)));
            }
        }
    }

    /// Terminal `response.completed` event, mapping `finish_reason` to status.
    pub(crate) fn completed_event(&mut self, output: Vec<Value>) -> Vec<u8> {
        let status = response_status_from_finish_reason(self.finish_reason.as_deref());
        let mut response = self.base_response(status, output);
        if let Some(details) = incomplete_reason_from_finish_reason(self.finish_reason.as_deref()) {
            response["incomplete_details"] = details;
        }
        response_event("response.completed", &response)
    }

    /// Terminal event for a mid-stream drop: status `incomplete` with a
    /// `stream_truncated` reason, still rendered as `response.completed`.
    pub(crate) fn truncated_event(&mut self, output: Vec<Value>) -> Vec<u8> {
        let mut response = self.base_response(ResponseStatus::Incomplete, output);
        response["incomplete_details"] = json!({ "reason": "stream_truncated" });
        response_event("response.completed", &response)
    }

    /// Terminal `response.failed` event carrying an error envelope.
    pub(crate) fn failed_event(&mut self, message: &str, error_type: &str) -> Vec<u8> {
        let output = self.completed_output_items();
        let mut response = self.base_response(ResponseStatus::Failed, output);
        response["error"] = json!({ "message": message, "type": error_type });
        response_event("response.failed", &response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::parse_sse_block;

    /// Parse one serialized SSE event into `(event_name, response_object)`.
    fn parse_event(bytes: &[u8]) -> (String, Value) {
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let (event, data) = parse_sse_block(&text);
        let payload: Value = serde_json::from_str(&data.unwrap()).unwrap();
        (event.unwrap(), payload["response"].clone())
    }

    fn echo_map(pairs: Value) -> Map<String, Value> {
        pairs.as_object().unwrap().clone()
    }

    #[test]
    fn ensure_started_emits_created_then_in_progress_once() {
        let mut env = ResponseEnvelopeState::new(Some("resp_bridge_x"));
        let events = env.ensure_started();
        assert_eq!(events.len(), 2);
        let (e0, r0) = parse_event(&events[0]);
        let (e1, r1) = parse_event(&events[1]);
        assert_eq!(e0, "response.created");
        assert_eq!(e1, "response.in_progress");
        assert_eq!(r0["status"], json!("in_progress"));
        assert_eq!(r0["id"], json!("resp_bridge_x"));
        assert_eq!(r1["status"], json!("in_progress"));
        // Second call is a no-op.
        assert!(env.ensure_started().is_empty());
    }

    #[test]
    fn base_response_echoes_only_present_fields() {
        let mut env = ResponseEnvelopeState::new(Some("r"));
        env.set_request_echo(Some(&echo_map(json!({
            "temperature": 0.5,
            "top_p": Value::Null,
        }))));
        let events = env.ensure_started();
        let (_, resp) = parse_event(&events[0]);
        assert_eq!(resp["temperature"], json!(0.5));
        // Null echo values are dropped, not seeded.
        assert!(resp.get("top_p").is_none());
    }

    #[test]
    fn apply_metadata_freezes_first_model_and_maps_usage() {
        let mut env = ResponseEnvelopeState::new(None);
        env.apply_metadata(&json!({ "model": "gpt-x", "created": 123 }));
        env.apply_metadata(&json!({ "model": "other-alias" }));
        env.apply_metadata(&json!({
            "usage": { "prompt_tokens": 10, "completion_tokens": 5 }
        }));
        assert_eq!(env.model, "gpt-x");
        assert_eq!(env.created_at, 123);
        let events = env.ensure_started();
        let (_, resp) = parse_event(&events[0]);
        assert_eq!(resp["model"], json!("gpt-x"));
        assert_eq!(resp["usage"]["input_tokens"], json!(10));
        assert_eq!(resp["usage"]["output_tokens"], json!(5));
        assert_eq!(resp["usage"]["total_tokens"], json!(15));
    }

    #[test]
    fn apply_metadata_ignores_empty_usage_and_zero_created() {
        let mut env = ResponseEnvelopeState::new(None);
        let before = env.created_at;
        env.apply_metadata(&json!({ "created": 0, "usage": {} }));
        assert_eq!(env.created_at, before);
        assert!(env.usage.is_none());
    }

    #[test]
    fn completed_event_maps_finish_reason_to_status() {
        let mut env = ResponseEnvelopeState::new(Some("r"));
        env.finish_reason = Some("length".to_owned());
        let bytes = env.completed_event(vec![json!({ "type": "message" })]);
        let (event, resp) = parse_event(&bytes);
        assert_eq!(event, "response.completed");
        assert_eq!(resp["status"], json!("incomplete"));
        assert_eq!(
            resp["incomplete_details"],
            json!({ "reason": "max_output_tokens" })
        );
        assert_eq!(resp["output"][0]["type"], json!("message"));
    }

    #[test]
    fn completed_event_without_finish_reason_is_completed() {
        let mut env = ResponseEnvelopeState::new(Some("r"));
        let bytes = env.completed_event(Vec::new());
        let (_, resp) = parse_event(&bytes);
        assert_eq!(resp["status"], json!("completed"));
        assert!(resp.get("incomplete_details").is_none());
    }

    #[test]
    fn truncated_event_is_incomplete_with_stream_truncated() {
        let mut env = ResponseEnvelopeState::new(Some("r"));
        let bytes = env.truncated_event(Vec::new());
        let (event, resp) = parse_event(&bytes);
        assert_eq!(event, "response.completed");
        assert_eq!(resp["status"], json!("incomplete"));
        assert_eq!(
            resp["incomplete_details"],
            json!({ "reason": "stream_truncated" })
        );
    }

    #[test]
    fn failed_event_carries_error_envelope() {
        let mut env = ResponseEnvelopeState::new(Some("r"));
        let bytes = env.failed_event("boom", "stream_error");
        let (event, resp) = parse_event(&bytes);
        assert_eq!(event, "response.failed");
        assert_eq!(resp["status"], json!("failed"));
        assert_eq!(
            resp["error"],
            json!({ "message": "boom", "type": "stream_error" })
        );
    }

    #[test]
    fn output_index_allocation_is_monotonic() {
        let mut env = ResponseEnvelopeState::new(None);
        assert_eq!(env.allocate_output_index(), 0);
        assert_eq!(env.allocate_output_index(), 1);
        assert_eq!(env.allocate_output_index(), 2);
    }

    #[test]
    fn completed_items_are_returned_in_index_order() {
        let mut env = ResponseEnvelopeState::new(None);
        env.append_completed_item(2, json!({ "n": 2 }));
        env.append_completed_item(0, json!({ "n": 0 }));
        env.append_completed_item(1, json!({ "n": 1 }));
        let items = env.completed_output_items();
        assert_eq!(items[0]["n"], json!(0));
        assert_eq!(items[1]["n"], json!(1));
        assert_eq!(items[2]["n"], json!(2));
    }

    #[test]
    fn item_ids_derive_from_response_id() {
        let env = ResponseEnvelopeState::new(Some("resp_bridge_abc"));
        assert_eq!(env.message_item_id(), "msg_resp_bridge_abc");
        assert_eq!(env.reasoning_item_id(), "rs_resp_bridge_abc");
    }
}
