//! Bidirectional Responses ↔ Chat Completions conversion.
//!
//! Both directions live here (mirroring the Python `responses_to_chat/` +
//! `chat_to_responses/` split, but a single file suffices in Rust — visibility
//! is `pub(crate)` and there are no circular-import constraints to route
//! around). The polymorphic interior stays `serde_json::Value`; only the
//! request/response envelopes are strongly typed.
//!
//! Phase 1 scope: text / reasoning / function-call / custom-tool /
//! tool-search / tool-output / generic-message on the request side, and the
//! full response assembly (reasoning + message + tool-call items, usage,
//! status, request-echo) on the response side. Namespace flatten/restore is
//! Phase 3 — here tool names map through identity.

use serde_json::{json, Map, Value};

use crate::id_gen;
use crate::reasoning;
use crate::sanitize::sanitize_string;
use crate::transform_loss::{TransformLoss, TransformLossCollector};
use crate::types::{ChatRequest, ResponsesRequest};

// --------------------------------------------------------------------------- //
// Request echo fields — single source of truth, mirrors response_semantics.py.
// --------------------------------------------------------------------------- //

pub(crate) const REQUEST_ECHO_FIELDS: &[&str] = &[
    "instructions",
    "max_output_tokens",
    "parallel_tool_calls",
    "previous_response_id",
    "reasoning",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "metadata",
];

// Chat request fields that pass through verbatim from the Responses request
// when present. Split into always-on and stream-only, mirroring
// responses_to_chat/constants.py EXTRA_CHAT_PASSTHROUGH_FIELDS.
const EXTRA_PASSTHROUGH_FIELDS: &[&str] = &[
    "metadata",
    "parallel_tool_calls",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "service_tier",
    "stop",
    "user",
    "logit_bias",
    "logprobs",
    "top_logprobs",
];

// --------------------------------------------------------------------------- //
// A Chat message under construction. content is None | String | Array.
// --------------------------------------------------------------------------- //

#[derive(Debug, Clone)]
struct ChatMessageBuilder {
    role: String,
    content: Value,
    tool_calls: Vec<Value>,
    tool_call_id: Option<String>,
    reasoning_content: Option<String>,
}

impl ChatMessageBuilder {
    fn new(role: &str) -> Self {
        Self {
            role: role.to_owned(),
            content: Value::Null,
            tool_calls: Vec::new(),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn with_content(role: &str, content: Value) -> Self {
        let mut m = Self::new(role);
        m.content = content;
        m
    }

    fn into_value(self) -> Value {
        let mut obj = Map::new();
        obj.insert("role".to_owned(), json!(self.role));
        // content is always present (null serializes as needed by upstream).
        obj.insert("content".to_owned(), self.content);
        if !self.tool_calls.is_empty() {
            obj.insert("tool_calls".to_owned(), Value::Array(self.tool_calls));
        }
        if let Some(id) = self.tool_call_id {
            obj.insert("tool_call_id".to_owned(), json!(id));
        }
        if let Some(rc) = self.reasoning_content {
            obj.insert("reasoning_content".to_owned(), json!(rc));
        }
        Value::Object(obj)
    }
}

// --------------------------------------------------------------------------- //
// Responses → Chat request
// --------------------------------------------------------------------------- //

/// Convert an inbound Responses request into an upstream Chat request.
///
/// Test-only convenience wrapper for the no-session case; the handler always
/// calls [`responses_to_chat_with_session`] directly.
#[cfg(test)]
pub fn responses_to_chat(payload: &ResponsesRequest, resolved_model: &str) -> ChatRequest {
    let tool_context = crate::context::build_tool_context_from_request(payload);
    responses_to_chat_with_session(payload, resolved_model, None, &tool_context)
}

/// Convert an inbound Responses request into an upstream Chat request, prefixing
/// the message list with `existing_messages` restored from a prior turn
/// (`previous_response_id` continuation). The stored history leads, then this
/// turn's instructions system message, then this turn's input items. Mirrors
/// the Python bridge's `responses_to_chat_request` + `_initial_messages`.
pub fn responses_to_chat_with_session(
    payload: &ResponsesRequest,
    resolved_model: &str,
    existing_messages: Option<&[Value]>,
    tool_context: &crate::context::BridgeToolContext,
) -> ChatRequest {
    let mut messages: Vec<Value> = match existing_messages {
        Some(prev) => prev.to_vec(),
        None => Vec::new(),
    };

    // Instructions become a leading system message.
    if let Some(instructions) = &payload.instructions {
        let text = instruction_text(instructions);
        if !text.trim().is_empty() {
            messages.push(ChatMessageBuilder::with_content("system", json!(text)).into_value());
        }
    }

    let mut loss = crate::transform_loss::TransformLossCollector::new();
    append_input_items(payload, &mut messages, &mut loss, tool_context);
    drain_transform_loss(&loss);
    let messages = collapse_system_messages_to_head(sanitize_messages(messages));

    let mut body = Map::new();
    body.insert("model".to_owned(), json!(resolved_model));
    body.insert("messages".to_owned(), Value::Array(messages));
    body.insert("stream".to_owned(), json!(payload.stream));

    if payload.stream {
        // stream_options is a named field, so it is read directly rather than
        // from `extra` (which never holds named fields).
        let stream_options = payload
            .stream_options
            .clone()
            .unwrap_or_else(|| json!({ "include_usage": true }));
        body.insert("stream_options".to_owned(), stream_options);
    }

    // Tools: the tool context has already normalized every declared Responses
    // tool into the Chat Completions nested `{type:function, function:{...}}`
    // shape, applying namespace flattening, custom-tool input schemas, and the
    // tool-search proxy. Reusing it keeps the request `tools` array consistent
    // with the reverse (chat→responses) name/namespace restore map.
    let chat_tools = tool_context.chat_tools();
    if !chat_tools.is_empty() {
        body.insert("tools".to_owned(), Value::Array(chat_tools.to_vec()));
        if let Some(tc) = &payload.tool_choice {
            body.insert(
                "tool_choice".to_owned(),
                responses_tool_choice_to_chat(tc, tool_context),
            );
        }
    }

    // Token limit: OpenAI o-series uses max_completion_tokens.
    if let Some(max_out) = payload.max_output_tokens {
        if is_openai_o_series(resolved_model) {
            body.insert("max_completion_tokens".to_owned(), json!(max_out));
        } else {
            body.insert("max_tokens".to_owned(), json!(max_out));
        }
    }

    // Sampling params. These are named fields on the request, so they are read
    // directly rather than from `extra` (a `#[serde(flatten)]` map that never
    // contains named fields).
    for (key, value) in [
        ("temperature", &payload.temperature),
        ("top_p", &payload.top_p),
    ] {
        if let Some(v) = value {
            if !v.is_null() {
                body.insert(key.to_owned(), v.clone());
            }
        }
    }

    // response_format is a named field, so it is read directly rather than
    // from `extra` (which never holds named fields).
    if let Some(rf) = &payload.response_format {
        if !rf.is_null() {
            body.insert("response_format".to_owned(), rf.clone());
        }
    }

    // Passthrough fields (always-on; stream-only omitted for brevity as they
    // overlap the always-on set in practice).
    for &field in EXTRA_PASSTHROUGH_FIELDS {
        if let Some(v) = payload.extra.get(field) {
            if !v.is_null() {
                body.insert(field.to_owned(), v.clone());
            }
        }
    }

    // Reasoning effort: normalize and encode per provider bucket.
    let effort_str = payload
        .reasoning
        .as_ref()
        .and_then(|r| r.get("effort"))
        .and_then(Value::as_str);
    let canonical = reasoning::normalize_canonical_effort(effort_str);
    if let Some(effort) = reasoning::wire_reasoning_effort(resolved_model, canonical) {
        body.insert("reasoning_effort".to_owned(), json!(effort));
    }

    ChatRequest { body }
}

fn append_input_items(
    payload: &ResponsesRequest,
    messages: &mut Vec<Value>,
    loss: &mut TransformLossCollector,
    tool_context: &crate::context::BridgeToolContext,
) {
    let items = iter_request_input_items(payload);
    let mut pending_tool_calls: Vec<Value> = Vec::new();
    let mut pending_reasoning: Option<String> = None;
    // Call ids already present in the (possibly session-restored) history, so a
    // continuation turn skips tool calls/outputs it already carries. Mirrors
    // the Python bridge's `existing_call_ids`.
    let skip_call_ids = existing_call_ids(messages);

    macro_rules! flush {
        () => {{
            flush_pending(messages, &mut pending_tool_calls, &mut pending_reasoning);
        }};
    }

    for item in items {
        let obj = match item.as_object() {
            Some(o) => o,
            None => {
                // iter_request_input_items lifts bare strings to input_text, so
                // a non-object here is genuinely malformed (number/bool/null).
                flush!();
                loss.record(
                    TransformLoss::SkippedNonDictItem,
                    None,
                    "Non-object input item skipped",
                );
                continue;
            }
        };
        let item_type = obj.get("type").and_then(Value::as_str).unwrap_or("");

        // Reasoning items.
        if item_type == "reasoning" {
            let text = reasoning_item_text(obj);
            let text = text.trim();
            if text.is_empty() {
                loss.record(
                    TransformLoss::DroppedEmptyReasoning,
                    non_empty(item_type),
                    "Reasoning item produced empty text after extraction and stripping",
                );
                continue;
            }
            if pending_tool_calls.is_empty() && append_reasoning_to_last_assistant(messages, text) {
                // backfilled onto the preceding assistant turn
            } else {
                pending_reasoning = Some(match pending_reasoning.take() {
                    Some(prev) => format!("{prev}\n\n{text}"),
                    None => text.to_owned(),
                });
            }
            continue;
        }

        // Text-like content → user message.
        if matches!(
            item_type,
            "input_text" | "output_text" | "text" | "latest_reminder"
        ) {
            flush!();
            let text = obj.get("text").and_then(Value::as_str).unwrap_or("");
            messages.push(ChatMessageBuilder::with_content("user", json!(text)).into_value());
            continue;
        }

        // Media items (input_image / input_audio) → user message content part.
        if matches!(item_type, "input_image" | "input_audio") {
            flush!();
            handle_media_item(obj, item_type, messages, loss);
            continue;
        }

        // Tool call items → accumulate.
        if handle_tool_call_item(
            obj,
            item_type,
            &mut pending_tool_calls,
            &mut pending_reasoning,
            &skip_call_ids,
            loss,
            tool_context,
        ) {
            continue;
        }

        // Tool output items → tool message (or orphan downgrade to user).
        if matches!(
            item_type,
            "function_call_output" | "custom_tool_call_output" | "tool_search_output"
        ) {
            let call_id = resolve_tool_call_id(obj);
            if should_skip(obj, &skip_call_ids) {
                loss.record(
                    TransformLoss::SkippedDuplicateToolCall,
                    non_empty(item_type),
                    "Duplicate tool output already in message history",
                );
                continue;
            }
            flush!();
            let content = if item_type == "function_call_output" {
                normalize_tool_output_content(obj.get("output"))
            } else {
                serde_json::to_string(obj).unwrap_or_else(|_| "{}".to_owned())
            };
            if has_matching_call(&call_id, messages) {
                let mut m = ChatMessageBuilder::with_content("tool", json!(content));
                m.tool_call_id = Some(call_id);
                messages.push(m.into_value());
            } else {
                // A tool message with no preceding assistant tool_call would be
                // rejected by Chat Completions; downgrade to a user message.
                loss.record(
                    TransformLoss::DowngradedOrphanToolOutput,
                    non_empty(item_type),
                    format!("Tool output {call_id} has no matching preceding tool call"),
                );
                let text = format!("Function call output ({call_id}): {content}");
                messages.push(ChatMessageBuilder::with_content("user", json!(text)).into_value());
            }
            continue;
        }

        // Generic role/content message.
        if obj.contains_key("role") || obj.contains_key("content") || item_type == "message" {
            flush!();
            messages.push(build_generic_message(obj, item_type, loss));
            continue;
        }

        // Unknown → skip permissively.
        flush!();
        loss.record(
            TransformLoss::SkippedUnknownItemType,
            non_empty(item_type),
            format!("Unrecognized input item type: {item_type:?}"),
        );
    }

    flush_pending(messages, &mut pending_tool_calls, &mut pending_reasoning);
}

/// The item `type` as a metric label, or `None` when the field was absent/empty
/// (which the metric renders as `"none"`).
fn non_empty(item_type: &str) -> Option<&str> {
    (!item_type.is_empty()).then_some(item_type)
}

/// Collect call ids already present in `messages`, from both assistant
/// `tool_calls[].id`/`call_id` and tool-role `tool_call_id`. Mirrors the Python
/// `existing_call_ids`.
fn existing_call_ids(messages: &[Value]) -> std::collections::HashSet<String> {
    let mut ids = std::collections::HashSet::new();
    for msg in messages {
        let Some(obj) = msg.as_object() else { continue };
        if let Some(id) = obj.get("tool_call_id").and_then(Value::as_str) {
            if !id.is_empty() {
                ids.insert(id.to_owned());
            }
        }
        if let Some(tcs) = obj.get("tool_calls").and_then(Value::as_array) {
            for tc in tcs {
                if let Some(id) = tc
                    .get("id")
                    .and_then(Value::as_str)
                    .or_else(|| tc.get("call_id").and_then(Value::as_str))
                {
                    if !id.is_empty() {
                        ids.insert(id.to_owned());
                    }
                }
            }
        }
    }
    ids
}

/// Whether an item's `call_id`/`id` was already seen in the session history.
/// Mirrors the Python `should_skip`.
fn should_skip(obj: &Map<String, Value>, skip_ids: &std::collections::HashSet<String>) -> bool {
    obj.get("call_id")
        .and_then(Value::as_str)
        .or_else(|| obj.get("id").and_then(Value::as_str))
        .is_some_and(|id| skip_ids.contains(id))
}

/// Whether `call_id` has a matching assistant `tool_calls[].id` in the flushed
/// message history. Mirrors the Python `has_matching_call` (the pending buffer
/// is always flushed into `messages` before this is called).
fn has_matching_call(call_id: &str, messages: &[Value]) -> bool {
    messages.iter().any(|msg| {
        msg.get("role").and_then(Value::as_str) == Some("assistant")
            && msg
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|tcs| {
                    tcs.iter()
                        .any(|tc| tc.get("id").and_then(Value::as_str) == Some(call_id))
                })
    })
}

/// Convert a top-level media item and push the resulting user message, or record
/// a transform-loss event when the URL/format is rejected. Mirrors the Python
/// `handle_media_item`.
fn handle_media_item(
    obj: &Map<String, Value>,
    item_type: &str,
    messages: &mut Vec<Value>,
    loss: &mut TransformLossCollector,
) {
    use crate::media::MediaConversion;
    let (conversion, loss_kind) = if item_type == "input_image" {
        (
            crate::media::image_part_from_input_item(obj),
            TransformLoss::SkippedUnsupportedImage,
        )
    } else {
        (
            crate::media::audio_part_from_input_item(obj),
            TransformLoss::SkippedUnsupportedAudio,
        )
    };
    match conversion {
        MediaConversion::Part(part) => {
            messages.push(ChatMessageBuilder::with_content("user", json!([part])).into_value());
        }
        MediaConversion::Rejected(reason) => {
            loss.record(loss_kind, non_empty(item_type), reason);
        }
    }
}

/// Drain a collector into the `bridge_transform_loss_total` metric and a single
/// warning log summarizing the events. Mirrors the Python request handler's
/// post-conversion loss reporting.
fn drain_transform_loss(loss: &TransformLossCollector) {
    if loss.is_empty() {
        return;
    }
    for event in loss.events() {
        crate::metrics::record_transform_loss(event.kind.name(), event.item_type.as_deref());
        tracing::debug!(
            kind = event.kind.name(),
            item_type = event.item_type.as_deref().unwrap_or("none"),
            "transform loss: {}",
            event.reason
        );
    }
    let summary = loss
        .events()
        .iter()
        .map(|e| {
            format!(
                "{}({})",
                e.kind.name(),
                e.item_type.as_deref().unwrap_or("n/a")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    tracing::warn!(
        "responses_to_chat transform loss: {} events [{}]",
        loss.events().len(),
        summary
    );
}

fn flush_pending(
    messages: &mut Vec<Value>,
    pending_tool_calls: &mut Vec<Value>,
    pending_reasoning: &mut Option<String>,
) {
    if pending_tool_calls.is_empty() && pending_reasoning.is_none() {
        return;
    }
    if !pending_tool_calls.is_empty() {
        let mut m = ChatMessageBuilder::new("assistant");
        m.tool_calls = std::mem::take(pending_tool_calls);
        m.reasoning_content = pending_reasoning.take();
        messages.push(m.into_value());
        return;
    }
    // reasoning-only assistant separator.
    let mut m = ChatMessageBuilder::with_content("assistant", json!(""));
    m.reasoning_content = pending_reasoning.take();
    messages.push(m.into_value());
}

fn handle_tool_call_item(
    obj: &Map<String, Value>,
    item_type: &str,
    pending_tool_calls: &mut Vec<Value>,
    pending_reasoning: &mut Option<String>,
    skip_call_ids: &std::collections::HashSet<String>,
    loss: &mut TransformLossCollector,
    tool_context: &crate::context::BridgeToolContext,
) -> bool {
    let (name, arguments): (String, String) = match item_type {
        "function_call" => {
            // A namespaced function call is flattened back to the Chat name the
            // upstream saw, so a continuation turn's tool_calls[].name matches
            // the tool schema. Mirrors `chat_name_for_function`.
            let raw_name = obj
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown_tool");
            let namespace = obj.get("namespace").and_then(Value::as_str);
            (
                tool_context.chat_name_for_function(raw_name, namespace),
                canonicalize_tool_arguments(obj.get("arguments")),
            )
        }
        "custom_tool_call" => (
            obj.get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown_tool")
                .to_owned(),
            // custom tool input → chat arguments (string passthrough in Phase 1).
            custom_tool_input_to_chat_arguments(obj.get("input")),
        ),
        "tool_search_call" => (
            crate::context::TOOL_SEARCH_PROXY_NAME.to_owned(),
            canonicalize_tool_arguments(obj.get("arguments")),
        ),
        _ => return false,
    };

    // A tool call whose id already appears in the session history is a
    // continuation duplicate; skip it (recognized, so return true).
    if should_skip(obj, skip_call_ids) {
        loss.record(
            TransformLoss::SkippedDuplicateToolCall,
            non_empty(item_type),
            "Duplicate tool call already in message history",
        );
        return true;
    }

    if let Some(rc) = obj.get("reasoning_content").and_then(Value::as_str) {
        *pending_reasoning = Some(match pending_reasoning.take() {
            Some(prev) => format!("{prev}\n\n{rc}"),
            None => rc.to_owned(),
        });
    }

    let call_id = resolve_tool_call_id(obj);
    pending_tool_calls.push(json!({
        "id": call_id,
        "type": "function",
        "function": { "name": name, "arguments": arguments },
    }));
    true
}

fn build_generic_message(
    obj: &Map<String, Value>,
    item_type: &str,
    loss: &mut TransformLossCollector,
) -> Value {
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");
    let chat_role = match role {
        "system" | "developer" => "system",
        "user" | "assistant" | "tool" => role,
        other => {
            loss.record(
                TransformLoss::DowngradedInvalidRole,
                non_empty(item_type),
                format!("Unrecognized message role {other:?} downgraded to user"),
            );
            "user"
        }
    };

    let content = chat_message_content_from_response_content(obj.get("content"));
    let mut m = ChatMessageBuilder::with_content(chat_role, content);

    if let Some(tcid) = obj.get("tool_call_id").and_then(Value::as_str) {
        m.tool_call_id = Some(tcid.to_owned());
    }
    if chat_role == "assistant" {
        if let Some(tcs) = obj.get("tool_calls").and_then(Value::as_array) {
            m.tool_calls = tcs.clone();
        }
    }
    if let Some(rc) = obj.get("reasoning_content").and_then(Value::as_str) {
        m.reasoning_content = Some(rc.to_owned());
    }
    m.into_value()
}

// --------------------------------------------------------------------------- //
// Response side: Chat body → Responses object
// --------------------------------------------------------------------------- //

/// Render an upstream Chat Completions body as a Responses object.
///
/// `response_id` is the bridge-owned top-level id; the caller supplies the same
/// value it persists and returns, so item ids match the streaming path exactly.
pub fn chat_to_responses(
    chat_body: &Value,
    fallback_model: &str,
    original_request: Option<&Map<String, Value>>,
    response_id: &str,
    tool_context: &crate::context::BridgeToolContext,
) -> Value {
    let empty = Value::Object(Map::new());
    let choice = chat_body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .unwrap_or(&empty);
    let message = choice.get("message").unwrap_or(&empty);

    let reasoning_text = extract_reasoning_text(message);
    let reasoning_text = if reasoning_text.is_empty() {
        String::new()
    } else {
        sanitize_string(&reasoning_text)
    };

    let model = chat_body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(fallback_model);
    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
    let created_at = chat_body.get("created").and_then(Value::as_i64);

    let mut output: Vec<Value> = Vec::new();

    if !reasoning_text.is_empty() {
        output.push(json!({
            "id": id_gen::reasoning_item_id(response_id),
            "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": reasoning_text }],
        }));
    }

    let parts = message_content_parts(message);
    let output_text = output_text_from_parts(&parts);
    if !parts.is_empty() {
        output.push(json!({
            "id": id_gen::message_item_id(response_id),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": parts,
        }));
    }

    output.extend(chat_tool_calls_to_response_items(
        message,
        &reasoning_text,
        tool_context,
    ));

    let mut response = Map::new();
    response.insert("id".to_owned(), json!(response_id));
    response.insert("object".to_owned(), json!("response"));
    response.insert(
        "status".to_owned(),
        json!(response_status_from_finish_reason(finish_reason).as_str()),
    );
    response.insert("model".to_owned(), json!(model));
    response.insert("output".to_owned(), Value::Array(output));
    response.insert("output_text".to_owned(), json!(output_text));
    response.insert("created_at".to_owned(), json!(created_at));
    response.insert("usage".to_owned(), map_chat_usage(chat_body.get("usage")));
    response.insert(
        "incomplete_details".to_owned(),
        incomplete_reason_from_finish_reason(finish_reason).unwrap_or(Value::Null),
    );

    // The Responses response object always carries the full spec field set.
    // Python's Pydantic model serializes every echo field (null default) even
    // when absent, so real clients (codex-tui) see a stable shape. Seed each
    // field as null, then let request-echo overwrite the ones the caller sent.
    for &key in REQUEST_ECHO_FIELDS {
        response.insert(key.to_owned(), Value::Null);
    }
    echo_request_fields(&mut response, original_request);
    Value::Object(response)
}

fn echo_request_fields(response: &mut Map<String, Value>, original: Option<&Map<String, Value>>) {
    let Some(original) = original else { return };
    for &key in REQUEST_ECHO_FIELDS {
        if let Some(value) = original.get(key) {
            if !value.is_null() {
                response.insert(key.to_owned(), value.clone());
            }
        }
    }
}

fn extract_reasoning_text(message: &Value) -> String {
    // The explicit reasoning-field lookup lives in the reasoning module (single
    // source of truth); it preserves original bytes, so trim here to match the
    // Responses summary shape.
    if let Some(obj) = message.as_object() {
        if let Some(field) = reasoning::extract_reasoning_field(obj) {
            let trimmed = field.trim();
            if !trimmed.is_empty() {
                return trimmed.to_owned();
            }
        }
    }
    // Inline <think> fallback.
    if let Some(content) = message.get("content").and_then(Value::as_str) {
        if let Some((reasoning, _answer)) = reasoning::split_inline_think(content) {
            if !reasoning.is_empty() {
                return reasoning;
            }
        }
    }
    String::new()
}

fn message_content_parts(message: &Value) -> Vec<Value> {
    let mut parts: Vec<Value> = Vec::new();
    let msg_annotations = extract_message_annotations(message);

    match message.get("content") {
        Some(Value::String(raw)) => {
            let stripped = match reasoning::split_inline_think(raw) {
                Some((reasoning, answer)) if !reasoning.is_empty() => answer,
                _ => raw.clone(),
            };
            if !stripped.is_empty() {
                parts.push(json!({
                    "type": "output_text",
                    "text": sanitize_string(&stripped),
                    "annotations": msg_annotations,
                }));
            }
        }
        Some(Value::Array(items)) => {
            for part in items {
                let Some(p) = part.as_object() else { continue };
                let ptype = p.get("type").and_then(Value::as_str).unwrap_or("");
                if matches!(ptype, "text" | "output_text") {
                    if let Some(text) = p.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            let part_ann = p
                                .get("annotations")
                                .and_then(Value::as_array)
                                .cloned()
                                .unwrap_or_default();
                            let mut merged = msg_annotations.clone();
                            for a in part_ann {
                                if !merged.contains(&a) {
                                    merged.push(a);
                                }
                            }
                            parts.push(json!({
                                "type": "output_text",
                                "text": sanitize_string(text),
                                "annotations": merged,
                            }));
                        }
                    }
                } else if ptype == "refusal" {
                    if let Some(refusal) = p.get("refusal").and_then(Value::as_str) {
                        if !refusal.is_empty() {
                            parts.push(json!({
                                "type": "refusal",
                                "refusal": sanitize_string(refusal),
                            }));
                        }
                    }
                }
            }
        }
        _ => {}
    }

    if let Some(refusal) = message.get("refusal").and_then(Value::as_str) {
        if !refusal.is_empty() {
            parts.push(json!({ "type": "refusal", "refusal": sanitize_string(refusal) }));
        }
    }

    parts
}

fn extract_message_annotations(message: &Value) -> Vec<Value> {
    message
        .get("annotations")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter(|a| a.is_object()).cloned().collect())
        .unwrap_or_default()
}

fn output_text_from_parts(parts: &[Value]) -> String {
    parts
        .iter()
        .filter(|p| p.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

fn chat_tool_calls_to_response_items(
    message: &Value,
    reasoning: &str,
    tool_context: &crate::context::BridgeToolContext,
) -> Vec<Value> {
    let mut output: Vec<Value> = Vec::new();
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tc in tool_calls {
            let Some(tc_obj) = tc.as_object() else {
                continue;
            };
            let function = tc_obj.get("function").and_then(Value::as_object);
            let call_id = tc_obj
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(id_gen::synthetic_tool_call_id);
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown_tool");
            let arguments = function.and_then(|f| f.get("arguments"));
            output.push(tool_call_to_response_item(
                &call_id,
                name,
                arguments,
                reasoning,
                tool_context,
            ));
        }
    }
    output
}

/// Convert a single Chat `tool_calls[]` entry back into a Responses output
/// item, classifying by the tool context: tool-search proxy → `tool_search_call`,
/// custom tool → `custom_tool_call`, nested-namespace call → `function_call`
/// with the action extracted from the arguments, and a plain/namespaced
/// function → `function_call` (restoring the original name + namespace).
/// Mirrors the Python `tool_call_to_response_item`.
fn tool_call_to_response_item(
    call_id: &str,
    name: &str,
    arguments: Option<&Value>,
    reasoning: &str,
    tool_context: &crate::context::BridgeToolContext,
) -> Value {
    let canonical = canonicalize_tool_arguments(arguments);
    let name_opt = Some(name);
    let spec = tool_context.lookup_chat_name(name_opt);

    let mut item = if tool_context.is_tool_search(name_opt) {
        json!({
            "id": id_gen::function_call_item_id(call_id),
            "type": "tool_search_call",
            "status": "completed",
            "call_id": call_id,
            "execution": "client",
            "arguments": crate::context::parse_tool_arguments_object(&canonical),
        })
    } else if tool_context.is_custom_tool(name_opt) {
        let restored = spec
            .map(|s| s.name.clone())
            .unwrap_or_else(|| name.to_owned());
        json!({
            "id": id_gen::custom_tool_call_item_id(call_id),
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": call_id,
            "name": restored,
            "input": crate::context::custom_tool_input_from_chat_arguments(&canonical),
        })
    } else if spec.map(|s| s.is_nested_namespace()).unwrap_or(false) {
        // Nested namespace call — the action lives inside the arguments JSON.
        let spec = spec.expect("nested namespace spec present");
        let resolution = crate::context::resolve_nested_namespace_arguments(spec, &canonical);
        let restored = resolution.action_name.unwrap_or_else(|| spec.name.clone());
        let mut nested = json!({
            "id": id_gen::function_call_item_id(call_id),
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": restored,
            "arguments": resolution.normalized_arguments,
        });
        if let Some(ns) = &spec.namespace {
            nested["namespace"] = json!(ns);
        }
        nested
    } else {
        let restored = spec
            .map(|s| s.name.clone())
            .unwrap_or_else(|| name.to_owned());
        let mut func = json!({
            "id": id_gen::function_call_item_id(call_id),
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": restored,
            "arguments": canonical,
        });
        if let Some(ns) = spec.and_then(|s| s.namespace.as_ref()) {
            func["namespace"] = json!(ns);
        }
        func
    };

    if !reasoning.is_empty() {
        item["reasoning_content"] = json!(sanitize_string(reasoning));
    }
    item
}

// --------------------------------------------------------------------------- //
// Semantics helpers (mirror response_semantics.py)
// --------------------------------------------------------------------------- //

/// A Responses top-level `status`, over its closed set of wire values. Replaces
/// the former stringly-typed status so a typo can't compile and the persist
/// guard is a total `match`. `as_str()` renders the exact protocol bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
}

impl ResponseStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Failed => "failed",
        }
    }

    /// Responses statuses safe to persist for `previous_response_id`
    /// continuation. `failed` / `incomplete` turns must not be saved, or a
    /// resume would replay a partial or invalid turn.
    pub(crate) fn is_persistable(self) -> bool {
        matches!(self, Self::Completed | Self::InProgress)
    }
}

pub(crate) fn response_status_from_finish_reason(finish_reason: Option<&str>) -> ResponseStatus {
    match finish_reason {
        Some("tool_calls") => ResponseStatus::InProgress,
        Some("length") | Some("content_filter") => ResponseStatus::Incomplete,
        _ => ResponseStatus::Completed,
    }
}

pub(crate) fn incomplete_reason_from_finish_reason(finish_reason: Option<&str>) -> Option<Value> {
    match finish_reason {
        Some("length") => Some(json!({ "reason": "max_output_tokens" })),
        Some("content_filter") => Some(json!({ "reason": "content_filter" })),
        _ => None,
    }
}

/// Whether a finalized response status is safe to persist for
/// `previous_response_id` continuation. `None` (never finalized) is not
/// persistable. Mirrors `should_persist_response_status`.
pub(crate) fn should_persist_response_status(status: Option<ResponseStatus>) -> bool {
    status.is_some_and(ResponseStatus::is_persistable)
}

/// Whether a Chat `finish_reason` maps to a persistable terminal state.
/// Mirrors `should_persist_finish_reason`.
pub(crate) fn should_persist_finish_reason(finish_reason: Option<&str>) -> bool {
    response_status_from_finish_reason(finish_reason).is_persistable()
}

/// Extract the first-choice `finish_reason` from a Chat Completions body.
/// Mirrors `chat_finish_reason`.
pub(crate) fn chat_finish_reason(chat_body: &Value) -> Option<String> {
    chat_body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

pub(crate) fn map_chat_usage(usage: Option<&Value>) -> Value {
    let Some(usage) = usage.and_then(Value::as_object) else {
        return json!({ "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 });
    };
    let get = |k: &str| usage.get(k).and_then(Value::as_i64).unwrap_or(0);
    let prompt = get("prompt_tokens");
    let completion = get("completion_tokens");
    let input = get("input_tokens").max(prompt);
    let output = get("output_tokens").max(completion);
    let total = get("total_tokens").max(input + output);

    let mut result = Map::new();
    result.insert("input_tokens".to_owned(), json!(input));
    result.insert("output_tokens".to_owned(), json!(output));
    result.insert("total_tokens".to_owned(), json!(total));

    let input_details = usage
        .get("input_tokens_details")
        .or_else(|| usage.get("prompt_tokens_details"));
    if let Some(d) = input_details {
        if !d.is_null() {
            result.insert("input_tokens_details".to_owned(), d.clone());
        }
    }
    let output_details = usage
        .get("output_tokens_details")
        .or_else(|| usage.get("completion_tokens_details"));
    if let Some(d) = output_details {
        if !d.is_null() {
            result.insert("output_tokens_details".to_owned(), d.clone());
        }
    }
    Value::Object(result)
}

// --------------------------------------------------------------------------- //
// Content / text helpers
// --------------------------------------------------------------------------- //

fn iter_request_input_items(payload: &ResponsesRequest) -> Vec<Value> {
    match &payload.input {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![json!({ "type": "input_text", "text": s })],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => json!({ "type": "input_text", "text": s }),
                other => other.clone(),
            })
            .collect(),
        Some(other) => vec![other.clone()],
    }
}

fn instruction_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let chunks: Vec<String> = items
                .iter()
                .filter_map(|part| match part {
                    Value::String(s) if !s.is_empty() => Some(s.clone()),
                    Value::Object(o) => o
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned),
                    _ => None,
                })
                .collect();
            chunks.join("\n\n")
        }
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn flatten_text_content(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let chunks: Vec<String> = items
                .iter()
                .filter_map(|item| match item {
                    Value::String(s) if !s.is_empty() => Some(s.clone()),
                    Value::Object(o) => {
                        let typ = o.get("type").and_then(Value::as_str).unwrap_or("");
                        if matches!(typ, "input_text" | "output_text" | "text") {
                            o.get("text")
                                .and_then(Value::as_str)
                                .filter(|s| !s.is_empty())
                                .map(str::to_owned)
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect();
            chunks.join("\n")
        }
        _ => String::new(),
    }
}

fn reasoning_item_text(item: &Map<String, Value>) -> String {
    // 1. summary list of summary_text parts.
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        let chunks: Vec<String> = summary
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        let text = chunks.join("\n\n");
        if !text.is_empty() {
            return text;
        }
    }
    // 2/3/4: bare string fields, in priority order.
    for field in ["text", "reasoning_content", "encrypted_content"] {
        if let Some(s) = item.get(field).and_then(Value::as_str) {
            if !s.is_empty() {
                return s.to_owned();
            }
        }
    }
    String::new()
}

pub(crate) fn chat_message_content_from_response_content(content: Option<&Value>) -> Value {
    let Some(content) = content else {
        return Value::Null;
    };
    match content {
        Value::Null => Value::Null,
        Value::String(s) => json!(s),
        Value::Array(items) => {
            let mut parts: Vec<Value> = Vec::new();
            for item in items {
                match item {
                    Value::String(s) if !s.is_empty() => {
                        parts.push(json!({ "type": "text", "text": sanitize_string(s) }));
                    }
                    Value::Object(o) => {
                        let typ = o.get("type").and_then(Value::as_str).unwrap_or("");
                        if matches!(typ, "input_text" | "output_text" | "text") {
                            if let Some(text) = o.get("text").and_then(Value::as_str) {
                                parts
                                    .push(json!({ "type": "text", "text": sanitize_string(text) }));
                            }
                        } else if typ == "refusal" {
                            if let Some(r) = o.get("refusal").and_then(Value::as_str) {
                                if !r.is_empty() {
                                    parts.push(json!({
                                        "type": "text",
                                        "text": format!("[refusal]: {}", sanitize_string(r)),
                                    }));
                                }
                            }
                        }
                        // image/audio parts land in Phase 2+ (multimodal).
                    }
                    _ => {}
                }
            }
            if parts.is_empty() {
                return json!("");
            }
            // All-text collapses to a joined string.
            if parts
                .iter()
                .all(|p| p.get("type").and_then(Value::as_str) == Some("text"))
            {
                let joined = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");
                return json!(joined);
            }
            Value::Array(parts)
        }
        other => json!(flatten_text_content(other)),
    }
}

fn normalize_tool_output_content(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(o)) => {
            if let Some(content) = o.get("content") {
                if content.is_array() {
                    let flat = flatten_text_content(content);
                    if !flat.is_empty() {
                        return flat;
                    }
                }
            }
            let typ = o.get("type").and_then(Value::as_str).unwrap_or("");
            if matches!(typ, "input_text" | "output_text" | "text") {
                if let Some(text) = o.get("text").and_then(Value::as_str) {
                    return text.to_owned();
                }
            }
            canonical_json_string(value.unwrap())
        }
        Some(arr @ Value::Array(_)) => {
            let flat = flatten_text_content(arr);
            if !flat.is_empty() {
                flat
            } else {
                canonical_json_string(arr)
            }
        }
        Some(other) => canonical_json_string(other),
    }
}

// --------------------------------------------------------------------------- //
// Message normalization (mirror message_normalization.py)
// --------------------------------------------------------------------------- //

fn sanitize_messages(messages: Vec<Value>) -> Vec<Value> {
    if messages.is_empty() {
        return messages;
    }

    // Step 1: normalize — keep user/system always; assistant with no content /
    // tool_calls / tool_call_id becomes a reasoning-only separator (content="").
    let mut sanitized: Vec<Value> = Vec::new();
    for msg in messages {
        let Some(obj) = msg.as_object() else { continue };
        let role = obj.get("role").and_then(Value::as_str).unwrap_or("");
        let has_content = obj.get("content").is_some_and(|c| {
            !c.is_null() && c.as_str() != Some("") && !c.as_array().is_some_and(|a| a.is_empty())
        });
        let has_tool_calls = obj
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty());
        let has_tool_call_id = obj
            .get("tool_call_id")
            .and_then(Value::as_str)
            .is_some_and(|s| !s.is_empty());

        if role == "assistant" && !has_content && !has_tool_calls && !has_tool_call_id {
            let mut sep = Map::new();
            sep.insert("role".to_owned(), json!("assistant"));
            sep.insert("content".to_owned(), json!(""));
            if let Some(rc) = obj.get("reasoning_content") {
                sep.insert("reasoning_content".to_owned(), rc.clone());
            }
            sanitized.push(Value::Object(sep));
            continue;
        }
        if role == "user" || role == "system" || has_content || has_tool_calls || has_tool_call_id {
            sanitized.push(msg);
        }
    }
    if sanitized.is_empty() {
        return sanitized;
    }

    // Step 2: merge adjacent same-role system messages.
    let mut it = sanitized.into_iter();
    let mut merged: Vec<Value> = vec![it.next().unwrap()];
    for msg in it {
        let prev_role = merged
            .last()
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str);
        let cur_role = msg.get("role").and_then(Value::as_str);
        if prev_role == cur_role && cur_role == Some("system") {
            let prev = merged.last().unwrap();
            let a = flatten_text_content(prev.get("content").unwrap_or(&Value::Null));
            let b = flatten_text_content(msg.get("content").unwrap_or(&Value::Null));
            let combined = [a.trim(), b.trim()]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            let content = if combined.is_empty() {
                Value::Null
            } else {
                json!(combined)
            };
            *merged.last_mut().unwrap() = json!({ "role": "system", "content": content });
        } else {
            merged.push(msg);
        }
    }

    // Step 2b: character sanitization of string fields.
    merged.into_iter().map(sanitize_message_strings).collect()
}

fn sanitize_message_strings(msg: Value) -> Value {
    let Some(obj) = msg.as_object() else {
        return msg;
    };
    let mut out = obj.clone();

    if let Some(Value::String(s)) = out.get("content") {
        out.insert("content".to_owned(), json!(sanitize_string(s)));
    } else if let Some(Value::Array(items)) = out.get("content") {
        let cleaned: Vec<Value> = items
            .iter()
            .map(|p| {
                if let Some(po) = p.as_object() {
                    if let Some(text) = po.get("text").and_then(Value::as_str) {
                        let mut np = po.clone();
                        np.insert("text".to_owned(), json!(sanitize_string(text)));
                        return Value::Object(np);
                    }
                }
                p.clone()
            })
            .collect();
        out.insert("content".to_owned(), Value::Array(cleaned));
    }

    for field in ["reasoning_content", "tool_call_id"] {
        if let Some(s) = out.get(field).and_then(Value::as_str) {
            out.insert(field.to_owned(), json!(sanitize_string(s)));
        }
    }

    if let Some(Value::Array(tcs)) = out.get("tool_calls") {
        let cleaned: Vec<Value> = tcs
            .iter()
            .map(|tc| {
                let Some(tco) = tc.as_object() else {
                    return tc.clone();
                };
                let mut ntc = tco.clone();
                if let Some(Value::Object(fn_obj)) = tco.get("function") {
                    let mut nfn = fn_obj.clone();
                    let name = fn_obj.get("name").and_then(Value::as_str).unwrap_or("");
                    let args = fn_obj
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    nfn.insert("name".to_owned(), json!(sanitize_string(name)));
                    nfn.insert("arguments".to_owned(), json!(sanitize_string(args)));
                    ntc.insert("function".to_owned(), Value::Object(nfn));
                }
                Value::Object(ntc)
            })
            .collect();
        out.insert("tool_calls".to_owned(), Value::Array(cleaned));
    }

    Value::Object(out)
}

fn collapse_system_messages_to_head(messages: Vec<Value>) -> Vec<Value> {
    let mut system_chunks: Vec<String> = Vec::new();
    let mut rest: Vec<Value> = Vec::new();
    for msg in messages {
        if msg.get("role").and_then(Value::as_str) == Some("system") {
            let text = flatten_text_content(msg.get("content").unwrap_or(&Value::Null));
            let text = text.trim();
            if !text.is_empty() {
                system_chunks.push(text.to_owned());
            }
            continue;
        }
        rest.push(msg);
    }
    if system_chunks.is_empty() {
        return rest;
    }
    let mut result = vec![json!({ "role": "system", "content": system_chunks.join("\n\n") })];
    result.extend(rest);
    result
}

fn append_reasoning_to_last_assistant(messages: &mut [Value], text: &str) -> bool {
    for msg in messages.iter_mut().rev() {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        if obj.get("role").and_then(Value::as_str) == Some("assistant") {
            let existing = obj.get("reasoning_content").and_then(Value::as_str);
            let combined = match existing {
                Some(prev) if !prev.is_empty() => format!("{prev}\n\n{text}"),
                _ => text.to_owned(),
            };
            obj.insert("reasoning_content".to_owned(), json!(combined));
            return true;
        }
        // Only backfill the immediately-preceding assistant; stop at any other role.
        return false;
    }
    false
}

// --------------------------------------------------------------------------- //
// Tool argument helpers (mirror tool_arguments.py canonicalization)
// --------------------------------------------------------------------------- //

/// Canonicalize tool arguments into a deterministic JSON string. A parseable
/// JSON value is re-serialized with sorted keys; an unparseable string passes
/// through unchanged; missing input yields "{}".
pub(crate) fn canonicalize_tool_arguments(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "{}".to_owned(),
        Some(Value::String(s)) => {
            // A string that is empty or whitespace-only canonicalizes to "{}",
            // matching the Python `raw = arguments.strip(); if not raw`.
            if s.trim().is_empty() {
                return "{}".to_owned();
            }
            // Try to parse then re-dump sorted; fall back to the raw string.
            match serde_json::from_str::<Value>(s) {
                Ok(parsed) => canonical_json_string(&parsed),
                Err(_) => s.clone(),
            }
        }
        Some(other) => canonical_json_string(other),
    }
}

fn custom_tool_input_to_chat_arguments(input: Option<&Value>) -> String {
    match input {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => canonical_json_string(other),
    }
}

/// Deterministic JSON serialization with sorted keys (serde_json sorts object
/// keys when the `preserve_order` feature is off, which is the default).
fn canonical_json_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned())
}

fn resolve_tool_call_id(obj: &Map<String, Value>) -> String {
    obj.get("call_id")
        .and_then(Value::as_str)
        .or_else(|| obj.get("id").and_then(Value::as_str))
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(id_gen::synthetic_tool_call_id)
}

fn is_openai_o_series(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    // o1 / o3 / o4 families use max_completion_tokens.
    m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4")
}

// --------------------------------------------------------------------------- //
// Tool-choice conversion (Responses → Chat Completions).
//
// Tool *definitions* are normalized by the `BridgeToolContext` registry (see
// `context.rs`), which owns namespace flattening, custom-tool input schemas,
// and the tool-search proxy. Only tool_choice rewriting lives here.
// --------------------------------------------------------------------------- //

/// Convert a Responses `tool_choice` into the Chat Completions form. String
/// modes (`auto`/`none`/`required`) pass through; an explicit tool object is
/// rewritten to `{type:function, function:{name}}`. Mirrors the Python
/// `_responses_tool_choice_to_chat`.
fn responses_tool_choice_to_chat(
    tool_choice: &Value,
    tool_context: &crate::context::BridgeToolContext,
) -> Value {
    let Some(obj) = tool_choice.as_object() else {
        return tool_choice.clone();
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("tool_search") => {
            json!({ "type": "function", "function": { "name": crate::context::TOOL_SEARCH_PROXY_NAME } })
        }
        Some("function") | Some("custom") => {
            match obj
                .get("name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                Some(name) => {
                    // A namespaced tool_choice is flattened to the Chat name the
                    // upstream schema declares. Mirrors the Python
                    // `_responses_tool_choice_to_chat`.
                    let namespace = obj.get("namespace").and_then(Value::as_str);
                    let chat_name = tool_context.chat_name_for_function(name, namespace);
                    json!({ "type": "function", "function": { "name": chat_name } })
                }
                None => tool_choice.clone(),
            }
        }
        _ => tool_choice.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `ResponsesRequest` carrying only an `input` field, for exercising
    /// the request-side input helpers without hand-constructing every field.
    fn req_with_input(input: Value) -> ResponsesRequest {
        serde_json::from_value(json!({ "input": input })).unwrap()
    }

    #[test]
    fn tool_choice_string_modes_pass_through() {
        let ctx = crate::context::BridgeToolContext::new();
        for mode in ["auto", "none", "required"] {
            assert_eq!(
                responses_tool_choice_to_chat(&json!(mode), &ctx),
                json!(mode)
            );
        }
    }

    #[test]
    fn tool_choice_explicit_object_is_rewritten_nested() {
        let ctx = crate::context::BridgeToolContext::new();
        let out = responses_tool_choice_to_chat(&json!({ "type": "function", "name": "f" }), &ctx);
        assert_eq!(
            out,
            json!({ "type": "function", "function": { "name": "f" } })
        );
    }

    #[test]
    fn tool_choice_tool_search_maps_to_proxy() {
        let ctx = crate::context::BridgeToolContext::new();
        let out = responses_tool_choice_to_chat(&json!({ "type": "tool_search" }), &ctx);
        assert_eq!(
            out,
            json!({ "type": "function", "function": { "name": crate::context::TOOL_SEARCH_PROXY_NAME } }),
        );
    }

    // ----------------------------------------------------------------------- //
    // Response side: status / incomplete-details / usage mapping
    // ----------------------------------------------------------------------- //

    #[test]
    fn status_from_finish_reason_matches_python_table() {
        assert_eq!(
            response_status_from_finish_reason(Some("tool_calls")).as_str(),
            "in_progress"
        );
        assert_eq!(
            response_status_from_finish_reason(Some("length")).as_str(),
            "incomplete"
        );
        assert_eq!(
            response_status_from_finish_reason(Some("content_filter")).as_str(),
            "incomplete"
        );
        assert_eq!(
            response_status_from_finish_reason(Some("stop")).as_str(),
            "completed"
        );
        assert_eq!(
            response_status_from_finish_reason(None).as_str(),
            "completed"
        );
    }

    #[test]
    fn incomplete_reason_only_for_length_and_filter() {
        assert_eq!(
            incomplete_reason_from_finish_reason(Some("length")),
            Some(json!({ "reason": "max_output_tokens" }))
        );
        assert_eq!(
            incomplete_reason_from_finish_reason(Some("content_filter")),
            Some(json!({ "reason": "content_filter" }))
        );
        assert_eq!(incomplete_reason_from_finish_reason(Some("stop")), None);
        assert_eq!(incomplete_reason_from_finish_reason(None), None);
    }

    #[test]
    fn usage_missing_yields_zeroed_object() {
        assert_eq!(
            map_chat_usage(None),
            json!({ "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 })
        );
    }

    #[test]
    fn usage_takes_max_of_old_and_new_token_fields() {
        // NewAPI-style: both prompt_tokens and zero-filled input_tokens present.
        let usage = json!({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "input_tokens": 0,
            "output_tokens": 0,
        });
        let out = map_chat_usage(Some(&usage));
        assert_eq!(out["input_tokens"], json!(10));
        assert_eq!(out["output_tokens"], json!(20));
        assert_eq!(out["total_tokens"], json!(30));
    }

    #[test]
    fn usage_prefers_explicit_total_when_larger() {
        let usage = json!({
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 99,
        });
        let out = map_chat_usage(Some(&usage));
        assert_eq!(out["total_tokens"], json!(99));
    }

    #[test]
    fn usage_carries_token_details_from_either_naming() {
        let usage = json!({
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "prompt_tokens_details": { "cached_tokens": 3 },
            "completion_tokens_details": { "reasoning_tokens": 7 },
        });
        let out = map_chat_usage(Some(&usage));
        assert_eq!(out["input_tokens_details"], json!({ "cached_tokens": 3 }));
        assert_eq!(
            out["output_tokens_details"],
            json!({ "reasoning_tokens": 7 })
        );
    }

    // ----------------------------------------------------------------------- //
    // Response side: reasoning + content extraction
    // ----------------------------------------------------------------------- //

    #[test]
    fn reasoning_prefers_reasoning_content_field() {
        let msg = json!({ "reasoning_content": " thinking ", "reasoning": "other" });
        assert_eq!(extract_reasoning_text(&msg), "thinking");
    }

    #[test]
    fn reasoning_falls_back_to_reasoning_field() {
        let msg = json!({ "reasoning": " deduced " });
        assert_eq!(extract_reasoning_text(&msg), "deduced");
    }

    #[test]
    fn reasoning_extracts_inline_think_block() {
        let msg = json!({ "content": "<think>step one</think>answer" });
        assert_eq!(extract_reasoning_text(&msg), "step one");
    }

    #[test]
    fn reasoning_absent_yields_empty() {
        let msg = json!({ "content": "plain answer" });
        assert_eq!(extract_reasoning_text(&msg), "");
    }

    #[test]
    fn content_string_becomes_single_output_text_part() {
        let msg = json!({ "content": "hello" });
        let parts = message_content_parts(&msg);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], json!("output_text"));
        assert_eq!(parts[0]["text"], json!("hello"));
        assert_eq!(parts[0]["annotations"], json!([]));
    }

    #[test]
    fn content_string_strips_inline_think_prefix() {
        let msg = json!({ "content": "<think>reason</think>final" });
        let parts = message_content_parts(&msg);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["text"], json!("final"));
    }

    #[test]
    fn content_array_merges_message_and_part_annotations() {
        let msg = json!({
            "annotations": [{ "type": "url_citation", "url": "a" }],
            "content": [{
                "type": "output_text",
                "text": "hi",
                "annotations": [{ "type": "url_citation", "url": "b" }],
            }],
        });
        let parts = message_content_parts(&msg);
        assert_eq!(parts.len(), 1);
        let ann = parts[0]["annotations"].as_array().unwrap();
        assert_eq!(ann.len(), 2);
    }

    #[test]
    fn content_array_refusal_part_is_preserved() {
        let msg = json!({ "content": [{ "type": "refusal", "refusal": "no" }] });
        let parts = message_content_parts(&msg);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], json!("refusal"));
        assert_eq!(parts[0]["refusal"], json!("no"));
    }

    #[test]
    fn top_level_refusal_field_appends_refusal_part() {
        let msg = json!({ "content": "text", "refusal": "denied" });
        let parts = message_content_parts(&msg);
        // output_text + refusal
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1]["type"], json!("refusal"));
        assert_eq!(parts[1]["refusal"], json!("denied"));
    }

    #[test]
    fn empty_content_string_yields_no_parts() {
        let msg = json!({ "content": "" });
        assert!(message_content_parts(&msg).is_empty());
    }

    #[test]
    fn output_text_joins_only_output_text_parts_with_newline() {
        let parts = vec![
            json!({ "type": "output_text", "text": "a" }),
            json!({ "type": "refusal", "refusal": "x" }),
            json!({ "type": "output_text", "text": "b" }),
        ];
        assert_eq!(output_text_from_parts(&parts), "a\nb");
    }

    // ----------------------------------------------------------------------- //
    // Response side: tool-call items
    // ----------------------------------------------------------------------- //

    #[test]
    fn tool_calls_become_function_call_items() {
        let msg = json!({
            "tool_calls": [{
                "id": "call_1",
                "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
            }],
        });
        let ctx = crate::context::BridgeToolContext::new();
        let items = chat_tool_calls_to_response_items(&msg, "", &ctx);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], json!("function_call"));
        assert_eq!(items[0]["call_id"], json!("call_1"));
        assert_eq!(items[0]["name"], json!("get_weather"));
        assert_eq!(items[0]["status"], json!("completed"));
        // arguments canonicalized (sorted keys, single-key here).
        assert_eq!(items[0]["arguments"], json!("{\"city\":\"Paris\"}"));
    }

    #[test]
    fn tool_call_item_carries_reasoning_when_present() {
        let ctx = crate::context::BridgeToolContext::new();
        let item = tool_call_to_response_item("c1", "f", Some(&json!("{}")), "why", &ctx);
        assert_eq!(item["reasoning_content"], json!("why"));
    }

    #[test]
    fn tool_call_item_omits_reasoning_when_empty() {
        let ctx = crate::context::BridgeToolContext::new();
        let item = tool_call_to_response_item("c1", "f", Some(&json!("{}")), "", &ctx);
        assert!(item.get("reasoning_content").is_none());
    }

    #[test]
    fn tool_call_missing_name_defaults_to_unknown_tool() {
        let msg = json!({ "tool_calls": [{ "id": "c", "function": {} }] });
        let ctx = crate::context::BridgeToolContext::new();
        let items = chat_tool_calls_to_response_items(&msg, "", &ctx);
        assert_eq!(items[0]["name"], json!("unknown_tool"));
    }

    // ----------------------------------------------------------------------- //
    // Content / text primitives
    // ----------------------------------------------------------------------- //

    #[test]
    fn instruction_text_from_string_and_array() {
        assert_eq!(instruction_text(&json!("go")), "go");
        let arr = json!(["one", { "text": "two" }, { "no_text": true }]);
        assert_eq!(instruction_text(&arr), "one\n\ntwo");
        assert_eq!(instruction_text(&Value::Null), "");
    }

    #[test]
    fn flatten_text_content_variants() {
        assert_eq!(flatten_text_content(&json!("s")), "s");
        assert_eq!(flatten_text_content(&Value::Null), "");
        let arr = json!([
            "bare",
            { "type": "input_text", "text": "a" },
            { "type": "output_text", "text": "b" },
            { "type": "image", "text": "ignored" },
        ]);
        assert_eq!(flatten_text_content(&arr), "bare\na\nb");
    }

    #[test]
    fn reasoning_item_text_prefers_summary_list() {
        let item = json!({
            "summary": [{ "text": "p1" }, { "text": "p2" }],
            "text": "ignored",
        });
        assert_eq!(reasoning_item_text(item.as_object().unwrap()), "p1\n\np2");
    }

    #[test]
    fn reasoning_item_text_falls_back_through_fields() {
        let item = json!({ "reasoning_content": "rc" });
        assert_eq!(reasoning_item_text(item.as_object().unwrap()), "rc");
        let enc = json!({ "encrypted_content": "ec" });
        assert_eq!(reasoning_item_text(enc.as_object().unwrap()), "ec");
        assert_eq!(reasoning_item_text(json!({}).as_object().unwrap()), "");
    }

    #[test]
    fn iter_request_input_items_string_wraps_as_input_text() {
        let payload = req_with_input(json!("hi"));
        let items = iter_request_input_items(&payload);
        assert_eq!(items, vec![json!({ "type": "input_text", "text": "hi" })]);
    }

    #[test]
    fn iter_request_input_items_array_lifts_bare_strings() {
        let payload = req_with_input(json!(["a", { "type": "message", "role": "user" }]));
        let items = iter_request_input_items(&payload);
        assert_eq!(items[0], json!({ "type": "input_text", "text": "a" }));
        assert_eq!(items[1], json!({ "type": "message", "role": "user" }));
    }

    #[test]
    fn iter_request_input_items_none_is_empty() {
        let payload = req_with_input(Value::Null);
        assert!(iter_request_input_items(&payload).is_empty());
    }

    #[test]
    fn chat_content_all_text_collapses_to_string() {
        let content = json!([
            { "type": "input_text", "text": "a" },
            { "type": "text", "text": "b" },
        ]);
        assert_eq!(
            chat_message_content_from_response_content(Some(&content)),
            json!("a\nb")
        );
    }

    #[test]
    fn chat_content_refusal_part_is_prefixed() {
        let content = json!([{ "type": "refusal", "refusal": "no" }]);
        assert_eq!(
            chat_message_content_from_response_content(Some(&content)),
            json!("[refusal]: no")
        );
    }

    #[test]
    fn chat_content_empty_array_becomes_empty_string() {
        let content = json!([]);
        assert_eq!(
            chat_message_content_from_response_content(Some(&content)),
            json!("")
        );
    }

    #[test]
    fn tool_output_flattens_string_and_content_array() {
        assert_eq!(normalize_tool_output_content(Some(&json!("raw"))), "raw");
        let obj = json!({ "content": [{ "type": "output_text", "text": "flat" }] });
        assert_eq!(normalize_tool_output_content(Some(&obj)), "flat");
        assert_eq!(normalize_tool_output_content(None), "");
    }

    // ----------------------------------------------------------------------- //
    // Message normalization
    // ----------------------------------------------------------------------- //

    #[test]
    fn sanitize_drops_empty_assistant_to_separator() {
        let msgs = vec![json!({
            "role": "assistant",
            "reasoning_content": "rc",
        })];
        let out = sanitize_messages(msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], json!("assistant"));
        assert_eq!(out[0]["content"], json!(""));
        assert_eq!(out[0]["reasoning_content"], json!("rc"));
    }

    #[test]
    fn sanitize_keeps_user_and_system_even_when_empty() {
        let msgs = vec![
            json!({ "role": "user", "content": "" }),
            json!({ "role": "system", "content": "" }),
        ];
        let out = sanitize_messages(msgs);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn sanitize_keeps_assistant_with_tool_calls() {
        let msgs = vec![json!({
            "role": "assistant",
            "tool_calls": [{ "id": "c", "function": { "name": "f", "arguments": "{}" } }],
        })];
        let out = sanitize_messages(msgs);
        assert_eq!(out.len(), 1);
        assert!(out[0].get("tool_calls").is_some());
    }

    #[test]
    fn sanitize_merges_adjacent_system_messages() {
        let msgs = vec![
            json!({ "role": "system", "content": "a" }),
            json!({ "role": "system", "content": "b" }),
        ];
        let out = sanitize_messages(msgs);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["content"], json!("a\n\nb"));
    }

    #[test]
    fn sanitize_cleans_control_chars_in_content() {
        let msgs = vec![json!({ "role": "user", "content": "hi\u{0000}there" })];
        let out = sanitize_messages(msgs);
        assert_eq!(out[0]["content"], json!("hithere"));
    }

    #[test]
    fn sanitize_cleans_tool_call_function_fields() {
        let msgs = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "c",
                "function": { "name": "f\u{0000}n", "arguments": "{\u{0000}}" },
            }],
        })];
        let out = sanitize_messages(msgs);
        let f = &out[0]["tool_calls"][0]["function"];
        assert_eq!(f["name"], json!("fn"));
        assert_eq!(f["arguments"], json!("{}"));
    }

    #[test]
    fn collapse_moves_all_system_text_to_head() {
        let msgs = vec![
            json!({ "role": "user", "content": "u1" }),
            json!({ "role": "system", "content": "s1" }),
            json!({ "role": "assistant", "content": "a1" }),
            json!({ "role": "system", "content": "s2" }),
        ];
        let out = collapse_system_messages_to_head(msgs);
        assert_eq!(out[0]["role"], json!("system"));
        assert_eq!(out[0]["content"], json!("s1\n\ns2"));
        assert_eq!(out[1]["role"], json!("user"));
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn collapse_no_system_leaves_messages_unchanged() {
        let msgs = vec![json!({ "role": "user", "content": "u" })];
        let out = collapse_system_messages_to_head(msgs.clone());
        assert_eq!(out, msgs);
    }

    #[test]
    fn append_reasoning_backfills_last_assistant() {
        let mut msgs = vec![
            json!({ "role": "user", "content": "u" }),
            json!({ "role": "assistant", "content": "a" }),
        ];
        assert!(append_reasoning_to_last_assistant(&mut msgs, "why"));
        assert_eq!(msgs[1]["reasoning_content"], json!("why"));
    }

    #[test]
    fn append_reasoning_concatenates_existing() {
        let mut msgs = vec![json!({
            "role": "assistant",
            "content": "a",
            "reasoning_content": "prev",
        })];
        append_reasoning_to_last_assistant(&mut msgs, "next");
        assert_eq!(msgs[0]["reasoning_content"], json!("prev\n\nnext"));
    }

    #[test]
    fn append_reasoning_stops_at_non_assistant_tail() {
        let mut msgs = vec![json!({ "role": "user", "content": "u" })];
        assert!(!append_reasoning_to_last_assistant(&mut msgs, "why"));
    }

    // ----------------------------------------------------------------------- //
    // Input-item assembly (append_input_items + helpers)
    // ----------------------------------------------------------------------- //

    /// Run `append_input_items` for the given `input` and return the messages.
    fn build_messages(input: Value) -> Vec<Value> {
        build_messages_with_loss(input).0
    }

    /// Run `append_input_items` and return both the messages and the collected
    /// transform-loss events, for tests asserting on degradation.
    fn build_messages_with_loss(input: Value) -> (Vec<Value>, TransformLossCollector) {
        let payload = req_with_input(input);
        let tool_context = crate::context::build_tool_context_from_request(&payload);
        let mut messages = Vec::new();
        let mut loss = TransformLossCollector::new();
        append_input_items(&payload, &mut messages, &mut loss, &tool_context);
        (messages, loss)
    }

    #[test]
    fn text_item_becomes_user_message() {
        let msgs = build_messages(json!([{ "type": "input_text", "text": "hello" }]));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], json!("user"));
        assert_eq!(msgs[0]["content"], json!("hello"));
    }

    #[test]
    fn function_call_item_becomes_assistant_tool_call() {
        let msgs = build_messages(json!([{
            "type": "function_call",
            "call_id": "c1",
            "name": "get_weather",
            "arguments": "{\"city\":\"Paris\"}",
        }]));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], json!("assistant"));
        let tc = &msgs[0]["tool_calls"][0];
        assert_eq!(tc["id"], json!("c1"));
        assert_eq!(tc["function"]["name"], json!("get_weather"));
        assert_eq!(tc["function"]["arguments"], json!("{\"city\":\"Paris\"}"));
    }

    #[test]
    fn matched_function_call_output_becomes_tool_message() {
        // A tool output whose call_id matches a preceding call becomes a proper
        // tool-role message.
        let msgs = build_messages(json!([
            { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c1", "output": "sunny" },
        ]));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], json!("tool"));
        assert_eq!(msgs[1]["tool_call_id"], json!("c1"));
        assert_eq!(msgs[1]["content"], json!("sunny"));
    }

    #[test]
    fn orphan_function_call_output_downgrades_to_user() {
        // A tool output with no matching preceding call would be rejected by
        // Chat Completions as an orphan tool message, so it is downgraded to a
        // user message and recorded as transform loss.
        let (msgs, loss) = build_messages_with_loss(json!([{
            "type": "function_call_output",
            "call_id": "c1",
            "output": "sunny",
        }]));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], json!("user"));
        assert_eq!(
            msgs[0]["content"],
            json!("Function call output (c1): sunny")
        );
        assert_eq!(loss.events().len(), 1);
        assert_eq!(
            loss.events()[0].kind,
            TransformLoss::DowngradedOrphanToolOutput
        );
    }

    #[test]
    fn call_then_output_pairs_across_flush() {
        let msgs = build_messages(json!([
            { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c1", "output": "done" },
        ]));
        // The pending tool-call flushes to an assistant message before the tool
        // output message is appended.
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], json!("assistant"));
        assert_eq!(msgs[0]["tool_calls"][0]["id"], json!("c1"));
        assert_eq!(msgs[1]["role"], json!("tool"));
        assert_eq!(msgs[1]["tool_call_id"], json!("c1"));
    }

    #[test]
    fn reasoning_before_tool_call_rides_on_assistant() {
        let msgs = build_messages(json!([
            { "type": "reasoning", "summary": [{ "text": "because" }] },
            { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
        ]));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], json!("assistant"));
        assert_eq!(msgs[0]["reasoning_content"], json!("because"));
        assert_eq!(msgs[0]["tool_calls"][0]["id"], json!("c1"));
    }

    #[test]
    fn reasoning_only_flushes_to_assistant_separator() {
        let msgs = build_messages(json!([{ "type": "reasoning", "text": "lonely" }]));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], json!("assistant"));
        assert_eq!(msgs[0]["content"], json!(""));
        assert_eq!(msgs[0]["reasoning_content"], json!("lonely"));
    }

    #[test]
    fn reasoning_backfills_preceding_assistant_message() {
        let msgs = build_messages(json!([
            { "type": "message", "role": "assistant", "content": "answer" },
            { "type": "reasoning", "text": "afterthought" },
        ]));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], json!("assistant"));
        assert_eq!(msgs[0]["reasoning_content"], json!("afterthought"));
    }

    #[test]
    fn custom_tool_call_uses_input_as_arguments() {
        let msgs = build_messages(json!([{
            "type": "custom_tool_call",
            "call_id": "c1",
            "name": "runner",
            "input": "raw payload",
        }]));
        assert_eq!(
            msgs[0]["tool_calls"][0]["function"]["name"],
            json!("runner")
        );
        assert_eq!(
            msgs[0]["tool_calls"][0]["function"]["arguments"],
            json!("raw payload")
        );
    }

    #[test]
    fn tool_search_call_maps_to_proxy_name() {
        let msgs = build_messages(json!([{
            "type": "tool_search_call",
            "call_id": "c1",
            "arguments": "{}",
        }]));
        assert_eq!(
            msgs[0]["tool_calls"][0]["function"]["name"],
            json!(crate::context::TOOL_SEARCH_PROXY_NAME)
        );
    }

    #[test]
    fn generic_message_maps_developer_role_to_system() {
        let msgs = build_messages(json!([{
            "type": "message",
            "role": "developer",
            "content": "sys text",
        }]));
        assert_eq!(msgs[0]["role"], json!("system"));
        assert_eq!(msgs[0]["content"], json!("sys text"));
    }

    #[test]
    fn multiple_tool_calls_accumulate_into_one_assistant() {
        let msgs = build_messages(json!([
            { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}" },
            { "type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}" },
        ]));
        assert_eq!(msgs.len(), 1);
        let tcs = msgs[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0]["id"], json!("c1"));
        assert_eq!(tcs[1]["id"], json!("c2"));
    }

    // ----------------------------------------------------------------------- //
    // Top-level request assembly: responses_to_chat
    // ----------------------------------------------------------------------- //

    /// Build a full `ResponsesRequest` from a raw JSON body.
    fn req_from(body: Value) -> ResponsesRequest {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn responses_to_chat_sets_model_messages_and_stream_flag() {
        let payload = req_from(json!({
            "model": "gpt-x",
            "input": [{ "type": "input_text", "text": "hi" }],
        }));
        let chat = responses_to_chat(&payload, "gpt-x");
        assert_eq!(chat.body["model"], json!("gpt-x"));
        assert_eq!(chat.body["stream"], json!(false));
        let msgs = chat.body["messages"].as_array().unwrap();
        assert_eq!(msgs.last().unwrap()["content"], json!("hi"));
    }

    #[test]
    fn responses_to_chat_prepends_instructions_as_system_head() {
        let payload = req_from(json!({
            "model": "m",
            "instructions": "be terse",
            "input": [{ "type": "input_text", "text": "hi" }],
        }));
        let chat = responses_to_chat(&payload, "m");
        let msgs = chat.body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], json!("system"));
        assert_eq!(msgs[0]["content"], json!("be terse"));
    }

    #[test]
    fn responses_to_chat_converts_tools_and_tool_choice() {
        let payload = req_from(json!({
            "model": "m",
            "input": "hi",
            "tools": [{ "type": "function", "name": "f", "parameters": { "type": "object" } }],
            "tool_choice": { "type": "function", "name": "f" },
        }));
        let chat = responses_to_chat(&payload, "m");
        assert_eq!(chat.body["tools"][0]["function"]["name"], json!("f"));
        assert_eq!(
            chat.body["tool_choice"],
            json!({ "type": "function", "function": { "name": "f" } })
        );
    }

    #[test]
    fn responses_to_chat_maps_max_output_tokens_to_max_tokens() {
        let payload = req_from(json!({
            "model": "m",
            "input": "hi",
            "max_output_tokens": 128,
        }));
        let chat = responses_to_chat(&payload, "m");
        assert_eq!(chat.body["max_tokens"], json!(128));
        assert!(chat.body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn responses_to_chat_uses_max_completion_tokens_for_o_series() {
        let payload = req_from(json!({
            "model": "o3-mini",
            "input": "hi",
            "max_output_tokens": 64,
        }));
        let chat = responses_to_chat(&payload, "o3-mini");
        assert_eq!(chat.body["max_completion_tokens"], json!(64));
        assert!(chat.body.get("max_tokens").is_none());
    }

    #[test]
    fn responses_to_chat_passes_sampling_params_through() {
        let payload = req_from(json!({
            "model": "m",
            "input": "hi",
            "temperature": 0.5,
            "top_p": 0.9,
        }));
        let chat = responses_to_chat(&payload, "m");
        assert_eq!(chat.body["temperature"], json!(0.5));
        assert_eq!(chat.body["top_p"], json!(0.9));
    }

    #[test]
    fn responses_to_chat_passes_response_format_through() {
        let payload = req_from(json!({
            "model": "m",
            "input": "hi",
            "response_format": { "type": "json_object" },
        }));
        let chat = responses_to_chat(&payload, "m");
        assert_eq!(
            chat.body["response_format"],
            json!({ "type": "json_object" })
        );
    }

    #[test]
    fn responses_to_chat_passes_extra_passthrough_fields_through() {
        let payload = req_from(json!({
            "model": "m",
            "input": "hi",
            "seed": 42,
            "presence_penalty": 0.3,
        }));
        let chat = responses_to_chat(&payload, "m");
        assert_eq!(chat.body["seed"], json!(42));
        assert_eq!(chat.body["presence_penalty"], json!(0.3));
    }

    #[test]
    fn responses_to_chat_stream_uses_named_stream_options() {
        let payload = req_from(json!({
            "model": "m",
            "input": "hi",
            "stream": true,
            "stream_options": { "include_usage": false },
        }));
        let chat = responses_to_chat(&payload, "m");
        assert_eq!(chat.body["stream"], json!(true));
        assert_eq!(
            chat.body["stream_options"],
            json!({ "include_usage": false })
        );
    }

    #[test]
    fn responses_to_chat_stream_defaults_stream_options() {
        let payload = req_from(json!({
            "model": "m",
            "input": "hi",
            "stream": true,
        }));
        let chat = responses_to_chat(&payload, "m");
        assert_eq!(
            chat.body["stream_options"],
            json!({ "include_usage": true })
        );
    }

    #[test]
    fn responses_to_chat_non_stream_omits_stream_options() {
        let payload = req_from(json!({ "model": "m", "input": "hi" }));
        let chat = responses_to_chat(&payload, "m");
        assert!(chat.body.get("stream_options").is_none());
    }

    #[test]
    fn responses_to_chat_omits_tools_when_empty() {
        let payload = req_from(json!({
            "model": "m",
            "input": "hi",
            "tools": [],
        }));
        let chat = responses_to_chat(&payload, "m");
        assert!(chat.body.get("tools").is_none());
        assert!(chat.body.get("tool_choice").is_none());
    }
}
