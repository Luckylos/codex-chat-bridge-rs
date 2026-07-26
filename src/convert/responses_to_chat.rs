//! Responses request -> upstream Chat Completions request.

use serde_json::{json, Map, Value};

use crate::protocol::{InputItemKind, ToolCallKind};
use crate::reasoning;
use crate::transform_loss::{TransformLoss, TransformLossCollector};
use crate::types::{ChatRequest, ResponsesRequest};

use super::message_normalization::{
    append_reasoning_to_last_assistant, chat_message_content_from_response_content,
    collapse_system_messages_to_head, instruction_text, non_empty, normalize_tool_output_content,
    reasoning_item_text, sanitize_messages,
};
use super::tool_arguments::{
    canonicalize_tool_arguments, custom_tool_input_to_chat_arguments, resolve_tool_call_id,
};

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
/// turn's instructions system message, then this turn's input items.
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
    let items = crate::context::iter_request_input_items(payload.input.as_ref());
    let mut pending_tool_calls: Vec<Value> = Vec::new();
    let mut pending_reasoning: Option<String> = None;
    // Call ids already present in the (possibly session-restored) history, so a
    // continuation turn skips tool calls/outputs it already carries.
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

        // Classify the item's `type` tag into a closed set so this dispatch is
        // exhaustive (a typo can't compile; a new protocol variant forces a
        // decision here). The payload stays `Value` — the reads below are the
        // same lenient, byte-exact accesses as before.
        match InputItemKind::classify(item_type) {
            // Reasoning items.
            InputItemKind::Reasoning => {
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
                if pending_tool_calls.is_empty()
                    && append_reasoning_to_last_assistant(messages, text)
                {
                    // backfilled onto the preceding assistant turn
                } else {
                    pending_reasoning = Some(match pending_reasoning.take() {
                        Some(prev) => format!("{prev}\n\n{text}"),
                        None => text.to_owned(),
                    });
                }
            }

            // Text-like content → user message.
            InputItemKind::Text => {
                flush!();
                let text = obj.get("text").and_then(Value::as_str).unwrap_or("");
                messages.push(ChatMessageBuilder::with_content("user", json!(text)).into_value());
            }

            // Media items (input_image / input_audio) → user message content part.
            InputItemKind::Image | InputItemKind::Audio => {
                flush!();
                handle_media_item(obj, item_type, messages, loss);
            }

            // Tool call items → accumulate.
            InputItemKind::FunctionCall
            | InputItemKind::CustomToolCall
            | InputItemKind::ToolSearchCall => {
                let kind = InputItemKind::classify(item_type)
                    .as_tool_call()
                    .expect("tool-call kinds map to a ToolCallKind");
                handle_tool_call_item(
                    obj,
                    kind,
                    &mut pending_tool_calls,
                    &mut pending_reasoning,
                    &skip_call_ids,
                    loss,
                    tool_context,
                );
            }

            // Tool output items → tool message (or orphan downgrade to user).
            InputItemKind::FunctionCallOutput
            | InputItemKind::CustomToolCallOutput
            | InputItemKind::ToolSearchOutput => {
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
                    // A tool message with no preceding assistant tool_call would
                    // be rejected by Chat Completions; downgrade to a user
                    // message.
                    loss.record(
                        TransformLoss::DowngradedOrphanToolOutput,
                        non_empty(item_type),
                        format!("Tool output {call_id} has no matching preceding tool call"),
                    );
                    let text = format!("Function call output ({call_id}): {content}");
                    messages
                        .push(ChatMessageBuilder::with_content("user", json!(text)).into_value());
                }
            }

            // Explicit `message` items, plus any unrecognized item that still
            // carries a role/content payload, become a generic message. An
            // unrecognized item with neither is skipped permissively. This
            // preserves the pre-refactor fallthrough exactly.
            InputItemKind::Message | InputItemKind::Other => {
                if item_type == "message" || obj.contains_key("role") || obj.contains_key("content")
                {
                    flush!();
                    messages.push(build_generic_message(obj, item_type, loss));
                } else {
                    flush!();
                    loss.record(
                        TransformLoss::SkippedUnknownItemType,
                        non_empty(item_type),
                        format!("Unrecognized input item type: {item_type:?}"),
                    );
                }
            }
        }
    }

    flush_pending(messages, &mut pending_tool_calls, &mut pending_reasoning);
}
/// Collect call ids already present in `messages`, from both assistant
/// `tool_calls[].id`/`call_id` and tool-role `tool_call_id`.
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
fn should_skip(obj: &Map<String, Value>, skip_ids: &std::collections::HashSet<String>) -> bool {
    obj.get("call_id")
        .and_then(Value::as_str)
        .or_else(|| obj.get("id").and_then(Value::as_str))
        .is_some_and(|id| skip_ids.contains(id))
}
/// Whether `call_id` has a matching assistant `tool_calls[].id` in the flushed
/// message history.
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
/// a transform-loss event when the URL/format is rejected.
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
/// warning log summarizing the events.
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
    kind: ToolCallKind,
    pending_tool_calls: &mut Vec<Value>,
    pending_reasoning: &mut Option<String>,
    skip_call_ids: &std::collections::HashSet<String>,
    loss: &mut TransformLossCollector,
    tool_context: &crate::context::BridgeToolContext,
) {
    let (name, arguments): (String, String) = match kind {
        ToolCallKind::Function => {
            // A namespaced function call is flattened back to the Chat name the
            // upstream saw, so a continuation turn's tool_calls[].name matches
            // the tool schema.
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
        ToolCallKind::Custom => (
            obj.get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown_tool")
                .to_owned(),
            // custom tool input → chat arguments (string passthrough in Phase 1).
            custom_tool_input_to_chat_arguments(obj.get("input")),
        ),
        ToolCallKind::Search => (
            crate::context::TOOL_SEARCH_PROXY_NAME.to_owned(),
            canonicalize_tool_arguments(obj.get("arguments")),
        ),
    };

    // A tool call whose id already appears in the session history is a
    // continuation duplicate; skip it.
    if should_skip(obj, skip_call_ids) {
        loss.record(
            TransformLoss::SkippedDuplicateToolCall,
            non_empty(kind.as_str()),
            "Duplicate tool call already in message history",
        );
        return;
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
fn is_openai_o_series(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    // o1 / o3 / o4 families use max_completion_tokens.
    m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4")
}
/// Convert a Responses `tool_choice` into the Chat Completions form. String
/// modes (`auto`/`none`/`required`) pass through; an explicit tool object is
/// rewritten to `{type:function, function:{name}}`.
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
                    // upstream schema declares.
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
    #[test]
    fn iter_request_input_items_string_wraps_as_input_text() {
        let payload = req_with_input(json!("hi"));
        let items = crate::context::iter_request_input_items(payload.input.as_ref());
        assert_eq!(items, vec![json!({ "type": "input_text", "text": "hi" })]);
    }
    #[test]
    fn iter_request_input_items_array_lifts_bare_strings() {
        let payload = req_with_input(json!(["a", { "type": "message", "role": "user" }]));
        let items = crate::context::iter_request_input_items(payload.input.as_ref());
        assert_eq!(items[0], json!({ "type": "input_text", "text": "a" }));
        assert_eq!(items[1], json!({ "type": "message", "role": "user" }));
    }
    #[test]
    fn iter_request_input_items_none_is_empty() {
        let payload = req_with_input(Value::Null);
        assert!(crate::context::iter_request_input_items(payload.input.as_ref()).is_empty());
    }
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
