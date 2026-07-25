//! Upstream Chat Completions body -> Responses object.

use serde_json::{json, Map, Value};

use crate::id_gen;
use crate::protocol::ContentPartType;
use crate::reasoning;
use crate::sanitize::sanitize_string;

use super::message_normalization::extract_message_annotations;
use super::semantics::{
    echo_request_fields, incomplete_reason_from_finish_reason, map_chat_usage,
    response_status_from_finish_reason, REQUEST_ECHO_FIELDS,
};
use super::tool_arguments::canonicalize_tool_arguments;

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
                // Note: this site matches `text | output_text` only (NOT
                // `input_text`), so it classifies explicitly rather than using
                // `is_text()`.
                match ContentPartType::classify(ptype) {
                    ContentPartType::Text | ContentPartType::OutputText => {
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
                    }
                    ContentPartType::Refusal => {
                        if let Some(refusal) = p.get("refusal").and_then(Value::as_str) {
                            if !refusal.is_empty() {
                                parts.push(json!({
                                    "type": "refusal",
                                    "refusal": sanitize_string(refusal),
                                }));
                            }
                        }
                    }
                    ContentPartType::InputText | ContentPartType::Other => {}
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
