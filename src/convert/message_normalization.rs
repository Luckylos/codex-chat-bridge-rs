//! Message content normalization: text flattening, sanitization, adjacent
//! system-message collapsing, and reasoning backfill.

use serde_json::{json, Map, Value};

use crate::protocol::ContentPartType;
use crate::sanitize::sanitize_string;

use super::tool_arguments::canonical_json_string;

/// The item `type` as a metric label, or `None` when the field was absent/empty
/// (which the metric renders as `"none"`).
pub(crate) fn non_empty(item_type: &str) -> Option<&str> {
    (!item_type.is_empty()).then_some(item_type)
}
pub(crate) fn extract_message_annotations(message: &Value) -> Vec<Value> {
    message
        .get("annotations")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter(|a| a.is_object()).cloned().collect())
        .unwrap_or_default()
}
pub(crate) fn instruction_text(value: &Value) -> String {
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
                        if ContentPartType::classify(typ).is_text() {
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
pub(crate) fn reasoning_item_text(item: &Map<String, Value>) -> String {
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
                        let part_type = ContentPartType::classify(typ);
                        if part_type.is_text() {
                            if let Some(text) = o.get("text").and_then(Value::as_str) {
                                parts
                                    .push(json!({ "type": "text", "text": sanitize_string(text) }));
                            }
                        } else if part_type == ContentPartType::Refusal {
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
pub(crate) fn normalize_tool_output_content(value: Option<&Value>) -> String {
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
            if ContentPartType::classify(typ).is_text() {
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
pub(crate) fn sanitize_messages(messages: Vec<Value>) -> Vec<Value> {
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
pub(crate) fn collapse_system_messages_to_head(messages: Vec<Value>) -> Vec<Value> {
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
pub(crate) fn append_reasoning_to_last_assistant(messages: &mut [Value], text: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
