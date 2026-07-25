//! Reasoning-content cache for session continuation.
//!
//! Mirrors the Python bridge's `protocol/reasoning_cache.py`. Some upstreams
//! omit `reasoning_content` on assistant messages that carry tool calls when
//! those messages are replayed on a continuation turn. To keep the reasoning
//! attached across turns, a `tool_call_id → reasoning_content` map is extracted
//! when a turn is saved and re-applied to any assistant message that is missing
//! its reasoning on the next turn.
//!
//! Messages are the bridge's `serde_json::Value` chat-message shape rather than
//! a strong struct, so the helpers reach into `role` / `tool_calls` /
//! `reasoning_content` fields directly.

use std::collections::BTreeMap;

use serde_json::Value;

/// Extract the id from a Chat Completions `tool_call` object, preferring `id`
/// then `call_id`, empty when neither is a non-empty string.
pub fn chat_tool_call_id(tool_call: &Value) -> String {
    for key in ["id", "call_id"] {
        if let Some(s) = tool_call.get(key).and_then(Value::as_str) {
            if !s.is_empty() {
                return s.to_owned();
            }
        }
    }
    String::new()
}

/// Build a `tool_call_id → reasoning_content` map from assistant messages that
/// carry both tool calls and non-blank reasoning.
pub fn extract_reasoning_cache(messages: &[Value]) -> BTreeMap<String, String> {
    let mut cache = BTreeMap::new();
    for msg in messages {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        if tool_calls.is_empty() {
            continue;
        }
        let reasoning = msg
            .get("reasoning_content")
            .and_then(Value::as_str)
            .unwrap_or("");
        if reasoning.trim().is_empty() {
            continue;
        }
        for tc in tool_calls {
            let id = chat_tool_call_id(tc);
            if !id.is_empty() {
                cache.insert(id, reasoning.to_owned());
            }
        }
    }
    cache
}

/// Restore cached reasoning into assistant messages missing `reasoning_content`.
///
/// Only messages that carry tool calls and lack non-blank reasoning are
/// touched; the first tool call with a cache hit supplies the reasoning.
pub fn apply_reasoning_cache(messages: &mut [Value], cache: &BTreeMap<String, String>) {
    if cache.is_empty() {
        return;
    }
    for msg in messages.iter_mut() {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let has_tool_calls = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|tc| !tc.is_empty());
        if !has_tool_calls {
            continue;
        }
        let has_reasoning = msg
            .get("reasoning_content")
            .and_then(Value::as_str)
            .is_some_and(|r| !r.trim().is_empty());
        if has_reasoning {
            continue;
        }
        let restored = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .and_then(|tool_calls| {
                tool_calls.iter().find_map(|tc| {
                    let id = chat_tool_call_id(tc);
                    cache.get(&id).cloned()
                })
            });
        if let Some(reasoning) = restored {
            if let Some(obj) = msg.as_object_mut() {
                obj.insert("reasoning_content".to_owned(), Value::from(reasoning));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_tool_call_id_prefers_id_then_call_id() {
        assert_eq!(chat_tool_call_id(&json!({ "id": "a" })), "a");
        assert_eq!(chat_tool_call_id(&json!({ "call_id": "b" })), "b");
        assert_eq!(chat_tool_call_id(&json!({ "id": "", "call_id": "c" })), "c");
        assert_eq!(chat_tool_call_id(&json!({})), "");
    }

    #[test]
    fn extract_maps_tool_call_ids_to_reasoning() {
        let messages = vec![json!({
            "role": "assistant",
            "reasoning_content": "because",
            "tool_calls": [
                { "id": "call_1", "type": "function", "function": { "name": "f", "arguments": "{}" } },
                { "id": "call_2", "type": "function", "function": { "name": "g", "arguments": "{}" } }
            ]
        })];
        let cache = extract_reasoning_cache(&messages);
        assert_eq!(cache.get("call_1"), Some(&"because".to_owned()));
        assert_eq!(cache.get("call_2"), Some(&"because".to_owned()));
    }

    #[test]
    fn extract_skips_blank_reasoning_and_non_tool_messages() {
        let messages = vec![
            json!({ "role": "user", "content": "hi" }),
            json!({
                "role": "assistant",
                "reasoning_content": "   ",
                "tool_calls": [{ "id": "call_1" }]
            }),
            json!({ "role": "assistant", "content": "no tools", "reasoning_content": "x" }),
        ];
        assert!(extract_reasoning_cache(&messages).is_empty());
    }

    #[test]
    fn apply_restores_missing_reasoning_from_first_hit() {
        let mut cache = BTreeMap::new();
        cache.insert("call_1".to_owned(), "recovered".to_owned());
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{ "id": "call_1", "type": "function", "function": { "name": "f", "arguments": "{}" } }]
        })];
        apply_reasoning_cache(&mut messages, &cache);
        assert_eq!(
            messages[0].get("reasoning_content").and_then(Value::as_str),
            Some("recovered")
        );
    }

    #[test]
    fn apply_leaves_existing_reasoning_untouched() {
        let mut cache = BTreeMap::new();
        cache.insert("call_1".to_owned(), "recovered".to_owned());
        let mut messages = vec![json!({
            "role": "assistant",
            "reasoning_content": "original",
            "tool_calls": [{ "id": "call_1" }]
        })];
        apply_reasoning_cache(&mut messages, &cache);
        assert_eq!(
            messages[0].get("reasoning_content").and_then(Value::as_str),
            Some("original")
        );
    }

    #[test]
    fn apply_is_noop_on_empty_cache() {
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{ "id": "call_1" }]
        })];
        apply_reasoning_cache(&mut messages, &BTreeMap::new());
        assert!(messages[0].get("reasoning_content").is_none());
    }
}
