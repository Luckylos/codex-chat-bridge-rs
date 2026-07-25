//! Tool-argument canonicalization and call-id resolution.

use serde_json::{Map, Value};

use crate::id_gen;

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
pub(crate) fn custom_tool_input_to_chat_arguments(input: Option<&Value>) -> String {
    match input {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => canonical_json_string(other),
    }
}
/// Deterministic JSON serialization with sorted keys (serde_json sorts object
/// keys when the `preserve_order` feature is off, which is the default).
pub(crate) fn canonical_json_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "{}".to_owned())
}
pub(crate) fn resolve_tool_call_id(obj: &Map<String, Value>) -> String {
    obj.get("call_id")
        .and_then(Value::as_str)
        .or_else(|| obj.get("id").and_then(Value::as_str))
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(id_gen::synthetic_tool_call_id)
}
