//! Protocol envelopes.
//!
//! Strong types for the request/response *envelopes* the bridge owns; the
//! polymorphic interior (input items, content parts, tool definitions) stays as
//! `serde_json::Value` because the Responses protocol has too many tagged-union
//! shapes for exhaustive structs to pay off — this mirrors the Python bridge's
//! TypedDict-at-the-edges / dict-in-the-middle split.

use serde::Deserialize;
use serde_json::{Map, Value};

/// Inbound `/v1/responses` request. Only the fields the bridge inspects are
/// named; everything else the client sends is preserved via `extra` so wire
/// passthrough stays lossless.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub instructions: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<Value>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    // KNOWN PORTING GAP (not dead field): the Responses structured-output `text`
    // config. The Python bridge reads `text.format` and injects it as the
    // outbound Chat `response_format` (`request._response_format_from_payload`),
    // so a client requesting structured output actually gets it. The Rust port
    // deserializes this field but does NOT yet consume it — retained, not
    // deleted, because dropping it would silently regress a live Python
    // capability. Wiring it into the request builder is a deferred feature, out
    // of scope for the behavior-neutral refactor. Currently 0 hits in 24h of
    // production traffic, which is why it read as "never used".
    #[serde(default)]
    #[allow(dead_code)]
    pub text: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub max_output_tokens: Option<i64>,
    #[serde(default)]
    pub temperature: Option<Value>,
    #[serde(default)]
    pub top_p: Option<Value>,
    #[serde(default)]
    pub n: Option<i64>,

    /// Every other client-sent field, retained for request-echo and passthrough.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ResponsesRequest {
    /// Re-serialize the original request into a plain map for request-echo.
    /// Mirrors Python's `model_dump(exclude_none=True, exclude_defaults=True)`
    /// closely enough for the echo fields, which are all explicitly named.
    pub fn to_echo_map(&self) -> Map<String, Value> {
        let mut map = Map::new();
        if let Some(v) = &self.instructions {
            map.insert("instructions".into(), v.clone());
        }
        if let Some(v) = self.max_output_tokens {
            map.insert("max_output_tokens".into(), Value::from(v));
        }
        if let Some(v) = &self.tools {
            map.insert("tools".into(), v.clone());
        }
        if let Some(v) = &self.tool_choice {
            map.insert("tool_choice".into(), v.clone());
        }
        if let Some(v) = &self.reasoning {
            map.insert("reasoning".into(), v.clone());
        }
        if let Some(v) = &self.temperature {
            map.insert("temperature".into(), v.clone());
        }
        if let Some(v) = &self.top_p {
            map.insert("top_p".into(), v.clone());
        }
        if let Some(v) = &self.previous_response_id {
            map.insert("previous_response_id".into(), Value::from(v.clone()));
        }
        for key in ["parallel_tool_calls", "metadata"] {
            if let Some(v) = self.extra.get(key) {
                map.insert(key.into(), v.clone());
            }
        }
        map
    }
}

/// Outbound Chat Completions request built by the conversion pipeline.
///
/// The bridge owns no strong Chat schema — the body is assembled field-by-field
/// as a `serde_json::Map` and serialized as-is to the upstream. Keeping it a
/// thin newtype (rather than a bare `Map`) gives the conversion and upstream
/// layers a named type to pass around rather than a bare map.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub body: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req_from(body: Value) -> ResponsesRequest {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn echo_map_includes_only_present_named_fields() {
        let req = req_from(json!({
            "model": "m",
            "instructions": "be terse",
            "temperature": 0.5,
        }));
        let echo = req.to_echo_map();
        assert_eq!(echo["instructions"], json!("be terse"));
        assert_eq!(echo["temperature"], json!(0.5));
        // Absent named fields do not appear.
        assert!(echo.get("tools").is_none());
        assert!(echo.get("top_p").is_none());
        assert!(echo.get("max_output_tokens").is_none());
    }

    #[test]
    fn echo_map_carries_tools_tool_choice_and_reasoning() {
        let req = req_from(json!({
            "model": "m",
            "tools": [{ "type": "function", "name": "f" }],
            "tool_choice": { "type": "function", "name": "f" },
            "reasoning": { "effort": "high" },
            "max_output_tokens": 128,
        }));
        let echo = req.to_echo_map();
        assert_eq!(echo["tools"][0]["name"], json!("f"));
        assert_eq!(echo["tool_choice"]["type"], json!("function"));
        assert_eq!(echo["reasoning"]["effort"], json!("high"));
        assert_eq!(echo["max_output_tokens"], json!(128));
    }

    #[test]
    fn echo_map_lifts_previous_response_id_and_extra_passthrough() {
        let req = req_from(json!({
            "model": "m",
            "previous_response_id": "resp_bridge_abc",
            "parallel_tool_calls": true,
            "metadata": { "trace": "x" },
        }));
        let echo = req.to_echo_map();
        assert_eq!(echo["previous_response_id"], json!("resp_bridge_abc"));
        assert_eq!(echo["parallel_tool_calls"], json!(true));
        assert_eq!(echo["metadata"], json!({ "trace": "x" }));
    }

    #[test]
    fn echo_map_empty_when_only_model_present() {
        let req = req_from(json!({ "model": "m" }));
        assert!(req.to_echo_map().is_empty());
    }
}
