//! Protocol envelopes.
//!
//! Strong types for the request/response *envelopes* the bridge owns; the
//! polymorphic interior (input items, content parts, tool definitions) stays as
//! `serde_json::Value` because the Responses protocol has too many tagged-union
//! shapes for exhaustive structs to pay off — this mirrors the Python bridge's
//! TypedDict-at-the-edges / dict-in-the-middle split.

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub n: Option<i64>,

    /// Every other client-sent field, retained for request-echo and passthrough.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// A single model entry in the `/v1/models` list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default = "default_object")]
    pub object: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

fn default_object() -> String {
    "model".to_owned()
}
