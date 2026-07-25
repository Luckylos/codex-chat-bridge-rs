//! Session resolution and persistence for `previous_response_id` continuation.
//!

use serde_json::{json, Map, Value};

use crate::context::{build_tool_context_from_request, BridgeToolContext};
use crate::error::BridgeError;
use crate::reasoning::extract_reasoning_field;
use crate::reasoning_cache::{apply_reasoning_cache, extract_reasoning_cache};
use crate::session_store::{get_session_store, SessionRecord};
use crate::types::ResponsesRequest;

/// Chat roles the bridge understands. An upstream role outside this set is
/// coerced to `assistant` (with a warning) so a malformed turn still persists
/// as something continuable rather than poisoning the session.
const VALID_ROLES: [&str; 4] = ["system", "user", "assistant", "tool"];

fn coerce_saved_chat_role(role: Option<&str>) -> String {
    match role {
        Some(r) if VALID_ROLES.contains(&r) => r.to_owned(),
        other => {
            tracing::warn!("Upstream returned unexpected role {other:?}; coercing to 'assistant'");
            "assistant".to_owned()
        }
    }
}

/// Extract an assistant message from an upstream Chat Completions body, for the
/// non-streaming path's session persistence. Returns `None` when the choice
/// carries no content, tool calls, refusal, or reasoning — an empty turn is not
/// worth saving.
pub fn assistant_message_from_chat_body(chat_body: &Value) -> Option<Value> {
    let choice = chat_body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())?;
    let message = choice.get("message").and_then(Value::as_object)?;
    if message.is_empty() {
        return None;
    }

    let role = coerce_saved_chat_role(message.get("role").and_then(Value::as_str));
    let content = message.get("content").cloned().unwrap_or(Value::Null);
    let has_content =
        !matches!(content, Value::Null) && !matches!(&content, Value::String(s) if s.is_empty());
    let tool_calls = message.get("tool_calls").and_then(Value::as_array);
    let has_tool_calls = tool_calls.map(|t| !t.is_empty()).unwrap_or(false);
    let refusal = message.get("refusal").and_then(Value::as_str);
    let has_refusal = refusal.map(|r| !r.is_empty()).unwrap_or(false);
    let reasoning = extract_reasoning_field(message);

    if !has_content && !has_tool_calls && !has_refusal && reasoning.is_none() {
        return None;
    }

    let mut out = Map::new();
    out.insert("role".to_owned(), json!(role));
    // A refusal with no content becomes the message body, matching Python.
    if has_content {
        out.insert("content".to_owned(), content);
    } else if let Some(r) = refusal.filter(|_| has_refusal) {
        out.insert("content".to_owned(), json!(format!("[refusal]: {r}")));
    } else {
        out.insert("content".to_owned(), Value::Null);
    }
    if let Some(tc) = tool_calls.filter(|_| has_tool_calls) {
        out.insert("tool_calls".to_owned(), Value::Array(tc.clone()));
    }
    if let Some(rc) = reasoning {
        out.insert("reasoning_content".to_owned(), json!(rc));
    }
    Some(Value::Object(out))
}

/// Resolve `previous_response_id` into `(messages, tool_context, model)`.
///
/// A missing id is a legitimate fresh session → all-`None`. A supplied id that
/// is unknown or expired is a hard 404. On a hit, cached reasoning is restored
/// into the stored assistant messages and the continuation turn's freshly
/// declared tools are merged into the stored context.
#[allow(clippy::type_complexity)]
pub fn resolve_session(
    payload: &ResponsesRequest,
) -> Result<
    (
        Option<Vec<Value>>,
        Option<BridgeToolContext>,
        Option<String>,
    ),
    BridgeError,
> {
    let Some(prev_id) = payload
        .previous_response_id
        .as_deref()
        .filter(|s| !s.is_empty())
    else {
        return Ok((None, None, None));
    };

    let store = get_session_store();
    let Some(mut record) = store.get(prev_id) else {
        return Err(BridgeError::SessionNotFound {
            message: format!("Previous response {prev_id} not found."),
        });
    };

    apply_reasoning_cache(&mut record.messages, &record.reasoning_cache);
    let mut merged_context = record.tool_context.clone();
    merged_context.merge(&build_tool_context_from_request(payload));
    Ok((
        Some(record.messages),
        Some(merged_context),
        Some(record.model),
    ))
}

/// Save a session snapshot for later `previous_response_id` continuation. The
/// assistant message (when present) is appended to the effective request
/// messages, and the reasoning cache is re-extracted from the full turn so a
/// later continuation can restore reasoning the upstream may drop.
pub fn save_session(
    response_id: &str,
    messages: &[Value],
    tool_context: &BridgeToolContext,
    model: &str,
    assistant_message: Option<Value>,
) {
    let mut saved_messages = messages.to_vec();
    if let Some(assistant) = assistant_message {
        saved_messages.push(assistant);
    }
    let cache = extract_reasoning_cache(&saved_messages);
    get_session_store().save(
        response_id,
        SessionRecord::new(
            saved_messages,
            tool_context.clone(),
            model.to_owned(),
            cache,
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assistant_message_from_text_body() {
        let body = json!({
            "choices": [{ "message": { "role": "assistant", "content": "hello" } }]
        });
        let msg = assistant_message_from_chat_body(&body).expect("message");
        assert_eq!(msg["role"], json!("assistant"));
        assert_eq!(msg["content"], json!("hello"));
    }

    #[test]
    fn assistant_message_from_tool_call_body() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{ "id": "call_1", "type": "function",
                        "function": { "name": "f", "arguments": "{}" } }]
                }
            }]
        });
        let msg = assistant_message_from_chat_body(&body).expect("message");
        assert_eq!(msg["tool_calls"][0]["id"], json!("call_1"));
    }

    #[test]
    fn refusal_becomes_content_when_no_text() {
        let body = json!({
            "choices": [{ "message": { "role": "assistant", "refusal": "cannot" } }]
        });
        let msg = assistant_message_from_chat_body(&body).expect("message");
        assert_eq!(msg["content"], json!("[refusal]: cannot"));
    }

    #[test]
    fn empty_message_is_none() {
        let body = json!({ "choices": [{ "message": { "role": "assistant" } }] });
        assert!(assistant_message_from_chat_body(&body).is_none());
    }

    #[test]
    fn unexpected_role_coerced_to_assistant() {
        let body = json!({
            "choices": [{ "message": { "role": "weird", "content": "x" } }]
        });
        let msg = assistant_message_from_chat_body(&body).expect("message");
        assert_eq!(msg["role"], json!("assistant"));
    }

    fn req_from(body: Value) -> ResponsesRequest {
        serde_json::from_value(body).unwrap()
    }

    #[test]
    fn no_previous_id_is_fresh_session() {
        let payload = req_from(json!({ "model": "m" }));
        let (messages, ctx, model) = resolve_session(&payload).expect("ok");
        assert!(messages.is_none() && ctx.is_none() && model.is_none());
    }

    #[test]
    fn unknown_previous_id_is_404() {
        let payload = req_from(json!({ "model": "m", "previous_response_id": "resp_missing" }));
        let err = resolve_session(&payload).expect_err("should 404");
        assert!(matches!(err, BridgeError::SessionNotFound { .. }));
    }
}
