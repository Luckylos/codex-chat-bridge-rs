//! Single source of truth for every client-visible identifier the bridge mints.
//!
//! The Responses protocol exposes several id families that must agree between
//! the non-streaming JSON renderer and the streaming state machine:
//!
//! * the top-level response id (`resp_bridge_<uuid>`), echoed back by clients
//!   as `previous_response_id` and used as the session-store key;
//! * per-turn output item ids derived from the response id — reasoning
//!   (`rs_<response_id>`) and message (`msg_<response_id>`);
//! * per-tool-call output item ids derived from the call id — function
//!   (`fc_<call_id>`) and custom tool (`ctc_<call_id>`);
//! * synthetic tool-call ids (`call_auto_<uuid>`) for payloads that omit one.
//!
//! Centralizing the constructors here makes a single call site own each id
//! shape, so both render paths are structurally incapable of disagreeing.
//! Synthetic ids use `uuid4` so they carry no shared mutable state and are safe
//! under any concurrency model.

use uuid::Uuid;

const RESPONSE_ID_PREFIX: &str = "resp_bridge_";

/// Mint a fresh top-level response id (`resp_bridge_<uuid>`).
///
/// This is the root from which every per-turn output item id is derived, and
/// the key under which the turn is persisted for `previous_response_id`
/// continuation.
pub fn new_response_id() -> String {
    format!("{RESPONSE_ID_PREFIX}{}", short_uuid(12))
}

/// Output item id for the reasoning summary item of a response.
pub fn reasoning_item_id(response_id: &str) -> String {
    format!("rs_{response_id}")
}

/// Output item id for the assistant message item of a response.
///
/// Some OpenAI-compatible validators (including NewAPI's Responses path)
/// require message item ids to begin with `msg` when echoed back via
/// `previous_response_id` continuation.
pub fn message_item_id(response_id: &str) -> String {
    format!("msg_{response_id}")
}

/// Output item id for a `function_call` tool item, derived from its call id.
pub fn function_call_item_id(call_id: &str) -> String {
    format!("fc_{call_id}")
}

/// Output item id for a `custom_tool_call` item, derived from its call id.
/// Emitted by the tool-namespace rendering that lands in Phase 3; the prefix
/// contract is locked in (and tested) now so both render paths agree.
pub fn custom_tool_call_item_id(call_id: &str) -> String {
    format!("ctc_{call_id}")
}

/// Synthetic tool-call id for payloads that omit one.
///
/// Used by both conversion directions when an upstream tool call (or a
/// Responses tool item) arrives without an id.
pub fn synthetic_tool_call_id() -> String {
    format!("call_auto_{}", short_uuid(16))
}

fn short_uuid(len: usize) -> String {
    Uuid::new_v4().simple().to_string()[..len].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_id_carries_prefix_and_fixed_length() {
        let id = new_response_id();
        assert!(id.starts_with(RESPONSE_ID_PREFIX));
        // prefix + 12 hex chars
        assert_eq!(id.len(), RESPONSE_ID_PREFIX.len() + 12);
    }

    #[test]
    fn response_ids_are_unique_per_call() {
        assert_ne!(new_response_id(), new_response_id());
    }

    #[test]
    fn item_ids_derive_from_response_id() {
        let rid = "resp_bridge_abc123";
        assert_eq!(reasoning_item_id(rid), "rs_resp_bridge_abc123");
        assert_eq!(message_item_id(rid), "msg_resp_bridge_abc123");
    }

    #[test]
    fn tool_item_ids_derive_from_call_id() {
        assert_eq!(function_call_item_id("call_1"), "fc_call_1");
        assert_eq!(custom_tool_call_item_id("call_1"), "ctc_call_1");
    }

    #[test]
    fn synthetic_call_id_is_prefixed_and_unique() {
        let a = synthetic_tool_call_id();
        assert!(a.starts_with("call_auto_"));
        assert_eq!(a.len(), "call_auto_".len() + 16);
        assert_ne!(a, synthetic_tool_call_id());
    }
}
