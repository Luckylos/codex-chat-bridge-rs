//! Bidirectional Responses <-> Chat Completions conversion.
//!
//! Split by data-flow direction, restoring the domain structure of the Python
//! original: [`responses_to_chat`] (request side) and [`chat_to_responses`]
//! (response side), over shared [`semantics`], [`message_normalization`], and
//! [`tool_arguments`] helpers. The polymorphic interior stays
//! `serde_json::Value`; only the request/response envelopes are strongly typed.

mod chat_to_responses;
mod message_normalization;
mod responses_to_chat;
mod semantics;
mod tool_arguments;

pub use chat_to_responses::chat_to_responses;
pub use responses_to_chat::responses_to_chat_with_session;

pub(crate) use message_normalization::chat_message_content_from_response_content;
pub(crate) use semantics::{
    chat_finish_reason, incomplete_reason_from_finish_reason, map_chat_usage,
    response_status_from_finish_reason, should_persist_finish_reason,
    should_persist_response_status, ResponseStatus, REQUEST_ECHO_FIELDS,
};
pub(crate) use tool_arguments::canonicalize_tool_arguments;
