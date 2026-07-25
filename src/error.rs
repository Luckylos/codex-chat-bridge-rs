//! Bridge error type and its HTTP rendering.
//!

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

// INTENTIONAL BEHAVIOR DIFFERENCE (not dead variants): `UnsupportedInputItem`
// and `Stream` are never *constructed* by the Rust port, and that is by design,
// not an oversight:
//   * Unsupported hosted tools — the Python bridge raises `UnsupportedInputItem`
//     under the reject/error policy; the Rust request path has no per-tool error
//     channel at build time and deliberately drops-with-warning instead
//     (see `context.rs` add_response_tool). Production default policy is Ignore,
//     so this path never triggers a client-visible error either way.
//   * Streaming faults — the Rust stream path surfaces failures as a terminal
//     `response.failed` SSE event via `envelope.failed_event`, NOT as a thrown
//     `Stream` error (a stream that already sent 200 + headers cannot switch to
//     an HTTP error body).
// Both variants are retained because their status/envelope *rendering* is the
// wire contract and is pinned by tests — keeping them documents the full error
// surface and lets the rendering stay proven. `#[allow(dead_code)]` is scoped to
// the two variants, with this justification, rather than a blanket module allow.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// Client sent something invalid → 400.
    #[error("{message}")]
    InvalidRequest {
        message: String,
        code: &'static str,
        detail: Option<Value>,
    },
    /// Unsupported Responses input item → 400, carries the offending item type.
    /// Never constructed (see module note): the request path drops-with-warning.
    #[allow(dead_code)]
    #[error("{message}")]
    UnsupportedInputItem { message: String, item_type: String },
    /// Upstream failed or returned an error status → 502 by default.
    #[error("{message}")]
    Upstream {
        message: String,
        code: &'static str,
        status: StatusCode,
        detail: Option<Value>,
    },
    /// Internal streaming fault → 500. Never constructed (see module note):
    /// stream faults emit a terminal `response.failed` event instead.
    #[allow(dead_code)]
    #[error("{message}")]
    Stream { message: String, code: &'static str },
    /// Session lookup miss for previous_response_id → 404.
    #[error("{message}")]
    SessionNotFound { message: String },
    /// Request body exceeded the configured byte budget → 413.
    #[error("{message}")]
    RequestBodyTooLarge { message: String },
}

impl BridgeError {
    pub fn invalid_request(message: impl Into<String>, code: &'static str) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            code,
            detail: None,
        }
    }

    pub fn upstream(message: impl Into<String>, code: &'static str) -> Self {
        Self::Upstream {
            message: message.into(),
            code,
            status: StatusCode::BAD_GATEWAY,
            detail: None,
        }
    }

    /// Upstream error carrying the upstream HTTP status and parsed error detail,
    /// so a terminal non-retryable upstream failure surfaces the original status
    /// and body to the client rather than a flat 502.
    pub fn upstream_with_status(
        message: impl Into<String>,
        code: &'static str,
        status: u16,
        detail: Option<Value>,
    ) -> Self {
        Self::Upstream {
            message: message.into(),
            code,
            status: StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
            detail,
        }
    }

    pub fn request_body_too_large(max_body_bytes: usize) -> Self {
        Self::RequestBodyTooLarge {
            message: format!("Request body too large: exceeds max {max_body_bytes} bytes"),
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::InvalidRequest { .. } | Self::UnsupportedInputItem { .. } => {
                StatusCode::BAD_REQUEST
            }
            Self::Upstream { status, .. } => *status,
            Self::Stream { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::SessionNotFound { .. } => StatusCode::NOT_FOUND,
            Self::RequestBodyTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
        }
    }

    fn error_type(&self) -> &'static str {
        match self {
            Self::InvalidRequest { .. } | Self::UnsupportedInputItem { .. } => {
                "invalid_request_error"
            }
            Self::Upstream { .. } => "upstream_error",
            Self::Stream { .. } => "stream_error",
            Self::SessionNotFound { .. } => "invalid_request_error",
            Self::RequestBodyTooLarge { .. } => "invalid_request_error",
        }
    }

    /// The `error` envelope body, matching the Python bridge's JSON shape.
    pub fn envelope(&self) -> Value {
        let (message, code): (&str, &str) = match self {
            Self::InvalidRequest { message, code, .. } => (message, code),
            Self::UnsupportedInputItem { message, .. } => (message, "unsupported_input_item"),
            Self::Upstream { message, code, .. } => (message, code),
            Self::Stream { message, code } => (message, code),
            Self::SessionNotFound { message } => (message, "previous_response_not_found"),
            Self::RequestBodyTooLarge { message } => (message, "request_too_large"),
        };
        let mut error = json!({
            "message": message,
            "type": self.error_type(),
            "code": code,
        });
        if let Self::UnsupportedInputItem { item_type, .. } = self {
            error["item_type"] = json!(item_type);
        }
        if let Self::InvalidRequest {
            detail: Some(d), ..
        }
        | Self::Upstream {
            detail: Some(d), ..
        } = self
        {
            let param = match d {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
            };
            error["param"] = json!(param);
        }
        json!({ "error": error })
    }
}

impl IntoResponse for BridgeError {
    fn into_response(self) -> Response {
        (self.status(), Json(self.envelope())).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_request_renders_400_envelope() {
        let err = BridgeError::invalid_request("bad n", "n_not_supported");
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        let env = err.envelope();
        assert_eq!(env["error"]["message"], json!("bad n"));
        assert_eq!(env["error"]["type"], json!("invalid_request_error"));
        assert_eq!(env["error"]["code"], json!("n_not_supported"));
        assert!(env["error"].get("param").is_none());
    }

    #[test]
    fn upstream_defaults_to_502() {
        let err = BridgeError::upstream("boom", "upstream_network_error");
        assert_eq!(err.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(err.envelope()["error"]["type"], json!("upstream_error"));
    }

    #[test]
    fn upstream_with_status_preserves_code_and_param() {
        let detail = json!({ "error": { "message": "rate limited" } });
        let err = BridgeError::upstream_with_status(
            "rate limited",
            "upstream_request_failed",
            429,
            Some(detail.clone()),
        );
        assert_eq!(err.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            err.envelope()["error"]["param"],
            json!(r#"{"error":{"message":"rate limited"}}"#)
        );
    }

    #[test]
    fn upstream_string_detail_passes_through_as_param() {
        let err = BridgeError::upstream_with_status(
            "boom",
            "upstream_request_failed",
            500,
            Some(json!("raw upstream text")),
        );
        assert_eq!(err.envelope()["error"]["param"], json!("raw upstream text"));
    }

    #[test]
    fn upstream_with_invalid_status_falls_back_to_502() {
        let err = BridgeError::upstream_with_status("x", "c", 0, None);
        assert_eq!(err.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn unsupported_input_item_carries_item_type() {
        let err = BridgeError::UnsupportedInputItem {
            message: "nope".to_owned(),
            item_type: "mystery".to_owned(),
        };
        assert_eq!(err.status(), StatusCode::BAD_REQUEST);
        let env = err.envelope();
        assert_eq!(env["error"]["item_type"], json!("mystery"));
        assert_eq!(env["error"]["code"], json!("unsupported_input_item"));
    }

    #[test]
    fn session_not_found_renders_404() {
        let err = BridgeError::SessionNotFound {
            message: "no such response".to_owned(),
        };
        assert_eq!(err.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            err.envelope()["error"]["code"],
            json!("previous_response_not_found")
        );
    }
}
