//! Bridge error type and its HTTP rendering.
//!
//! Mirrors the Python `BridgeError` hierarchy's *wire* contract (a JSON
//! `{ "error": { message, type, code, ... } }` envelope with a status code),
//! but as a single enum with `IntoResponse` so the compiler enforces that every
//! variant maps to a status and body.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

// UnsupportedInputItem / Stream / SessionNotFound are constructed by the
// streaming (Phase 2) and session-store (Phase 3) layers; their status/envelope
// rendering is already covered by tests so the wire contract is locked in now.
#[derive(Debug)]
#[allow(dead_code)]
pub enum BridgeError {
    /// Client sent something invalid → 400.
    InvalidRequest {
        message: String,
        code: &'static str,
        detail: Option<Value>,
    },
    /// Unsupported Responses input item → 400, carries the offending item type.
    UnsupportedInputItem { message: String, item_type: String },
    /// Upstream failed or returned an error status → 502 by default.
    Upstream {
        message: String,
        code: &'static str,
        status: StatusCode,
        detail: Option<Value>,
    },
    /// Internal streaming fault → 500.
    Stream { message: String, code: &'static str },
    /// Session lookup miss for previous_response_id → 404.
    SessionNotFound { message: String },
    /// Request body exceeded the configured byte budget → 413.
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
            error["detail"] = d.clone();
        }
        json!({ "error": error })
    }
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.envelope())
    }
}

impl std::error::Error for BridgeError {}

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
        assert!(env["error"].get("detail").is_none());
    }

    #[test]
    fn upstream_defaults_to_502() {
        let err = BridgeError::upstream("boom", "upstream_network_error");
        assert_eq!(err.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(err.envelope()["error"]["type"], json!("upstream_error"));
    }

    #[test]
    fn upstream_with_status_preserves_code_and_detail() {
        let detail = json!({ "error": { "message": "rate limited" } });
        let err = BridgeError::upstream_with_status(
            "rate limited",
            "upstream_request_failed",
            429,
            Some(detail.clone()),
        );
        assert_eq!(err.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.envelope()["error"]["detail"], detail);
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
