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

#[derive(Debug)]
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

    fn status(&self) -> StatusCode {
        match self {
            Self::InvalidRequest { .. } | Self::UnsupportedInputItem { .. } => {
                StatusCode::BAD_REQUEST
            }
            Self::Upstream { status, .. } => *status,
            Self::Stream { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            Self::SessionNotFound { .. } => StatusCode::NOT_FOUND,
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
