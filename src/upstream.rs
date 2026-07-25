//! Upstream HTTP client.
//!
//! Phase 0 surface: a shared `reqwest::Client` (connection pooling + keep-alive
//! come for free) and `list_models`, used both by `/v1/models` and the startup
//! connectivity probe. The conversion/streaming paths land in later phases.

use std::time::Duration;

use serde_json::Value;

use crate::config::Config;
use crate::error::BridgeError;

/// Thin wrapper over a pooled `reqwest::Client` bound to the bridge config.
#[derive(Clone)]
pub struct Upstream {
    client: reqwest::Client,
    config: Config,
}

impl Upstream {
    pub fn new(config: Config) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(config.upstream_timeout_seconds))
            .build()
            .expect("reqwest client builds with a valid timeout");
        Self { client, config }
    }

    fn auth_headers(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.config.upstream_api_key.is_empty() {
            req
        } else {
            req.bearer_auth(&self.config.upstream_api_key)
        }
    }

    /// Fetch the upstream model catalogue, returning the raw `data` array.
    ///
    /// Mirrors the Python bridge: a bare list or a `{"data": [...]}` envelope
    /// both resolve to the inner list; anything else yields an empty list.
    pub async fn list_models(&self) -> Result<Vec<Value>, BridgeError> {
        let url = self.config.models_url();
        let resp = self
            .auth_headers(self.client.get(&url))
            .send()
            .await
            .map_err(|e| BridgeError::upstream(e.to_string(), "upstream_models_unavailable"))?;

        let resp = resp.error_for_status().map_err(|e| {
            let status = e.status().map(|s| s.as_u16()).unwrap_or(502);
            crate::metrics::record_upstream_error("models", &status.to_string());
            BridgeError::upstream(e.to_string(), "upstream_models_unavailable")
        })?;

        let body: Value = resp
            .json()
            .await
            .map_err(|e| BridgeError::upstream(e.to_string(), "upstream_models_unavailable"))?;

        Ok(match body {
            Value::Array(items) => items,
            Value::Object(mut obj) => match obj.remove("data") {
                Some(Value::Array(items)) => items,
                _ => Vec::new(),
            },
            _ => Vec::new(),
        })
    }
}
