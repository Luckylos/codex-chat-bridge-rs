//! Upstream HTTP client.
//!
//! A shared `reqwest::Client` (connection pooling + keep-alive come for free)
//! plus the two request shapes the bridge needs:
//!
//! * [`Upstream::list_models`] — control-plane catalogue fetch, also used by the
//!   startup connectivity probe.
//! * [`Upstream::create_chat_completion`] — the non-streaming hot path. Wraps
//!   the outbound body in two retry layers matching the Python bridge:
//!   an inner **compat** layer that rewrites/strips fields a specific upstream
//!   rejects (400 + narrow 5xx) and an outer **transport** layer that retries
//!   429/5xx and network faults with exponential backoff + jitter.
//!
//! The streaming path lands in a later phase.

use std::time::Duration;

use futures::StreamExt;
use rand::Rng;
use serde_json::{Map, Value};

use crate::compat::{self, ReasoningState};
use crate::config::Config;
use crate::error::BridgeError;

/// Statuses the compat layer inspects for a possible body rewrite.
const COMPAT_STATUSES: [u16; 3] = [400, 500, 503];
/// Statuses the transport layer retries with backoff.
const RETRYABLE_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

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

    /// Whether the upstream is configured to serve streaming responses.
    pub fn upstream_streaming(&self) -> bool {
        self.config.upstream_streaming
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
    pub async fn list_models(&self) -> Result<Vec<Value>, BridgeError> {
        let url = self.config.models_url();
        let resp = self
            .auth_headers(self.client.get(&url))
            .send()
            .await
            .map_err(|e| BridgeError::upstream(e.to_string(), "upstream_models_unavailable"))?;

        let resp = resp.error_for_status().map_err(|e| {
            let status = e.status().map_or(502, |s| s.as_u16());
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

    /// POST a Chat Completions body and return the parsed JSON response.
    ///
    /// Runs the compat + transport retry stack. `body` is the raw chat request
    /// assembled by the conversion layer; it is seeded into the compat state
    /// machine which may rewrite it across attempts.
    pub async fn create_chat_completion(
        &self,
        body: Map<String, Value>,
    ) -> Result<Value, BridgeError> {
        let resp = self.send_with_retry(body, false).await?;
        resp.json()
            .await
            .map_err(|e| BridgeError::upstream(e.to_string(), "upstream_bad_response"))
    }

    /// POST a Chat Completions body with `stream: true` and return the upstream
    /// SSE body as a byte stream. Shares the compat + transport retry stack with
    /// [`Upstream::create_chat_completion`]; retries happen before the first
    /// byte, after which the live stream is handed to the caller untouched.
    pub async fn create_chat_completion_stream(
        &self,
        body: Map<String, Value>,
    ) -> Result<impl futures::Stream<Item = Vec<u8>>, BridgeError> {
        let resp = self.send_with_retry(body, true).await?;
        let byte_stream = resp.bytes_stream();
        Ok(async_stream::stream! {
            futures::pin_mut!(byte_stream);
            while let Some(item) = byte_stream.next().await {
                match item {
                    Ok(bytes) if !bytes.is_empty() => yield bytes.to_vec(),
                    Ok(_) => {}
                    Err(e) => {
                        // A mid-stream transport fault cannot be retried once
                        // bytes have been relayed; surface it as an SSE error
                        // frame so the consumer finalizes as failed.
                        let err = serde_json::json!({
                            "error": {
                                "message": format!("Upstream stream error: {e}"),
                                "type": "upstream_stream_error",
                            }
                        });
                        yield crate::sse::serialize_event(Some("error"), &err);
                        break;
                    }
                }
            }
        })
    }

    /// Forward a raw Chat Completions body verbatim, returning the upstream
    /// response with its body unread.
    ///
    /// Transparent relay path for native `/v1/chat/completions` clients: no
    /// reasoning-policy rewrite, no compat body mutation, no response
    /// transformation. Unlike [`Upstream::create_chat_completion`], upstream
    /// 4xx/5xx bodies are returned as-is — the caller owns the request body, so
    /// the bridge does not second-guess client parameters. Only network faults
    /// and retryable statuses (429/5xx) trigger a retry, and the body is never
    /// mutated between attempts.
    pub async fn relay_chat_completion(
        &self,
        body: Map<String, Value>,
    ) -> Result<reqwest::Response, BridgeError> {
        let is_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        // Records the whole relay retry loop's duration on any exit path,
        let _phase = crate::metrics::PhaseTimer::start(
            crate::metrics::PhaseKind::Facade,
            "relay_retry",
            is_stream,
        );
        let url = self.config.chat_completions_url();
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let max_retries = self.config.upstream_max_retries;

        for attempt in 0..=max_retries {
            let send_result = {
                let _send = crate::metrics::PhaseTimer::start(
                    crate::metrics::PhaseKind::Transport,
                    "send",
                    is_stream,
                );
                self.auth_headers(self.client.post(&url))
                    .json(&body)
                    .send()
                    .await
            };
            match send_result {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    // Retryable status with attempts left → backoff and retry.
                    // Every other status (success or client/permanent error) is
                    // relayed to the caller verbatim.
                    if RETRYABLE_STATUSES.contains(&status) && attempt < max_retries {
                        self.backoff(attempt).await;
                        continue;
                    }
                    return Ok(resp);
                }
                Err(net) => {
                    if attempt < max_retries {
                        tracing::warn!(
                            "relay network error (attempt {}/{}): {net}",
                            attempt + 1,
                            max_retries + 1
                        );
                        self.backoff(attempt).await;
                        continue;
                    }
                    crate::metrics::record_upstream_error(&model, "network");
                    return Err(BridgeError::upstream(
                        format!("Upstream relay failed: {net}"),
                        "upstream_relay_failed",
                    ));
                }
            }
        }

        Err(BridgeError::upstream(
            "relay retry loop exhausted without a conclusive result",
            "retry_exhausted",
        ))
    }

    /// The shared compat + transport retry loop, returning the successful
    /// upstream response (body unread) or a terminal [`BridgeError`].
    async fn send_with_retry(
        &self,
        body: Map<String, Value>,
        is_stream: bool,
    ) -> Result<reqwest::Response, BridgeError> {
        // Records the whole transport retry loop's duration on any exit path,
        let _phase = crate::metrics::PhaseTimer::start(
            crate::metrics::PhaseKind::Facade,
            "request_retry",
            is_stream,
        );
        let url = self.config.chat_completions_url();
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();

        let max_retries = self.config.upstream_max_retries;
        let mut state = compat::initial_state(body);

        for attempt in 0..=max_retries {
            match self.send_with_compat(&url, &mut state, is_stream).await {
                Ok(SendOutcome::Success(resp)) => return Ok(resp),
                Ok(SendOutcome::TerminalError { status, detail }) => {
                    // Retryable transport status with attempts left → backoff.
                    if RETRYABLE_STATUSES.contains(&status) && attempt < max_retries {
                        self.backoff(attempt).await;
                        continue;
                    }
                    crate::metrics::record_upstream_error(&model, &status.to_string());
                    return Err(BridgeError::upstream_with_status(
                        upstream_error_message(&detail, status),
                        "upstream_request_failed",
                        status,
                        Some(detail),
                    ));
                }
                Err(net) => {
                    // Network fault: retry with backoff if attempts remain.
                    if attempt < max_retries {
                        tracing::warn!(
                            "upstream network error (attempt {}/{}): {net}",
                            attempt + 1,
                            max_retries + 1
                        );
                        self.backoff(attempt).await;
                        continue;
                    }
                    crate::metrics::record_upstream_error(&model, "network");
                    return Err(BridgeError::upstream(net, "upstream_network_error"));
                }
            }
        }

        Err(BridgeError::upstream(
            "retry loop exhausted without a conclusive result",
            "retry_exhausted",
        ))
    }

    /// One transport attempt, itself looping over compat rewrites. Returns a
    /// [`SendOutcome`] for HTTP-level results or `Err(String)` for a network
    /// fault the caller may retry.
    async fn send_with_compat(
        &self,
        url: &str,
        state: &mut ReasoningState,
        is_stream: bool,
    ) -> Result<SendOutcome, String> {
        // The whole compat rewrite loop is one `compat_cycle` phase, matching
        // the Python `send_with_compat_retry` timing.
        let _cycle = crate::metrics::PhaseTimer::start(
            crate::metrics::PhaseKind::Facade,
            "compat_cycle",
            is_stream,
        );
        let mut hops = 0usize;

        loop {
            let resp = {
                let _send = crate::metrics::PhaseTimer::start(
                    crate::metrics::PhaseKind::Transport,
                    "send",
                    is_stream,
                );
                self.auth_headers(self.client.post(url))
                    .json(state.body())
                    .send()
                    .await
                    .map_err(|e| e.to_string())?
            };

            let status = resp.status().as_u16();
            if !COMPAT_STATUSES.contains(&status) {
                if resp.status().is_success() {
                    // Return the response unread: the caller decides whether to
                    // parse it as JSON (buffered) or relay it as a byte stream.
                    return Ok(SendOutcome::Success(resp));
                }
                let detail = read_error_detail(resp, is_stream).await;
                return Ok(SendOutcome::TerminalError { status, detail });
            }

            // Compat-eligible status: read the error body and see if a rewrite
            // applies. `next_retry` mutates `state` in place on a hit.
            let detail = read_error_detail(resp, is_stream).await;
            let error_text = detail_text(&detail);

            match compat::next_retry(state, &error_text, status) {
                Some(label) => {
                    hops += 1;
                    if hops > compat::MAX_COMPAT_HOPS {
                        tracing::warn!("compat retry hop cap exceeded ({hops})");
                        return Ok(SendOutcome::TerminalError { status, detail });
                    }
                    tracing::info!("upstream {status} compat retry: {label}");
                    continue;
                }
                None => return Ok(SendOutcome::TerminalError { status, detail }),
            }
        }
    }

    /// Exponential backoff with jitter: `0.5 * 2^attempt + rand(0, 0.5)`,
    /// capped at 30s. Matches the Python `backoff_delay`.
    async fn backoff(&self, attempt: u32) {
        let base = 0.5_f64;
        let delay = (base * 2f64.powi(attempt as i32)).min(30.0);
        let jitter = rand::thread_rng().gen_range(0.0..base);
        tokio::time::sleep(Duration::from_secs_f64(delay + jitter)).await;
    }
}

/// Result of a single transport attempt. `Success` carries the unread response
/// so the caller chooses between buffered JSON and a live byte-stream relay.
enum SendOutcome {
    Success(reqwest::Response),
    TerminalError { status: u16, detail: Value },
}

/// Read an upstream error response into a JSON detail value: the parsed body
/// when it is JSON, otherwise the raw text wrapped as a string, otherwise null.
async fn read_error_detail(resp: reqwest::Response, is_stream: bool) -> Value {
    // Times the error-body read as a distinct transport phase.
    let _phase = crate::metrics::PhaseTimer::start(
        crate::metrics::PhaseKind::Transport,
        "read_error_text",
        is_stream,
    );
    match resp.text().await {
        Ok(text) if !text.trim().is_empty() => {
            serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text))
        }
        _ => Value::Null,
    }
}

/// Flatten an error detail into searchable text for compat predicate matching.
fn detail_text(detail: &Value) -> String {
    match detail {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Extract a human-readable message from an upstream error detail, matching
/// `extract_upstream_error_message`: prefer `error.message`, then a bare
/// string, else a generic HTTP status line.
fn upstream_error_message(detail: &Value, status: u16) -> String {
    if let Some(msg) = detail
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
    {
        return msg.to_owned();
    }
    if let Value::String(s) = detail {
        if !s.trim().is_empty() {
            return s.clone();
        }
    }
    format!("Upstream returned HTTP {status}")
}

/// Parse and validate a raw Chat Completions request body for the relay path.
///
/// Returns the validated body as an object map ready to forward verbatim, or a
/// `BridgeError` with the OpenAI-style 400 code the client sees. The relay does
/// not convert or mutate the body, so validation is limited to the two fields
/// an upstream Chat Completions call always requires: a non-empty `model` and a
/// non-empty `messages` array.
pub fn validate_relay_body(raw: &[u8]) -> Result<Map<String, Value>, BridgeError> {
    let payload: Value = serde_json::from_slice(raw).map_err(|_| {
        BridgeError::invalid_request("Request body is not valid JSON.", "invalid_json_body")
    })?;

    let obj = match payload {
        Value::Object(map) => map,
        _ => {
            return Err(BridgeError::invalid_request(
                "Chat Completions request body must be a JSON object.",
                "invalid_chat_body",
            ));
        }
    };

    let model_ok = obj
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|m| !m.trim().is_empty());
    if !model_ok {
        return Err(BridgeError::invalid_request(
            "Chat Completions request is missing required field: model.",
            "missing_model",
        ));
    }

    let messages_ok = obj
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|m| !m.is_empty());
    if !messages_ok {
        return Err(BridgeError::invalid_request(
            "Chat Completions request is missing required field: messages.",
            "missing_messages",
        ));
    }

    Ok(obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detail_text_flattens_by_variant() {
        assert_eq!(detail_text(&json!("boom")), "boom");
        assert_eq!(detail_text(&Value::Null), "");
        // Objects stringify to compact JSON for keyword matching.
        let obj = detail_text(&json!({ "error": { "message": "top_p bad" } }));
        assert!(obj.contains("top_p bad"));
    }

    #[test]
    fn error_message_prefers_nested_error_message() {
        let detail = json!({ "error": { "message": "rate limited" } });
        assert_eq!(upstream_error_message(&detail, 429), "rate limited");
    }

    #[test]
    fn error_message_falls_back_to_bare_string() {
        let detail = json!("plain failure");
        assert_eq!(upstream_error_message(&detail, 500), "plain failure");
    }

    #[test]
    fn error_message_defaults_to_status_line() {
        assert_eq!(
            upstream_error_message(&Value::Null, 502),
            "Upstream returned HTTP 502"
        );
        // Empty string detail is treated as absent.
        assert_eq!(
            upstream_error_message(&json!("   "), 504),
            "Upstream returned HTTP 504"
        );
    }

    #[test]
    fn compat_and_retryable_status_sets_are_disjoint_on_400() {
        // 400 is compat-only (never blindly retried); 429/5xx are transport.
        assert!(COMPAT_STATUSES.contains(&400));
        assert!(!RETRYABLE_STATUSES.contains(&400));
        assert!(RETRYABLE_STATUSES.contains(&429));
    }

    fn relay_err_code(raw: &[u8]) -> String {
        match validate_relay_body(raw) {
            Ok(_) => panic!("expected validation error"),
            Err(e) => e.envelope()["error"]["code"].as_str().unwrap().to_owned(),
        }
    }

    #[test]
    fn validate_relay_body_accepts_minimal_valid_request() {
        let raw = json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hi" }],
        })
        .to_string();
        let obj = validate_relay_body(raw.as_bytes()).expect("valid body");
        // The body is returned verbatim for forwarding, extra fields intact.
        assert_eq!(obj.get("model").and_then(Value::as_str), Some("gpt-4o"));
        assert!(obj.get("messages").and_then(Value::as_array).is_some());
    }

    #[test]
    fn validate_relay_body_preserves_extra_fields_verbatim() {
        let raw = json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hi" }],
            "temperature": 0.7,
            "stream": true,
        })
        .to_string();
        let obj = validate_relay_body(raw.as_bytes()).expect("valid body");
        assert_eq!(obj.get("temperature"), Some(&json!(0.7)));
        assert_eq!(obj.get("stream"), Some(&json!(true)));
    }

    #[test]
    fn validate_relay_body_rejects_non_json() {
        assert_eq!(relay_err_code(b"not json at all"), "invalid_json_body");
    }

    #[test]
    fn validate_relay_body_rejects_non_object() {
        assert_eq!(relay_err_code(b"[1, 2, 3]"), "invalid_chat_body");
    }

    #[test]
    fn validate_relay_body_rejects_missing_or_blank_model() {
        assert_eq!(
            relay_err_code(
                json!({ "messages": [{ "role": "user" }] })
                    .to_string()
                    .as_bytes()
            ),
            "missing_model"
        );
        assert_eq!(
            relay_err_code(
                json!({ "model": "  ", "messages": [{ "role": "user" }] })
                    .to_string()
                    .as_bytes()
            ),
            "missing_model"
        );
    }

    #[test]
    fn validate_relay_body_rejects_missing_or_empty_messages() {
        assert_eq!(
            relay_err_code(json!({ "model": "gpt-4o" }).to_string().as_bytes()),
            "missing_messages"
        );
        assert_eq!(
            relay_err_code(
                json!({ "model": "gpt-4o", "messages": [] })
                    .to_string()
                    .as_bytes()
            ),
            "missing_messages"
        );
    }

    // --- Retry / compat orchestration (send_with_retry + send_with_compat) ---
    //
    // These exercise the transport+compat *wiring* — the loops, the attempt
    // counting, the compat-rewrite-then-resend path, and the terminal-vs-retry
    // branch — which the compat.rs unit tests (predicate logic only) and the
    // shadow oracle (happy path only) cannot reach. A mock upstream on an
    // ephemeral port counts hits; `start_paused` gives the backoff sleeps a
    // virtual clock so no real wall-time elapses.

    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::{Json, Router};

    /// Spawn a throwaway upstream on an ephemeral port; return its base URL.
    /// Binds the listener synchronously (so the port is accepting before we
    /// hand back the URL) and only moves the bound listener into the serve
    /// task, eliminating the connect-before-ready race.
    async fn spawn_upstream(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock upstream");
        let addr = listener.local_addr().expect("mock upstream addr");
        tokio::spawn(async move {
            axum::serve(listener, router)
                .await
                .expect("serve mock upstream");
        });
        // Poll the port until the accept loop is actually serving, so the first
        // client attempt cannot race ahead of readiness and see a spurious 502.
        for _ in 0..50 {
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        format!("http://{addr}")
    }

    fn upstream_for(base: &str) -> Upstream {
        Upstream::new(Config::for_test(base))
    }

    /// An upstream with an explicit retry budget. Network-fault tests use `0`
    /// so they assert the terminal mapping without paying the backoff sleep.
    fn upstream_with_retries(base: &str, max_retries: u32) -> Upstream {
        let mut config = Config::for_test(base);
        config.upstream_max_retries = max_retries;
        Upstream::new(config)
    }

    /// A port that nothing listens on, so a request fails at connect time —
    /// a real transport fault rather than an HTTP error status.
    const DEAD_UPSTREAM: &str = "http://127.0.0.1:1";

    fn chat_body() -> Map<String, Value> {
        json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hi" }],
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[tokio::test]
    async fn transport_retries_500_then_succeeds() {
        // First attempt 500 (retryable), second attempt 200: the transport loop
        // must resend after backoff and return the eventual success.
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let upstream = Router::new().route(
            "/chat/completions",
            post(|| async {
                let n = HITS.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response()
                } else {
                    Json(json!({ "id": "ok", "object": "chat.completion" })).into_response()
                }
            }),
        );
        let base = spawn_upstream(upstream).await;
        let out = upstream_for(&base)
            .create_chat_completion(chat_body())
            .await;

        assert_eq!(out.unwrap()["id"], "ok");
        assert_eq!(HITS.load(Ordering::SeqCst), 2, "should send exactly twice");
    }

    #[tokio::test]
    async fn transport_exhausts_retries_then_returns_status() {
        // Persistent 503: with max_retries=2 the loop makes 3 attempts total,
        // then surfaces the upstream status (not a flat 502) and stops.
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let upstream = Router::new().route(
            "/chat/completions",
            post(|| async {
                HITS.fetch_add(1, Ordering::SeqCst);
                (axum::http::StatusCode::SERVICE_UNAVAILABLE, "down").into_response()
            }),
        );
        let base = spawn_upstream(upstream).await;
        let err = upstream_for(&base)
            .create_chat_completion(chat_body())
            .await
            .unwrap_err();

        match err {
            BridgeError::Upstream { status, .. } => {
                assert_eq!(status.as_u16(), 503, "terminal status is surfaced");
            }
            other => panic!("expected Upstream error, got {other:?}"),
        }
        assert_eq!(
            HITS.load(Ordering::SeqCst),
            3,
            "1 initial + 2 retries = 3 attempts"
        );
    }

    #[tokio::test]
    async fn compat_rewrites_then_resends_on_400() {
        // A 400 whose body names top_p triggers the compat top_p clamp rule;
        // the compat loop must rewrite the body in place and resend within the
        // SAME attempt (no backoff), then succeed. The second hit must carry the
        // clamped top_p, proving the rewrite was actually applied to the wire.
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let upstream = Router::new().route(
            "/chat/completions",
            post(
                |axum::extract::Json(body): axum::extract::Json<Value>| async move {
                    let n = HITS.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        (
                            axum::http::StatusCode::BAD_REQUEST,
                            "invalid value for top_p",
                        )
                            .into_response()
                    } else {
                        // Prove the resend carries the clamped value, not the original.
                        assert_eq!(body["top_p"], json!(0.999), "resend must be rewritten");
                        Json(json!({ "id": "ok" })).into_response()
                    }
                },
            ),
        );
        let base = spawn_upstream(upstream).await;
        let mut body = chat_body();
        body.insert("top_p".to_owned(), json!(5.0));
        let out = upstream_for(&base).create_chat_completion(body).await;

        assert_eq!(out.unwrap()["id"], "ok");
        assert_eq!(HITS.load(Ordering::SeqCst), 2, "rewrite then one resend");
    }

    // --- Transport faults (connect-time failures, not HTTP error statuses) ---
    //
    // The status-code retry paths are covered above. These cover the `Err(net)`
    // arm: a request that never gets an HTTP response at all, which is what an
    // upstream outage or connection reset actually looks like.

    #[tokio::test]
    async fn network_fault_maps_to_upstream_error() {
        let err = upstream_with_retries(DEAD_UPSTREAM, 0)
            .create_chat_completion(chat_body())
            .await
            .expect_err("connect to a dead port must fail");

        match err {
            BridgeError::Upstream { code, .. } => {
                assert_eq!(code, "upstream_network_error");
            }
            other => panic!("expected an upstream error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn network_fault_is_retried_before_giving_up() {
        // With one retry the loop must make two attempts and still surface the
        // terminal network error rather than hanging or panicking.
        let err = upstream_with_retries(DEAD_UPSTREAM, 1)
            .create_chat_completion(chat_body())
            .await
            .expect_err("dead port must still fail after retrying");

        assert!(matches!(err, BridgeError::Upstream { .. }));
    }

    #[tokio::test]
    async fn relay_network_fault_maps_to_relay_failed() {
        // The relay path has its own `Err(net)` arm with a distinct code, so a
        // regression that crossed the two wouldn't be caught by the test above.
        let err = upstream_with_retries(DEAD_UPSTREAM, 0)
            .relay_chat_completion(chat_body())
            .await
            .expect_err("relay to a dead port must fail");

        match err {
            BridgeError::Upstream { code, .. } => {
                assert_eq!(code, "upstream_relay_failed");
            }
            other => panic!("expected an upstream error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_network_fault_maps_to_upstream_error() {
        // The success arm is an opaque `impl Stream` (no `Debug`), so match
        // rather than `expect_err`.
        match upstream_with_retries(DEAD_UPSTREAM, 0)
            .create_chat_completion_stream(chat_body())
            .await
        {
            Ok(_) => panic!("streaming from a dead port must fail"),
            Err(err) => assert!(matches!(err, BridgeError::Upstream { .. })),
        }
    }

    #[tokio::test]
    async fn list_models_network_fault_is_an_upstream_error() {
        let err = upstream_with_retries(DEAD_UPSTREAM, 0)
            .list_models()
            .await
            .expect_err("listing models from a dead port must fail");

        assert!(matches!(err, BridgeError::Upstream { .. }));
    }

    #[tokio::test]
    async fn non_retryable_status_returns_immediately() {
        // 401 is neither compat-eligible nor retryable: one attempt, no resend,
        // the status passes straight through.
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let upstream = Router::new().route(
            "/chat/completions",
            post(|| async {
                HITS.fetch_add(1, Ordering::SeqCst);
                (axum::http::StatusCode::UNAUTHORIZED, "nope").into_response()
            }),
        );
        let base = spawn_upstream(upstream).await;
        let err = upstream_for(&base)
            .create_chat_completion(chat_body())
            .await
            .unwrap_err();

        match err {
            BridgeError::Upstream { status, .. } => assert_eq!(status.as_u16(), 401),
            other => panic!("expected Upstream error, got {other:?}"),
        }
        assert_eq!(HITS.load(Ordering::SeqCst), 1, "no retry on 401");
    }
}
