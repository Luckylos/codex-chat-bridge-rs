//! codex-chat-bridge (Rust) — entry point and HTTP surface.
//!
//! Phase 0 scaffold: process startup, config load + upstream connectivity
//! probe, and the read-only endpoints (`/health`, `/metrics`, `/v1/models`).
//! The `/v1/responses` conversion pipeline and `/v1/chat/completions` relay
//! land in later phases; their routes are registered now returning 501 so the
//! external surface shape is stable from the start.

mod compat;
mod config;
mod context;
mod convert;
mod error;
mod id_gen;
mod media;
mod metrics;
mod middleware;
#[cfg(test)]
mod parity_golden;
#[cfg(test)]
mod parity_stream_golden;
#[cfg(test)]
mod proptests;
mod protocol;
mod reasoning;
mod reasoning_cache;
mod sanitize;
mod session_bridge;
mod session_store;
mod sha256;
mod sse;
mod stream_chat_to_responses;
mod stream_envelope;
mod stream_events;
mod stream_inline_think;
mod stream_message;
mod stream_reasoning;
mod stream_responses_state;
mod stream_tools;
mod transform_loss;
mod types;
mod upstream;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use serde_json::json;

use crate::config::Config;
use crate::error::BridgeError;
use crate::types::ResponsesRequest;
use crate::upstream::Upstream;

/// Shared application state handed to every request handler.
struct AppState {
    upstream: Upstream,
    /// Result of the startup upstream connectivity probe: `Some(true/false)`
    /// once checked, `None` until then.
    upstream_reachable: AtomicBool,
    upstream_checked: AtomicBool,
    /// Full resolved config, consumed by the middleware stack (inbound auth,
    /// body-size boundary).
    config: Config,
    /// Concurrency permits gating the model routes; shared across requests.
    concurrency: std::sync::Arc<tokio::sync::Semaphore>,
}

type SharedState = Arc<AppState>;

pub async fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "codex_chat_bridge=info,tower_http=info".into()),
        )
        .init();

    let config = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Startup config validation failed: {e}");
            std::process::exit(1);
        }
    };

    if config.upstream_api_key.is_empty() {
        tracing::warn!(
            "BRIDGE_UPSTREAM_API_KEY is not set — upstream requests will lack Bearer auth."
        );
    }

    // Seed the process-global unsupported-tool policy so the tool-context
    // builder can read it without an AppState handle.
    config::init_global_unsupported_tool_policy(config.unsupported_tool_policy);

    let host = config.host.clone();
    let port = config.port;
    let upstream = Upstream::new(config.clone());
    let concurrency = middleware::new_semaphore(config.max_concurrent_requests);

    let state: SharedState = Arc::new(AppState {
        upstream: upstream.clone(),
        upstream_reachable: AtomicBool::new(false),
        upstream_checked: AtomicBool::new(false),
        config,
        concurrency,
    });

    // Startup connectivity probe: non-fatal.
    match upstream.list_models().await {
        Ok(models) => {
            state.upstream_reachable.store(true, Ordering::Relaxed);
            state.upstream_checked.store(true, Ordering::Relaxed);
            tracing::info!(
                "Upstream connectivity: ok ({} upstream models)",
                models.len()
            );
        }
        Err(e) => {
            state.upstream_reachable.store(false, Ordering::Relaxed);
            state.upstream_checked.store(true, Ordering::Relaxed);
            tracing::warn!("Upstream connectivity check failed: {e}");
        }
    }

    let app = build_app(state);

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("codex-chat-bridge listening on http://{addr}");

    let served = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    if let Err(e) = served {
        tracing::error!("Server error: {e}");
        std::process::exit(1);
    }
    tracing::info!("Shutdown complete.");
}

/// Assemble the router and middleware stack over a ready [`AppState`]. Split
/// from `main` so integration tests can exercise the full stack in-process via
/// `tower::ServiceExt::oneshot` without binding a socket.
fn build_app(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(list_models))
        .route("/v1/responses", post(create_response))
        // Compact shares the Responses pipeline verbatim, matching the Python
        // bridge's `_create_response_impl` reuse.
        .route("/v1/responses/compact", post(create_response))
        .route("/v1/chat/completions", post(chat_completions))
        // Layers run outer→inner in reverse registration order: request_log
        // (correlation + metrics) wraps everything, then the body-size
        // boundary, then the model-route concurrency limit closest to the
        // handlers.
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::concurrency_limit,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middleware::body_limit,
        ))
        .layer(axum::middleware::from_fn(middleware::request_log))
        .with_state(state)
}

/// Resolve when the process receives SIGTERM (container stop) or SIGINT
/// (Ctrl-C), so `axum::serve` stops accepting new connections and drains
/// in-flight requests before exit.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("SIGINT received, shutting down gracefully."),
        () = terminate => tracing::info!("SIGTERM received, shutting down gracefully."),
    }
}

/// `GET /health` — liveness plus the cached upstream-reachability tri-state.
async fn health(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let reachable = if state.upstream_checked.load(Ordering::Relaxed) {
        json!(state.upstream_reachable.load(Ordering::Relaxed))
    } else {
        json!(null)
    };
    Json(json!({
        "ok": true,
        "service": "codex-chat-bridge",
        "upstream_reachable": reachable,
    }))
}

/// `GET /metrics` — Prometheus text exposition format.
async fn metrics_endpoint() -> Response {
    let (content_type, body) = metrics::render();
    ([(axum::http::header::CONTENT_TYPE, content_type)], body).into_response()
}

/// `GET /v1/models` — proxy the upstream model catalogue as an OpenAI list.
async fn list_models(
    State(state): State<SharedState>,
) -> Result<Json<serde_json::Value>, BridgeError> {
    let models = state.upstream.list_models().await?;
    Ok(Json(json!({ "object": "list", "data": models })))
}

/// `POST /v1/chat/completions` — transparent relay for native Chat clients.
///
/// The body is forwarded verbatim to the upstream aggregator: no Responses↔Chat
/// conversion, no reasoning-policy rewrite, no compat mutation, no session
/// persistence. The bridge only adds transport-level retry (network faults +
/// 429/5xx) and passes the upstream status code, body, and SSE stream through
/// unchanged.
async fn chat_completions(
    State(state): State<SharedState>,
    body: axum::body::Bytes,
) -> Result<Response, BridgeError> {
    use futures::StreamExt;

    let obj = upstream::validate_relay_body(&body)?;
    let resp = state.upstream.relay_chat_completion(obj).await?;

    let status =
        axum::http::StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_owned();

    // Relay the upstream body as a byte stream so SSE and buffered JSON alike
    // pass through without the bridge materializing the whole body.
    let byte_stream = resp
        .bytes_stream()
        .map(|item| item.map_err(std::io::Error::other));
    let axum_body = axum::body::Body::from_stream(byte_stream);

    Ok((
        status,
        [(axum::http::header::CONTENT_TYPE, content_type)],
        axum_body,
    )
        .into_response())
}

/// `POST /v1/responses` — the core conversion pipeline.
///
/// Fresh (no `previous_response_id`) requests, streaming and non-streaming.
/// The request is converted to a Chat Completions body, sent upstream with the
/// full compat + transport retry, and the upstream turn is rendered back as a
/// Responses object (non-streaming) or a Responses SSE event stream. Session
/// continuation (`previous_response_id`) lands in a later phase.
async fn create_response(
    State(state): State<SharedState>,
    Json(payload): Json<ResponsesRequest>,
) -> Result<Response, BridgeError> {
    // n != 1 is unsupported — the bridge maps one Responses turn to one Chat
    // completion, matching the Python bridge's explicit rejection.
    if payload.n.is_some_and(|n| n != 1) {
        return Err(BridgeError::invalid_request(
            "Responses requests with n != 1 are not supported by this bridge.",
            "unsupported_n",
        ));
    }

    // Resolve `previous_response_id` into stored history + merged tool context
    // + prior model. A supplied-but-unknown id is a hard 404.
    let (existing_messages, session_context, session_model) =
        session_bridge::resolve_session(&payload)?;

    let mut resolved_model = payload.model.as_deref().unwrap_or("").trim().to_owned();
    if resolved_model.is_empty() {
        if let Some(m) = session_model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
        {
            resolved_model = m.to_owned();
        }
    }
    if resolved_model.is_empty() {
        return Err(BridgeError::invalid_request(
            "Responses request is missing required field: model.",
            "missing_model",
        ));
    }

    // On a continuation the stored (merged) context wins; otherwise build fresh.
    let tool_context =
        session_context.unwrap_or_else(|| context::build_tool_context_from_request(&payload));
    let chat_request = convert::responses_to_chat_with_session(
        &payload,
        &resolved_model,
        existing_messages.as_deref(),
        &tool_context,
    );
    let bridge_id = id_gen::new_response_id();
    let echo = payload.to_echo_map();

    if payload.stream {
        return create_response_streaming(
            state,
            chat_request,
            tool_context,
            bridge_id,
            echo,
            resolved_model,
        )
        .await;
    }

    let chat_body = state
        .upstream
        .create_chat_completion(chat_request.body.clone())
        .await?;
    let response_body = convert::chat_to_responses(
        &chat_body,
        &resolved_model,
        Some(&echo),
        &bridge_id,
        &tool_context,
    );

    // Persist the turn for later continuation when the finish_reason is
    // persistable and the turn produced an assistant message.
    let assistant_message = session_bridge::assistant_message_from_chat_body(&chat_body);
    let finish_reason = convert::chat_finish_reason(&chat_body);
    if assistant_message.is_some()
        && convert::should_persist_finish_reason(finish_reason.as_deref())
    {
        let messages = chat_request
            .body
            .get("messages")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        session_bridge::save_session(
            &bridge_id,
            &messages,
            &tool_context,
            &resolved_model,
            assistant_message,
        );
    }

    Ok(Json(response_body).into_response())
}

/// Streaming `/v1/responses`: emit a `text/event-stream` of Responses SSE
/// events. When the upstream serves streaming, its byte stream is decoded and
/// converted incrementally; otherwise the turn is buffered upstream and the
/// buffered body is rendered as a single burst of SSE events.
async fn create_response_streaming(
    state: SharedState,
    chat_request: crate::types::ChatRequest,
    tool_context: crate::context::BridgeToolContext,
    bridge_id: String,
    echo: serde_json::Map<String, serde_json::Value>,
    resolved_model: String,
) -> Result<Response, BridgeError> {
    use futures::StreamExt;

    // The effective request messages are the persist snapshot's base; the
    // finalized assistant message is appended at save time.
    let request_messages = chat_request
        .body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();

    let body = if state.upstream.upstream_streaming() {
        let upstream = state
            .upstream
            .create_chat_completion_stream(chat_request.body)
            .await?;
        // Capture everything the finalize-time persist needs. The assistant
        // message is reconstructed from stream state after finalize.
        let persist = stream_chat_to_responses::StreamPersist {
            response_id: bridge_id.clone(),
            messages: request_messages,
            tool_context: tool_context.clone(),
            model: resolved_model,
        };
        let events = stream_chat_to_responses::create_responses_sse_stream(
            upstream,
            tool_context,
            Some(bridge_id),
            Some(echo),
            Some(persist),
        );
        let byte_stream = events.map(Ok::<_, std::convert::Infallible>);
        axum::body::Body::from_stream(byte_stream)
    } else {
        let chat_body = state
            .upstream
            .create_chat_completion(chat_request.body)
            .await?;
        // Persistability is decided from the buffered body upfront, matching
        // the Python bridge's buffer-then-SSE path.
        let assistant_message = session_bridge::assistant_message_from_chat_body(&chat_body);
        let finish_reason = convert::chat_finish_reason(&chat_body);
        if assistant_message.is_some()
            && convert::should_persist_finish_reason(finish_reason.as_deref())
        {
            session_bridge::save_session(
                &bridge_id,
                &request_messages,
                &tool_context,
                &resolved_model,
                assistant_message,
            );
        }
        let events = stream_chat_to_responses::sse_events_from_buffered_chat(
            &chat_body,
            tool_context,
            Some(&bridge_id),
            Some(&echo),
        );
        let byte_stream =
            futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
        axum::body::Body::from_stream(byte_stream)
    };

    Ok((
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        body,
    )
        .into_response())
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use std::sync::atomic::AtomicUsize;
    use tower::ServiceExt;

    /// Build an [`AppState`] pointed at `upstream_base_url` with a chosen
    /// concurrency budget and body cap; probe flags are left unchecked so
    /// `/health` reports `null` reachability.
    fn test_state_with(
        upstream_base_url: &str,
        max_concurrent: usize,
        max_body_bytes: usize,
    ) -> SharedState {
        let mut config = Config::for_test(upstream_base_url);
        config.max_concurrent_requests = max_concurrent;
        config.max_body_bytes = max_body_bytes;
        let concurrency = middleware::new_semaphore(config.max_concurrent_requests);
        Arc::new(AppState {
            upstream: Upstream::new(config.clone()),
            upstream_reachable: AtomicBool::new(false),
            upstream_checked: AtomicBool::new(false),
            config,
            concurrency,
        })
    }

    /// Convenience state with default budgets for tests that only touch the
    /// control plane or exercise validation.
    fn test_state(upstream_base_url: &str) -> SharedState {
        test_state_with(upstream_base_url, 20, 10 * 1024 * 1024)
    }

    /// Spawn a throwaway upstream server on an ephemeral port and return its
    /// base URL (no trailing slash). The router is supplied by the caller.
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
        format!("http://{addr}")
    }

    /// Build the valid Chat Completions relay body used across tests.
    fn valid_chat_body() -> String {
        json!({
            "model": "gpt-4o",
            "messages": [{ "role": "user", "content": "hi" }],
        })
        .to_string()
    }

    #[tokio::test]
    async fn health_ok_and_echoes_correlation_id() {
        let app = build_app(test_state("http://127.0.0.1:1"));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("x-request-id", "corr-abc123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("x-bridge-request-id").unwrap(),
            "corr-abc123"
        );
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["ok"], json!(true));
        assert_eq!(body["upstream_reachable"], json!(null));
    }

    #[tokio::test]
    async fn missing_correlation_id_is_generated() {
        let app = build_app(test_state("http://127.0.0.1:1"));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let id = resp
            .headers()
            .get("x-bridge-request-id")
            .unwrap()
            .to_str()
            .unwrap();
        // Freshly minted ids are 32-char simple uuids.
        assert_eq!(id.len(), 32);
    }

    #[tokio::test]
    async fn oversized_body_is_rejected_with_413() {
        let app = build_app(test_state_with("http://127.0.0.1:1", 20, 32));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("x".repeat(1024)))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], json!("request_too_large"));
    }

    #[tokio::test]
    async fn relay_forwards_body_and_passes_status_through() {
        let upstream = Router::new().route(
            "/chat/completions",
            post(|| async { Json(json!({ "id": "relayed", "object": "chat.completion" })) }),
        );
        let base = spawn_upstream(upstream).await;
        let app = build_app(test_state(&base));

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(valid_chat_body()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["id"], json!("relayed"));
    }

    #[tokio::test]
    async fn relay_rejects_invalid_body_before_upstream() {
        // No upstream is reachable; the 400 must come from local validation.
        let app = build_app(test_state("http://127.0.0.1:1"));
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(json!({ "messages": [] }).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], json!("missing_model"));
    }

    #[tokio::test]
    async fn concurrency_limit_serializes_model_requests() {
        // A single permit forces the second request to wait for the first, so
        // the two upstream calls never overlap. Each request is driven in its
        // own task that collects the body, releasing the permit on completion —
        // the body guard holds the permit until the body stream finishes.
        static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
        static MAX_SEEN: AtomicUsize = AtomicUsize::new(0);

        let upstream = Router::new().route(
            "/chat/completions",
            post(|| async {
                let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
                MAX_SEEN.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
                Json(json!({ "id": "ok" }))
            }),
        );
        let base = spawn_upstream(upstream).await;
        let state = test_state_with(&base, 1, 10 * 1024 * 1024);

        let drive = |state: SharedState| {
            tokio::spawn(async move {
                let resp = build_app(state)
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/v1/chat/completions")
                            .header("content-type", "application/json")
                            .body(Body::from(valid_chat_body()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                let status = resp.status();
                // Collecting the body runs the guard's Drop, releasing the permit.
                resp.into_body().collect().await.unwrap();
                status
            })
        };

        let t1 = drive(state.clone());
        let t2 = drive(state.clone());
        let (a, b) = tokio::join!(t1, t2);
        assert_eq!(a.unwrap(), StatusCode::OK);
        assert_eq!(b.unwrap(), StatusCode::OK);
        assert_eq!(MAX_SEEN.load(Ordering::SeqCst), 1);
    }
}
