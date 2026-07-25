//! codex-chat-bridge (Rust) — entry point and HTTP surface.
//!
//! Phase 0 scaffold: process startup, config load + upstream connectivity
//! probe, and the read-only endpoints (`/health`, `/metrics`, `/v1/models`).
//! The `/v1/responses` conversion pipeline and `/v1/chat/completions` relay
//! land in later phases; their routes are registered now returning 501 so the
//! external surface shape is stable from the start.

mod config;
mod error;
mod metrics;
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
use crate::upstream::Upstream;

/// Shared application state handed to every request handler.
struct AppState {
    upstream: Upstream,
    /// Result of the startup upstream connectivity probe: `Some(true/false)`
    /// once checked, `None` until then. Mirrors the Python bridge's
    /// `app.state.health_upstream_reachable`.
    upstream_reachable: AtomicBool,
    upstream_checked: AtomicBool,
}

type SharedState = Arc<AppState>;

#[tokio::main]
async fn main() {
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

    let host = config.host.clone();
    let port = config.port;
    let upstream = Upstream::new(config.clone());

    let state: SharedState = Arc::new(AppState {
        upstream: upstream.clone(),
        upstream_reachable: AtomicBool::new(false),
        upstream_checked: AtomicBool::new(false),
    });

    // Startup connectivity probe: non-fatal, mirrors the Python lifespan.
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

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/models", get(list_models))
        .route("/v1/responses", post(not_implemented))
        .route("/v1/responses/compact", post(not_implemented))
        .route("/v1/chat/completions", post(not_implemented))
        .with_state(state);

    let addr = format!("{host}:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("codex-chat-bridge listening on http://{addr}");

    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("Server error: {e}");
        std::process::exit(1);
    }
    tracing::info!("Shutdown complete.");
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

/// Placeholder for routes whose pipeline lands in a later phase.
async fn not_implemented() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json!({
            "error": {
                "message": "This endpoint is not implemented yet in the Rust bridge.",
                "type": "not_implemented",
                "code": "not_implemented",
            }
        })),
    )
        .into_response()
}
