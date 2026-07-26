//! HTTP middleware stack.
//!
//! Three thin layers, applied outer→inner: request-log/correlation, request
//! body-size boundary, and a model-route concurrency limit. Each layer whose
//! effect must span the full (possibly streamed) response body — request
//! timing, the in-flight gauge, the concurrency permit — attaches a RAII guard
//! to the response body via [`attach_body_guard`], so the guard's `Drop` runs
//! when the body finishes or the client disconnects.
//!
//! The bridge is a thin trusted-network layer sitting behind new-api / the
//! relay; inbound authentication is intentionally delegated to that upstream
//! boundary rather than re-implemented here.

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::header;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::BridgeError;
use crate::metrics;
use crate::AppState;

/// Response header carrying the resolved request id back to the client.
const REQUEST_ID_HEADER: &str = "x-bridge-request-id";
/// Model routes subject to the concurrency limit; control-plane routes stay
/// responsive when model capacity is saturated.
const CONCURRENCY_LIMITED_PATHS: [&str; 3] = [
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/responses/compact",
];

// --------------------------------------------------------------------------- //
// Body guard: run a value's Drop after the response body completes.
// --------------------------------------------------------------------------- //

/// Re-wrap a response body so `guard` is held until the body stream finishes or
/// is dropped (client disconnect). This makes a guard's lifetime span the full
/// streamed response, not just handler return.
fn attach_body_guard<G: Send + 'static>(resp: Response, guard: G) -> Response {
    let (parts, body) = resp.into_parts();
    let data_stream = body.into_data_stream();
    let guarded = async_stream::stream! {
        let _guard = guard;
        futures::pin_mut!(data_stream);
        while let Some(item) = data_stream.next().await {
            yield item;
        }
    };
    Response::from_parts(parts, Body::from_stream(guarded))
}

// --------------------------------------------------------------------------- //
// Request-log + correlation (outermost).
// --------------------------------------------------------------------------- //

/// Records a completed request's metrics, emits a structured per-request log
/// line, and decrements the in-flight gauge on drop, so the observation spans
/// the full streamed-body lifetime (the log fires when the body finishes or the
/// client disconnects, not at handler return).
struct RequestMetricsGuard {
    request_id: String,
    method: String,
    path: String,
    status: u16,
    start: Instant,
}

impl Drop for RequestMetricsGuard {
    fn drop(&mut self) {
        let duration_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        metrics::record_request_full(&self.method, &self.path, self.status, duration_ms);
        metrics::dec_in_flight();
        // Per-request access log as a structured JSON line
        // (request_id / method / path / status / duration_ms).
        tracing::info!(
            request_id = %self.request_id,
            method = %self.method,
            path = %self.path,
            status = self.status,
            duration_ms = duration_ms as u64,
            "request completed"
        );
    }
}

/// Validate an incoming `x-request-id`: non-empty, ≤128 chars, printable ASCII.
/// Otherwise mint a fresh hex id.
fn resolve_request_id(req: &Request) -> String {
    if let Some(raw) = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
    {
        let candidate = raw.trim();
        if !candidate.is_empty()
            && candidate.len() <= 128
            && candidate.bytes().all(|b| (b'!'..=b'~').contains(&b))
        {
            return candidate.to_owned();
        }
    }
    uuid::Uuid::new_v4().simple().to_string()
}

/// Outermost layer: assign/propagate a request id, echo it on the response, and
/// record request metrics over the full response lifetime.
pub async fn request_log(req: Request, next: Next) -> Response {
    let method = req.method().as_str().to_owned();
    let path = req.uri().path().to_owned();
    let request_id = resolve_request_id(&req);

    metrics::inc_in_flight();
    let start = Instant::now();

    let mut resp = next.run(req).await;
    let status = resp.status().as_u16();

    // Echo the correlation id, replacing any upstream-set value.
    if let Ok(value) = header::HeaderValue::from_str(&request_id) {
        resp.headers_mut().insert(REQUEST_ID_HEADER, value);
    }

    let guard = RequestMetricsGuard {
        request_id,
        method,
        path,
        status,
        start,
    };
    attach_body_guard(resp, guard)
}

// --------------------------------------------------------------------------- //
// Request-body size boundary.
// --------------------------------------------------------------------------- //

/// Reject bodies larger than the configured budget. A declared `Content-Length`
/// over budget is rejected upfront; otherwise the body is buffered up to the
/// limit and rejected the moment it is exceeded, closing chunked-transfer and
/// misleading-header bypasses.
pub async fn body_limit(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let max = state.config.max_body_bytes;

    if let Some(declared) = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
    {
        if declared > max {
            return BridgeError::request_body_too_large(max).into_response();
        }
    }

    let (parts, body) = req.into_parts();
    match axum::body::to_bytes(body, max).await {
        Ok(bytes) => {
            next.run(Request::from_parts(parts, Body::from(bytes)))
                .await
        }
        Err(_) => BridgeError::request_body_too_large(max).into_response(),
    }
}

// --------------------------------------------------------------------------- //
// Concurrency limit.
// --------------------------------------------------------------------------- //

/// Holds a concurrency permit and the usage-gauge decrement until the response
/// body completes.
struct ConcurrencyGuard {
    _permit: OwnedSemaphorePermit,
}

impl Drop for ConcurrencyGuard {
    fn drop(&mut self) {
        metrics::dec_concurrency();
    }
}

/// Limit in-flight model requests to the configured capacity, holding the permit
/// until the full (possibly streamed) response body is delivered. Control-plane
/// routes are never gated.
pub async fn concurrency_limit(
    State(state): State<Arc<AppState>>,
    req: Request,
    next: Next,
) -> Response {
    if !CONCURRENCY_LIMITED_PATHS.contains(&req.uri().path()) {
        return next.run(req).await;
    }

    // Bounded wait: absorb a short burst, but shed sustained overload with 503
    // rather than growing an unbounded backlog behind a slow upstream. The
    // permit acquire itself never fails (the semaphore is never closed); only
    // the timeout arm rejects.
    let wait = std::time::Duration::from_secs_f64(state.config.queue_timeout_seconds);
    let permit =
        match tokio::time::timeout(wait, Arc::clone(&state.concurrency).acquire_owned()).await {
            Ok(permit) => permit.expect("concurrency semaphore is never closed"),
            Err(_) => {
                metrics::inc_shed();
                return BridgeError::overloaded(state.config.queue_timeout_seconds).into_response();
            }
        };
    metrics::inc_concurrency();
    let guard = ConcurrencyGuard { _permit: permit };

    let resp = next.run(req).await;
    attach_body_guard(resp, guard)
}

/// Build a concurrency semaphore with the configured permit count.
pub fn new_semaphore(permits: usize) -> Arc<Semaphore> {
    Arc::new(Semaphore::new(permits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request as HttpRequest;

    fn req_with_request_id(value: &str) -> Request {
        HttpRequest::builder()
            .header("x-request-id", value)
            .body(Body::empty())
            .unwrap()
    }

    #[test]
    fn valid_request_id_is_preserved() {
        let req = req_with_request_id("abc-123_XYZ");
        assert_eq!(resolve_request_id(&req), "abc-123_XYZ");
    }

    #[test]
    fn blank_request_id_is_replaced() {
        let req = req_with_request_id("   ");
        // A fresh uuid simple form is 32 hex chars.
        assert_eq!(resolve_request_id(&req).len(), 32);
    }

    #[test]
    fn oversized_request_id_is_replaced() {
        let long = "a".repeat(129);
        let req = req_with_request_id(&long);
        assert_eq!(resolve_request_id(&req).len(), 32);
    }

    #[test]
    fn non_printable_request_id_is_replaced() {
        let req = req_with_request_id("bad id with space");
        assert_eq!(resolve_request_id(&req).len(), 32);
    }
}
