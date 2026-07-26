//! Prometheus metrics, registered once in a lazily-initialized registry.
//!
//! Metric names, labels, and histogram buckets match the Python bridge so
//! existing Grafana dashboards keep working against the Rust process.

use std::sync::OnceLock;
use std::time::Instant;

use prometheus::{
    Encoder, HistogramVec, IntCounter, IntCounterVec, IntGauge, Registry, TextEncoder,
};

struct Metrics {
    registry: Registry,
    requests_total: IntCounterVec,
    requests_in_flight: IntGauge,
    request_duration_ms: HistogramVec,
    upstream_errors_total: IntCounterVec,
    // Low-level transport phase timing, labelled by phase and stream flag.
    // Observed via `observe_transport_phase` by the upstream client.
    upstream_transport_phase_duration_ms: HistogramVec,
    // Higher-level facade/orchestration phase timing, same label set.
    upstream_phase_duration_ms: HistogramVec,
    // Current concurrent request count, moved by the concurrency-limit layer.
    concurrency_usage: IntGauge,
    // Responses→Chat conversion loss events by kind and item type, so
    // previously silent degradation is observable.
    transform_loss_total: IntCounterVec,
    // Count of U+FFFD replacements the streaming decoder emitted for invalid
    // upstream bytes, so silent stream corruption is observable.
    stream_decode_replacements_total: IntCounter,
    // Requests shed with 503 because the concurrency-queue wait timed out, so
    // sustained overload (vs. a healthy burst) is visible in the metrics.
    requests_shed_total: IntCounter,
}

/// Request-duration buckets in milliseconds, matching the Python histogram.
const DURATION_BUCKETS_MS: &[f64] = &[
    10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0,
];

/// Upstream phase-duration buckets in milliseconds, matching the Python
/// transport + facade histograms.
const PHASE_BUCKETS_MS: &[f64] = &[
    0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
];

fn metrics() -> &'static Metrics {
    static METRICS: OnceLock<Metrics> = OnceLock::new();
    METRICS.get_or_init(|| {
        let registry = Registry::new();

        let requests_total = IntCounterVec::new(
            prometheus::opts!(
                "bridge_requests_total",
                "Total requests by method, path, status"
            ),
            &["method", "path", "status"],
        )
        .expect("valid metric");
        let requests_in_flight =
            IntGauge::new("bridge_requests_in_flight", "Currently in-flight requests")
                .expect("valid metric");
        let request_duration_ms = HistogramVec::new(
            prometheus::histogram_opts!(
                "bridge_request_duration_ms",
                "Request duration in ms",
                DURATION_BUCKETS_MS.to_vec()
            ),
            &["method", "path"],
        )
        .expect("valid metric");
        let upstream_errors_total = IntCounterVec::new(
            prometheus::opts!(
                "bridge_upstream_errors_total",
                "Upstream errors by model and upstream status code"
            ),
            &["model", "status_code"],
        )
        .expect("valid metric");
        let upstream_transport_phase_duration_ms = HistogramVec::new(
            prometheus::histogram_opts!(
                "bridge_upstream_transport_phase_duration_ms",
                "Upstream transport phase duration in ms",
                PHASE_BUCKETS_MS.to_vec()
            ),
            &["phase", "stream"],
        )
        .expect("valid metric");
        let upstream_phase_duration_ms = HistogramVec::new(
            prometheus::histogram_opts!(
                "bridge_upstream_phase_duration_ms",
                "Upstream facade/orchestration phase duration in ms",
                PHASE_BUCKETS_MS.to_vec()
            ),
            &["phase", "stream"],
        )
        .expect("valid metric");
        let concurrency_usage = IntGauge::new(
            "bridge_concurrency_usage",
            "Current concurrent request count (from semaphore)",
        )
        .expect("valid metric");
        let transform_loss_total = IntCounterVec::new(
            prometheus::opts!(
                "bridge_transform_loss_total",
                "Responses→Chat conversion loss events by kind and item type"
            ),
            &["kind", "item_type"],
        )
        .expect("valid metric");
        let stream_decode_replacements_total = IntCounter::new(
            "bridge_stream_decode_replacements_total",
            "U+FFFD replacements emitted while decoding upstream stream bytes",
        )
        .expect("valid metric");
        let requests_shed_total = IntCounter::new(
            "bridge_requests_shed_total",
            "Requests rejected with 503 after the concurrency-queue wait timed out",
        )
        .expect("valid metric");

        for collector in [
            Box::new(requests_total.clone()) as Box<dyn prometheus::core::Collector>,
            Box::new(requests_in_flight.clone()),
            Box::new(request_duration_ms.clone()),
            Box::new(upstream_errors_total.clone()),
            Box::new(upstream_transport_phase_duration_ms.clone()),
            Box::new(upstream_phase_duration_ms.clone()),
            Box::new(concurrency_usage.clone()),
            Box::new(transform_loss_total.clone()),
            Box::new(stream_decode_replacements_total.clone()),
            Box::new(requests_shed_total.clone()),
        ] {
            registry.register(collector).expect("register");
        }

        Metrics {
            registry,
            requests_total,
            requests_in_flight,
            request_duration_ms,
            upstream_errors_total,
            upstream_transport_phase_duration_ms,
            upstream_phase_duration_ms,
            concurrency_usage,
            transform_loss_total,
            stream_decode_replacements_total,
            requests_shed_total,
        }
    })
}

/// Record a completed request: increment the labelled counter and observe its
/// duration. `method`/`path`/`status` are the access-log labels.
pub fn record_request_full(method: &str, path: &str, status: u16, duration_ms: f64) {
    let m = metrics();
    m.requests_total
        .with_label_values(&[method, path, &status.to_string()])
        .inc();
    m.request_duration_ms
        .with_label_values(&[method, path])
        .observe(duration_ms);
}

/// Increment the in-flight gauge; paired with [`dec_in_flight`].
pub fn inc_in_flight() {
    metrics().requests_in_flight.inc();
}

/// Decrement the in-flight gauge.
pub fn dec_in_flight() {
    metrics().requests_in_flight.dec();
}

/// Increment the concurrency-usage gauge (a model route holds a permit).
pub fn inc_concurrency() {
    metrics().concurrency_usage.inc();
}

/// Decrement the concurrency-usage gauge.
pub fn dec_concurrency() {
    metrics().concurrency_usage.dec();
}

/// Count a request shed with 503 because the concurrency-queue wait timed out.
pub fn inc_shed() {
    metrics().requests_shed_total.inc();
}

pub fn record_upstream_error(model: &str, status_code: &str) {
    metrics()
        .upstream_errors_total
        .with_label_values(&[model, status_code])
        .inc();
}

/// Observe a low-level upstream transport phase duration (milliseconds),
/// labelled by phase and whether the request was streaming.
pub fn observe_transport_phase(phase: &str, stream: bool, duration_ms: f64) {
    metrics()
        .upstream_transport_phase_duration_ms
        .with_label_values(&[phase, stream_label(stream)])
        .observe(duration_ms);
}

/// Observe a higher-level upstream facade/orchestration phase duration
/// (milliseconds), labelled by phase and whether the request was streaming.
pub fn observe_phase(phase: &str, stream: bool, duration_ms: f64) {
    metrics()
        .upstream_phase_duration_ms
        .with_label_values(&[phase, stream_label(stream)])
        .observe(duration_ms);
}

/// Record a single Responses→Chat transform-loss event by kind and item type.
/// `item_type` is `"none"` when the offending item had no `type` field, matching
/// the Python collector's label for a missing type.
pub fn record_transform_loss(kind: &str, item_type: Option<&str>) {
    metrics()
        .transform_loss_total
        .with_label_values(&[kind, item_type.unwrap_or("none")])
        .inc();
}

/// Which of the two upstream phase histograms an RAII [`PhaseTimer`] feeds.
#[derive(Clone, Copy)]
pub enum PhaseKind {
    /// Low-level transport phase (`send` / `close` / `read_error_text`).
    Transport,
    /// Higher-level facade/orchestration phase (`request_retry` / `compat_cycle`).
    Facade,
}

/// RAII timer that observes an upstream phase duration on drop, so the metric
/// is recorded on every exit path (success, error, or early return). Duration
/// is rounded to whole milliseconds.
pub struct PhaseTimer {
    kind: PhaseKind,
    phase: &'static str,
    stream: bool,
    start: Instant,
}

impl PhaseTimer {
    /// Start timing `phase` for the given histogram `kind` and stream flag.
    pub fn start(kind: PhaseKind, phase: &'static str, stream: bool) -> Self {
        Self {
            kind,
            phase,
            stream,
            start: Instant::now(),
        }
    }
}

impl Drop for PhaseTimer {
    fn drop(&mut self) {
        let duration_ms = (self.start.elapsed().as_secs_f64() * 1000.0).round();
        match self.kind {
            PhaseKind::Transport => observe_transport_phase(self.phase, self.stream, duration_ms),
            PhaseKind::Facade => observe_phase(self.phase, self.stream, duration_ms),
        }
    }
}

/// Render the `stream` histogram label the way the Python bridge does:
/// `"stream"` for a streaming request, `"body"` for a buffered one.
fn stream_label(stream: bool) -> &'static str {
    if stream {
        "stream"
    } else {
        "body"
    }
}

/// Count U+FFFD replacements emitted while decoding upstream stream bytes.
pub fn record_stream_decode_replacements(count: u64) {
    if count > 0 {
        metrics().stream_decode_replacements_total.inc_by(count);
    }
}

/// Render all metrics in the Prometheus text exposition format.
pub fn render() -> (String, Vec<u8>) {
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();
    let families = metrics().registry.gather();
    encoder
        .encode(&families, &mut buffer)
        .expect("encode metrics");
    (encoder.format_type().to_owned(), buffer)
}
