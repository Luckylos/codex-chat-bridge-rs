//! Prometheus metrics, registered once in a lazily-initialized registry.
//!
//! Metric names match the Python bridge so existing Grafana dashboards keep
//! working against the Rust process.

use std::sync::OnceLock;

use prometheus::{Encoder, HistogramVec, IntCounterVec, Registry, TextEncoder};

struct Metrics {
    registry: Registry,
    requests_total: IntCounterVec,
    upstream_errors_total: IntCounterVec,
    upstream_phase_seconds: HistogramVec,
}

fn metrics() -> &'static Metrics {
    static METRICS: OnceLock<Metrics> = OnceLock::new();
    METRICS.get_or_init(|| {
        let registry = Registry::new();

        let requests_total = IntCounterVec::new(
            prometheus::opts!(
                "bridge_requests_total",
                "Total bridge requests by path and status"
            ),
            &["path", "status"],
        )
        .expect("valid metric");
        let upstream_errors_total = IntCounterVec::new(
            prometheus::opts!(
                "bridge_upstream_errors_total",
                "Upstream errors by model and status"
            ),
            &["model", "status_code"],
        )
        .expect("valid metric");
        let upstream_phase_seconds = HistogramVec::new(
            prometheus::histogram_opts!(
                "bridge_upstream_phase_seconds",
                "Upstream request phase timing in seconds"
            ),
            &["phase"],
        )
        .expect("valid metric");

        registry
            .register(Box::new(requests_total.clone()))
            .expect("register");
        registry
            .register(Box::new(upstream_errors_total.clone()))
            .expect("register");
        registry
            .register(Box::new(upstream_phase_seconds.clone()))
            .expect("register");

        Metrics {
            registry,
            requests_total,
            upstream_errors_total,
            upstream_phase_seconds,
        }
    })
}

pub fn record_request(path: &str, status: u16) {
    metrics()
        .requests_total
        .with_label_values(&[path, &status.to_string()])
        .inc();
}

pub fn record_upstream_error(model: &str, status_code: &str) {
    metrics()
        .upstream_errors_total
        .with_label_values(&[model, status_code])
        .inc();
}

pub fn observe_phase(phase: &str, seconds: f64) {
    metrics()
        .upstream_phase_seconds
        .with_label_values(&[phase])
        .observe(seconds);
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
