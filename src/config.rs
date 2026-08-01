//! Bridge configuration.
//!
//! Parsed once from `BRIDGE_*` environment variables at startup. The env var
//! names, defaults, and validation rules are the external contract shared with
//! the Python bridge (drop-in replacement); the internal shape is idiomatic
//! Rust — a plain struct whose fields the compiler guarantees are all set, so
//! no declarative registry or "unset" sentinel is needed.

use std::time::Duration;

use serde_json::{Map, Value};

/// A configuration error surfaced at startup so a malformed override fails
/// loudly instead of silently degrading.
#[derive(Debug)]
pub struct ConfigError {
    pub message: String,
    pub code: &'static str,
}

impl ConfigError {
    fn new(message: impl Into<String>, code: &'static str) -> Self {
        Self {
            message: message.into(),
            code,
        }
    }
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for ConfigError {}

/// Policy for Responses hosted tools the bridge cannot map to Chat Completions.
/// A real enum instead of a validated string — invalid values are rejected at
/// parse time and every consumer matches exhaustively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedToolPolicy {
    Ignore,
    Reject,
    Error,
    Passthrough,
}

/// Process-global unsupported-tool policy, seeded once at startup by
/// [`init_global_unsupported_tool_policy`].
static GLOBAL_UNSUPPORTED_TOOL_POLICY: std::sync::OnceLock<UnsupportedToolPolicy> =
    std::sync::OnceLock::new();

/// Seed the process-global unsupported-tool policy. Called once at startup
/// after config parse. Subsequent calls are no-ops (the first value wins),
/// matching the immutable-settings contract.
pub fn init_global_unsupported_tool_policy(policy: UnsupportedToolPolicy) {
    let _ = GLOBAL_UNSUPPORTED_TOOL_POLICY.set(policy);
}

/// Read the process-global unsupported-tool policy, or `Ignore` when unset
/// (e.g. in unit tests that never call [`init_global_unsupported_tool_policy`]).
pub fn global_unsupported_tool_policy() -> UnsupportedToolPolicy {
    GLOBAL_UNSUPPORTED_TOOL_POLICY
        .get()
        .copied()
        .unwrap_or(UnsupportedToolPolicy::Ignore)
}

impl UnsupportedToolPolicy {
    fn parse(raw: &str) -> Result<Self, ConfigError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ignore" => Ok(Self::Ignore),
            "reject" => Ok(Self::Reject),
            "error" => Ok(Self::Error),
            "passthrough" => Ok(Self::Passthrough),
            other => Err(ConfigError::new(
                format!(
                    "BRIDGE_UNSUPPORTED_TOOL_POLICY must be one of \
                     [error, ignore, passthrough, reject], got {other:?}"
                ),
                "tool_policy_invalid",
            )),
        }
    }
}

/// The upstream HTTP time budgets, split by the phase each one bounds.
///
/// Deliberately carries **no total-request budget**: a streamed LLM response
/// legitimately runs for minutes, so capping the request as a whole truncates
/// the generation mid-flight — the client sees an early EOF and the gateway
/// records a usage-less, unbillable turn. Because no "the timeout" scalar is
/// reachable from this type, no call site can reintroduce a total cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpstreamTimeouts {
    /// Maximum time to establish the connection.
    connect: Duration,
    /// Maximum gap *between* received bytes. Resets on every byte, so it trips
    /// only on a genuinely stalled upstream, never on a slow-but-alive stream.
    idle: Duration,
}

impl UpstreamTimeouts {
    /// Build from the operator-facing `BRIDGE_UPSTREAM_TIMEOUT_SECONDS` budget,
    /// which applies **per phase** rather than to the request as a whole.
    pub fn try_from_seconds(seconds: f64) -> Result<Self, ConfigError> {
        if !seconds.is_finite() || seconds <= 0.0 {
            return Err(ConfigError::new(
                format!(
                    "BRIDGE_UPSTREAM_TIMEOUT_SECONDS must be a finite number > 0, got {seconds}"
                ),
                "timeout_invalid",
            ));
        }
        let budget = Duration::from_secs_f64(seconds);
        Ok(Self {
            connect: budget,
            idle: budget,
        })
    }

    /// Apply the budgets to a `reqwest` client builder. The single place these
    /// map onto an HTTP client, so the total-request `.timeout()` is never wired
    /// in — keeping the invariant enforced here instead of at each call site.
    pub fn apply(&self, builder: reqwest::ClientBuilder) -> reqwest::ClientBuilder {
        builder
            .connect_timeout(self.connect)
            .read_timeout(self.idle)
    }
}

/// Fully-resolved bridge configuration. Every field is populated by
/// [`Config::from_env`]; there is no partially-constructed state.
///
#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub upstream_base_url: String,
    pub upstream_api_key: String,
    pub upstream_timeouts: UpstreamTimeouts,
    pub upstream_streaming: bool,
    pub upstream_max_retries: u32,
    pub max_concurrent_requests: usize,
    /// How long a request waits for a concurrency permit before the bridge
    /// sheds it with 429. Bounds the queue so a slow upstream plus sustained
    /// inflow cannot grow an unbounded backlog — a short burst is absorbed, a
    /// persistent overload is rejected fast rather than piling up latency.
    pub queue_timeout_seconds: f64,
    pub max_body_bytes: usize,
    pub unsupported_tool_policy: UnsupportedToolPolicy,
    // KNOWN PORTING GAP (not dead config): these three fields are the outbound
    // "upstream body policy" — the Python bridge parses the same env vars AND
    // applies them at the single send boundary (`upstream_body_policy.apply()`),
    // so a `tool_denylist` / `drop_params` / `extra_params` operator override
    // reshapes the bytes that leave the bridge without a code change. The Rust
    // port parses + validates them (below, with tests) but does NOT yet apply
    // them on the send path — so they read as never-used. They are retained,
    // not deleted, because deleting them would silently drop a live Python
    // capability. Wiring `.apply()` into `upstream.rs` is a deferred feature,
    // out of scope for the behavior-neutral refactor. Currently no-op in prod
    // (the env vars are unset), which is why removing them looked "safe".
    #[allow(dead_code)]
    pub upstream_tool_denylist: Vec<String>,
    #[allow(dead_code)]
    pub upstream_drop_params: Vec<String>,
    #[allow(dead_code)]
    pub upstream_extra_params: Map<String, Value>,
}

impl Config {
    /// Parse and validate configuration from the environment. Returns the first
    /// validation error encountered so startup can abort with a clear message.
    pub fn from_env() -> Result<Self, ConfigError> {
        let upstream_base_url = str_env("BRIDGE_UPSTREAM_BASE_URL", "")
            .trim_end_matches('/')
            .to_owned();
        if upstream_base_url.is_empty() {
            return Err(ConfigError::new(
                "BRIDGE_UPSTREAM_BASE_URL is required but not set. \
                 Example: BRIDGE_UPSTREAM_BASE_URL=https://newapi.example.com/v1",
                "upstream_base_url_empty",
            ));
        }

        let upstream_timeouts =
            UpstreamTimeouts::try_from_seconds(f64_env("BRIDGE_UPSTREAM_TIMEOUT_SECONDS", 60.0)?)?;

        let max_concurrent_requests = usize_env("BRIDGE_MAX_CONCURRENT_REQUESTS", 20)?;
        if max_concurrent_requests < 1 {
            return Err(ConfigError::new(
                "BRIDGE_MAX_CONCURRENT_REQUESTS must be >= 1",
                "concurrency_invalid",
            ));
        }

        let queue_timeout_seconds = f64_env("BRIDGE_QUEUE_TIMEOUT_SECONDS", 10.0)?;
        if !queue_timeout_seconds.is_finite() || queue_timeout_seconds <= 0.0 {
            return Err(ConfigError::new(
                format!(
                    "BRIDGE_QUEUE_TIMEOUT_SECONDS must be a finite number > 0, \
                     got {queue_timeout_seconds}"
                ),
                "queue_timeout_invalid",
            ));
        }

        let max_body_bytes = usize_env("BRIDGE_MAX_BODY_BYTES", 10 * 1024 * 1024)?;
        if max_body_bytes < 1 {
            return Err(ConfigError::new(
                "BRIDGE_MAX_BODY_BYTES must be >= 1",
                "max_body_bytes_invalid",
            ));
        }

        Ok(Self {
            host: str_env("BRIDGE_HOST", "0.0.0.0"),
            port: u16_env("BRIDGE_PORT", 18090)?,
            upstream_base_url,
            upstream_api_key: str_env("BRIDGE_UPSTREAM_API_KEY", ""),
            upstream_timeouts,
            upstream_streaming: bool_env("BRIDGE_UPSTREAM_STREAMING", true),
            upstream_max_retries: u32_env("BRIDGE_UPSTREAM_MAX_RETRIES", 2)?,
            max_concurrent_requests,
            queue_timeout_seconds,
            max_body_bytes,
            unsupported_tool_policy: UnsupportedToolPolicy::parse(&str_env(
                "BRIDGE_UNSUPPORTED_TOOL_POLICY",
                "ignore",
            ))?,
            upstream_tool_denylist: csv_env("BRIDGE_UPSTREAM_TOOL_DENYLIST"),
            upstream_drop_params: json_str_list_env("BRIDGE_UPSTREAM_DROP_PARAMS")?,
            upstream_extra_params: json_object_env("BRIDGE_UPSTREAM_EXTRA_PARAMS")?,
        })
    }

    /// The `/v1/chat/completions` upstream URL.
    pub fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.upstream_base_url)
    }

    /// The `/v1/models` upstream URL.
    pub fn models_url(&self) -> String {
        format!("{}/models", self.upstream_base_url)
    }

    /// Build a config for tests without touching the environment, pointing the
    /// upstream at a caller-supplied base URL (e.g. a mock server). Every other
    /// field takes the same default `from_env` would apply.
    #[cfg(test)]
    pub fn for_test(upstream_base_url: impl Into<String>) -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            port: 0,
            upstream_base_url: upstream_base_url.into(),
            upstream_api_key: String::new(),
            upstream_timeouts: UpstreamTimeouts::try_from_seconds(60.0)
                .expect("60s is a valid timeout budget"),
            upstream_streaming: false,
            upstream_max_retries: 2,
            max_concurrent_requests: 20,
            queue_timeout_seconds: 10.0,
            max_body_bytes: 10 * 1024 * 1024,
            unsupported_tool_policy: UnsupportedToolPolicy::Ignore,
            upstream_tool_denylist: Vec::new(),
            upstream_drop_params: Vec::new(),
            upstream_extra_params: Map::new(),
        }
    }
}

// --------------------------------------------------------------------------- #
// Primitive env parsers. Each applies the field default when the var is unset;
// shape errors return ConfigError so a bad override aborts startup.
// --------------------------------------------------------------------------- #

fn str_env(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn bool_env(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        ),
        Err(_) => default,
    }
}

fn f64_env(key: &str, default: f64) -> Result<f64, ConfigError> {
    match std::env::var(key) {
        Ok(raw) => raw.trim().parse().map_err(|_| {
            ConfigError::new(
                format!("{key} must be a number, got {raw:?}"),
                "config_invalid_float",
            )
        }),
        Err(_) => Ok(default),
    }
}

fn u32_env(key: &str, default: u32) -> Result<u32, ConfigError> {
    int_env(key, default)
}

fn u16_env(key: &str, default: u16) -> Result<u16, ConfigError> {
    int_env(key, default)
}

fn usize_env(key: &str, default: usize) -> Result<usize, ConfigError> {
    int_env(key, default)
}

/// Parse any unsigned integer env var, or the default when unset.
fn int_env<T>(key: &str, default: T) -> Result<T, ConfigError>
where
    T: std::str::FromStr,
{
    match std::env::var(key) {
        Ok(raw) => raw.trim().parse().map_err(|_| {
            ConfigError::new(
                format!("{key} must be an integer, got {raw:?}"),
                "config_invalid_int",
            )
        }),
        Err(_) => Ok(default),
    }
}

/// Comma-separated list → trimmed, non-empty items. Unset yields an empty vec.
fn csv_env(key: &str) -> Vec<String> {
    match std::env::var(key) {
        Ok(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// JSON array of strings → vec. Empty/unset yields `[]`. Rejects non-array
/// shapes and non-string members so a malformed override fails at startup.
fn json_str_list_env(key: &str) -> Result<Vec<String>, ConfigError> {
    let raw = match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => v,
        _ => return Ok(Vec::new()),
    };
    let parsed: Value = serde_json::from_str(&raw).map_err(|e| {
        ConfigError::new(
            format!("{key} must be a JSON array of strings, got invalid JSON: {e}"),
            "config_invalid_json_array",
        )
    })?;
    match parsed {
        Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                Value::String(s) => Ok(s),
                other => Err(ConfigError::new(
                    format!("{key} must be a JSON array of strings, got member {other}"),
                    "config_invalid_json_array",
                )),
            })
            .collect(),
        other => Err(ConfigError::new(
            format!("{key} must be a JSON array of strings, got {other}"),
            "config_invalid_json_array",
        )),
    }
}

/// JSON object env var → map. Empty/unset yields `{}`. Rejects non-object
/// shapes so a malformed override fails at startup rather than corrupting body.
fn json_object_env(key: &str) -> Result<Map<String, Value>, ConfigError> {
    let raw = match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => v,
        _ => return Ok(Map::new()),
    };
    let parsed: Value = serde_json::from_str(&raw).map_err(|e| {
        ConfigError::new(
            format!("{key} must be a JSON object, got invalid JSON: {e}"),
            "config_invalid_json_object",
        )
    })?;
    match parsed {
        Value::Object(map) => Ok(map),
        other => Err(ConfigError::new(
            format!("{key} must be a JSON object, got {other}"),
            "config_invalid_json_object",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize env mutation across tests — std::env is process-global.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn clear_bridge_env() {
        for (k, _) in std::env::vars() {
            if k.starts_with("BRIDGE_") {
                std::env::remove_var(k);
            }
        }
    }

    #[test]
    fn requires_upstream_base_url() {
        let _g = lock();
        clear_bridge_env();
        let err = Config::from_env().unwrap_err();
        assert_eq!(err.code, "upstream_base_url_empty");
    }

    #[test]
    fn defaults_are_applied() {
        let _g = lock();
        clear_bridge_env();
        std::env::set_var("BRIDGE_UPSTREAM_BASE_URL", "http://up/v1/");
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.upstream_base_url, "http://up/v1"); // trailing slash trimmed
        assert_eq!(cfg.port, 18090);
        assert_eq!(
            cfg.upstream_timeouts,
            UpstreamTimeouts::try_from_seconds(60.0).unwrap()
        );
        assert!(cfg.upstream_streaming);
        assert_eq!(cfg.max_concurrent_requests, 20);
        assert_eq!(cfg.unsupported_tool_policy, UnsupportedToolPolicy::Ignore);
        assert_eq!(cfg.models_url(), "http://up/v1/models");
        clear_bridge_env();
    }

    #[test]
    fn rejects_bad_tool_policy() {
        let _g = lock();
        clear_bridge_env();
        std::env::set_var("BRIDGE_UPSTREAM_BASE_URL", "http://up/v1");
        std::env::set_var("BRIDGE_UNSUPPORTED_TOOL_POLICY", "bogus");
        let err = Config::from_env().unwrap_err();
        assert_eq!(err.code, "tool_policy_invalid");
        clear_bridge_env();
    }

    #[test]
    fn upstream_timeouts_reject_non_positive_and_non_finite_budgets() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = UpstreamTimeouts::try_from_seconds(bad).unwrap_err();
            assert_eq!(err.code, "timeout_invalid");
        }
    }

    #[test]
    fn upstream_timeouts_apply_the_budget_to_every_phase() {
        // The budget is per-phase, so both phases carry the full value rather
        // than splitting it between them.
        let t = UpstreamTimeouts::try_from_seconds(2.5).unwrap();
        assert_eq!(t.connect, Duration::from_secs_f64(2.5));
        assert_eq!(t.idle, Duration::from_secs_f64(2.5));
    }

    #[test]
    fn rejects_non_finite_timeout() {
        let _g = lock();
        clear_bridge_env();
        std::env::set_var("BRIDGE_UPSTREAM_BASE_URL", "http://up/v1");
        std::env::set_var("BRIDGE_UPSTREAM_TIMEOUT_SECONDS", "0");
        let err = Config::from_env().unwrap_err();
        assert_eq!(err.code, "timeout_invalid");
        clear_bridge_env();
    }

    #[test]
    fn parses_json_and_csv_collections() {
        let _g = lock();
        clear_bridge_env();
        std::env::set_var("BRIDGE_UPSTREAM_BASE_URL", "http://up/v1");
        std::env::set_var("BRIDGE_UPSTREAM_TOOL_DENYLIST", "a, b ,,c");
        std::env::set_var("BRIDGE_UPSTREAM_DROP_PARAMS", r#"["x","y"]"#);
        std::env::set_var("BRIDGE_UPSTREAM_EXTRA_PARAMS", r#"{"k":1}"#);
        let cfg = Config::from_env().unwrap();
        assert_eq!(cfg.upstream_tool_denylist, ["a", "b", "c"]);
        assert_eq!(cfg.upstream_drop_params, ["x", "y"]);
        assert_eq!(cfg.upstream_extra_params.get("k"), Some(&Value::from(1)));
        clear_bridge_env();
    }

    #[test]
    fn rejects_non_object_extra_params() {
        let _g = lock();
        clear_bridge_env();
        std::env::set_var("BRIDGE_UPSTREAM_BASE_URL", "http://up/v1");
        std::env::set_var("BRIDGE_UPSTREAM_EXTRA_PARAMS", "[1,2]");
        let err = Config::from_env().unwrap_err();
        assert_eq!(err.code, "config_invalid_json_object");
        clear_bridge_env();
    }
}
