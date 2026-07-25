//! Upstream 400/5xx compatibility retry policy.
//!
//! When an OpenAI-compatible upstream rejects a request with a 400 (or a
//! narrow set of 5xx stream-open failures), the offending field can often be
//! stripped or clamped and the request retried. This module owns that state
//! machine, exposed as three pieces:
//!
//! * [`ReasoningState`] carries the mutable request body plus the reasoning
//!   wire-mode and the set of compat labels already applied, so no rule can
//!   re-fire and loop.
//! * [`initial_state`] seeds the wire-mode from the model bucket and any
//!   caller-supplied `reasoning_effort` / `thinking`, stripping reasoning
//!   fields the target provider does not accept — matching
//!   `build_initial_reasoning_state`.
//! * [`next_retry`] inspects an error body + status and returns the next body
//!   to try (with a label for logging) or `None` when nothing more can help.
//!
//! The rules fire in a fixed precedence, with an owned `Map` mutated in place
//! and a `HashSet<&'static str>` tracking applied labels so no rule can
//! re-fire and loop.

use std::collections::HashSet;

use serde_json::{json, Map, Value};

use crate::reasoning::{normalize_canonical_effort, CanonicalEffort};

/// How reasoning parameters are encoded on the wire for the current attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireMode {
    /// Send no reasoning parameters; preserve provider defaults.
    ProviderDefault,
    /// Send an explicit `reasoning_effort`.
    EffortOnly,
}

/// Mutable per-request state threaded through the compat retry loop.
#[derive(Debug, Clone)]
pub struct ReasoningState {
    body: Map<String, Value>,
    canonical_effort: CanonicalEffort,
    wire_mode: WireMode,
    applied: HashSet<&'static str>,
}

impl ReasoningState {
    /// The body to send on the current attempt.
    pub fn body(&self) -> &Map<String, Value> {
        &self.body
    }
}

fn model_of(body: &Map<String, Value>) -> &str {
    body.get("model").and_then(Value::as_str).unwrap_or("")
}

/// Model buckets that never accept `reasoning_effort` (kimi/moonshot). Kept
/// local rather than exported from `reasoning` because the wire-mode seeding
/// here needs the bucket decision, not the effort-string mapping.
fn is_passthrough_model(model: &str) -> bool {
    // wire_reasoning_effort returns None for passthrough buckets regardless of
    // effort; reuse it as the single source of truth for the bucket split so
    // the two modules cannot drift.
    crate::reasoning::wire_reasoning_effort(model, CanonicalEffort::High).is_none()
}

fn strip_reasoning_fields(body: &mut Map<String, Value>) {
    body.remove("thinking");
    body.remove("reasoning_effort");
}

/// Infer the canonical effort already encoded in a raw chat body — either an
/// explicit `reasoning_effort` string or a `thinking: {type: "disabled"}`
/// block.
fn infer_effort(body: &Map<String, Value>) -> CanonicalEffort {
    let explicit = normalize_canonical_effort(body.get("reasoning_effort").and_then(Value::as_str));
    if explicit != CanonicalEffort::Unspecified {
        return explicit;
    }
    if let Some(thinking) = body.get("thinking").and_then(Value::as_object) {
        if thinking.get("type").and_then(Value::as_str) == Some("disabled") {
            return CanonicalEffort::None;
        }
    }
    CanonicalEffort::Unspecified
}

fn effort_wire_str(effort: CanonicalEffort) -> &'static str {
    match effort {
        CanonicalEffort::None => "none",
        CanonicalEffort::High => "high",
        CanonicalEffort::XHigh => "xhigh",
        CanonicalEffort::Unspecified => "unspecified",
    }
}

/// Seed the reasoning state from a freshly built chat body. Strips reasoning
/// fields the target provider cannot accept and re-encodes `reasoning_effort`
/// for effort-bucket models. Equivalent to `build_initial_reasoning_state`.
pub fn initial_state(mut body: Map<String, Value>) -> ReasoningState {
    let canonical_effort = infer_effort(&body);
    let passthrough = is_passthrough_model(model_of(&body));

    let wire_mode = if canonical_effort == CanonicalEffort::Unspecified || passthrough {
        WireMode::ProviderDefault
    } else {
        WireMode::EffortOnly
    };

    strip_reasoning_fields(&mut body);
    if wire_mode == WireMode::EffortOnly && canonical_effort != CanonicalEffort::Unspecified {
        body.insert(
            "reasoning_effort".to_owned(),
            json!(effort_wire_str(canonical_effort)),
        );
    }

    ReasoningState {
        body,
        canonical_effort,
        wire_mode,
        applied: HashSet::new(),
    }
}

fn error_mentions(error: &str, needle: &str) -> bool {
    error.to_ascii_lowercase().contains(needle)
}

/// A generic single-field compat rule: match on the error body + current body,
/// then rewrite one field.
struct GenericRule {
    label: &'static str,
    matches: fn(&Map<String, Value>, &str) -> bool,
    rewrite: fn(&mut Map<String, Value>),
}

fn top_p_out_of_range(body: &Map<String, Value>, error: &str) -> bool {
    body.get("top_p").is_some_and(|v| !v.is_null()) && error_mentions(error, "top_p")
}
fn clamp_top_p(body: &mut Map<String, Value>) {
    body.insert("top_p".to_owned(), json!(0.999));
}

fn include_usage_rejected(body: &Map<String, Value>, error: &str) -> bool {
    body.get("stream_options")
        .and_then(Value::as_object)
        .and_then(|o| o.get("include_usage"))
        .and_then(Value::as_bool)
        == Some(true)
        && error_mentions(error, "include_usage")
}
fn disable_include_usage(body: &mut Map<String, Value>) {
    let mut opts = body
        .get("stream_options")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    opts.insert("include_usage".to_owned(), json!(false));
    body.insert("stream_options".to_owned(), Value::Object(opts));
}

fn stream_options_rejected(body: &Map<String, Value>, error: &str) -> bool {
    body.get("stream_options").is_some_and(|v| !v.is_null())
        && error_mentions(error, "stream_options")
}
fn strip_stream_options(body: &mut Map<String, Value>) {
    body.remove("stream_options");
}

fn parallel_tool_calls_rejected(body: &Map<String, Value>, error: &str) -> bool {
    body.get("parallel_tool_calls")
        .is_some_and(|v| !v.is_null())
        && error_mentions(error, "parallel_tool_calls")
}
fn strip_parallel_tool_calls(body: &mut Map<String, Value>) {
    body.remove("parallel_tool_calls");
}

const GENERIC_RULES: &[GenericRule] = &[
    GenericRule {
        label: "top_p_out_of_range",
        matches: top_p_out_of_range,
        rewrite: clamp_top_p,
    },
    GenericRule {
        label: "include_usage_rejected",
        matches: include_usage_rejected,
        rewrite: disable_include_usage,
    },
    GenericRule {
        label: "stream_options_rejected",
        matches: stream_options_rejected,
        rewrite: strip_stream_options,
    },
    GenericRule {
        label: "parallel_tool_calls_rejected",
        matches: parallel_tool_calls_rejected,
        rewrite: strip_parallel_tool_calls,
    },
];

/// The retry hop cap: one hop per generic rule plus a small margin for the
/// reasoning-mode transitions.
pub const MAX_COMPAT_HOPS: usize = GENERIC_RULES.len() + 3;

/// Compute the next retry state after an upstream error, or `None` when no
/// compat rule applies. `status` is the upstream HTTP status; `error` is the
/// upstream error body text. On `Some`, `state` has been advanced in place with
/// the rewritten body and the applied label recorded.
///
/// Precedence matches `UpstreamCompatPolicy.retry_state`:
/// 1. generic single-field rules (400 only),
/// 2. explicit-tool-choice disable-reasoning (400 tool_choice/thinking, or
///    narrow 5xx empty-stream),
/// 3. reasoning-effort fallback to provider_default (400 only),
/// 4. raw `thinking` strip (400 only).
pub fn next_retry(state: &mut ReasoningState, error: &str, status: u16) -> Option<&'static str> {
    if status == 400 {
        for rule in GENERIC_RULES {
            if state.applied.contains(rule.label) {
                continue;
            }
            if (rule.matches)(&state.body, error) {
                (rule.rewrite)(&mut state.body);
                state.applied.insert(rule.label);
                return Some(rule.label);
            }
        }
    }

    if let Some(label) = explicit_tool_choice_retry(state, error, status) {
        return Some(label);
    }

    if status != 400 {
        return None;
    }

    if let Some(label) = reasoning_fallback(state, error) {
        return Some(label);
    }

    raw_thinking_strip(state, error)
}

fn has_tool_choice_object(body: &Map<String, Value>) -> bool {
    body.get("tool_choice").is_some_and(Value::is_object)
}

fn explicit_tool_choice_retry(
    state: &mut ReasoningState,
    error: &str,
    status: u16,
) -> Option<&'static str> {
    const LABEL: &str = "explicit_tool_choice_disable_reasoning";
    if state.canonical_effort != CanonicalEffort::Unspecified {
        return None;
    }
    if state.applied.contains(LABEL) {
        return None;
    }
    if !has_tool_choice_object(&state.body) {
        return None;
    }

    if status == 400 {
        if !error_mentions(error, "tool_choice") || !error_mentions(error, "thinking mode") {
            return None;
        }
    } else if matches!(status, 500 | 503) {
        if state.body.get("stream").and_then(Value::as_bool) != Some(true) {
            return None;
        }
        if !error_mentions(error, "empty_stream")
            && !error_mentions(error, "upstream stream closed before first payload")
        {
            return None;
        }
    } else {
        return None;
    }

    state.body.remove("thinking");
    state
        .body
        .insert("reasoning_effort".to_owned(), json!("none"));
    state.canonical_effort = CanonicalEffort::None;
    state.wire_mode = WireMode::EffortOnly;
    state.applied.insert(LABEL);
    Some(LABEL)
}

fn reasoning_fallback(state: &mut ReasoningState, error: &str) -> Option<&'static str> {
    const LABEL: &str = "unsupported_reasoning_effort_to_provider_default";
    if !error_mentions(error, "reasoning_effort") {
        return None;
    }
    if state.wire_mode == WireMode::ProviderDefault {
        return None;
    }
    if state.applied.contains(LABEL) {
        return None;
    }
    strip_reasoning_fields(&mut state.body);
    state.wire_mode = WireMode::ProviderDefault;
    state.applied.insert(LABEL);
    Some(LABEL)
}

fn raw_thinking_strip(state: &mut ReasoningState, error: &str) -> Option<&'static str> {
    const LABEL: &str = "unsupported_thinking_strip_raw_thinking";
    if state.wire_mode != WireMode::ProviderDefault {
        return None;
    }
    if state.applied.contains(LABEL) {
        return None;
    }
    state.body.get("thinking")?;
    if !error_mentions(error, "thinking") {
        return None;
    }
    state.body.remove("thinking");
    state.applied.insert(LABEL);
    Some(LABEL)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_with(model: &str, extra: Value) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("model".to_owned(), json!(model));
        if let Value::Object(o) = extra {
            m.extend(o);
        }
        m
    }

    #[test]
    fn effort_model_encodes_reasoning_effort() {
        let state = initial_state(body_with(
            "deepseek-v3",
            json!({ "reasoning_effort": "high" }),
        ));
        assert_eq!(
            state.body().get("reasoning_effort").and_then(Value::as_str),
            Some("high")
        );
    }

    #[test]
    fn passthrough_model_strips_effort() {
        let state = initial_state(body_with(
            "moonshot/kimi-k2",
            json!({ "reasoning_effort": "high" }),
        ));
        assert!(state.body().get("reasoning_effort").is_none());
    }

    #[test]
    fn top_p_clamp_fires_once() {
        let mut state = initial_state(body_with("gpt-4o", json!({ "top_p": 1.5 })));
        let label = next_retry(&mut state, "top_p must be <= 1", 400);
        assert_eq!(label, Some("top_p_out_of_range"));
        assert_eq!(
            state.body().get("top_p").and_then(Value::as_f64),
            Some(0.999)
        );
        // Same error again — rule already applied, no re-fire.
        let again = next_retry(&mut state, "top_p must be <= 1", 400);
        assert_eq!(again, None);
    }

    #[test]
    fn reasoning_fallback_to_provider_default() {
        let mut state = initial_state(body_with(
            "deepseek-v3",
            json!({ "reasoning_effort": "high" }),
        ));
        let label = next_retry(&mut state, "reasoning_effort not supported", 400);
        assert_eq!(
            label,
            Some("unsupported_reasoning_effort_to_provider_default")
        );
        assert!(state.body().get("reasoning_effort").is_none());
    }

    // ----------------------------------------------------------------------- //
    // Generic single-field rules
    // ----------------------------------------------------------------------- //

    #[test]
    fn include_usage_disabled_on_rejection() {
        let mut state = initial_state(body_with(
            "gpt-4o",
            json!({ "stream_options": { "include_usage": true } }),
        ));
        let label = next_retry(&mut state, "include_usage is not supported", 400);
        assert_eq!(label, Some("include_usage_rejected"));
        assert_eq!(
            state.body()["stream_options"]["include_usage"],
            json!(false)
        );
    }

    #[test]
    fn stream_options_stripped_on_rejection() {
        let mut state = initial_state(body_with(
            "gpt-4o",
            json!({ "stream_options": { "include_usage": false } }),
        ));
        // include_usage rule needs `true`, so only the strip rule matches here.
        let label = next_retry(&mut state, "stream_options not allowed", 400);
        assert_eq!(label, Some("stream_options_rejected"));
        assert!(state.body().get("stream_options").is_none());
    }

    #[test]
    fn parallel_tool_calls_stripped_on_rejection() {
        let mut state = initial_state(body_with("gpt-4o", json!({ "parallel_tool_calls": true })));
        let label = next_retry(&mut state, "parallel_tool_calls unsupported", 400);
        assert_eq!(label, Some("parallel_tool_calls_rejected"));
        assert!(state.body().get("parallel_tool_calls").is_none());
    }

    #[test]
    fn generic_rules_do_not_fire_on_non_400() {
        let mut state = initial_state(body_with("gpt-4o", json!({ "top_p": 1.5 })));
        assert_eq!(next_retry(&mut state, "top_p too big", 500), None);
    }

    #[test]
    fn unmatched_error_yields_none() {
        let mut state = initial_state(body_with("gpt-4o", json!({ "top_p": 1.5 })));
        assert_eq!(next_retry(&mut state, "some unrelated error", 400), None);
    }

    // ----------------------------------------------------------------------- //
    // Explicit tool-choice disable-reasoning
    // ----------------------------------------------------------------------- //

    #[test]
    fn explicit_tool_choice_disables_reasoning_on_400() {
        let mut state = initial_state(body_with(
            "gpt-4o",
            json!({ "tool_choice": { "type": "function", "function": { "name": "f" } } }),
        ));
        let label = next_retry(
            &mut state,
            "tool_choice cannot be used with thinking mode",
            400,
        );
        assert_eq!(label, Some("explicit_tool_choice_disable_reasoning"));
        assert_eq!(state.body()["reasoning_effort"], json!("none"));
        assert!(state.body().get("thinking").is_none());
    }

    #[test]
    fn explicit_tool_choice_needs_both_keywords() {
        let mut state = initial_state(body_with(
            "gpt-4o",
            json!({ "tool_choice": { "type": "function" } }),
        ));
        // Only "tool_choice" present, missing "thinking mode" — no fire.
        assert_eq!(next_retry(&mut state, "tool_choice invalid", 400), None);
    }

    #[test]
    fn explicit_tool_choice_fires_on_5xx_empty_stream() {
        let mut state = initial_state(body_with(
            "gpt-4o",
            json!({
                "tool_choice": { "type": "function" },
                "stream": true,
            }),
        ));
        let label = next_retry(&mut state, "empty_stream from upstream", 503);
        assert_eq!(label, Some("explicit_tool_choice_disable_reasoning"));
        assert_eq!(state.body()["reasoning_effort"], json!("none"));
    }

    #[test]
    fn explicit_tool_choice_5xx_requires_stream_true() {
        let mut state = initial_state(body_with(
            "gpt-4o",
            json!({ "tool_choice": { "type": "function" } }),
        ));
        // Not streaming → the 5xx branch bails.
        assert_eq!(next_retry(&mut state, "empty_stream", 503), None);
    }

    #[test]
    fn explicit_tool_choice_skipped_when_effort_specified() {
        let mut state = initial_state(body_with(
            "deepseek-v3",
            json!({
                "reasoning_effort": "high",
                "tool_choice": { "type": "function" },
            }),
        ));
        // canonical_effort != Unspecified → the rule is inhibited; the error
        // also mentions reasoning_effort, so the fallback rule fires instead.
        let label = next_retry(
            &mut state,
            "tool_choice with thinking mode: reasoning_effort issue",
            400,
        );
        assert_eq!(
            label,
            Some("unsupported_reasoning_effort_to_provider_default")
        );
    }

    // ----------------------------------------------------------------------- //
    // Raw thinking strip
    // ----------------------------------------------------------------------- //

    #[test]
    fn raw_thinking_stripped_in_provider_default_mode() {
        // No effort → ProviderDefault wire mode; thinking survives initial strip
        // only if re-added, so inject it post-seed via a fresh body.
        let mut body = body_with("moonshot/kimi-k2", json!({}));
        body.insert("thinking".to_owned(), json!({ "type": "enabled" }));
        let mut state = initial_state(body);
        // initial_state strips thinking, so re-insert to model a passthrough
        // upstream that still received it another way.
        state
            .body
            .insert("thinking".to_owned(), json!({ "type": "enabled" }));
        let label = next_retry(&mut state, "thinking is not supported", 400);
        assert_eq!(label, Some("unsupported_thinking_strip_raw_thinking"));
        assert!(state.body().get("thinking").is_none());
    }

    #[test]
    fn thinking_disabled_block_infers_none_effort() {
        let state = initial_state(body_with(
            "deepseek-v3",
            json!({ "thinking": { "type": "disabled" } }),
        ));
        // Effort None → EffortOnly wire mode encodes reasoning_effort=none.
        assert_eq!(state.body().get("reasoning_effort"), Some(&json!("none")));
    }
}
