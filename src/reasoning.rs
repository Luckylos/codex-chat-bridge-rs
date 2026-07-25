//! Reasoning-effort policy.
//!
//! Normalizes caller effort into the frozen canonical set and decides the wire
//! encoding by provider bucket. External behavior matches the Python bridge's
//! `reasoning_policy.py`; the internal shape is a small enum + match rather than
//! the Python dataclass/regex-tuple layout.

use std::sync::OnceLock;

use regex_lite::Regex;

/// Canonical effort after normalization. `Unspecified` means the caller sent no
/// effort at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalEffort {
    Unspecified,
    None,
    High,
    XHigh,
}

impl CanonicalEffort {
    /// The wire value sent as `reasoning_effort`. `None`/`Unspecified` never
    /// reach the wire (they map to provider_default), so they render as the
    /// closest string only for completeness.
    fn wire_str(self) -> &'static str {
        match self {
            CanonicalEffort::Unspecified => "unspecified",
            CanonicalEffort::None => "none",
            CanonicalEffort::High => "high",
            CanonicalEffort::XHigh => "xhigh",
        }
    }
}

/// Provider bucket derived from the model name. `Effort` accepts a
/// `reasoning_effort` field; `Passthrough` (kimi/moonshot) never sends reasoning
/// params and preserves provider defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bucket {
    Effort,
    Passthrough,
}

struct BucketRule {
    pattern: Regex,
    bucket: Bucket,
}

fn bucket_rules() -> &'static [BucketRule] {
    static RULES: OnceLock<Vec<BucketRule>> = OnceLock::new();
    RULES
        .get_or_init(|| {
            vec![
                BucketRule {
                    pattern: Regex::new(r"(?i)(?:^|[/\-])(kimi|moonshot)").unwrap(),
                    bucket: Bucket::Passthrough,
                },
                BucketRule {
                    pattern: Regex::new(r"(?i)(?:^|[/\-])(deepseek)").unwrap(),
                    bucket: Bucket::Effort,
                },
                BucketRule {
                    pattern: Regex::new(r"(?i)(?:^|[/\-])(glm|zhipu|bigmodel)").unwrap(),
                    bucket: Bucket::Effort,
                },
            ]
        })
        .as_slice()
}

fn select_bucket(model: &str) -> Bucket {
    let model = model.trim();
    for rule in bucket_rules() {
        if rule.pattern.is_match(model) {
            return rule.bucket;
        }
    }
    Bucket::Effort
}

/// Normalize an arbitrary caller-supplied effort value into the canonical set.
/// Mirrors the Python normalization table exactly.
pub fn normalize_canonical_effort(value: Option<&str>) -> CanonicalEffort {
    let raw = match value {
        Some(v) => v.trim().to_ascii_lowercase(),
        None => return CanonicalEffort::Unspecified,
    };
    if raw.is_empty() {
        return CanonicalEffort::Unspecified;
    }
    match raw.as_str() {
        "off" | "disabled" | "false" | "none" | "minimal" => CanonicalEffort::None,
        "low" | "medium" | "high" => CanonicalEffort::High,
        "xhigh" | "max" => CanonicalEffort::XHigh,
        // Unknown effort maps to High (matches Python; a warning there, silent here).
        _ => CanonicalEffort::High,
    }
}

/// Resolve the `reasoning_effort` value to place on the outbound chat request,
/// given the model and the caller's canonical effort. Returns `None` when no
/// reasoning param should be sent (provider_default).
pub fn wire_reasoning_effort(model: &str, effort: CanonicalEffort) -> Option<&'static str> {
    if effort == CanonicalEffort::Unspecified {
        return None;
    }
    match select_bucket(model) {
        Bucket::Effort => Some(effort.wire_str()),
        Bucket::Passthrough => None,
    }
}

// --------------------------------------------------------------------------- #
// Inline `<think>` parsing and explicit reasoning-field extraction.
//
// Mirrors the Python bridge's `reasoning/inline.py` (`split_inline_think`) and
// `reasoning/field.py` (`extract_reasoning_field`). The streaming partial-tag
// helpers (`could_be_partial_think_open`, `trailing_partial_close_len`) and the
// anchored open-match / close-search exposers are used by the inline-think SSE
// state machine.
// --------------------------------------------------------------------------- #

fn think_open_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)<(?:think|thinking)\s*>").expect("valid regex"))
}

fn think_close_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)</(?:think|thinking)\s*>").expect("valid regex"))
}

fn think_tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)</?(?:think|thinking)\s*>").expect("valid regex"))
}

/// Split a complete string that may contain one leading think block into
/// `(reasoning, answer)`. Returns `None` when there is no opening tag — the
/// caller keeps the original text as the answer.
///
/// Semantics match `split_inline_think`: an unterminated open tag consumes the
/// rest as reasoning (answer empty); a matched pair extracts the interior as
/// reasoning and stitches the surrounding text as the answer. Empty reasoning
/// after stripping is reported as `None`.
pub fn split_inline_think(text: &str) -> Option<(String, String)> {
    let open = think_open_re().find(text)?;

    let after_open = open.end();
    match think_close_re().find_at(text, after_open) {
        None => {
            let reasoning = think_tag_re().replace_all(&text[after_open..], "");
            let reasoning = reasoning.trim();
            Some((reasoning.to_owned(), String::new()))
        }
        Some(close) => {
            let reasoning = text[after_open..close.start()].trim();
            let answer = format!("{}{}", &text[..open.start()], &text[close.end()..]);
            Some((reasoning.to_owned(), answer.trim().to_owned()))
        }
    }
}

// --------------------------------------------------------------------------- #
// Streaming inline-think helpers (mirror `reasoning/inline.py`).
//
// Used by the inline-think SSE state machine to detect a `<think>` prefix and
// a `</think>` close tag that may be split across chunk boundaries.
// --------------------------------------------------------------------------- #

const OPEN_TAG: &str = "<think>";
const OPEN_TAG_ALT: &str = "<thinking>";
const OPEN_STEMS: [&str; 2] = ["<think", "<thinking"];

const CLOSE_TAG: &str = "</think>";
const CLOSE_TAG_ALT: &str = "</thinking>";
const CLOSE_STEMS: [&str; 2] = ["</think", "</thinking"];

/// Match an open think tag anchored at the *start* of `buf` (mirrors Python's
/// `THINK_OPEN_RE.match`). Returns the byte offset just past the tag when it
/// matches at position 0, else `None`.
pub fn match_think_open_at_start(buf: &str) -> Option<usize> {
    let m = think_open_re().find(buf)?;
    if m.start() == 0 {
        Some(m.end())
    } else {
        None
    }
}

/// Search for the first close think tag anywhere in `text` (mirrors
/// `THINK_CLOSE_RE.search`). Returns `(start, end)` byte offsets.
pub fn find_think_close(text: &str) -> Option<(usize, usize)> {
    think_close_re().find(text).map(|m| (m.start(), m.end()))
}

/// Return true if `buffer` could still grow into a valid open think tag.
/// Mirrors `could_be_partial_think_open`.
pub fn could_be_partial_think_open(buffer: &str) -> bool {
    let b = buffer.trim_start().to_ascii_lowercase();
    if b.is_empty() {
        return false;
    }
    if OPEN_TAG.starts_with(&b) || OPEN_TAG_ALT.starts_with(&b) {
        return true;
    }
    for stem in OPEN_STEMS {
        if let Some(suffix) = b.strip_prefix(stem) {
            if !suffix.is_empty() && suffix.chars().all(char::is_whitespace) {
                return true;
            }
        }
    }
    false
}

/// Length (in bytes) of the longest trailing run of `text` that could start a
/// close think tag, so a `</think>` split across chunks can be held back.
/// Mirrors `trailing_partial_close_len`. Returns 0 when no suffix qualifies.
pub fn trailing_partial_close_len(text: &str) -> usize {
    let lowered = text.to_ascii_lowercase();
    // Operate on chars to keep suffix slicing on valid boundaries; return the
    // byte length of the qualifying suffix so callers can slice `text`.
    let chars: Vec<char> = lowered.chars().collect();
    let max_len = chars.len().min(CLOSE_TAG_ALT.chars().count());
    for size in (1..=max_len).rev() {
        let candidate: String = chars[chars.len() - size..].iter().collect();
        let qualifies = CLOSE_TAG.starts_with(&candidate)
            || CLOSE_TAG_ALT.starts_with(&candidate)
            || CLOSE_STEMS.iter().any(|stem| {
                candidate.strip_prefix(stem).is_some_and(|suffix| {
                    !suffix.is_empty() && suffix.chars().all(char::is_whitespace)
                })
            });
        if qualifies {
            return candidate.len();
        }
    }
    0
}

/// Return the first non-empty `reasoning_content` / `reasoning` string field
/// from a chat message object, preserving the original bytes (no trimming).
/// Blank/whitespace-only strings are treated as absent.
pub fn extract_reasoning_field(
    message: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    for key in ["reasoning_content", "reasoning"] {
        if let Some(v) = message.get(key).and_then(|v| v.as_str()) {
            if !v.trim().is_empty() {
                return Some(v.to_owned());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_table_matches_python() {
        assert_eq!(
            normalize_canonical_effort(None),
            CanonicalEffort::Unspecified
        );
        assert_eq!(
            normalize_canonical_effort(Some("")),
            CanonicalEffort::Unspecified
        );
        assert_eq!(
            normalize_canonical_effort(Some("off")),
            CanonicalEffort::None
        );
        assert_eq!(
            normalize_canonical_effort(Some("minimal")),
            CanonicalEffort::None
        );
        assert_eq!(
            normalize_canonical_effort(Some("low")),
            CanonicalEffort::High
        );
        assert_eq!(
            normalize_canonical_effort(Some("high")),
            CanonicalEffort::High
        );
        assert_eq!(
            normalize_canonical_effort(Some("max")),
            CanonicalEffort::XHigh
        );
        assert_eq!(
            normalize_canonical_effort(Some("turbo")),
            CanonicalEffort::High
        );
    }

    #[test]
    fn kimi_never_sends_effort() {
        assert_eq!(
            wire_reasoning_effort("moonshot/kimi-k2", CanonicalEffort::High),
            None
        );
        assert_eq!(
            wire_reasoning_effort("deepseek-v3", CanonicalEffort::High),
            Some("high")
        );
        assert_eq!(
            wire_reasoning_effort("gpt-4o", CanonicalEffort::XHigh),
            Some("xhigh")
        );
        assert_eq!(
            wire_reasoning_effort("gpt-4o", CanonicalEffort::Unspecified),
            None
        );
    }
}
