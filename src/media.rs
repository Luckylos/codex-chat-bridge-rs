//! Media input conversion (`input_image` / `input_audio`) with SSRF guards.
//!
//! Top-level Responses `input_image` / `input_audio` items convert to Chat
//! Completions `image_url` / `input_audio` content parts. Before a URL is
//! forwarded upstream it passes an SSRF check: only `https://` to a public host
//! or an inline `data:` URI (constrained to the matching media MIME prefix) is
//! allowed. `http://`, `file://`, and `https://` pointing at loopback/private/
//! link-local/reserved IPs (cloud-metadata exfiltration vectors) are rejected.
//!
//! Mirrors the Python bridge's `responses_to_chat/media.py`.

use std::net::IpAddr;

use serde_json::{json, Map, Value};

/// Audio formats accepted when synthesizing a `data:` URI, so an attacker
/// cannot inject an arbitrary MIME type through the `format` field.
const ALLOWED_AUDIO_FORMATS: &[&str] = &["wav", "mp3", "flac", "ogg", "m4a", "aac"];

/// Outcome of converting a media item: either a Chat content part, or a
/// rejection reason to be recorded as transform loss by the caller.
pub enum MediaConversion {
    Part(Value),
    Rejected(String),
}

/// Return true for hosts that must not be reachable through the bridge.
///
/// Blocks loopback, private, link-local (incl. cloud metadata 169.254.169.254),
/// and unspecified addresses. Non-IP-literal hostnames are allowed — the bridge
/// cannot safely resolve them here and defers to the upstream's egress policy
/// (DNS-rebinding defense is out of scope for a static scheme/host check).
fn is_blocked_host(host: &str) -> bool {
    if host.is_empty() {
        return true;
    }
    // Strip IPv6 literal brackets before parsing.
    let candidate = host.trim_start_matches('[').trim_end_matches(']');
    let Ok(ip) = candidate.parse::<IpAddr>() else {
        // Not an IP literal (a DNS name) — allow.
        return false;
    };
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast()
                || v4.is_documentation()
                // Carrier-grade NAT / shared address space 100.64.0.0/10.
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xC0) == 0x40)
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                // Unique-local fc00::/7.
                || (v6.segments()[0] & 0xFE00) == 0xFC00
                // Link-local fe80::/10.
                || (v6.segments()[0] & 0xFFC0) == 0xFE80
        }
    }
}

/// Whether a media URL is safe to forward upstream.
///
/// Allowed: `https://` to a non-blocked host, or a `data:` URI (optionally
/// constrained to a MIME prefix). Everything else is rejected.
fn is_safe_media_url(url: &str, allowed_data_prefix: Option<&str>) -> bool {
    if url.is_empty() {
        return false;
    }
    if url.starts_with("data:") {
        return match allowed_data_prefix {
            Some(prefix) => url.starts_with(prefix),
            None => true,
        };
    }
    let Some(after_scheme) = url.strip_prefix("https://") else {
        return false;
    };
    let host = host_from_authority(after_scheme);
    !is_blocked_host(host)
}

/// Extract the host from the authority portion of a `https://` URL body
/// (everything after the scheme). Strips userinfo and port, and preserves an
/// IPv6 literal's brackets for `is_blocked_host` to trim.
fn host_from_authority(after_scheme: &str) -> &str {
    // Authority ends at the first '/', '?', or '#'.
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    // Drop userinfo.
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    // IPv6 literal: host is inside brackets; a port may follow "]".
    if let Some(inner) = authority.strip_prefix('[') {
        return match inner.split_once(']') {
            Some((h, _)) => h,
            None => inner,
        };
    }
    // IPv4/hostname: strip a trailing ":port".
    authority.split(':').next().unwrap_or(authority)
}

fn is_safe_image_url(url: &str) -> bool {
    is_safe_media_url(url, Some("data:image/"))
}

fn is_safe_audio_url(url: &str) -> bool {
    is_safe_media_url(url, Some("data:audio/"))
}

/// Convert a Responses `input_image` item to a Chat `image_url` content part.
///
/// Accepts either a flat `image_url` string or a nested `{ "url": ... }` object
/// (whose extra keys, e.g. `detail`, are preserved). The URL must pass the SSRF
/// check. A top-level `detail` field is folded into the payload when absent.
pub fn image_part_from_input_item(item: &Map<String, Value>) -> MediaConversion {
    let (url, mut payload) = match item.get("image_url") {
        Some(Value::String(s)) if !s.is_empty() => (s.clone(), Map::new()),
        Some(Value::Object(obj)) => match obj.get("url").and_then(Value::as_str) {
            Some(url) if !url.is_empty() => (url.to_owned(), obj.clone()),
            _ => return MediaConversion::Rejected("input_image missing url".to_owned()),
        },
        _ => return MediaConversion::Rejected("input_image missing url".to_owned()),
    };

    if !is_safe_image_url(&url) {
        return MediaConversion::Rejected(format!(
            "Rejected unsafe image URL (only https:// to public hosts and data:image/ allowed): {}",
            truncate(&url, 60)
        ));
    }

    payload.insert("url".to_owned(), json!(url));
    if let Some(detail) = item.get("detail").and_then(Value::as_str) {
        if !detail.is_empty() {
            payload
                .entry("detail".to_owned())
                .or_insert_with(|| json!(detail));
        }
    }
    MediaConversion::Part(json!({ "type": "image_url", "image_url": Value::Object(payload) }))
}

/// Merge the flat and nested `input_audio` shapes into one `(url, data, format)`
/// source. Nested `input_audio` keys take precedence; flat keys are fallback.
fn audio_source(item: &Map<String, Value>) -> (Option<String>, Option<String>, Option<String>) {
    let mut url = item
        .get("audio_url")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mut data = item.get("data").and_then(Value::as_str).map(str::to_owned);
    let mut format = item
        .get("format")
        .and_then(Value::as_str)
        .map(str::to_owned);

    if let Some(nested) = item.get("input_audio").and_then(Value::as_object) {
        if let Some(v) = nested.get("url").and_then(Value::as_str) {
            url = Some(v.to_owned());
        }
        if let Some(v) = nested.get("data").and_then(Value::as_str) {
            data = Some(v.to_owned());
        }
        if let Some(v) = nested.get("format").and_then(Value::as_str) {
            format = Some(v.to_owned());
        }
    }
    (url, data, format)
}

/// Convert a Responses `input_audio` item to a Chat `input_audio` content part.
///
/// Accepts the nested (`{ "input_audio": {...} }`) and flat
/// (`{ "data": ..., "format": ... }` / `{ "audio_url": ... }`) shapes:
///   - a URL string → `{ "input_audio": { "url": ... } }` (SSRF-checked);
///   - a pre-built `data:` URI in `data` → passed through unchanged;
///   - base64 `data` + `format` → `{ "input_audio": { "data": "data:audio/…" } }`
///     with the format constrained to the allowed set.
pub fn audio_part_from_input_item(item: &Map<String, Value>) -> MediaConversion {
    let (url, data, format) = audio_source(item);

    if let Some(url) = url.filter(|s| !s.is_empty()) {
        if !is_safe_audio_url(&url) {
            return MediaConversion::Rejected(format!(
                "Rejected unsafe audio URL (only https:// to public hosts and data:audio/ allowed): {}",
                truncate(&url, 60)
            ));
        }
        return MediaConversion::Part(json!({
            "type": "input_audio",
            "input_audio": { "url": url },
        }));
    }

    if let Some(data) = data.filter(|s| !s.is_empty()) {
        if data.starts_with("data:") {
            return MediaConversion::Part(json!({
                "type": "input_audio",
                "input_audio": { "data": data },
            }));
        }
        let fmt = format
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("wav")
            .to_ascii_lowercase();
        if !ALLOWED_AUDIO_FORMATS.contains(&fmt.as_str()) {
            return MediaConversion::Rejected(format!(
                "Rejected unsupported audio format: {fmt:?} (allowed: {ALLOWED_AUDIO_FORMATS:?})"
            ));
        }
        return MediaConversion::Part(json!({
            "type": "input_audio",
            "input_audio": { "data": format!("data:audio/{fmt};base64,{data}") },
        }));
    }

    MediaConversion::Rejected("input_audio missing url and data".to_owned())
}

/// Truncate a string to at most `max` chars for inclusion in a loss reason,
/// without splitting a multi-byte char.
fn truncate(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(conv: MediaConversion) -> Value {
        match conv {
            MediaConversion::Part(v) => v,
            MediaConversion::Rejected(r) => panic!("expected part, got rejection: {r}"),
        }
    }

    fn reason(conv: MediaConversion) -> String {
        match conv {
            MediaConversion::Rejected(r) => r,
            MediaConversion::Part(v) => panic!("expected rejection, got part: {v}"),
        }
    }

    // ---- SSRF host classification ----

    #[test]
    fn blocks_loopback_private_linklocal_metadata() {
        assert!(is_blocked_host("127.0.0.1"));
        assert!(is_blocked_host("10.0.0.5"));
        assert!(is_blocked_host("192.168.1.1"));
        assert!(is_blocked_host("172.16.0.1"));
        assert!(is_blocked_host("169.254.169.254")); // cloud metadata
        assert!(is_blocked_host("0.0.0.0"));
        assert!(is_blocked_host("100.64.0.1")); // CGNAT
        assert!(is_blocked_host("[::1]"));
        assert!(is_blocked_host("[fe80::1]"));
        assert!(is_blocked_host("[fc00::1]"));
        assert!(is_blocked_host(""));
    }

    #[test]
    fn allows_public_hosts_and_dns_names() {
        assert!(!is_blocked_host("93.184.216.34")); // example.com
        assert!(!is_blocked_host("example.com"));
        assert!(!is_blocked_host("cdn.openai.com"));
        assert!(!is_blocked_host("[2606:2800:220:1::1]"));
    }

    #[test]
    fn host_extraction_strips_port_userinfo_path() {
        assert_eq!(host_from_authority("example.com/path"), "example.com");
        assert_eq!(host_from_authority("example.com:443/x"), "example.com");
        assert_eq!(host_from_authority("user:pw@example.com/x"), "example.com");
        assert_eq!(host_from_authority("[::1]:8443/x"), "::1");
        assert_eq!(host_from_authority("host?q=1"), "host");
    }

    // ---- URL safety ----

    #[test]
    fn image_url_safety_rules() {
        assert!(is_safe_image_url("https://cdn.example.com/a.png"));
        assert!(is_safe_image_url("data:image/png;base64,AAAA"));
        assert!(!is_safe_image_url("data:audio/wav;base64,AAAA")); // wrong prefix
        assert!(!is_safe_image_url("http://example.com/a.png")); // not https
        assert!(!is_safe_image_url("https://127.0.0.1/a.png")); // SSRF
        assert!(!is_safe_image_url(
            "https://169.254.169.254/latest/meta-data"
        ));
        assert!(!is_safe_image_url("file:///etc/passwd"));
        assert!(!is_safe_image_url(""));
    }

    #[test]
    fn audio_url_safety_rules() {
        assert!(is_safe_audio_url("https://cdn.example.com/a.mp3"));
        assert!(is_safe_audio_url("data:audio/wav;base64,AAAA"));
        assert!(!is_safe_audio_url("data:image/png;base64,AAAA"));
        assert!(!is_safe_audio_url("https://10.0.0.1/a.mp3"));
    }

    // ---- image conversion ----

    #[test]
    fn image_flat_url_builds_part() {
        let item = json!({ "type": "input_image", "image_url": "https://ex.com/a.png" });
        let p = part(image_part_from_input_item(item.as_object().unwrap()));
        assert_eq!(p["type"], json!("image_url"));
        assert_eq!(p["image_url"]["url"], json!("https://ex.com/a.png"));
    }

    #[test]
    fn image_nested_object_preserves_detail() {
        let item = json!({
            "type": "input_image",
            "image_url": { "url": "https://ex.com/a.png", "detail": "high" },
        });
        let p = part(image_part_from_input_item(item.as_object().unwrap()));
        assert_eq!(p["image_url"]["detail"], json!("high"));
    }

    #[test]
    fn image_top_level_detail_folded_in() {
        let item = json!({
            "type": "input_image",
            "image_url": "https://ex.com/a.png",
            "detail": "low",
        });
        let p = part(image_part_from_input_item(item.as_object().unwrap()));
        assert_eq!(p["image_url"]["detail"], json!("low"));
    }

    #[test]
    fn image_unsafe_url_rejected() {
        let item = json!({ "type": "input_image", "image_url": "http://127.0.0.1/a.png" });
        let r = reason(image_part_from_input_item(item.as_object().unwrap()));
        assert!(r.contains("unsafe image URL"));
    }

    #[test]
    fn image_missing_url_rejected() {
        let item = json!({ "type": "input_image" });
        let r = reason(image_part_from_input_item(item.as_object().unwrap()));
        assert!(r.contains("missing url"));
    }

    // ---- audio conversion ----

    #[test]
    fn audio_url_builds_part() {
        let item = json!({ "type": "input_audio", "audio_url": "https://ex.com/a.mp3" });
        let p = part(audio_part_from_input_item(item.as_object().unwrap()));
        assert_eq!(p["input_audio"]["url"], json!("https://ex.com/a.mp3"));
    }

    #[test]
    fn audio_nested_shape_takes_precedence() {
        let item = json!({
            "type": "input_audio",
            "format": "mp3",
            "input_audio": { "data": "QUJD", "format": "flac" },
        });
        let p = part(audio_part_from_input_item(item.as_object().unwrap()));
        assert_eq!(
            p["input_audio"]["data"],
            json!("data:audio/flac;base64,QUJD")
        );
    }

    #[test]
    fn audio_prebuilt_data_uri_passthrough() {
        let item = json!({
            "type": "input_audio",
            "data": "data:audio/wav;base64,QUJD",
        });
        let p = part(audio_part_from_input_item(item.as_object().unwrap()));
        assert_eq!(
            p["input_audio"]["data"],
            json!("data:audio/wav;base64,QUJD")
        );
    }

    #[test]
    fn audio_base64_with_format_synthesizes_data_uri() {
        let item = json!({ "type": "input_audio", "data": "QUJD", "format": "mp3" });
        let p = part(audio_part_from_input_item(item.as_object().unwrap()));
        assert_eq!(
            p["input_audio"]["data"],
            json!("data:audio/mp3;base64,QUJD")
        );
    }

    #[test]
    fn audio_base64_defaults_format_to_wav() {
        let item = json!({ "type": "input_audio", "data": "QUJD" });
        let p = part(audio_part_from_input_item(item.as_object().unwrap()));
        assert_eq!(
            p["input_audio"]["data"],
            json!("data:audio/wav;base64,QUJD")
        );
    }

    #[test]
    fn audio_unsupported_format_rejected() {
        let item = json!({ "type": "input_audio", "data": "QUJD", "format": "exe" });
        let r = reason(audio_part_from_input_item(item.as_object().unwrap()));
        assert!(r.contains("unsupported audio format"));
    }

    #[test]
    fn audio_unsafe_url_rejected() {
        let item = json!({ "type": "input_audio", "audio_url": "https://192.168.0.1/a.mp3" });
        let r = reason(audio_part_from_input_item(item.as_object().unwrap()));
        assert!(r.contains("unsafe audio URL"));
    }

    #[test]
    fn audio_missing_everything_rejected() {
        let item = json!({ "type": "input_audio" });
        let r = reason(audio_part_from_input_item(item.as_object().unwrap()));
        assert!(r.contains("missing url and data"));
    }
}
