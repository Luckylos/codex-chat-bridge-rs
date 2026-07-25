//! Shared text sanitization.
//!
//! Mirrors the Python bridge's `text_utils.sanitize_string`: strip NUL and
//! control characters below `0x20`, but preserve `\n`, `\r`, `\t`. Both
//! conversion directions route text/reasoning/call-id/tool-name strings
//! through here before they reach the upstream or the client, so raw control
//! characters cannot cause upstream injection or parser confusion.

/// Remove control characters `< 0x20` except newline/carriage-return/tab.
///
/// Returns the input unchanged when it contains no such characters, so the
/// common clean-string case allocates nothing.
pub fn sanitize_string(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    // Fast path: nothing to strip.
    if !value
        .chars()
        .any(|ch| (ch as u32) < 0x20 && !matches!(ch, '\n' | '\r' | '\t'))
    {
        return value.to_owned();
    }
    value
        .chars()
        .filter(|&ch| ch >= ' ' || matches!(ch, '\n' | '\r' | '\t'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_whitespace_controls() {
        assert_eq!(sanitize_string("a\nb\tc\rd"), "a\nb\tc\rd");
    }

    #[test]
    fn strips_low_controls_and_nul() {
        assert_eq!(sanitize_string("a\u{0}b\u{1}c\u{1f}d"), "abcd");
    }

    #[test]
    fn clean_string_unchanged() {
        assert_eq!(sanitize_string("hello world"), "hello world");
        assert_eq!(sanitize_string(""), "");
    }
}
