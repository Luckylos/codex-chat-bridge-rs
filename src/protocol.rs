//! Typed discriminants for the Responses protocol shapes the converter
//! *dispatches on*.
//!
//! These are **classifier enums**, not serde-modeled structs: they turn the
//! `type` tag of an otherwise-`Value` item into a closed set the compiler can
//! check, so every dispatch site is an exhaustive `match` (a typo can't
//! compile; adding a protocol variant forces a decision at each branch). The
//! item *payload* stays `serde_json::Value` — lenient, byte-exact reads are
//! preserved and nothing round-trips through a stricter typed decode.
//!
//! This extends the house style already set by `convert::ResponseStatus`
//! (classify → enum → exhaustive match); it does not introduce a new
//! serde-deserialization layer.

/// The kind of a top-level Responses **input item**, as dispatched by
/// `append_input_items`. `Other` covers any unrecognized tag (routed to the
/// generic-message path or a transform-loss skip by the caller); the raw tag
/// string is still read from the item for the loss metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputItemKind {
    Reasoning,
    /// Text-like content lifted to a user message
    /// (`input_text` / `output_text` / `text` / `latest_reminder`).
    Text,
    Image,
    Audio,
    FunctionCall,
    CustomToolCall,
    ToolSearchCall,
    FunctionCallOutput,
    CustomToolCallOutput,
    ToolSearchOutput,
    Message,
    Other,
}

impl InputItemKind {
    pub(crate) fn classify(item_type: &str) -> Self {
        match item_type {
            "reasoning" => Self::Reasoning,
            "input_text" | "output_text" | "text" | "latest_reminder" => Self::Text,
            "input_image" => Self::Image,
            "input_audio" => Self::Audio,
            "function_call" => Self::FunctionCall,
            "custom_tool_call" => Self::CustomToolCall,
            "tool_search_call" => Self::ToolSearchCall,
            "function_call_output" => Self::FunctionCallOutput,
            "custom_tool_call_output" => Self::CustomToolCallOutput,
            "tool_search_output" => Self::ToolSearchOutput,
            "message" => Self::Message,
            _ => Self::Other,
        }
    }

    /// The three input-item kinds that are accumulated as pending tool calls.
    pub(crate) fn as_tool_call(self) -> Option<ToolCallKind> {
        match self {
            Self::FunctionCall => Some(ToolCallKind::Function),
            Self::CustomToolCall => Some(ToolCallKind::Custom),
            Self::ToolSearchCall => Some(ToolCallKind::Search),
            _ => None,
        }
    }
}

/// The three tool-call input-item kinds, as dispatched by
/// `handle_tool_call_item`. Split out so that handler's dispatch is exhaustive
/// over exactly the calls it accepts (no `_ => unreachable`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolCallKind {
    Function,
    Custom,
    Search,
}

impl ToolCallKind {
    /// The exact protocol `type` string — used as the transform-loss metric
    /// label so telemetry is byte-identical to the pre-refactor dispatch.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function_call",
            Self::Custom => "custom_tool_call",
            Self::Search => "tool_search_call",
        }
    }
}

/// The kind of a message **content part**, as dispatched by the content
/// extraction/normalization helpers. Centralizes the
/// `input_text | output_text | text` set that was duplicated as string
/// literals across several sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContentPartType {
    InputText,
    OutputText,
    Text,
    Refusal,
    Other,
}

impl ContentPartType {
    pub(crate) fn classify(part_type: &str) -> Self {
        match part_type {
            "input_text" => Self::InputText,
            "output_text" => Self::OutputText,
            "text" => Self::Text,
            "refusal" => Self::Refusal,
            _ => Self::Other,
        }
    }

    /// Any text-carrying part (`input_text` / `output_text` / `text`). The
    /// single source of truth for the set that was previously an inline
    /// `matches!(typ, "input_text" | "output_text" | "text")` at 3+ call sites.
    pub(crate) fn is_text(self) -> bool {
        matches!(self, Self::InputText | Self::OutputText | Self::Text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_item_kind_classifies_every_known_tag() {
        for (tag, kind) in [
            ("reasoning", InputItemKind::Reasoning),
            ("input_text", InputItemKind::Text),
            ("output_text", InputItemKind::Text),
            ("text", InputItemKind::Text),
            ("latest_reminder", InputItemKind::Text),
            ("input_image", InputItemKind::Image),
            ("input_audio", InputItemKind::Audio),
            ("function_call", InputItemKind::FunctionCall),
            ("custom_tool_call", InputItemKind::CustomToolCall),
            ("tool_search_call", InputItemKind::ToolSearchCall),
            ("function_call_output", InputItemKind::FunctionCallOutput),
            (
                "custom_tool_call_output",
                InputItemKind::CustomToolCallOutput,
            ),
            ("tool_search_output", InputItemKind::ToolSearchOutput),
            ("message", InputItemKind::Message),
        ] {
            assert_eq!(InputItemKind::classify(tag), kind, "tag={tag}");
        }
    }

    #[test]
    fn input_item_kind_unknown_and_empty_are_other() {
        assert_eq!(InputItemKind::classify(""), InputItemKind::Other);
        assert_eq!(InputItemKind::classify("wat"), InputItemKind::Other);
    }

    #[test]
    fn tool_call_kinds_round_trip_through_tag() {
        for kind in [
            ToolCallKind::Function,
            ToolCallKind::Custom,
            ToolCallKind::Search,
        ] {
            assert_eq!(
                InputItemKind::classify(kind.as_str()).as_tool_call(),
                Some(kind)
            );
        }
    }

    #[test]
    fn non_tool_call_kinds_have_no_tool_call() {
        assert_eq!(InputItemKind::Reasoning.as_tool_call(), None);
        assert_eq!(InputItemKind::Message.as_tool_call(), None);
        assert_eq!(InputItemKind::FunctionCallOutput.as_tool_call(), None);
    }

    #[test]
    fn content_part_is_text_covers_exactly_the_three() {
        assert!(ContentPartType::classify("input_text").is_text());
        assert!(ContentPartType::classify("output_text").is_text());
        assert!(ContentPartType::classify("text").is_text());
        assert!(!ContentPartType::classify("refusal").is_text());
        assert!(!ContentPartType::classify("image_url").is_text());
    }
}
