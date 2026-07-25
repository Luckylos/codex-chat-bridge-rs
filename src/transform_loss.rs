//! Responses→Chat conversion loss taxonomy and collector.
//!
//! Protocol conversion is permissive: malformed, duplicate, or unconvertible
//! input items are dropped or downgraded rather than rejected. That keeps the
//! bridge robust, but silent degradation is invisible in production. This
//! module makes each degradation explicit and testable — the conversion path
//! records a [`TransformLoss`] event per drop/downgrade into a
//! [`TransformLossCollector`], which the request handler drains into the
//! `bridge_transform_loss_total` metric and a warning log.
//!

/// A category of data loss or degradation during Responses→Chat conversion.
///
/// The variant names are the metric `kind` label, so they must stay stable
/// (they match the Python `ProviderTransformLoss` enum member names verbatim).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformLoss {
    /// Input item was not a JSON object (e.g. number, null) and was skipped.
    SkippedNonDictItem,
    /// Input item had an unrecognized `type`; dropped permissively.
    SkippedUnknownItemType,
    /// `input_image` item could not be converted (unsupported format/URL).
    SkippedUnsupportedImage,
    /// `input_audio` item could not be converted (unsupported format/URL).
    SkippedUnsupportedAudio,
    /// Tool call/output item duplicated one already in the message history.
    SkippedDuplicateToolCall,
    /// Tool output had no matching preceding tool call; downgraded to a user
    /// message to avoid Chat Completions rejecting an orphan tool message.
    DowngradedOrphanToolOutput,
    /// Reasoning item produced empty text after stripping; dropped.
    DroppedEmptyReasoning,
    /// Generic message item had an unrecognized role; downgraded to `user`.
    DowngradedInvalidRole,
}

impl TransformLoss {
    /// The stable string used as the metric `kind` label. Matches the Python
    /// enum member `.name`.
    pub fn name(self) -> &'static str {
        match self {
            Self::SkippedNonDictItem => "SkippedNonDictItem",
            Self::SkippedUnknownItemType => "SkippedUnknownItemType",
            Self::SkippedUnsupportedImage => "SkippedUnsupportedImage",
            Self::SkippedUnsupportedAudio => "SkippedUnsupportedAudio",
            Self::SkippedDuplicateToolCall => "SkippedDuplicateToolCall",
            Self::DowngradedOrphanToolOutput => "DowngradedOrphanToolOutput",
            Self::DroppedEmptyReasoning => "DroppedEmptyReasoning",
            Self::DowngradedInvalidRole => "DowngradedInvalidRole",
        }
    }
}

/// A single observed transform-loss event.
///
/// `item_type` is the offending item's `type` field when present; `None` maps
/// to the `"none"` metric label. `reason` is a human-readable explanation for
/// the warning log.
#[derive(Debug, Clone)]
pub struct TransformLossEvent {
    pub kind: TransformLoss,
    pub item_type: Option<String>,
    pub reason: String,
}

/// Accumulates [`TransformLossEvent`]s during a single conversion pass.
///
/// A single collector threads through `append_input_items`; call sites record
/// unconditionally. The handler drains it once after conversion. Unlike the
/// Python version there is no null-object variant — an empty `Vec` allocates
/// nothing until the first `record`, so the common lossless path is free.
#[derive(Debug, Default)]
pub struct TransformLossCollector {
    events: Vec<TransformLossEvent>,
}

impl TransformLossCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one loss event.
    pub fn record(
        &mut self,
        kind: TransformLoss,
        item_type: Option<&str>,
        reason: impl Into<String>,
    ) {
        self.events.push(TransformLossEvent {
            kind,
            item_type: item_type.map(str::to_owned),
            reason: reason.into(),
        });
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn events(&self) -> &[TransformLossEvent] {
        &self.events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_name_matches_python_member_names() {
        assert_eq!(
            TransformLoss::SkippedNonDictItem.name(),
            "SkippedNonDictItem"
        );
        assert_eq!(
            TransformLoss::DowngradedOrphanToolOutput.name(),
            "DowngradedOrphanToolOutput"
        );
        assert_eq!(
            TransformLoss::DowngradedInvalidRole.name(),
            "DowngradedInvalidRole"
        );
    }

    #[test]
    fn collector_starts_empty_and_accumulates() {
        let mut c = TransformLossCollector::new();
        assert!(c.is_empty());
        c.record(
            TransformLoss::SkippedUnknownItemType,
            Some("mystery"),
            "unrecognized",
        );
        c.record(TransformLoss::DroppedEmptyReasoning, None, "empty");
        assert!(!c.is_empty());
        assert_eq!(c.events().len(), 2);
        assert_eq!(c.events()[0].item_type.as_deref(), Some("mystery"));
        assert_eq!(c.events()[1].item_type, None);
    }
}
