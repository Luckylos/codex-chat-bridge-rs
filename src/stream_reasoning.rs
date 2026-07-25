//! Reasoning increment state machine for the streaming path.
//!
//! Mirrors the Python bridge's `stream_state/reasoning.py`. Tracks a single
//! reasoning summary block across the stream:
//!
//! * the first delta lazily emits `output_item.added` (a `reasoning` item in
//!   `in_progress`) plus an empty `reasoning_summary_part.added`;
//! * each subsequent delta accumulates text and emits a
//!   `reasoning_summary_text.delta`;
//! * `finalize` closes the block with the text/part `.done` events and an
//!   `output_item.done`, and registers the completed item with the envelope so
//!   the terminal response carries it in output order.
//!
//! Summary index is always 0 — the bridge collapses reasoning into one summary
//! block, matching the non-streaming renderer in `convert::chat_to_responses`.
//!
//! Driven by the top-level stream orchestrator that lands in a later layer, so
//! the state machine reads as dead until it wires in. Tests lock in the event
//! sequence now.

use serde_json::json;

use crate::stream_envelope::ResponseEnvelopeState;
use crate::stream_events;

/// Streaming reasoning accumulator. One instance per response stream.
/// Lifecycle of the reasoning output item. The transitions are strictly
/// forward (`NotStarted → Open → Done`); collapsing the former
/// `item_added`/`done` boolean pair into one enum makes the nonsensical
/// "done but never started" combination unrepresentable.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Lifecycle {
    #[default]
    NotStarted,
    Open,
    Done,
}

#[derive(Debug, Default)]
pub struct ReasoningState {
    text: String,
    lifecycle: Lifecycle,
    output_index: i64,
}

impl ReasoningState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulated reasoning text seen so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Lazily open the reasoning item on first delta. Idempotent.
    fn ensure_started(&mut self, envelope: &mut ResponseEnvelopeState) -> Vec<Vec<u8>> {
        if self.lifecycle != Lifecycle::NotStarted {
            return Vec::new();
        }
        self.lifecycle = Lifecycle::Open;
        self.output_index = envelope.allocate_output_index();
        let item_id = envelope.reasoning_item_id();
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
        });
        vec![
            stream_events::output_item_added(self.output_index, item),
            stream_events::reasoning_summary_part_added(
                &item_id,
                self.output_index,
                0,
                json!({ "type": "summary_text", "text": "" }),
            ),
        ]
    }

    /// Push a reasoning delta. Opens the item on first call, then emits a
    /// summary-text delta. A no-op once the block is finalized.
    pub fn push_delta(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        delta: &str,
    ) -> Vec<Vec<u8>> {
        if self.lifecycle == Lifecycle::Done {
            return Vec::new();
        }
        let mut events = self.ensure_started(envelope);
        self.text.push_str(delta);
        let item_id = envelope.reasoning_item_id();
        events.push(stream_events::reasoning_summary_text_delta(
            &item_id,
            self.output_index,
            0,
            delta,
        ));
        events
    }

    /// Close the reasoning block. Emits text/part `.done` + `output_item.done`
    /// and registers the completed item with the envelope. A no-op if the block
    /// was never opened or is already done.
    pub fn finalize(&mut self, envelope: &mut ResponseEnvelopeState) -> Vec<Vec<u8>> {
        if self.lifecycle != Lifecycle::Open {
            return Vec::new();
        }
        self.lifecycle = Lifecycle::Done;
        let item_id = envelope.reasoning_item_id();
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": self.text }],
        });
        let summary_part = json!({ "type": "summary_text", "text": self.text });
        envelope.append_completed_item(self.output_index, item.clone());
        vec![
            stream_events::reasoning_summary_text_done(&item_id, self.output_index, 0, &self.text),
            stream_events::reasoning_summary_part_done(
                &item_id,
                self.output_index,
                0,
                summary_part,
            ),
            stream_events::output_item_done(self.output_index, item),
        ]
    }

    /// The reasoning text to attach to a tool call, if the block is still open.
    /// Trimmed (reasoning is structural, not user-visible). Empty once done.
    pub fn active_text_for_tools(&self) -> String {
        if self.text.is_empty() || self.lifecycle == Lifecycle::Done {
            String::new()
        } else {
            self.text.trim().to_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn parse_event(bytes: &[u8]) -> (String, Value) {
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let mut event = String::new();
        let mut data = String::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                event = rest.to_owned();
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data = rest.to_owned();
            }
        }
        (event, serde_json::from_str(&data).unwrap())
    }

    fn env() -> ResponseEnvelopeState {
        ResponseEnvelopeState::new(Some("resp_bridge_abc"))
    }

    #[test]
    fn first_delta_opens_item_and_summary_part() {
        let mut e = env();
        let mut r = ReasoningState::new();
        let events = r.push_delta(&mut e, "hel");
        assert_eq!(events.len(), 3);
        let (ev0, d0) = parse_event(&events[0]);
        assert_eq!(ev0, "response.output_item.added");
        assert_eq!(d0["item"]["type"], json!("reasoning"));
        assert_eq!(d0["item"]["status"], json!("in_progress"));
        assert_eq!(d0["output_index"], json!(0));
        let (ev1, d1) = parse_event(&events[1]);
        assert_eq!(ev1, "response.reasoning_summary_part.added");
        assert_eq!(d1["part"]["text"], json!(""));
        let (ev2, d2) = parse_event(&events[2]);
        assert_eq!(ev2, "response.reasoning_summary_text.delta");
        assert_eq!(d2["delta"], json!("hel"));
        assert_eq!(d2["summary_index"], json!(0));
    }

    #[test]
    fn subsequent_delta_only_emits_text_delta() {
        let mut e = env();
        let mut r = ReasoningState::new();
        r.push_delta(&mut e, "hel");
        let events = r.push_delta(&mut e, "lo");
        assert_eq!(events.len(), 1);
        let (ev, d) = parse_event(&events[0]);
        assert_eq!(ev, "response.reasoning_summary_text.delta");
        assert_eq!(d["delta"], json!("lo"));
        assert_eq!(r.text(), "hello");
    }

    #[test]
    fn finalize_closes_with_done_events_and_registers_item() {
        let mut e = env();
        let mut r = ReasoningState::new();
        r.push_delta(&mut e, "think");
        let events = r.finalize(&mut e);
        assert_eq!(events.len(), 3);
        let (ev0, d0) = parse_event(&events[0]);
        assert_eq!(ev0, "response.reasoning_summary_text.done");
        assert_eq!(d0["text"], json!("think"));
        let (ev1, _) = parse_event(&events[1]);
        assert_eq!(ev1, "response.reasoning_summary_part.done");
        let (ev2, d2) = parse_event(&events[2]);
        assert_eq!(ev2, "response.output_item.done");
        assert_eq!(d2["item"]["summary"][0]["text"], json!("think"));
        // The completed item is registered with the envelope in output order.
        assert_eq!(e.completed_output_items()[0]["type"], json!("reasoning"));
    }

    #[test]
    fn finalize_without_start_is_noop() {
        let mut e = env();
        let mut r = ReasoningState::new();
        assert!(r.finalize(&mut e).is_empty());
    }

    #[test]
    fn delta_after_finalize_is_noop() {
        let mut e = env();
        let mut r = ReasoningState::new();
        r.push_delta(&mut e, "x");
        r.finalize(&mut e);
        assert!(r.push_delta(&mut e, "y").is_empty());
        assert_eq!(r.text(), "x");
    }

    #[test]
    fn active_text_for_tools_trims_while_open_empty_once_done() {
        let mut e = env();
        let mut r = ReasoningState::new();
        r.push_delta(&mut e, "  spaced  ");
        assert_eq!(r.active_text_for_tools(), "spaced");
        r.finalize(&mut e);
        assert_eq!(r.active_text_for_tools(), "");
    }
}
