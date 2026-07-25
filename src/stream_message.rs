//! Message increment state machine for the streaming path.
//!
//! Mirrors the Python bridge's `stream_state/message.py`. Tracks the assistant
//! `message` output item across the stream — its ordered content parts (text
//! segments and refusals) and per-segment annotations:
//!
//! * the first text/refusal lazily emits `output_item.added` (a `message` item
//!   in `in_progress`);
//! * each text segment emits `content_part.added` when opened and accumulates
//!   `output_text.delta`s; `flush_open_text_part` (or `finalize`) closes it with
//!   `output_text.done` + `content_part.done`;
//! * a refusal is emitted as an immediately-closed part (`content_part.added` +
//!   `content_part.done`);
//! * `finalize` closes every open text part, then emits `output_item.done` and
//!   registers the completed `message` item with the envelope in output order.
//!
//! Annotations arriving before a text segment is open are held pending and
//! attached to the next segment; annotations arriving mid-segment extend the
//! open one. `content_index` is the part's position in the content array.
//!
//! Python keeps a parallel `parts` list beside `segments`; because the two grow
//! in lockstep (one part per segment) the Rust model derives the content parts
//! from `segments` directly, dropping the duplicate.
//!
//! Driven by the top-level stream orchestrator that lands in a later layer, so
//! the state machine reads as dead until it wires in. Tests lock in the event
//! sequence now.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::stream_envelope::ResponseEnvelopeState;
use crate::stream_events;

/// One content part of the assistant message, in emission order.
enum Segment {
    Text {
        content_index: usize,
        text: String,
        annotations: Vec<Value>,
    },
    Refusal {
        refusal: String,
    },
}

/// Lifecycle of the `message` output item. Transitions run strictly forward
/// (`NotStarted → Open → Done`); folding the former `item_added`/`item_done`
/// boolean pair into one enum makes the nonsensical "done but never started"
/// combination unrepresentable.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Lifecycle {
    #[default]
    NotStarted,
    Open,
    Done,
}

#[derive(Default)]
pub struct MessageState {
    segments: Vec<Segment>,
    lifecycle: Lifecycle,
    output_index: Option<i64>,
    /// content_index values whose text part has already been closed.
    text_part_done: HashSet<usize>,
    /// Annotations seen before a text segment is open, attached to the next one.
    pending_annotations: Vec<Value>,
}

impl MessageState {
    pub fn new() -> Self {
        Self::default()
    }

    /// The concatenated text of all output_text segments (for session persist).
    pub fn text(&self) -> String {
        let mut out = String::new();
        for seg in &self.segments {
            if let Segment::Text { text, .. } = seg {
                out.push_str(text);
            }
        }
        out
    }

    /// Index of the last segment if it is an open (not-yet-done) text segment.
    fn current_text_segment_index(&self) -> Option<usize> {
        let idx = self.segments.len().checked_sub(1)?;
        match &self.segments[idx] {
            Segment::Text { content_index, .. } if !self.text_part_done.contains(content_index) => {
                Some(idx)
            }
            _ => None,
        }
    }

    fn drain_pending_annotations(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.pending_annotations)
    }

    /// Accumulate annotations for the current or next text segment. Only object
    /// entries are kept; a non-array input is ignored.
    pub fn add_annotations(&mut self, annotations: Option<&Value>) {
        let Some(arr) = annotations.and_then(Value::as_array) else {
            return;
        };
        let normalized: Vec<Value> = arr.iter().filter(|a| a.is_object()).cloned().collect();
        if normalized.is_empty() {
            return;
        }
        if let Some(idx) = self.current_text_segment_index() {
            if let Segment::Text { annotations, .. } = &mut self.segments[idx] {
                annotations.extend(normalized);
                return;
            }
        }
        self.pending_annotations.extend(normalized);
    }

    /// Lazily open the `message` item on the first text/refusal. Idempotent.
    fn ensure_message_item_started(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
    ) -> Vec<Vec<u8>> {
        if self.lifecycle != Lifecycle::NotStarted {
            return Vec::new();
        }
        self.lifecycle = Lifecycle::Open;
        self.output_index = Some(envelope.allocate_output_index());
        let item = json!({
            "id": envelope.message_item_id(),
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        });
        vec![stream_events::output_item_added(
            self.output_index.unwrap_or(0),
            item,
        )]
    }

    /// Open a fresh text segment, draining any pending annotations onto it.
    /// Returns the new segment index plus the `content_part.added` events.
    fn start_text_segment(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
    ) -> (usize, Vec<Vec<u8>>) {
        let mut events = self.ensure_message_item_started(envelope);
        let content_index = self.segments.len();
        let annotations = self.drain_pending_annotations();
        self.segments.push(Segment::Text {
            content_index,
            text: String::new(),
            annotations: annotations.clone(),
        });
        let mid = envelope.message_item_id();
        let oi = self.output_index.unwrap_or(0);
        events.push(stream_events::content_part_added(
            &mid,
            oi,
            content_index,
            json!({ "type": "output_text", "text": "", "annotations": annotations }),
        ));
        (self.segments.len() - 1, events)
    }

    /// Append a text delta, opening a segment if none is currently open.
    pub fn push_text_delta(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        delta: &str,
    ) -> Vec<Vec<u8>> {
        if self.lifecycle == Lifecycle::Done || delta.is_empty() {
            return Vec::new();
        }
        let (seg_idx, mut events) = match self.current_text_segment_index() {
            Some(idx) => {
                let events = self.ensure_message_item_started(envelope);
                if !self.pending_annotations.is_empty() {
                    let drained = self.drain_pending_annotations();
                    if let Segment::Text { annotations, .. } = &mut self.segments[idx] {
                        annotations.extend(drained);
                    }
                }
                (idx, events)
            }
            None => self.start_text_segment(envelope),
        };
        let mid = envelope.message_item_id();
        let oi = self.output_index.unwrap_or(0);
        if let Segment::Text {
            text,
            content_index,
            ..
        } = &mut self.segments[seg_idx]
        {
            text.push_str(delta);
            let ci = *content_index;
            events.push(stream_events::output_text_delta(&mid, oi, ci, delta));
        }
        events
    }

    /// Emit a refusal as an immediately-closed content part.
    pub fn push_refusal_part(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        refusal: &str,
    ) -> Vec<Vec<u8>> {
        if refusal.is_empty() || self.lifecycle == Lifecycle::Done {
            return Vec::new();
        }
        let mut events = self.ensure_message_item_started(envelope);
        let content_index = self.segments.len();
        self.segments.push(Segment::Refusal {
            refusal: refusal.to_owned(),
        });
        let part = json!({ "type": "refusal", "refusal": refusal });
        let mid = envelope.message_item_id();
        let oi = self.output_index.unwrap_or(0);
        events.push(stream_events::content_part_added(
            &mid,
            oi,
            content_index,
            part.clone(),
        ));
        events.push(stream_events::content_part_done(
            &mid,
            oi,
            content_index,
            part,
        ));
        events
    }

    /// Render the ordered content parts for the completed message item.
    pub fn content_parts(&self) -> Vec<Value> {
        self.segments
            .iter()
            .map(|seg| match seg {
                Segment::Text {
                    text, annotations, ..
                } => json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": annotations,
                }),
                Segment::Refusal { refusal, .. } => json!({
                    "type": "refusal",
                    "refusal": refusal,
                }),
            })
            .collect()
    }

    /// Close a single text segment (idempotent per content_index).
    fn finalize_text_segment(
        &mut self,
        envelope: &ResponseEnvelopeState,
        seg_idx: usize,
    ) -> Vec<Vec<u8>> {
        let (content_index, text, annotations) = match &self.segments[seg_idx] {
            Segment::Text {
                content_index,
                text,
                annotations,
            } => (*content_index, text.clone(), annotations.clone()),
            Segment::Refusal { .. } => return Vec::new(),
        };
        if self.text_part_done.contains(&content_index) {
            return Vec::new();
        }
        self.text_part_done.insert(content_index);
        let text_part = json!({ "type": "output_text", "text": text, "annotations": annotations });
        let mid = envelope.message_item_id();
        let oi = self.output_index.unwrap_or(0);
        vec![
            stream_events::output_text_done(&mid, oi, content_index, &text),
            stream_events::content_part_done(&mid, oi, content_index, text_part),
        ]
    }

    /// Close the currently-open text segment, if any.
    pub fn flush_open_text_part(&mut self, envelope: &mut ResponseEnvelopeState) -> Vec<Vec<u8>> {
        if self.lifecycle == Lifecycle::Done {
            return Vec::new();
        }
        match self.current_text_segment_index() {
            Some(idx) => self.finalize_text_segment(envelope, idx),
            None => Vec::new(),
        }
    }

    /// Close every open text part, emit `output_item.done`, and register the
    /// completed message item with the envelope. A no-op if never opened or
    /// already done.
    pub fn finalize(&mut self, envelope: &mut ResponseEnvelopeState) -> Vec<Vec<u8>> {
        if self.lifecycle != Lifecycle::Open {
            return Vec::new();
        }
        self.lifecycle = Lifecycle::Done;
        let mut events: Vec<Vec<u8>> = Vec::new();
        let text_indices: Vec<usize> = self
            .segments
            .iter()
            .enumerate()
            .filter_map(|(i, s)| matches!(s, Segment::Text { .. }).then_some(i))
            .collect();
        for idx in text_indices {
            events.extend(self.finalize_text_segment(envelope, idx));
        }
        let content = self.content_parts();
        let item = json!({
            "id": envelope.message_item_id(),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": content,
        });
        let oi = self.output_index.unwrap_or(0);
        envelope.append_completed_item(oi, item.clone());
        events.push(stream_events::output_item_done(oi, item));
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn env() -> ResponseEnvelopeState {
        ResponseEnvelopeState::new(Some("resp_bridge_abc"))
    }

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

    fn events_of(raw: &[Vec<u8>]) -> Vec<(String, Value)> {
        raw.iter().map(|b| parse_event(b)).collect()
    }

    #[test]
    fn first_delta_opens_item_and_content_part_then_delta() {
        let mut env = env();
        let mut msg = MessageState::new();
        let evs = events_of(&msg.push_text_delta(&mut env, "hello"));
        let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
            ]
        );
        assert_eq!(evs[0].1["item"]["type"], json!("message"));
        assert_eq!(evs[0].1["item"]["status"], json!("in_progress"));
        assert_eq!(evs[2].1["delta"], json!("hello"));
        assert_eq!(evs[2].1["content_index"], json!(0));
    }

    #[test]
    fn subsequent_delta_only_emits_text_delta() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.push_text_delta(&mut env, "a");
        let evs = events_of(&msg.push_text_delta(&mut env, "b"));
        let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["response.output_text.delta"]);
        assert_eq!(evs[0].1["delta"], json!("b"));
        assert_eq!(msg.text(), "ab");
    }

    #[test]
    fn finalize_closes_text_and_emits_item_done_completed() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.push_text_delta(&mut env, "hi");
        let evs = events_of(&msg.finalize(&mut env));
        let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
            ]
        );
        assert_eq!(evs[0].1["text"], json!("hi"));
        assert_eq!(evs[2].1["item"]["status"], json!("completed"));
        assert_eq!(evs[2].1["item"]["content"][0]["text"], json!("hi"));
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.push_text_delta(&mut env, "hi");
        assert!(!msg.finalize(&mut env).is_empty());
        assert!(msg.finalize(&mut env).is_empty());
    }

    #[test]
    fn finalize_without_content_is_noop() {
        let mut env = env();
        let mut msg = MessageState::new();
        assert!(msg.finalize(&mut env).is_empty());
    }

    #[test]
    fn refusal_emits_added_then_done() {
        let mut env = env();
        let mut msg = MessageState::new();
        let evs = events_of(&msg.push_refusal_part(&mut env, "no"));
        let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.output_item.added",
                "response.content_part.added",
                "response.content_part.done",
            ]
        );
        assert_eq!(evs[1].1["part"]["type"], json!("refusal"));
        assert_eq!(evs[1].1["part"]["refusal"], json!("no"));
    }

    #[test]
    fn pending_annotations_attach_to_next_text_segment() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.add_annotations(Some(&json!([{ "type": "url_citation", "url": "x" }])));
        let evs = events_of(&msg.push_text_delta(&mut env, "hi"));
        // content_part.added carries the drained annotations.
        let part = &evs[1].1["part"];
        assert_eq!(part["annotations"][0]["type"], json!("url_citation"));
    }

    #[test]
    fn mid_segment_annotations_extend_open_segment() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.push_text_delta(&mut env, "hi");
        msg.add_annotations(Some(&json!([{ "type": "a" }])));
        let evs = events_of(&msg.finalize(&mut env));
        // content_part.done carries the annotation added mid-segment.
        assert_eq!(evs[1].1["part"]["annotations"][0]["type"], json!("a"));
    }

    #[test]
    fn add_annotations_ignores_non_array_and_non_object() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.add_annotations(Some(&json!("nope")));
        msg.add_annotations(Some(&json!([1, "two", true])));
        let evs = events_of(&msg.push_text_delta(&mut env, "hi"));
        assert_eq!(evs[1].1["part"]["annotations"], json!([]));
    }

    #[test]
    fn flush_then_new_delta_starts_second_segment() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.push_text_delta(&mut env, "a");
        let closed = events_of(&msg.flush_open_text_part(&mut env));
        let closed_names: Vec<&str> = closed.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            closed_names,
            ["response.output_text.done", "response.content_part.done"]
        );
        let evs = events_of(&msg.push_text_delta(&mut env, "b"));
        // A new content part opens at content_index 1.
        let names: Vec<&str> = evs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            ["response.content_part.added", "response.output_text.delta"]
        );
        assert_eq!(evs[0].1["content_index"], json!(1));
        assert_eq!(evs[1].1["content_index"], json!(1));
    }

    #[test]
    fn content_parts_reflects_text_and_refusal_order() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.push_text_delta(&mut env, "hi");
        msg.flush_open_text_part(&mut env);
        msg.push_refusal_part(&mut env, "no");
        let parts = msg.content_parts();
        assert_eq!(parts[0]["type"], json!("output_text"));
        assert_eq!(parts[0]["text"], json!("hi"));
        assert_eq!(parts[1]["type"], json!("refusal"));
        assert_eq!(parts[1]["refusal"], json!("no"));
    }

    #[test]
    fn push_after_done_is_suppressed() {
        let mut env = env();
        let mut msg = MessageState::new();
        msg.push_text_delta(&mut env, "hi");
        msg.finalize(&mut env);
        assert!(msg.push_text_delta(&mut env, "more").is_empty());
        assert!(msg.push_refusal_part(&mut env, "no").is_empty());
    }
}
