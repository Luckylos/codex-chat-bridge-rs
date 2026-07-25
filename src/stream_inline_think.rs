//! Inline-`<think>` three-phase state machine for streaming content deltas.
//!
//! Mirrors the Python bridge's `inline_think_sm.py`. A `content` delta stream
//! may embed a leading `<think>…</think>` block that must be routed to the
//! reasoning summary rather than emitted as visible text. Three phases:
//!
//! * `Detecting` — buffer until we can tell whether the content opens with a
//!   `<think>` tag (possibly split across chunks);
//! * `Reasoning` — inside the think block, route to the reasoning machine,
//!   holding back a trailing run that could still grow into a `</think>` close
//!   tag split across chunks;
//! * `Text` — past the think block, emit as plain text.
//!
//! The machine borrows the envelope / reasoning / message sub-states by mutable
//! reference (not the whole orchestrator) so the caller can split-borrow struct
//! fields without a self-aliasing conflict.
//!
//! Driven by the streaming orchestrator; reads as dead until that wires in.
#![allow(dead_code)]

use crate::reasoning::{
    could_be_partial_think_open, find_think_close, match_think_open_at_start,
    trailing_partial_close_len,
};
use crate::stream_envelope::ResponseEnvelopeState;
use crate::stream_message::MessageState;
use crate::stream_reasoning::ReasoningState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Detecting,
    Reasoning,
    Text,
}

/// Three-phase inline-think detector. Mirrors `InlineThinkStateMachine`.
pub struct InlineThinkStateMachine {
    phase: Phase,
    buffer: String,
    /// Trailing bytes of a `Reasoning`-phase chunk that could still grow into a
    /// `</think>` close tag split across chunk boundaries.
    reason_tail: String,
}

impl Default for InlineThinkStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl InlineThinkStateMachine {
    pub fn new() -> Self {
        Self {
            phase: Phase::Detecting,
            buffer: String::new(),
            reason_tail: String::new(),
        }
    }

    pub fn is_text_phase(&self) -> bool {
        self.phase == Phase::Text
    }

    pub fn is_detecting_or_reasoning(&self) -> bool {
        self.phase != Phase::Text
    }

    /// Route a content delta through inline-think detection. Mirrors
    /// `push_content_delta`.
    pub fn push_content_delta(
        &mut self,
        delta: &str,
        envelope: &mut ResponseEnvelopeState,
        reasoning: &mut ReasoningState,
        message: &mut MessageState,
    ) -> Vec<Vec<u8>> {
        match self.phase {
            Phase::Text => message.push_text_delta(envelope, delta),
            Phase::Reasoning => self.route_reasoning_delta(delta, envelope, reasoning, message),
            Phase::Detecting => self.route_detecting_delta(delta, envelope, reasoning, message),
        }
    }

    fn route_detecting_delta(
        &mut self,
        delta: &str,
        envelope: &mut ResponseEnvelopeState,
        reasoning: &mut ReasoningState,
        message: &mut MessageState,
    ) -> Vec<Vec<u8>> {
        self.buffer.push_str(delta);
        // `lstrip()` for detection only — the buffer keeps original bytes so a
        // non-think flush preserves leading whitespace.
        let trimmed = self.buffer.trim_start();

        if let Some(end) = match_think_open_at_start(trimmed) {
            // Detected `<think>` open tag — switch to reasoning phase.
            self.phase = Phase::Reasoning;
            let after_tag = trimmed[end..].to_owned();
            self.buffer.clear();
            if after_tag.is_empty() {
                return Vec::new();
            }
            return self.route_reasoning_delta(&after_tag, envelope, reasoning, message);
        }

        if could_be_partial_think_open(trimmed) {
            // Not enough data yet — keep buffering silently.
            return Vec::new();
        }

        // Not a `<think>` prefix — flush the whole buffer as text.
        self.phase = Phase::Text;
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let buffered = std::mem::take(&mut self.buffer);
        message.push_text_delta(envelope, &buffered)
    }

    fn route_reasoning_delta(
        &mut self,
        delta: &str,
        envelope: &mut ResponseEnvelopeState,
        reasoning: &mut ReasoningState,
        message: &mut MessageState,
    ) -> Vec<Vec<u8>> {
        let mut combined = std::mem::take(&mut self.reason_tail);
        combined.push_str(delta);
        let mut events = Vec::new();

        if let Some((start, end)) = find_think_close(&combined) {
            let pre = &combined[..start];
            if !pre.is_empty() {
                events.extend(reasoning.push_delta(envelope, pre));
            }
            events.extend(reasoning.finalize(envelope));
            self.phase = Phase::Text;
            let post = &combined[end..];
            if !post.is_empty() {
                events.extend(message.push_text_delta(envelope, post));
            }
            return events;
        }

        let hold = trailing_partial_close_len(&combined);
        if hold > 0 {
            let split = combined.len() - hold;
            self.reason_tail = combined[split..].to_owned();
            combined.truncate(split);
        }
        if !combined.is_empty() {
            events.extend(reasoning.push_delta(envelope, &combined));
        }
        events
    }

    fn drain_reason_tail(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        reasoning: &mut ReasoningState,
    ) -> Vec<Vec<u8>> {
        if self.reason_tail.is_empty() {
            return Vec::new();
        }
        let tail = std::mem::take(&mut self.reason_tail);
        reasoning.push_delta(envelope, &tail)
    }

    /// Flush buffered content during stream finalization. Mirrors
    /// `flush_on_finalize`.
    pub fn flush_on_finalize(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        reasoning: &mut ReasoningState,
        message: &mut MessageState,
    ) -> Vec<Vec<u8>> {
        let mut events = Vec::new();
        match self.phase {
            Phase::Reasoning => {
                // Unclosed think block — treat accumulated reasoning as-is.
                events.extend(self.drain_reason_tail(envelope, reasoning));
                events.extend(reasoning.finalize(envelope));
            }
            Phase::Detecting if !self.buffer.is_empty() => {
                // Never saw a think tag — emit the buffer as text.
                let buffered = std::mem::take(&mut self.buffer);
                events.extend(message.push_text_delta(envelope, &buffered));
            }
            _ => {}
        }
        self.phase = Phase::Text;
        events
    }

    /// Force-flush buffered/reasoning content as text when tool calls arrive
    /// before think detection completes. Mirrors `force_to_text`.
    pub fn force_to_text(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        reasoning: &mut ReasoningState,
        message: &mut MessageState,
    ) -> Vec<Vec<u8>> {
        let mut events = Vec::new();
        if self.phase != Phase::Text {
            events.extend(self.drain_reason_tail(envelope, reasoning));
            events.extend(reasoning.finalize(envelope));
            if !self.buffer.is_empty() {
                let buffered = std::mem::take(&mut self.buffer);
                events.extend(message.push_text_delta(envelope, &buffered));
            }
            self.phase = Phase::Text;
        }
        events
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
        let value = if data.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&data).unwrap()
        };
        (event, value)
    }

    /// Collect the event `type` names in order, for order-sensitive assertions.
    fn event_types(events: &[Vec<u8>]) -> Vec<String> {
        events.iter().map(|e| parse_event(e).0).collect()
    }

    struct Harness {
        sm: InlineThinkStateMachine,
        envelope: ResponseEnvelopeState,
        reasoning: ReasoningState,
        message: MessageState,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                sm: InlineThinkStateMachine::new(),
                envelope: ResponseEnvelopeState::new(Some("resp_bridge_abc")),
                reasoning: ReasoningState::new(),
                message: MessageState::new(),
            }
        }

        fn push(&mut self, delta: &str) -> Vec<Vec<u8>> {
            self.sm.push_content_delta(
                delta,
                &mut self.envelope,
                &mut self.reasoning,
                &mut self.message,
            )
        }

        fn finalize(&mut self) -> Vec<Vec<u8>> {
            self.sm
                .flush_on_finalize(&mut self.envelope, &mut self.reasoning, &mut self.message)
        }

        fn force(&mut self) -> Vec<Vec<u8>> {
            self.sm
                .force_to_text(&mut self.envelope, &mut self.reasoning, &mut self.message)
        }
    }

    #[test]
    fn plain_text_without_think_flushes_as_text() {
        let mut h = Harness::new();
        // "Hello" cannot be a think prefix, so it flushes immediately as text.
        let events = h.push("Hello");
        let types = event_types(&events);
        assert!(types.contains(&"response.output_item.added".to_owned()));
        assert!(types.contains(&"response.content_part.added".to_owned()));
        assert!(types.contains(&"response.output_text.delta".to_owned()));
        assert!(h.sm.is_text_phase());
    }

    #[test]
    fn single_chunk_think_block_routes_reasoning_then_text() {
        let mut h = Harness::new();
        let events = h.push("<think>pondering</think>answer");
        let types = event_types(&events);
        // Reasoning opens + summary delta, finalizes, then text emits.
        assert!(types.contains(&"response.reasoning_summary_text.delta".to_owned()));
        assert!(types.contains(&"response.reasoning_summary_text.done".to_owned()));
        assert!(types.contains(&"response.output_text.delta".to_owned()));
        assert!(h.sm.is_text_phase());
    }

    #[test]
    fn think_open_split_across_chunks_buffers_then_detects() {
        let mut h = Harness::new();
        // Partial "<thi" is a possible prefix — buffered silently, no events.
        let events = h.push("<thi");
        assert!(events.is_empty());
        assert!(h.sm.is_detecting_or_reasoning());
        // Completing the tag + content switches to reasoning.
        let events = h.push("nk>secret");
        let types = event_types(&events);
        assert!(types.contains(&"response.reasoning_summary_text.delta".to_owned()));
    }

    #[test]
    fn close_tag_split_across_chunks_is_held_back() {
        let mut h = Harness::new();
        h.push("<think>abc");
        // "</thi" could start a close tag — held back, not emitted as reasoning.
        let events = h.push("</thi");
        let delta_texts: Vec<String> = events
            .iter()
            .filter_map(|e| {
                let (ev, data) = parse_event(e);
                if ev == "response.reasoning_summary_text.delta" {
                    Some(data["delta"].as_str().unwrap_or_default().to_owned())
                } else {
                    None
                }
            })
            .collect();
        // The held-back "</thi" must not appear in any reasoning delta.
        assert!(delta_texts.iter().all(|t| !t.contains("</thi")));
        // Completing the close tag finalizes reasoning and emits trailing text.
        let events = h.push("nk>done");
        let types = event_types(&events);
        assert!(types.contains(&"response.reasoning_summary_text.done".to_owned()));
        assert!(types.contains(&"response.output_text.delta".to_owned()));
    }

    #[test]
    fn unclosed_think_block_finalizes_reasoning_on_flush() {
        let mut h = Harness::new();
        h.push("<think>still thinking");
        let events = h.finalize();
        let types = event_types(&events);
        assert!(types.contains(&"response.reasoning_summary_text.done".to_owned()));
        assert!(h.sm.is_text_phase());
    }

    #[test]
    fn detecting_buffer_flushed_as_text_on_finalize() {
        let mut h = Harness::new();
        // A lone "<" stays in detecting (could be a partial tag).
        let events = h.push("<");
        assert!(events.is_empty());
        // Finalize flushes the buffer as text since no tag ever completed.
        let events = h.finalize();
        let types = event_types(&events);
        assert!(types.contains(&"response.output_text.delta".to_owned()));
    }

    #[test]
    fn force_to_text_flushes_pending_reasoning() {
        let mut h = Harness::new();
        h.push("<think>partial");
        let events = h.force();
        let types = event_types(&events);
        // Pending reasoning is finalized so a following tool call is well-ordered.
        assert!(types.contains(&"response.reasoning_summary_text.done".to_owned()));
        assert!(h.sm.is_text_phase());
    }

    #[test]
    fn force_to_text_is_noop_in_text_phase() {
        let mut h = Harness::new();
        h.push("plain");
        assert!(h.sm.is_text_phase());
        let events = h.force();
        assert!(events.is_empty());
    }
}
