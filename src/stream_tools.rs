//! Tool-call increment state machine for the streaming path.
//!
//! Mirrors the Python bridge's `stream_state/tools.py` (plus the supporting
//! `tool_types` / `tool_items` / `tool_progress` modules). Tracks any number
//! of parallel tool calls, keyed by their upstream chat `index`, across the
//! stream:
//!
//! * each `tool_calls[]` delta accumulates id / name / arguments;
//! * the first delta carrying a name or call id lazily emits
//!   `output_item.added` (an `in_progress` `function_call` /
//!   `custom_tool_call` / `tool_search_call` item);
//! * subsequent argument deltas stream out as `function_call_arguments.delta`
//!   (chunked at 64 bytes) or `custom_tool_call_input.delta`;
//! * `finalize` closes every open call with its `.done` events plus
//!   `output_item.done`, and registers the completed items with the envelope.
//!
//! Output-index allocation differs from the reasoning / message machines: tool
//! calls claim a contiguous base so parallel calls preserve their upstream
//! `index` ordering (`base + index`), rather than allocating one at a time.
//!
//! Phase 1 scope: the nested-namespace buffering path (`nested_buffered` /
//! `try_resolve_nested_buffer` / `degrade_buffered_namespace`) from the Python
//! bridge is Phase 3; here namespace resolution is the identity mapping, so a
//! tool call is added as soon as it carries a name or id.
//!
//! Driven by the top-level stream orchestrator that lands in a later layer, so
//! the store reads as dead until it wires in. Tests lock in the event
//! sequence now.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::context::{
    self, custom_tool_input_from_chat_arguments, parse_tool_arguments_object, BridgeToolContext,
};
use crate::convert::canonicalize_tool_arguments;
use crate::id_gen;
use crate::stream_envelope::ResponseEnvelopeState;
use crate::stream_events;

/// Chunk size for incremental `function_call_arguments.delta` emission.
const ARGUMENTS_CHUNK_SIZE: usize = 64;

/// Upper bound on how many bytes of a nested-namespace call's arguments we will
/// buffer while waiting for the `action` selector to become parseable before
/// degrading to a namespace-level call. Mirrors `NAMESPACE_BUFFER_MAX_BYTES`.
const NAMESPACE_BUFFER_MAX_BYTES: usize = 4096;

/// Per-call accumulation state. Mirrors `tool_types.ToolCallState`.
#[derive(Debug, Default)]
struct ToolCallState {
    output_index: Option<i64>,
    item_id: String,
    call_id: String,
    name: String,
    /// Restored namespace for a nested-namespace call, set once its `action`
    /// resolves (or on degradation). Drives the response-side `namespace`
    /// field. Mirrors `ToolCallState.namespace`.
    namespace: Option<String>,
    arguments: String,
    added: bool,
    done: bool,
    reasoning_content: String,
    emitted_custom_input: String,
    /// How much of `arguments` has already streamed out as delta events, so
    /// incremental replay resumes at the right offset and finalize does not
    /// re-emit already-sent content.
    emitted_arguments: String,
    /// True while a nested-namespace call is holding back its `output_item.added`
    /// until the `action` selector can be parsed out of the arguments buffer.
    nested_buffered: bool,
    /// True once a nested-namespace call has been resolved (action extracted)
    /// or degraded to a namespace-level call, so the buffering path is skipped.
    nested_resolved: bool,
}

/// Tool classification. Mirrors `tool_types.ToolKind`.
#[derive(Debug, Clone, Copy)]
struct ToolKind {
    is_custom: bool,
    is_tool_search: bool,
}

impl ToolKind {
    fn response_item_type(&self) -> &'static str {
        if self.is_custom {
            "custom_tool_call"
        } else if self.is_tool_search {
            "tool_search_call"
        } else {
            "function_call"
        }
    }
}

/// A completed tool item plus the derived strings the finalize path needs.
/// Mirrors `tool_types.CompletedToolEmission`.
struct CompletedToolEmission {
    item: Value,
    arguments: String,
    input_text: Option<String>,
}

fn resolve_tool_kind(ctx: &BridgeToolContext, name: &str) -> ToolKind {
    let name_opt = if name.is_empty() { None } else { Some(name) };
    ToolKind {
        is_custom: ctx.is_custom_tool(name_opt),
        is_tool_search: ctx.is_tool_search(name_opt),
    }
}

/// The `ToolSpec` for `name` when it refers to a nested-namespace tool, else
/// `None`. Mirrors `tool_namespace.nested_namespace_spec`.
fn nested_namespace_spec<'a>(
    ctx: &'a BridgeToolContext,
    name: &str,
) -> Option<&'a crate::context::ToolSpec> {
    if name.is_empty() {
        return None;
    }
    ctx.lookup_chat_name(Some(name))
        .filter(|spec| spec.is_nested_namespace())
}

/// Try to resolve a buffered nested-namespace call: if the `action` selector is
/// now parseable, rewrite the state's name/namespace/arguments to the concrete
/// action and clear the buffering flag. Returns whether it resolved. Mirrors
/// `tool_namespace.try_resolve_nested_buffer`.
fn try_resolve_nested_buffer(ctx: &BridgeToolContext, state: &mut ToolCallState) -> bool {
    let Some(spec) = nested_namespace_spec(ctx, &state.name) else {
        return false;
    };
    let resolution = crate::context::resolve_nested_namespace_arguments(spec, &state.arguments);
    let Some(action_name) = resolution.action_name else {
        return false;
    };
    state.namespace = spec.namespace.clone();
    state.name = action_name;
    state.arguments = resolution.normalized_arguments;
    state.nested_buffered = false;
    state.nested_resolved = true;
    true
}

/// Degrade a still-unresolved buffered namespace call to a namespace-level call
/// (keep the namespace name, normalize the arguments). Mirrors
/// `tool_namespace.degrade_buffered_namespace`.
fn degrade_buffered_namespace(ctx: &BridgeToolContext, state: &mut ToolCallState) {
    if let Some(spec) = nested_namespace_spec(ctx, &state.name) {
        let resolution = crate::context::resolve_nested_namespace_arguments(spec, &state.arguments);
        state.namespace = spec.namespace.clone();
        state.name = spec.name.clone();
        state.arguments = resolution.normalized_arguments;
    }
    state.nested_buffered = false;
    state.nested_resolved = true;
}

/// At finalize, flush a still-buffered nested call: resolve its action if
/// possible, otherwise degrade to namespace-level. Mirrors
/// `tool_namespace.flush_buffered_nested_state`.
fn flush_buffered_nested_state(ctx: &BridgeToolContext, state: &mut ToolCallState) {
    if !state.nested_buffered || state.added {
        return;
    }
    if try_resolve_nested_buffer(ctx, state) {
        return;
    }
    degrade_buffered_namespace(ctx, state);
}

/// Assign a synthetic call id / fallback name and derive the item id from the
/// call id. Mirrors `tool_items.ensure_tool_identity`.
fn ensure_tool_identity(state: &mut ToolCallState, kind: ToolKind) {
    if state.call_id.is_empty() {
        state.call_id = id_gen::synthetic_tool_call_id();
    }
    if state.name.is_empty() {
        state.name = "unknown_tool".to_owned();
    }
    state.item_id = if kind.is_custom {
        id_gen::custom_tool_call_item_id(&state.call_id)
    } else {
        id_gen::function_call_item_id(&state.call_id)
    };
}

/// Resolve the response-side `(name, namespace)`. Mirrors
/// `tool_items._response_name_and_namespace`. Phase 1 namespace is identity.
fn response_name_and_namespace(
    state: &ToolCallState,
    kind: ToolKind,
    ctx: &BridgeToolContext,
) -> (Option<String>, Option<String>) {
    if kind.is_tool_search {
        return (None, None);
    }
    if kind.is_custom {
        return (Some(state.name.clone()), None);
    }
    // A resolved/degraded nested-namespace call already carries its restored
    // namespace + action name directly on the state; use them verbatim.
    if let Some(ns) = &state.namespace {
        return (Some(state.name.clone()), Some(ns.clone()));
    }
    let (namespace, restored) = ctx.restore_namespace_and_name(&state.name);
    match namespace {
        Some(ns) => (Some(restored), Some(ns)),
        None => (Some(state.name.clone()), None),
    }
}

fn with_reasoning_content(state: &ToolCallState, mut item: Value) -> Value {
    if !state.reasoning_content.is_empty() {
        if let Value::Object(map) = &mut item {
            map.insert(
                "reasoning_content".to_owned(),
                Value::String(state.reasoning_content.clone()),
            );
        }
    }
    item
}

/// Build the `in_progress` output item for `output_item.added`. Mirrors
/// `tool_items.build_in_progress_item`.
fn build_in_progress_item(state: &ToolCallState, kind: ToolKind, ctx: &BridgeToolContext) -> Value {
    let (name, namespace) = response_name_and_namespace(state, kind, ctx);
    let mut map = serde_json::Map::new();
    map.insert("id".to_owned(), json!(state.item_id));
    map.insert("type".to_owned(), json!(kind.response_item_type()));
    map.insert("status".to_owned(), json!("in_progress"));
    map.insert("call_id".to_owned(), json!(state.call_id));
    if let Some(name) = name {
        map.insert("name".to_owned(), json!(name));
    }
    if let Some(namespace) = namespace {
        map.insert("namespace".to_owned(), json!(namespace));
    }
    if kind.is_tool_search {
        map.insert("execution".to_owned(), json!("client"));
        map.insert("arguments".to_owned(), json!({}));
    } else if kind.is_custom {
        map.insert("input".to_owned(), json!(""));
    } else {
        map.insert("arguments".to_owned(), json!(""));
    }
    with_reasoning_content(state, Value::Object(map))
}

/// Build the completed output item for `output_item.done`. Mirrors
/// `tool_items.build_completed_item`.
fn build_completed_item(
    state: &ToolCallState,
    kind: ToolKind,
    ctx: &BridgeToolContext,
) -> CompletedToolEmission {
    let arguments = canonicalize_tool_arguments(Some(&Value::String(state.arguments.clone())));
    let (name, namespace) = response_name_and_namespace(state, kind, ctx);

    if kind.is_custom {
        let input_text = custom_tool_input_from_chat_arguments(&arguments);
        let mut map = serde_json::Map::new();
        map.insert("id".to_owned(), json!(state.item_id));
        map.insert("type".to_owned(), json!("custom_tool_call"));
        map.insert("status".to_owned(), json!("completed"));
        map.insert("call_id".to_owned(), json!(state.call_id));
        if let Some(name) = name {
            map.insert("name".to_owned(), json!(name));
        }
        map.insert("input".to_owned(), json!(input_text));
        let item = with_reasoning_content(state, Value::Object(map));
        return CompletedToolEmission {
            item,
            arguments,
            input_text: Some(input_text),
        };
    }

    if kind.is_tool_search {
        let mut map = serde_json::Map::new();
        map.insert("id".to_owned(), json!(state.item_id));
        map.insert("type".to_owned(), json!("tool_search_call"));
        map.insert("status".to_owned(), json!("completed"));
        map.insert("call_id".to_owned(), json!(state.call_id));
        map.insert("execution".to_owned(), json!("client"));
        map.insert(
            "arguments".to_owned(),
            parse_tool_arguments_object(&arguments),
        );
        let item = with_reasoning_content(state, Value::Object(map));
        return CompletedToolEmission {
            item,
            arguments,
            input_text: None,
        };
    }

    let mut map = serde_json::Map::new();
    map.insert("id".to_owned(), json!(state.item_id));
    map.insert("type".to_owned(), json!("function_call"));
    map.insert("status".to_owned(), json!("completed"));
    map.insert("call_id".to_owned(), json!(state.call_id));
    if let Some(name) = name {
        map.insert("name".to_owned(), json!(name));
    }
    if let Some(namespace) = namespace {
        map.insert("namespace".to_owned(), json!(namespace));
    }
    map.insert("arguments".to_owned(), json!(arguments));
    let item = with_reasoning_content(state, Value::Object(map));
    CompletedToolEmission {
        item,
        arguments,
        input_text: None,
    }
}

/// Accumulate one `tool_calls[]` delta into `state`. Returns the raw arguments
/// fragment when present. Mirrors `tool_progress.apply_tool_call_delta`.
fn apply_tool_call_delta(
    state: &mut ToolCallState,
    tool_call: &Value,
    reasoning: Option<&str>,
) -> Option<String> {
    if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
        if !id.is_empty() {
            state.call_id = id.to_owned();
        }
    }
    let function = tool_call.get("function").and_then(Value::as_object);
    if let Some(function) = function {
        if let Some(name) = function.get("name").and_then(Value::as_str) {
            if !name.is_empty() {
                state.name = name.to_owned();
            }
        }
        if let Some(args) = function.get("arguments").and_then(Value::as_str) {
            if !args.is_empty() {
                state.arguments.push_str(args);
                if let Some(reasoning) = reasoning {
                    if !reasoning.is_empty() && state.reasoning_content.is_empty() {
                        state.reasoning_content = reasoning.to_owned();
                    }
                }
                return Some(args.to_owned());
            }
        }
    }
    if let Some(reasoning) = reasoning {
        if !reasoning.is_empty() && state.reasoning_content.is_empty() {
            state.reasoning_content = reasoning.to_owned();
        }
    }
    None
}

/// Emit the not-yet-sent tail of `arguments` as chunked delta events. Mirrors
/// `tool_progress.emit_arguments_incremental`.
fn emit_arguments_incremental(state: &mut ToolCallState) -> Vec<Vec<u8>> {
    let full = state.arguments.clone();
    let already = state.emitted_arguments.clone();
    if full.is_empty() || full == already {
        return Vec::new();
    }
    let pending = if full.starts_with(&already) {
        full[already.len()..].to_owned()
    } else {
        state.emitted_arguments = String::new();
        full.clone()
    };
    if pending.is_empty() {
        return Vec::new();
    }
    let output_index = state.output_index.unwrap_or(0);
    let mut events = Vec::new();
    let bytes = pending.as_bytes();
    let mut offset = 0;
    while offset < bytes.len() {
        let end = char_boundary_end(&pending, offset, ARGUMENTS_CHUNK_SIZE);
        let chunk = &pending[offset..end];
        events.push(stream_events::function_arguments_delta(
            &state.item_id,
            output_index,
            chunk,
        ));
        offset = end;
    }
    state.emitted_arguments = format!("{already}{pending}");
    events
}

/// The end byte offset for a chunk of up to `max` bytes starting at `start`,
/// snapped back to a UTF-8 char boundary so we never split a multi-byte char.
fn char_boundary_end(s: &str, start: usize, max: usize) -> usize {
    let mut end = (start + max).min(s.len());
    while end > start && !s.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Emit a `custom_tool_call_input.delta` for the newly-decoded input prefix.
/// Mirrors `tool_progress.emit_custom_input_delta_events`.
fn emit_custom_input_delta_events(state: &mut ToolCallState) -> Vec<Vec<u8>> {
    let prefix = match context::partial_custom_tool_input_from_chat_arguments(&state.arguments) {
        Some(prefix) => prefix,
        None => return Vec::new(),
    };
    if prefix == state.emitted_custom_input {
        return Vec::new();
    }
    if !prefix.starts_with(&state.emitted_custom_input) {
        state.emitted_custom_input = String::new();
    }
    let delta = prefix[state.emitted_custom_input.len()..].to_owned();
    if delta.is_empty() {
        return Vec::new();
    }
    let output_index = state.output_index.unwrap_or(0);
    state.emitted_custom_input = prefix;
    vec![stream_events::custom_input_delta(
        &state.item_id,
        output_index,
        &delta,
    )]
}

/// Streaming store for all tool calls in a turn. Mirrors
/// `tools.ToolStateStore`.
pub struct ToolStateStore {
    tool_context: BridgeToolContext,
    tool_calls: BTreeMap<i64, ToolCallState>,
    finalized: bool,
    tool_output_base: Option<i64>,
}

impl ToolStateStore {
    pub fn new(tool_context: BridgeToolContext) -> Self {
        Self {
            tool_context,
            tool_calls: BTreeMap::new(),
            finalized: false,
            tool_output_base: None,
        }
    }

    /// Read-only projection of the state at `index`, or `default` when the
    /// index has no state yet. Collapses the repeated
    /// `self.tool_calls.get(&index).map(..).unwrap_or(..)` lookups.
    fn state_or<T>(&self, index: i64, default: T, f: impl FnOnce(&ToolCallState) -> T) -> T {
        self.tool_calls.get(&index).map(f).unwrap_or(default)
    }

    /// Mutate the state at `index` when present. Centralizes the
    /// `if let Some(state) = self.tool_calls.get_mut(&index)` side-effect dance.
    fn modify_state(&mut self, index: i64, f: impl FnOnce(&mut ToolCallState)) {
        if let Some(state) = self.tool_calls.get_mut(&index) {
            f(state);
        }
    }

    /// Like `modify_state` but also lends the shared tool context, for the
    /// nested-namespace helpers that resolve specs against it. Split-borrows
    /// the two fields so the borrow checker sees they are disjoint.
    fn with_ctx_state_mut(
        &mut self,
        index: i64,
        f: impl FnOnce(&BridgeToolContext, &mut ToolCallState),
    ) {
        let ctx = &self.tool_context;
        if let Some(state) = self.tool_calls.get_mut(&index) {
            f(ctx, state);
        }
    }

    /// Claim a stable output index for `state`, `base + index`, so parallel
    /// tool calls preserve their upstream ordering. Mirrors
    /// `ToolStateStore._ensure_output_index`.
    fn ensure_output_index(&mut self, envelope: &mut ResponseEnvelopeState, index: i64) -> i64 {
        if let Some(existing) = self.tool_calls.get(&index).and_then(|s| s.output_index) {
            return existing;
        }
        let base = *self
            .tool_output_base
            .get_or_insert_with(|| envelope.peek_next_output_index());
        let output_index = base + index;
        envelope.advance_output_index_to(output_index + 1);
        if let Some(state) = self.tool_calls.get_mut(&index) {
            state.output_index = Some(output_index);
        }
        output_index
    }

    /// Emit `output_item.added` on the first delta that carries an identity.
    /// Mirrors `ToolStateStore._ensure_added`.
    fn ensure_added(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        index: i64,
    ) -> (Vec<Vec<u8>>, ToolKind) {
        let name = self
            .tool_calls
            .get(&index)
            .map(|s| s.name.clone())
            .unwrap_or_default();
        let kind = resolve_tool_kind(&self.tool_context, &name);
        let already_added = self
            .tool_calls
            .get(&index)
            .map(|s| s.added)
            .unwrap_or(false);
        if already_added {
            return (Vec::new(), kind);
        }
        if let Some(state) = self.tool_calls.get_mut(&index) {
            state.added = true;
            ensure_tool_identity(state, kind);
        }
        let output_index = self.ensure_output_index(envelope, index);
        let state = self.tool_calls.get(&index).expect("state present");
        let item = build_in_progress_item(state, kind, &self.tool_context);
        (
            vec![stream_events::output_item_added(output_index, item)],
            kind,
        )
    }

    /// Begin buffering a nested-namespace call: claim its output index but hold
    /// back `output_item.added` until the `action` selector is parseable.
    /// Mirrors `ToolStateStore._maybe_start_nested_buffer`.
    fn maybe_start_nested_buffer(&mut self, envelope: &mut ResponseEnvelopeState, index: i64) {
        let should_buffer = match self.tool_calls.get(&index) {
            Some(state) => {
                !state.added
                    && !state.nested_buffered
                    && !state.nested_resolved
                    && nested_namespace_spec(&self.tool_context, &state.name).is_some()
            }
            None => false,
        };
        if !should_buffer {
            return;
        }
        self.ensure_output_index(envelope, index);
        if let Some(state) = self.tool_calls.get_mut(&index) {
            state.nested_buffered = true;
        }
    }

    /// Try to resolve a buffered nested call this delta: if its action is now
    /// parseable, emit the deferred `output_item.added` plus any accumulated
    /// arguments. Mirrors `ToolStateStore._emit_buffered_nested_events`.
    fn emit_buffered_nested_events(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        index: i64,
    ) -> Vec<Vec<u8>> {
        let resolved = {
            let ctx = &self.tool_context;
            match self.tool_calls.get_mut(&index) {
                Some(state) => try_resolve_nested_buffer(ctx, state),
                None => false,
            }
        };
        if !resolved {
            return Vec::new();
        }
        let (mut events, kind) = self.ensure_added(envelope, index);
        if !events.is_empty() && !kind.is_custom {
            if let Some(state) = self.tool_calls.get_mut(&index) {
                if !state.arguments.is_empty() {
                    events.extend(emit_arguments_incremental(state));
                }
            }
        }
        events
    }

    /// Accumulate a `tool_calls[]` delta and emit the incremental events it
    /// produces. Mirrors `ToolStateStore.push_delta`, including the
    /// nested-namespace buffering path (buffer until `action` resolves, or
    /// degrade to a namespace-level call once the buffer overflows).
    pub fn push_delta(
        &mut self,
        envelope: &mut ResponseEnvelopeState,
        tool_call: &Value,
        reasoning: Option<&str>,
    ) -> Vec<Vec<u8>> {
        if self.finalized {
            return Vec::new();
        }
        let index = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
        let (args_delta, has_identity, already_added) = {
            let state = self.tool_calls.entry(index).or_default();
            let args_delta = apply_tool_call_delta(state, tool_call, reasoning);
            (
                args_delta,
                !state.call_id.is_empty() || !state.name.is_empty(),
                state.added,
            )
        };

        // A nested-namespace call is held back until its `action` can be parsed
        // out of the arguments buffer.
        if !already_added && has_identity {
            self.maybe_start_nested_buffer(envelope, index);
        }
        let nested_buffered = self.state_or(index, false, |s| s.nested_buffered);
        if nested_buffered {
            let events = self.emit_buffered_nested_events(envelope, index);
            let still_buffered = self.state_or(index, false, |s| s.nested_buffered);
            if !events.is_empty() || !still_buffered {
                return events;
            }
            // Still unresolved: degrade to a namespace-level call once the
            // buffer grows past the cap, so a never-closing action selector
            // can't stall the stream indefinitely.
            let overflowed = self.state_or(index, false, |s| {
                s.arguments.len() > NAMESPACE_BUFFER_MAX_BYTES
            });
            if overflowed {
                self.with_ctx_state_mut(index, degrade_buffered_namespace);
                let (mut added_events, _) = self.ensure_added(envelope, index);
                self.modify_state(index, |state| {
                    if !state.arguments.is_empty() {
                        added_events.extend(emit_arguments_incremental(state));
                    }
                });
                return added_events;
            }
            return Vec::new();
        }

        let added_now = !already_added && has_identity;

        if !has_identity {
            return Vec::new();
        }

        let (mut events, kind) = self.ensure_added(envelope, index);

        if kind.is_custom {
            if let Some(state) = self.tool_calls.get_mut(&index) {
                events.extend(emit_custom_input_delta_events(state));
            }
            return events;
        }

        if added_now {
            if let Some(state) = self.tool_calls.get_mut(&index) {
                if !state.arguments.is_empty() {
                    events.extend(emit_arguments_incremental(state));
                    return events;
                }
            }
        }

        if let Some(args_delta) = args_delta {
            if !added_now {
                if let Some(state) = self.tool_calls.get_mut(&index) {
                    let output_index = state.output_index.unwrap_or(0);
                    events.push(stream_events::function_arguments_delta(
                        &state.item_id,
                        output_index,
                        &args_delta,
                    ));
                    // Intentionally do NOT advance `emitted_arguments` here.
                    // The raw per-chunk delta path only mirrors upstream
                    // fragments; `emitted_arguments` tracks what the
                    // *normalized* incremental replay has sent. Leaving it
                    // untouched lets `finalize_state` detect that the
                    // canonicalized full arguments differ from what was
                    // streamed and emit one authoritative merged delta before
                    // `.done`, so a client that reassembles from deltas still
                    // converges on the canonical arguments. Mirrors Python
                    // `tools.push_delta` (which never updates emitted_arguments
                    // on this branch).
                }
            }
        }
        events
    }

    /// Close every open tool call: flush residual arguments/input, emit the
    /// `.done` events + `output_item.done`, and register completed items with
    /// the envelope. Mirrors `ToolStateStore.finalize`.
    pub fn finalize(&mut self, envelope: &mut ResponseEnvelopeState) -> Vec<Vec<u8>> {
        if self.finalized {
            return Vec::new();
        }
        self.finalized = true;
        let indices: Vec<i64> = self.tool_calls.keys().copied().collect();
        let mut events = Vec::new();
        for index in indices {
            if self.tool_calls.get(&index).map(|s| s.done).unwrap_or(false) {
                continue;
            }
            events.extend(self.finalize_state(envelope, index));
        }
        events
    }

    /// Build the Chat-side `tool_calls[]` array for session persistence.
    ///
    /// Only calls that acquired a name are kept (an index that never resolved
    /// an identity is a partial artifact, not a real call). Arguments are
    /// sanitized at construction. Mirrors the tool_calls branch of
    /// `ResponsesStreamState.build_assistant_message`. Phase 1 has no
    /// chat-name/chat-arguments split, so `name`/`arguments` are used directly.
    pub fn persisted_tool_calls(&self) -> Vec<Value> {
        self.tool_calls
            .values()
            .filter(|state| !state.name.is_empty())
            .map(|state| {
                let call_id = if state.call_id.is_empty() {
                    id_gen::synthetic_tool_call_id()
                } else {
                    state.call_id.clone()
                };
                json!({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": state.name,
                        "arguments": crate::sanitize::sanitize_string(&state.arguments),
                    },
                })
            })
            .collect()
    }

    /// Mirrors `ToolStateStore._finalize_state`.
    fn finalize_state(&mut self, envelope: &mut ResponseEnvelopeState, index: i64) -> Vec<Vec<u8>> {
        let mut events = Vec::new();
        // Flush any still-buffered nested-namespace call: resolve its action if
        // the buffer now parses, otherwise degrade to a namespace-level call.
        self.with_ctx_state_mut(index, flush_buffered_nested_state);
        let has_identity = self.state_or(index, false, |s| {
            !s.call_id.is_empty() || !s.name.is_empty()
        });
        let already_added = self.state_or(index, false, |s| s.added);
        if !already_added && has_identity {
            let (added_events, _) = self.ensure_added(envelope, index);
            events.extend(added_events);
        }

        let name = self.state_or(index, String::new(), |s| s.name.clone());
        let kind = resolve_tool_kind(&self.tool_context, &name);

        if !kind.is_custom {
            self.modify_state(index, |state| {
                if state.emitted_arguments != state.arguments {
                    events.extend(emit_arguments_incremental(state));
                }
            });
        }

        let emission = {
            let state = self.tool_calls.get(&index).expect("state present");
            build_completed_item(state, kind, &self.tool_context)
        };

        let (item_id, output_index) = {
            let state = self.tool_calls.get_mut(&index).expect("state present");
            state.done = true;
            (state.item_id.clone(), state.output_index.unwrap_or(0))
        };

        if kind.is_custom {
            let input_text = emission.input_text.clone().unwrap_or_default();
            let emitted = self
                .tool_calls
                .get(&index)
                .map(|s| s.emitted_custom_input.clone())
                .unwrap_or_default();
            if input_text != emitted {
                let residual = if input_text.starts_with(&emitted) {
                    input_text[emitted.len()..].to_owned()
                } else {
                    input_text.clone()
                };
                if !residual.is_empty() {
                    events.push(stream_events::custom_input_delta(
                        &item_id,
                        output_index,
                        &residual,
                    ));
                }
                if let Some(state) = self.tool_calls.get_mut(&index) {
                    state.emitted_custom_input = input_text.clone();
                }
            }
            events.push(stream_events::custom_input_done(
                &item_id,
                output_index,
                &input_text,
            ));
        } else {
            events.push(stream_events::function_arguments_done(
                &item_id,
                output_index,
                &emission.arguments,
            ));
        }
        events.push(stream_events::output_item_done(
            output_index,
            emission.item.clone(),
        ));
        envelope.append_completed_item(output_index, emission.item);
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn function_delta(index: i64, id: &str, name: &str, args: &str) -> Value {
        json!({
            "index": index,
            "id": id,
            "function": { "name": name, "arguments": args },
        })
    }

    #[test]
    fn function_call_added_then_arguments_then_done() {
        let mut store = ToolStateStore::new(BridgeToolContext::new());
        let mut envelope = env();

        let events = store.push_delta(
            &mut envelope,
            &function_delta(0, "call_1", "get_weather", "{\"city\":"),
            None,
        );
        // First delta: output_item.added, then a function_call_arguments.delta
        // replaying the accumulated arguments.
        assert_eq!(events.len(), 2);
        let (added_event, added) = parse_event(&events[0]);
        assert_eq!(added_event, "response.output_item.added");
        assert_eq!(added["item"]["type"], json!("function_call"));
        assert_eq!(added["item"]["status"], json!("in_progress"));
        assert_eq!(added["item"]["call_id"], json!("call_1"));
        assert_eq!(added["item"]["name"], json!("get_weather"));
        assert_eq!(added["output_index"], json!(0));
        let (delta_event, delta) = parse_event(&events[1]);
        assert_eq!(delta_event, "response.function_call_arguments.delta");
        assert_eq!(delta["delta"], json!("{\"city\":"));

        // Second delta: only the new fragment streams out.
        let events = store.push_delta(
            &mut envelope,
            &function_delta(0, "call_1", "get_weather", "\"NYC\"}"),
            None,
        );
        assert_eq!(events.len(), 1);
        let (_, delta) = parse_event(&events[0]);
        assert_eq!(delta["delta"], json!("\"NYC\"}"));

        // Finalize: the first delta already replayed `{"city":` via
        // `emit_arguments_incremental` (which advanced `emitted_arguments`),
        // but the second delta streamed `"NYC"}` through the raw per-chunk
        // path, which intentionally does NOT advance `emitted_arguments`. So
        // finalize sees the canonical `emitted_arguments` (`{"city":`) still
        // trails the full `arguments` (`{"city":"NYC"}`) and emits one
        // `function_call_arguments.delta` carrying just the unreplayed tail
        // `"NYC"}` before `.done`. This lets a client reassembling from the
        // normalized-delta stream converge on the canonical value. Then come
        // arguments.done + output_item.done. Mirrors Python `tools.finalize`.
        let events = store.finalize(&mut envelope);
        assert_eq!(events.len(), 3);
        let (merged_event, merged) = parse_event(&events[0]);
        assert_eq!(merged_event, "response.function_call_arguments.delta");
        assert_eq!(merged["delta"], json!("\"NYC\"}"));
        let (done_event, done) = parse_event(&events[1]);
        assert_eq!(done_event, "response.function_call_arguments.done");
        assert_eq!(done["arguments"], json!("{\"city\":\"NYC\"}"));
        let (item_done_event, item_done) = parse_event(&events[2]);
        assert_eq!(item_done_event, "response.output_item.done");
        assert_eq!(item_done["item"]["status"], json!("completed"));
        assert_eq!(item_done["item"]["arguments"], json!("{\"city\":\"NYC\"}"));
    }

    #[test]
    fn parallel_calls_keep_index_ordering() {
        let mut store = ToolStateStore::new(BridgeToolContext::new());
        let mut envelope = env();
        // Pre-allocate an output index (e.g. a preceding message) so the base
        // is non-zero and we can see base+index at work.
        assert_eq!(envelope.allocate_output_index(), 0);

        store.push_delta(
            &mut envelope,
            &function_delta(0, "call_a", "tool_a", "{}"),
            None,
        );
        store.push_delta(
            &mut envelope,
            &function_delta(1, "call_b", "tool_b", "{}"),
            None,
        );

        let events = store.finalize(&mut envelope);
        // Two calls × (arguments.done + output_item.done) = 4 events.
        let item_done: Vec<Value> = events
            .iter()
            .map(|e| parse_event(e))
            .filter(|(name, _)| name == "response.output_item.done")
            .map(|(_, v)| v)
            .collect();
        assert_eq!(item_done.len(), 2);
        assert_eq!(item_done[0]["output_index"], json!(1));
        assert_eq!(item_done[0]["item"]["call_id"], json!("call_a"));
        assert_eq!(item_done[1]["output_index"], json!(2));
        assert_eq!(item_done[1]["item"]["call_id"], json!("call_b"));
    }

    #[test]
    fn custom_tool_streams_input_not_arguments() {
        let mut ctx = BridgeToolContext::new();
        ctx.add_custom_tool_name("run_script");
        let mut store = ToolStateStore::new(ctx);
        let mut envelope = env();

        let events = store.push_delta(
            &mut envelope,
            &function_delta(0, "call_c", "run_script", "{\"input\": \"echo "),
            None,
        );
        // added + custom_tool_call_input.delta with the decoded prefix.
        assert_eq!(events.len(), 2);
        let (added_event, added) = parse_event(&events[0]);
        assert_eq!(added_event, "response.output_item.added");
        assert_eq!(added["item"]["type"], json!("custom_tool_call"));
        assert_eq!(added["item"]["input"], json!(""));
        let (delta_event, delta) = parse_event(&events[1]);
        assert_eq!(delta_event, "response.custom_tool_call_input.delta");
        assert_eq!(delta["delta"], json!("echo "));

        let events = store.push_delta(
            &mut envelope,
            &function_delta(0, "call_c", "run_script", "hi\"}"),
            None,
        );
        let (delta_event, delta) = parse_event(&events[0]);
        assert_eq!(delta_event, "response.custom_tool_call_input.delta");
        assert_eq!(delta["delta"], json!("hi"));

        let events = store.finalize(&mut envelope);
        let (done_event, done) = parse_event(&events[0]);
        assert_eq!(done_event, "response.custom_tool_call_input.done");
        assert_eq!(done["input"], json!("echo hi"));
        let (item_done_event, item_done) = parse_event(&events[1]);
        assert_eq!(item_done_event, "response.output_item.done");
        assert_eq!(item_done["item"]["type"], json!("custom_tool_call"));
        assert_eq!(item_done["item"]["input"], json!("echo hi"));
    }

    #[test]
    fn tool_search_call_arguments_object() {
        let mut ctx = BridgeToolContext::new();
        ctx.set_tool_search_enabled(true);
        let mut store = ToolStateStore::new(ctx);
        let mut envelope = env();

        store.push_delta(
            &mut envelope,
            &function_delta(
                0,
                "call_s",
                context::TOOL_SEARCH_PROXY_NAME,
                "{\"query\": \"rust\"}",
            ),
            None,
        );
        let events = store.finalize(&mut envelope);
        let (_, item_done) = parse_event(events.last().unwrap());
        assert_eq!(item_done["item"]["type"], json!("tool_search_call"));
        assert_eq!(item_done["item"]["execution"], json!("client"));
        assert_eq!(item_done["item"]["arguments"], json!({ "query": "rust" }));
        // tool_search items carry no name.
        assert!(item_done["item"].get("name").is_none());
    }

    #[test]
    fn synthetic_call_id_when_only_name_present() {
        let mut store = ToolStateStore::new(BridgeToolContext::new());
        let mut envelope = env();
        // Name but no id — identity is established (name is enough), so the
        // item opens immediately with a synthesized call id. Mirrors the
        // Python guard `state.call_id or state.name`.
        let events = store.push_delta(
            &mut envelope,
            &json!({ "index": 0, "function": { "name": "do_thing", "arguments": "{}" } }),
            None,
        );
        let (added_event, added) = parse_event(&events[0]);
        assert_eq!(added_event, "response.output_item.added");
        assert!(added["item"]["call_id"]
            .as_str()
            .unwrap()
            .starts_with("call_"));
        assert_eq!(added["item"]["name"], json!("do_thing"));

        // A tool call carrying neither id nor name never establishes identity,
        // so it emits nothing on push and is skipped without an added item.
        let mut store = ToolStateStore::new(BridgeToolContext::new());
        let mut envelope = env();
        let events = store.push_delta(
            &mut envelope,
            &json!({ "index": 0, "function": { "arguments": "{}" } }),
            None,
        );
        assert!(events.is_empty());
    }

    #[test]
    fn reasoning_content_attached_to_item() {
        let mut store = ToolStateStore::new(BridgeToolContext::new());
        let mut envelope = env();
        store.push_delta(
            &mut envelope,
            &function_delta(0, "call_r", "tool_r", "{}"),
            Some("thinking about it"),
        );
        let events = store.finalize(&mut envelope);
        let (_, item_done) = parse_event(events.last().unwrap());
        assert_eq!(
            item_done["item"]["reasoning_content"],
            json!("thinking about it")
        );
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut store = ToolStateStore::new(BridgeToolContext::new());
        let mut envelope = env();
        store.push_delta(
            &mut envelope,
            &function_delta(0, "call_1", "tool", "{}"),
            None,
        );
        let first = store.finalize(&mut envelope);
        assert!(!first.is_empty());
        let second = store.finalize(&mut envelope);
        assert!(second.is_empty());
    }

    /// Build a context registering a nested `db` namespace with `read`/`write`
    /// actions, returning `(context, flattened_chat_name)`.
    fn nested_ctx(strategy: &str) -> (BridgeToolContext, String) {
        let mut ctx = BridgeToolContext::new();
        let chat_name = ctx.add_nested_namespace_for_test("db", &["read", "write"], strategy);
        (ctx, chat_name)
    }

    #[test]
    fn nested_namespace_buffers_added_until_action_resolves() {
        let (ctx, chat_name) = nested_ctx("nested_oneof");
        let mut store = ToolStateStore::new(ctx);
        let mut envelope = env();

        // First delta carries the name but no parseable action yet — the call
        // is buffered, so nothing is emitted.
        let events = store.push_delta(
            &mut envelope,
            &function_delta(0, "call_n", &chat_name, "{\"acti"),
            None,
        );
        assert!(events.is_empty());

        // Completing the arguments resolves the action → deferred added + args.
        let events = store.push_delta(
            &mut envelope,
            &function_delta(0, "call_n", &chat_name, "on\":\"read\",\"id\":\"7\"}"),
            None,
        );
        let (added_event, added) = parse_event(&events[0]);
        assert_eq!(added_event, "response.output_item.added");
        assert_eq!(added["item"]["type"], json!("function_call"));
        // Name is restored to the concrete action, namespace to the original.
        assert_eq!(added["item"]["name"], json!("read"));
        assert_eq!(added["item"]["namespace"], json!("db"));

        let events = store.finalize(&mut envelope);
        let (_, item_done) = parse_event(events.last().unwrap());
        assert_eq!(item_done["item"]["name"], json!("read"));
        assert_eq!(item_done["item"]["namespace"], json!("db"));
        // The `action` envelope is unwrapped, leaving just the action args.
        assert_eq!(item_done["item"]["arguments"], json!("{\"id\":\"7\"}"));
    }

    #[test]
    fn nested_anyof_flattens_params_envelope() {
        let (ctx, chat_name) = nested_ctx("nested_anyof");
        let mut store = ToolStateStore::new(ctx);
        let mut envelope = env();

        store.push_delta(
            &mut envelope,
            &function_delta(
                0,
                "call_n",
                &chat_name,
                "{\"action\":\"read\",\"params\":{\"id\":\"7\"}}",
            ),
            None,
        );
        let events = store.finalize(&mut envelope);
        let (_, item_done) = parse_event(events.last().unwrap());
        assert_eq!(item_done["item"]["name"], json!("read"));
        assert_eq!(item_done["item"]["namespace"], json!("db"));
        // `params` is flattened back out into the top-level arguments.
        assert_eq!(item_done["item"]["arguments"], json!("{\"id\":\"7\"}"));
    }

    #[test]
    fn nested_namespace_degrades_when_action_never_resolves() {
        let (ctx, chat_name) = nested_ctx("nested_oneof");
        let mut store = ToolStateStore::new(ctx);
        let mut envelope = env();

        // A delta whose action never becomes valid stays buffered on push...
        let events = store.push_delta(
            &mut envelope,
            &function_delta(0, "call_n", &chat_name, "{\"other\":\"x\"}"),
            None,
        );
        assert!(events.is_empty());

        // ...and at finalize degrades to a namespace-level call.
        let events = store.finalize(&mut envelope);
        let (_, item_done) = parse_event(events.last().unwrap());
        assert_eq!(item_done["item"]["name"], json!("db"));
        assert_eq!(item_done["item"]["namespace"], json!("db"));
    }

    #[test]
    fn nested_namespace_degrades_on_buffer_overflow() {
        let (ctx, chat_name) = nested_ctx("nested_oneof");
        let mut store = ToolStateStore::new(ctx);
        let mut envelope = env();

        // Flood the buffer past NAMESPACE_BUFFER_MAX_BYTES without ever closing
        // the action selector: the call degrades mid-stream rather than stalling.
        let filler = format!(
            "{{\"junk\":\"{}",
            "x".repeat(NAMESPACE_BUFFER_MAX_BYTES + 16)
        );
        let events = store.push_delta(
            &mut envelope,
            &function_delta(0, "call_n", &chat_name, &filler),
            None,
        );
        let (added_event, added) = parse_event(&events[0]);
        assert_eq!(added_event, "response.output_item.added");
        assert_eq!(added["item"]["name"], json!("db"));
        assert_eq!(added["item"]["namespace"], json!("db"));
    }
}
