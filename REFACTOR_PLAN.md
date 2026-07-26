# codex-chat-bridge-rs — Clean-Room Refactor Plan (Highest-Standard, Idiomatic Rust)

Date: 2026-07-25. Supersedes the prior "surgical, Python-parity-anchored" plan.

> **Status (2026-07-26 re-audit): this plan is closed out.** Kept as a decision
> record, not a live task list. Per-phase outcome:
>
> | Phase | Outcome | Evidence |
> |-------|---------|----------|
> | 0 — freeze oracle + lib/bin split | done | lib + bin crate, golden fixtures committed |
> | 1 — typed id newtypes | **dropped** (see below) | no prefix scattering, no misuse surface |
> | 2 — interior protocol enums | done, different layout | `protocol.rs`: `InputItemKind`, `ToolCallKind`, `ContentPartType` (flat module, not `domain/`) |
> | 3 — split convert by direction | done | `convert/{responses_to_chat,chat_to_responses,semantics,message_normalization,tool_arguments}.rs` |
> | 4 — thiserror error type | done | `error.rs` with `#[derive(thiserror::Error)]` |
> | 5 — ToolSpec discriminants → enums | done, different layout | `ToolKind` in `context.rs` (no `tools/` dir) |
> | 6 — stream_tools re-lookup cleanup | done | `read_state` / `modify_state` / `with_ctx_state_mut` |
> | 7 — sweep + polish | done | module-level `dead_code` allows removed; provenance comments replaced |
>
> Where the delivered layout differs from the plan (Phases 2 and 5 landed as flat
> modules rather than `domain/` and `tools/` directories), the structural goal —
> exhaustive typed dispatch, single ownership per concern — is met; a directory
> level for 3 enums and one enum respectively would be indirection without
> benefit at this crate size.

## Directive

Refactor to the **best possible idiomatic-Rust state**, as if this were a fresh
top-tier greenfield project. **Do not consult the Python version for design.**
The `Mirrors X.py` comments are archaeology and get deleted, not preserved.

## The one non-negotiable constraint (ranks above "new project")

This is a **live production service** that new-api → channel 101 → clients depend
on *right now*. Clean-room means total freedom over **internal structure**; it
does **not** mean redesigning the external contract.

- **External behavior MUST NOT change.** The enforceable, semantically-correct
  invariant is **structural + semantic + SSE-event-order** equivalence — NOT
  raw byte-identity. Byte-identity is (a) impossible for streams (upstream token
  chunking is non-deterministic — the same prompt yields a different delta count
  per run, already proven), (b) unenforced by the oracle (`shape()` ignores
  scalar values and sorts keys), and (c) semantically unnecessary (JSON object
  key order carries no meaning). What must hold: same JSON envelope *structure*,
  same field *values*, same SSE event names in the same *order*, same status
  codes, same error shapes, same id prefixes.
- **The one place bytes DO matter is already guarded**: serialized tool-call
  `arguments` strings (`canonicalize_tool_arguments` → `to_string`) are
  wire-visible bytes the client re-parses; these are pinned byte-for-byte by the
  golden fixtures (`tests/parity/golden.json`, Value-equality) and must stay so.
- Therefore the **oracle flips**: it was "Python production". It is now the
  **pre-refactor Rust HEAD, frozen as tag `pre-refactor-baseline`**. The crate
  becomes its own differential oracle. `scripts/shadow_diff.py` runs the frozen
  baseline binary on :18091 vs the refactor build on :18095 and asserts
  structural + semantic + SSE-order equivalence. This safety net is what lets us
  refactor aggressively. **Gap to close in Phase 0**: the current oracle's
  `shape()` drops scalar values — add value-level assertions on the non-stream
  envelope + a full-response golden snapshot so semantic (not just structural)
  drift is caught.

## Why clean-room reverses earlier rejections

The prior review rejected interior strong-typing citing "would break Python
parity" and "丢字段风险". Both die under clean-room:

- `#[serde(flatten)] extra: Map<String,Value>` already exists on
  `ResponsesRequest` (types.rs:52) — proof that **tagged enum + flatten-extra +
  `Unknown` catch-all** preserves lossless passthrough *and* gains exhaustive
  compile-time dispatch. Best of both.
- "mirror Python's dict-in-the-middle" was the *only* argument for staying on
  `Value` in the interior. That argument is now void.

So the plan is materially more ambitious than the surgical one.

---

## Target architecture

```
src/
  lib.rs            # crate root: pub mod wiring, so tests/ + benches/ can link
  main.rs           # thin: parse env, build Router, serve, signal handling ONLY
  domain/
    ids.rs          # ResponseId, CallId, ItemId newtypes (typed, Display, prefix-encapsulated)
    protocol.rs     # tagged enums for input items / content parts / tool defs
    reasoning.rs    # CanonicalEffort, ReasoningBucket enums (already good, moved)
  responses_to_chat/    # request direction (was convert.rs input half)
  chat_to_responses/    # response direction (was convert.rs output half)
  message_normalization.rs   # the ~170-line cohesive subsystem, extracted
  tool_arguments.rs          # canonicalize + custom-input + nested-namespace, unified
  tools/            # context (registry) + stream_tools (incremental FSM)
  streaming/        # envelope, events, message/reasoning/inline-think FSMs, orchestrator
  http/             # middleware, error, routes
  upstream.rs       # unchanged transport
  config.rs metrics.rs sse.rs sha256.rs session_store.rs session_bridge.rs
tests/              # integration tests extracted from main.rs
benches/            # criterion micro-benches on the hot transform path (new)
```

---

## Phases (each: independent commit, all 6 gates green before next)

### Gate set (every phase)
1. `cargo build --release`
2. `cargo test` (all pass)
3. `cargo clippy --all-targets -- -D warnings` **and** a clean pass under
   `-W clippy::pedantic` triage (fix real ones, `#[allow]` w/ justification only
   where pedantic is wrong)
4. `cargo fmt --check`
5. **`scripts/shadow_diff.py` baseline-vs-refactor: multi-model ALL ALIGNED**
   (structural + semantic + SSE-order equivalence oracle — a fail reverts the
   phase). Value-level assertions must be on, not just `shape()`.
6. RSS/fd/panic health after a concurrent stress burst

### Phase 0 — Freeze the oracle + scaffolding
- Tag `pre-refactor-baseline` at HEAD `35bf058`; build that binary once, keep it
  at `bin/codex-chat-bridge-baseline` for the differential harness.
- Point `shadow_diff.py` at baseline:18091 vs refactor:18095 (both same env).
- **Harden the oracle first (it currently under-checks):** `shape()` drops
  scalar values, so a value-level regression (wrong status string, wrong id
  prefix, dropped field content) would pass silently. Add: (a) value-equality on
  the full non-stream `/v1/responses` envelope, (b) a committed full-response
  golden snapshot per model, (c) keep the collapsed-SSE-order check for streams.
  Do this before Phase 1 or the whole safety net is weaker than it looks.
- Delete 8 stale `#![allow(dead_code)]` (verified 0-warning without them).
- **lib.rs/main.rs split**; move `mod integration_tests` out of main.rs into
  `tests/`. This alone is a large idiomatic win (doc-tests, benches, faster
  incremental test builds).

### Phase 1 — Typed ids (domain/ids.rs) — **NOT DONE, deliberately dropped**

Original intent: `ResponseId`/`CallId`/`ItemId` newtypes encapsulating the
prefix logic (`resp_bridge_`, `fc_`, `ctc_`, `rs_`, `msg_`), banning raw
`format!("fc_{}")` scattering.

**Verdict (2026-07-26 re-audit): dropped as pure type-safety churn.** The two
problems newtypes would solve do not exist in this codebase:

1. **No scattering to ban.** `format!` with an id prefix appears in
   **zero** sites outside `id_gen.rs`. Prefix logic is already encapsulated at
   a single point — the stated goal is already met, by module boundary instead
   of by type.
2. **No misuse surface to close.** No function signature takes two `&str` id
   parameters, so the error a newtype prevents (passing a `call_id` where a
   `response_id` is expected) is not physically expressible today.

Cost would be 3 new types, ~11 signature changes, all call sites and tests
rewritten; benefit is zero catchable defects. This is the same call Phase 2's
DESIGN OVERRIDE already made: **types where we branch, plain values where we
pass through.** Phase 1 lands on the "pass through" side.

Revisit only if a future change introduces same-typed adjacent id parameters
or a second id-minting site.

### Phase 2 — Interior protocol enums (protocol.rs) — the big one
- Replace the stringly-typed `get("type").as_str()` dispatch sites with typed
  discriminants so every dispatch is an exhaustive `match` (a typo can't
  compile; a new protocol variant forces a decision at each site).
- **DESIGN OVERRIDE (supersedes the earlier `#[serde(tag)]` prescription):**
  use **classifier enums**, NOT `#[serde(tag="type")]` full-deserialize enums.
  Evidence that classifier is strictly better here:
  1. Input items are **consumed** (→ Chat messages), never re-emitted. The
     "lossless round-trip" concern only ever applied to the top-level
     `ResponsesRequest` (already handled via `#[serde(flatten)] extra`). There
     is no round-trip on interior items, so untagged-`Unknown(Value)` +
     round-trip proptests solve a problem these sites don't have.
  2. serde `from_value` is **stricter** than the current lenient reads
     (`get("text").as_str().unwrap_or("")` yields `""` on a number; a
     `text: String` field would error the whole item → behavior drift). To
     stay byte-exact the fields would have to be `Option<Value>`, which
     reintroduces the exact dynamic reads we're removing — net churn, no gain.
  3. The codebase already established the right pattern: `ResponseStatus` is a
     classifier enum (`from_finish_reason → enum → exhaustive match`) with the
     payload left as `Value`. That wins typo-can't-compile + exhaustive
     dispatch with **zero** behavior-drift risk. Phase 2 extends that house
     style; it does not invent a new serde-modeling layer.
- Concrete deliverables (`protocol.rs`):
  * `InputItemType` — the 7-way input-item dispatcher in `append_input_items`
    (folding in `handle_tool_call_item`); the big one. Exhaustive `match`,
    `Other` keeps the raw tag string for the transform-loss metric.
  * `ContentPartType` — centralizes the `matches!(typ,
    "input_text"|"output_text"|"text")` literal duplicated across 3+
    content-extraction sites (DRY + typo-safety).
- **Scope discipline (honest exclusions):**
  * `output_text_from_parts` filter and the `.all(type=="text")` predicate are
    checks on parts the bridge **itself just built** (self-constructed data),
    not external-type dispatch → left as-is.
  * `responses_tool_choice_to_chat` and the `role` match in
    `build_generic_message` are already clean total matches with an
    *intended* passthrough/downgrade catch-all; enum-izing buys no
    "decision-forced" safety and only adds indirection → left as-is.
  * Payload field reads stay on `Value` (lenient, byte-exact) — type-safety
    where we branch, passthrough where we don't.
- Highest risk of behavioral drift → lean hardest on the oracle after each site.

### Phase 3 — Split convert.rs by direction
- `responses_to_chat/` (23 fns) + `chat_to_responses/` (7 fns) as sibling
  modules; extract `message_normalization.rs` and `tool_arguments.rs`.
- Pure module-boundary moves; oracle guards behavior.

### Phase 4 — thiserror-based error type (http/error.rs)
- Replace hand-rolled `Display`/`Error` with `#[derive(thiserror::Error)]`
  (the dep is already declared). `envelope()` unchanged → wire error bytes
  identical (already covered by the `param` sorted-json tests + oracle Case 5).

### Phase 5 — ToolSpec discriminants → enums (tools/)
- `kind` (function/custom/tool_search/namespace) and `namespace_strategy`
  (nested_oneof/nested_anyof/flat) String → enum; all comparison sites become
  exhaustive matches.

### Phase 6 — stream_tools push_delta/finalize_state cleanup
- Collapse the ~9 borrow-checker-forced `get_mut(&index)` re-lookups via a
  `with_state(index, |s| …)` helper. Nested-buffer tri-state logic unchanged.

### Phase 7 — Sweep + polish
- Delete all `Mirrors X.py` comments (139) — replace with genuine module-purpose
  docs written fresh, not Python cross-refs.
- `benches/` criterion on the transform hot path (establish a perf floor).
- Final `-W clippy::pedantic` triage pass across the whole crate.
- Doc-test the public API of each domain module.

---

## Delivery gate
- All phases' commits on a `refactor/clean-room` branch.
- Full `shadow_diff.py` multi-model ALL ALIGNED on the final build.
- Coverage ≥ prior 87.69%; core modules stay 98-100%.
- Then fast-forward to master, rebuild release, atomic-mv deploy, verify via
  new-api channel-test + journal, exactly as the cutover runbook.

## Explicitly still NOT done (even at highest standard, these are net-negative)
- Redesigning the external API surface (violates the one hard constraint).
- Splitting genuine protocol dispatchers into artificial sub-fns (branch count =
  protocol arity is *inherent*; typed-enum match is the idiomatic end state, not
  further fragmentation).
- Config knobs / abstraction for hypothetical future providers (YAGNI).
