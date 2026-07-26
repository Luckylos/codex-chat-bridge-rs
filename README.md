# codex-chat-bridge (Rust)

Transparent protocol bridge: a **Responses-speaking client** talks to a
**Chat Completions upstream**. The bridge accepts OpenAI *Responses API* traffic
(`/v1/responses`), converts it to *Chat Completions* on the way out, and
converts the reply — including streaming SSE — back to Responses events on the
way in. `/v1/chat/completions` is relayed straight through.

```
client (Responses API) → codex-chat-bridge :18090 → upstream (Chat Completions)
```

It targets a single upstream (typically a NewAPI entrypoint that handles
provider aggregation and model routing); the bridge itself does only protocol
conversion.

## Status

This Rust implementation is the **production service** on `:18090`. It is a
byte-for-byte-behavior port of the original Python `codex-chat-bridge`, which is
now **retired** and kept read-only as a historical reference at
[`Luckylos/codex-chat-bridge`](https://github.com/Luckylos/codex-chat-bridge).
No Python process runs in production; the systemd unit `codex-chat-bridge`
executes the Rust binary.

The two repositories are independent git trees with no shared history — the
Python repo is the archive of the prior generation, this repo is the current
one.

## Why a rewrite

The Python service was correct but the conversion pipeline (Responses ⇄ Chat,
streaming state machines, reasoning/tool lifecycle) is the kind of stateful,
hot-path logic where Rust's type system pays for itself: illegal states are made
unrepresentable rather than defended against at runtime.

- **Closed-set enums over stringly-typed state.** Message/reasoning lifecycles
  (`NotStarted → Open → Done`) and response status are enums with total
  matches — a typo can't compile, and a missed variant is a build error.
- **Strong types at the boundary, `Value` only where the protocol is genuinely
  polymorphic.** Request entrypoints are typed structs; conversion stops at
  `serde_json::Value` only where the wire format is legitimately open-ended.
- **One source of truth per concern.** Session persistence, id generation, and
  input-item iteration each have a single implementation the rest of the crate
  calls.

The design rationale and the module-by-module review live in
[`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) and
[`REFACTOR_PLAN.md`](REFACTOR_PLAN.md).

## HTTP surface

| Method + path                | Purpose                                              |
|------------------------------|------------------------------------------------------|
| `GET /health`                | Liveness + upstream connectivity probe               |
| `GET /metrics`               | Phase-timing / operational metrics                   |
| `GET /v1/models`             | Upstream model catalogue passthrough                 |
| `POST /v1/responses`         | Responses → Chat conversion (streaming + non-stream) |
| `POST /v1/responses/compact` | Same handler, compaction entrypoint                  |
| `POST /v1/chat/completions`  | Chat Completions relay (verbatim forward + retries)  |

## Configuration (environment)

| Env var                          | Default        | Purpose                                                    |
|----------------------------------|----------------|------------------------------------------------------------|
| `BRIDGE_UPSTREAM_BASE_URL`       | (required)     | Upstream entrypoint                                        |
| `BRIDGE_UPSTREAM_API_KEY`        | empty          | Upstream API key                                           |
| `BRIDGE_UPSTREAM_TIMEOUT_SECONDS`| `60`           | Upstream request timeout                                   |
| `BRIDGE_UPSTREAM_STREAMING`      | `true`         | Request the upstream in streaming mode                     |
| `BRIDGE_UPSTREAM_MAX_RETRIES`    | `2`            | Max 400-compat / transport retry attempts                  |
| `BRIDGE_MAX_CONCURRENT_REQUESTS` | `20`           | Concurrency limit (permit-gated)                           |
| `BRIDGE_MAX_BODY_BYTES`          | `10485760`     | Request-body ceiling (10 MiB); over-limit → 413            |
| `BRIDGE_UNSUPPORTED_TOOL_POLICY` | `ignore`       | Unmappable Responses builtin-tool policy (`ignore`/`reject`/`error`/`passthrough`; invalid → `ConfigError` at startup) |
| `BRIDGE_INBOUND_API_KEYS`        | empty          | Inbound auth keys (comma-separated). When set, `/v1/*` requires `Authorization: Bearer <key>`; `/health` and `/metrics` are unauthenticated. |
| `BRIDGE_HOST`                    | `0.0.0.0`      | Listen host                                                |
| `BRIDGE_PORT`                    | `18090`        | Listen port                                                |

**Security note:** with `BRIDGE_INBOUND_API_KEYS` empty there is **no inbound
authentication** — the bridge will forward for any reachable client using the
upstream key. Deploy it inside a trusted network, or set inbound keys.

## Build, test, run

```sh
cargo build --release
cargo test                          # 289 tests (unit + parity golden + proptests)
cargo fmt --all -- --check          # non-negotiable in CI
cargo clippy --all-targets -- -D warnings

systemctl restart codex-chat-bridge
curl -s localhost:18090/health
```

CI (`.github/workflows/ci.yml`) gates every push and PR on `fmt` + `clippy -D
warnings` + the full test suite.

## Correctness methodology

External behavior must not change relative to the retired Python service. The
enforced invariant is **structural + semantic + SSE-event-order equivalence**,
not raw byte-identity (stream token chunking is non-deterministic upstream, so
byte-identity is neither achievable nor meaningful for streams).

- **Parity golden tests** (`src/parity_golden.rs`, `src/parity_stream_golden.rs`)
  pin the conversion output against captured fixtures in `tests/parity/`.
- **Property tests** (`src/proptests.rs`) fuzz the converters for invariants
  that must hold across arbitrary inputs.
- **Retry/compat orchestration tests** drive the upstream client against a mock
  server to prove the transport + compat retry loops behave as specified.
