//! codex-chat-bridge (Rust) — binary entry point.
//!
//! All logic lives in the `codex_chat_bridge` library crate; this binary is a
//! thin wrapper that owns only the async runtime and delegates to
//! [`codex_chat_bridge::run`]. Keeping the crate as a lib + bin lets the
//! integration tests, doc-tests, and benches link the same code.

#[tokio::main]
async fn main() {
    codex_chat_bridge::run().await;
}
