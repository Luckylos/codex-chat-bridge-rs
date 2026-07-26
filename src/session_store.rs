//! In-memory session store for `previous_response_id` continuation.
//!

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::context::BridgeToolContext;

const DEFAULT_TTL: Duration = Duration::from_secs(3600);
const DEFAULT_MAX_SESSIONS: usize = 500;

/// A state snapshot of a single Responses API response.
#[derive(Clone)]
pub struct SessionRecord {
    pub messages: Vec<Value>,
    pub tool_context: BridgeToolContext,
    pub model: String,
    pub reasoning_cache: BTreeMap<String, String>,
}

impl SessionRecord {
    pub fn new(
        messages: Vec<Value>,
        tool_context: BridgeToolContext,
        model: String,
        reasoning_cache: BTreeMap<String, String>,
    ) -> Self {
        Self {
            messages,
            tool_context,
            model,
            reasoning_cache,
        }
    }
}

/// Internal stored entry: a record plus its last-access timestamp for TTL.
struct StoredEntry {
    record: SessionRecord,
    last_accessed_at: Instant,
}

/// In-memory session store, indexed by response_id.
pub struct SessionStore {
    ttl: Duration,
    max_sessions: usize,
    sessions: Mutex<BTreeMap<String, StoredEntry>>,
}

impl SessionStore {
    pub fn new(ttl: Duration, max_sessions: usize) -> Self {
        Self {
            ttl,
            max_sessions,
            sessions: Mutex::new(BTreeMap::new()),
        }
    }

    /// Lock the session map, recovering from a poisoned mutex instead of
    /// propagating the panic. Every mutation here is a single statement over a
    /// structurally-valid `BTreeMap`, so a thread that panicked mid-access left
    /// the map intact; poisoning must not turn every subsequent request into a
    /// panic which, under systemd `Restart=on-failure`, becomes a crash loop.
    /// Mirrors the recovery idiom already used in `config.rs`.
    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, StoredEntry>> {
        self.sessions.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Look up a session; expired entries are treated as missing. A live access
    /// renews the entry's TTL. The returned record is a clone, so the caller
    /// can mutate it freely without touching stored state.
    pub fn get(&self, response_id: &str) -> Option<SessionRecord> {
        let mut sessions = self.lock();
        let now = Instant::now();
        let entry = sessions.get_mut(response_id)?;
        if now.duration_since(entry.last_accessed_at) > self.ttl {
            sessions.remove(response_id);
            return None;
        }
        entry.last_accessed_at = now;
        Some(entry.record.clone())
    }

    /// Save session state, renewing its timestamp and triggering lazy cleanup
    /// plus cap enforcement. Stores a clone so later mutation of the passed
    /// record cannot reach into the store.
    pub fn save(&self, response_id: &str, record: SessionRecord) {
        let mut sessions = self.lock();
        let now = Instant::now();
        sessions.insert(
            response_id.to_owned(),
            StoredEntry {
                record,
                last_accessed_at: now,
            },
        );
        self.enforce_cap(&mut sessions);
        self.cleanup(&mut sessions, now);
    }

    /// Drop entries whose TTL has elapsed.
    fn cleanup(&self, sessions: &mut BTreeMap<String, StoredEntry>, now: Instant) {
        let stale: Vec<String> = sessions
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.last_accessed_at) > self.ttl)
            .map(|(id, _)| id.clone())
            .collect();
        for id in stale {
            sessions.remove(&id);
        }
    }

    /// Evict the least-recently-accessed entries until under the cap.
    fn enforce_cap(&self, sessions: &mut BTreeMap<String, StoredEntry>) {
        while sessions.len() > self.max_sessions {
            let oldest = sessions
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed_at)
                .map(|(id, _)| id.clone());
            match oldest {
                Some(id) => {
                    sessions.remove(&id);
                }
                None => break,
            }
        }
    }

    #[cfg(test)]
    pub fn active_count(&self) -> usize {
        self.lock().len()
    }
}

static GLOBAL_STORE: OnceLock<SessionStore> = OnceLock::new();

/// The process-global session store singleton.
pub fn get_session_store() -> &'static SessionStore {
    GLOBAL_STORE.get_or_init(|| SessionStore::new(DEFAULT_TTL, DEFAULT_MAX_SESSIONS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn record(model: &str) -> SessionRecord {
        SessionRecord::new(
            vec![json!({ "role": "user", "content": "hi" })],
            BridgeToolContext::new(),
            model.to_owned(),
            BTreeMap::new(),
        )
    }

    #[test]
    fn save_then_get_roundtrips() {
        let store = SessionStore::new(DEFAULT_TTL, DEFAULT_MAX_SESSIONS);
        store.save("resp_1", record("gpt-x"));
        let got = store.get("resp_1").expect("record present");
        assert_eq!(got.model, "gpt-x");
        assert_eq!(got.messages.len(), 1);
    }

    #[test]
    fn missing_id_is_none() {
        let store = SessionStore::new(DEFAULT_TTL, DEFAULT_MAX_SESSIONS);
        assert!(store.get("nope").is_none());
    }

    #[test]
    fn expired_entry_is_evicted_on_get() {
        let store = SessionStore::new(Duration::from_millis(0), DEFAULT_MAX_SESSIONS);
        store.save("resp_1", record("gpt-x"));
        // TTL is zero, so any elapsed time expires the entry.
        std::thread::sleep(Duration::from_millis(1));
        assert!(store.get("resp_1").is_none());
        assert_eq!(store.active_count(), 0);
    }

    #[test]
    fn cap_evicts_when_exceeded() {
        let store = SessionStore::new(DEFAULT_TTL, 2);
        store.save("a", record("m"));
        std::thread::sleep(Duration::from_millis(1));
        store.save("b", record("m"));
        std::thread::sleep(Duration::from_millis(1));
        store.save("c", record("m"));
        // Cap is 2, so the oldest ("a") is evicted.
        assert_eq!(store.active_count(), 2);
        assert!(store.get("a").is_none());
        assert!(store.get("b").is_some());
        assert!(store.get("c").is_some());
    }

    #[test]
    fn stored_record_is_isolated_from_later_mutation() {
        let store = SessionStore::new(DEFAULT_TTL, DEFAULT_MAX_SESSIONS);
        store.save("resp_1", record("gpt-x"));
        let mut got = store.get("resp_1").expect("record present");
        got.messages
            .push(json!({ "role": "assistant", "content": "mutated" }));
        // The stored copy is untouched.
        let again = store.get("resp_1").expect("record present");
        assert_eq!(again.messages.len(), 1);
    }

    #[test]
    fn poisoned_mutex_recovers_instead_of_propagating_panic() {
        // Simulate a thread panicking while holding the lock: the mutex is left
        // poisoned. Subsequent access must recover the guard rather than panic —
        // otherwise every later request would panic and, under systemd
        // Restart=on-failure, turn into a crash loop.
        use std::sync::Arc;

        let store = Arc::new(SessionStore::new(DEFAULT_TTL, DEFAULT_MAX_SESSIONS));
        store.save("resp_1", record("gpt-x"));

        let poisoner = Arc::clone(&store);
        let handle = std::thread::spawn(move || {
            let _guard = poisoner.lock();
            panic!("poison the session mutex");
        });
        assert!(handle.join().is_err(), "the poisoning thread must panic");

        // The map was structurally intact when the panic hit, so the stored
        // record survives and access still works.
        let got = store.get("resp_1").expect("record survives poisoning");
        assert_eq!(got.model, "gpt-x");
        store.save("resp_2", record("m"));
        assert_eq!(store.active_count(), 2);
    }
}
