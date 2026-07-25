//! Python↔Rust byte-level parity test.
//!
//! Loads the golden fixture generated from the authoritative Python bridge
//! (`tests/parity/generate_golden.py`) and asserts the Rust port produces
//! byte-identical output for every deterministic pure transform. Regenerate the
//! fixture after any intentional behavior change:
//!
//! ```text
//! uv run --project /opt/codex-chat-bridge python \
//!     tests/parity/generate_golden.py
//! ```

#![cfg(test)]

use serde_json::Value;

/// The fixture is embedded at compile time so the test needs no runtime IO and
/// fails loudly (a build error) if the file is ever removed.
const GOLDEN: &str = include_str!("../tests/parity/golden.json");

/// Drive one Rust transform for a fixture record, returning its output as the
/// JSON value shape the Python side recorded (string, or null for the
/// `Option`-returning partial-input helper).
fn run_case(func: &str, input: &Value) -> Value {
    match func {
        "sanitize_string" => {
            Value::String(crate::sanitize::sanitize_string(input.as_str().unwrap()))
        }
        "flatten_namespace_tool_name" => {
            let pair = input.as_array().unwrap();
            let ns = pair[0].as_str().unwrap();
            let name = pair[1].as_str().unwrap();
            Value::String(crate::context::flatten_namespace_tool_name(ns, name))
        }
        "short_sha256_hex" => Value::String(crate::sha256::sha256_hex_16(
            input.as_str().unwrap().as_bytes(),
        )),
        "custom_tool_input_from_chat_arguments" => Value::String(
            crate::context::custom_tool_input_from_chat_arguments(input.as_str().unwrap()),
        ),
        "partial_custom_tool_input_from_chat_arguments" => {
            match crate::context::partial_custom_tool_input_from_chat_arguments(
                input.as_str().unwrap(),
            ) {
                Some(s) => Value::String(s),
                None => Value::Null,
            }
        }
        "canonicalize_tool_arguments" => {
            // The Python generator preserves the input's JSON type: a string
            // vector arrives as a JSON string, a dict/list as that value, and
            // `None` as JSON null. `canonicalize_tool_arguments` takes the value
            // by reference, so a null input maps to `None`.
            let arg = match input {
                Value::Null => None,
                other => Some(other),
            };
            Value::String(crate::convert::canonicalize_tool_arguments(arg))
        }
        other => panic!("unknown parity function in fixture: {other}"),
    }
}

#[test]
fn rust_output_matches_python_golden() {
    let records: Vec<Value> = serde_json::from_str(GOLDEN).expect("golden.json parses");
    assert!(!records.is_empty(), "golden fixture is empty");

    let mut mismatches = Vec::new();
    for record in &records {
        let func = record["fn"].as_str().expect("record has fn");
        let input = &record["input"];
        let expected = &record["output"];
        let actual = run_case(func, input);
        if &actual != expected {
            mismatches.push(format!(
                "fn={func} input={input} expected={expected} actual={actual}"
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "Rust output diverged from the Python golden fixture in {} case(s):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
