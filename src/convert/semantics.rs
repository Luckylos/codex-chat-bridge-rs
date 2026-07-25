//! Response semantics: status classification, finish-reason mapping, usage
//! normalization, and request-echo fields. Shared by both directions and the
//! streaming state machine.

use serde_json::{json, Map, Value};

pub(crate) const REQUEST_ECHO_FIELDS: &[&str] = &[
    "instructions",
    "max_output_tokens",
    "parallel_tool_calls",
    "previous_response_id",
    "reasoning",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "metadata",
];
pub(crate) fn echo_request_fields(
    response: &mut Map<String, Value>,
    original: Option<&Map<String, Value>>,
) {
    let Some(original) = original else { return };
    for &key in REQUEST_ECHO_FIELDS {
        if let Some(value) = original.get(key) {
            if !value.is_null() {
                response.insert(key.to_owned(), value.clone());
            }
        }
    }
}
/// A Responses top-level `status`, over its closed set of wire values. Replaces
/// the former stringly-typed status so a typo can't compile and the persist
/// guard is a total `match`. `as_str()` renders the exact protocol bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
}
impl ResponseStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Failed => "failed",
        }
    }

    /// Responses statuses safe to persist for `previous_response_id`
    /// continuation. `failed` / `incomplete` turns must not be saved, or a
    /// resume would replay a partial or invalid turn.
    pub(crate) fn is_persistable(self) -> bool {
        matches!(self, Self::Completed | Self::InProgress)
    }
}
pub(crate) fn response_status_from_finish_reason(finish_reason: Option<&str>) -> ResponseStatus {
    match finish_reason {
        Some("tool_calls") => ResponseStatus::InProgress,
        Some("length") | Some("content_filter") => ResponseStatus::Incomplete,
        _ => ResponseStatus::Completed,
    }
}
pub(crate) fn incomplete_reason_from_finish_reason(finish_reason: Option<&str>) -> Option<Value> {
    match finish_reason {
        Some("length") => Some(json!({ "reason": "max_output_tokens" })),
        Some("content_filter") => Some(json!({ "reason": "content_filter" })),
        _ => None,
    }
}
/// Whether a finalized response status is safe to persist for
/// `previous_response_id` continuation. `None` (never finalized) is not
/// persistable.
pub(crate) fn should_persist_response_status(status: Option<ResponseStatus>) -> bool {
    status.is_some_and(ResponseStatus::is_persistable)
}
/// Whether a Chat `finish_reason` maps to a persistable terminal state.
pub(crate) fn should_persist_finish_reason(finish_reason: Option<&str>) -> bool {
    response_status_from_finish_reason(finish_reason).is_persistable()
}
/// Extract the first-choice `finish_reason` from a Chat Completions body.
pub(crate) fn chat_finish_reason(chat_body: &Value) -> Option<String> {
    chat_body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}
pub(crate) fn map_chat_usage(usage: Option<&Value>) -> Value {
    let Some(usage) = usage.and_then(Value::as_object) else {
        return json!({ "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 });
    };
    let get = |k: &str| usage.get(k).and_then(Value::as_i64).unwrap_or(0);
    let prompt = get("prompt_tokens");
    let completion = get("completion_tokens");
    let input = get("input_tokens").max(prompt);
    let output = get("output_tokens").max(completion);
    let total = get("total_tokens").max(input + output);

    let mut result = Map::new();
    result.insert("input_tokens".to_owned(), json!(input));
    result.insert("output_tokens".to_owned(), json!(output));
    result.insert("total_tokens".to_owned(), json!(total));

    let input_details = usage
        .get("input_tokens_details")
        .or_else(|| usage.get("prompt_tokens_details"));
    if let Some(d) = input_details {
        if !d.is_null() {
            result.insert("input_tokens_details".to_owned(), d.clone());
        }
    }
    let output_details = usage
        .get("output_tokens_details")
        .or_else(|| usage.get("completion_tokens_details"));
    if let Some(d) = output_details {
        if !d.is_null() {
            result.insert("output_tokens_details".to_owned(), d.clone());
        }
    }
    Value::Object(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_from_finish_reason_matches_python_table() {
        assert_eq!(
            response_status_from_finish_reason(Some("tool_calls")).as_str(),
            "in_progress"
        );
        assert_eq!(
            response_status_from_finish_reason(Some("length")).as_str(),
            "incomplete"
        );
        assert_eq!(
            response_status_from_finish_reason(Some("content_filter")).as_str(),
            "incomplete"
        );
        assert_eq!(
            response_status_from_finish_reason(Some("stop")).as_str(),
            "completed"
        );
        assert_eq!(
            response_status_from_finish_reason(None).as_str(),
            "completed"
        );
    }
    #[test]
    fn incomplete_reason_only_for_length_and_filter() {
        assert_eq!(
            incomplete_reason_from_finish_reason(Some("length")),
            Some(json!({ "reason": "max_output_tokens" }))
        );
        assert_eq!(
            incomplete_reason_from_finish_reason(Some("content_filter")),
            Some(json!({ "reason": "content_filter" }))
        );
        assert_eq!(incomplete_reason_from_finish_reason(Some("stop")), None);
        assert_eq!(incomplete_reason_from_finish_reason(None), None);
    }
    #[test]
    fn usage_missing_yields_zeroed_object() {
        assert_eq!(
            map_chat_usage(None),
            json!({ "input_tokens": 0, "output_tokens": 0, "total_tokens": 0 })
        );
    }
    #[test]
    fn usage_takes_max_of_old_and_new_token_fields() {
        // NewAPI-style: both prompt_tokens and zero-filled input_tokens present.
        let usage = json!({
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "input_tokens": 0,
            "output_tokens": 0,
        });
        let out = map_chat_usage(Some(&usage));
        assert_eq!(out["input_tokens"], json!(10));
        assert_eq!(out["output_tokens"], json!(20));
        assert_eq!(out["total_tokens"], json!(30));
    }
    #[test]
    fn usage_prefers_explicit_total_when_larger() {
        let usage = json!({
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "total_tokens": 99,
        });
        let out = map_chat_usage(Some(&usage));
        assert_eq!(out["total_tokens"], json!(99));
    }
    #[test]
    fn usage_carries_token_details_from_either_naming() {
        let usage = json!({
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "prompt_tokens_details": { "cached_tokens": 3 },
            "completion_tokens_details": { "reasoning_tokens": 7 },
        });
        let out = map_chat_usage(Some(&usage));
        assert_eq!(out["input_tokens_details"], json!({ "cached_tokens": 3 }));
        assert_eq!(
            out["output_tokens_details"],
            json!({ "reasoning_tokens": 7 })
        );
    }
}
