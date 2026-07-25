//! Bridge tool-context: the request-scoped tool registry.
//!

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

use crate::config::UnsupportedToolPolicy;

/// The Chat Completions function name that proxies a Responses `tool_search`
/// call. Matches the Python bridge's `bridge_context.constants`.
pub const TOOL_SEARCH_PROXY_NAME: &str = "tool_search";

/// The single Chat function argument used to carry a custom tool's freeform
/// input. Matches the Python bridge's `CUSTOM_TOOL_INPUT_FIELD`.
pub const CUSTOM_TOOL_INPUT_FIELD: &str = "input";

/// Maximum length of a Chat Completions function name. Matches the Python
/// bridge's `CHAT_TOOL_NAME_MAX_LEN`.
pub const CHAT_TOOL_NAME_MAX_LEN: usize = 64;

/// Responses hosted-tool types the bridge cannot map to a Chat function.
/// Matches the Python bridge's `HOSTED_TOOL_TYPES`.
const HOSTED_TOOL_TYPES: &[&str] = &[
    "web_search",
    "file_search",
    "computer_use",
    "computer_use_preview",
    "code_interpreter",
    "mcp",
];

fn is_hosted_tool_type(tool_type: &str) -> bool {
    HOSTED_TOOL_TYPES.contains(&tool_type)
}

/// Flatten `namespace` + `name` into a single Chat function name.
///
/// The natural join is `namespace__name`. When that exceeds
/// [`CHAT_TOOL_NAME_MAX_LEN`], it is truncated and disambiguated with a
/// `__<16-hex-sha256>` suffix so distinct long names never collide.
pub fn flatten_namespace_tool_name(namespace: &str, name: &str) -> String {
    let full_name = format!("{namespace}__{name}");
    // Python measures length and slices by Unicode code point, not UTF-8 byte,
    // so mirror that: count/take `char`s. For ASCII names (the common case)
    // this is identical to byte semantics.
    if full_name.chars().count() <= CHAT_TOOL_NAME_MAX_LEN {
        return full_name;
    }
    let suffix = format!("__{}", crate::sha256::sha256_hex_16(full_name.as_bytes()));
    let prefix_len = CHAT_TOOL_NAME_MAX_LEN - suffix.chars().count();
    let prefix: String = full_name.chars().take(prefix_len).collect();
    format!("{prefix}{suffix}")
}

/// Extract a tool name from a Responses tool value: a bare string, a flat
/// `{name}`, or a nested `{function:{name}}`.
fn tool_name_from_value(tool: &Value) -> Option<String> {
    if let Some(s) = tool.as_str() {
        let candidate = s.trim();
        return (!candidate.is_empty()).then(|| candidate.to_owned());
    }
    let obj = tool.as_object()?;
    if let Some(name) = obj
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
    {
        return Some(name.to_owned());
    }
    obj.get("function")
        .and_then(Value::as_object)
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_owned)
}

// --------------------------------------------------------------------------- //
// ToolSpec
// --------------------------------------------------------------------------- //

/// The origin kind of a registered Chat tool. Replaces the former stringly-typed
/// `kind` field so an invalid value cannot be constructed and every match is
/// total. `as_str()` renders the exact Python `ToolSpec.kind` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    Function,
    Custom,
    ToolSearch,
    Namespace,
}

/// The schema strategy for a namespace tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamespaceStrategy {
    NestedOneof,
    NestedAnyof,
    Flat,
}

impl NamespaceStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NestedOneof => "nested_oneof",
            Self::NestedAnyof => "nested_anyof",
            Self::Flat => "flat",
        }
    }

    /// Parse a (lowercased) strategy string, defaulting unknown/absent to
    /// `Flat`. The caller warns on the unrecognized-but-non-flat case, so the
    /// raw string is inspected there, not here.
    fn from_str_or_flat(s: &str) -> Self {
        match s {
            "nested_oneof" => Self::NestedOneof,
            "nested_anyof" => Self::NestedAnyof,
            _ => Self::Flat,
        }
    }

    /// True for the action-selector strategies (as opposed to `Flat`).
    fn is_nested(self) -> bool {
        matches!(self, Self::NestedOneof | Self::NestedAnyof)
    }
}

/// The origin of a registered Chat tool, used to translate upstream tool calls
/// back into the correct Responses item shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    /// The tool's origin kind.
    pub kind: ToolKind,
    /// The original Responses-side tool name.
    pub name: String,
    /// The namespace this tool belongs to, when any.
    pub namespace: Option<String>,
    /// The namespace schema strategy, when `kind == Namespace`.
    pub namespace_strategy: Option<NamespaceStrategy>,
    /// Sub-tool action names when `kind == Namespace`.
    pub actions: Option<Vec<String>>,
}

impl ToolSpec {
    fn function(name: impl Into<String>, namespace: Option<String>) -> Self {
        Self {
            kind: ToolKind::Function,
            name: name.into(),
            namespace,
            namespace_strategy: None,
            actions: None,
        }
    }

    fn custom(name: impl Into<String>) -> Self {
        Self {
            kind: ToolKind::Custom,
            name: name.into(),
            namespace: None,
            namespace_strategy: None,
            actions: None,
        }
    }

    fn tool_search() -> Self {
        Self {
            kind: ToolKind::ToolSearch,
            name: TOOL_SEARCH_PROXY_NAME.to_owned(),
            namespace: None,
            namespace_strategy: None,
            actions: None,
        }
    }

    /// True when this is a namespace tool using a nested (action-selector)
    /// strategy.
    pub fn is_nested_namespace(&self) -> bool {
        self.kind == ToolKind::Namespace
            && self
                .namespace_strategy
                .map(NamespaceStrategy::is_nested)
                .unwrap_or(false)
    }
}

// --------------------------------------------------------------------------- //
// Nested namespace argument resolution
// --------------------------------------------------------------------------- //

/// The result of normalizing a nested-namespace tool call's arguments: the
/// selected action name (when it matched a known action) and the argument JSON
/// with the `action`/`params` envelope unwrapped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedNamespaceResolution {
    pub action_name: Option<String>,
    pub normalized_arguments: String,
}

/// Normalize nested-namespace arguments and extract a validated action name.
///
/// The namespace schema encodes the concrete action in an `action` field; for
/// `nested_anyof` the real arguments live under `params` and are flattened back
/// out. When the payload is incomplete/malformed or the action is unknown, the
/// original argument string is preserved.
pub fn resolve_nested_namespace_arguments(
    spec: &ToolSpec,
    arguments_json: &str,
) -> NestedNamespaceResolution {
    if arguments_json.is_empty() {
        return NestedNamespaceResolution {
            action_name: None,
            normalized_arguments: arguments_json.to_owned(),
        };
    }
    let Ok(Value::Object(mut parsed)) = serde_json::from_str::<Value>(arguments_json) else {
        return NestedNamespaceResolution {
            action_name: None,
            normalized_arguments: arguments_json.to_owned(),
        };
    };

    let raw_action = parsed.remove("action");
    if spec.namespace_strategy == Some(NamespaceStrategy::NestedAnyof) {
        if let Some(Value::Object(params)) = parsed.remove("params") {
            for (k, v) in params {
                parsed.insert(k, v);
            }
        }
    }

    let normalized_arguments =
        serde_json::to_string(&Value::Object(parsed)).unwrap_or_else(|_| "{}".to_owned());
    let action_name = raw_action
        .and_then(|a| a.as_str().map(str::to_owned))
        .filter(|a| {
            spec.actions
                .as_ref()
                .map(|acts| acts.iter().any(|x| x == a))
                .unwrap_or(false)
        });
    NestedNamespaceResolution {
        action_name,
        normalized_arguments,
    }
}

// --------------------------------------------------------------------------- //
// BridgeToolContext
// --------------------------------------------------------------------------- //

/// Request-scoped tool registry. Owns the upstream `chat_tools` array and the
/// forward/reverse namespace maps.
#[derive(Debug, Default, Clone)]
pub struct BridgeToolContext {
    custom_tool_names: BTreeSet<String>,
    tool_search_enabled: bool,
    chat_name_to_spec: BTreeMap<String, ToolSpec>,
    chat_tools: Vec<Value>,
    seen_chat_names: BTreeSet<String>,
    namespace_name_to_chat_name: BTreeMap<(String, String), String>,
    registered_namespaces: BTreeSet<String>,
}

impl BridgeToolContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// The upstream Chat Completions `tools` array. Empty when no tools were
    /// declared.
    pub fn chat_tools(&self) -> &[Value] {
        &self.chat_tools
    }

    pub fn is_custom_tool(&self, chat_name: Option<&str>) -> bool {
        matches!(chat_name, Some(name) if !name.is_empty() && self.custom_tool_names.contains(name))
    }

    /// Test-only: register a custom tool by name through the real registration
    /// path. Cross-module tests (e.g. `stream_tools`) build a context this way
    /// without hand-constructing a full Responses tool payload.
    #[cfg(test)]
    pub fn add_custom_tool_name(&mut self, name: &str) {
        self.add_custom_tool(&json!({ "type": "custom", "name": name }));
    }

    /// Test-only: enable the tool-search proxy through the real registration
    /// path.
    #[cfg(test)]
    pub fn set_tool_search_enabled(&mut self, enabled: bool) {
        if enabled {
            self.add_tool_search_tool();
        }
    }

    /// Test-only: register a nested-namespace tool through the real
    /// registration path and return the flattened Chat name upstream sees.
    /// Cross-module tests (e.g. `stream_tools`) build a nested-namespace
    /// context this way without hand-constructing a full Responses payload.
    #[cfg(test)]
    pub fn add_nested_namespace_for_test(
        &mut self,
        namespace: &str,
        actions: &[&str],
        strategy: &str,
    ) -> String {
        let children: Vec<Value> = actions
            .iter()
            .map(|action| {
                json!({
                    "type": "function",
                    "name": action,
                    "parameters": { "type": "object", "properties": {} },
                })
            })
            .collect();
        self.add_namespace_tool(&json!({
            "type": "namespace",
            "name": namespace,
            "strategy": strategy,
            "tools": children,
        }));
        flatten_namespace_tool_name(namespace, namespace)
    }

    pub fn is_tool_search(&self, chat_name: Option<&str>) -> bool {
        self.tool_search_enabled
            && matches!(chat_name, Some(name) if name == TOOL_SEARCH_PROXY_NAME)
    }

    /// Look up the [`ToolSpec`] registered under a Chat function name.
    pub fn lookup_chat_name(&self, chat_name: Option<&str>) -> Option<&ToolSpec> {
        chat_name.and_then(|name| self.chat_name_to_spec.get(name))
    }

    /// Forward map: the Chat function name for a Responses `(name, namespace)`.
    /// A registered namespaced tool uses its (possibly hash-suffixed) flattened
    /// name; an unregistered namespaced name falls back to a fresh flatten; a
    /// plain name passes through.
    pub fn chat_name_for_function(&self, name: &str, namespace: Option<&str>) -> String {
        if let Some(ns) = namespace {
            if let Some(chat_name) = self
                .namespace_name_to_chat_name
                .get(&(ns.to_owned(), name.to_owned()))
            {
                return chat_name.clone();
            }
            return flatten_namespace_tool_name(ns, name);
        }
        name.to_owned()
    }

    /// Reverse map: recover the original `(namespace, name)` from a Chat
    /// function name. Falls back to splitting on the last `__` only when the
    /// prefix is a registered namespace, so a plain name containing `__` is not
    /// mis-parsed.
    pub fn restore_namespace_and_name(&self, chat_name: &str) -> (Option<String>, String) {
        if let Some(spec) = self.chat_name_to_spec.get(chat_name) {
            return (spec.namespace.clone(), spec.name.clone());
        }
        if let Some((ns, n)) = chat_name.rsplit_once("__") {
            if !ns.is_empty() && !n.is_empty() && self.registered_namespaces.contains(ns) {
                return (Some(ns.to_owned()), n.to_owned());
            }
        }
        (None, chat_name.to_owned())
    }

    /// Fold another context's registrations into this one. Used to merge a
    /// continuation turn's freshly-declared tools into the stored session
    /// context. Hosted (passthrough) tools are copied verbatim; every other
    /// tool is re-registered by its spec.
    pub fn merge(&mut self, other: &BridgeToolContext) {
        for chat_tool in &other.chat_tools {
            if let Some(tool_type) = chat_tool.get("type").and_then(Value::as_str) {
                if is_hosted_tool_type(tool_type) {
                    if !self.chat_tools.contains(chat_tool) {
                        self.chat_tools.push(chat_tool.clone());
                    }
                    continue;
                }
            }
            let chat_name = chat_tool
                .get("function")
                .and_then(Value::as_object)
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty());
            let Some(chat_name) = chat_name else { continue };
            if let Some(spec) = other.chat_name_to_spec.get(chat_name) {
                self.register_chat_tool(chat_name, spec.clone(), chat_tool.clone());
            }
        }
        if other.tool_search_enabled {
            self.add_tool_search_tool();
        }
    }

    /// Register a normalized Chat tool into the registry, deduplicating by Chat
    /// name.
    fn register_chat_tool(&mut self, chat_name: &str, spec: ToolSpec, chat_tool: Value) {
        if chat_name.trim().is_empty() || self.seen_chat_names.contains(chat_name) {
            return;
        }
        self.seen_chat_names.insert(chat_name.to_owned());
        if let Some(ns) = &spec.namespace {
            self.namespace_name_to_chat_name
                .insert((ns.clone(), spec.name.clone()), chat_name.to_owned());
        }
        if spec.kind == ToolKind::Custom {
            self.custom_tool_names.insert(chat_name.to_owned());
        }
        if spec.kind == ToolKind::ToolSearch {
            self.tool_search_enabled = true;
        }
        self.chat_name_to_spec.insert(chat_name.to_owned(), spec);
        self.chat_tools.push(chat_tool);
    }

    /// Register a Responses function tool (flat or nested `function`), applying
    /// namespace flattening.
    fn add_function_tool(&mut self, tool: &Value, namespace: Option<&str>) {
        let function = tool
            .get("function")
            .filter(|f| f.is_object())
            .unwrap_or(tool);
        let Some(name) = tool_name_from_value(function) else {
            return;
        };
        let chat_name = self.chat_name_for_function(&name, namespace);
        // A function tool is normally an object; a bare-string tool leaves
        // `source` as None, in which case parameters/description are absent
        // (an empty map's `.get` and `None` behave identically here).
        let source = function.as_object();
        let parameters = source
            .and_then(|s| s.get("parameters"))
            .filter(|p| p.is_object())
            .cloned()
            .unwrap_or_else(|| json!({}));
        let chat_tool = json!({
            "type": "function",
            "function": {
                "name": chat_name,
                "description": source
                    .and_then(|s| s.get("description"))
                    .cloned()
                    .unwrap_or(Value::Null),
                "parameters": parameters,
            },
        });
        let spec = ToolSpec::function(name, namespace.map(str::to_owned));
        self.register_chat_tool(&chat_name, spec, chat_tool);
    }

    /// Register a custom tool as a single-`input`-string Chat function.
    fn add_custom_tool(&mut self, tool: &Value) {
        let Some(name) = tool_name_from_value(tool) else {
            return;
        };
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("Custom Codex tool.");
        let chat_tool = json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        CUSTOM_TOOL_INPUT_FIELD: {
                            "type": "string",
                            "description": "Input to pass to the custom Codex tool.",
                        }
                    },
                    "required": [CUSTOM_TOOL_INPUT_FIELD],
                },
            },
        });
        self.register_chat_tool(&name, ToolSpec::custom(name.clone()), chat_tool);
    }

    /// Register the synthetic tool-search proxy function.
    fn add_tool_search_tool(&mut self) {
        let chat_tool = json!({
            "type": "function",
            "function": {
                "name": TOOL_SEARCH_PROXY_NAME,
                "description": "Search and load Codex tools, plugins, connectors, and MCP namespaces for the current task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for tools or connectors to load.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tool groups to return.",
                        },
                    },
                    "required": ["query"],
                },
            },
        });
        self.register_chat_tool(TOOL_SEARCH_PROXY_NAME, ToolSpec::tool_search(), chat_tool);
    }

    /// Register a namespace tool, flattening children (flat strategy) or merging
    /// them into a single action-selector schema (nested strategies).
    fn add_namespace_tool(&mut self, namespace_tool: &Value) {
        let namespace = namespace_tool.get("name").and_then(Value::as_str);
        let children = namespace_tool
            .get("tools")
            .or_else(|| namespace_tool.get("children"))
            .and_then(Value::as_array);
        let (Some(namespace), Some(children)) = (namespace, children) else {
            return;
        };
        if namespace.trim().is_empty() {
            return;
        }
        self.registered_namespaces.insert(namespace.to_owned());

        let raw_strategy = namespace_tool
            .get("strategy")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| "flat".to_owned());
        let strategy = NamespaceStrategy::from_str_or_flat(&raw_strategy);

        if strategy.is_nested() {
            self.add_nested_namespace_tool(namespace, children, strategy);
            return;
        }
        if raw_strategy != "flat" {
            tracing::warn!(
                "Unrecognized namespace strategy {raw_strategy:?}, falling back to flat"
            );
        }
        for child in children {
            if child.get("type").and_then(Value::as_str) == Some("function") {
                self.add_function_tool(child, Some(namespace));
            }
        }
    }

    /// Build and register a single merged Chat tool for a nested namespace.
    fn add_nested_namespace_tool(
        &mut self,
        namespace: &str,
        children: &[Value],
        strategy: NamespaceStrategy,
    ) {
        let (sub_tools, action_names) = collect_function_children(children);
        if sub_tools.is_empty() {
            return;
        }
        let chat_name = flatten_namespace_tool_name(namespace, namespace);
        let schema = if strategy == NamespaceStrategy::NestedOneof {
            build_oneof_schema(&sub_tools)
        } else {
            build_anyof_schema(&sub_tools, &action_names)
        };
        let chat_tool = json!({
            "type": "function",
            "function": {
                "name": chat_name,
                "description": format!(
                    "Namespace tool: {namespace} (strategy: {})",
                    strategy.as_str()
                ),
                "parameters": schema,
            },
        });
        let spec = ToolSpec {
            kind: ToolKind::Namespace,
            name: namespace.to_owned(),
            namespace: Some(namespace.to_owned()),
            namespace_strategy: Some(strategy),
            actions: Some(action_names),
        };
        self.register_chat_tool(&chat_name, spec, chat_tool);
    }

    /// Register a single Responses tool by type, applying the hosted-tool
    /// policy.
    fn add_response_tool(&mut self, tool: &Value, policy: UnsupportedToolPolicy) {
        // A bare string is shorthand for a custom tool by name.
        if let Some(name) = tool.as_str() {
            if !name.trim().is_empty() {
                self.add_custom_tool(&json!({ "type": "custom", "name": name }));
            }
            return;
        }
        let Some(obj) = tool.as_object() else {
            return;
        };
        let tool_type = obj.get("type").and_then(Value::as_str).unwrap_or("");

        if is_hosted_tool_type(tool_type) {
            match policy {
                UnsupportedToolPolicy::Reject | UnsupportedToolPolicy::Error => {
                    // The Python bridge raises here; the Rust request path has no
                    // per-tool error channel at build time, so we drop with a
                    // loud warning rather than aborting the whole request.
                    tracing::warn!(
                        "Hosted Responses tool type {tool_type:?} is not supported \
                         (policy={policy:?}); dropping"
                    );
                }
                UnsupportedToolPolicy::Passthrough => {
                    if !self.chat_tools.contains(tool) {
                        self.chat_tools.push(tool.clone());
                    }
                }
                UnsupportedToolPolicy::Ignore => {
                    tracing::warn!(
                        "Ignoring unsupported hosted tool type: {tool_type} (policy=ignore)"
                    );
                }
            }
            return;
        }

        match tool_type {
            "function" => self.add_function_tool(tool, None),
            "custom" => self.add_custom_tool(tool),
            "tool_search" => self.add_tool_search_tool(),
            "namespace" => self.add_namespace_tool(tool),
            other => {
                tracing::warn!("Unrecognized tool type {other:?} — no handler, tool dropped");
            }
        }
    }
}

/// Extract valid function children and their action names, in order.
fn collect_function_children(children: &[Value]) -> (Vec<Value>, Vec<String>) {
    let mut sub_tools = Vec::new();
    let mut action_names = Vec::new();
    for child in children {
        if child.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        let func = child
            .get("function")
            .filter(|f| f.is_object())
            .cloned()
            .unwrap_or_else(|| child.clone());
        let Some(name) = tool_name_from_value(&func) else {
            continue;
        };
        sub_tools.push(func);
        action_names.push(name);
    }
    (sub_tools, action_names)
}

/// `nested_oneof`: one variant per action, discriminated by an `action` enum.
fn build_oneof_schema(sub_tools: &[Value]) -> Value {
    let mut variants = Vec::new();
    for func in sub_tools {
        let action_name = tool_name_from_value(func).unwrap_or_else(|| "unknown".to_owned());
        let params = func.get("parameters").and_then(Value::as_object);
        let mut props = params
            .and_then(|p| p.get("properties"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        props.insert(
            "action".to_owned(),
            json!({ "type": "string", "enum": [action_name] }),
        );
        let mut required: Vec<Value> = params
            .and_then(|p| p.get("required"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if !required.iter().any(|r| r.as_str() == Some("action")) {
            required.insert(0, json!("action"));
        }
        let mut variant = Map::new();
        variant.insert("type".to_owned(), json!("object"));
        variant.insert("properties".to_owned(), Value::Object(props));
        variant.insert("required".to_owned(), Value::Array(required));
        if let Some(ap) = params.and_then(|p| p.get("additionalProperties")) {
            variant.insert("additionalProperties".to_owned(), ap.clone());
        }
        variants.push(Value::Object(variant));
    }
    json!({ "type": "object", "oneOf": variants })
}

/// `nested_anyof`: a single `action` enum plus a merged `params` anyOf.
fn build_anyof_schema(sub_tools: &[Value], action_names: &[String]) -> Value {
    let param_variants: Vec<Value> = sub_tools
        .iter()
        .map(|func| {
            func.get("parameters")
                .filter(|p| p.is_object())
                .cloned()
                .unwrap_or_else(|| json!({}))
        })
        .collect();
    let params_schema = match param_variants.len() {
        0 => json!({}),
        1 => param_variants.into_iter().next().unwrap(),
        _ => json!({ "anyOf": param_variants }),
    };
    json!({
        "type": "object",
        "properties": {
            "action": { "type": "string", "enum": action_names },
            "params": params_schema,
        },
        "required": ["action"],
    })
}

// --------------------------------------------------------------------------- //
// Request-side tool-context builder
// --------------------------------------------------------------------------- //

/// Build a [`BridgeToolContext`] from a Responses request: register each declared
/// tool (flattening namespaces, building the Chat `tools` array), then scan the
/// request input for `custom_tool_call` names and `tool_search_output` tool
/// lists.
pub fn build_tool_context_from_request(
    payload: &crate::types::ResponsesRequest,
) -> BridgeToolContext {
    let policy = crate::config::global_unsupported_tool_policy();
    let mut context = BridgeToolContext::new();

    if let Some(tools) = payload.tools.as_ref().and_then(Value::as_array) {
        for tool in tools {
            context.add_response_tool(tool, policy);
        }
    }

    for item in iter_request_input_items(payload.input.as_ref()) {
        if let Some(obj) = item.as_object() {
            if obj.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
                if let Some(name) = obj
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                {
                    context.add_custom_tool(&json!({ "type": "custom", "name": name }));
                }
            }
        }
        collect_tool_search_output_tools(&mut context, &item, policy);
    }

    context
}

/// Normalize a request `input` into a flat list of items, wrapping bare strings
/// as `input_text` parts.
fn iter_request_input_items(input: Option<&Value>) -> Vec<Value> {
    match input {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![json!({ "type": "input_text", "text": s })],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => json!({ "type": "input_text", "text": s }),
                other => other.clone(),
            })
            .collect(),
        Some(other) => vec![other.clone()],
    }
}

/// Register tools carried inside `tool_search_output` items (recursing into
/// arrays but not into the output payload itself).
fn collect_tool_search_output_tools(
    context: &mut BridgeToolContext,
    value: &Value,
    policy: UnsupportedToolPolicy,
) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_tool_search_output_tools(context, item, policy);
            }
        }
        Value::Object(obj)
            if obj.get("type").and_then(Value::as_str) == Some("tool_search_output") =>
        {
            if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
                for tool in tools {
                    context.add_response_tool(tool, policy);
                }
            }
        }
        _ => {}
    }
}

// --------------------------------------------------------------------------- //
// Custom-tool input helpers
// --------------------------------------------------------------------------- //

/// Extract the *fully parsed* custom-tool input from a chat arguments JSON
/// string.
pub fn custom_tool_input_from_chat_arguments(arguments: &str) -> String {
    if arguments.trim().is_empty() {
        return String::new();
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => match map.get(CUSTOM_TOOL_INPUT_FIELD) {
            Some(Value::String(s)) => s.clone(),
            _ => arguments.to_owned(),
        },
        _ => arguments.to_owned(),
    }
}

/// Parse a chat arguments string into an object for a `tool_search_call`.
pub fn parse_tool_arguments_object(arguments: &str) -> Value {
    if arguments.trim().is_empty() {
        return Value::Object(Map::new());
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => json!({ "query": arguments }),
    }
}

/// Extract a *partial* custom-tool input prefix from an in-flight (possibly
/// incomplete) chat arguments string.
pub fn partial_custom_tool_input_from_chat_arguments(arguments: &str) -> Option<String> {
    if arguments.trim().is_empty() {
        return None;
    }

    let needle = format!("\"{CUSTOM_TOOL_INPUT_FIELD}\"");
    let key_pos = arguments.find(&needle)?;
    let colon_rel = arguments[key_pos..].find(':')?;
    let mut value_start = key_pos + colon_rel + 1;

    let bytes = arguments.as_bytes();
    while value_start < bytes.len() && matches!(bytes[value_start], b' ' | b'\t' | b'\r' | b'\n') {
        value_start += 1;
    }
    if value_start >= bytes.len() || bytes[value_start] != b'"' {
        return None;
    }

    Some(partial_json_string_prefix(&arguments[value_start + 1..]))
}

/// Decode a JSON string body (everything after the opening quote) up to the
/// first unescaped closing quote or a truncation point.
fn partial_json_string_prefix(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::new();
    let mut i = 0;
    let len = chars.len();

    while i < len {
        let ch = chars[i];
        if ch == '"' {
            break;
        }
        if ch != '\\' {
            result.push(ch);
            i += 1;
            continue;
        }
        // Escape sequence.
        if i + 1 >= len {
            break;
        }
        let esc = chars[i + 1];
        if esc == 'u' {
            if i + 6 > len {
                break;
            }
            let hex: String = chars[i + 2..i + 6].iter().collect();
            match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                Some(decoded) => result.push(decoded),
                None => break,
            }
            i += 6;
            continue;
        }
        let mapped = match esc {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => '\u{0008}',
            'f' => '\u{000C}',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            _ => break,
        };
        result.push(mapped);
        i += 2;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn req(tools: Value) -> crate::types::ResponsesRequest {
        serde_json::from_value(json!({ "tools": tools })).unwrap()
    }

    #[test]
    fn custom_and_tool_search_classification() {
        let ctx = build_tool_context_from_request(&req(json!([
            { "type": "custom", "name": "my_custom" },
            { "type": "tool_search" },
        ])));
        assert!(ctx.is_custom_tool(Some("my_custom")));
        assert!(!ctx.is_custom_tool(Some("other")));
        assert!(!ctx.is_custom_tool(None));
        assert!(ctx.is_tool_search(Some(TOOL_SEARCH_PROXY_NAME)));
        assert!(!ctx.is_tool_search(Some("my_custom")));
    }

    #[test]
    fn function_tool_becomes_chat_tool() {
        let ctx = build_tool_context_from_request(&req(json!([{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
        }])));
        assert_eq!(ctx.chat_tools().len(), 1);
        let f = &ctx.chat_tools()[0]["function"];
        assert_eq!(f["name"], json!("get_weather"));
        assert_eq!(f["description"], json!("Get weather"));
    }

    #[test]
    fn flat_namespace_flattens_child_names() {
        let ctx = build_tool_context_from_request(&req(json!([{
            "type": "namespace",
            "name": "github",
            "strategy": "flat",
            "tools": [
                { "type": "function", "name": "create_issue", "parameters": {} },
            ],
        }])));
        assert_eq!(ctx.chat_tools().len(), 1);
        assert_eq!(
            ctx.chat_tools()[0]["function"]["name"],
            json!("github__create_issue")
        );
        // Forward and reverse maps round-trip.
        assert_eq!(
            ctx.chat_name_for_function("create_issue", Some("github")),
            "github__create_issue"
        );
        assert_eq!(
            ctx.restore_namespace_and_name("github__create_issue"),
            (Some("github".to_owned()), "create_issue".to_owned())
        );
    }

    #[test]
    fn long_namespace_name_is_hashed() {
        let long = "a".repeat(70);
        let flat = flatten_namespace_tool_name("ns", &long);
        assert_eq!(flat.len(), CHAT_TOOL_NAME_MAX_LEN);
        assert!(flat.contains("__"));
        // Deterministic.
        assert_eq!(flat, flatten_namespace_tool_name("ns", &long));
    }

    #[test]
    fn restore_only_splits_registered_namespaces() {
        let ctx = BridgeToolContext::new();
        // Unregistered "foo__bar" must not be mis-parsed as namespaced.
        assert_eq!(
            ctx.restore_namespace_and_name("foo__bar"),
            (None, "foo__bar".to_owned())
        );
    }

    #[test]
    fn nested_oneof_builds_action_selector() {
        let ctx = build_tool_context_from_request(&req(json!([{
            "type": "namespace",
            "name": "db",
            "strategy": "nested_oneof",
            "tools": [
                { "type": "function", "name": "read", "parameters": { "type": "object", "properties": { "id": { "type": "string" } }, "required": ["id"] } },
                { "type": "function", "name": "write", "parameters": { "type": "object", "properties": { "row": { "type": "object" } } } },
            ],
        }])));
        assert_eq!(ctx.chat_tools().len(), 1);
        let schema = &ctx.chat_tools()[0]["function"]["parameters"];
        let variants = schema["oneOf"].as_array().unwrap();
        assert_eq!(variants.len(), 2);
        // First variant carries an action enum locked to "read" plus its params.
        assert_eq!(
            variants[0]["properties"]["action"]["enum"][0],
            json!("read")
        );
        assert!(variants[0]["required"]
            .as_array()
            .unwrap()
            .contains(&json!("action")));
        let spec = ctx
            .lookup_chat_name(Some(
                ctx.chat_tools()[0]["function"]["name"].as_str().unwrap(),
            ))
            .unwrap();
        assert!(spec.is_nested_namespace());
        assert_eq!(spec.actions.as_ref().unwrap(), &["read", "write"]);
    }

    #[test]
    fn nested_anyof_resolution_flattens_params() {
        let spec = ToolSpec {
            kind: ToolKind::Namespace,
            name: "db".to_owned(),
            namespace: Some("db".to_owned()),
            namespace_strategy: Some(NamespaceStrategy::NestedAnyof),
            actions: Some(vec!["read".to_owned()]),
        };
        let res =
            resolve_nested_namespace_arguments(&spec, r#"{"action":"read","params":{"id":"7"}}"#);
        assert_eq!(res.action_name.as_deref(), Some("read"));
        assert_eq!(res.normalized_arguments, r#"{"id":"7"}"#);
    }

    #[test]
    fn nested_resolution_unknown_action_is_none() {
        let spec = ToolSpec {
            kind: ToolKind::Namespace,
            name: "db".to_owned(),
            namespace: Some("db".to_owned()),
            namespace_strategy: Some(NamespaceStrategy::NestedOneof),
            actions: Some(vec!["read".to_owned()]),
        };
        let res = resolve_nested_namespace_arguments(&spec, r#"{"action":"delete"}"#);
        assert_eq!(res.action_name, None);
        assert_eq!(res.normalized_arguments, "{}");
    }

    #[test]
    fn hosted_tool_ignored_by_default() {
        let ctx = build_tool_context_from_request(&req(json!([
            { "type": "web_search" },
            { "type": "function", "name": "f", "parameters": {} },
        ])));
        // Hosted tool dropped under the default ignore policy; function kept.
        assert_eq!(ctx.chat_tools().len(), 1);
        assert_eq!(ctx.chat_tools()[0]["function"]["name"], json!("f"));
    }

    #[test]
    fn merge_unions_tools_and_flags() {
        let mut base = build_tool_context_from_request(&req(json!([
            { "type": "function", "name": "a", "parameters": {} },
        ])));
        let other = build_tool_context_from_request(&req(json!([
            { "type": "function", "name": "b", "parameters": {} },
            { "type": "tool_search" },
        ])));
        base.merge(&other);
        let names: Vec<&str> = base
            .chat_tools()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
        assert!(names.contains(&TOOL_SEARCH_PROXY_NAME));
        assert!(base.is_tool_search(Some(TOOL_SEARCH_PROXY_NAME)));
    }

    #[test]
    fn dedup_by_chat_name() {
        let ctx = build_tool_context_from_request(&req(json!([
            { "type": "function", "name": "dup", "parameters": {} },
            { "type": "function", "name": "dup", "parameters": {} },
        ])));
        assert_eq!(ctx.chat_tools().len(), 1);
    }

    #[test]
    fn custom_input_full_extraction() {
        assert_eq!(custom_tool_input_from_chat_arguments(""), "");
        assert_eq!(
            custom_tool_input_from_chat_arguments(r#"{"input": "hello"}"#),
            "hello"
        );
        assert_eq!(
            custom_tool_input_from_chat_arguments(r#"{"input": 42}"#),
            r#"{"input": 42}"#
        );
        assert_eq!(
            custom_tool_input_from_chat_arguments("not json"),
            "not json"
        );
    }

    #[test]
    fn tool_search_arguments_object() {
        assert_eq!(parse_tool_arguments_object(""), json!({}));
        assert_eq!(
            parse_tool_arguments_object(r#"{"q": 1}"#),
            json!({ "q": 1 })
        );
        assert_eq!(
            parse_tool_arguments_object("raw text"),
            json!({ "query": "raw text" })
        );
    }

    #[test]
    fn partial_input_prefix_streaming() {
        assert_eq!(partial_custom_tool_input_from_chat_arguments("{"), None);
        assert_eq!(
            partial_custom_tool_input_from_chat_arguments(r#"{"in"#),
            None
        );
        assert_eq!(
            partial_custom_tool_input_from_chat_arguments(r#"{"input": "hel"#),
            Some("hel".to_owned())
        );
        assert_eq!(
            partial_custom_tool_input_from_chat_arguments(r#"{"input": "hello"}"#),
            Some("hello".to_owned())
        );
    }

    #[test]
    fn partial_input_decodes_escapes() {
        assert_eq!(
            partial_custom_tool_input_from_chat_arguments(r#"{"input": "line\n\ttab"#),
            Some("line\n\ttab".to_owned())
        );
        assert_eq!(
            partial_custom_tool_input_from_chat_arguments(r#"{"input": "trail\"#),
            Some("trail".to_owned())
        );
        assert_eq!(
            partial_custom_tool_input_from_chat_arguments(r#"{"input": "A\u0042C"#),
            Some("ABC".to_owned())
        );
    }
}
