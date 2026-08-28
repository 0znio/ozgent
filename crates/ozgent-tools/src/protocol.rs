//! Wire types for the JSON-RPC 2.0 dialect spoken with the Python worker.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROTOCOL_VERSION: u32 = 1;

// Codes mirrored from `ozgent_tools.worker`.
pub const PARSE_ERROR: i32 = -32700;
pub const INVALID_REQUEST: i32 = -32600;
pub const METHOD_NOT_FOUND: i32 = -32601;
pub const INVALID_PARAMS: i32 = -32602;
pub const INTERNAL_ERROR: i32 = -32603;
/// The tool ran and failed in a way the model should see.
pub const TOOL_ERROR: i32 = -32000;
pub const TOOL_CANCELLED: i32 = -32001;

#[derive(Debug, Serialize)]
pub struct Request<'a> {
    pub jsonrpc: &'static str,
    pub id: u64,
    pub method: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl<'a> Request<'a> {
    pub fn new(id: u64, method: &'a str, params: Option<Value>) -> Self {
        Self { jsonrpc: "2.0", id, method, params }
    }
}

#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: Option<u64>,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<RpcError>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    /// Whether the model should be encouraged to try the call again.
    pub fn retryable(&self) -> bool {
        self.data
            .as_ref()
            .and_then(|d| d.get("retryable"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// A failure the model caused and can correct, as opposed to one that
    /// indicates a broken tool. The distinction decides whether the error goes
    /// back into the conversation or is surfaced to the user.
    pub fn is_model_fault(&self) -> bool {
        matches!(self.code, INVALID_PARAMS | TOOL_ERROR)
    }
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Payload of the `initialize` reply.
#[derive(Debug, Deserialize)]
pub struct InitializeResult {
    pub protocol_version: u32,
    pub worker_version: String,
    pub python: String,
    #[serde(default)]
    pub tools: Vec<ToolManifest>,
    /// Non-fatal load failures, one per tool file that could not be imported.
    #[serde(default)]
    pub errors: Vec<String>,
}

/// A tool as the worker describes it.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolManifest {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub input_schema: Value,
    #[serde(default)]
    pub output_schema: Option<Value>,
    /// The Python file the tool was defined in. Empty when the worker is an
    /// older build that did not report one, which is not worth an error.
    #[serde(default)]
    pub source: String,
    /// What the tool does to the world. A worker that does not report one —
    /// or a tool whose author has not said — lands on `unknown`, which is
    /// asked about rather than assumed harmless.
    #[serde(default)]
    pub effect: ozgent_core::permission::Effect,
}

impl From<ToolManifest> for ozgent_core::ToolSpec {
    fn from(m: ToolManifest) -> Self {
        Self {
            name: m.name,
            description: m.description,
            input_schema: m.input_schema,
            output_schema: m.output_schema,
            effect: m.effect,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_errors() {
        let bad_args = RpcError { code: INVALID_PARAMS, message: "x".into(), data: None };
        assert!(bad_args.is_model_fault(), "the model can fix its own arguments");

        let crashed = RpcError { code: INTERNAL_ERROR, message: "x".into(), data: None };
        assert!(!crashed.is_model_fault(), "a tool bug is not the model's fault");
    }

    #[test]
    fn reads_the_retryable_flag() {
        let e = RpcError {
            code: TOOL_ERROR,
            message: "rate limited".into(),
            data: Some(serde_json::json!({ "retryable": true })),
        };
        assert!(e.retryable());

        let plain = RpcError { code: TOOL_ERROR, message: "no".into(), data: None };
        assert!(!plain.retryable());
    }
}
