//! The wire format, and what to make of what comes back.
//!
//! Everything here is a pure function of a JSON value, and that is deliberate:
//! an MCP server is someone else's program, so this code spends most of its
//! effort on replies that are wrong, partial or hostile, and those are cases
//! that can only be covered properly if they can be written down as a literal.

use ozgent_core::ToolSpec;
use ozgent_core::permission::Effect;
use serde_json::{Value, json};

/// The revision of the specification ozgent speaks.
///
/// Sent on `initialize`; a server that speaks something else answers with the
/// version it chose, and ozgent goes along with it rather than refusing —
/// a client that will only talk to one revision stops working every time the
/// specification moves.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// A JSON-RPC request.
pub fn request(id: u64, method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
}

/// A JSON-RPC notification — no id, and no reply is expected or waited for.
pub fn notification(method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "method": method, "params": params })
}

/// What a server said went wrong.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct Failure {
    pub code: i64,
    pub message: String,
}

/// Pull the result out of a JSON-RPC response.
pub fn result_of(response: &Value) -> Result<Value, Failure> {
    if let Some(error) = response.get("error") {
        return Err(Failure {
            code: error.get("code").and_then(|c| c.as_i64()).unwrap_or(0),
            message: error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("the server gave no reason")
                .to_string(),
        });
    }
    // A response with neither is malformed. Treating a missing `result` as an
    // empty one would turn a broken server into a tool that silently returns
    // nothing.
    response.get("result").cloned().ok_or_else(|| Failure {
        code: 0,
        message: "the reply had neither a result nor an error".into(),
    })
}

/// The parameters ozgent introduces itself with.
pub fn initialize_params() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        // Only what ozgent actually honours. Claiming `sampling` or
        // `elicitation` would invite the server to send requests back that
        // nothing here answers, and it would wait.
        "capabilities": {},
        "clientInfo": { "name": "ozgent", "version": env!("CARGO_PKG_VERSION") },
    })
}

/// What a server said about itself.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    pub protocol_version: String,
    /// Whether it offers tools at all.
    pub has_tools: bool,
}

pub fn server_info(result: &Value) -> ServerInfo {
    let info = result.get("serverInfo");
    ServerInfo {
        name: info
            .and_then(|i| i.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("an unnamed server")
            .to_string(),
        version: info
            .and_then(|i| i.get("version"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        protocol_version: result
            .get("protocolVersion")
            .and_then(|v| v.as_str())
            .unwrap_or(PROTOCOL_VERSION)
            .to_string(),
        has_tools: result.get("capabilities").and_then(|c| c.get("tools")).is_some(),
    }
}

/// One tool, as the server describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct Listed {
    /// The name on the server, which is what `tools/call` must be given.
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Option<Value>,
    /// `readOnlyHint`, if the server offered one.
    pub read_only: Option<bool>,
}

/// Read a `tools/list` result. Returns the tools and the pagination cursor.
///
/// Anything without a usable name is dropped rather than being allowed to
/// become a tool the model can name but nothing can call.
pub fn parse_tools(result: &Value) -> (Vec<Listed>, Option<String>) {
    let cursor = result
        .get("nextCursor")
        .and_then(|c| c.as_str())
        .filter(|c| !c.is_empty())
        .map(str::to_string);

    let Some(list) = result.get("tools").and_then(|t| t.as_array()) else {
        return (Vec::new(), cursor);
    };

    let tools = list
        .iter()
        .filter_map(|tool| {
            let name = tool.get("name")?.as_str()?.trim();
            if name.is_empty() {
                return None;
            }
            Some(Listed {
                name: name.to_string(),
                description: tool
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string(),
                // A schema is what tells the model how to call the tool. A
                // missing one becomes an empty object rather than null, which
                // some templates render as the word "null".
                input_schema: tool
                    .get("inputSchema")
                    .cloned()
                    .filter(|s| s.is_object())
                    .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
                output_schema: tool.get("outputSchema").cloned().filter(|s| s.is_object()),
                read_only: tool
                    .get("annotations")
                    .and_then(|a| a.get("readOnlyHint"))
                    .and_then(|r| r.as_bool()),
            })
        })
        .collect();

    (tools, cursor)
}

/// What ozgent will treat a tool as doing.
///
/// The specification is explicit that annotations are *hints* and that a
/// client must not trust them unless the server is trusted. ozgent's own rule
/// already says a tool that does not declare its effect is asked about,
/// because silence must not be read as harmless — and a claim by the thing
/// being asked about is not better evidence than silence.
///
/// So with `trust_hints` off, every tool is `Unknown` and every call asks.
/// With it on, the operator has said they trust this server, and
/// `readOnlyHint` is believed.
pub fn effect_of(tool: &Listed, trust_hints: bool) -> Effect {
    if !trust_hints {
        return Effect::Unknown;
    }
    match tool.read_only {
        Some(true) => Effect::Read,
        // Not read-only is a change of some kind. MCP has no notion of running
        // a program, so `Write` is as specific as this can honestly be.
        Some(false) => Effect::Write,
        None => Effect::Unknown,
    }
}

/// Turn a listed tool into the spec the model is shown.
pub fn to_spec(server: &str, tool: &Listed, trust_hints: bool) -> ToolSpec {
    ToolSpec {
        name: ozgent_core::mcp::tool_name(server, &tool.name),
        description: tool.description.clone(),
        input_schema: tool.input_schema.clone(),
        output_schema: tool.output_schema.clone(),
        effect: effect_of(tool, trust_hints),
    }
}

/// What a `tools/call` produced.
///
/// `Err` is the tool reporting failure — `isError` — which is a normal thing
/// for a tool to do and reaches the model as a result, not as a crash.
pub fn parse_call(result: &Value) -> Result<Value, String> {
    let failed = result.get("isError").and_then(|e| e.as_bool()).unwrap_or(false);
    let text = content_text(result);

    if failed {
        return Err(if text.trim().is_empty() {
            "the tool reported an error and said nothing more".to_string()
        } else {
            text
        });
    }

    // A server that returned structured output meant that to be the result;
    // the text block beside it is the same thing rendered for a human.
    if let Some(structured) = result.get("structuredContent").filter(|s| !s.is_null()) {
        return Ok(structured.clone());
    }

    if text.trim().is_empty() {
        // Every content block was an image, a resource, or nothing at all.
        return Ok(describe_blocks(result));
    }

    // A tool whose text *is* JSON is far more useful to the model as data than
    // as a string containing punctuation.
    match serde_json::from_str::<Value>(text.trim()) {
        Ok(value) if value.is_object() || value.is_array() => Ok(value),
        _ => Ok(json!({ "text": text })),
    }
}

/// Every text block, joined.
fn content_text(result: &Value) -> String {
    let Some(blocks) = result.get("content").and_then(|c| c.as_array()) else {
        return String::new();
    };
    blocks
        .iter()
        .filter_map(|block| match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => block.get("text").and_then(|t| t.as_str()).map(str::to_string),
            // An embedded resource carries its own text when it has one.
            Some("resource") => block
                .get("resource")
                .and_then(|r| r.get("text"))
                .and_then(|t| t.as_str())
                .map(str::to_string),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Say what came back when none of it was text.
///
/// The model cannot be handed an image through a tool result here, so it is
/// told one exists rather than being given an empty object and left to
/// conclude the tool did nothing.
fn describe_blocks(result: &Value) -> Value {
    let Some(blocks) = result.get("content").and_then(|c| c.as_array()) else {
        return json!({});
    };
    let kinds: Vec<String> = blocks
        .iter()
        .filter_map(|b| b.get("type").and_then(|t| t.as_str()).map(str::to_string))
        .collect();
    if kinds.is_empty() {
        return json!({});
    }
    json!({ "text": format!("The tool returned {} that cannot be shown here.", kinds.join(", ")) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listed(read_only: Option<bool>) -> Listed {
        Listed {
            name: "search".into(),
            description: "Search".into(),
            input_schema: json!({ "type": "object" }),
            output_schema: None,
            read_only,
        }
    }

    #[test]
    fn a_request_carries_the_id_it_will_be_answered_with() {
        let r = request(7, "tools/list", json!({}));
        assert_eq!(r["jsonrpc"], "2.0");
        assert_eq!(r["id"], 7);
        assert_eq!(r["method"], "tools/list");
    }

    #[test]
    fn a_notification_has_no_id_at_all() {
        // A server that saw an id would try to answer, and nothing is waiting.
        let n = notification("notifications/initialized", json!({}));
        assert!(n.get("id").is_none(), "{n}");
    }

    #[test]
    fn an_error_reply_is_reported_with_what_the_server_said() {
        let response = json!({ "jsonrpc": "2.0", "id": 1,
            "error": { "code": -32602, "message": "unknown tool" } });
        assert_eq!(
            result_of(&response),
            Err(Failure { code: -32602, message: "unknown tool".into() })
        );
    }

    #[test]
    fn a_reply_with_neither_result_nor_error_is_a_failure() {
        // Treating a missing result as an empty one turns a broken server into
        // a tool that silently returns nothing, which is much harder to see.
        let response = json!({ "jsonrpc": "2.0", "id": 1 });
        assert!(result_of(&response).is_err());
    }

    #[test]
    fn ozgent_claims_only_what_it_can_actually_do() {
        // Claiming `sampling` invites the server to send a request back that
        // nothing answers, and it waits.
        let p = initialize_params();
        assert_eq!(p["capabilities"], json!({}));
        assert_eq!(p["clientInfo"]["name"], "ozgent");
    }

    #[test]
    fn a_servers_chosen_protocol_version_is_taken_as_given() {
        let info = server_info(&json!({
            "protocolVersion": "2024-11-05",
            "serverInfo": { "name": "files", "version": "1.2" },
            "capabilities": { "tools": {} }
        }));
        assert_eq!(info.protocol_version, "2024-11-05");
        assert_eq!(info.name, "files");
        assert!(info.has_tools);
    }

    #[test]
    fn a_server_offering_no_tools_says_so() {
        let info = server_info(&json!({ "capabilities": { "resources": {} } }));
        assert!(!info.has_tools);
    }

    #[test]
    fn tools_are_read_with_their_schemas() {
        let (tools, cursor) = parse_tools(&json!({
            "tools": [{
                "name": "search",
                "description": "Search the issues",
                "inputSchema": { "type": "object", "properties": { "q": { "type": "string" } } },
                "annotations": { "readOnlyHint": true }
            }]
        }));
        assert_eq!(cursor, None);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
        assert_eq!(tools[0].read_only, Some(true));
        assert_eq!(tools[0].input_schema["properties"]["q"]["type"], "string");
    }

    #[test]
    fn a_tool_with_no_usable_name_is_dropped() {
        // Otherwise the model is shown something it can name and nothing can
        // call.
        let (tools, _) = parse_tools(&json!({
            "tools": [{ "description": "no name" }, { "name": "  " }, { "name": "ok" }]
        }));
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "ok");
    }

    #[test]
    fn a_missing_schema_becomes_an_empty_object_not_null() {
        // A null schema renders as the word "null" in some chat templates.
        let (tools, _) = parse_tools(&json!({ "tools": [{ "name": "x" }] }));
        assert_eq!(tools[0].input_schema["type"], "object");
        assert!(tools[0].input_schema.is_object());
    }

    #[test]
    fn pagination_is_reported_so_the_rest_can_be_fetched() {
        let (_, cursor) = parse_tools(&json!({ "tools": [], "nextCursor": "page2" }));
        assert_eq!(cursor.as_deref(), Some("page2"));
        let (_, none) = parse_tools(&json!({ "tools": [], "nextCursor": "" }));
        assert_eq!(none, None);
    }

    #[test]
    fn a_reply_that_is_not_a_tool_list_yields_nothing_rather_than_panicking() {
        assert_eq!(parse_tools(&json!({})).0.len(), 0);
        assert_eq!(parse_tools(&json!(null)).0.len(), 0);
        assert_eq!(parse_tools(&json!({ "tools": "lots" })).0.len(), 0);
    }

    // ------------------------------------------------------------- effects

    #[test]
    fn a_servers_claim_about_itself_is_not_believed_by_default() {
        // The property this module exists to protect. A read-only hint decides
        // whether a call runs without anyone being asked, and it is written by
        // the same party that wants the call to happen.
        assert_eq!(effect_of(&listed(Some(true)), false), Effect::Unknown);
        assert_eq!(effect_of(&listed(Some(false)), false), Effect::Unknown);
        assert_eq!(effect_of(&listed(None), false), Effect::Unknown);
    }

    #[test]
    fn a_trusted_servers_hints_are_used() {
        assert_eq!(effect_of(&listed(Some(true)), true), Effect::Read);
        assert_eq!(effect_of(&listed(Some(false)), true), Effect::Write);
    }

    #[test]
    fn silence_is_still_not_harmless_even_from_a_trusted_server() {
        // Trusting a server means believing what it says, not filling in what
        // it did not say.
        assert_eq!(effect_of(&listed(None), true), Effect::Unknown);
    }

    #[test]
    fn a_spec_is_named_after_its_server() {
        let spec = to_spec("github", &listed(Some(true)), true);
        assert_eq!(spec.name, "github_search");
        assert_eq!(spec.effect, Effect::Read);
    }

    // --------------------------------------------------------------- calls

    #[test]
    fn text_content_comes_back_as_text() {
        let out = parse_call(&json!({ "content": [{ "type": "text", "text": "42 issues" }] }));
        assert_eq!(out, Ok(json!({ "text": "42 issues" })));
    }

    #[test]
    fn structured_output_is_preferred_to_its_rendering() {
        // A server sending both means the text to be the human form of the
        // same thing; handing the model the prose loses the data.
        let out = parse_call(&json!({
            "content": [{ "type": "text", "text": "42 issues" }],
            "structuredContent": { "count": 42 }
        }));
        assert_eq!(out, Ok(json!({ "count": 42 })));
    }

    #[test]
    fn text_that_is_json_is_handed_over_as_data() {
        let out = parse_call(&json!({
            "content": [{ "type": "text", "text": "{\"count\": 2}" }]
        }));
        assert_eq!(out, Ok(json!({ "count": 2 })));
    }

    #[test]
    fn text_that_merely_looks_numeric_stays_text() {
        // `42` parses as JSON but is not a result the model should be handed
        // as a bare number where an object was expected.
        assert_eq!(
            parse_call(&json!({ "content": [{ "type": "text", "text": "42" }] })),
            Ok(json!({ "text": "42" }))
        );
    }

    #[test]
    fn several_text_blocks_are_joined() {
        let out = parse_call(&json!({
            "content": [
                { "type": "text", "text": "first" },
                { "type": "text", "text": "second" }
            ]
        }));
        assert_eq!(out, Ok(json!({ "text": "first\nsecond" })));
    }

    #[test]
    fn an_embedded_resource_contributes_its_text() {
        let out = parse_call(&json!({
            "content": [{ "type": "resource",
                          "resource": { "uri": "file:///a", "text": "contents" } }]
        }));
        assert_eq!(out, Ok(json!({ "text": "contents" })));
    }

    #[test]
    fn a_tool_reporting_failure_is_an_error_with_its_own_words() {
        let out = parse_call(&json!({
            "isError": true,
            "content": [{ "type": "text", "text": "the repository does not exist" }]
        }));
        assert_eq!(out, Err("the repository does not exist".into()));
    }

    #[test]
    fn a_failure_with_nothing_said_still_says_something() {
        let out = parse_call(&json!({ "isError": true, "content": [] }));
        assert!(out.unwrap_err().contains("said nothing more"));
    }

    #[test]
    fn a_result_that_is_only_an_image_says_so_rather_than_looking_empty() {
        // An empty object reads to the model as "the tool did nothing", and it
        // tries again.
        let out = parse_call(&json!({
            "content": [{ "type": "image", "data": "…", "mimeType": "image/png" }]
        }))
        .unwrap();
        assert!(out["text"].as_str().unwrap().contains("image"), "{out}");
    }

    #[test]
    fn a_result_with_no_content_at_all_is_empty_and_not_an_error() {
        // A tool that did something and has nothing to report is normal.
        assert_eq!(parse_call(&json!({})), Ok(json!({})));
    }
}
