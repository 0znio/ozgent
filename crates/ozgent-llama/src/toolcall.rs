//! Extracting tool calls from model output.
//!
//! There is no single format. Each model family invented its own, and a local
//! runtime has to read all of them, because the user picks the model:
//!
//! ```text
//! <tool_call>{"name": "x", "arguments": {}}</tool_call>   Qwen, Hermes
//! [TOOL_CALLS][{"name": "x", "arguments": {}}]            Mistral
//! <function=x>{"a": 1}</function>                         some Llama tunes
//! <|python_tag|>{"name": "x", ...}                        Llama 3.x
//! ```json { "name": "x", "arguments": {} } ```            models with no tool training
//! ```
//!
//! Brace matching is done by a string-aware scanner rather than a regex, since
//! arguments nest and may contain braces inside string literals.

use ozgent_core::ToolCall;
use serde_json::Value;

/// Markers that introduce a tool call, and the marker that ends it if any.
const OPENERS: &[(&str, Option<&str>)] = &[
    ("<tool_call>", Some("</tool_call>")),
    ("<|tool_call|>", Some("<|/tool_call|>")),
    ("[TOOL_CALLS]", None),
    ("<|python_tag|>", None),
    ("<function=", Some("</function>")),
];

/// What the model produced, separated into what to show and what to run.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Parsed {
    /// Text to render, with tool-call syntax removed.
    pub text: String,
    pub calls: Vec<ToolCall>,
}

impl Parsed {
    pub fn has_calls(&self) -> bool {
        !self.calls.is_empty()
    }
}

/// Split model output into visible text and tool calls.
///
/// Nothing is invented: if no recognisable call is present the text is
/// returned unchanged, so ordinary prose that merely mentions JSON is safe.
pub fn extract(output: &str) -> Parsed {
    let mut calls = Vec::new();
    let mut text = String::new();
    let mut rest = output;

    while let Some((idx, opener, closer)) = next_opener(rest) {
        text.push_str(&rest[..idx]);
        let after = &rest[idx + opener.len()..];

        // `<function=name>` carries the name in the marker itself.
        let explicit_name = if opener == "<function=" {
            match after.find('>') {
                Some(end) => {
                    let name = after[..end].trim().to_string();
                    rest = &after[end + 1..];
                    Some(name)
                }
                None => {
                    text.push_str(opener);
                    rest = after;
                    continue;
                }
            }
        } else {
            rest = after;
            None
        };

        // Bound the search at the closer so a malformed call cannot swallow
        // the remainder of the response.
        let (body, consumed) = match closer.and_then(|c| rest.find(c).map(|i| (i, c))) {
            Some((end, c)) => (&rest[..end], end + c.len()),
            None => (rest, rest.len()),
        };

        // Qwen 3.5 nests the two markers: `<tool_call>` wrapping
        // `<function=name>`. The outer opener wins the scan above, so unwrap
        // here or the name never reaches the parameter parser.
        let (name, body) = match explicit_name.clone() {
            Some(n) => (Some(n), body),
            None => match unwrap_function(body) {
                Some((n, inner)) => (Some(n), inner),
                None => (None, body),
            },
        };

        let before = calls.len();
        // A `<function=` body is `<parameter=x>` blocks rather than JSON.
        // Tried first when the marker named the function, because a value can
        // itself contain a brace and the JSON scan would then harvest a
        // fragment of prose as if it were the arguments.
        if let Some(name) = name.as_deref() {
            if let Some(call) = from_parameters(body, name) {
                calls.push(call);
            }
        }
        // Ling 3.0 names the function on the opener's own line and then
        // lists `<arg_key>`/`<arg_value>` pairs — a third format again, and
        // the reason its calls used to cost a grammar retry every round.
        if calls.len() == before {
            if let Some(call) = from_arg_pairs(body, name.as_deref()) {
                calls.push(call);
            }
        }
        if calls.len() == before {
            harvest(body, name.as_deref(), &mut calls);
        }
        if calls.len() == before {
            // Nothing parseable: keep the text rather than silently dropping it.
            text.push_str(opener);
            text.push_str(body);
        }
        rest = &rest[consumed..];
    }
    text.push_str(rest);

    // Fall back to fenced JSON, which is how an untrained model complies.
    if calls.is_empty() {
        if let Some((cleaned, fenced)) = from_fenced_json(&text) {
            text = cleaned;
            calls = fenced;
        }
    }

    for (i, call) in calls.iter_mut().enumerate() {
        if call.id.is_empty() {
            call.id = format!("call_{i}");
        }
    }
    Parsed { text: text.trim().to_string(), calls }
}

fn next_opener(text: &str) -> Option<(usize, &'static str, Option<&'static str>)> {
    OPENERS
        .iter()
        .filter_map(|(open, close)| text.find(open).map(|i| (i, *open, *close)))
        .min_by_key(|(i, _, _)| *i)
}

/// Pull every JSON object or array of objects out of a tool-call body.
/// Parse `<arg_key>k</arg_key><arg_value>v</arg_value>` pairs.
///
/// Ling 3.0's format, which puts the function name directly after the opener:
///
/// ```text
/// <tool_call>web_search
/// <arg_key>query</arg_key>
/// <arg_value>llama.cpp</arg_value>
/// </tool_call>
/// ```
///
/// `name` is used when the marker already carried one; otherwise the first
/// line of the body is the name. Returns `None` when there is no `<arg_key>`
/// at all, so a JSON body falls through to `harvest`.
fn from_arg_pairs(body: &str, name: Option<&str>) -> Option<ToolCall> {
    const KEY_OPEN: &str = "<arg_key>";
    const KEY_CLOSE: &str = "</arg_key>";
    const VAL_OPEN: &str = "<arg_value>";
    const VAL_CLOSE: &str = "</arg_value>";

    let first_key = body.find(KEY_OPEN)?;
    let name = match name {
        Some(n) => n.to_string(),
        // Everything before the first key, which is the name on its own line.
        None => {
            let head = body[..first_key].trim();
            let head = head.lines().next().unwrap_or("").trim();
            if head.is_empty() {
                return None;
            }
            head.to_string()
        }
    };

    let mut arguments = serde_json::Map::new();
    let mut rest = body;
    while let Some(k) = rest.find(KEY_OPEN) {
        let after_key = &rest[k + KEY_OPEN.len()..];
        let Some(k_end) = after_key.find(KEY_CLOSE) else { break };
        let key = after_key[..k_end].trim().to_string();

        let after = &after_key[k_end + KEY_CLOSE.len()..];
        let Some(v) = after.find(VAL_OPEN) else { break };
        let after_val = &after[v + VAL_OPEN.len()..];
        // An unclosed final value still carries its text; a call cut off by
        // the token limit is better read than thrown away.
        let (raw, consumed) = match after_val.find(VAL_CLOSE) {
            Some(end) => (&after_val[..end], end + VAL_CLOSE.len()),
            None => (after_val, after_val.len()),
        };

        if !key.is_empty() {
            arguments.insert(key, coerce(raw.trim()));
        }
        rest = &after_val[consumed..];
    }

    (!arguments.is_empty()).then(|| ToolCall {
        id: String::new(),
        name,
        arguments: Value::Object(arguments),
    })
}

/// Split `<function=name>…</function>` into its name and its body.
///
/// Returns `None` when there is no function marker, leaving the body to be
/// read as JSON.
fn unwrap_function(body: &str) -> Option<(String, &str)> {
    const OPEN: &str = "<function=";
    let start = body.find(OPEN)?;
    let after = &body[start + OPEN.len()..];
    let name_end = after.find('>')?;
    let name = after[..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let inner = &after[name_end + 1..];
    let inner = match inner.find("</function>") {
        Some(end) => &inner[..end],
        None => inner,
    };
    Some((name.to_string(), inner))
}

/// Parse `<parameter=name>value</parameter>` blocks into arguments.
///
/// The format Qwen 3.5 is trained on and describes in its own template:
///
/// ```text
/// <tool_call>
/// <function=web_search>
/// <parameter=query>
/// stocks to watch
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// Returns `None` when there is no parameter block at all, so a body that is
/// really JSON falls through to `harvest` rather than becoming a call with no
/// arguments.
fn from_parameters(body: &str, name: &str) -> Option<ToolCall> {
    const OPEN: &str = "<parameter=";
    const CLOSE: &str = "</parameter>";

    let mut arguments = serde_json::Map::new();
    let mut rest = body;
    let mut found = false;

    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(name_end) = after.find('>') else { break };
        let key = after[..name_end].trim().to_string();
        let value_part = &after[name_end + 1..];

        // An unclosed final parameter still carries its value; a truncated
        // call is better read than discarded.
        let (raw, consumed) = match value_part.find(CLOSE) {
            Some(end) => (&value_part[..end], end + CLOSE.len()),
            None => (value_part, value_part.len()),
        };

        if !key.is_empty() {
            found = true;
            arguments.insert(key, coerce(raw.trim()));
        }
        rest = &value_part[consumed..];
    }

    found.then(|| ToolCall {
        id: String::new(),
        name: name.to_string(),
        arguments: Value::Object(arguments),
    })
}

/// Read a parameter value as the type it is written as.
///
/// The format carries no types, so `5` and `true` arrive as text where the
/// schema wants a number and a boolean. Only whole scalars are converted:
/// anything else — including `5 stocks`, which is prose that begins with a
/// digit — stays the string it was written as.
fn coerce(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        Ok(v @ (Value::Bool(_) | Value::Number(_) | Value::Null)) => v,
        _ => Value::String(raw.to_string()),
    }
}

fn harvest(body: &str, explicit_name: Option<&str>, out: &mut Vec<ToolCall>) {
    let mut cursor = 0;
    while cursor < body.len() {
        let Some(start) = body[cursor..].find(['{', '[']).map(|i| cursor + i) else {
            break;
        };
        let Some(end) = balanced_end(body, start) else {
            break;
        };
        let slice = &body[start..end];

        if let Ok(value) = serde_json::from_str::<Value>(slice) {
            match value {
                Value::Array(items) => {
                    for item in items {
                        if let Some(c) = to_call(&item, explicit_name) {
                            out.push(c);
                        }
                    }
                }
                other => {
                    if let Some(c) = to_call(&other, explicit_name) {
                        out.push(c);
                    }
                }
            }
        }
        cursor = end;
    }
}

/// Interpret one JSON value as a tool call.
///
/// Accepts the several key spellings models use, and treats an object with no
/// recognisable name as arguments when the name came from the marker.
fn to_call(value: &Value, explicit_name: Option<&str>) -> Option<ToolCall> {
    let obj = value.as_object()?;

    // OpenAI nests the real call under `function`.
    if let Some(inner) = obj.get("function").and_then(Value::as_object) {
        let name = inner.get("name")?.as_str()?.to_string();
        return Some(ToolCall {
            id: obj.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
            name,
            arguments: normalize_arguments(inner.get("arguments")),
        });
    }

    let name = obj
        .get("name")
        .or_else(|| obj.get("tool"))
        .or_else(|| obj.get("tool_name"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| explicit_name.map(str::to_string))?;

    if name.is_empty() {
        return None;
    }

    let arguments = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .or_else(|| obj.get("args"))
        .or_else(|| obj.get("input"));

    // With a name from the marker, the whole object is the argument set.
    let arguments = match (arguments, explicit_name) {
        (Some(a), _) => normalize_arguments(Some(a)),
        (None, Some(_)) => value.clone(),
        (None, None) => Value::Object(Default::default()),
    };

    Some(ToolCall {
        id: obj.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
        name,
        arguments,
    })
}

/// Arguments are sometimes a JSON *string* containing JSON, which is how the
/// OpenAI wire format encodes them.
fn normalize_arguments(value: Option<&Value>) -> Value {
    match value {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(v) => v.clone(),
        None => Value::Object(Default::default()),
    }
}

/// Index just past the balanced bracket that opens at `start`.
///
/// String-aware, so braces inside quoted values do not throw off the count.
fn balanced_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let open = bytes[start];
    let close = match open {
        b'{' => b'}',
        b'[' => b']',
        _ => return None,
    };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            x if x == open => depth += 1,
            x if x == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Recognise a fenced JSON block that is really a tool call.
///
/// Deliberately strict: the object must carry both a name and arguments, so a
/// model showing the user an example JSON payload is not executed.
fn from_fenced_json(text: &str) -> Option<(String, Vec<ToolCall>)> {
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut rest = text;

    while let Some(idx) = rest.find("```") {
        let after_fence = &rest[idx + 3..];
        let Some(nl) = after_fence.find('\n') else { break };
        let lang = after_fence[..nl].trim();
        let body_start = idx + 3 + nl + 1;
        let Some(end_rel) = rest[body_start..].find("```") else { break };
        let body = &rest[body_start..body_start + end_rel];

        let is_json = lang.is_empty() || lang.eq_ignore_ascii_case("json");
        let parsed = is_json
            .then(|| serde_json::from_str::<Value>(body.trim()).ok())
            .flatten();

        let is_call = parsed.as_ref().is_some_and(|v| {
            v.as_object().is_some_and(|o| {
                o.contains_key("name")
                    && (o.contains_key("arguments") || o.contains_key("parameters"))
            })
        });

        if is_call {
            if let Some(c) = parsed.as_ref().and_then(|v| to_call(v, None)) {
                calls.push(c);
            }
            cleaned.push_str(&rest[..idx]);
        } else {
            cleaned.push_str(&rest[..body_start + end_rel + 3]);
        }
        rest = &rest[body_start + end_rel + 3..];
    }

    if calls.is_empty() {
        return None;
    }
    cleaned.push_str(rest);
    Some((cleaned, calls))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_the_qwen_hermes_format() {
        let out = extract(
            r#"Let me look.<tool_call>{"name": "web_search", "arguments": {"query": "rust"}}</tool_call>"#,
        );
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "web_search");
        assert_eq!(out.calls[0].arguments, json!({"query": "rust"}));
        assert_eq!(out.text, "Let me look.", "call syntax must not reach the user");
    }

    #[test]
    fn parses_the_mistral_format_with_several_calls() {
        let out = extract(
            r#"[TOOL_CALLS][{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]"#,
        );
        assert_eq!(out.calls.len(), 2);
        assert_eq!(out.calls[0].name, "a");
        assert_eq!(out.calls[1].name, "b");
    }

    #[test]
    fn parses_the_function_marker_format() {
        let out = extract(r#"<function=get_weather>{"city": "Oslo"}</function>"#);
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "get_weather");
        assert_eq!(out.calls[0].arguments, json!({"city": "Oslo"}));
    }

    #[test]
    fn parses_the_python_tag_format() {
        let out = extract(r#"<|python_tag|>{"name": "calc", "arguments": {"expr": "2+2"}}"#);
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "calc");
    }

    #[test]
    fn parses_the_openai_nested_shape_with_stringified_arguments() {
        let out = extract(
            r#"<tool_call>{"id":"abc","function":{"name":"search","arguments":"{\"q\":\"cats\"}"}}</tool_call>"#,
        );
        assert_eq!(out.calls[0].name, "search");
        assert_eq!(out.calls[0].arguments, json!({"q": "cats"}), "stringified JSON must be decoded");
        assert_eq!(out.calls[0].id, "abc", "a provided id must be kept");
    }

    #[test]
    fn accepts_alternative_key_spellings() {
        for body in [
            r#"{"name": "t", "parameters": {"a": 1}}"#,
            r#"{"tool": "t", "args": {"a": 1}}"#,
            r#"{"tool_name": "t", "input": {"a": 1}}"#,
        ] {
            let out = extract(&format!("<tool_call>{body}</tool_call>"));
            assert_eq!(out.calls.len(), 1, "failed on {body}");
            assert_eq!(out.calls[0].name, "t");
            assert_eq!(out.calls[0].arguments, json!({"a": 1}));
        }
    }

    #[test]
    fn handles_braces_inside_string_arguments() {
        // A regex-based scanner gets this wrong.
        let out = extract(
            r#"<tool_call>{"name": "run", "arguments": {"code": "if (x) { y(); }"}}</tool_call>"#,
        );
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].arguments["code"], "if (x) { y(); }");
    }

    #[test]
    fn handles_escaped_quotes_inside_arguments() {
        let out = extract(
            r#"<tool_call>{"name": "echo", "arguments": {"text": "she said \"hi\" }"}}</tool_call>"#,
        );
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].arguments["text"], r#"she said "hi" }"#);
    }

    #[test]
    fn recognises_a_fenced_json_call_from_an_untrained_model() {
        let out = extract("I'll search.\n\n```json\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"x\"}}\n```");
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].name, "web_search");
        assert!(!out.text.contains("web_search"), "the fence must be removed: {:?}", out.text);
    }

    #[test]
    fn an_illustrative_json_block_is_not_executed() {
        // The model is showing the user a payload, not calling anything.
        let text = "The config looks like:\n\n```json\n{\"model\": \"gemma4\", \"gpu_layers\": 20}\n```";
        let out = extract(text);
        assert!(out.calls.is_empty(), "must not execute example JSON: {:?}", out.calls);
        assert!(out.text.contains("gpu_layers"), "the example must still be shown");
    }

    #[test]
    fn ordinary_prose_is_returned_untouched() {
        let text = "A tool call uses braces { like this } but this is just prose.";
        let out = extract(text);
        assert!(out.calls.is_empty());
        assert_eq!(out.text, text);
    }

    #[test]
    fn malformed_json_is_shown_rather_than_silently_dropped() {
        let out = extract("<tool_call>{not valid json at all</tool_call>");
        assert!(out.calls.is_empty());
        assert!(out.text.contains("not valid json"), "content must not vanish: {:?}", out.text);
    }

    #[test]
    fn an_unterminated_call_does_not_swallow_later_text() {
        let out = extract(r#"<tool_call>{"name": "a", "arguments": {}}"#);
        assert_eq!(out.calls.len(), 1, "a call with no closer should still parse");
        assert_eq!(out.calls[0].name, "a");
    }

    #[test]
    fn ids_are_assigned_when_the_model_gives_none() {
        let out = extract(
            r#"<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>"#,
        );
        assert_eq!(out.calls.len(), 2);
        assert_ne!(out.calls[0].id, out.calls[1].id, "ids must be distinct to correlate results");
        assert!(!out.calls[0].id.is_empty());
    }

    #[test]
    fn text_around_a_call_is_preserved_in_order() {
        let out = extract(
            r#"Before. <tool_call>{"name":"a","arguments":{}}</tool_call> After."#,
        );
        assert_eq!(out.text, "Before.  After.".trim());
        assert!(out.text.starts_with("Before."));
        assert!(out.text.ends_with("After."));
    }

    #[test]
    fn empty_output_is_handled() {
        let out = extract("");
        assert!(out.text.is_empty());
        assert!(!out.has_calls());
    }

    #[test]
    fn a_call_with_no_name_is_rejected() {
        let out = extract(r#"<tool_call>{"arguments": {"a": 1}}</tool_call>"#);
        assert!(out.calls.is_empty(), "a nameless call is not runnable");
    }
}

/// Hides tool-call syntax while it is being generated.
///
/// A tool call is a request to the runtime, not prose for the user, so the
/// raw `<tool_call>{…}` must never reach the terminal. As with reasoning tags,
/// the opener arrives split across tokens, so the gate holds back only the
/// longest suffix that could still become one.
#[derive(Debug, Default)]
pub struct StreamGate {
    buf: String,
    suppressing: bool,
}

impl StreamGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a tool call has started and output is being withheld.
    pub fn suppressing(&self) -> bool {
        self.suppressing
    }

    /// Feed generated text, returning only what the user should see.
    pub fn push(&mut self, text: &str) -> String {
        if self.suppressing {
            return String::new();
        }
        self.buf.push_str(text);

        // Once an opener appears, everything from it onward belongs to the
        // call, including whatever follows in later tokens.
        if let Some(idx) = OPENERS
            .iter()
            .filter_map(|(open, _)| self.buf.find(open))
            .min()
        {
            let visible = self.buf[..idx].to_string();
            self.buf.clear();
            self.suppressing = true;
            return visible;
        }

        let keep = OPENERS
            .iter()
            .map(|(open, _)| partial_suffix(&self.buf, open))
            .max()
            .unwrap_or(0);
        let split = self.buf.len() - keep;
        self.buf.drain(..split).collect()
    }

    /// Flush anything held back that never became a tool call.
    pub fn finish(&mut self) -> String {
        if self.suppressing {
            self.buf.clear();
            return String::new();
        }
        std::mem::take(&mut self.buf)
    }
}

/// Longest suffix of `haystack` that is a proper prefix of `tag`.
fn partial_suffix(haystack: &str, tag: &str) -> usize {
    let max = tag.len().min(haystack.len());
    for len in (1..=max).rev() {
        let start = haystack.len() - len;
        if !haystack.is_char_boundary(start) {
            continue;
        }
        if len < tag.len() && tag.as_bytes().starts_with(&haystack.as_bytes()[start..]) {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod native_format_tests {
    use super::*;

    const CALL: &str = concat!(
        "<tool_call>\n<function=web_search>\n",
        "<parameter=query>\nIndian stock market\n</parameter>\n",
        "<parameter=count>\n5\n</parameter>\n",
        "</function>\n</tool_call>"
    );

    #[test]
    fn qwens_own_format_is_parsed() {
        // The format Qwen 3.5's template describes to the model. Unparsed, it
        // read as a malformed call, cost a grammar retry every round, and left
        // the model off-contract for the rest of the turn.
        let p = extract(CALL);
        assert_eq!(p.calls.len(), 1, "{:?}", p);
        assert_eq!(p.calls[0].name, "web_search");
        assert_eq!(p.calls[0].arguments["query"], "Indian stock market");
    }

    #[test]
    fn numbers_and_booleans_come_back_as_themselves() {
        let p = extract(CALL);
        assert_eq!(p.calls[0].arguments["count"], 5, "a count is a number");

        let flags = "<function=t>\n<parameter=on>\ntrue</parameter>\n</function>";
        assert_eq!(extract(flags).calls[0].arguments["on"], true);
    }

    #[test]
    fn prose_beginning_with_a_digit_stays_a_string() {
        // `5 best stocks` must not become the number 5.
        let call = "<function=t>\n<parameter=q>\n5 best stocks</parameter>\n</function>";
        assert_eq!(extract(call).calls[0].arguments["q"], "5 best stocks");
    }

    #[test]
    fn a_value_containing_braces_survives_intact() {
        // The reason parameters are tried before the JSON scan: this body
        // would otherwise be harvested as a JSON fragment.
        let call = concat!(
            "<function=write_file>\n<parameter=content>\n",
            "fn main() { println!(\"hi\"); }\n</parameter>\n</function>"
        );
        let p = extract(call);
        assert_eq!(p.calls.len(), 1, "{:?}", p);
        assert!(
            p.calls[0].arguments["content"].as_str().unwrap().contains("println!"),
            "{:?}",
            p.calls[0].arguments
        );
    }

    #[test]
    fn a_multiline_value_keeps_its_lines() {
        let call = "<function=t>\n<parameter=body>\nline one\nline two\n</parameter>\n</function>";
        assert_eq!(extract(call).calls[0].arguments["body"], "line one\nline two");
    }

    #[test]
    fn a_truncated_final_parameter_is_still_read() {
        // The generation ran out mid-call; the arguments are all there.
        let call = "<function=t>\n<parameter=q>\nsomething";
        let p = extract(call);
        assert_eq!(p.calls.len(), 1, "{:?}", p);
        assert_eq!(p.calls[0].arguments["q"], "something");
    }

    const LING_CALL: &str = concat!(
        "<tool_call>web_search\n",
        "<arg_key>query</arg_key>\n<arg_value>llama.cpp</arg_value>\n",
        "<arg_key>count</arg_key>\n<arg_value>5</arg_value>\n",
        "</tool_call>"
    );

    #[test]
    fn lings_own_format_is_parsed() {
        // Ling names the function on the opener's line. Unparsed, every call
        // it made cost a grammar retry and pushed it off its own format.
        let p = extract(LING_CALL);
        assert_eq!(p.calls.len(), 1, "{p:?}");
        assert_eq!(p.calls[0].name, "web_search");
        assert_eq!(p.calls[0].arguments["query"], "llama.cpp");
        assert_eq!(p.calls[0].arguments["count"], 5);
    }

    #[test]
    fn a_ling_call_with_no_arguments_is_not_a_call() {
        // `<tool_call>` around prose must not become a call named after the
        // first line of that prose.
        assert!(extract("<tool_call>I was thinking about this</tool_call>").calls.is_empty());
    }

    #[test]
    fn a_truncated_ling_value_is_still_read() {
        let call = "<tool_call>web_search\n<arg_key>query</arg_key>\n<arg_value>llama";
        let p = extract(call);
        assert_eq!(p.calls.len(), 1, "{p:?}");
        assert_eq!(p.calls[0].arguments["query"], "llama");
    }

    #[test]
    fn all_three_formats_coexist() {
        // One parser, three vocabularies; adding one must not cost another.
        for (label, text) in [
            ("qwen", CALL),
            ("ling", LING_CALL),
            ("json", r#"<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>"#),
        ] {
            let p = extract(text);
            assert_eq!(p.calls.len(), 1, "{label}: {p:?}");
            assert_eq!(p.calls[0].name, "web_search", "{label}");
        }
    }

    #[test]
    fn the_json_format_still_works() {
        // Adding one format must not cost the other; most models use this one.
        let p = extract(r#"<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>"#);
        assert_eq!(p.calls.len(), 1, "{:?}", p);
        assert_eq!(p.calls[0].name, "web_search");
        assert_eq!(p.calls[0].arguments["query"], "x");
    }

    #[test]
    fn a_function_marker_with_a_json_body_still_parses() {
        let p = extract("<function=web_search>{\"query\": \"x\"}</function>");
        assert_eq!(p.calls.len(), 1, "{:?}", p);
        assert_eq!(p.calls[0].arguments["query"], "x");
    }

    #[test]
    fn prose_around_a_call_is_kept_and_the_call_is_not() {
        let p = extract(&format!("Let me look that up.\n{CALL}"));
        assert!(p.text.contains("Let me look that up"), "{:?}", p.text);
        assert!(!p.text.contains("<parameter="), "{:?}", p.text);
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    fn drip(input: &str) -> (String, bool) {
        let mut g = StreamGate::new();
        let mut out = String::new();
        for ch in input.chars() {
            out.push_str(&g.push(&ch.to_string()));
        }
        out.push_str(&g.finish());
        (out, g.suppressing())
    }

    #[test]
    fn ordinary_text_passes_through() {
        let (out, suppressed) = drip("Just a normal answer.");
        assert_eq!(out, "Just a normal answer.");
        assert!(!suppressed);
    }

    #[test]
    fn a_tool_call_is_hidden_from_the_user() {
        let (out, suppressed) = drip(r#"Let me check.<tool_call>{"name":"x","arguments":{}}</tool_call>"#);
        assert_eq!(out, "Let me check.", "the call must not reach the terminal");
        assert!(suppressed);
    }

    #[test]
    fn an_opener_split_across_tokens_is_still_caught() {
        // The failure this gate exists to prevent.
        let mut g = StreamGate::new();
        let mut out = String::new();
        for tok in ["Sure.", "<tool", "_call>", "{\"name\""] {
            out.push_str(&g.push(tok));
        }
        out.push_str(&g.finish());
        assert_eq!(out, "Sure.");
        assert!(g.suppressing());
    }

    #[test]
    fn text_merely_resembling_an_opener_is_released() {
        let (out, suppressed) = drip("compare <tool and <too here");
        assert_eq!(out, "compare <tool and <too here");
        assert!(!suppressed);
    }

    #[test]
    fn every_recognised_opener_is_gated() {
        for (open, _) in OPENERS {
            let (out, suppressed) = drip(&format!("text{open}rest"));
            assert_eq!(out, "text", "failed to gate {open}");
            assert!(suppressed);
        }
    }

    #[test]
    fn nothing_leaks_after_suppression_starts() {
        let mut g = StreamGate::new();
        let _ = g.push("<tool_call>");
        assert_eq!(g.push("{\"name\": \"x\"}"), "");
        assert_eq!(g.push("</tool_call> trailing"), "");
        assert_eq!(g.finish(), "");
    }

    #[test]
    fn multibyte_text_is_not_split() {
        let (out, _) = drip("日本語のテキスト 🎉");
        assert_eq!(out, "日本語のテキスト 🎉");
    }
}
