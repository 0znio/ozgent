//! Generating GBNF grammars from tool schemas.
//!
//! Constrained decoding masks the logits so only tokens that keep the output
//! syntactically valid can be sampled. Applied to tool calling this is
//! decisive: a model with no tool training cannot emit malformed JSON, cannot
//! invent a tool name, and cannot misspell a parameter — the grammar makes
//! those tokens unreachable rather than merely unlikely.
//!
//! The converter handles the subset of JSON Schema that
//! `ozgent_tools.schema` emits, which is by construction the subset every
//! ozgent tool uses.

use ozgent_core::ToolSpec;
use serde_json::Value;
use std::fmt::Write;

/// Primitive rules, emitted only when referenced.
///
/// A grammar carrying rules nothing uses is larger to compile and harder to
/// debug, and llama.cpp rejects some combinations outright. Each entry lists
/// the rules it depends on so the closure can be computed.
const PRIMITIVES: &[(&str, &str, &[&str])] = &[
    ("ws", r"ws ::= [ \t\n]*", &[]),
    ("char", r#"char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])"#, &[]),
    ("string", r#"string ::= "\"" char* "\"" ws"#, &["char", "ws"]),
    ("integer", r#"integer ::= "-"? ("0" | [1-9] [0-9]*) ws"#, &["ws"]),
    ("number", r#"number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ws"#, &["ws"]),
    ("boolean", r#"boolean ::= ("true" | "false") ws"#, &["ws"]),
    ("null", r#"null ::= "null" ws"#, &["ws"]),
    (
        "value",
        "value ::= string | number | boolean | null",
        &["string", "number", "boolean", "null"],
    ),
];

/// Emit the transitive closure of the primitive rules `needed`.
fn emit_primitives(needed: &std::collections::BTreeSet<String>) -> String {
    let mut wanted: std::collections::BTreeSet<String> = needed.clone();
    // `ws` separates every token, so it is always required.
    wanted.insert("ws".into());

    // Close over dependencies; the table is small so a fixed point is cheap.
    loop {
        let before = wanted.len();
        for (name, _, deps) in PRIMITIVES {
            if wanted.contains(*name) {
                for d in *deps {
                    wanted.insert((*d).to_string());
                }
            }
        }
        if wanted.len() == before {
            break;
        }
    }

    // Definition order follows the table, so dependencies appear first.
    PRIMITIVES
        .iter()
        .filter(|(name, _, _)| wanted.contains(*name))
        .map(|(_, rule, _)| *rule)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a grammar that admits exactly one well-formed call to one of `tools`.
///
/// Returns `None` when there are no tools, since an empty alternation is not a
/// valid grammar.
pub fn tool_call_grammar(tools: &[ToolSpec]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }

    let mut body = String::new();
    let mut needed = std::collections::BTreeSet::new();
    let alternatives: Vec<String> = (0..tools.len()).map(|i| format!("call{i}")).collect();

    for (i, tool) in tools.iter().enumerate() {
        let args = schema_rule(&tool.input_schema, &format!("args{i}"), &mut body, &mut needed);
        let _ = writeln!(
            body,
            "call{i} ::= \"{{\" ws \"\\\"name\\\"\" ws \":\" ws \"\\\"{}\\\"\" ws \",\" ws \"\\\"arguments\\\"\" ws \":\" ws {args} \"}}\" ws",
            escape(&tool.name)
        );
    }

    // Wrap in the marker the parser reads. Without it the grammar yields bare
    // JSON, which `extract` treats as prose — the call is perfectly formed and
    // then silently ignored.
    let mut out = format!(
        "root ::= \"<tool_call>\" ({}) \"</tool_call>\"\n",
        alternatives.join(" | ")
    );
    out.push_str(&emit_primitives(&needed));
    out.push('\n');
    out.push_str(&body);
    Some(out)
}

/// Emit a rule for `schema` named `name`, returning the rule reference to use.
fn schema_rule(
    schema: &Value,
    name: &str,
    out: &mut String,
    needed: &mut std::collections::BTreeSet<String>,
) -> String {
    let Some(obj) = schema.as_object() else {
        return "value".into();
    };

    // An enum is the tightest constraint available: only the listed literals.
    if let Some(Value::Array(values)) = obj.get("enum") {
        let alts: Vec<String> = values.iter().map(literal).collect();
        if !alts.is_empty() {
            needed.insert("ws".into());
            let _ = writeln!(out, "{name} ::= ({}) ws", alts.join(" | "));
            return name.to_string();
        }
    }

    let ty = obj.get("type").and_then(|t| match t {
        Value::String(s) => Some(s.clone()),
        // A nullable field is `["string", "null"]`; take the concrete half.
        Value::Array(a) => a.iter().find_map(|v| {
            v.as_str().filter(|s| *s != "null").map(str::to_string)
        }),
        _ => None,
    });

    let mut primitive = |n: &str| {
        needed.insert(n.to_string());
        n.to_string()
    };
    match ty.as_deref() {
        Some("string") => primitive("string"),
        Some("integer") => primitive("integer"),
        Some("number") => primitive("number"),
        Some("boolean") => primitive("boolean"),
        Some("array") => {
            let item = match obj.get("items") {
                Some(i) => schema_rule(i, &format!("{name}-item"), out, needed),
                None => {
                    needed.insert("value".into());
                    "value".into()
                }
            };
            needed.insert("ws".into());
            let _ = writeln!(out, "{name} ::= \"[\" ws ({item} (\",\" ws {item})*)? \"]\" ws");
            name.to_string()
        }
        Some("object") => object_rule(obj, name, out, needed),
        _ => {
            needed.insert("value".into());
            "value".into()
        }
    }
}

/// Build a rule for an object with known properties.
///
/// Required properties are emitted in a fixed order and optional ones are
/// wrapped in `( ... )?`. Fixing the order is what keeps the grammar small: a
/// grammar admitting every permutation grows factorially.
fn object_rule(
    obj: &serde_json::Map<String, Value>,
    name: &str,
    out: &mut String,
    needed: &mut std::collections::BTreeSet<String>,
) -> String {
    let Some(Value::Object(props)) = obj.get("properties") else {
        needed.insert("value".into());
        return "value".into();
    };
    if props.is_empty() {
        needed.insert("ws".into());
        let _ = writeln!(out, "{name} ::= \"{{\" ws \"}}\" ws");
        return name.to_string();
    }

    let required: Vec<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    // Required first so the model commits to the essential arguments before
    // any optional ones.
    let mut ordered: Vec<(&String, &Value)> = Vec::new();
    for key in &required {
        if let Some((k, v)) = props.get_key_value(*key) {
            ordered.push((k, v));
        }
    }
    for (k, v) in props {
        if !required.contains(&k.as_str()) {
            ordered.push((k, v));
        }
    }

    let mut parts: Vec<String> = Vec::new();
    for (index, (key, sub)) in ordered.iter().enumerate() {
        let rule = schema_rule(sub, &format!("{name}-{}", sanitize(key)), out, needed);
        let is_required = required.contains(&key.as_str());
        // A separator is needed only when something precedes this property.
        let comma = if index == 0 { String::new() } else { "\",\" ws ".to_string() };
        let piece = format!("{comma}\"\\\"{}\\\"\" ws \":\" ws {rule}", escape(key));
        parts.push(if is_required { piece } else { format!("({piece})?") });
    }

    needed.insert("ws".into());
    let _ = writeln!(out, "{name} ::= \"{{\" ws {} \"}}\" ws", parts.join(" "));
    name.to_string()
}

fn literal(v: &Value) -> String {
    match v {
        Value::String(s) => format!("\"\\\"{}\\\"\"", escape(s)),
        other => format!("\"{other}\""),
    }
}

/// Escape a value that will sit inside a JSON string, inside a GBNF literal.
///
/// Two layers are needed and missing either corrupts the grammar. The JSON
/// layer turns `"` into `\"` so the emitted text is valid JSON; the GBNF layer
/// then escapes that backslash and quote again so the literal does not
/// terminate early. Escaping once produces a grammar that matches *invalid*
/// JSON, which the tool-call parser will then reject.
fn escape(s: &str) -> String {
    let json = s.replace('\\', "\\\\").replace('"', "\\\"");
    json.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Make a string usable as a GBNF rule name.
///
/// llama.cpp's grammar parser accepts letters, digits and hyphens in an
/// identifier — but **not underscores**. An underscore silently ends the name,
/// so `args0_category` parses as rule `args0` followed by junk, and the whole
/// grammar is rejected with a confusing message.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(name: &str, schema: Value) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: String::new(),
            input_schema: schema,
            output_schema: None,
        }
    }

    #[test]
    fn the_grammar_output_is_parseable_by_the_tool_call_extractor() {
        // The grammar and the parser must agree on the wrapper, or a perfectly
        // formed call is generated and then thrown away.
        let g = tool_call_grammar(&[spec(
            "get_temperature",
            json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
        )])
        .unwrap();
        assert!(g.contains("<tool_call>"), "grammar must emit the marker:\n{g}");
        assert!(g.contains("</tool_call>"), "{g}");

        // What that grammar can produce must round-trip through extract().
        let produced = r#"<tool_call>{"name": "get_temperature", "arguments": {"city": "Lima"}}</tool_call>"#;
        let parsed = crate::toolcall::extract(produced);
        assert_eq!(parsed.calls.len(), 1, "extractor must accept grammar output");
        assert_eq!(parsed.calls[0].name, "get_temperature");
        assert_eq!(parsed.calls[0].arguments["city"], "Lima");
    }

    #[test]
    fn no_tools_yields_no_grammar() {
        // An empty alternation would be a syntactically invalid grammar.
        assert!(tool_call_grammar(&[]).is_none());
    }

    #[test]
    fn the_tool_name_is_a_fixed_literal() {
        let g = tool_call_grammar(&[spec(
            "web_search",
            json!({"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
        )])
        .unwrap();

        assert!(g.contains(r#""\"web_search\"""#), "the name must be pinned:\n{g}");
        assert!(g.starts_with(r#"root ::= "<tool_call>""#), "{g}");
    }

    #[test]
    fn several_tools_become_alternatives() {
        let g = tool_call_grammar(&[
            spec("a", json!({"type": "object", "properties": {}})),
            spec("b", json!({"type": "object", "properties": {}})),
        ])
        .unwrap();
        assert!(g.contains("(call0 | call1)"), "{g}");
        assert!(g.contains(r#""\"a\"""#) && g.contains(r#""\"b\"""#));
    }

    #[test]
    fn enums_become_literal_alternations() {
        let g = tool_call_grammar(&[spec(
            "search",
            json!({
                "type": "object",
                "properties": {"category": {"enum": ["web", "news"]}},
                "required": ["category"]
            }),
        )])
        .unwrap();
        assert!(g.contains(r#""\"web\"""#), "{g}");
        assert!(g.contains(r#""\"news\"""#), "{g}");
    }

    #[test]
    fn primitive_types_map_to_prelude_rules() {
        let g = tool_call_grammar(&[spec(
            "t",
            json!({
                "type": "object",
                "properties": {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "n": {"type": "number"},
                    "b": {"type": "boolean"}
                },
                "required": ["s", "i", "n", "b"]
            }),
        )])
        .unwrap();
        for rule in ["string", "integer", "number", "boolean"] {
            assert!(g.contains(&format!("{rule} ::=")), "{rule} rule missing:\n{g}");
        }
        // Only what is used: this schema has no array, so `value` is absent.
        assert!(!g.contains("value ::="), "unused rules must not be emitted:\n{g}");
    }

    #[test]
    fn optional_properties_are_wrapped_but_required_ones_are_not() {
        let g = tool_call_grammar(&[spec(
            "t",
            json!({
                "type": "object",
                "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["q"]
            }),
        )])
        .unwrap();

        let args = g.lines().find(|l| l.starts_with("args0 ::=")).expect("args rule");
        assert!(args.contains(r#"(","#), "optional property should be in a group: {args}");
        // The required key appears outside any optional group, i.e. first.
        let q_at = args.find(r#"\"q\""#).unwrap();
        let n_at = args.find(r#"\"n\""#).unwrap();
        assert!(q_at < n_at, "required properties come first: {args}");
    }

    #[test]
    fn no_generated_rule_name_contains_an_underscore() {
        // The failure mode is silent and the resulting error is misleading, so
        // this is checked across every construct that names a sub-rule.
        let g = tool_call_grammar(&[spec(
            "my_tool",
            json!({
                "type": "object",
                "properties": {
                    "some_field": {"enum": ["a", "b"]},
                    "other_list": {"type": "array", "items": {"type": "string"}},
                    "nested_obj": {"type": "object", "properties": {"inner_key": {"type": "string"}}}
                },
                "required": ["some_field"]
            }),
        )])
        .unwrap();

        for line in g.lines() {
            if let Some((head, _)) = line.split_once("::=") {
                assert!(!head.contains('_'), "bad rule name: {:?}", head.trim());
            }
        }
    }

    #[test]
    fn arrays_produce_a_repetition_rule() {
        let g = tool_call_grammar(&[spec(
            "t",
            json!({
                "type": "object",
                "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
                "required": ["tags"]
            }),
        )])
        .unwrap();
        assert!(g.contains("args0-tags ::= \"[\""), "{g}");
        assert!(g.contains("(\",\" ws string)*"), "repetition expected:\n{g}");
    }

    #[test]
    fn an_empty_property_set_still_produces_a_valid_object() {
        let g = tool_call_grammar(&[spec("t", json!({"type": "object", "properties": {}}))]).unwrap();
        assert!(g.contains(r#"args0 ::= "{" ws "}""#), "{g}");
    }

    #[test]
    fn nullable_types_use_the_concrete_half() {
        let g = tool_call_grammar(&[spec(
            "t",
            json!({
                "type": "object",
                "properties": {"maybe": {"type": ["string", "null"]}},
                "required": ["maybe"]
            }),
        )])
        .unwrap();
        assert!(g.contains("string"), "{g}");
    }

    #[test]
    fn quotes_in_names_are_escaped() {
        // A tool name is attacker-controlled only insofar as a user wrote it,
        // but an unescaped quote silently corrupts the whole grammar.
        let g = tool_call_grammar(&[spec("we\"ird", json!({"type": "object", "properties": {}}))]).unwrap();
        // Two layers: JSON needs \" and GBNF needs that backslash escaped too.
        assert!(g.contains(r#"we\\\"ird"#), "quote must be double-escaped:\n{g}");
        // A single-escaped name would close the literal early, leaving a
        // stray `ird\"` token outside any quotes.
        assert!(!g.contains(r#""\"we\"ird\"""#), "literal terminates early:\n{g}");
    }

    #[test]
    fn property_names_are_sanitised_into_rule_names() {
        let g = tool_call_grammar(&[spec(
            "t",
            json!({
                "type": "object",
                "properties": {"max_results": {"enum": [1, 2]}},
                "required": ["max_results"]
            }),
        )])
        .unwrap();
        // The JSON key keeps its underscore; the rule name must not, because
        // llama.cpp's identifier parser stops at one.
        assert!(g.contains(r#"\"max_results\""#), "the key must be preserved:\n{g}");
        for line in g.lines() {
            if let Some((head, _)) = line.split_once("::=") {
                assert!(
                    !head.contains('_'),
                    "rule name {:?} contains an underscore, which llama.cpp rejects",
                    head.trim()
                );
            }
        }
    }

    /// Remove quoted literals and character classes, leaving only the
    /// grammar's structural identifiers.
    fn structural_only(body: &str) -> String {
        let mut out = String::new();
        let mut chars = body.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '"' => {
                    // Skip to the closing quote, honouring backslash escapes.
                    while let Some(n) = chars.next() {
                        if n == '\\' {
                            chars.next();
                        } else if n == '"' {
                            break;
                        }
                    }
                }
                '[' => {
                    while let Some(n) = chars.next() {
                        if n == '\\' {
                            chars.next();
                        } else if n == ']' {
                            break;
                        }
                    }
                }
                other => out.push(other),
            }
        }
        out
    }

    #[test]
    fn every_referenced_rule_is_defined() {
        // A dangling reference makes llama.cpp reject the grammar at load
        // time, which would surface as tool calling simply not working.
        let g = tool_call_grammar(&[spec(
            "web_search",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"enum": ["web", "news"]},
                    "count": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["query"]
            }),
        )])
        .unwrap();

        let defined: std::collections::HashSet<String> = g
            .lines()
            .filter_map(|l| l.split_once("::="))
            .map(|(head, _)| head.trim().to_string())
            .collect();

        assert!(defined.contains("root"), "a grammar needs a root rule");

        for line in g.lines() {
            let Some((_, body)) = line.split_once("::=") else { continue };
            for token in structural_only(body)
                .split(|c: char| !c.is_ascii_alphanumeric() && c != '-')
                .filter(|t| !t.is_empty())
                .filter(|t| !t.chars().next().unwrap().is_ascii_digit())
            {
                assert!(
                    defined.contains(token),
                    "undefined rule {token:?} referenced in: {line}"
                );
            }
        }
    }

    #[test]
    fn structural_only_strips_literals_and_classes() {
        assert_eq!(structural_only(r#" "{" ws "}" "#).trim(), "ws");
        assert_eq!(structural_only(r"[ \t\n]* ws").trim(), "* ws");
        assert_eq!(structural_only(r#" "\"name\"" ws "#).trim(), "ws");
    }
}

