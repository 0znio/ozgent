//! `{{ node.field }}` — how one step reads another's output.
//!
//! Deliberately not a language. There is no arithmetic, no function calls and
//! no way to reach anything but the outputs of steps that have already run.
//! A workflow editor invites people to paste in things they were sent, and an
//! expression evaluator is the shortest path from "paste a template" to
//! "arbitrary code runs on the machine hosting it".
//!
//! What it does is a path lookup: a name, then dotted keys and `[n]` indices.
//! Everything else in a template is literal text.

use serde_json::Value;
use std::collections::BTreeMap;

/// What the steps that have already run produced, by step id.
pub type Context = BTreeMap<String, Value>;

/// Fill in every `{{ … }}` in `template`.
///
/// An expression that resolves to a string is inserted as-is; anything else is
/// inserted as compact JSON, so `{{ search.results }}` in a prompt gives the
/// model the data rather than the word `[object Object]`.
///
/// An expression that resolves to nothing is left standing, exactly as written.
/// Blanking it would turn "this step has not run yet" into "this step returned
/// an empty string", which is the same prompt with a different meaning and no
/// way to tell them apart.
pub fn interpolate(template: &str, context: &Context) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;

    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            // An unclosed `{{` is text. Someone is writing about the syntax.
            out.push_str(&rest[start..]);
            return out;
        };
        let expression = &after[..end];
        match resolve(expression, context) {
            Some(Value::String(s)) => out.push_str(&s),
            Some(value) => out.push_str(&value.to_string()),
            None => {
                out.push_str("{{");
                out.push_str(expression);
                out.push_str("}}");
            }
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    out
}

/// Resolve a whole template that is exactly one expression, keeping its type.
///
/// `{{ search.count }}` as an entire field should stay the number 4, not become
/// the string "4" — a tool with a typed schema would reject the string.
pub fn resolve_value(template: &str, context: &Context) -> Value {
    let trimmed = template.trim();
    if let Some(inner) = trimmed.strip_prefix("{{").and_then(|t| t.strip_suffix("}}")) {
        // Only when the *whole* field is one expression; `{{a}} and {{b}}` is
        // text that happens to start with one.
        if !inner.contains("}}") {
            if let Some(value) = resolve(inner, context) {
                return value;
            }
        }
    }
    Value::String(interpolate(template, context))
}

/// Fill in every string inside a JSON structure, in place.
///
/// This is how a step's parameters are prepared: the shape stays whatever the
/// tool's schema asks for, and only the leaves are substituted.
pub fn fill(value: &Value, context: &Context) -> Value {
    match value {
        Value::String(s) => resolve_value(s, context),
        Value::Array(items) => Value::Array(items.iter().map(|v| fill(v, context)).collect()),
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), fill(v, context))).collect())
        }
        other => other.clone(),
    }
}

/// Look up one expression. `None` when the path leads nowhere.
pub fn resolve(expression: &str, context: &Context) -> Option<Value> {
    let expression = expression.trim();
    if expression.is_empty() {
        return None;
    }

    let (head, rest) = split_head(expression);
    let mut current = context.get(head)?.clone();

    for step in Path::new(rest) {
        current = match (step, current) {
            (Step::Key(k), Value::Object(map)) => map.get(&k)?.clone(),
            (Step::Index(i), Value::Array(items)) => items.get(i)?.clone(),
            // A path into something that is not a container is a mistake in
            // the template, not an empty value.
            _ => return None,
        };
    }
    Some(current)
}

/// Split the step id from the path into its output.
fn split_head(expression: &str) -> (&str, &str) {
    let cut = expression
        .find(['.', '['])
        .unwrap_or(expression.len());
    let (head, rest) = expression.split_at(cut);
    (head.trim(), rest)
}

#[derive(Debug, PartialEq)]
enum Step {
    Key(String),
    Index(usize),
}

/// Walks `.key` and `[0]` segments.
struct Path<'a>(&'a str);

impl<'a> Path<'a> {
    fn new(text: &'a str) -> Self {
        Self(text)
    }
}

impl Iterator for Path<'_> {
    type Item = Step;

    fn next(&mut self) -> Option<Step> {
        let text = self.0.trim_start();
        if let Some(rest) = text.strip_prefix('.') {
            let cut = rest.find(['.', '[']).unwrap_or(rest.len());
            let (key, tail) = rest.split_at(cut);
            self.0 = tail;
            let key = key.trim();
            return (!key.is_empty()).then(|| Step::Key(key.to_string()));
        }
        if let Some(rest) = text.strip_prefix('[') {
            let end = rest.find(']')?;
            let index: usize = rest[..end].trim().parse().ok()?;
            self.0 = &rest[end + 1..];
            return Some(Step::Index(index));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn context() -> Context {
        Context::from([
            ("trigger".to_string(), json!({ "city": "Oslo", "count": 3 })),
            (
                "search".to_string(),
                json!({
                    "results": [
                        { "title": "First", "url": "https://a.test" },
                        { "title": "Second", "url": "https://b.test" }
                    ],
                    "count": 2
                }),
            ),
        ])
    }

    #[test]
    fn a_plain_template_is_left_alone() {
        assert_eq!(interpolate("just text", &context()), "just text");
        assert_eq!(interpolate("", &context()), "");
    }

    #[test]
    fn a_reference_is_replaced_by_what_the_step_produced() {
        assert_eq!(interpolate("in {{ trigger.city }} today", &context()), "in Oslo today");
    }

    #[test]
    fn paths_walk_objects_and_arrays() {
        let c = context();
        assert_eq!(resolve("search.results[0].title", &c), Some(json!("First")));
        assert_eq!(resolve("search.results[1].url", &c), Some(json!("https://b.test")));
        assert_eq!(resolve("search.count", &c), Some(json!(2)));
    }

    #[test]
    fn whitespace_inside_the_braces_does_not_matter() {
        let c = context();
        assert_eq!(interpolate("{{trigger.city}}", &c), "Oslo");
        assert_eq!(interpolate("{{  trigger.city  }}", &c), "Oslo");
    }

    #[test]
    fn a_whole_field_that_is_one_reference_keeps_its_type() {
        // A tool whose schema says `count: integer` rejects the string "3",
        // so substituting into a whole field must not stringify it.
        let c = context();
        assert_eq!(resolve_value("{{ trigger.count }}", &c), json!(3));
        assert_eq!(resolve_value("{{ search.results }}", &c), c["search"]["results"]);
        // But a reference with text around it is text.
        assert_eq!(resolve_value("about {{ trigger.count }}", &c), json!("about 3"));
    }

    #[test]
    fn structured_data_lands_as_json_and_not_as_a_debug_string() {
        // The point of putting a step's output in a prompt is to give the
        // model the data.
        let text = interpolate("results: {{ search.results }}", &context());
        assert!(text.contains("\"title\":\"First\""), "{text}");
        assert!(!text.contains("object"), "{text}");
    }

    #[test]
    fn an_unknown_reference_is_left_standing_rather_than_blanked() {
        // "has not run yet" and "returned nothing" are different, and a
        // blanked expression makes them indistinguishable in the prompt.
        let c = context();
        assert_eq!(interpolate("x {{ nope.field }} y", &c), "x {{ nope.field }} y");
        assert_eq!(interpolate("{{ trigger.missing }}", &c), "{{ trigger.missing }}");
        assert_eq!(interpolate("{{ search.results[9] }}", &c), "{{ search.results[9] }}");
    }

    #[test]
    fn several_references_in_one_template_all_resolve() {
        let out = interpolate("{{ trigger.city }} has {{ search.count }} results", &context());
        assert_eq!(out, "Oslo has 2 results");
    }

    #[test]
    fn an_unclosed_brace_is_text_and_not_an_error() {
        // Someone writing about the syntax, or halfway through typing it.
        assert_eq!(interpolate("use {{ trigger.city", &context()), "use {{ trigger.city");
    }

    #[test]
    fn nothing_but_step_outputs_can_be_reached() {
        // The property that keeps this from being a scripting language: there
        // is no environment, no filesystem, no process, and no way to call
        // anything.
        let c = context();
        for expression in [
            "process.env.HOME",
            "constructor",
            "__proto__.polluted",
            "../../etc/passwd",
            "trigger.constructor.name",
        ] {
            assert_eq!(resolve(expression, &c), None, "{expression} resolved");
        }
    }

    #[test]
    fn filling_a_structure_substitutes_only_its_leaves() {
        let c = context();
        let params = json!({
            "query": "news about {{ trigger.city }}",
            "count": "{{ trigger.count }}",
            "nested": { "list": ["{{ trigger.city }}", 7] }
        });
        assert_eq!(
            fill(&params, &c),
            json!({
                "query": "news about Oslo",
                "count": 3,
                "nested": { "list": ["Oslo", 7] }
            })
        );
    }

    #[test]
    fn a_reference_to_a_container_used_as_a_path_prefix_fails_cleanly() {
        let c = context();
        assert_eq!(resolve("search.count.deeper", &c), None);
        assert_eq!(resolve("search.results.title", &c), None);
    }
}
