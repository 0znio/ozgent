//! Describing the available tools to a model.
//!
//! Shared by the terminal and the web server so both front ends offer tools in
//! exactly the same words — a model that learns the format in one must not
//! meet a different one in the other.

pub fn tool_preamble(tools: &[ozgent_core::ToolSpec]) -> String {
    let mut out = String::from(
        "You can call tools. To call one, reply with only this, and nothing else:\n\
         <tool_call>{\"name\": \"<tool>\", \"arguments\": {…}}</tool_call>\n\
         Wait for the result before answering.\n\n\
         Call a tool only to get something you do not already have. If the \
         answer is in front of you — in this conversation, or in an image or \
         file attached to it — answer from that directly. Do not search to \
         confirm what you can already see, and do not search just because you \
         are unsure; say what you can see and what you cannot.\n\n\
         Available tools:\n",
    );
    for t in tools {
        out.push_str(&format!("- {}: {}\n", t.name, first_line(&t.description)));
        if let Some(props) = t.input_schema.get("properties").and_then(|p| p.as_object()) {
            let required: Vec<&str> = t
                .input_schema
                .get("required")
                .and_then(|r| r.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            for (name, schema) in props {
                let ty = schema.get("type").and_then(|v| v.as_str()).unwrap_or("any");
                let req = if required.contains(&name.as_str()) { " (required)" } else { "" };
                let desc = schema
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(|d| format!(" — {d}"))
                    .unwrap_or_default();
                out.push_str(&format!("    {name}: {ty}{req}{desc}\n"));
            }
        }
    }
    out
}

/// Extra instruction for a turn that carries images or audio.
///
/// A model shown a picture will still reach for a search engine unless told
/// plainly that it can see: the observed failure was a search for
/// "floating island waterfall anime art style red figure" — every term lifted
/// from the image it was already looking at — followed by an answer describing
/// the search results and a request that the user describe the picture.
pub const MEDIA_RULE: &str =
    "The user has attached media to this message and you can see it directly. \
     Describe or answer from what is actually there. Do not search the web for \
     it, and do not ask the user to describe it back to you.";

/// First line of a description, for one-line summaries.
pub fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}
#[cfg(test)]
mod tests {
    use super::*;
    use ozgent_core::ToolSpec;

    fn spec(name: &str, description: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: description.into(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
            output_schema: None,
        }
    }

    #[test]
    fn the_preamble_tells_the_model_not_to_search_for_what_it_can_see() {
        // Observed failure: shown an image, the model searched the web for
        // terms lifted from that image, then described the results instead.
        let text = tool_preamble(&[spec("web_search", "Look up current facts on the web.")]);
        let lower = text.to_lowercase();
        assert!(lower.contains("attached"), "must mention attachments: {text}");
        assert!(
            lower.contains("already have") || lower.contains("already see"),
            "must say not to fetch what is present: {text}"
        );
    }

    #[test]
    fn only_the_first_line_of_a_description_reaches_the_model() {
        // This is why the rule lives here and in the first line of a docstring:
        // anything written below it is never shown.
        let text = tool_preamble(&[spec(
            "demo",
            "The summary line.\nA second line the model never sees.",
        )]);
        assert!(text.contains("The summary line."));
        assert!(!text.contains("never sees"), "later lines must not leak in");
    }

    #[test]
    fn the_media_rule_forbids_both_searching_and_asking_back() {
        let lower = MEDIA_RULE.to_lowercase();
        assert!(lower.contains("see it directly"));
        assert!(lower.contains("do not search"));
        assert!(lower.contains("describe it back"), "asking the user to describe it is the tell");
    }

    #[test]
    fn every_tool_is_listed_with_its_summary() {
        let text = tool_preamble(&[spec("alpha", "Does alpha."), spec("beta", "Does beta.")]);
        assert!(text.contains("- alpha: Does alpha."));
        assert!(text.contains("- beta: Does beta."));
    }
}
