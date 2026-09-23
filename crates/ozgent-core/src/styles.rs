//! Response styles and personas: how a model answers in ozgent's own chats.
//!
//! Two per-model settings, both in the model's options:
//!
//! * `system_prompt` — a persona or standing instruction the user wrote
//!   ("You are my Rust reviewer; be blunt").
//! * `style` — how answers are shaped, by name: a built-in below, or one the
//!   user defined as `[styles.<name>]`.
//!
//! They reach the chats a person has with ozgent — the web page, the
//! terminal, the messaging channels — and never an API caller's request, which
//! brings its own system prompt and would be surprised by one of ours.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A style: a name to pick it by, a line to describe it, and the instruction
/// that shapes answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Style {
    pub name: String,
    pub title: String,
    pub prompt: String,
    /// False for the built-ins, which cannot be edited or removed.
    pub custom: bool,
}

/// `[styles.<name>]`: a style the user wrote.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomStyle {
    /// Shown in lists. Defaults to the name.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    pub prompt: String,
}

/// The built-in styles, as (name, title, instruction).
///
/// Written as instructions about *form*, not persona, so any of them combines
/// with whatever system prompt a model has.
pub const BUILTIN: &[(&str, &str, &str)] = &[
    (
        "concise",
        "Concise — short answers, no padding",
        "Answer concisely. Lead with the answer, use only the sentences or bullet points it needs, \
         and leave out preamble, restating the question, and closing summaries.",
    ),
    (
        "detailed",
        "Detailed — thorough, with reasoning and examples",
        "Give thorough, well-organised answers: explain the reasoning, cover edge cases and \
         trade-offs, include an example where it helps, and use headings or lists when the answer \
         is long.",
    ),
    (
        "to-the-point",
        "To the point — just what was asked",
        "Answer only what was asked: the direct answer first and nothing else unless it changes the \
         answer. No background, no caveats, no suggestions unless requested. One line when one line \
         will do.",
    ),
    (
        "adhd",
        "ADHD-friendly — scannable, one idea at a time",
        "Format every answer for easy focus and scanning. Start with a one-line answer or TL;DR. Then \
         use short bullet points, one idea per bullet, with the key words in bold. Break any task \
         into small numbered steps. No long paragraphs, no tangents. If there is a next action, end \
         with it on its own line.",
    ),
    (
        "beginner",
        "Beginner — plain words, no jargon",
        "Explain as to a curious beginner: plain words, define any technical term the first time it \
         appears, build from the basics, and use a concrete analogy or example.",
    ),
    (
        "expert",
        "Expert — precise and technical",
        "Assume an expert reader. Be precise and technical, skip the basics, use correct terminology, \
         and give specifics — numbers, names, versions — rather than generalities.",
    ),
    (
        "casual",
        "Casual — warm and conversational",
        "Write in a warm, relaxed, conversational tone, like a knowledgeable friend: plain language \
         and contractions, still accurate and still to the point.",
    ),
    (
        "formal",
        "Formal — professional register",
        "Write in a formal, professional register suitable for business or academic use: complete \
         sentences, measured tone, no slang or emoji.",
    ),
    (
        "tutor",
        "Tutor — hints and questions, not answers",
        "Act as a patient tutor. Guide the user towards the answer with questions and hints, one \
         step at a time, and give the full answer only when they ask for it directly.",
    ),
];

/// A style name as the user types it: lowercase, spaces as hyphens.
pub fn normalise_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().split_whitespace().collect::<Vec<_>>().join("-")
}

/// Whether `name` can name a custom style: letters, digits and hyphens, not a
/// built-in's name, and short enough to type.
pub fn valid_custom_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 40 {
        return Err("a style name is 1 to 40 characters".into());
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err("a style name uses letters, digits and hyphens only".into());
    }
    if matches!(name, "off" | "none" | "default") || BUILTIN.iter().any(|(n, _, _)| *n == name) {
        return Err(format!("{name:?} is taken by a built-in style"));
    }
    Ok(())
}

/// Every style there is: the built-ins, then the user's own.
pub fn all(custom: &BTreeMap<String, CustomStyle>) -> Vec<Style> {
    let mut out: Vec<Style> = BUILTIN
        .iter()
        .map(|(n, t, p)| Style { name: n.to_string(), title: t.to_string(), prompt: p.to_string(), custom: false })
        .collect();
    out.extend(custom.iter().map(|(n, c)| Style {
        name: n.clone(),
        title: if c.title.trim().is_empty() { n.clone() } else { c.title.clone() },
        prompt: c.prompt.clone(),
        custom: true,
    }));
    out
}

/// The style called `name`, if there is one.
pub fn find(custom: &BTreeMap<String, CustomStyle>, name: &str) -> Option<Style> {
    let name = normalise_name(name);
    all(custom).into_iter().find(|s| s.name == name)
}

/// The system prompt a chat with this model begins with, before ozgent's own
/// context lines: the user's persona, then the style's instruction.
pub fn compose(
    system_prompt: Option<&str>,
    style: Option<&str>,
    custom: &BTreeMap<String, CustomStyle>,
) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(p) = system_prompt.map(str::trim).filter(|p| !p.is_empty()) {
        parts.push(p.to_string());
    }
    if let Some(s) = style.and_then(|n| find(custom, n)) {
        parts.push(format!("Response style: {}", s.prompt));
    }
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_ins_have_unique_valid_names() {
        let mut names: Vec<_> = BUILTIN.iter().map(|(n, _, _)| *n).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), BUILTIN.len());
        for (n, _, p) in BUILTIN {
            assert_eq!(normalise_name(n), *n);
            assert!(p.len() > 40);
        }
    }

    #[test]
    fn a_custom_style_cannot_shadow_a_built_in() {
        assert!(valid_custom_name("concise").is_err());
        assert!(valid_custom_name("off").is_err());
        assert!(valid_custom_name("pirate").is_ok());
        assert!(valid_custom_name("two words").is_err());
    }

    #[test]
    fn persona_and_style_combine_in_order() {
        let mut custom = BTreeMap::new();
        custom.insert("pirate".into(), CustomStyle { title: String::new(), prompt: "Talk like a pirate.".into() });
        let p = compose(Some("You review Rust."), Some("Pirate"), &custom).unwrap();
        assert!(p.starts_with("You review Rust.") && p.ends_with("Talk like a pirate."), "{p}");
        assert!(compose(None, Some("concise"), &custom).unwrap().contains("concisely"));
        assert_eq!(compose(Some("  "), Some("nonexistent"), &custom), None);
        assert_eq!(compose(None, None, &custom), None);
    }
}
