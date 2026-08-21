//! Extracting durable facts from conversation.
//!
//! Following the passive-extraction approach: after a turn completes, a cheap
//! pass proposes short standalone statements worth remembering. It runs off
//! the critical path, so it never adds latency to a reply.
//!
//! This module owns the prompt, the parsing, and deduplication. The model call
//! itself is injected, so extraction is testable without an engine.

use crate::store::{Scope, Store, StoreError};

/// The instruction given to the extractor model.
///
/// It insists on self-contained statements because a fact retrieved months
/// later has no surrounding conversation to disambiguate it — "he prefers
/// tabs" is useless without a name.
pub const EXTRACTION_PROMPT: &str = "\
From the exchange below, list durable facts worth remembering about the user or \
their project. Rules:
- One fact per line, no numbering or bullets.
- Each must stand alone without the conversation: resolve every pronoun.
- Only lasting facts: preferences, decisions, constraints, names, setups.
- Skip anything transient, hypothetical, or already obvious.
- If there is nothing worth keeping, reply with exactly: NONE";

/// A fact proposed by the extractor, before it is stored.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub scope: Scope,
}

/// Parse the extractor's reply into candidates.
///
/// Models decorate lists despite instructions, so bullets and numbering are
/// stripped rather than trusted.
pub fn parse_extraction(reply: &str) -> Vec<Candidate> {
    let trimmed = reply.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        return Vec::new();
    }

    trimmed
        .lines()
        .map(strip_list_marker)
        .map(str::trim)
        .filter(|l| l.len() > 3)
        .filter(|l| !l.eq_ignore_ascii_case("none"))
        .map(|l| Candidate { text: l.to_string(), scope: classify_scope(l) })
        .collect()
}

fn strip_list_marker(line: &str) -> &str {
    let l = line.trim();
    for marker in ["- ", "* ", "• ", "– "] {
        if let Some(rest) = l.strip_prefix(marker) {
            return rest;
        }
    }
    // Numbered forms: "1. ", "2) ".
    let digits: String = l.chars().take_while(char::is_ascii_digit).collect();
    if !digits.is_empty() {
        let rest = &l[digits.len()..];
        for sep in [". ", ") ", "- "] {
            if let Some(r) = rest.strip_prefix(sep) {
                return r;
            }
        }
    }
    l
}

/// Guess whether a fact is about the user generally or only this conversation.
///
/// Getting this wrong is cheap in one direction and expensive in the other: a
/// user-scoped fact follows them everywhere, so the default is the narrower
/// conversation scope.
fn classify_scope(text: &str) -> Scope {
    let lower = text.to_ascii_lowercase();
    const USER_MARKERS: &[&str] = &[
        "the user prefers", "the user likes", "the user uses", "the user's name",
        "the user works", "the user is", "the user always", "the user never",
        "prefers", "always uses",
    ];
    if USER_MARKERS.iter().any(|m| lower.contains(m)) {
        Scope::User
    } else {
        Scope::Conversation
    }
}

/// Store candidates, skipping near-duplicates of what is already known.
///
/// Returns the ids actually inserted.
pub fn store_candidates(
    store: &Store,
    conversation_id: i64,
    source_message_id: Option<i64>,
    candidates: &[Candidate],
) -> Result<Vec<i64>, StoreError> {
    let existing: Vec<String> = store
        .facts_for(conversation_id)?
        .into_iter()
        .map(|f| normalize_for_compare(&f.text))
        .collect();

    let mut inserted = Vec::new();
    let mut seen = existing;

    for c in candidates {
        let key = normalize_for_compare(&c.text);
        if seen.iter().any(|e| is_duplicate(e, &key)) {
            continue;
        }
        let id = store.add_fact(Some(conversation_id), c.scope, &c.text, source_message_id)?;
        seen.push(key);
        inserted.push(id);
    }
    Ok(inserted)
}

fn normalize_for_compare(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Duplicate detection by token overlap.
///
/// Exact matching misses trivial rewordings, which is how a fact store fills
/// with near-copies of the same statement.
fn is_duplicate(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let at: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let bt: std::collections::HashSet<&str> = b.split_whitespace().collect();
    if at.is_empty() || bt.is_empty() {
        return false;
    }
    let overlap = at.intersection(&bt).count() as f32;
    overlap / at.len().min(bt.len()) as f32 > 0.85
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_lines() {
        let out = parse_extraction("The user runs Arch Linux.\nThe user has an RTX 5050.");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "The user runs Arch Linux.");
    }

    #[test]
    fn strips_bullets_and_numbering() {
        let out = parse_extraction("- alpha fact here\n* beta fact here\n1. gamma fact here\n2) delta fact here");
        let texts: Vec<&str> = out.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, vec!["alpha fact here", "beta fact here", "gamma fact here", "delta fact here"]);
    }

    #[test]
    fn none_yields_no_candidates() {
        assert!(parse_extraction("NONE").is_empty());
        assert!(parse_extraction("none").is_empty());
        assert!(parse_extraction("   ").is_empty());
        assert!(parse_extraction("").is_empty());
    }

    #[test]
    fn user_scope_is_recognised_but_defaults_narrow() {
        let out = parse_extraction("The user prefers tabs over spaces.\nThe build uses cmake 4.4.");
        assert_eq!(out[0].scope, Scope::User, "explicit preference is about the user");
        assert_eq!(out[1].scope, Scope::Conversation, "default must be the narrow scope");
    }

    #[test]
    fn duplicate_detection_catches_rewordings() {
        assert!(is_duplicate("the user runs arch linux", "the user runs arch linux"));
        assert!(!is_duplicate("the user runs arch linux", "the user runs ubuntu instead"));
        assert!(!is_duplicate("", "anything"));
    }

    #[test]
    fn normalisation_ignores_punctuation_and_case() {
        assert_eq!(
            normalize_for_compare("The User's Name is Ada!"),
            normalize_for_compare("the users name is ada")
        );
    }
}
