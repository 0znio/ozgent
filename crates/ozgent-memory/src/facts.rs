//! Extracting durable facts from conversation.
//!
//! Following the passive-extraction approach: after a turn completes, a cheap
//! pass proposes short standalone statements worth remembering. It runs off
//! the critical path, so it never adds latency to a reply.
//!
//! This module owns the prompt, the parsing, and deduplication. The model call
//! itself is injected, so extraction is testable without an engine.

use crate::store::{Scope, Store, StoreError, Triple};

/// The instruction given to the extractor model.
///
/// It insists on self-contained statements because a fact retrieved months
/// later has no surrounding conversation to disambiguate it — "he prefers
/// tabs" is useless without a name.
pub const EXTRACTION_PROMPT: &str = "\
From the exchange below, list durable facts worth remembering about the user or \
their project. Rules:
- One fact per line, no numbering or bullets.
- Write each as: subject | relation | value
  For example: the user | preferred editor | Helix
- The subject is who or what the fact is about; the relation is the property \
being stated; the value is what it is. Name the same property the same way \
every time, so a later correction can be recognised as one.
- If a fact genuinely will not fit that shape, write it as a plain sentence \
instead, standing alone without the conversation: resolve every pronoun.
- Only lasting facts: preferences, decisions, constraints, names, setups.
- Skip anything transient, hypothetical, or already obvious.
- If there is nothing worth keeping, reply with exactly: NONE";

/// A fact proposed by the extractor, before it is stored.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub scope: Scope,
    /// Set when the extractor managed the `subject | relation | value` shape.
    /// Only these can supersede an earlier statement of the same property.
    pub triple: Option<Triple>,
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
        .map(parse_candidate)
        .collect()
}

/// Turn one extracted line into a candidate.
///
/// A line in `subject | relation | value` form becomes a triple and a readable
/// sentence; anything else is kept verbatim as prose. Falling back rather than
/// rejecting matters — a fact the extractor could not decompose is still worth
/// remembering, it just cannot take part in supersession.
fn parse_candidate(line: &str) -> Candidate {
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    if let [subject, relation, value] = parts[..] {
        if !subject.is_empty() && !relation.is_empty() && !value.is_empty() {
            let text = format!("{subject} — {relation}: {value}");
            let scope = classify_scope(&text);
            return Candidate {
                text,
                scope,
                triple: Some(Triple {
                    subject: subject.to_string(),
                    relation: relation.to_string(),
                    value: value.to_string(),
                }),
            };
        }
    }
    Candidate { text: line.to_string(), scope: classify_scope(line), triple: None }
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

/// Store candidates, superseding what they correct.
///
/// The old behaviour was to skip anything resembling a known fact, which got
/// corrections exactly backwards: told "I prefer tabs" and later "I prefer
/// spaces", a similarity test either dropped the correction and kept the stale
/// value, or kept both and handed the model a contradiction. Neither is
/// recoverable at read time.
///
/// A candidate carrying a triple is therefore matched on subject and relation
/// — an exact key, not a resemblance — and if a live fact already states that
/// property, it is superseded by this one. Same value or different, the later
/// statement is what the user last said. Prose candidates have no such key and
/// keep the old near-duplicate skip, which is the best available for them.
///
/// Returns the ids actually inserted.
pub fn store_candidates(
    store: &Store,
    conversation_id: i64,
    source_message_id: Option<i64>,
    candidates: &[Candidate],
) -> Result<Vec<i64>, StoreError> {
    let mut seen: Vec<String> = store
        .facts_for(conversation_id)?
        .into_iter()
        .filter(|f| f.triple.is_none())
        .map(|f| normalize_for_compare(&f.text))
        .collect();

    let mut inserted = Vec::new();

    for c in candidates {
        match &c.triple {
            Some(triple) => {
                let prior = store.live_fact_for_property(Some(conversation_id), triple)?;
                // Nothing to record when the value is unchanged: superseding a
                // fact with an identical one would churn the store and lose
                // the original's age for no gain.
                if prior.as_ref().and_then(|p| p.triple.as_ref()).is_some_and(|t| {
                    t.value.trim().eq_ignore_ascii_case(triple.value.trim())
                }) {
                    continue;
                }
                let id = store.add_fact_with_triple(
                    Some(conversation_id),
                    c.scope,
                    &c.text,
                    source_message_id,
                    Some(triple),
                )?;
                if let Some(prior) = prior {
                    store.supersede_fact(prior.id, id)?;
                }
                inserted.push(id);
            }
            None => {
                let key = normalize_for_compare(&c.text);
                if seen.iter().any(|e| is_duplicate(e, &key)) {
                    continue;
                }
                let id =
                    store.add_fact(Some(conversation_id), c.scope, &c.text, source_message_id)?;
                seen.push(key);
                inserted.push(id);
            }
        }
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

    #[test]
    fn a_line_in_triple_form_becomes_a_triple() {
        let c = parse_candidate("the user | preferred editor | Helix");
        let t = c.triple.expect("triple");
        assert_eq!(t.subject, "the user");
        assert_eq!(t.relation, "preferred editor");
        assert_eq!(t.value, "Helix");
        assert!(c.text.contains("Helix"), "{}", c.text);
    }

    #[test]
    fn prose_still_works() {
        // Not everything decomposes, and a fact that does not is still worth
        // keeping — it simply cannot supersede anything.
        let c = parse_candidate("The deploy script must run before the tests.");
        assert!(c.triple.is_none());
        assert_eq!(c.text, "The deploy script must run before the tests.");
    }

    #[test]
    fn a_half_formed_triple_falls_back_to_prose() {
        assert!(parse_candidate("the user |  | Helix").triple.is_none());
        assert!(parse_candidate("a | b").triple.is_none());
    }

    #[test]
    fn the_same_property_stated_differently_supersedes() {
        let store = Store::open_in_memory().expect("store");
        let conv = store.create_conversation("t", None).expect("conv");
        let first = parse_candidate("the user | preferred editor | Vim");
        store_candidates(&store, conv, None, &[first]).expect("first");

        let second = parse_candidate("the user | Preferred Editor | Helix");
        store_candidates(&store, conv, None, &[second]).expect("second");

        let live = store.facts_for(conv).expect("facts");
        let editors: Vec<&str> = live
            .iter()
            .filter(|f| f.triple.as_ref().is_some_and(|t| t.relation.to_lowercase().contains("editor")))
            .map(|f| f.text.as_str())
            .collect();
        assert_eq!(editors.len(), 1, "one live answer, got {editors:?}");
        assert!(editors[0].contains("Helix"), "the correction must win: {editors:?}");
    }

    #[test]
    fn restating_the_same_value_changes_nothing() {
        let store = Store::open_in_memory().expect("store");
        let conv = store.create_conversation("t", None).expect("conv");
        let c = parse_candidate("the user | preferred editor | Helix");
        let first = store_candidates(&store, conv, None, &[c.clone()]).expect("first");
        let again = store_candidates(&store, conv, None, &[c]).expect("again");
        assert_eq!(first.len(), 1);
        assert!(again.is_empty(), "an unchanged value should not be re-stored");
    }

    #[test]
    fn different_properties_coexist() {
        // Supersession keys on the property, so unrelated facts about the same
        // subject must not evict each other.
        let store = Store::open_in_memory().expect("store");
        let conv = store.create_conversation("t", None).expect("conv");
        store_candidates(&store, conv, None, &[
            parse_candidate("the user | preferred editor | Helix"),
            parse_candidate("the user | preferred shell | zsh"),
        ])
        .expect("store");
        assert_eq!(store.facts_for(conv).expect("facts").len(), 2);
    }
}
