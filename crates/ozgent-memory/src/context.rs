//! Assembling a context window from memory.
//!
//! A hundred-turn conversation cannot be resent every turn, but the model must
//! still answer a question about turn three. Context is therefore built in
//! tiers, each with a different claim on the budget:
//!
//! 1. **Pinned facts** — always present, never subject to retrieval.
//! 2. **Recent window** — the last few turns verbatim, because conversation is
//!    mostly local and pronouns resolve against it.
//! 3. **Retrieved** — older messages and facts that the current question
//!    actually touches, found by hybrid search and included only if they fit.
//!
//! Anything that does not fit is dropped rather than truncated: half a message
//! is worse than none, because the model cannot tell it was cut.

use crate::embed::Embedder;
use crate::retrieve::{Hit, Retriever};
use crate::store::{Fact, OwnerKind, Store, StoreError, StoredMessage};
use ozgent_core::{Message, Role};

/// Characters of one recalled item shown in the prompt.
pub const EXCERPT_CHARS: usize = 800;

/// How much room each tier gets.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Total tokens available for the whole prompt.
    pub total: usize,
    /// Held back so the model has room to answer.
    pub reserve_for_reply: usize,
    /// How many trailing messages are always included verbatim.
    pub recent_messages: usize,
    /// Ceiling on retrieved items, to stop a vague question flooding context.
    pub max_retrieved: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            total: 4096,
            reserve_for_reply: 512,
            recent_messages: 8,
            max_retrieved: 6,
        }
    }
}

impl Budget {
    /// Tokens actually available for prompt content.
    pub fn usable(&self) -> usize {
        self.total.saturating_sub(self.reserve_for_reply)
    }
}

/// The result of assembly, with enough detail to explain itself to the user.
#[derive(Debug, Clone, Default)]
pub struct AssembledContext {
    pub pinned: Vec<Fact>,
    pub retrieved: Vec<Hit>,
    pub recent: Vec<StoredMessage>,
    pub tokens_used: usize,
    /// Messages in the conversation that were left out entirely.
    pub messages_elided: usize,
}

impl AssembledContext {
    /// Whether anything was recalled from beyond the recent window.
    pub fn used_recall(&self) -> bool {
        !self.retrieved.is_empty()
    }

    /// Render as messages for the model.
    ///
    /// The system prompt alone goes first, the same in every conversation,
    /// so the start of the prompt is served from the cache that holds it.
    /// What memory adds — pinned facts, excerpts recalled from beyond the
    /// recent window — goes with the newest message instead: it differs from
    /// one message to the next, and in the system prompt it made every new
    /// conversation re-read the whole start (5,162 tokens, 3 s on a 4B).
    /// It is still labelled for what it is, so the model is never misled
    /// about what was said in the window.
    pub fn to_messages(&self, system_prompt: Option<&str>) -> Vec<Message> {
        let mut out = Vec::new();
        if let Some(sp) = system_prompt.map(str::trim).filter(|s| !s.is_empty()) {
            out.push(Message::system(sp));
        }

        let mut note = String::new();
        if !self.pinned.is_empty() {
            note.push_str("What you know about the user:\n");
            for f in &self.pinned {
                note.push_str("- ");
                note.push_str(f.text.trim());
                note.push('\n');
            }
        }
        if !self.retrieved.is_empty() {
            if !note.is_empty() {
                note.push('\n');
            }
            note.push_str("From memory — earlier in this conversation, or known about the user:\n");
            for hit in &self.retrieved {
                match hit.seq {
                    Some(seq) => note.push_str(&format!("- (message {seq}) ")),
                    None => note.push_str("- "),
                }
                note.push_str(hit.text.trim());
                note.push('\n');
            }
        }

        let newest_user = self.recent.iter().rposition(|m| m.role == "user");
        for (i, m) in self.recent.iter().enumerate() {
            let mut text = m.content.clone();
            if Some(i) == newest_user && !note.is_empty() {
                text = format!("[{}]\n\n{text}", note.trim());
            }
            out.push(Message {
                role: match m.role.as_str() {
                    "system" => Role::System,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    _ => Role::User,
                },
                content: vec![ozgent_core::Part::Text { text }],
                thinking: None,
                tool_calls: Vec::new(),
                tool_call_id: m.tool_call_id.clone(),
            });
        }
        out
    }
}

/// A message without the memory note `to_messages` puts in front of it:
/// what the person actually wrote, for anything that searches by it.
pub fn without_memory_note(text: &str) -> &str {
    if text.starts_with("[What you know about the user") || text.starts_with("[From memory") {
        if let Some(at) = text.find("]\n\n") {
            return &text[at + 3..];
        }
    }
    text
}

/// Whether a recalled text has anything to do with the question: a word of
/// four letters or more in common, near enough (a plural, a tense). The
/// retrievers rank; they do not judge, and the best of nothing relevant is
/// still returned — a saved preference for a news source came back for
/// "what is the capital of France?".
fn relevant(text: &str, query: &str) -> bool {
    const COMMON: &[&str] = &[
        "the", "and", "you", "are", "for", "was", "did", "how", "who", "why", "not", "can", "get", "has", "had",
        "her", "him", "his", "its", "our", "out", "all", "any", "one", "use", "what", "which", "that", "this",
        "with", "have", "from", "about", "your", "there", "they", "them", "then", "when", "where", "would",
        "could", "should", "does", "tell", "know", "like", "just", "some", "much", "many", "more", "most",
    ];
    fn words(t: &str) -> Vec<String> {
        t.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3 || w.chars().any(|c| c.is_ascii_digit()))
            .map(|w| {
                let w = w.to_lowercase();
                if w.len() > 4 {
                    w.strip_suffix("ing").or_else(|| w.strip_suffix("ed")).or_else(|| w.strip_suffix('s')).unwrap_or(&w).to_string()
                } else {
                    w
                }
            })
            .filter(|w| !COMMON.contains(&w.as_str()))
            .collect()
    }
    let asked = words(query);
    words(text).iter().any(|w| asked.contains(w))
}

/// Builds context for a turn.
pub struct ContextBuilder<'a> {
    store: &'a Store,
    embedder: &'a dyn Embedder,
    pub budget: Budget,
    /// The query's vector, when the caller embedded it beforehand.
    query_vector: Option<Vec<f32>>,
}

impl<'a> ContextBuilder<'a> {
    pub fn new(store: &'a Store, embedder: &'a dyn Embedder) -> Self {
        Self { store, embedder, budget: Budget::default(), query_vector: None }
    }

    /// Use a vector for the query computed beforehand, so the builder does
    /// not embed it while the caller holds the store. Empty: keywords only.
    pub fn with_query_vector(mut self, vector: Vec<f32>) -> Self {
        self.query_vector = Some(vector);
        self
    }

    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Assemble context for `query` in `conversation_id`.
    pub fn build(
        &self,
        conversation_id: i64,
        query: &str,
    ) -> Result<AssembledContext, StoreError> {
        let mut ctx = AssembledContext::default();
        let mut used = 0usize;
        let budget = self.budget.usable();

        // Tier 1: pinned facts, which are small and always earn their place.
        for fact in self.store.pinned_facts(conversation_id)? {
            used += estimate_tokens(&fact.text);
            ctx.pinned.push(fact);
        }

        // Tier 2: the recent window, oldest-first.
        let recent = self
            .store
            .recent_messages(conversation_id, self.budget.recent_messages as i64)?;

        // Trim from the front if the window alone overruns the budget: the
        // newest turns matter most.
        let mut window: Vec<StoredMessage> = Vec::new();
        for m in recent.into_iter().rev() {
            let cost = estimate_tokens(&m.content);
            if used + cost > budget && !window.is_empty() {
                break;
            }
            used += cost;
            window.push(m);
        }
        window.reverse();
        ctx.recent = window;

        // Tier 3: whatever the question actually reaches back for.
        let in_window: std::collections::HashSet<i64> =
            ctx.recent.iter().map(|m| m.id).collect();

        // What the window already shows is not searched for again, and a
        // conversation that fits in the window costs no query embedding.
        let mut retriever = Retriever::new(self.store, self.embedder).excluding(in_window.iter().copied());
        if let Some(v) = &self.query_vector {
            retriever = retriever.with_query_vector(v.clone());
        }
        let hits = retriever.search(conversation_id, query, self.budget.max_retrieved * 3)?;

        for mut hit in hits {
            if ctx.retrieved.len() >= self.budget.max_retrieved {
                break;
            }
            if !relevant(&hit.text, query) {
                continue;
            }
            // What tools returned reaches the model another way — the last
            // turn's rides along with its reply, older ones through the
            // memory tool, excerpted where they answer the question. Here it
            // could only be cut from its start, and a model shown the start
            // of a long article answered from that.
            if hit.kind == OwnerKind::Message
                && self.store.get_message(hit.id).ok().flatten().is_some_and(|m| m.role == "tool")
            {
                continue;
            }
            // An excerpt, not the item: a page a tool read can be tens of
            // thousands of characters, and the model can read the rest with
            // its memory tool.
            if let Some((at, _)) = hit.text.char_indices().nth(EXCERPT_CHARS) {
                hit.text = format!("{}…", &hit.text[..at]);
            }
            // Never repeat something already shown verbatim.
            if hit.kind == OwnerKind::Message && in_window.contains(&hit.id) {
                continue;
            }
            if hit.kind == OwnerKind::Fact && ctx.pinned.iter().any(|f| f.id == hit.id) {
                continue;
            }
            let cost = estimate_tokens(&hit.text);
            if used + cost > budget {
                continue; // a smaller later hit may still fit
            }
            used += cost;
            ctx.retrieved.push(hit);
        }

        let total = self.store.message_count(conversation_id)? as usize;
        ctx.messages_elided = total
            .saturating_sub(ctx.recent.len())
            .saturating_sub(ctx.retrieved.iter().filter(|h| h.kind == OwnerKind::Message).count());
        ctx.tokens_used = used;
        Ok(ctx)
    }
}

/// Rough token count.
///
/// Four characters per token is the usual English approximation. It is only
/// used for budgeting, and it deliberately errs high on short strings so the
/// assembler does not overfill. The real tokenizer replaces it once a model is
/// loaded.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.chars().count().div_ceil(4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_scales_with_length() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("a") >= 1);
        assert!(estimate_tokens(&"x".repeat(400)) > estimate_tokens(&"x".repeat(40)));
    }

    #[test]
    fn budget_reserves_room_for_the_reply() {
        let b = Budget { total: 1000, reserve_for_reply: 300, ..Default::default() };
        assert_eq!(b.usable(), 700);
    }

    #[test]
    fn usable_budget_never_underflows() {
        let b = Budget { total: 100, reserve_for_reply: 500, ..Default::default() };
        assert_eq!(b.usable(), 0, "must saturate rather than wrap");
    }
}

#[cfg(test)]
mod note {
    #[test]
    fn the_note_comes_off_and_nothing_else_does() {
        assert_eq!(super::without_memory_note("[From memory — x:\n- a [b] c]\n\n[14:09] what now?"), "[14:09] what now?");
        assert_eq!(super::without_memory_note("[14:09] plain"), "[14:09] plain");
    }
}
