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
    /// Facts and recalled excerpts go in a system message rather than being
    /// forged as conversation turns, so the model is never misled about what
    /// was actually said in this window.
    pub fn to_messages(&self, system_prompt: Option<&str>) -> Vec<Message> {
        let mut out = Vec::new();
        let mut preamble = String::new();

        if let Some(sp) = system_prompt {
            preamble.push_str(sp.trim());
        }

        if !self.pinned.is_empty() {
            if !preamble.is_empty() {
                preamble.push_str("\n\n");
            }
            preamble.push_str("What you know about the user:\n");
            for f in &self.pinned {
                preamble.push_str("- ");
                preamble.push_str(f.text.trim());
                preamble.push('\n');
            }
        }

        if !self.retrieved.is_empty() {
            if !preamble.is_empty() {
                preamble.push_str("\n");
            }
            preamble.push_str(
                "\nRelevant excerpts from earlier in this conversation, \
                 outside the messages shown below:\n",
            );
            for hit in &self.retrieved {
                match hit.seq {
                    Some(seq) => preamble.push_str(&format!("- (message {seq}) ")),
                    None => preamble.push_str("- "),
                }
                preamble.push_str(hit.text.trim());
                preamble.push('\n');
            }
        }

        if !preamble.trim().is_empty() {
            out.push(Message::system(preamble.trim()));
        }

        for m in &self.recent {
            out.push(Message {
                role: match m.role.as_str() {
                    "system" => Role::System,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    _ => Role::User,
                },
                content: vec![ozgent_core::Part::Text { text: m.content.clone() }],
                thinking: None,
                tool_calls: Vec::new(),
                tool_call_id: m.tool_call_id.clone(),
            });
        }
        out
    }
}

/// Builds context for a turn.
pub struct ContextBuilder<'a> {
    store: &'a Store,
    embedder: &'a dyn Embedder,
    pub budget: Budget,
}

impl<'a> ContextBuilder<'a> {
    pub fn new(store: &'a Store, embedder: &'a dyn Embedder) -> Self {
        Self { store, embedder, budget: Budget::default() }
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

        let retriever = Retriever::new(self.store, self.embedder);
        let hits = retriever.search(conversation_id, query, self.budget.max_retrieved * 3)?;

        for hit in hits {
            if ctx.retrieved.len() >= self.budget.max_retrieved {
                break;
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
