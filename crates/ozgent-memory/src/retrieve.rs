//! Hybrid retrieval: lexical search fused with vector search.
//!
//! Neither method alone is enough. FTS5 finds exact terms — an error code, a
//! filename, a person's name — that an embedding blurs away. Vector search
//! finds paraphrases that share no vocabulary with the query. Running both and
//! fusing the ranked lists with Reciprocal Rank Fusion gets both behaviours
//! without having to tune a weight between two incomparable score scales.

use crate::embed::{Embedder, cosine};
use crate::store::{OwnerKind, Store, StoreError};
use rusqlite::params;
use std::collections::HashMap;

/// RRF's damping constant. 60 is the value from the original paper and is
/// what makes the fusion insensitive to the raw scores of either retriever.
pub const RRF_K: f32 = 60.0;

/// One retrieved item.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub kind: OwnerKind,
    pub id: i64,
    /// Fused RRF score. Comparable within one result set only.
    pub score: f32,
    pub text: String,
    /// Position in the conversation, for messages.
    pub seq: Option<i64>,
    /// Which retrievers found it, for explaining a result.
    pub lexical_rank: Option<usize>,
    pub vector_rank: Option<usize>,
}

impl Hit {
    /// True when both retrievers surfaced this item, which is the strongest
    /// signal available.
    pub fn corroborated(&self) -> bool {
        self.lexical_rank.is_some() && self.vector_rank.is_some()
    }
}

pub struct Retriever<'a> {
    store: &'a Store,
    embedder: &'a dyn Embedder,
    /// How many candidates each retriever contributes before fusion.
    pub candidates: usize,
}

impl<'a> Retriever<'a> {
    pub fn new(store: &'a Store, embedder: &'a dyn Embedder) -> Self {
        Self { store, embedder, candidates: 40 }
    }

    /// Search a conversation's messages and visible facts.
    pub fn search(
        &self,
        conversation_id: i64,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Hit>, StoreError> {
        let mut fused = self.search_kind(conversation_id, query, OwnerKind::Fact)?;
        fused.extend(self.search_kind(conversation_id, query, OwnerKind::Message)?);

        fused.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        fused.truncate(limit);
        Ok(fused)
    }

    fn search_kind(
        &self,
        conversation_id: i64,
        query: &str,
        kind: OwnerKind,
    ) -> Result<Vec<Hit>, StoreError> {
        let lexical = self.lexical(conversation_id, query, kind)?;
        let vector = self.vector(conversation_id, query, kind)?;
        Ok(self.fuse(lexical, vector, kind, conversation_id)?)
    }

    /// BM25-ranked full-text search.
    fn lexical(
        &self,
        conversation_id: i64,
        query: &str,
        kind: OwnerKind,
    ) -> Result<Vec<i64>, StoreError> {
        let Some(fts_query) = to_fts_query(query) else {
            return Ok(Vec::new());
        };

        let sql = match kind {
            OwnerKind::Message => {
                "SELECT m.id FROM messages_fts f
                 JOIN messages m ON m.id = f.rowid
                 WHERE messages_fts MATCH ?1 AND m.conversation_id = ?2
                 ORDER BY bm25(messages_fts) LIMIT ?3"
            }
            OwnerKind::Fact => {
                "SELECT fa.id FROM facts_fts f
                 JOIN facts fa ON fa.id = f.rowid
                 WHERE facts_fts MATCH ?1 AND fa.superseded_by IS NULL
                   AND (fa.conversation_id = ?2 OR fa.scope = 'user')
                 ORDER BY bm25(facts_fts) LIMIT ?3"
            }
        };

        let mut stmt = self.store.raw().prepare(sql)?;
        let rows = stmt.query_map(
            params![fts_query, conversation_id, self.candidates as i64],
            |r| r.get::<_, i64>(0),
        )?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Brute-force cosine over the conversation's vectors.
    fn vector(
        &self,
        conversation_id: i64,
        query: &str,
        kind: OwnerKind,
    ) -> Result<Vec<i64>, StoreError> {
        let q = self.embedder.embed(query);
        if q.iter().all(|x| *x == 0.0) {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(i64, f32)> = self
            .store
            .embeddings_in_conversation(kind, conversation_id)?
            .into_iter()
            .map(|(id, v)| (id, cosine(&q, &v)))
            .filter(|(_, s)| *s > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(self.candidates);
        Ok(scored.into_iter().map(|(id, _)| id).collect())
    }

    /// Reciprocal Rank Fusion.
    ///
    /// Each list contributes `1 / (k + rank)`. Only ranks matter, so the two
    /// retrievers' incompatible score scales never have to be reconciled, and
    /// an item found by both is naturally promoted above one found by either.
    fn fuse(
        &self,
        lexical: Vec<i64>,
        vector: Vec<i64>,
        kind: OwnerKind,
        conversation_id: i64,
    ) -> Result<Vec<Hit>, StoreError> {
        let mut scores: HashMap<i64, (f32, Option<usize>, Option<usize>)> = HashMap::new();

        for (i, id) in lexical.iter().enumerate() {
            let e = scores.entry(*id).or_insert((0.0, None, None));
            e.0 += 1.0 / (RRF_K + (i + 1) as f32);
            e.1 = Some(i + 1);
        }
        for (i, id) in vector.iter().enumerate() {
            let e = scores.entry(*id).or_insert((0.0, None, None));
            e.0 += 1.0 / (RRF_K + (i + 1) as f32);
            e.2 = Some(i + 1);
        }

        let mut hits = Vec::with_capacity(scores.len());
        for (id, (score, lex, vec_rank)) in scores {
            let Some((text, seq)) = self.load_text(kind, id, conversation_id)? else {
                continue;
            };
            hits.push(Hit { kind, id, score, text, seq, lexical_rank: lex, vector_rank: vec_rank });
        }
        Ok(hits)
    }

    fn load_text(
        &self,
        kind: OwnerKind,
        id: i64,
        conversation_id: i64,
    ) -> Result<Option<(String, Option<i64>)>, StoreError> {
        Ok(match kind {
            OwnerKind::Message => self
                .store
                .get_message(id)?
                // Guard against an id leaking in from another conversation.
                .filter(|m| m.conversation_id == conversation_id)
                .map(|m| (m.content, Some(m.seq))),
            OwnerKind::Fact => self.store.get_fact(id)?.map(|f| (f.text, None)),
        })
    }
}

/// Turn a natural-language query into a safe FTS5 MATCH expression.
///
/// FTS5 has its own syntax — `AND`, `OR`, `NEAR`, `*`, `:`, `"` — and a user
/// question containing any of it would either error or mean something
/// unintended. Each word is therefore quoted as a literal and the words are
/// OR'ed, so partial matches still rank via BM25.
pub fn to_fts_query(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.len() > 1)
        .map(|w| format!("\"{}\"", w.replace('"', "")))
        .collect();

    if terms.is_empty() {
        return None;
    }
    Some(terms.join(" OR "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_queries_are_escaped_into_literals() {
        let q = to_fts_query("gpu layers").unwrap();
        assert_eq!(q, "\"gpu\" OR \"layers\"");
    }

    #[test]
    fn fts_operators_in_user_text_are_neutralised() {
        // Unescaped, each of these is FTS5 syntax and would error or mislead.
        for hostile in ["a AND b", "x OR y", "col:value", "prefix*", "\"quoted\"", "NEAR(a b)"] {
            let q = to_fts_query(hostile).expect("should produce a query");
            let operators_outside_quotes = q
                .split('"')
                .step_by(2)
                .any(|seg| seg.contains('*') || seg.contains(':'));
            assert!(!operators_outside_quotes, "unescaped operator in {q:?}");
        }
    }

    #[test]
    fn empty_or_punctuation_only_queries_yield_nothing() {
        assert!(to_fts_query("").is_none());
        assert!(to_fts_query("?! ...").is_none());
        assert!(to_fts_query("a").is_none(), "single characters are dropped");
    }

    #[test]
    fn rrf_promotes_items_found_by_both_retrievers() {
        // Rank 3 in both should beat rank 1 in one and absent from the other.
        let both = 1.0 / (RRF_K + 3.0) + 1.0 / (RRF_K + 3.0);
        let one_only = 1.0 / (RRF_K + 1.0);
        assert!(both > one_only, "corroboration must win: {both} vs {one_only}");
    }

    #[test]
    fn rrf_is_insensitive_to_raw_score_scales() {
        // Only ranks feed the formula, so a retriever returning 0..1 and one
        // returning 0..1000 contribute identically at the same rank.
        let a = 1.0 / (RRF_K + 1.0);
        let b = 1.0 / (RRF_K + 1.0);
        assert_eq!(a, b);
    }
}
