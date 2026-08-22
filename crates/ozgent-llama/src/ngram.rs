//! Context lookup drafting for self-speculative decoding.
//!
//! Speculation normally needs a second, smaller model. This instead mines the
//! text already in front of us: if the tokens just emitted have appeared
//! before, whatever followed them last time is a plausible continuation. The
//! large model verifies several drafted tokens in a single batch, and because
//! a batch of eight costs barely more than a batch of one on a
//! bandwidth-starved GPU, every accepted token is close to free.
//!
//! Output is bit-identical to unaccelerated decoding — a wrong draft is
//! rejected and the correctly sampled token used instead — so this is purely a
//! latency win, never an approximation.
//!
//! # Why longest-match rather than a fixed n-gram
//!
//! Keying on a fixed two tokens is cheap but imprecise: in a 150k vocabulary a
//! two-token key collides constantly, so the continuation it names is usually
//! wrong, and every wrong draft costs a verification slot. This keeps the
//! short key for O(1) lookup but then *ranks* the candidate positions by how
//! far the match extends backwards, and drafts from the longest one. Precision
//! comes from the ranking, not from a longer key, so the index stays small
//! while the draft quality rises sharply.
//!
//! It shines where models repeat themselves: quoting a file, editing code,
//! rewriting a passage, echoing tool output. On free prose it rarely hits,
//! which is why acceptance is tracked over a rolling window and drafting backs
//! off when it stops paying — then probes occasionally, so a turn that starts
//! as prose and moves on to quoting code picks the acceleration back up.

use std::collections::{HashMap, VecDeque};

/// Length of the key used for O(1) candidate lookup.
///
/// Deliberately short. Precision comes from [`NgramCache::draft`] extending
/// each candidate backwards, not from the key.
pub const NGRAM: usize = 2;

/// How far back a candidate match is scored. Beyond this the ranking is
/// already unambiguous and the extra comparisons are wasted.
const MAX_MATCH: usize = 32;

/// Occurrences retained per key. Hot keys in a long context would otherwise
/// grow without bound, and the oldest occurrences are the least predictive.
const MAX_POSITIONS: usize = 64;

/// Verification rounds kept in the acceptance window.
const WINDOW: usize = 16;

/// Suppressed rounds before drafting is retried, so a change of workload can
/// re-enable acceleration mid-turn.
const PROBE_AFTER: u32 = 24;

/// Drafted tokens needed before the acceptance rate is trusted.
const MIN_EVIDENCE: u32 = 24;

/// The tokens seen so far, indexed for continuation lookup.
#[derive(Debug, Default)]
pub struct NgramCache {
    tokens: Vec<i32>,
    /// Key to the positions where it occurs. A position `p` means
    /// `tokens[p..p + NGRAM]` equals the key, so the continuation is at
    /// `p + NGRAM`.
    index: HashMap<[i32; NGRAM], Vec<u32>>,
    /// Per-round `(proposed, accepted)`, most recent last.
    window: VecDeque<(u16, u16)>,
    proposed: u64,
    accepted: u64,
    /// Consecutive rounds drafting has been suppressed for.
    suppressed: u32,
}

impl NgramCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokens indexed so far. The caller uses this to feed only what is new.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Index a whole sequence, e.g. the prompt before generation starts.
    pub fn extend(&mut self, tokens: &[i32]) {
        self.tokens.reserve(tokens.len());
        for t in tokens {
            self.push_token(*t);
        }
    }

    /// Append one token and index the key that now ends at it.
    ///
    /// Every token goes through here, including ones accepted from a draft —
    /// skipping those would leave holes in the index exactly where the text is
    /// most repetitive, which is where drafting pays best.
    pub fn push_token(&mut self, token: i32) {
        self.tokens.push(token);
        // The key that just became complete starts NGRAM+1 tokens from the end;
        // its continuation is the token just pushed.
        if self.tokens.len() > NGRAM {
            let p = self.tokens.len() - NGRAM - 1;
            let key: [i32; NGRAM] = self.tokens[p..p + NGRAM]
                .try_into()
                .expect("slice is NGRAM long by construction");
            let slot = self.index.entry(key).or_default();
            if slot.len() >= MAX_POSITIONS {
                slot.remove(0);
            }
            slot.push(p as u32);
        }
    }

    /// Propose up to `max` tokens continuing the sequence.
    ///
    /// Returns an empty draft when nothing matches, which is the common case
    /// on novel text and costs nothing.
    /// `min_reach` rejects a match that does not extend at least that far
    /// behind the key. A two-token key collides constantly in ordinary prose
    /// and predicts badly there, while genuine repetition — quoting a file,
    /// reissuing a structure — matches for tens of tokens. Filtering on reach
    /// separates the two before any draft is decoded, which costs nothing:
    /// it is a hash lookup and a backward scan, not a forward pass.
    pub fn draft(&self, max: usize, min_reach: usize) -> Vec<i32> {
        if max == 0 || self.tokens.len() <= NGRAM {
            return Vec::new();
        }
        let cur = self.tokens.len() - NGRAM;
        let key: [i32; NGRAM] = self.tokens[cur..]
            .try_into()
            .expect("slice is NGRAM long by construction");
        let Some(positions) = self.index.get(&key) else {
            return Vec::new();
        };

        // Rank candidates by how far the match extends behind the key. Later
        // positions win ties: in a passage repeated more than once, the most
        // recent copy is the better predictor.
        let mut best: Option<(usize, u32)> = None;
        for &p in positions {
            let p = p as usize;
            if p >= cur {
                continue; // the current occurrence has no continuation yet
            }
            if p + NGRAM >= self.tokens.len() {
                continue;
            }
            let reach = self.match_length(p, cur);
            if (reach as usize) < min_reach {
                continue;
            }
            if best.is_none_or(|(score, _)| reach >= score) {
                best = Some((reach, p as u32));
            }
        }

        let Some((_, p)) = best else { return Vec::new() };
        let from = p as usize + NGRAM;
        let end = (from + max).min(self.tokens.len());
        self.tokens[from..end].to_vec()
    }

    /// How many tokens before `a` and `b` agree, capped at [`MAX_MATCH`].
    fn match_length(&self, a: usize, b: usize) -> usize {
        let mut n = 0;
        while n < MAX_MATCH
            && a > n
            && b > n
            && self.tokens[a - 1 - n] == self.tokens[b - 1 - n]
        {
            n += 1;
        }
        n
    }

    /// How many tokens to draft, given how well drafting has been going.
    ///
    /// Drafting too long when acceptance is poor wastes a whole batch on
    /// tokens that will be thrown away; drafting too short when it is good
    /// leaves free tokens on the table. `cap` is the hard ceiling imposed by
    /// remaining context and batch size.
    pub fn suggest_len(&self, cap: usize) -> usize {
        if cap == 0 {
            return 0;
        }
        let len = match self.acceptance() {
            // No evidence yet: probe with a moderate draft.
            None => cap.min(8),
            Some(r) if r >= 0.75 => cap,
            Some(r) if r >= 0.50 => (cap / 2).max(4),
            Some(r) if r >= 0.25 => 4,
            Some(_) => 2,
        };
        len.min(cap).max(1)
    }

    /// Record the outcome of a verified draft.
    pub fn observe(&mut self, proposed: usize, accepted: usize) {
        self.proposed += proposed as u64;
        self.accepted += accepted as u64;
        if self.window.len() >= WINDOW {
            self.window.pop_front();
        }
        self.window.push_back((proposed as u16, accepted as u16));
    }

    /// Fraction of drafted tokens accepted over the recent window, or `None`
    /// before there is enough evidence to judge.
    ///
    /// Deliberately a rolling window rather than a running total: a turn that
    /// opens with prose and then starts quoting a file should not be judged
    /// forever on how the prose went.
    pub fn acceptance(&self) -> Option<f32> {
        let (p, a) = self
            .window
            .iter()
            .fold((0u32, 0u32), |(p, a), (x, y)| (p + *x as u32, a + *y as u32));
        (p >= MIN_EVIDENCE).then(|| a as f32 / p as f32)
    }

    /// Whether drafting is still worth the verification cost.
    ///
    /// Below the threshold each rejected draft costs more than the occasional
    /// hit saves, so it is suppressed — but only for a while. Every
    /// [`PROBE_AFTER`] rounds one draft is let through to test whether the
    /// text has become predictable again.
    pub fn worth_drafting(&mut self, min_acceptance: f32, probe_every: u32) -> bool {
        match self.acceptance() {
            None => true,
            Some(rate) if rate >= min_acceptance => {
                self.suppressed = 0;
                true
            }
            Some(_) => {
                if self.suppressed >= probe_every {
                    self.suppressed = 0;
                    true
                } else {
                    self.suppressed += 1;
                    false
                }
            }
        }
    }

    /// Lifetime `(proposed, accepted)`, for reporting.
    pub fn stats(&self) -> (u64, u64) {
        (self.proposed, self.accepted)
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(tokens: &[i32]) -> NgramCache {
        let mut c = NgramCache::new();
        c.extend(tokens);
        c
    }

    #[test]
    fn drafts_a_repeated_sequence() {
        // Having seen 1,2 -> 3,4,5 and then arrived back at 1,2, the same
        // continuation should replay.
        let c = seeded(&[1, 2, 3, 4, 5, 9, 1, 2]);
        assert_eq!(c.draft(3, 0), vec![3, 4, 5]);
    }

    #[test]
    fn an_unseen_context_drafts_nothing() {
        let c = seeded(&[1, 2, 3, 4, 90, 91]);
        assert!(c.draft(4, 0).is_empty());
    }

    #[test]
    fn drafting_is_capped_by_the_requested_length() {
        let c = seeded(&[1, 2, 3, 4, 5, 9, 1, 2]);
        assert_eq!(c.draft(2, 0).len(), 2);
        assert_eq!(c.draft(0, 0).len(), 0);
    }

    #[test]
    fn a_short_context_cannot_be_keyed() {
        let c = seeded(&[1, 2]);
        assert!(c.draft(4, 0).is_empty(), "needs more than NGRAM tokens");
    }

    #[test]
    fn the_longer_match_wins_over_the_more_recent_one() {
        // This is the whole point of the rewrite. Key [7,8] occurs twice:
        // once preceded by 5,6 and once by 1,2. The live context is preceded
        // by 5,6, so the *earlier* occurrence is the better predictor even
        // though a recency-only rule would pick the later one.
        //   pos 0:        5,6,7,8 -> 100
        //   pos 5:        1,2,7,8 -> 200
        //   live:     ...,5,6,7,8 -> ?
        let c = seeded(&[5, 6, 7, 8, 100, 1, 2, 7, 8, 200, 42, 5, 6, 7, 8]);
        assert_eq!(
            c.draft(1, 0),
            vec![100],
            "the candidate whose earlier context also matches should win"
        );
    }

    #[test]
    fn recency_breaks_ties_when_matches_are_equally_long() {
        // Both occurrences of 1,2 are preceded by nothing that matches, so the
        // scores tie and the most recent continuation should win.
        let c = seeded(&[1, 2, 7, 0, 1, 2, 9, 0, 1, 2]);
        assert_eq!(c.draft(1, 0), vec![9]);
    }

    #[test]
    fn tokens_accepted_from_a_draft_are_still_indexed() {
        // Regression guard: transitions inside an accepted draft must be
        // learned too, or the index develops holes exactly where the text is
        // most repetitive.
        let mut c = NgramCache::new();
        c.extend(&[1, 2]);
        for t in [3, 4, 5] {
            c.push_token(t);
        }
        c.extend(&[9, 1, 2]);
        assert_eq!(c.draft(3, 0), vec![3, 4, 5]);
    }

    #[test]
    fn len_tracks_every_token_so_callers_can_feed_only_the_new_ones() {
        let mut c = NgramCache::new();
        c.extend(&[1, 2, 3]);
        assert_eq!(c.len(), 3);
        c.push_token(4);
        assert_eq!(c.len(), 4);
    }

    #[test]
    fn acceptance_needs_evidence_before_it_judges() {
        let mut c = NgramCache::new();
        assert!(c.acceptance().is_none(), "no verdict without data");
        assert!(c.worth_drafting(0.2, PROBE_AFTER), "must not disable itself prematurely");
        c.observe(20, 2);
        c.observe(20, 2);
        assert_eq!(c.acceptance(), Some(0.1));
        assert!(!c.worth_drafting(0.2, PROBE_AFTER), "a 10% hit rate is not worth verifying");
    }

    #[test]
    fn a_good_acceptance_rate_keeps_drafting_enabled() {
        let mut c = NgramCache::new();
        c.observe(20, 15);
        c.observe(20, 15);
        assert!(c.worth_drafting(0.2, PROBE_AFTER));
        assert_eq!(c.stats(), (40, 30));
    }

    #[test]
    fn suppressed_drafting_probes_again_rather_than_giving_up_for_good() {
        // A turn that opens with prose and later starts quoting code must be
        // able to pick the acceleration back up.
        let mut c = NgramCache::new();
        c.observe(20, 0);
        c.observe(20, 0);
        assert!(!c.worth_drafting(0.2, PROBE_AFTER));

        let mut probes = 0;
        for _ in 0..(PROBE_AFTER * 2 + 4) {
            if c.worth_drafting(0.2, PROBE_AFTER) {
                probes += 1;
            }
        }
        assert!(probes >= 2, "expected periodic probes, saw {probes}");
    }

    #[test]
    fn acceptance_is_rolling_so_an_old_bad_patch_is_forgiven() {
        let mut c = NgramCache::new();
        for _ in 0..WINDOW {
            c.observe(8, 0);
        }
        assert_eq!(c.acceptance(), Some(0.0));
        for _ in 0..WINDOW {
            c.observe(8, 8);
        }
        assert_eq!(
            c.acceptance(),
            Some(1.0),
            "the window should have rolled past the bad patch"
        );
    }

    #[test]
    fn draft_length_follows_the_acceptance_rate() {
        let mut c = NgramCache::new();
        assert_eq!(c.suggest_len(16), 8, "probe moderately with no evidence");

        c.observe(20, 19);
        c.observe(20, 19);
        assert_eq!(c.suggest_len(16), 16, "a high hit rate earns a full draft");

        let mut poor = NgramCache::new();
        poor.observe(20, 1);
        poor.observe(20, 1);
        assert_eq!(poor.suggest_len(16), 2, "a poor hit rate drafts barely at all");
    }

    #[test]
    fn suggest_len_respects_a_hard_cap() {
        let mut c = NgramCache::new();
        c.observe(20, 19);
        c.observe(20, 19);
        assert_eq!(c.suggest_len(3), 3);
        assert_eq!(c.suggest_len(0), 0);
    }

    #[test]
    fn an_empty_cache_reports_itself_empty() {
        let mut c = NgramCache::new();
        assert!(c.is_empty());
        c.extend(&[1, 2, 3]);
        assert!(!c.is_empty());
    }

    #[test]
    fn extending_with_too_few_tokens_is_safe() {
        let c = seeded(&[1, 2]);
        assert!(c.is_empty());
        assert!(c.draft(4, 0).is_empty());
    }

    #[test]
    fn a_self_loop_does_not_draft_past_the_end() {
        // A run of identical tokens must not produce an unbounded or
        // out-of-range draft.
        let c = seeded(&[7; 64]);
        let d = c.draft(16, 0);
        assert!(d.len() <= 16, "draft exceeded its cap: {}", d.len());
        assert!(d.iter().all(|t| *t == 7));
    }

    #[test]
    fn positions_per_key_stay_bounded() {
        // A hot key in a long context must not grow without bound.
        let mut c = NgramCache::new();
        for _ in 0..(MAX_POSITIONS * 4) {
            c.extend(&[1, 2, 3]);
        }
        let longest = c.index.values().map(Vec::len).max().unwrap_or(0);
        assert!(longest <= MAX_POSITIONS, "kept {longest} positions");
    }
}
