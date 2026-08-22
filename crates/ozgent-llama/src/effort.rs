//! Bounding how long a reasoning model may think.
//!
//! Asking a model to "think briefly" in the prompt is advice it routinely
//! ignores, and cutting the stream once it has thought too long wastes the
//! tokens already spent and leaves an unterminated block the parser then has to
//! guess about. Writing the model's `</think>` for it does neither: the block
//! closes properly, the model reads its own reasoning as finished, and the
//! answer follows normally.
//!
//! This only watches. Deciding when to inject, and injecting, belongs to the
//! generation loop that owns the batch.

/// Text that closes a reasoning block, matching what the models emit.
pub const CLOSE: &str = "</think>\n\n";

/// The same tag set the reasoning filter recognises.
///
/// Shared rather than duplicated: a model whose reasoning the filter can see
/// but the budget cannot would think without limit while appearing bounded,
/// and the two lists drifting apart is exactly how that happens.
use crate::thinking::DEFAULT_TAGS;

/// Longest tag, plus room for it to straddle a token boundary.
const WINDOW: usize = 24;

/// Tracks whether generation is inside a reasoning block, and for how long.
#[derive(Debug)]
pub struct ThinkBudget {
    budget: u32,
    spent: u32,
    inside: bool,
    /// Set once the block has been closed — by the model or by us — so a later
    /// mention of the tag in the answer cannot reopen it.
    finished: bool,
    tail: String,
}

impl ThinkBudget {
    /// `budget` of zero disables the limit entirely.
    pub fn new(budget: u32) -> Self {
        Self { budget, spent: 0, inside: false, finished: false, tail: String::new() }
    }

    /// Start already inside a reasoning block.
    ///
    /// Chat templates for reasoning models open `<think>` in the prompt, so
    /// generation begins inside the block and the opening tag never appears in
    /// the output. A watcher waiting to see one would wait forever.
    pub fn resumed(budget: u32) -> Self {
        Self { budget, spent: 0, inside: true, finished: false, tail: String::new() }
    }

    /// Whether `prompt` leaves a reasoning block open.
    pub fn opens_thinking(prompt: &str) -> bool {
        crate::thinking::open_at_end(prompt).is_some()
    }

    /// Whether the block is open and the budget is gone.
    pub fn exhausted(&self) -> bool {
        self.inside && !self.finished && self.budget > 0 && self.spent >= self.budget
    }

    pub fn inside(&self) -> bool {
        self.inside
    }

    pub fn spent(&self) -> u32 {
        self.spent
    }

    /// Record that the block was closed for the model.
    pub fn closed(&mut self) {
        self.inside = false;
        self.finished = true;
    }

    /// Feed one token's worth of decoded text.
    pub fn observe(&mut self, text: &str) {
        if self.finished || text.is_empty() {
            return;
        }
        if self.inside {
            self.spent += 1;
        }

        self.tail.push_str(text);
        if self.tail.len() > WINDOW {
            let mut cut = self.tail.len() - WINDOW;
            while !self.tail.is_char_boundary(cut) {
                cut += 1;
            }
            self.tail.drain(..cut);
        }

        // Closing is checked first: a token carrying both tags is the model
        // emitting an empty block, which is finished, not open.
        if DEFAULT_TAGS.iter().any(|t| self.tail.contains(t.close)) {
            self.closed();
            return;
        }
        if !self.inside && DEFAULT_TAGS.iter().any(|t| self.tail.contains(t.open)) {
            self.inside = true;
            self.spent = 0;
            self.tail.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_that_opens_a_block_is_detected() {
        assert!(ThinkBudget::opens_thinking("...<|im_start|>assistant\n<think>\n"));
        // The suppression prefill opens and closes, so it leaves nothing open.
        assert!(!ThinkBudget::opens_thinking("...<think>\n\n</think>\n\n"));
        assert!(!ThinkBudget::opens_thinking("a plain prompt"));
    }

    #[test]
    fn a_resumed_budget_counts_from_the_first_token() {
        let mut b = ThinkBudget::resumed(2);
        assert!(b.inside(), "the prompt already opened the block");
        b.observe("one");
        b.observe("two");
        assert!(b.exhausted());
    }

    #[test]
    fn a_budget_of_zero_never_fires() {
        let mut b = ThinkBudget::new(0);
        b.observe("<think>");
        for _ in 0..10_000 {
            b.observe("word ");
        }
        assert!(!b.exhausted(), "zero means unlimited");
    }

    #[test]
    fn nothing_fires_before_the_block_opens() {
        let mut b = ThinkBudget::new(4);
        for _ in 0..50 {
            b.observe("ordinary prose ");
        }
        assert!(!b.exhausted());
        assert!(!b.inside());
    }

    #[test]
    fn the_budget_counts_only_tokens_inside_the_block() {
        let mut b = ThinkBudget::new(3);
        b.observe("preamble ");
        b.observe("<think>");
        assert!(b.inside());
        b.observe("one");
        b.observe("two");
        assert!(!b.exhausted(), "two of three spent");
        b.observe("three");
        assert!(b.exhausted());
    }

    #[test]
    fn a_model_that_closes_on_its_own_is_left_alone() {
        let mut b = ThinkBudget::new(2);
        b.observe("<think>");
        b.observe("brief");
        b.observe("</think>");
        assert!(!b.inside());
        for _ in 0..20 {
            b.observe("answer ");
        }
        assert!(!b.exhausted(), "the answer must never be interrupted");
    }

    #[test]
    fn an_empty_block_counts_as_finished() {
        // The suppression path prefills `<think>\n\n</think>`; seeing both tags
        // must not be read as an open block.
        let mut b = ThinkBudget::new(2);
        b.observe("<think>\n\n</think>\n\n");
        assert!(!b.inside());
        b.observe("the answer");
        assert!(!b.exhausted());
    }

    #[test]
    fn a_tag_split_across_tokens_is_still_seen() {
        let mut b = ThinkBudget::new(2);
        b.observe("<th");
        b.observe("ink>");
        assert!(b.inside(), "the window spans token boundaries");
    }

    #[test]
    fn mentioning_the_tag_in_the_answer_cannot_reopen_it() {
        let mut b = ThinkBudget::new(2);
        b.observe("<think>");
        b.observe("</think>");
        b.observe("you write <think> like this");
        assert!(!b.inside());
        assert!(!b.exhausted());
    }
}
