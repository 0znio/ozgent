//! Splitting reasoning traces out of a token stream.
//!
//! Reasoning models wrap their scratchpad in tags. Two things make this harder
//! than a regex over the finished text:
//!
//! 1. We are streaming. Output must be emitted as it arrives, so the filter can
//!    only hold back the few characters that might still turn out to be part of
//!    a tag.
//! 2. Tags arrive split across token boundaries. `<think>` routinely appears as
//!    `<`, `th`, `ink>`, so any check against a single token is wrong.
//!
//! The filter therefore keeps the longest suffix of its buffer that is a proper
//! prefix of some tag, and releases everything before it.

use ozgent_core::ThinkingMode;

/// A piece of classified output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chunk {
    /// Visible answer text.
    Answer(String),
    /// Reasoning. Suppressed entirely when the mode is [`ThinkingMode::Off`].
    Thinking(String),
}

/// An open/close tag pair a model might use for reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TagPair {
    pub open: &'static str,
    pub close: &'static str,
}

/// Tag pairs in use across current reasoning models.
pub const DEFAULT_TAGS: &[TagPair] = &[
    TagPair { open: "<think>", close: "</think>" },
    TagPair { open: "<thinking>", close: "</thinking>" },
    TagPair { open: "<|thinking|>", close: "<|/thinking|>" },
    TagPair { open: "<reasoning>", close: "</reasoning>" },
    TagPair { open: "<|channel|>analysis<|message|>", close: "<|end|>" },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Emitting answer text, watching for an opening tag.
    Answer,
    /// Inside a reasoning span, watching for the matching close tag.
    Thinking { close: &'static str },
}

pub struct ThinkingFilter {
    mode: ThinkingMode,
    tags: &'static [TagPair],
    state: State,
    /// Text not yet classifiable, because it may be a partial tag.
    buf: String,
    /// Whether any reasoning was seen, so callers can report it.
    saw_thinking: bool,
    /// Suppresses the blank line a model usually leaves after a close tag.
    trim_next_answer: bool,
}

impl ThinkingFilter {
    pub fn new(mode: ThinkingMode) -> Self {
        Self {
            mode,
            tags: DEFAULT_TAGS,
            state: State::Answer,
            buf: String::new(),
            saw_thinking: false,
            trim_next_answer: false,
        }
    }

    /// Start already inside a reasoning span.
    ///
    /// Needed when the prompt prefills an opening tag to force a reasoning
    /// model into (or out of) thinking: the model never emits the open tag
    /// itself, only the close.
    pub fn starting_inside(mut self, close: &'static str) -> Self {
        self.state = State::Thinking { close };
        self
    }

    pub fn with_tags(mut self, tags: &'static [TagPair]) -> Self {
        self.tags = tags;
        self
    }

    pub fn saw_thinking(&self) -> bool {
        self.saw_thinking
    }

    /// Whether the model is currently mid-reasoning, for a UI spinner.
    pub fn is_thinking(&self) -> bool {
        matches!(self.state, State::Thinking { .. })
    }

    /// Feed newly generated text and take whatever can now be classified.
    pub fn push(&mut self, text: &str) -> Vec<Chunk> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Answer => {
                    if !self.step_answer(&mut out) {
                        break;
                    }
                }
                State::Thinking { close } => {
                    if !self.step_thinking(close, &mut out) {
                        break;
                    }
                }
            }
        }
        out
    }

    /// Flush at end of stream. Anything still buffered was never a tag.
    pub fn finish(&mut self) -> Vec<Chunk> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        if !rest.is_empty() {
            match self.state {
                State::Answer => self.emit_answer(rest, &mut out),
                State::Thinking { .. } => self.emit_thinking(rest, &mut out),
            }
        }
        out
    }

    /// Returns true if the state changed and the loop should run again.
    fn step_answer(&mut self, out: &mut Vec<Chunk>) -> bool {
        // Earliest opening tag wins, so a model using two vocabularies cannot
        // desynchronise the machine.
        let hit = self
            .tags
            .iter()
            .filter_map(|t| self.buf.find(t.open).map(|i| (i, *t)))
            .min_by_key(|(i, _)| *i);

        if let Some((idx, tag)) = hit {
            let before = self.buf[..idx].to_string();
            self.buf.drain(..idx + tag.open.len());
            if !before.is_empty() {
                self.emit_answer(before, out);
            }
            self.state = State::Thinking { close: tag.close };
            self.saw_thinking = true;
            return true;
        }

        // No complete tag. Hold back only what could still become one.
        let keep = self
            .tags
            .iter()
            .map(|t| partial_suffix_len(&self.buf, t.open))
            .max()
            .unwrap_or(0);

        let split = self.buf.len() - keep;
        if split > 0 {
            let ready: String = self.buf.drain(..split).collect();
            self.emit_answer(ready, out);
        }
        false
    }

    fn step_thinking(&mut self, close: &'static str, out: &mut Vec<Chunk>) -> bool {
        if let Some(idx) = self.buf.find(close) {
            let before = self.buf[..idx].to_string();
            self.buf.drain(..idx + close.len());
            if !before.is_empty() {
                self.emit_thinking(before, out);
            }
            self.state = State::Answer;
            // Models almost always emit "\n\n" right after closing; dropping it
            // stops the answer starting with a blank line.
            self.trim_next_answer = true;
            return true;
        }

        let keep = partial_suffix_len(&self.buf, close);
        let split = self.buf.len() - keep;
        if split > 0 {
            let ready: String = self.buf.drain(..split).collect();
            self.emit_thinking(ready, out);
        }
        false
    }

    fn emit_answer(&mut self, mut text: String, out: &mut Vec<Chunk>) {
        if self.trim_next_answer {
            let trimmed = text.trim_start_matches(['\n', '\r']).to_string();
            // Only stop trimming once real content arrives, since the newlines
            // may themselves be split across chunks.
            if !trimmed.is_empty() {
                self.trim_next_answer = false;
            }
            text = trimmed;
        }
        if !text.is_empty() {
            out.push(Chunk::Answer(text));
        }
    }

    fn emit_thinking(&mut self, text: String, out: &mut Vec<Chunk>) {
        self.saw_thinking = true;
        if self.mode != ThinkingMode::Off && !text.is_empty() {
            out.push(Chunk::Thinking(text));
        }
    }
}

/// Length of the longest suffix of `haystack` that is a proper prefix of `tag`.
///
/// This is exactly the amount that must stay buffered: it might be the start of
/// a tag whose remainder has not arrived yet.
fn partial_suffix_len(haystack: &str, tag: &str) -> usize {
    let max = tag.len().min(haystack.len()).saturating_sub(0);
    for len in (1..=max).rev() {
        let start = haystack.len() - len;
        // A split must not land inside a multi-byte character.
        if !haystack.is_char_boundary(start) {
            continue;
        }
        if len < tag.len() && tag.as_bytes().starts_with(&haystack.as_bytes()[start..]) {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the filter one character at a time — the worst case for tag
    /// splitting, and a superset of any real tokenisation.
    fn run_char_by_char(mode: ThinkingMode, input: &str) -> Vec<Chunk> {
        let mut f = ThinkingFilter::new(mode);
        let mut out = Vec::new();
        for ch in input.chars() {
            out.extend(f.push(&ch.to_string()));
        }
        out.extend(f.finish());
        out
    }

    fn answer(chunks: &[Chunk]) -> String {
        chunks.iter().filter_map(|c| match c {
            Chunk::Answer(s) => Some(s.as_str()),
            _ => None,
        }).collect()
    }

    fn thinking(chunks: &[Chunk]) -> String {
        chunks.iter().filter_map(|c| match c {
            Chunk::Thinking(s) => Some(s.as_str()),
            _ => None,
        }).collect()
    }

    #[test]
    fn separates_thinking_from_answer() {
        let out = run_char_by_char(ThinkingMode::On, "<think>weighing it up</think>The answer is 4.");
        assert_eq!(thinking(&out), "weighing it up");
        assert_eq!(answer(&out), "The answer is 4.");
    }

    #[test]
    fn suppresses_thinking_when_off() {
        let out = run_char_by_char(ThinkingMode::Off, "<think>secret reasoning</think>Hello.");
        assert_eq!(thinking(&out), "", "reasoning must not reach the user");
        assert_eq!(answer(&out), "Hello.");
    }

    #[test]
    fn tags_split_across_token_boundaries_are_still_matched() {
        // The exact failure mode a naive per-token check has.
        let mut f = ThinkingFilter::new(ThinkingMode::On);
        let mut out = Vec::new();
        for tok in ["<", "th", "ink", ">", "hmm", "</", "thi", "nk>", "Done."] {
            out.extend(f.push(tok));
        }
        out.extend(f.finish());

        assert_eq!(thinking(&out), "hmm");
        assert_eq!(answer(&out), "Done.", "no tag fragment may leak into the answer");
    }

    #[test]
    fn text_that_merely_looks_like_a_tag_is_released() {
        let out = run_char_by_char(ThinkingMode::On, "use < and <thi as operators");
        assert_eq!(answer(&out), "use < and <thi as operators");
        assert_eq!(thinking(&out), "");
    }

    #[test]
    fn answer_before_thinking_is_preserved() {
        let out = run_char_by_char(ThinkingMode::On, "Sure. <think>why</think>Because.");
        assert_eq!(answer(&out), "Sure. Because.");
        assert_eq!(thinking(&out), "why");
    }

    #[test]
    fn blank_lines_after_the_close_tag_are_trimmed() {
        let out = run_char_by_char(ThinkingMode::On, "<think>x</think>\n\nReal answer.");
        assert_eq!(answer(&out), "Real answer.", "answer must not start with blank lines");
    }

    #[test]
    fn handles_multiple_thinking_spans() {
        let out = run_char_by_char(ThinkingMode::On, "<think>a</think>One.<think>b</think>Two.");
        assert_eq!(thinking(&out), "ab");
        assert_eq!(answer(&out), "One.Two.");
    }

    #[test]
    fn unclosed_thinking_is_flushed_at_end_of_stream() {
        let out = run_char_by_char(ThinkingMode::On, "<think>ran out of tokens");
        assert_eq!(thinking(&out), "ran out of tokens");
        assert_eq!(answer(&out), "");
    }

    #[test]
    fn prefilled_open_tag_means_starting_inside() {
        // DeepSeek-R1 style: we prefill `<think>`, so only the close arrives.
        let mut f = ThinkingFilter::new(ThinkingMode::On).starting_inside("</think>");
        let mut out = Vec::new();
        for tok in ["reason", "ing", "</think>", "Answer."] {
            out.extend(f.push(tok));
        }
        out.extend(f.finish());

        assert_eq!(thinking(&out), "reasoning");
        assert_eq!(answer(&out), "Answer.");
    }

    #[test]
    fn recognises_alternative_tag_vocabularies() {
        for (input, think, ans) in [
            ("<thinking>a</thinking>b", "a", "b"),
            ("<|thinking|>a<|/thinking|>b", "a", "b"),
            ("<reasoning>a</reasoning>b", "a", "b"),
            ("<|channel|>analysis<|message|>a<|end|>b", "a", "b"),
        ] {
            let out = run_char_by_char(ThinkingMode::On, input);
            assert_eq!(thinking(&out), think, "thinking mismatch for {input}");
            assert_eq!(answer(&out), ans, "answer mismatch for {input}");
        }
    }

    #[test]
    fn multibyte_characters_are_never_split() {
        let out = run_char_by_char(ThinkingMode::On, "<think>日本語の推論</think>答えは4です。🎉");
        assert_eq!(thinking(&out), "日本語の推論");
        assert_eq!(answer(&out), "答えは4です。🎉");
    }

    #[test]
    fn plain_output_with_no_tags_passes_through_unchanged() {
        let text = "Just a normal answer with no reasoning at all.";
        let out = run_char_by_char(ThinkingMode::On, text);
        assert_eq!(answer(&out), text);
        assert!(!out.iter().any(|c| matches!(c, Chunk::Thinking(_))));
    }

    #[test]
    fn streaming_releases_text_eagerly() {
        // Everything except a possible tag prefix must come out immediately;
        // otherwise the UI stalls waiting for a tag that never arrives.
        let mut f = ThinkingFilter::new(ThinkingMode::On);
        let out = f.push("Hello there");
        assert_eq!(answer(&out), "Hello there", "text must not be withheld");
    }

    #[test]
    fn reports_whether_reasoning_occurred() {
        let mut f = ThinkingFilter::new(ThinkingMode::Off);
        assert!(!f.saw_thinking());
        let _ = f.push("<think>x</think>y");
        assert!(f.saw_thinking(), "suppressed reasoning still counts as seen");
        assert!(!f.is_thinking(), "the span closed");
    }
}
