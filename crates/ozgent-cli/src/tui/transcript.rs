//! What the conversation looks like, as lines.
//!
//! A full-screen application cannot let anything print where it likes: the
//! bottom rows belong to the prompt and the status bars, and a stray
//! `println!` would scroll the whole layout by one. So everything that used
//! to be printed is appended here instead, already rendered to ANSI, and the
//! frame decides which of it is on screen.
//!
//! Lines are stored pre-wrapped to the width they were rendered at. A resize
//! therefore needs the *source* back, not the lines — which is why a block
//! keeps the text it came from and can render itself again.

use ozgent_render::{MarkdownRenderer, Theme, display_width};

/// Where a block came from, so it can be laid out again at a new width.
#[derive(Debug, Clone)]
pub enum Source {
    /// Markdown, re-rendered when the terminal is resized.
    Markdown(String),
    /// Already-styled text that wraps but is not parsed — a tool result, a
    /// note, a heading ozgent wrote itself.
    Plain(String),
    /// Exactly these lines, at any width. Boxes and rules draw themselves and
    /// must not be re-flowed into nonsense.
    Fixed(Vec<String>),
    /// A reply being written: reasoning above, answer below.
    ///
    /// One block rather than two because they are re-rendered together on
    /// every token, and because the reasoning has to disappear cleanly the
    /// moment the answer starts — which it cannot do if it is already a
    /// committed block of its own.
    Reply { thinking: Option<String>, answer: String },
}

/// One unit of transcript: a message, a note, a tool card.
#[derive(Debug, Clone)]
pub struct Block {
    pub source: Source,
    /// Rendered lines at the current width.
    lines: Vec<String>,
}

impl Block {
    pub fn markdown(text: impl Into<String>) -> Self {
        Self { source: Source::Markdown(text.into()), lines: Vec::new() }
    }
    pub fn plain(text: impl Into<String>) -> Self {
        Self { source: Source::Plain(text.into()), lines: Vec::new() }
    }
    pub fn fixed(lines: Vec<String>) -> Self {
        Self { source: Source::Fixed(lines), lines: Vec::new() }
    }
    pub fn reply(thinking: Option<String>, answer: impl Into<String>) -> Self {
        Self { source: Source::Reply { thinking, answer: answer.into() }, lines: Vec::new() }
    }

    fn render(&mut self, theme: &Theme, width: usize) {
        self.lines = match &self.source {
            Source::Fixed(lines) => lines.clone(),
            Source::Plain(text) => wrap_styled(text, width),
            Source::Markdown(text) => render_markdown(theme, width, text),
            Source::Reply { thinking, answer } => {
                let mut lines = Vec::new();
                // Reasoning is subordinate to the answer and is styled that
                // way rather than parsed: a model's scratch notes are not
                // markdown, and rendering them as such turns a stray `#` into
                // a heading three times the weight of the reply.
                if let Some(text) = thinking.as_ref().filter(|t| !t.trim().is_empty()) {
                    let styled = theme.style(theme.thinking, text.trim());
                    lines.extend(wrap_styled(&styled, width));
                    if !answer.trim().is_empty() {
                        lines.push(String::new());
                    }
                }
                if !answer.is_empty() {
                    lines.extend(render_markdown(theme, width, answer));
                }
                lines
            }
        };
    }
}

/// Every block, and the window of it currently on screen.
pub struct Transcript {
    blocks: Vec<Block>,
    /// The block being written to right now, replaced on every token.
    ///
    /// Separate from `blocks` because a streaming reply is re-rendered from
    /// its whole source each time a token arrives: markdown cannot be
    /// appended to line by line — a closing fence changes how everything
    /// since the opening one is drawn.
    live: Option<Block>,
    theme: Theme,
    width: usize,
    /// Lines scrolled back from the bottom. Zero means following the tail,
    /// which is the state a new line must not silently leave.
    scroll: usize,
}

impl Transcript {
    pub fn new(theme: Theme, width: usize) -> Self {
        Self { blocks: Vec::new(), live: None, theme, width, scroll: 0 }
    }

    /// Exercised by the tests, which is the only caller that needs it: the
    /// application always has a banner in the transcript by the time anything
    /// asks.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty() && self.live.is_none()
    }

    /// Re-render everything for a new width.
    ///
    /// The whole point of keeping sources: lines wrapped to eighty columns are
    /// wrong at a hundred and twenty, and there is no way back to the text
    /// from the lines once colour and indentation have been baked in.
    pub fn resize(&mut self, width: usize) {
        if width == self.width {
            return;
        }
        self.width = width;
        for block in &mut self.blocks {
            block.render(&self.theme, width);
        }
        if let Some(live) = &mut self.live {
            live.render(&self.theme, width);
        }
    }

    pub fn push(&mut self, mut block: Block) {
        let before = self.total_lines();
        block.render(&self.theme, self.width);
        self.blocks.push(block);
        self.hold_position(before);
    }

    /// Convenience for the many one-line notes ozgent prints.
    pub fn note(&mut self, text: impl Into<String>) {
        self.push(Block::plain(text));
    }

    pub fn blank(&mut self) {
        self.push(Block::fixed(vec![String::new()]));
    }

    /// Replace the block being streamed into.
    pub fn set_live(&mut self, mut block: Block) {
        let before = self.total_lines();
        block.render(&self.theme, self.width);
        self.live = Some(block);
        self.hold_position(before);
    }

    /// Turn the live block into a permanent one.
    pub fn commit(&mut self) {
        if let Some(live) = self.live.take() {
            self.blocks.push(live);
        }
    }

    /// Every line, in order.
    fn lines(&self) -> impl Iterator<Item = &String> {
        self.blocks.iter().chain(self.live.iter()).flat_map(|b| b.lines.iter())
    }

    pub fn total_lines(&self) -> usize {
        self.lines().count()
    }

    /// The `height` lines that should be on screen.
    ///
    /// Padded at the top when there is not enough transcript to fill the
    /// window, so a new conversation starts at the bottom next to the prompt
    /// rather than floating at the top of an empty screen.
    pub fn visible(&self, height: usize) -> Vec<&str> {
        let all: Vec<&String> = self.lines().collect();
        let end = all.len().saturating_sub(self.scroll);
        let start = end.saturating_sub(height);
        all[start..end].iter().map(|s| s.as_str()).collect()
    }

    /// Scroll back by `lines`, stopping at the top.
    pub fn scroll_up(&mut self, lines: usize, height: usize) {
        let max = self.total_lines().saturating_sub(height);
        self.scroll = (self.scroll + lines).min(max);
    }

    pub fn scroll_down(&mut self, lines: usize) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub fn scroll_to_tail(&mut self) {
        self.scroll = 0;
    }

    #[allow(dead_code)] // asserted by the scrolling tests
    pub fn is_at_tail(&self) -> bool {
        self.scroll == 0
    }

    /// Lines below the window, hidden by scrolling back.
    ///
    /// Worth showing: while scrolled back the view is deliberately held still,
    /// so a reply arriving underneath produces no visible change at all. With
    /// nothing saying why, a working session looks like a frozen one.
    pub fn hidden_below(&self) -> usize {
        self.scroll
    }

    /// Keep the same lines in view when content is added below them.
    ///
    /// `scroll` counts lines back from the *end*, so appending moves the
    /// window forward on its own: reading back through a long answer while
    /// the model keeps writing would drag the text out from under you, one
    /// line per token. Growing `scroll` by however much was added holds the
    /// view still. At the tail there is nothing to hold — that is the
    /// following case, and it stays following.
    ///
    /// The delta can be negative: a streamed reply is re-rendered whole on
    /// every token, and a closing fence can make it shorter than it was.
    fn hold_position(&mut self, before: usize) {
        if self.scroll == 0 {
            return;
        }
        let after = self.total_lines();
        self.scroll = (self.scroll + after).saturating_sub(before);
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
        self.live = None;
        self.scroll = 0;
    }
}

fn render_markdown(theme: &Theme, width: usize, text: &str) -> Vec<String> {
    MarkdownRenderer::new(theme.clone(), width)
        .render(text)
        .lines()
        .map(str::to_string)
        .collect()
}

/// Break already-styled text into lines that fit `width`.
///
/// Splits on spaces and never in the middle of an escape sequence, which is
/// why this cannot simply be `str::chars().chunks()`: cutting a colour code in
/// half spills the colour over the rest of the screen.
fn wrap_styled(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for paragraph in text.split('\n') {
        if display_width(paragraph) <= width {
            out.push(paragraph.to_string());
            continue;
        }
        let mut line = String::new();
        let mut used = 0;
        for word in paragraph.split_inclusive(' ') {
            let w = display_width(word);
            if used + w > width && used > 0 {
                out.push(std::mem::take(&mut line));
                used = 0;
            }
            // A single word longer than the line is hard-split, one display
            // cell at a time, so a URL cannot push the border off screen.
            if w > width {
                for c in word.chars() {
                    let cw = display_width(&c.to_string());
                    if used + cw > width {
                        out.push(std::mem::take(&mut line));
                        used = 0;
                    }
                    line.push(c);
                    used += cw;
                }
                continue;
            }
            line.push_str(word);
            used += w;
        }
        out.push(line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript() -> Transcript {
        Transcript::new(Theme::plain(), 40)
    }

    #[test]
    fn a_note_becomes_one_line() {
        let mut t = transcript();
        t.note("hello");
        assert_eq!(t.visible(10), vec!["hello"]);
    }

    #[test]
    fn the_window_shows_the_last_lines_by_default() {
        let mut t = transcript();
        for i in 0..20 {
            t.note(format!("line {i}"));
        }
        assert_eq!(t.visible(3), vec!["line 17", "line 18", "line 19"]);
    }

    #[test]
    fn scrolling_back_moves_the_window_and_stops_at_the_top() {
        let mut t = transcript();
        for i in 0..10 {
            t.note(format!("line {i}"));
        }
        t.scroll_up(3, 4);
        assert_eq!(t.visible(4), vec!["line 3", "line 4", "line 5", "line 6"]);

        // Far past the top: the first line stays the first line.
        t.scroll_up(1000, 4);
        assert_eq!(t.visible(4), vec!["line 0", "line 1", "line 2", "line 3"]);
    }

    #[test]
    fn scrolling_down_returns_to_the_tail() {
        let mut t = transcript();
        for i in 0..10 {
            t.note(format!("line {i}"));
        }
        t.scroll_up(5, 3);
        assert!(!t.is_at_tail());
        t.scroll_down(100);
        assert!(t.is_at_tail());
        assert_eq!(t.visible(2), vec!["line 8", "line 9"]);
    }

    #[test]
    fn a_short_transcript_is_not_padded_into_nonsense() {
        let mut t = transcript();
        t.note("only");
        assert_eq!(t.visible(10), vec!["only"], "asking for ten lines must not invent nine");
    }

    #[test]
    fn the_live_block_is_replaced_not_appended() {
        // Markdown cannot be appended line by line: a closing fence changes
        // how everything since the opening one is drawn.
        let mut t = transcript();
        t.set_live(Block::plain("wri"));
        t.set_live(Block::plain("writing"));
        assert_eq!(t.visible(10), vec!["writing"]);
    }

    #[test]
    fn committing_keeps_the_live_block_and_clears_the_slot() {
        let mut t = transcript();
        t.set_live(Block::plain("done"));
        t.commit();
        t.set_live(Block::plain("next"));
        assert_eq!(t.visible(10), vec!["done", "next"]);
    }

    #[test]
    fn a_resize_reflows_from_the_source() {
        // Lines wrapped to forty columns are wrong at eighty, and there is no
        // way back to the text once wrapping has happened.
        let mut t = Transcript::new(Theme::plain(), 20);
        t.note("one two three four five six seven eight");
        let narrow = t.visible(10).len();
        t.resize(80);
        let wide = t.visible(10).len();
        assert!(narrow > wide, "{narrow} lines at 20 columns, {wide} at 80");
    }

    #[test]
    fn fixed_lines_survive_a_resize_unchanged() {
        // A box draws its own borders; reflowing it would produce rubble.
        let mut t = Transcript::new(Theme::plain(), 20);
        t.push(Block::fixed(vec!["╭────╮".into(), "╰────╯".into()]));
        t.resize(100);
        assert_eq!(t.visible(10), vec!["╭────╮", "╰────╯"]);
    }

    #[test]
    fn wrapping_never_cuts_an_escape_sequence() {
        // A colour code cut in half spills over the rest of the screen.
        let styled = format!("\x1b[31m{}\x1b[0m", "word ".repeat(20));
        for line in wrap_styled(&styled, 20) {
            let opens = line.matches('\x1b').count();
            let terminated = line.matches('m').count();
            assert!(opens <= terminated, "a broken escape in {line:?}");
        }
    }

    #[test]
    fn a_word_longer_than_the_line_is_split_rather_than_overflowing() {
        let lines = wrap_styled(&"z".repeat(60), 20);
        assert!(lines.iter().all(|l| display_width(l) <= 20), "{lines:?}");
        assert_eq!(lines.concat(), "z".repeat(60), "no characters may be lost");
    }

    #[test]
    fn a_reply_shows_its_reasoning_above_its_answer() {
        let mut t = transcript();
        t.set_live(Block::reply(Some("weighing it up".into()), "The answer."));
        let out = t.visible(10).join("\n");
        let think = out.find("weighing").expect("reasoning is shown");
        let answer = out.find("The answer").expect("the answer is shown");
        assert!(think < answer, "reasoning belongs above the reply it led to");
    }

    #[test]
    fn an_empty_reasoning_block_takes_no_room() {
        // A model that decides not to reason still emits <think></think>.
        let mut t = transcript();
        t.set_live(Block::reply(Some("   ".into()), "Just the answer."));
        let out = t.visible(10);
        assert_eq!(out.len(), 1, "{out:?}");
    }

    #[test]
    fn reasoning_is_not_parsed_as_markdown() {
        // A stray `#` in a model's scratch notes must not become a heading
        // heavier than the reply underneath it.
        let mut t = transcript();
        t.set_live(Block::reply(Some("# not a heading".into()), ""));
        assert!(t.visible(10).join("").contains("# not a heading"));
    }

    #[test]
    fn a_reply_reflows_on_resize_like_anything_else() {
        let mut t = Transcript::new(Theme::plain(), 20);
        t.set_live(Block::reply(None, "one two three four five six seven eight nine"));
        let narrow = t.visible(20).len();
        t.resize(80);
        assert!(t.visible(20).len() < narrow);
    }

    #[test]
    fn new_content_does_not_drag_the_view_out_from_under_a_reader() {
        // The bug this exists for: `scroll` counts back from the end, so
        // appending moves the window forward on its own. Reading back through
        // a long answer while the model wrote lost a line per token.
        let mut t = transcript();
        for i in 0..20 {
            t.note(format!("line {i}"));
        }
        t.scroll_up(8, 4);
        let held: Vec<String> = t.visible(4).iter().map(|s| s.to_string()).collect();

        for i in 20..25 {
            t.note(format!("line {i}"));
        }
        assert_eq!(t.visible(4), held, "the same lines must still be on screen");
    }

    #[test]
    fn a_streamed_reply_getting_shorter_does_not_break_the_hold() {
        // A reply is re-rendered whole on each token, and a closing fence can
        // make it shorter than it was; the delta is negative there.
        let mut t = transcript();
        for i in 0..20 {
            t.note(format!("line {i}"));
        }
        t.scroll_up(6, 4);
        let held: Vec<String> = t.visible(4).iter().map(|s| s.to_string()).collect();

        t.set_live(Block::plain("a\nb\nc\nd"));
        t.set_live(Block::plain("a"));
        assert_eq!(t.visible(4), held);
    }

    #[test]
    fn at_the_tail_new_content_is_followed() {
        // The other half: someone who has not scrolled wants to see the reply
        // as it arrives.
        let mut t = transcript();
        t.note("first");
        assert!(t.is_at_tail());
        for i in 0..10 {
            t.note(format!("line {i}"));
        }
        assert_eq!(t.visible(1), vec!["line 9"]);
    }

    #[test]
    fn clearing_empties_everything_including_the_scroll() {
        let mut t = transcript();
        for i in 0..30 {
            t.note(format!("{i}"));
        }
        t.scroll_up(10, 5);
        t.clear();
        assert!(t.is_empty());
        assert!(t.is_at_tail());
        assert!(t.visible(5).is_empty());
    }
}
