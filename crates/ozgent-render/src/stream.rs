//! Incremental rendering of a model's markdown as it streams.
//!
//! Re-rendering the whole response on every token flickers and destroys
//! scrollback, so output is split in two:
//!
//! * **Committed** text is finished — a closed paragraph, a completed code
//!   line — and is written once, permanently. It scrolls away like any other
//!   terminal output and is never touched again.
//! * The **volatile** tail is the block still being written. It is erased and
//!   redrawn in place as tokens arrive.
//!
//! Code fences commit line by line rather than waiting for the closing fence,
//! because models emit long code blocks and redrawing a 200-line block per
//! token would be unusable.

use crate::markdown::MarkdownRenderer;
use crate::theme::Style;
use std::io::{self, Write};
use unicode_width::UnicodeWidthStr;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Mode {
    Normal,
    /// Inside an open fence; `fence` is the exact marker that will close it.
    Code { fence: String },
}

pub struct StreamRenderer {
    r: MarkdownRenderer,
    mode: Mode,
    /// Source of the block being accumulated, in normal mode.
    block: String,
    /// The current line, up to the newline that has not arrived yet.
    line: String,
    /// Terminal lines the volatile region currently occupies.
    volatile: usize,
    /// Whether anything has been written, to suppress a leading blank line.
    started: bool,
}

impl StreamRenderer {
    pub fn new(r: MarkdownRenderer) -> Self {
        Self {
            r,
            mode: Mode::Normal,
            block: String::new(),
            line: String::new(),
            volatile: 0,
            started: false,
        }
    }

    /// Feed newly generated markdown and update the terminal.
    pub fn push<W: Write>(&mut self, text: &str, out: &mut W) -> io::Result<()> {
        self.erase(out)?;

        for ch in text.chars() {
            if ch == '\n' {
                let line = std::mem::take(&mut self.line);
                self.commit_line(&line, out)?;
            } else {
                self.line.push(ch);
            }
        }

        self.draw_volatile(out)?;
        out.flush()
    }

    /// Flush everything at end of stream and leave the cursor on a fresh line.
    pub fn finish<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        self.erase(out)?;

        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            self.commit_line(&line, out)?;
        }
        if matches!(self.mode, Mode::Code { .. }) {
            // An unterminated fence still ends the block.
            self.mode = Mode::Normal;
        }
        self.flush_block(out)?;
        self.volatile = 0;
        out.flush()
    }

    /// Handle one completed source line.
    fn commit_line<W: Write>(&mut self, line: &str, out: &mut W) -> io::Result<()> {
        match &self.mode {
            Mode::Code { fence } => {
                if is_closing_fence(line, fence) {
                    self.mode = Mode::Normal;
                    self.write_line("", out)?;
                } else {
                    let styled = self.r.theme().style(self.r.theme().code_block, line);
                    self.write_line(&format!("  {styled}"), out)?;
                }
            }
            Mode::Normal => {
                if let Some(fence) = opening_fence(line) {
                    // The prose before the fence is complete now.
                    self.flush_block(out)?;
                    let lang = line.trim_start().trim_start_matches(&fence).trim();
                    if !lang.is_empty() {
                        let label = self.r.theme().style(Style::dim(), &format!("  {lang}"));
                        self.write_line(&label, out)?;
                    }
                    self.mode = Mode::Code { fence };
                } else if line.trim().is_empty() {
                    self.flush_block(out)?;
                } else {
                    self.block.push_str(line);
                    self.block.push('\n');
                }
            }
        }
        Ok(())
    }

    /// Render and permanently emit the accumulated non-code block.
    fn flush_block<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        if self.block.trim().is_empty() {
            self.block.clear();
            return Ok(());
        }
        let source = std::mem::take(&mut self.block);
        let rendered = self.r.render(&source);
        for line in rendered.lines() {
            self.write_line(line, out)?;
        }
        self.write_line("", out)
    }

    /// Draw the not-yet-final tail, remembering its height so it can be erased.
    fn draw_volatile<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        let rendered = match &self.mode {
            Mode::Code { .. } => {
                if self.line.is_empty() {
                    String::new()
                } else {
                    let styled = self.r.theme().style(self.r.theme().code_block, &self.line);
                    format!("  {styled}")
                }
            }
            Mode::Normal => {
                let mut source = self.block.clone();
                source.push_str(&self.line);
                if source.trim().is_empty() {
                    String::new()
                } else {
                    self.r.render(&source)
                }
            }
        };

        // The trailing newline has to go: it would leave the cursor on the
        // line *below* the volatile region, and the erase sequence works
        // upward from wherever the cursor is.
        let rendered = rendered.trim_end_matches('\n');
        if rendered.is_empty() {
            self.volatile = 0;
            return Ok(());
        }

        out.write_all(rendered.as_bytes())?;
        self.started = true;
        self.volatile = rendered
            .lines()
            .map(|l| display_lines(l, self.r.width()))
            .sum::<usize>()
            .max(1);
        Ok(())
    }

    /// Erase the previously drawn volatile region.
    fn erase<W: Write>(&mut self, out: &mut W) -> io::Result<()> {
        if self.volatile == 0 {
            return Ok(());
        }
        // The cursor sits at the end of the last volatile line, so return to
        // column 0, step up over the rest, then clear to the end of screen.
        write!(out, "\r")?;
        if self.volatile > 1 {
            write!(out, "\x1b[{}A", self.volatile - 1)?;
        }
        write!(out, "\x1b[0J")?;
        self.volatile = 0;
        Ok(())
    }

    /// Write one permanent line.
    fn write_line<W: Write>(&mut self, line: &str, out: &mut W) -> io::Result<()> {
        if line.is_empty() && !self.started {
            return Ok(()); // no leading blank line
        }
        out.write_all(line.as_bytes())?;
        out.write_all(b"\n")?;
        self.started = true;
        Ok(())
    }
}

/// The fence marker if this line opens a code block.
///
/// A fence is three or more backticks or tildes; the exact run is returned so
/// the closing fence can be required to be at least as long.
fn opening_fence(line: &str) -> Option<String> {
    let t = line.trim_start();
    let ch = t.chars().next()?;
    if ch != '`' && ch != '~' {
        return None;
    }
    let len = t.chars().take_while(|c| *c == ch).count();
    (len >= 3).then(|| std::iter::repeat_n(ch, len).collect())
}

/// A fence closes on a line of at least as many of the same character.
fn is_closing_fence(line: &str, fence: &str) -> bool {
    let t = line.trim();
    let ch = fence.chars().next().unwrap_or('`');
    t.len() >= fence.len() && t.chars().all(|c| c == ch) && !t.is_empty()
}

/// How many terminal rows a rendered line occupies once the terminal wraps it.
fn display_lines(line: &str, width: usize) -> usize {
    let w = strip_ansi_width(line);
    if w == 0 || width == 0 {
        return 1;
    }
    w.div_ceil(width).max(1)
}

/// Display width ignoring SGR escape sequences, which occupy no columns.
fn strip_ansi_width(s: &str) -> usize {
    let mut total = 0;
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            total += ch.to_string().width();
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Theme;

    /// Feed text in fixed-size pieces and return everything written.
    fn stream(chunks: &[&str], width: usize) -> String {
        let mut s = StreamRenderer::new(MarkdownRenderer::plain(width));
        let mut buf: Vec<u8> = Vec::new();
        for c in chunks {
            s.push(c, &mut buf).unwrap();
        }
        s.finish(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    /// What the user is left looking at: apply the erase sequences.
    fn final_screen(raw: &str) -> String {
        let mut lines: Vec<String> = vec![String::new()];
        let mut chars = raw.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                '\n' => lines.push(String::new()),
                '\r' => {
                    lines.last_mut().unwrap().clear();
                }
                '\x1b' => {
                    let mut seq = String::new();
                    for c in chars.by_ref() {
                        if c.is_ascii_alphabetic() {
                            seq.push(c);
                            break;
                        }
                        seq.push(c);
                    }
                    if let Some(n) = seq.strip_suffix('A') {
                        let up: usize = n.trim_start_matches('[').parse().unwrap_or(1);
                        for _ in 0..up {
                            if lines.len() > 1 {
                                lines.pop();
                            }
                        }
                        lines.last_mut().unwrap().clear();
                    } else if seq.ends_with('J') {
                        // Clear from cursor to end of screen.
                    }
                }
                c => lines.last_mut().unwrap().push(c),
            }
        }
        lines.join("\n").trim_end().to_string()
    }

    /// Split text into `n`-character pieces, simulating tokenisation.
    fn chunked(text: &str, n: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        chars.chunks(n).map(|c| c.iter().collect()).collect()
    }

    fn stream_str(text: &str, chunk: usize, width: usize) -> String {
        let pieces = chunked(text, chunk);
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        final_screen(&stream(&refs, width))
    }

    #[test]
    fn a_simple_paragraph_renders_once_complete() {
        let screen = stream_str("Hello there, world.", 3, 60);
        assert_eq!(screen, "Hello there, world.");
    }

    #[test]
    fn chunking_does_not_change_the_result() {
        let md = "# Title\n\nSome **bold** text and a list:\n\n- one\n- two\n\nDone.";
        let reference = stream_str(md, 1000, 60);
        for size in [1, 2, 3, 5, 13] {
            assert_eq!(
                stream_str(md, size, 60),
                reference,
                "output differs when streamed {size} chars at a time"
            );
        }
    }

    #[test]
    fn markdown_syntax_never_survives_to_the_screen() {
        let screen = stream_str("Use **bold** and *italic* and `code` here.", 2, 60);
        assert!(!screen.contains("**"), "{screen:?}");
        assert!(!screen.contains('`'), "{screen:?}");
        assert_eq!(screen, "Use bold and italic and code here.");
    }

    #[test]
    fn code_fences_render_as_code_and_lose_their_markers() {
        let screen = stream_str("```rust\nfn main() {}\n```\n", 4, 60);
        assert!(screen.contains("fn main() {}"), "{screen:?}");
        assert!(!screen.contains("```"), "fence markers leaked: {screen:?}");
        assert!(screen.contains("rust"), "language label expected: {screen:?}");
    }

    #[test]
    fn code_lines_commit_immediately_rather_than_waiting_for_the_fence() {
        // A long code block must not be redrawn in full on every token, so
        // completed lines are written permanently as they arrive.
        let mut s = StreamRenderer::new(MarkdownRenderer::plain(60));
        let mut buf: Vec<u8> = Vec::new();
        s.push("```python\nline_one()\nline_two()\n", &mut buf).unwrap();

        let before_close = String::from_utf8(buf.clone()).unwrap();
        assert!(before_close.contains("line_one()"), "{before_close:?}");
        assert!(before_close.contains("line_two()"), "{before_close:?}");
        // Committed content is never erased, so no cursor-up appears between
        // the two code lines.
        let between = &before_close[before_close.find("line_one()").unwrap()
            ..before_close.find("line_two()").unwrap()];
        assert!(!between.contains("\x1b["), "code lines were redrawn: {between:?}");
    }

    #[test]
    fn text_after_a_code_block_still_renders() {
        let screen = stream_str("Intro:\n\n```\nx = 1\n```\n\nOutro text.", 3, 60);
        assert!(screen.contains("Intro:"), "{screen:?}");
        assert!(screen.contains("x = 1"), "{screen:?}");
        assert!(screen.contains("Outro text."), "{screen:?}");
    }

    #[test]
    fn an_unterminated_fence_is_still_flushed_at_the_end() {
        let screen = stream_str("```\nincomplete code", 3, 60);
        assert!(screen.contains("incomplete code"), "content must not be lost: {screen:?}");
    }

    #[test]
    fn lists_render_with_markers() {
        let screen = stream_str("- alpha\n- beta\n", 2, 60);
        assert!(screen.contains("• alpha"), "{screen:?}");
        assert!(screen.contains("• beta"), "{screen:?}");
    }

    #[test]
    fn partial_output_is_visible_before_the_block_ends() {
        // The whole point of streaming: text appears without waiting for a
        // blank line to close the paragraph.
        let mut s = StreamRenderer::new(MarkdownRenderer::plain(60));
        let mut buf: Vec<u8> = Vec::new();
        s.push("Partial senten", &mut buf).unwrap();
        let screen = final_screen(&String::from_utf8(buf).unwrap());
        assert!(screen.contains("Partial senten"), "nothing shown yet: {screen:?}");
    }

    #[test]
    fn the_volatile_region_is_erased_before_being_redrawn() {
        let mut s = StreamRenderer::new(MarkdownRenderer::plain(60));
        let mut buf: Vec<u8> = Vec::new();
        s.push("one", &mut buf).unwrap();
        let after_first = buf.len();
        s.push(" two", &mut buf).unwrap();

        let second = String::from_utf8(buf[after_first..].to_vec()).unwrap();
        assert!(second.contains("\x1b[0J"), "expected an erase: {second:?}");
        // And the screen must show the joined text exactly once.
        let screen = final_screen(&String::from_utf8(buf).unwrap());
        assert_eq!(screen, "one two");
    }

    #[test]
    fn nothing_is_duplicated_across_many_updates() {
        let screen = stream_str("The quick brown fox jumps over the lazy dog.", 1, 60);
        assert_eq!(screen.matches("quick").count(), 1, "duplicated text: {screen:?}");
        assert_eq!(screen, "The quick brown fox jumps over the lazy dog.");
    }

    #[test]
    fn wrapped_paragraphs_are_erased_correctly() {
        // A multi-line volatile region needs the right cursor-up count, or
        // earlier lines are left behind as duplicates.
        let text = "word ".repeat(30);
        let screen = stream_str(&text, 7, 30);
        assert_eq!(screen.matches("word").count(), 30, "lines duplicated: {screen:?}");
        for line in screen.lines() {
            assert!(line.width() <= 30, "overflow: {line:?}");
        }
    }

    #[test]
    fn styled_output_measures_width_without_counting_escapes() {
        // If escapes counted toward width the erase height would be wrong.
        let mut s = StreamRenderer::new(MarkdownRenderer::new(Theme::default(), 40));
        let mut buf: Vec<u8> = Vec::new();
        s.push("**bold text here**", &mut buf).unwrap();
        s.push(" more", &mut buf).unwrap();
        s.finish(&mut buf).unwrap();

        let raw = String::from_utf8(buf).unwrap();
        assert!(raw.contains("\x1b["), "should be styled");
        let screen = final_screen(&raw);
        assert_eq!(screen.matches("bold text here").count(), 1, "{screen:?}");
    }

    #[test]
    fn ansi_aware_width_ignores_escape_sequences() {
        assert_eq!(strip_ansi_width("\x1b[1mbold\x1b[0m"), 4);
        assert_eq!(strip_ansi_width("plain"), 5);
        assert_eq!(strip_ansi_width("日本"), 4, "CJK counts two columns each");
    }

    #[test]
    fn empty_stream_produces_nothing() {
        assert_eq!(stream(&[], 60), "");
        assert_eq!(stream(&[""], 60), "");
    }

    #[test]
    fn headings_and_paragraphs_keep_their_order() {
        let screen = stream_str("# One\n\nbody one\n\n## Two\n\nbody two", 3, 60);
        let pos = |s: &str| screen.find(s).unwrap_or_else(|| panic!("missing {s:?} in {screen:?}"));
        assert!(pos("One") < pos("body one"));
        assert!(pos("body one") < pos("Two"));
        assert!(pos("Two") < pos("body two"));
    }
}
