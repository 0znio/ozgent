//! A status line pinned to the bottom of the terminal.
//!
//! The chat is a scrolling transcript, not a full-screen application, and that
//! is deliberate: native scrollback, selection and copy all keep working. But
//! a scrolling transcript has nowhere to put the facts that are true *now* —
//! how full the context is, how fast the last reply came out, which model is
//! answering — and printing them after every turn would bury the conversation
//! in chrome.
//!
//! vim solves this with a line the text never scrolls over, and so does this:
//! the terminal's scrolling region is shrunk by one row (`DECSTBM`) and the
//! freed row is drawn on directly. Everything else in the program keeps
//! printing exactly as before and simply cannot reach that row.
//!
//! Two details are load-bearing:
//!
//! - **`DECSTBM` homes the cursor**, so every write here is wrapped in
//!   save/restore. Without that the next line of a streaming reply would land
//!   in the top-left corner.
//! - **A resize resets the region** in most terminals, so the size is
//!   re-read and the region re-installed on every redraw rather than once at
//!   start-up. That is also what makes the line follow the window when it
//!   moves.

use std::io::{IsTerminal, Write};

use ozgent_render::{Style, Theme};

/// One labelled fact, and how badly it wants to stay when space runs out.
///
/// Ordering matters: a status line that drops the model name to keep the
/// sampler settings has kept the wrong thing.
pub struct Segment {
    pub text: String,
    /// Lower is more important. Segment 0 is never dropped.
    pub priority: u8,
}

impl Segment {
    pub fn new(priority: u8, text: impl Into<String>) -> Self {
        Self { text: text.into(), priority }
    }
}

/// Fit segments into `width` columns, dropping the least important first.
///
/// Returns the plain text of the line, padded to exactly `width` so the
/// highlight runs edge to edge the way vim's does. Kept separate from drawing
/// so the layout can be tested without a terminal.
pub fn compose(segments: &[Segment], width: usize) -> String {
    const SEP: &str = "  ·  ";

    let mut keep: Vec<&Segment> = segments.iter().collect();
    loop {
        let line = join(&keep, SEP);
        // The line is padded with a leading and trailing space, so it needs
        // two columns beyond its own content.
        if line.chars().count() + 2 <= width || keep.len() <= 1 {
            let mut out = format!(" {line} ");
            let len = out.chars().count();
            if len < width {
                out.push_str(&" ".repeat(width - len));
            } else if len > width {
                // Only reachable once a single segment is left and even that
                // does not fit. Truncating beats wrapping, which would scroll
                // the transcript by a line on every redraw.
                out = out.chars().take(width).collect();
            }
            return out;
        }
        // Drop one of the least important remaining segments.
        let worst = keep.iter().enumerate().max_by_key(|(i, s)| (s.priority, *i));
        let Some((index, _)) = worst else { return String::new() };
        keep.remove(index);
    }
}

fn join(segments: &[&Segment], sep: &str) -> String {
    segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(sep)
}

/// The bottom row, and the shrunken scrolling region that protects it.
pub struct StatusLine {
    /// False when stderr is not a terminal, which disables every write here.
    active: bool,
    /// Last size the region was installed for, so a resize is noticed.
    size: (usize, usize),
    /// What to draw. Kept so a resize or a redraw after `/clear` can repaint
    /// without the caller having to remember the last state.
    line: String,
    styled: bool,
}

impl StatusLine {
    /// Claim the bottom row, or do nothing at all when there is no terminal.
    ///
    /// A pipe, a test harness and a CI log all end up here, and every one of
    /// them wants the escape sequences absent rather than merely invisible.
    pub fn install(theme: &Theme) -> Self {
        let active = std::io::stderr().is_terminal();
        let mut status = Self {
            active,
            size: (0, 0),
            line: String::new(),
            styled: theme.enabled,
        };
        if active {
            // Scroll the transcript up by one so the row being claimed is
            // blank, rather than covering whatever was printed there.
            eprint!("\n\x1b[1A");
            status.reserve();
        }
        status
    }

    /// Set the region and remember the size it was set for.
    fn reserve(&mut self) {
        let (cols, rows) = ozgent_render::terminal_size();
        self.size = (cols, rows);
        if rows > 1 {
            // Save and restore around it: DECSTBM parks the cursor at the
            // top-left corner, which would otherwise overwrite the transcript.
            eprint!("\x1b7\x1b[1;{}r\x1b8", rows - 1);
        }
    }

    /// Replace what the line says and repaint it.
    pub fn set(&mut self, segments: &[Segment]) {
        if !self.active {
            return;
        }
        let (cols, _) = ozgent_render::terminal_size();
        self.line = compose(segments, cols);
        self.draw();
    }

    /// Repaint the current text, re-installing the region if the window moved.
    ///
    /// Called before every prompt because a terminal resize resets the region
    /// silently, and because rustyline's own redraws can scroll the line away.
    pub fn refresh(&mut self) {
        if !self.active {
            return;
        }
        if ozgent_render::terminal_size() != self.size {
            self.reserve();
            // Width changed, so the padding is wrong; but the segments are
            // gone. Re-pad what is there rather than showing a short bar.
            let text: String = self.line.trim_end().to_string();
            self.line = compose(&[Segment::new(0, text.trim())], self.size.0);
        }
        self.draw();
    }

    fn draw(&self) {
        if !self.active || self.line.is_empty() {
            return;
        }
        let (_, rows) = self.size;
        // Reverse video is how vim marks its status line, and it inverts
        // whatever palette the terminal already uses rather than imposing a
        // colour that may be unreadable on the user's background.
        let (on, off) =
            if self.styled { ("\x1b[7m", Style::RESET) } else { ("", "") };
        eprint!("\x1b7\x1b[{rows};1H\x1b[2K{on}{}{off}\x1b8", self.line);
        let _ = std::io::stderr().flush();
    }

    /// Give the row back and restore full-screen scrolling.
    ///
    /// Leaving the region in place would hand the user's shell a terminal that
    /// refuses to use its last line, which looks like ozgent broke it.
    pub fn remove(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let (_, rows) = self.size;
        eprint!("\x1b7\x1b[{rows};1H\x1b[2K\x1b8\x1b[r");
        let _ = std::io::stderr().flush();
    }
}

impl Drop for StatusLine {
    fn drop(&mut self) {
        self.remove();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs() -> Vec<Segment> {
        vec![
            Segment::new(0, "qwen3.5:4b"),
            Segment::new(1, "ctx 3.2k/32k"),
            Segment::new(2, "48.6 tok/s"),
            Segment::new(3, "temp 0.7"),
        ]
    }

    #[test]
    fn a_wide_line_shows_everything_and_fills_the_width() {
        let line = compose(&segs(), 100);
        assert_eq!(line.chars().count(), 100, "the highlight must reach both edges");
        assert!(line.contains("qwen3.5:4b"));
        assert!(line.contains("temp 0.7"));
    }

    #[test]
    fn a_narrow_line_drops_the_least_important_first() {
        // 40 columns is a phone-sized terminal, and the model name is the one
        // fact that must survive it.
        let line = compose(&segs(), 40);
        assert_eq!(line.chars().count(), 40);
        assert!(line.contains("qwen3.5:4b"), "got {line:?}");
        assert!(!line.contains("temp 0.7"), "the sampler is the first to go: {line:?}");
    }

    #[test]
    fn dropping_stops_before_the_line_is_empty() {
        let line = compose(&segs(), 12);
        assert_eq!(line.chars().count(), 12);
        assert!(line.trim().starts_with("qwen"), "got {line:?}");
    }

    #[test]
    fn a_single_oversized_segment_is_truncated_not_wrapped() {
        // Wrapping would scroll the transcript by one line on every repaint.
        let long = Segment::new(0, "x".repeat(200));
        assert_eq!(compose(&[long], 30).chars().count(), 30);
    }

    #[test]
    fn segments_are_dropped_by_priority_not_by_position() {
        let out = compose(
            &[
                Segment::new(0, "keep"),
                Segment::new(9, "drop-me"),
                Segment::new(1, "also-keep"),
            ],
            22,
        );
        assert!(out.contains("keep") && out.contains("also-keep"), "got {out:?}");
        assert!(!out.contains("drop-me"), "got {out:?}");
    }

    #[test]
    fn without_a_terminal_nothing_is_written() {
        // The test harness has no tty, which is exactly the case being
        // asserted: an inactive line must not emit escape sequences.
        let mut s = StatusLine::install(&Theme::plain());
        assert!(!s.active);
        s.set(&segs());
        assert!(s.line.is_empty());
    }
}
