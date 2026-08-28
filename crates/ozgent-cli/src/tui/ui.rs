//! The application: what is on the screen, and what the keys do to it.
//!
//! Everything the chat used to print goes through here. In full-screen mode it
//! becomes a block in the transcript and the frame is redrawn; with no
//! terminal — a pipe, a test harness, `ozgent run` in a script — it is printed
//! as before. Keeping both is not politeness: the piped path is how ozgent is
//! scripted, and it must not grow a dependency on a terminal it does not have.

use std::time::{Duration, Instant};

use ozgent_core::permission::{Choice, Effect};
use ozgent_render::{Color, Style, Theme, display_width};

use super::frame::{self, Layout, MAX_PROMPT_ROWS};
use super::{Block, Editor, Key, Screen, Transcript};
use crate::input::{Input, Prompt};
use crate::status::Segment;

/// How often a frame is repainted while a reply streams.
///
/// Re-rendering markdown is linear in the answer so far, and a token arrives
/// every twenty milliseconds on a fast model; repainting on each one spends
/// more time drawing than generating. At this rate the text still appears to
/// flow and the cost stays flat.
const FRAME: Duration = Duration::from_millis(50);

/// Lines moved per notch of the wheel, matching a terminal's own scrollback.
const WHEEL: usize = 3;

/// Braille dots, because they turn without the line changing width — a
/// spinner made of `|/-\` shifts everything after it by a column each frame.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// How often the spinner advances. Slow enough not to strobe, fast enough to
/// read as motion rather than as something stuck.
pub const SPIN: Duration = Duration::from_millis(90);

/// What the user did at the prompt.
pub enum Submission {
    Line(String),
    /// Ctrl-C: abandon this line, keep the session.
    Cancelled,
    /// Ctrl-D, or the end of piped input.
    Eof,
}

/// The screen, or the absence of one.
pub struct Ui {
    screen: Option<Screen>,
    transcript: Transcript,
    editor: Editor,
    theme: Theme,
    /// Drawn on the status bar. Owned by the caller's `status_segments`.
    status: Vec<Segment>,
    /// What the permission bar says when nothing is being asked.
    posture: String,
    /// The question on the permission bar, while there is one.
    question: Option<Question>,
    /// The line whose marker is currently spinning, without its marker.
    activity: Option<String>,
    spin: usize,
    size: (usize, usize),
    last_frame: Instant,
    /// Kept even in full-screen mode, where its editor is never used: it owns
    /// the history file, and a session that took over the terminal must still
    /// leave its history where the next one — piped or not — will find it.
    prompt: Prompt,
    /// Whether `prompt` is also doing the line editing, which it is only when
    /// there is no screen.
    piped: bool,
}

/// A permission question, as the bar shows it.
struct Question {
    tool: String,
    effect: Effect,
    summary: String,
}

impl Ui {
    /// Take over the terminal, or fall back to printing.
    pub fn new(theme: Theme, history: Option<std::path::PathBuf>) -> Self {
        let screen = Screen::open().ok();
        let size = ozgent_render::terminal_size();
        // History is loaded from the same file either way, so switching
        // between a terminal and a pipe does not split it in two.
        let prompt = Prompt::new(history);
        let past = prompt.history_lines();

        Self {
            transcript: Transcript::new(theme.clone(), size.0),
            editor: Editor::new(past),
            theme,
            status: Vec::new(),
            posture: String::new(),
            question: None,
            activity: None,
            spin: 0,
            size,
            last_frame: Instant::now() - FRAME,
            piped: screen.is_none(),
            prompt,
            screen,
        }
    }

    /// Write the history file.
    ///
    /// The full-screen editor keeps its own list, so it is copied across
    /// before saving; otherwise a session that took over the terminal would
    /// record nothing at all.
    pub fn save_history(&mut self) {
        if !self.piped {
            let typed: Vec<String> = self.editor.history().to_vec();
            for line in typed {
                self.prompt.remember(&line);
            }
        }
        self.prompt.save();
    }

    /// Announce something that is about to take time.
    ///
    /// The line goes in immediately with a settled marker; [`Ui::tick`] swaps
    /// in the next spinner frame while it runs, and [`Ui::settle`] puts the
    /// marker back. `text` carries no marker of its own — this owns that
    /// column, so the spinner and the dot cannot disagree about where it is.
    pub fn begin_activity(&mut self, text: impl Into<String>) {
        // Replacing rather than appending when one is already standing. A tool
        // call is announced by name the moment the model commits to it, and
        // again with its arguments once they are written; those are the same
        // event and must be the same line, or a write shows up twice.
        let replacing = self.activity.is_some();
        self.activity = Some(text.into());
        let line = self.activity_line(replacing);
        if self.screen.is_some() {
            if replacing {
                self.transcript.set_last(Block::plain(line));
            } else {
                self.spin = 0;
                self.transcript.push(Block::plain(line));
            }
            self.render();
        } else if !replacing {
            eprintln!("{line}");
        }
    }

    /// Advance the spinner. Does nothing when nothing is running.
    ///
    /// Self-throttling, so a caller driving it from a per-token callback and
    /// one driving it from a timer can both simply call it.
    pub fn tick(&mut self) {
        if self.activity.is_none() || self.screen.is_none() {
            return;
        }
        if self.last_frame.elapsed() < SPIN {
            return;
        }
        self.spin = self.spin.wrapping_add(1);
        let line = self.activity_line(true);
        self.transcript.set_last(Block::plain(line));
        self.render();
    }

    /// Stop the spinner, leaving the line as a plain record of what happened.
    pub fn settle(&mut self) {
        if self.activity.is_none() {
            return;
        }
        let line = self.activity_line(false);
        if self.screen.is_some() {
            self.transcript.set_last(Block::plain(line));
            self.render();
        }
        self.activity = None;
    }

    fn activity_line(&self, running: bool) -> String {
        let text = self.activity.as_deref().unwrap_or_default();
        let marker = if running {
            // Amber while it is happening, which is the one thing amber means.
            self.theme.style(
                Style::color(Color::Yellow),
                SPINNER[self.spin % SPINNER.len()],
            )
        } else {
            self.theme.style(Style::color(Color::Green), "●")
        };
        format!("{marker} {text}")
    }

    /// Whether the terminal was taken over.
    #[allow(dead_code)] // asserted by the fallback test
    pub fn full_screen(&self) -> bool {
        self.screen.is_some()
    }

    // ------------------------------------------------------------ output

    /// One line of ozgent's own chrome — a note, a heading, a tool result.
    ///
    /// Painted at once. Appending without painting means nothing appears until
    /// something else happens to redraw, and the something else is usually the
    /// reply that comes *after* the slow thing you were waiting on — so a tool
    /// call announced before it ran showed up only once it had finished.
    pub fn say(&mut self, text: impl Into<String>) {
        let text = text.into();
        if self.screen.is_some() {
            self.transcript.note(text);
            self.render();
        } else {
            eprintln!("{text}");
        }
    }

    pub fn blank(&mut self) {
        if self.screen.is_some() {
            self.transcript.blank();
            self.render();
        } else {
            eprintln!();
        }
    }

    /// Render markdown into the transcript.
    pub fn markdown(&mut self, text: impl Into<String>) {
        let text = text.into();
        if self.screen.is_some() {
            self.transcript.push(Block::markdown(text));
            self.render();
        } else {
            let width = ozgent_render::terminal_width();
            print!("{}", ozgent_render::MarkdownRenderer::new(self.theme.clone(), width).render(&text));
        }
    }

    /// Replace the reply being written, repainting no more often than [`FRAME`].
    ///
    /// `force` bypasses the throttle for the last token of a turn, which would
    /// otherwise sit unpainted until something else happened.
    pub fn stream(&mut self, thinking: Option<&str>, answer: &str, force: bool) {
        if self.screen.is_none() {
            return;
        }
        if !force && self.last_frame.elapsed() < FRAME {
            return;
        }
        self.transcript.set_live(Block::reply(thinking.map(str::to_string), answer.to_string()));
        self.render();
    }

    /// Keep the streamed reply as part of the conversation.
    pub fn commit(&mut self) {
        self.transcript.commit();
    }

    pub fn clear(&mut self) {
        if self.screen.is_some() {
            self.transcript.clear();
            self.render();
        } else {
            crate::chat::clear_screen();
        }
    }

    pub fn set_status(&mut self, segments: Vec<Segment>) {
        self.status = segments;
    }

    /// What the permission bar says when it is not asking anything.
    pub fn set_posture(&mut self, text: impl Into<String>) {
        self.posture = text.into();
    }

    // ------------------------------------------------------------- input

    /// Read one submission, drawing frames and handling scrolling until then.
    pub fn read(&mut self, marker: &str) -> Submission {
        if self.piped {
            return match self.prompt.read(marker) {
                Input::Line(l) => Submission::Line(l),
                Input::Cancelled => Submission::Cancelled,
                Input::Eof => Submission::Eof,
            };
        }

        // Painted once, then only when something changes. The obvious loop —
        // render, poll, repeat — repaints the whole screen twenty times a
        // second while the user sits thinking about what to type, which is
        // pure waste on a local terminal and visible lag over ssh.
        self.render();
        loop {
            let key = match self.screen.as_ref().and_then(|s| s.key(FRAME).ok().flatten()) {
                Some(k) => k,
                // Nothing arrived, so nothing on screen can have changed.
                None => continue,
            };
            match key {
                Key::Enter => {
                    if self.editor.is_empty() {
                        continue;
                    }
                    return Submission::Line(self.editor.take());
                }
                // Ctrl-C on a line abandons the line; on an empty one it is
                // the same "nothing happened" the scrolling REPL gave.
                Key::Interrupt => {
                    self.editor.clear();
                    return Submission::Cancelled;
                }
                Key::Eof if self.editor.is_empty() => return Submission::Eof,
                other => {
                    self.edit(other);
                    self.render();
                }
            }
        }
    }

    /// Ask a one-off question and read the answer in the prompt box.
    ///
    /// Used where the chat needs a value that is not a message — an API key,
    /// a filename. In full-screen mode `read_line` on stdin cannot be used at
    /// all: the terminal is in raw mode, so there are no lines to read.
    pub fn ask_text(&mut self, label: &str) -> Option<String> {
        if self.piped {
            eprint!("{label}");
            let _ = std::io::Write::flush(&mut std::io::stderr());
            let mut line = String::new();
            return match std::io::stdin().read_line(&mut line) {
                Ok(0) | Err(_) => None,
                Ok(_) => Some(line.trim().to_string()),
            };
        }
        let previous = std::mem::replace(&mut self.posture, label.to_string());
        // The field is empty for the question and restored afterwards, so a
        // half-typed message is not eaten by a prompt that interrupted it.
        let stashed = self.editor.take();
        self.render();
        let answer = loop {
            let Some(key) = self.screen.as_ref().and_then(|s| s.key(FRAME).ok().flatten()) else {
                continue;
            };
            match key {
                Key::Enter => break Some(self.editor.take()),
                Key::Escape | Key::Interrupt => break None,
                other => {
                    self.edit(other);
                    self.render();
                }
            }
        };
        self.posture = previous;
        self.editor.clear();
        self.editor.insert_str(&stashed);
        answer.map(|a| a.trim().to_string())
    }

    /// Apply one key to the field or the view.
    fn edit(&mut self, key: Key) {
        let page = self.layout().transcript.1.max(1);
        match key {
            Key::Char(c) => self.editor.insert(c),
            Key::SoftEnter => self.editor.insert('\n'),
            Key::Backspace => self.editor.backspace(),
            Key::Delete => self.editor.delete(),
            Key::Left => self.editor.left(),
            Key::Right => self.editor.right(),
            Key::Home => self.editor.home(),
            Key::End => self.editor.end(),
            Key::KillToEnd => self.editor.kill_to_end(),
            Key::KillToStart => self.editor.kill_to_start(),
            Key::KillWord => self.editor.kill_word(),
            Key::Up => self.editor.previous(),
            Key::Down => self.editor.next(),
            Key::PageUp => self.transcript.scroll_up(page / 2, page),
            Key::PageDown => self.transcript.scroll_down(page / 2),
            // Three lines a notch, which is what a terminal's own scrollback
            // does and therefore what the hand already expects.
            Key::ScrollUp => self.transcript.scroll_up(WHEEL, page),
            Key::ScrollDown => self.transcript.scroll_down(WHEEL),
            Key::Escape => self.transcript.scroll_to_tail(),
            Key::Resize => self.resize(),
            Key::Enter | Key::Interrupt | Key::Eof | Key::Tab | Key::Ignored => {}
        }
    }

    /// Poll for a key while something else is happening.
    ///
    /// Returns true when the user asked to stop. Called between tokens, so it
    /// must not block: a timeout of zero is the whole point.
    pub fn poll_interrupt(&mut self) -> bool {
        if self.screen.is_none() {
            return crate::input::interrupted();
        }
        // Drained into a list first: `edit` takes `&mut self`, and holding a
        // borrow of the screen across it is exactly the shape the borrow
        // checker refuses.
        let mut pressed = Vec::new();
        while let Some(screen) = self.screen.as_ref() {
            match screen.key(Duration::ZERO) {
                Ok(Some(key)) => pressed.push(key),
                _ => break,
            }
        }
        let mut stop = false;
        for key in pressed {
            match key {
                Key::Interrupt => stop = true,
                // Reading back through a long answer while the model keeps
                // writing is a real thing to want.
                Key::PageUp
                | Key::PageDown
                | Key::ScrollUp
                | Key::ScrollDown
                | Key::Escape
                | Key::Resize => self.edit(key),
                _ => {}
            }
        }
        stop || crate::input::interrupted()
    }

    fn resize(&mut self) {
        self.size = ozgent_render::terminal_size();
        self.transcript.resize(self.size.0);
    }

    // -------------------------------------------------------- permission

    /// Ask on the permission bar, and wait there for an answer.
    ///
    /// The bar rather than a dialog: the question is about the call the user
    /// can see at the tail of the transcript, and a box drawn over it would
    /// hide the very thing being asked about.
    pub fn ask_permission(
        &mut self,
        tool: &str,
        effect: Effect,
        arguments: &serde_json::Value,
    ) -> Choice {
        if self.screen.is_none() {
            return crate::permission::ask(&self.theme, tool, effect, arguments);
        }
        self.question = Some(Question {
            tool: tool.to_string(),
            effect,
            summary: summarise(arguments, 60),
        });

        self.render();
        let choice = loop {
            let Some(key) = self.screen.as_ref().and_then(|s| s.key(FRAME).ok().flatten()) else {
                continue;
            };
            match key {
                Key::Char('1') | Key::Char('y') | Key::Enter => break Choice::Once,
                Key::Char('2') | Key::Char('s') => break Choice::Session,
                Key::Char('3') | Key::Char('a') => break Choice::Always,
                // Escape and Ctrl-C both mean no, because both are what a
                // person reaches for when they want something to stop.
                Key::Char('4') | Key::Char('n') | Key::Escape | Key::Interrupt => {
                    break Choice::Deny
                }
                // The call being asked about may be off the top of the
                // screen; scrolling to read it is part of answering.
                Key::PageUp | Key::PageDown | Key::ScrollUp | Key::ScrollDown | Key::Resize => {
                    self.edit(key);
                    self.render();
                }
                _ => {}
            }
        };
        self.question = None;
        choice
    }

    // ------------------------------------------------------------ frames

    fn layout(&self) -> Layout {
        let (width, height) = self.size;
        let rows = self.editor.layout(width.saturating_sub(4).max(1)).0.len();
        Layout::compute(width, height, rows.min(MAX_PROMPT_ROWS))
    }

    /// Paint the whole screen.
    pub fn render(&mut self) {
        // A resize can arrive as a signal rather than an event on some
        // terminals, so the size is verified rather than trusted.
        if ozgent_render::terminal_size() != self.size {
            self.resize();
        }
        let Some(_) = self.screen.as_ref() else { return };
        let layout = self.layout();
        self.last_frame = Instant::now();

        let mut rows: Vec<String> = Vec::with_capacity(layout.height);

        // The transcript, bottom-aligned: a new conversation starts next to
        // the prompt rather than floating at the top of an empty screen.
        let visible = self.transcript.visible(layout.transcript.1);
        let blanks = layout.transcript.1.saturating_sub(visible.len());
        rows.extend(std::iter::repeat_n(String::new(), blanks));
        rows.extend(visible.iter().map(|l| frame::truncate(l, layout.width)));

        // The prompt box.
        let (lines, caret) = self.editor.layout(layout.prompt_width());
        let visible_rows = layout.prompt.1.saturating_sub(2);
        // The field scrolls inside its own box once it outgrows it, so the
        // caret is always on screen even in a long paste.
        let first = caret.row.saturating_sub(visible_rows.saturating_sub(1));
        let shown: Vec<String> =
            lines.iter().skip(first).take(visible_rows).cloned().collect();
        // While scrolled back the view is held still on purpose, so a reply
        // arriving underneath changes nothing on screen. Say what is down
        // there, and how to get back to it.
        let hidden = self.transcript.hidden_below();
        let hint = (hidden > 0).then(|| format!("↓ {hidden} more · Esc to follow"));
        rows.extend(frame::prompt_box(
            &self.theme,
            &shown,
            layout.width,
            "› ",
            Style::dim(),
            hint.as_deref(),
        ));

        if layout.permission.is_some() {
            rows.push(self.permission_bar(layout.width));
        }
        if layout.status.is_some() {
            rows.push(frame::bar(&self.theme, &self.status_text(layout.width), layout.width));
        }
        rows.truncate(layout.height);

        // The caret sits inside the box: one row down for the top border, two
        // columns in for the border and the space.
        let caret_row = layout.prompt.0 + 1 + caret.row.saturating_sub(first);
        let caret_col = 2 + display_width("› ") + caret.col;
        let place = self.question.is_none().then_some((caret_row, caret_col.min(layout.width - 1)));

        if let Some(screen) = self.screen.as_mut() {
            let _ = screen.draw(&rows, place);
        }
    }

    /// The bar between the prompt and the status line.
    ///
    /// Two jobs, and it is the same bar for both on purpose: what tools may do
    /// without asking, and — when one is asking — the question. A user who has
    /// been reading "run programs: ask" for a week knows exactly where the
    /// question will appear when it does.
    fn permission_bar(&self, width: usize) -> String {
        let Some(q) = &self.question else {
            let text = if self.posture.is_empty() {
                "tools · no policy loaded".to_string()
            } else {
                self.posture.clone()
            };
            return pad(&self.theme.style(Style::dim(), &text), width);
        };

        let accent = match q.effect {
            Effect::Execute => Color::Red,
            Effect::Write => Color::Yellow,
            Effect::Read | Effect::Unknown => Color::Cyan,
        };
        let head = self.theme.style(
            Style { bold: true, color: Some(accent), ..Default::default() },
            &format!(" {} {} ", glyph(q.effect), q.tool),
        );
        let body = self.theme.style(Style::default(), &q.summary);
        let keys = self.theme.style(
            Style::dim(),
            "  1 yes · 2 session · 3 always · 4 no",
        );
        pad(&format!("{head}{body}{keys}"), width)
    }

    fn status_text(&self, width: usize) -> String {
        // `compose` pads to the width and adds a space at each end; the bar
        // does the same. Asked for two columns less and trimmed at both ends,
        // the two paddings compose into one instead of doubling.
        let line = crate::status::compose(&self.status, width.saturating_sub(2));
        line.trim().to_string()
    }

    /// Give the terminal back.
    pub fn close(&mut self) {
        if let Some(mut screen) = self.screen.take() {
            screen.close();
        }
    }

}

fn glyph(effect: Effect) -> &'static str {
    match effect {
        Effect::Execute => "▶",
        Effect::Write => "✎",
        Effect::Read => "◇",
        Effect::Unknown => "?",
    }
}

fn pad(text: &str, width: usize) -> String {
    let used = display_width(text);
    if used >= width {
        return frame::truncate(text, width);
    }
    format!("{text}{}", " ".repeat(width - used))
}

/// One line describing what a call would do.
fn summarise(arguments: &serde_json::Value, width: usize) -> String {
    let Some(object) = arguments.as_object() else {
        return clip(&arguments.to_string(), width);
    };
    if object.is_empty() {
        return "no arguments".to_string();
    }
    // Whichever argument names the thing acted on leads: a path identifies a
    // write far better than its mode does, and a command is its own summary.
    let lead = ["command", "path", "url", "query", "file"]
        .iter()
        .find_map(|k| object.get(*k))
        .or_else(|| object.values().next());
    let text = match lead {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };
    clip(&text.replace('\n', "⏎"), width)
}

fn clip(text: &str, width: usize) -> String {
    if display_width(text) <= width {
        return text.to_string();
    }
    let head: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_command_is_its_own_summary() {
        assert_eq!(summarise(&json!({ "command": "cargo test" }), 60), "cargo test");
    }

    #[test]
    fn the_argument_that_names_the_target_leads() {
        // `path` identifies a write; `mode` does not.
        let out = summarise(&json!({ "mode": "append", "path": "/etc/hosts" }), 60);
        assert_eq!(out, "/etc/hosts");
    }

    #[test]
    fn a_call_with_no_arguments_says_so_rather_than_showing_braces() {
        assert_eq!(summarise(&json!({}), 60), "no arguments");
    }

    #[test]
    fn a_long_value_is_clipped_to_the_bar() {
        let out = summarise(&json!({ "command": "x".repeat(300) }), 40);
        assert_eq!(display_width(&out), 40);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn a_newline_never_reaches_the_bar() {
        // A bar is one row; a newline in it would push every row below down.
        let out = summarise(&json!({ "content": "one\ntwo" }), 60);
        assert!(!out.contains('\n'), "{out:?}");
    }

    #[test]
    fn padding_fills_exactly_and_never_overflows() {
        assert_eq!(display_width(&pad("short", 20)), 20);
        assert_eq!(display_width(&pad(&"x".repeat(50), 20)), 20);
    }

    #[test]
    fn the_spinner_keeps_the_line_the_same_width() {
        // A spinner made of `|/-\\` shifts everything after it by a column
        // each frame, which reads as the text jittering rather than turning.
        let widths: std::collections::BTreeSet<usize> =
            SPINNER.iter().map(|f| display_width(f)).collect();
        assert_eq!(widths.len(), 1, "frames differ in width: {SPINNER:?}");
    }

    #[test]
    fn every_spinner_frame_is_distinct() {
        let unique: std::collections::BTreeSet<&&str> = SPINNER.iter().collect();
        assert_eq!(unique.len(), SPINNER.len(), "a repeated frame reads as a stall");
    }

    #[test]
    fn without_a_terminal_the_ui_falls_back_rather_than_failing() {
        // The piped path is how ozgent is scripted; it must not need a tty.
        let ui = Ui::new(Theme::plain(), None);
        assert!(!ui.full_screen());
    }

    #[test]
    fn every_effect_has_a_glyph_of_its_own() {
        let all = [Effect::Read, Effect::Write, Effect::Execute, Effect::Unknown];
        let glyphs: std::collections::BTreeSet<&str> = all.iter().map(|e| glyph(*e)).collect();
        assert_eq!(glyphs.len(), all.len(), "two effects sharing a glyph read as one");
    }
}
