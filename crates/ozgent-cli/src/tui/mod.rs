//! The full-screen terminal interface.
//!
//! ozgent's chat used to be a scrolling REPL. That has real virtues — native
//! scrollback, selection, copy — but it has nowhere to put anything that is
//! true *now* rather than true *then*: how full the context is, what the last
//! reply cost, which tools may act without asking. Those facts kept being
//! printed into the transcript, where they immediately became history.
//!
//! So the terminal is taken over, the way vim and htop take it over. The
//! screen is the application: a transcript that scrolls, a prompt box that
//! grows with what is typed, a bar saying what tools are allowed to do, and a
//! status line. Nothing prints itself any more; everything is a region that
//! is redrawn.
//!
//! Three things this has to get right, none of them optional:
//!
//! * **The terminal must be given back.** Raw mode and the alternate screen
//!   are global state. Leaving either behind hands the user a shell with no
//!   echo, which looks like ozgent broke their machine. [`Screen`] restores
//!   both on drop, and the panic hook restores them before the message prints.
//! * **A resize re-lays out, it does not re-wrap.** Lines are wrapped when
//!   they are rendered, and there is no way back to the text from them, so
//!   the transcript keeps every block's source and renders again.
//! * **It degrades.** Off a terminal there is nothing to take over, so
//!   [`Screen::open`] fails and the caller falls back to plain printing.

pub mod editor;
pub mod frame;
pub mod transcript;
pub mod ui;

use std::io::{IsTerminal, Write};

use ozgent_render::crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    terminal,
};

pub use editor::Editor;
pub use transcript::{Block, Transcript};
pub use ui::{Submission, Ui};

/// The terminal, taken over for as long as this lives.
pub struct Screen {
    /// False once the terminal has been handed back, so a second restore —
    /// from `Drop` after an explicit `close` — does nothing.
    open: bool,
}

impl Screen {
    /// Take over the terminal, or report that there is nothing to take over.
    pub fn open() -> std::io::Result<Self> {
        // Both ends, not just the output. `echo hi | ozgent chat` has a
        // terminal to draw on and no terminal to read keys from, and taking
        // the screen over there would leave the piped message unread while
        // the application waited for a keypress that cannot arrive.
        if !std::io::stdout().is_terminal() || !std::io::stdin().is_terminal() {
            return Err(std::io::Error::other("not a terminal"));
        }
        install_panic_hook();
        terminal::enable_raw_mode()?;
        let mut out = std::io::stdout();
        // The alternate screen means the user's scrollback is still there when
        // ozgent exits — the shell they started from comes back untouched.
        ozgent_render::crossterm::execute!(
            out,
            terminal::EnterAlternateScreen,
            cursor::Hide,
        )?;
        out.flush()?;
        Ok(Self { open: true })
    }

    /// Give the terminal back.
    pub fn close(&mut self) {
        if !self.open {
            return;
        }
        self.open = false;
        restore();
    }

    /// Paint a whole frame in one write.
    ///
    /// One write rather than a sequence of them: drawing row by row lets the
    /// terminal display a half-finished frame, which reads as flicker on every
    /// token of a streaming reply.
    pub fn draw(&mut self, rows: &[String], caret: Option<(usize, usize)>) -> std::io::Result<()> {
        let mut screen = String::with_capacity(rows.iter().map(String::len).sum::<usize>() + 64);
        screen.push_str("\x1b[H");
        for (i, row) in rows.iter().enumerate() {
            if i > 0 {
                screen.push_str("\r\n");
            }
            // Clear to the end of each line rather than clearing the screen
            // first: clearing everything and then filling it in is the other
            // classic way to make a redraw flicker.
            screen.push_str(row);
            screen.push_str("\x1b[K");
        }
        screen.push_str("\x1b[J");
        match caret {
            Some((row, col)) => {
                screen.push_str(&format!("\x1b[{};{}H", row + 1, col + 1));
                screen.push_str("\x1b[?25h");
            }
            None => screen.push_str("\x1b[?25l"),
        }
        let mut out = std::io::stdout();
        out.write_all(screen.as_bytes())?;
        out.flush()
    }

    /// The next key, or `None` if nothing arrived within `timeout`.
    ///
    /// A timeout rather than a blocking read so the caller can keep a spinner
    /// moving and notice a resize while nobody is typing.
    pub fn key(&self, timeout: std::time::Duration) -> std::io::Result<Option<Key>> {
        if !event::poll(timeout)? {
            return Ok(None);
        }
        Ok(translate(event::read()?))
    }

}

impl Drop for Screen {
    fn drop(&mut self) {
        self.close();
    }
}

/// Put the terminal back the way it was found.
fn restore() {
    let mut out = std::io::stdout();
    let _ = ozgent_render::crossterm::execute!(
        out,
        cursor::Show,
        terminal::LeaveAlternateScreen,
    );
    let _ = terminal::disable_raw_mode();
    let _ = out.flush();
}

/// Restore the terminal before a panic message is printed.
///
/// Without this a panic prints its backtrace into the alternate screen, which
/// then disappears — leaving a shell in raw mode and no explanation anywhere.
fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore();
            previous(info);
        }));
    });
}

/// What the user pressed, in the terms this application thinks in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Enter,
    /// A newline inside the field rather than a submission.
    SoftEnter,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    Tab,
    Escape,
    /// Ctrl-C: stop what is happening, keep the session.
    Interrupt,
    /// Ctrl-D on an empty line: leave.
    Eof,
    KillToEnd,
    KillToStart,
    KillWord,
    Resize,
    /// A key this application has no use for.
    Ignored,
}

/// Map a terminal event onto a [`Key`].
///
/// Split out so the bindings can be tested without a terminal — the mapping is
/// where a binding silently goes missing, not the reading.
pub fn translate(event: Event) -> Option<Key> {
    let key = match event {
        Event::Resize(..) => return Some(Key::Resize),
        Event::Key(k) => k,
        _ => return None,
    };
    // Windows terminals report press *and* release; acting on both types
    // everything twice.
    if key.kind == KeyEventKind::Release {
        return None;
    }
    Some(from_key(key))
}

fn from_key(key: KeyEvent) -> Key {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

    match key.code {
        // Alt-Enter and Shift-Enter both mean "another line" in every editor
        // people arrive from, and terminals disagree about which they send.
        KeyCode::Enter if alt || shift => Key::SoftEnter,
        KeyCode::Enter => Key::Enter,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Delete => Key::Delete,
        KeyCode::Left => Key::Left,
        KeyCode::Right => Key::Right,
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::Tab => Key::Tab,
        KeyCode::Esc => Key::Escape,
        KeyCode::Char(c) if ctrl => match c {
            'c' => Key::Interrupt,
            'd' => Key::Eof,
            'a' => Key::Home,
            'e' => Key::End,
            'k' => Key::KillToEnd,
            'u' => Key::KillToStart,
            'w' => Key::KillWord,
            'j' => Key::SoftEnter,
            'l' => Key::Resize, // redraw, which a resize already means
            'p' => Key::Up,
            'n' => Key::Down,
            'b' => Key::Left,
            'f' => Key::Right,
            'h' => Key::Backspace,
            _ => Key::Ignored,
        },
        KeyCode::Char(c) => Key::Char(c),
        _ => Key::Ignored,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(code: KeyCode, modifiers: KeyModifiers) -> Option<Key> {
        translate(Event::Key(KeyEvent::new(code, modifiers)))
    }

    #[test]
    fn plain_typing_is_a_character() {
        assert_eq!(press(KeyCode::Char('a'), KeyModifiers::NONE), Some(Key::Char('a')));
        assert_eq!(press(KeyCode::Char('A'), KeyModifiers::SHIFT), Some(Key::Char('A')));
    }

    #[test]
    fn enter_submits_but_alt_enter_does_not() {
        assert_eq!(press(KeyCode::Enter, KeyModifiers::NONE), Some(Key::Enter));
        assert_eq!(press(KeyCode::Enter, KeyModifiers::ALT), Some(Key::SoftEnter));
        assert_eq!(press(KeyCode::Enter, KeyModifiers::SHIFT), Some(Key::SoftEnter));
    }

    #[test]
    fn the_emacs_bindings_people_already_have_are_present() {
        for (c, expected) in [
            ('a', Key::Home),
            ('e', Key::End),
            ('k', Key::KillToEnd),
            ('u', Key::KillToStart),
            ('w', Key::KillWord),
            ('b', Key::Left),
            ('f', Key::Right),
            ('p', Key::Up),
            ('n', Key::Down),
        ] {
            assert_eq!(press(KeyCode::Char(c), KeyModifiers::CONTROL), Some(expected), "ctrl-{c}");
        }
    }

    #[test]
    fn interrupt_and_eof_are_kept_apart() {
        // Ctrl-C abandons the answer; Ctrl-D leaves. Confusing them loses the
        // conversation.
        assert_eq!(press(KeyCode::Char('c'), KeyModifiers::CONTROL), Some(Key::Interrupt));
        assert_eq!(press(KeyCode::Char('d'), KeyModifiers::CONTROL), Some(Key::Eof));
    }

    #[test]
    fn a_resize_is_a_key_like_any_other() {
        assert_eq!(translate(Event::Resize(80, 24)), Some(Key::Resize));
    }

    #[test]
    fn a_key_release_is_not_a_second_press() {
        // Windows terminals report both; acting on each types everything twice.
        let mut key = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
        key.kind = KeyEventKind::Release;
        assert_eq!(translate(Event::Key(key)), None);
    }

    #[test]
    fn an_unmapped_control_key_is_ignored_rather_than_typed() {
        // Ctrl-G must not insert a `g`.
        assert_eq!(press(KeyCode::Char('g'), KeyModifiers::CONTROL), Some(Key::Ignored));
    }

    #[test]
    fn opening_a_screen_without_a_terminal_fails_rather_than_corrupting_one() {
        // The test harness has no tty, which is the case being asserted: the
        // caller has to be able to fall back to plain printing.
        assert!(Screen::open().is_err());
    }
}
