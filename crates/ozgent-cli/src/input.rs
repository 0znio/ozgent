//! Terminal input: line editing, history, and interruption.
//!
//! A chat REPL that cannot recall the previous message or fix a typo is
//! tiring to use, and one where Ctrl-C kills the process mid-answer loses the
//! conversation. Both are handled here so `chat.rs` stays about the
//! conversation rather than the terminal.

use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use rustyline::{Config, EditMode, Editor};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set while the user has asked to stop the current response.
///
/// A single global is right here: there is one foreground generation at a
/// time, and the signal handler cannot carry state.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Install the Ctrl-C handler.
///
/// Idempotent, because installing twice would replace the first handler and
/// leave the flag permanently unset.
pub fn install_interrupt_handler() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let _ = ctrlc::set_handler(|| {
            // Only ever set the flag: doing real work in a signal handler is
            // unsafe, and the generate loop polls this between tokens.
            INTERRUPTED.store(true, Ordering::SeqCst);
        });
    });
}

/// Clear the flag before starting work that can be interrupted.
pub fn arm_interrupt() {
    INTERRUPTED.store(false, Ordering::SeqCst);
}

/// Whether the user has asked to stop.
pub fn interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

/// What the user did at the prompt.
pub enum Input {
    Line(String),
    /// Ctrl-C: abandon this line, keep the session.
    Cancelled,
    /// Ctrl-D or end of piped input.
    Eof,
}

pub struct Prompt {
    editor: Editor<(), FileHistory>,
    history_path: Option<PathBuf>,
    /// True when input is not a terminal, e.g. a pipe or a test harness.
    piped: bool,
}

impl Prompt {
    pub fn new(history_path: Option<PathBuf>) -> Self {
        let config = Config::builder()
            .edit_mode(EditMode::Emacs)
            .history_ignore_space(true)
            .history_ignore_dups(true)
            .map(|b| b.build())
            .unwrap_or_default();

        let mut editor = Editor::<(), FileHistory>::with_config(config)
            .unwrap_or_else(|_| Editor::new().expect("terminal editor"));

        if let Some(path) = &history_path {
            // A missing history file is the normal first-run case.
            let _ = editor.load_history(path);
        }

        Self {
            editor,
            history_path,
            piped: !std::io::IsTerminal::is_terminal(&std::io::stdin()),
        }
    }

    /// Read one line.
    ///
    /// History is recorded explicitly rather than via `auto_add_history`,
    /// which only fires on the interactive path — so a session driven from a
    /// script would silently record nothing.
    pub fn read(&mut self, prompt: &str) -> Input {
        match self.editor.readline(prompt) {
            Ok(line) => {
                let trimmed = line.trim();
                // Skip blanks and `/exit` — neither is worth recalling.
                if !trimmed.is_empty() && !matches!(trimmed, "/exit" | "/quit" | "/q") {
                    let _ = self.editor.add_history_entry(trimmed);
                }
                Input::Line(line)
            }
            Err(ReadlineError::Interrupted) => Input::Cancelled,
            Err(ReadlineError::Eof) => Input::Eof,
            Err(_) => Input::Eof,
        }
    }

    pub fn is_piped(&self) -> bool {
        self.piped
    }

    /// Persist history. Failure is not worth interrupting the user over.
    pub fn save(&mut self) {
        if let Some(path) = &self.history_path {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = self.editor.save_history(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_survives_a_save_and_reload() {
        let dir = std::env::temp_dir().join(format!("ozgent-hist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history");

        {
            let mut p = Prompt::new(Some(path.clone()));
            let _ = p.editor.add_history_entry("remember me");
            p.save();
        }
        assert!(path.exists(), "history file should be written");

        let reloaded = Prompt::new(Some(path.clone()));
        assert!(
            reloaded.editor.history().iter().any(|e| e.contains("remember me")),
            "history should survive a restart"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_interrupt_flag_round_trips() {
        arm_interrupt();
        assert!(!interrupted(), "arming must clear the flag");
        INTERRUPTED.store(true, Ordering::SeqCst);
        assert!(interrupted());
        arm_interrupt();
        assert!(!interrupted(), "re-arming must clear it again");
    }

    #[test]
    fn installing_the_handler_twice_is_safe() {
        // Installing twice would otherwise replace the handler and leave the
        // flag permanently unset.
        install_interrupt_handler();
        install_interrupt_handler();
        arm_interrupt();
        assert!(!interrupted());
    }
}
