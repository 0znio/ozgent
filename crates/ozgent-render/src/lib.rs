//! Terminal rendering for model output.

pub mod markdown;
pub mod stream;
pub mod theme;

pub use markdown::MarkdownRenderer;
pub use stream::StreamRenderer;
pub use theme::{Color, Style, Theme};

/// Width of the terminal, or a readable default when output is not a tty.
///
/// Read fresh on every call rather than cached, because a terminal can be
/// resized in the middle of a session and text wrapped to the old width is
/// the most visible way to look broken. The call is an `ioctl`, so asking
/// again per rendered block costs nothing worth saving.
///
/// The whole width is used. A prose-readability cap was tempting and wrong:
/// on a wide terminal it leaves half the window empty while tables and code
/// blocks — which have their own natural width — are truncated for no reason.
pub fn terminal_width() -> usize {
    terminal_size().0
}

/// Columns and rows, falling back to a readable default off a terminal.
pub fn terminal_size() -> (usize, usize) {
    match crossterm::terminal::size() {
        // A zero from `ioctl` means "unknown", not "no columns", and dividing
        // by it later would be a very quiet crash.
        Ok((w, h)) if w > 0 && h > 0 => ((w as usize).max(20), h as usize),
        _ => (80, 24),
    }
}
