//! Terminal rendering for model output.

pub mod markdown;
pub mod stream;
pub mod theme;

// Re-exported rather than added to `ozgent-cli`'s own dependencies. It is
// already in the graph through this crate, and every distinct dependency set
// in this workspace gets its own llama.cpp build directory — a quarter-hour
// compile and half a gigabyte to buy an import path.
pub use crossterm;

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

/// Columns `text` occupies on screen, ignoring SGR escapes.
///
/// Not `len()` and not `chars().count()`: an escape sequence takes no columns
/// and a CJK character takes two, so anything that draws a box or pads a
/// column has to measure rather than count.
pub fn display_width(text: &str) -> usize {
    use unicode_width::UnicodeWidthChar;

    let mut width = 0;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // CSI ... final byte in @..~; anything else is a two-character
            // escape, whose second character is consumed by taking one more.
            for next in chars.by_ref() {
                if next != '[' && !next.is_ascii_digit() && next != ';' {
                    break;
                }
            }
            continue;
        }
        width += c.width().unwrap_or(0);
    }
    width
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_take_no_columns() {
        assert_eq!(display_width("\x1b[1mbold\x1b[0m"), 4);
    }

    #[test]
    fn wide_characters_take_two() {
        assert_eq!(display_width("\u{65e5}\u{672c}"), 4);
        assert_eq!(display_width("plain"), 5);
    }

    #[test]
    fn a_size_is_never_zero_columns() {
        // Dividing by a width of zero later is a very quiet crash.
        let (cols, rows) = terminal_size();
        assert!(cols >= 20 && rows >= 1);
    }
}
