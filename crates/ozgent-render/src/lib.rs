//! Terminal rendering for model output.

pub mod markdown;
pub mod stream;
pub mod theme;

pub use markdown::MarkdownRenderer;
pub use stream::StreamRenderer;
pub use theme::{Color, Style, Theme};

/// Width of the terminal, or a readable default when output is not a tty.
pub fn terminal_width() -> usize {
    crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(80)
        .clamp(40, 120)
}
