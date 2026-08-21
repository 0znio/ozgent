//! Terminal styling.
//!
//! Colours are the 8 ANSI names rather than fixed RGB, so output follows
//! whatever palette the user's terminal already uses and stays legible on both
//! light and dark backgrounds.

/// SGR attributes for one run of text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Style {
    pub bold: bool,
    pub italic: bool,
    pub dim: bool,
    pub underline: bool,
    pub strikethrough: bool,
    pub color: Option<Color>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BrightBlack,
}

impl Color {
    fn code(self) -> u8 {
        match self {
            Self::Red => 31,
            Self::Green => 32,
            Self::Yellow => 33,
            Self::Blue => 34,
            Self::Magenta => 35,
            Self::Cyan => 36,
            Self::White => 37,
            Self::BrightBlack => 90,
        }
    }
}

impl Style {
    pub fn bold() -> Self {
        Self { bold: true, ..Default::default() }
    }
    pub fn dim() -> Self {
        Self { dim: true, ..Default::default() }
    }
    pub fn color(c: Color) -> Self {
        Self { color: Some(c), ..Default::default() }
    }

    pub fn is_plain(&self) -> bool {
        *self == Self::default()
    }

    /// The SGR escape that turns this style on.
    pub fn prefix(&self) -> String {
        if self.is_plain() {
            return String::new();
        }
        let mut codes: Vec<String> = Vec::new();
        if self.bold {
            codes.push("1".into());
        }
        if self.dim {
            codes.push("2".into());
        }
        if self.italic {
            codes.push("3".into());
        }
        if self.underline {
            codes.push("4".into());
        }
        if self.strikethrough {
            codes.push("9".into());
        }
        if let Some(c) = self.color {
            codes.push(c.code().to_string());
        }
        format!("\x1b[{}m", codes.join(";"))
    }

    pub const RESET: &'static str = "\x1b[0m";

    /// Wrap `text` in this style.
    pub fn apply(&self, text: &str) -> String {
        if self.is_plain() || text.is_empty() {
            return text.to_string();
        }
        format!("{}{}{}", self.prefix(), text, Self::RESET)
    }
}

/// Which style each markdown construct gets.
#[derive(Debug, Clone)]
pub struct Theme {
    pub heading: [Style; 6],
    pub emphasis: Style,
    pub strong: Style,
    pub strikethrough: Style,
    pub code_inline: Style,
    pub code_block: Style,
    pub link: Style,
    pub link_url: Style,
    pub block_quote: Style,
    pub list_marker: Style,
    pub rule: Style,
    pub table_header: Style,
    /// Reasoning traces, kept visually subordinate to the answer.
    pub thinking: Style,
    /// Prefix drawn on each blockquote line.
    pub quote_prefix: String,
    /// Whether to colour at all. Off when piping to a file.
    pub enabled: bool,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            heading: [
                Style { bold: true, color: Some(Color::Magenta), ..Default::default() },
                Style { bold: true, color: Some(Color::Blue), ..Default::default() },
                Style { bold: true, color: Some(Color::Cyan), ..Default::default() },
                Style { bold: true, ..Default::default() },
                Style { bold: true, dim: true, ..Default::default() },
                Style { dim: true, ..Default::default() },
            ],
            emphasis: Style { italic: true, ..Default::default() },
            strong: Style::bold(),
            strikethrough: Style { strikethrough: true, dim: true, ..Default::default() },
            code_inline: Style::color(Color::Yellow),
            code_block: Style::color(Color::Green),
            link: Style { underline: true, color: Some(Color::Blue), ..Default::default() },
            link_url: Style::dim(),
            block_quote: Style::dim(),
            list_marker: Style::color(Color::Cyan),
            rule: Style::dim(),
            table_header: Style::bold(),
            thinking: Style { dim: true, italic: true, ..Default::default() },
            quote_prefix: "│ ".into(),
            enabled: true,
        }
    }
}

impl Theme {
    /// A theme that emits no escape sequences, for pipes and tests.
    pub fn plain() -> Self {
        Self { enabled: false, ..Default::default() }
    }

    pub fn style(&self, s: Style, text: &str) -> String {
        if self.enabled { s.apply(text) } else { text.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_style_adds_nothing() {
        assert_eq!(Style::default().apply("hi"), "hi");
    }

    #[test]
    fn styles_wrap_and_reset() {
        let s = Style::bold().apply("hi");
        assert!(s.starts_with("\x1b[1m"));
        assert!(s.ends_with(Style::RESET));
    }

    #[test]
    fn combined_attributes_share_one_escape() {
        let s = Style { bold: true, italic: true, color: Some(Color::Red), ..Default::default() };
        assert_eq!(s.prefix(), "\x1b[1;3;31m");
    }

    #[test]
    fn a_disabled_theme_emits_no_escapes() {
        let t = Theme::plain();
        let out = t.style(Style::bold(), "hi");
        assert_eq!(out, "hi", "piping to a file must not get escape codes");
    }
}
