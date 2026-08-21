//! Markdown to styled terminal text.
//!
//! Rendering is span-based: inline content accumulates as (text, style) pairs
//! and is wrapped to the terminal width only once the block ends. Wrapping
//! measures with `unicode-width`, because escape sequences have no width and
//! CJK characters have two columns, so `str::len` would misplace every break.

use crate::theme::{Style, Theme};
use pulldown_cmark::{Alignment, CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// A run of text sharing one style.
#[derive(Debug, Clone)]
struct Span {
    text: String,
    style: Style,
}

pub struct MarkdownRenderer {
    theme: Theme,
    width: usize,
}

impl MarkdownRenderer {
    pub fn new(theme: Theme, width: usize) -> Self {
        // Very narrow terminals make wrapping meaningless; clamp to something
        // that can still hold an indented list marker plus a word.
        Self { theme, width: width.max(20) }
    }

    pub fn plain(width: usize) -> Self {
        Self::new(Theme::plain(), width)
    }

    pub fn theme(&self) -> &Theme {
        &self.theme
    }

    pub fn width(&self) -> usize {
        self.width
    }

    /// Render a complete markdown document.
    pub fn render(&self, markdown: &str) -> String {
        let mut options = Options::empty();
        options.insert(Options::ENABLE_STRIKETHROUGH);
        options.insert(Options::ENABLE_TABLES);
        options.insert(Options::ENABLE_TASKLISTS);
        options.insert(Options::ENABLE_FOOTNOTES);

        let mut ctx = Context::new(self);
        for event in Parser::new_ext(markdown, options) {
            ctx.handle(event);
        }
        ctx.finish()
    }
}

/// Where a block's text begins, and how continuation lines line up under it.
#[derive(Debug, Clone, Copy)]
struct Indent {
    first: usize,
    rest: usize,
}

struct Context<'a> {
    r: &'a MarkdownRenderer,
    out: String,
    spans: Vec<Span>,
    styles: Vec<Style>,
    /// One entry per open list; `Some(n)` counts an ordered list.
    lists: Vec<Option<u64>>,
    quote_depth: usize,
    /// Set while inside a fenced or indented code block.
    code_lang: Option<String>,
    code_buf: String,
    link_url: Option<String>,
    /// Text of the link, to decide whether the URL adds information.
    link_text: String,
    table: Option<Table>,
    /// Marker for the list item currently being built.
    pending_marker: Option<String>,
}

#[derive(Default)]
struct Table {
    alignments: Vec<Alignment>,
    rows: Vec<Vec<String>>,
    current: Vec<String>,
    in_head: bool,
}

impl<'a> Context<'a> {
    fn new(r: &'a MarkdownRenderer) -> Self {
        Self {
            r,
            out: String::new(),
            spans: Vec::new(),
            styles: Vec::new(),
            lists: Vec::new(),
            quote_depth: 0,
            code_lang: None,
            code_buf: String::new(),
            link_url: None,
            link_text: String::new(),
            table: None,
            pending_marker: None,
        }
    }

    fn style(&self) -> Style {
        self.styles.last().copied().unwrap_or_default()
    }

    fn push_style(&mut self, mut s: Style) {
        // Nested emphasis composes rather than replacing, so `**_x_**` is both.
        let base = self.style();
        s.bold |= base.bold;
        s.italic |= base.italic;
        s.dim |= base.dim;
        s.underline |= base.underline;
        s.strikethrough |= base.strikethrough;
        s.color = s.color.or(base.color);
        self.styles.push(s);
    }

    fn text(&mut self, text: &str) {
        let style = self.style();
        self.spans.push(Span { text: text.to_string(), style });
    }

    fn handle(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),

            Event::Text(t) => {
                if self.code_lang.is_some() {
                    self.code_buf.push_str(&t);
                } else if let Some(table) = self.table.as_mut() {
                    table.current.last_mut().map(|c| c.push_str(&t));
                } else {
                    if self.link_url.is_some() {
                        self.link_text.push_str(&t);
                    }
                    self.text(&t);
                }
            }

            Event::Code(t) => {
                let style = self.r.theme.code_inline;
                if let Some(table) = self.table.as_mut() {
                    table.current.last_mut().map(|c| c.push_str(&t));
                } else {
                    self.spans.push(Span { text: t.to_string(), style });
                }
            }

            Event::SoftBreak => {
                if self.table.is_none() {
                    self.text(" ");
                }
            }
            Event::HardBreak => self.text("\n"),

            Event::Rule => {
                self.blank_line();
                let line = "─".repeat(self.r.width.min(60));
                let styled = self.r.theme.style(self.r.theme.rule, &line);
                self.out.push_str(&styled);
                self.out.push('\n');
                self.blank_line();
            }

            Event::TaskListMarker(done) => {
                let mark = if done { "[x] " } else { "[ ] " };
                let style = self.r.theme.list_marker;
                self.spans.push(Span { text: mark.into(), style });
            }

            // Raw HTML has no terminal representation; dropping it is better
            // than printing tags at the user.
            Event::Html(_) | Event::InlineHtml(_) => {}
            Event::FootnoteReference(name) => {
                let style = self.r.theme.link;
                self.spans.push(Span { text: format!("[^{name}]"), style });
            }
            _ => {}
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level, .. } => {
                self.push_style(self.r.theme.heading[heading_index(level)]);
            }
            Tag::CodeBlock(kind) => {
                self.code_lang = Some(match kind {
                    CodeBlockKind::Fenced(l) => l.to_string(),
                    CodeBlockKind::Indented => String::new(),
                });
                self.code_buf.clear();
            }
            Tag::List(start) => {
                // The parent item's own text is still buffered; it has to be
                // emitted at the parent's indent before the level changes.
                let indent = self.current_indent();
                self.flush_spans(indent);
                self.lists.push(start);
            }
            Tag::Item => {
                let marker = match self.lists.last_mut() {
                    Some(Some(n)) => {
                        let m = format!("{n}. ");
                        *n += 1;
                        m
                    }
                    _ => "• ".to_string(),
                };
                self.pending_marker = Some(marker);
            }
            Tag::BlockQuote(_) => {
                self.blank_line();
                self.quote_depth += 1;
            }
            Tag::Emphasis => self.push_style(self.r.theme.emphasis),
            Tag::Strong => self.push_style(self.r.theme.strong),
            Tag::Strikethrough => self.push_style(self.r.theme.strikethrough),
            Tag::Link { dest_url, .. } => {
                self.link_url = Some(dest_url.to_string());
                self.link_text.clear();
                self.push_style(self.r.theme.link);
            }
            Tag::Image { dest_url, .. } => {
                let style = self.r.theme.link;
                self.spans.push(Span { text: format!("🖼 {dest_url}"), style });
            }
            Tag::Table(alignments) => {
                self.table = Some(Table { alignments, in_head: true, ..Default::default() });
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(t) = self.table.as_mut() {
                    t.current = Vec::new();
                }
            }
            Tag::TableCell => {
                if let Some(t) = self.table.as_mut() {
                    t.current.push(String::new());
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                let indent = self.current_indent();
                self.flush_spans(indent);
                self.blank_line();
            }
            TagEnd::Heading(_) => {
                let indent = self.current_indent();
                self.flush_spans(indent);
                self.styles.pop();
                self.blank_line();
            }
            TagEnd::CodeBlock => {
                let lang = self.code_lang.take().unwrap_or_default();
                let body = std::mem::take(&mut self.code_buf);
                self.emit_code_block(&lang, &body);
            }
            TagEnd::List(_) => {
                self.lists.pop();
                if self.lists.is_empty() {
                    self.blank_line();
                }
            }
            TagEnd::Item => {
                let indent = self.current_indent();
                self.flush_spans(indent);
            }
            TagEnd::BlockQuote(_) => {
                self.quote_depth = self.quote_depth.saturating_sub(1);
            }
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.styles.pop();
            }
            TagEnd::Link => {
                self.styles.pop();
                // Showing the URL only when it differs from the visible text
                // keeps `[https://x](https://x)` from printing twice.
                if let Some(url) = self.link_url.take() {
                    if url != self.link_text && !self.link_text.is_empty() {
                        let style = self.r.theme.link_url;
                        self.spans.push(Span { text: format!(" ({url})"), style });
                    }
                }
            }
            TagEnd::TableHead => {
                if let Some(t) = self.table.as_mut() {
                    let row = std::mem::take(&mut t.current);
                    t.rows.push(row);
                    t.in_head = false;
                }
            }
            TagEnd::TableRow => {
                if let Some(t) = self.table.as_mut() {
                    let row = std::mem::take(&mut t.current);
                    t.rows.push(row);
                }
            }
            TagEnd::Table => {
                if let Some(t) = self.table.take() {
                    self.emit_table(t);
                }
            }
            _ => {}
        }
    }

    /// Indentation for the block being built: two spaces per list level, with
    /// continuation lines hanging under the marker.
    fn current_indent(&self) -> Indent {
        let list_depth = self.lists.len().saturating_sub(1);
        let base = list_depth * 2;
        match &self.pending_marker {
            Some(m) => Indent { first: base, rest: base + m.width() },
            None => {
                let extra = if self.lists.is_empty() { 0 } else { 2 };
                Indent { first: base + extra, rest: base + extra }
            }
        }
    }

    fn flush_spans(&mut self, indent: Indent) {
        if self.spans.is_empty() && self.pending_marker.is_none() {
            return;
        }
        let mut spans = std::mem::take(&mut self.spans);
        if let Some(marker) = self.pending_marker.take() {
            spans.insert(
                0,
                Span { text: marker, style: self.r.theme.list_marker },
            );
        }

        let quote_prefix = self.quote_prefix();
        let available = self
            .r
            .width
            .saturating_sub(quote_prefix.width())
            .max(20);

        for line in wrap(&spans, available, indent) {
            let rendered = line
                .iter()
                .map(|s| self.r.theme.style(s.style, &s.text))
                .collect::<String>();
            if rendered.trim().is_empty() && quote_prefix.is_empty() {
                continue;
            }
            if !quote_prefix.is_empty() {
                let styled = self.r.theme.style(self.r.theme.block_quote, &quote_prefix);
                self.out.push_str(&styled);
            }
            self.out.push_str(&rendered);
            self.out.push('\n');
        }
    }

    fn quote_prefix(&self) -> String {
        self.r.theme.quote_prefix.repeat(self.quote_depth)
    }

    fn emit_code_block(&mut self, lang: &str, body: &str) {
        self.blank_line();
        let indent = " ".repeat(if self.lists.is_empty() { 2 } else { self.lists.len() * 2 + 2 });

        if !lang.is_empty() {
            let label = self.r.theme.style(Style::dim(), &format!("{indent}{lang}"));
            self.out.push_str(&label);
            self.out.push('\n');
        }

        for line in body.trim_end_matches('\n').split('\n') {
            self.out.push_str(&indent);
            self.out
                .push_str(&self.r.theme.style(self.r.theme.code_block, line));
            self.out.push('\n');
        }
        self.blank_line();
    }

    fn emit_table(&mut self, t: Table) {
        if t.rows.is_empty() {
            return;
        }
        let columns = t.rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut widths = vec![0usize; columns];
        for row in &t.rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.width());
            }
        }

        // Shrink proportionally if the table would overflow the terminal.
        let total: usize = widths.iter().sum::<usize>() + 3 * columns.saturating_sub(1) + 1;
        if total > self.r.width && columns > 0 {
            let budget = self.r.width.saturating_sub(3 * columns.saturating_sub(1) + 1);
            let sum: usize = widths.iter().sum();
            if sum > 0 {
                for w in widths.iter_mut() {
                    *w = ((*w * budget) / sum).max(3);
                }
            }
        }

        self.blank_line();
        for (r, row) in t.rows.iter().enumerate() {
            let mut line = String::new();
            for (i, width) in widths.iter().enumerate() {
                let cell = row.get(i).map(String::as_str).unwrap_or("");
                let cell = truncate(cell, *width);
                let align = t.alignments.get(i).copied().unwrap_or(Alignment::None);
                let padded = pad(&cell, *width, align);
                let styled = if r == 0 {
                    self.r.theme.style(self.r.theme.table_header, &padded)
                } else {
                    padded
                };
                if i > 0 {
                    line.push_str(&self.r.theme.style(Style::dim(), " │ "));
                }
                line.push_str(&styled);
            }
            self.out.push_str(line.trim_end());
            self.out.push('\n');

            if r == 0 {
                let sep: Vec<String> = widths.iter().map(|w| "─".repeat(*w)).collect();
                let joined = sep.join("─┼─");
                self.out.push_str(&self.r.theme.style(Style::dim(), &joined));
                self.out.push('\n');
            }
        }
        self.blank_line();
    }

    /// Append a blank line, collapsing runs of them.
    fn blank_line(&mut self) {
        if self.out.is_empty() || self.out.ends_with("\n\n") {
            return;
        }
        if !self.out.ends_with('\n') {
            self.out.push('\n');
        }
        self.out.push('\n');
    }

    fn finish(mut self) -> String {
        let indent = self.current_indent();
        self.flush_spans(indent);
        while self.out.ends_with("\n\n") {
            self.out.pop();
        }
        self.out
    }
}

fn heading_index(level: HeadingLevel) -> usize {
    match level {
        HeadingLevel::H1 => 0,
        HeadingLevel::H2 => 1,
        HeadingLevel::H3 => 2,
        HeadingLevel::H4 => 3,
        HeadingLevel::H5 => 4,
        HeadingLevel::H6 => 5,
    }
}

/// Break styled spans into display lines that fit `width` columns.
///
/// Splitting happens at word boundaries within spans, so a style never leaks
/// across a line break and a long styled run still wraps.
fn wrap(spans: &[Span], width: usize, indent: Indent) -> Vec<Vec<Span>> {
    let mut lines: Vec<Vec<Span>> = Vec::new();
    let mut current: Vec<Span> = Vec::new();
    let mut used = indent.first;

    let start_line = |current: &mut Vec<Span>, lines: &mut Vec<Vec<Span>>, used: &mut usize| {
        lines.push(std::mem::take(current));
        *used = indent.rest;
    };

    let pad = |n: usize| Span { text: " ".repeat(n), style: Style::default() };
    if indent.first > 0 {
        current.push(pad(indent.first));
    }

    for span in spans {
        for (i, segment) in span.text.split('\n').enumerate() {
            if i > 0 {
                // An explicit hard break.
                start_line(&mut current, &mut lines, &mut used);
                if indent.rest > 0 {
                    current.push(pad(indent.rest));
                }
            }
            for word in split_words(segment) {
                let w = word.width();
                let is_space = word.chars().all(char::is_whitespace);

                if used + w > width && !current.is_empty() {
                    if is_space {
                        // Never start a line with the space that overflowed.
                        start_line(&mut current, &mut lines, &mut used);
                        if indent.rest > 0 {
                            current.push(pad(indent.rest));
                        }
                        continue;
                    }
                    start_line(&mut current, &mut lines, &mut used);
                    if indent.rest > 0 {
                        current.push(pad(indent.rest));
                    }
                }

                // A single word longer than the line gets hard-split rather
                // than overflowing the terminal.
                if w > width {
                    for chunk in hard_split(&word, width.saturating_sub(used).max(1), width) {
                        let cw = chunk.width();
                        if used + cw > width && !current.is_empty() {
                            start_line(&mut current, &mut lines, &mut used);
                            if indent.rest > 0 {
                                current.push(pad(indent.rest));
                            }
                        }
                        used += cw;
                        current.push(Span { text: chunk, style: span.style });
                    }
                    continue;
                }

                used += w;
                current.push(Span { text: word, style: span.style });
            }
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Split into words, keeping the whitespace that follows each one.
fn split_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_space = false;

    for ch in text.chars() {
        let is_space = ch.is_whitespace();
        if !buf.is_empty() && is_space != in_space {
            out.push(std::mem::take(&mut buf));
        }
        in_space = is_space;
        buf.push(ch);
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Break an unbreakable word into line-sized chunks.
///
/// The first chunk gets whatever remains on the current line; every later
/// chunk gets a full line.
fn hard_split(word: &str, first_budget: usize, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut budget = first_budget.max(1);
    for ch in word.chars() {
        let w = ch.width().unwrap_or(0);
        if buf.width() + w > budget && !buf.is_empty() {
            out.push(std::mem::take(&mut buf));
            budget = width.max(1);
        }
        buf.push(ch);
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

fn truncate(s: &str, width: usize) -> String {
    if s.width() <= width {
        return s.to_string();
    }
    let mut out = String::new();
    for ch in s.chars() {
        if out.width() + ch.width().unwrap_or(0) + 1 > width {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}

fn pad(s: &str, width: usize, align: Alignment) -> String {
    let deficit = width.saturating_sub(s.width());
    match align {
        Alignment::Right => format!("{}{}", " ".repeat(deficit), s),
        Alignment::Center => {
            let left = deficit / 2;
            format!("{}{}{}", " ".repeat(left), s, " ".repeat(deficit - left))
        }
        _ => format!("{}{}", s, " ".repeat(deficit)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(md: &str) -> String {
        MarkdownRenderer::plain(60).render(md)
    }

    #[test]
    fn renders_a_heading() {
        assert_eq!(plain("# Title").trim(), "Title");
    }

    #[test]
    fn strips_inline_emphasis_markers() {
        let out = plain("Some **bold** and *italic* and `code`.");
        assert_eq!(out.trim(), "Some bold and italic and code.");
        assert!(!out.contains('*'), "markers must not survive: {out:?}");
    }

    #[test]
    fn wraps_to_the_given_width() {
        let text = "word ".repeat(40);
        let out = MarkdownRenderer::plain(30).render(&text);
        for line in out.lines() {
            assert!(line.width() <= 30, "line exceeds width: {:?} ({})", line, line.width());
        }
        assert!(out.lines().count() > 1, "long text must wrap");
    }

    #[test]
    fn wraps_cjk_by_display_width_not_byte_length() {
        // Each character is two columns wide; a byte-length check would fit
        // twice as many and overflow the terminal.
        let out = MarkdownRenderer::plain(20).render(&"日".repeat(40));
        for line in out.lines() {
            assert!(line.width() <= 20, "CJK line overflows: {:?} ({})", line, line.width());
        }
    }

    #[test]
    fn renders_bullet_lists_with_markers() {
        let out = plain("- one\n- two");
        assert!(out.contains("• one"), "{out:?}");
        assert!(out.contains("• two"), "{out:?}");
    }

    #[test]
    fn numbers_ordered_lists_sequentially() {
        let out = plain("1. first\n1. second\n1. third");
        assert!(out.contains("1. first"), "{out:?}");
        assert!(out.contains("2. second"), "source numbering is renumbered: {out:?}");
        assert!(out.contains("3. third"), "{out:?}");
    }

    #[test]
    fn indents_nested_lists() {
        let out = plain("- outer\n  - inner");
        let inner = out.lines().find(|l| l.contains("inner")).unwrap();
        let outer = out.lines().find(|l| l.contains("outer")).unwrap();
        let lead = |s: &str| s.len() - s.trim_start().len();
        assert!(lead(inner) > lead(outer), "nested item must indent: {out:?}");
    }

    #[test]
    fn continuation_lines_hang_under_the_marker() {
        let out = MarkdownRenderer::plain(30).render(&format!("- {}", "word ".repeat(20)));
        let lines: Vec<&str> = out.lines().filter(|l| !l.trim().is_empty()).collect();
        assert!(lines.len() > 1, "should wrap: {out:?}");
        assert!(
            lines[1].starts_with("  "),
            "continuation must align under the text, got {:?}",
            lines[1]
        );
    }

    #[test]
    fn renders_fenced_code_preserving_lines_and_indentation() {
        let out = plain("```rust\nfn main() {\n    println!(\"hi\");\n}\n```");
        assert!(out.contains("fn main() {"), "{out:?}");
        assert!(out.contains("    println!"), "inner indentation must survive: {out:?}");
        assert!(out.contains("rust"), "language label should show: {out:?}");
        assert!(!out.contains("```"), "fence markers must not survive: {out:?}");
    }

    #[test]
    fn code_blocks_are_not_wrapped() {
        // Wrapping code would corrupt it; long lines must stay intact.
        let long = "x".repeat(100);
        let out = MarkdownRenderer::plain(40).render(&format!("```\n{long}\n```"));
        assert!(out.contains(&long), "code line must not be wrapped: {out:?}");
    }

    #[test]
    fn renders_block_quotes_with_a_prefix() {
        let out = plain("> quoted text");
        assert!(out.contains("│ quoted text"), "{out:?}");
    }

    #[test]
    fn renders_tables_with_aligned_columns() {
        let out = plain("| a | bbbb |\n|---|------|\n| 1 | 2 |");
        assert!(out.contains("│"), "cells should be separated: {out:?}");
        assert!(out.contains("─┼─"), "header rule expected: {out:?}");
        assert!(out.contains('a') && out.contains("bbbb") && out.contains('1'), "{out:?}");
    }

    #[test]
    fn shows_a_link_url_only_when_it_adds_information() {
        let labelled = plain("[docs](https://example.com)");
        assert!(labelled.contains("docs"), "{labelled:?}");
        assert!(labelled.contains("https://example.com"), "{labelled:?}");

        let bare = plain("[https://example.com](https://example.com)");
        assert_eq!(bare.matches("https://example.com").count(), 1, "no duplicate: {bare:?}");
    }

    #[test]
    fn renders_task_lists() {
        let out = plain("- [x] done\n- [ ] todo");
        assert!(out.contains("[x] done"), "{out:?}");
        assert!(out.contains("[ ] todo"), "{out:?}");
    }

    #[test]
    fn applies_ansi_when_the_theme_is_enabled() {
        let out = MarkdownRenderer::new(Theme::default(), 60).render("**bold**");
        assert!(out.contains("\x1b["), "styled output expected: {out:?}");
        assert!(out.contains("bold"));
    }

    #[test]
    fn emits_no_ansi_when_the_theme_is_plain() {
        let out = plain("# H\n\n**bold** `code` [l](http://x)\n\n- item");
        assert!(!out.contains('\x1b'), "plain theme must be escape-free: {out:?}");
    }

    #[test]
    fn collapses_runs_of_blank_lines() {
        let out = plain("a\n\n\n\n\nb");
        assert!(!out.contains("\n\n\n"), "excess blank lines: {out:?}");
    }

    #[test]
    fn a_word_longer_than_the_line_is_hard_split() {
        let out = MarkdownRenderer::plain(20).render(&"z".repeat(60));
        for line in out.lines() {
            assert!(line.width() <= 20, "unsplit long word overflows: {:?}", line);
        }
        assert_eq!(out.replace('\n', "").len(), 60, "no characters lost: {out:?}");
    }

    #[test]
    fn horizontal_rules_render() {
        let out = plain("a\n\n---\n\nb");
        assert!(out.contains('─'), "{out:?}");
    }

    #[test]
    fn empty_input_produces_empty_output() {
        assert_eq!(plain(""), "");
    }
}
