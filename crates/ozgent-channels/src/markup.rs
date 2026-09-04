//! Rendering a model's markdown for a chat app.
//!
//! Neither Telegram nor WhatsApp reads markdown. Telegram reads a small,
//! strictly-validated subset of HTML: send it a tag it does not know, or one
//! left unclosed, and it rejects the *whole* message with a 400 rather than
//! dropping the tag — so a stray `<` in a code sample loses the reply. WhatsApp
//! has no markup language at all, only four wrapper characters it applies
//! opportunistically to plain text.
//!
//! So both are generated from the parsed document rather than by patching the
//! source, which is what makes escaping total: every run of text goes through
//! the flavour's escape on its way out, and every tag this module opens it also
//! closes.
//!
//! What markdown has and chat apps do not — headings, lists, tables, rules — is
//! rendered as text that reads correctly rather than dropped. A table on a
//! phone is going to be ugly whatever happens; being able to read the numbers
//! beats being shown nothing.

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

/// Which chat app the output is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavour {
    /// Telegram's `parse_mode=HTML` subset.
    TelegramHtml,
    /// WhatsApp's `*bold*`, `_italic_`, `~strike~`, `` `mono` ``.
    WhatsApp,
    /// No markup at all. The fallback when a provider rejects the marked-up
    /// form, and the only safe thing to send when the reason is unknown.
    Plain,
}

/// Render markdown for a chat app.
pub fn render(markdown: &str, flavour: Flavour) -> String {
    let mut w = Writer::new(flavour);
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_TASKLISTS);
    for event in Parser::new_ext(markdown, options) {
        w.event(event);
    }
    w.finish()
}

/// Escape text so a flavour treats it as text and nothing else.
pub fn escape(text: &str, flavour: Flavour) -> String {
    match flavour {
        // The only three characters Telegram's parser gives meaning to. `"`
        // matters inside an attribute value, which is handled where hrefs are
        // written rather than here.
        Flavour::TelegramHtml => {
            let mut out = String::with_capacity(text.len());
            for c in text.chars() {
                match c {
                    '&' => out.push_str("&amp;"),
                    '<' => out.push_str("&lt;"),
                    '>' => out.push_str("&gt;"),
                    _ => out.push(c),
                }
            }
            out
        }
        // WhatsApp has no escape character. Its markup only fires on a matched
        // pair around non-space text, so a lone `*` is already literal and
        // there is nothing honest to do about a matched one.
        Flavour::WhatsApp | Flavour::Plain => text.to_string(),
    }
}

struct Writer {
    flavour: Flavour,
    out: String,
    /// Open ordered-list counters, outermost first. A `None` is a bullet list.
    lists: Vec<Option<u64>>,
    /// Text captured instead of written, for constructs whose content is
    /// needed whole: a code block's body, a link's label.
    capture: Option<String>,
    /// Language of the fence being captured.
    lang: Option<String>,
    /// Set just after opening a block container, so its first child does
    /// not push a blank line in between. Without it `<blockquote>` and its
    /// text end up separated, which Telegram renders as an empty quote
    /// followed by loose text.
    fresh: bool,
    /// Cells of the table row being built.
    row: Vec<String>,
    in_table: bool,
    quote: usize,
}

impl Writer {
    fn new(flavour: Flavour) -> Self {
        Self {
            flavour,
            out: String::new(),
            lists: Vec::new(),
            capture: None,
            lang: None,
            fresh: false,
            row: Vec::new(),
            in_table: false,
            quote: 0,
        }
    }

    fn finish(mut self) -> String {
        while self.out.ends_with('\n') {
            self.out.pop();
        }
        self.out
    }

    /// Write already-escaped or generated markup verbatim.
    fn raw(&mut self, s: &str) {
        match &mut self.capture {
            Some(buf) => buf.push_str(s),
            None => self.out.push_str(s),
        }
    }

    /// Write text, escaping it for the flavour.
    fn text(&mut self, s: &str) {
        let escaped = escape(s, self.flavour);
        self.raw(&escaped);
    }

    /// Start a block, leaving exactly one blank line before it.
    fn block(&mut self) {
        if self.out.is_empty() || std::mem::take(&mut self.fresh) {
            return;
        }
        self.trim_newlines();
        self.out.push_str("\n\n");
    }

    /// Start a line without forcing a blank one.
    fn line(&mut self) {
        if std::mem::take(&mut self.fresh) {
            return;
        }
        if !self.out.is_empty() && !self.out.ends_with('\n') {
            self.out.push('\n');
        }
    }

    fn trim_newlines(&mut self) {
        while self.out.ends_with('\n') {
            self.out.pop();
        }
    }

    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(t) => self.text(&t),
            Event::Code(t) => match self.flavour {
                Flavour::TelegramHtml => {
                    let escaped = escape(&t, self.flavour);
                    self.raw(&format!("<code>{escaped}</code>"));
                }
                Flavour::WhatsApp => self.raw(&format!("`{t}`")),
                Flavour::Plain => self.text(&t),
            },
            Event::SoftBreak => self.raw(" "),
            Event::HardBreak => self.raw("\n"),
            Event::Rule => {
                self.block();
                self.raw("──────────");
            }
            // Raw HTML in a model's reply is not markup to pass through: it is
            // either something the model wrote as an example, or something a
            // web page put in its mouth. Either way it goes out as text.
            Event::Html(t) | Event::InlineHtml(t) => self.text(&t),
            Event::TaskListMarker(done) => {
                self.raw(if done { "[x] " } else { "[ ] " });
            }
            Event::FootnoteReference(name) => self.text(&format!("[{name}]")),
            Event::InlineMath(t) | Event::DisplayMath(t) => self.text(&t),
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {
                if self.lists.is_empty() {
                    self.block();
                } else {
                    self.line();
                }
            }
            // No chat app has headings. Bold on its own line is what everyone
            // means by one, and it keeps the document's shape legible.
            Tag::Heading { .. } => {
                self.block();
                self.raw(self.open_bold());
            }
            Tag::BlockQuote(_) => {
                self.block();
                self.quote += 1;
                if self.flavour == Flavour::TelegramHtml {
                    self.raw("<blockquote>");
                }
                self.fresh = true;
            }
            Tag::CodeBlock(kind) => {
                self.block();
                self.lang = match kind {
                    CodeBlockKind::Fenced(info) => {
                        let lang = info.split_whitespace().next().unwrap_or("").to_string();
                        (!lang.is_empty()).then_some(lang)
                    }
                    CodeBlockKind::Indented => None,
                };
                self.capture = Some(String::new());
            }
            Tag::List(start) => {
                if self.lists.is_empty() {
                    self.block();
                } else {
                    self.line();
                }
                self.lists.push(start);
            }
            Tag::Item => {
                self.line();
                let depth = self.lists.len().saturating_sub(1);
                let indent = "  ".repeat(depth);
                let marker = match self.lists.last_mut() {
                    Some(Some(n)) => {
                        let m = format!("{n}. ");
                        *n += 1;
                        m
                    }
                    _ => "• ".to_string(),
                };
                self.raw(&format!("{indent}{marker}"));
            }
            Tag::Emphasis => self.raw(self.open_italic()),
            Tag::Strong => self.raw(self.open_bold()),
            Tag::Strikethrough => self.raw(self.open_strike()),
            Tag::Link { dest_url, .. } => {
                // The label is needed before the URL can be written for
                // Telegram, and needed *instead of* it nowhere — so capture.
                self.lang = Some(dest_url.to_string());
                self.capture = Some(String::new());
            }
            // An image cannot be shown inside a text message. Its alt text is
            // the only part that carries meaning.
            Tag::Image { dest_url, .. } => {
                self.lang = Some(dest_url.to_string());
                self.capture = Some(String::new());
            }
            Tag::Table(_) => {
                self.block();
                self.in_table = true;
            }
            Tag::TableHead | Tag::TableRow => {
                self.row.clear();
            }
            Tag::TableCell => {
                self.capture = Some(String::new());
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.line(),
            TagEnd::Heading(_) => {
                self.raw(self.close_bold());
                self.line();
            }
            TagEnd::BlockQuote(_) => {
                self.quote = self.quote.saturating_sub(1);
                if self.flavour == Flavour::TelegramHtml {
                    // The closing tag hugs the text: a newline before it is
                    // rendered by Telegram as a blank final line in the quote.
                    self.trim_newlines();
                    self.raw("</blockquote>");
                }
                self.line();
            }
            TagEnd::CodeBlock => {
                let code = self.capture.take().unwrap_or_default();
                let lang = self.lang.take();
                let code = code.trim_end_matches('\n').to_string();
                match self.flavour {
                    Flavour::TelegramHtml => {
                        // Already escaped: the body was captured through
                        // `text`, so escaping again turns `&lt;` into
                        // `&amp;lt;` and the code sample shows entities.
                        let body = code.as_str();
                        match lang.as_deref() {
                            Some(l) => {
                                let l = escape(l, Flavour::TelegramHtml);
                                self.raw(&format!(
                                    "<pre><code class=\"language-{l}\">{body}</code></pre>"
                                ));
                            }
                            None => self.raw(&format!("<pre>{body}</pre>")),
                        }
                    }
                    // WhatsApp's fence needs its own lines or it is shown
                    // literally, and it takes no language.
                    Flavour::WhatsApp => self.raw(&format!("```\n{code}\n```")),
                    Flavour::Plain => self.raw(&code),
                }
                self.line();
            }
            TagEnd::List(_) => {
                self.lists.pop();
                if self.lists.is_empty() {
                    self.line();
                }
            }
            TagEnd::Item => self.line(),
            TagEnd::Emphasis => self.raw(self.close_italic()),
            TagEnd::Strong => self.raw(self.close_bold()),
            TagEnd::Strikethrough => self.raw(self.close_strike()),
            TagEnd::Link => {
                let label = self.capture.take().unwrap_or_default();
                let url = self.lang.take().unwrap_or_default();
                self.write_link(&label, &url);
            }
            TagEnd::Image => {
                let alt = self.capture.take().unwrap_or_default();
                let url = self.lang.take().unwrap_or_default();
                let label = if alt.trim().is_empty() { "image".to_string() } else { alt };
                self.write_link(&label, &url);
            }
            TagEnd::TableCell => {
                let cell = self.capture.take().unwrap_or_default();
                self.row.push(cell.trim().to_string());
            }
            TagEnd::TableHead => {
                let row = std::mem::take(&mut self.row);
                self.line();
                let text = row.join(" · ");
                self.raw(&format!("{}{text}{}", self.open_bold(), self.close_bold()));
                self.line();
            }
            TagEnd::TableRow => {
                let row = std::mem::take(&mut self.row);
                self.line();
                self.raw(&row.join(" · "));
                self.line();
            }
            TagEnd::Table => {
                self.in_table = false;
                self.line();
            }
            _ => {}
        }
    }

    fn write_link(&mut self, label: &str, url: &str) {
        match self.flavour {
            Flavour::TelegramHtml => {
                // The label is already escaped — it was captured through
                // `text`. The URL has not been, and goes inside an attribute,
                // so it needs `"` handled as well.
                let href = escape(url, Flavour::TelegramHtml).replace('"', "&quot;");
                self.raw(&format!("<a href=\"{href}\">{label}</a>"));
            }
            // Neither has link markup, and both linkify a bare URL. Repeating
            // a URL that is already its own label helps nobody.
            Flavour::WhatsApp | Flavour::Plain => {
                if label.trim() == url.trim() || url.trim().is_empty() {
                    self.raw(label);
                } else {
                    self.raw(&format!("{label} ({url})"));
                }
            }
        }
    }

    fn open_bold(&self) -> &'static str {
        match self.flavour {
            Flavour::TelegramHtml => "<b>",
            Flavour::WhatsApp => "*",
            Flavour::Plain => "",
        }
    }
    fn close_bold(&self) -> &'static str {
        match self.flavour {
            Flavour::TelegramHtml => "</b>",
            Flavour::WhatsApp => "*",
            Flavour::Plain => "",
        }
    }
    fn open_italic(&self) -> &'static str {
        match self.flavour {
            Flavour::TelegramHtml => "<i>",
            Flavour::WhatsApp => "_",
            Flavour::Plain => "",
        }
    }
    fn close_italic(&self) -> &'static str {
        match self.flavour {
            Flavour::TelegramHtml => "</i>",
            Flavour::WhatsApp => "_",
            Flavour::Plain => "",
        }
    }
    fn open_strike(&self) -> &'static str {
        match self.flavour {
            Flavour::TelegramHtml => "<s>",
            Flavour::WhatsApp => "~",
            Flavour::Plain => "",
        }
    }
    fn close_strike(&self) -> &'static str {
        match self.flavour {
            Flavour::TelegramHtml => "</s>",
            Flavour::WhatsApp => "~",
            Flavour::Plain => "",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tg(md: &str) -> String {
        render(md, Flavour::TelegramHtml)
    }
    fn wa(md: &str) -> String {
        render(md, Flavour::WhatsApp)
    }

    #[test]
    fn emphasis_becomes_the_flavours_own_markup() {
        assert_eq!(tg("**bold** and *italic*"), "<b>bold</b> and <i>italic</i>");
        assert_eq!(wa("**bold** and *italic*"), "*bold* and _italic_");
    }

    #[test]
    fn html_in_the_reply_is_shown_and_never_sent_as_markup() {
        // The failure this pins is not cosmetic. Telegram rejects a message
        // containing a tag it does not know, so an unescaped `<div>` in a code
        // sample loses the entire reply; and passing a model's raw HTML
        // through would let a fetched page inject markup into a chat.
        assert_eq!(tg("use <div> here"), "use &lt;div&gt; here");
        assert_eq!(tg("a & b"), "a &amp; b");
        assert_eq!(tg("<script>alert(1)</script>"), "&lt;script&gt;alert(1)&lt;/script&gt;");
    }

    #[test]
    fn a_code_block_keeps_its_language_and_escapes_its_body() {
        assert_eq!(
            tg("```rust\nlet a = b < c;\n```"),
            "<pre><code class=\"language-rust\">let a = b &lt; c;</code></pre>"
        );
        assert_eq!(tg("```\nplain\n```"), "<pre>plain</pre>");
    }

    #[test]
    fn whatsapp_fences_need_their_own_lines() {
        // Inline, WhatsApp shows the backticks instead of applying them.
        assert_eq!(wa("```\nlet a = 1;\n```"), "```\nlet a = 1;\n```");
    }

    #[test]
    fn inline_code_is_escaped_inside_its_tag() {
        assert_eq!(tg("call `f<T>()` now"), "call <code>f&lt;T&gt;()</code> now");
        assert_eq!(wa("call `f()` now"), "call `f()` now");
    }

    #[test]
    fn a_heading_becomes_bold_on_its_own_line() {
        assert_eq!(tg("# Title\n\nbody"), "<b>Title</b>\n\nbody");
        assert_eq!(wa("## Title\n\nbody"), "*Title*\n\nbody");
    }

    #[test]
    fn lists_are_rendered_as_text_because_no_chat_app_has_them() {
        assert_eq!(tg("- one\n- two"), "• one\n• two");
        assert_eq!(tg("1. one\n2. two"), "1. one\n2. two");
    }

    #[test]
    fn a_nested_list_is_indented_rather_than_flattened() {
        let out = tg("- one\n  - inner\n- two");
        assert_eq!(out, "• one\n  • inner\n• two");
    }

    #[test]
    fn an_ordered_list_counts_from_where_it_says() {
        assert_eq!(tg("3. three\n4. four"), "3. three\n4. four");
    }

    #[test]
    fn a_link_keeps_its_label_and_quotes_its_url() {
        assert_eq!(tg("[docs](https://x.test/a?b=1&c=2)"), "<a href=\"https://x.test/a?b=1&amp;c=2\">docs</a>");
        // A quote in an href would otherwise close the attribute and let the
        // rest of the URL be read as markup.
        assert!(tg("[x](https://x.test/\"onmouseover=)").contains("&quot;"));
        assert_eq!(wa("[docs](https://x.test)"), "docs (https://x.test)");
    }

    #[test]
    fn a_bare_url_is_not_repeated_on_a_flavour_that_linkifies() {
        assert_eq!(wa("[https://x.test](https://x.test)"), "https://x.test");
    }

    #[test]
    fn a_table_is_readable_rather_than_dropped() {
        let md = "| a | b |\n|---|---|\n| 1 | 2 |";
        assert_eq!(wa(md), "*a · b*\n1 · 2");
    }

    #[test]
    fn a_blockquote_uses_telegrams_own_tag() {
        assert_eq!(tg("> quoted"), "<blockquote>quoted</blockquote>");
        assert_eq!(wa("> quoted"), "quoted");
    }

    #[test]
    fn plain_strips_every_marker_and_keeps_the_words() {
        let md = "# Title\n\n**bold** `code` [x](https://y.test)\n\n- one";
        let out = render(md, Flavour::Plain);
        assert_eq!(out, "Title\n\nbold code x (https://y.test)\n\n• one");
        assert!(!out.contains('<'), "no markup at all");
    }

    #[test]
    fn every_tag_opened_for_telegram_is_closed() {
        // Telegram rejects the whole message on an unbalanced tag, so this is
        // checked over a document using every construct rather than trusted.
        let md = "# H\n\n**b** *i* ~~s~~ `c` [l](https://x.test)\n\n> q\n\n```rs\nx\n```\n\n- a\n- b\n\n| h |\n|---|\n| v |";
        let out = tg(md);
        // Openers are matched by prefix: `<code>` and `<code class="…">`
        // are both closed by `</code>`, and counting the bare form alone
        // would report the fenced one as unbalanced.
        for (open, close) in [("<b>", "</b>"), ("<i>", "</i>"), ("<s>", "</s>"), ("<code", "</code>"), ("<pre>", "</pre>"), ("<blockquote>", "</blockquote>"), ("<a ", "</a>")] {
            assert_eq!(
                out.matches(open).count(),
                out.matches(close).count(),
                "{open} unbalanced in {out:?}"
            );
        }
    }

    #[test]
    fn a_half_written_reply_still_renders() {
        // Streaming revises a message on partial markdown: an unclosed fence
        // and an unclosed emphasis are the normal mid-generation state, not an
        // error, and neither may produce broken markup.
        let out = tg("here is code:\n\n```rust\nfn main() {");
        assert!(out.contains("<pre><code"), "{out}");
        assert_eq!(out.matches("<pre>").count(), out.matches("</pre>").count());

        let out = tg("this is **half");
        assert_eq!(out.matches("<b>").count(), out.matches("</b>").count());
    }

    #[test]
    fn trailing_blank_lines_are_not_sent() {
        // Telegram refuses an empty message and pads a trailing newline into
        // visible whitespace.
        assert_eq!(tg("text\n\n\n"), "text");
        assert_eq!(tg(""), "");
    }
}
