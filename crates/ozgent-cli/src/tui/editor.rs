//! The input line.
//!
//! A full-screen application owns its keystrokes, so the line editing rustyline
//! used to provide has to exist here instead. This is deliberately not a
//! general editor: it is one field that can hold several lines, with the
//! emacs bindings people's fingers already know and a history that survives
//! restarts.

/// Where the cursor is, in rows and columns of the box.
pub struct Caret {
    pub row: usize,
    pub col: usize,
}

pub struct Editor {
    text: String,
    /// Byte offset of the cursor. Bytes rather than characters because every
    /// operation here is a slice, and a character index would have to be
    /// converted at each one.
    cursor: usize,
    history: Vec<String>,
    /// Position while walking history. `None` means editing the live line,
    /// which is kept aside so walking back and forward again restores it.
    browsing: Option<usize>,
    stashed: String,
}

impl Editor {
    pub fn new(history: Vec<String>) -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            history,
            browsing: None,
            stashed: String::new(),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Byte offset of the cursor.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Replace `start..end` with `with` and put the cursor after it.
    ///
    /// How a suggestion is accepted: the partial `@sto` is swapped for the
    /// whole name in one step, so a single undo-free edit cannot leave half
    /// of each behind.
    pub fn replace(&mut self, start: usize, end: usize, with: &str) {
        let end = end.min(self.text.len());
        let start = start.min(end);
        self.text.replace_range(start..end, with);
        self.cursor = start + with.len();
        self.browsing = None;
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.browsing = None;
    }

    /// Take the line and record it in history.
    pub fn take(&mut self) -> String {
        let text = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.browsing = None;
        // Neither blanks nor an immediate repeat: both make the history
        // tedious to walk without recording anything the user wanted back.
        if !text.trim().is_empty() && self.history.last().map(String::as_str) != Some(text.as_str())
        {
            self.history.push(text.clone());
        }
        text
    }

    pub fn insert(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    pub fn insert_str(&mut self, s: &str) {
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    pub fn backspace(&mut self) {
        if let Some(prev) = self.prev_boundary() {
            self.text.replace_range(prev..self.cursor, "");
            self.cursor = prev;
        }
    }

    pub fn delete(&mut self) {
        if let Some(next) = self.next_boundary() {
            self.text.replace_range(self.cursor..next, "");
        }
    }

    pub fn left(&mut self) {
        if let Some(prev) = self.prev_boundary() {
            self.cursor = prev;
        }
    }

    pub fn right(&mut self) {
        if let Some(next) = self.next_boundary() {
            self.cursor = next;
        }
    }

    /// To the start of the current visual line, not of the whole field.
    pub fn home(&mut self) {
        self.cursor = self.line_start();
    }

    pub fn end(&mut self) {
        self.cursor = self.line_end();
    }

    /// Delete to the end of the line, emacs-style.
    pub fn kill_to_end(&mut self) {
        let end = self.line_end();
        self.text.replace_range(self.cursor..end, "");
    }

    pub fn kill_to_start(&mut self) {
        let start = self.line_start();
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Delete the word before the cursor.
    pub fn kill_word(&mut self) {
        let head = &self.text[..self.cursor];
        let trimmed = head.trim_end_matches(char::is_whitespace);
        let start = trimmed
            .rfind(char::is_whitespace)
            .map(|i| i + trimmed[i..].chars().next().map_or(1, char::len_utf8))
            .unwrap_or(0);
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Step back through history, keeping the half-typed line to come back to.
    pub fn previous(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.browsing {
            None => {
                self.stashed = self.text.clone();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.browsing = Some(next);
        self.text = self.history[next].clone();
        self.cursor = self.text.len();
    }

    pub fn next(&mut self) {
        let Some(i) = self.browsing else { return };
        if i + 1 < self.history.len() {
            self.browsing = Some(i + 1);
            self.text = self.history[i + 1].clone();
        } else {
            self.browsing = None;
            self.text = std::mem::take(&mut self.stashed);
        }
        self.cursor = self.text.len();
    }

    /// How the field lays out in a box `width` columns wide.
    ///
    /// Returns the visual lines and where the caret sits among them, which the
    /// frame needs both to draw the box and to park the terminal cursor.
    pub fn layout(&self, width: usize) -> (Vec<String>, Caret) {
        let width = width.max(1);
        let mut lines = Vec::new();
        let mut caret = Caret { row: 0, col: 0 };
        let mut offset = 0;

        for segment in self.text.split('\n') {
            // A segment wraps into as many rows as it needs; an empty one
            // still occupies a row, or a trailing newline would vanish.
            let mut row_text = String::new();
            let mut row_width = 0;
            let mut first = true;
            for c in segment.chars() {
                let w = ozgent_render::display_width(&c.to_string());
                if row_width + w > width {
                    lines.push(std::mem::take(&mut row_text));
                    row_width = 0;
                    first = false;
                }
                if offset == self.cursor {
                    caret = Caret { row: lines.len(), col: row_width };
                }
                row_text.push(c);
                row_width += w;
                offset += c.len_utf8();
            }
            let _ = first;
            if offset == self.cursor {
                caret = Caret { row: lines.len(), col: row_width };
            }
            lines.push(row_text);
            // The newline itself.
            offset += 1;
        }
        (lines, caret)
    }

    fn prev_boundary(&self) -> Option<usize> {
        self.text[..self.cursor].chars().next_back().map(|c| self.cursor - c.len_utf8())
    }

    fn next_boundary(&self) -> Option<usize> {
        self.text[self.cursor..].chars().next().map(|c| self.cursor + c.len_utf8())
    }

    fn line_start(&self) -> usize {
        self.text[..self.cursor].rfind('\n').map(|i| i + 1).unwrap_or(0)
    }

    fn line_end(&self) -> usize {
        self.text[self.cursor..]
            .find('\n')
            .map(|i| self.cursor + i)
            .unwrap_or(self.text.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(text: &str) -> Editor {
        let mut e = Editor::new(Vec::new());
        e.insert_str(text);
        e
    }

    #[test]
    fn typing_and_backspacing_agree_on_where_the_cursor_is() {
        let mut e = typed("hello");
        e.backspace();
        assert_eq!(e.text(), "hell");
        e.insert('o');
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn editing_in_the_middle_inserts_where_the_cursor_is() {
        let mut e = typed("helo");
        e.left();
        e.insert('l');
        assert_eq!(e.text(), "hello");
    }

    #[test]
    fn multibyte_characters_are_one_step_not_several() {
        // Stepping by bytes would land inside a character and panic on the
        // next slice.
        let mut e = typed("héllo");
        e.home();
        e.right();
        e.right();
        e.backspace();
        assert_eq!(e.text(), "hllo");
    }

    #[test]
    fn emoji_do_not_break_deletion() {
        let mut e = typed("a🙂b");
        e.left();
        e.backspace();
        assert_eq!(e.text(), "ab");
    }

    #[test]
    fn home_and_end_work_per_line_not_per_field() {
        let mut e = typed("first\nsecond");
        e.home();
        e.insert('X');
        assert_eq!(e.text(), "first\nXsecond");
        e.end();
        e.insert('Y');
        assert_eq!(e.text(), "first\nXsecondY");
    }

    #[test]
    fn killing_to_the_end_stops_at_the_newline() {
        let mut e = typed("keep\ndrop this");
        e.home();
        e.kill_to_end();
        assert_eq!(e.text(), "keep\n");
    }

    #[test]
    fn killing_a_word_takes_the_trailing_space_with_it() {
        let mut e = typed("one two three ");
        e.kill_word();
        assert_eq!(e.text(), "one two ");
        e.kill_word();
        assert_eq!(e.text(), "one ");
    }

    #[test]
    fn history_walks_back_and_returns_what_was_being_typed() {
        let mut e = Editor::new(vec!["first".into(), "second".into()]);
        e.insert_str("half typed");
        e.previous();
        assert_eq!(e.text(), "second");
        e.previous();
        assert_eq!(e.text(), "first");
        e.next();
        assert_eq!(e.text(), "second");
        e.next();
        assert_eq!(e.text(), "half typed", "the unfinished line must come back");
    }

    #[test]
    fn taking_a_line_records_it_once() {
        let mut e = typed("hello");
        assert_eq!(e.take(), "hello");
        e.insert_str("hello");
        e.take();
        assert_eq!(e.history(), ["hello"], "an immediate repeat is noise");
    }

    #[test]
    fn a_blank_line_is_not_history() {
        let mut e = typed("   ");
        e.take();
        assert!(e.history().is_empty());
    }

    #[test]
    fn the_caret_lands_where_the_text_says_it_should() {
        let e = typed("hello");
        let (lines, caret) = e.layout(40);
        assert_eq!(lines, ["hello"]);
        assert_eq!((caret.row, caret.col), (0, 5));
    }

    #[test]
    fn a_long_line_wraps_and_the_caret_follows_it() {
        let e = typed(&"x".repeat(25));
        let (lines, caret) = e.layout(10);
        assert_eq!(lines.len(), 3);
        assert_eq!(caret.row, 2, "the caret is on the last row");
        assert_eq!(caret.col, 5);
    }

    #[test]
    fn an_explicit_newline_makes_a_row_of_its_own() {
        let e = typed("a\n\nb");
        let (lines, _) = e.layout(40);
        assert_eq!(lines, ["a", "", "b"], "a blank line must not vanish");
    }

    #[test]
    fn the_caret_at_the_start_is_at_the_origin() {
        let mut e = typed("hello");
        e.home();
        let (_, caret) = e.layout(40);
        assert_eq!((caret.row, caret.col), (0, 0));
    }

    #[test]
    fn an_empty_field_still_has_one_row() {
        // The box has to draw something, and zero rows would collapse it.
        let e = Editor::new(Vec::new());
        let (lines, caret) = e.layout(40);
        assert_eq!(lines, [""]);
        assert_eq!((caret.row, caret.col), (0, 0));
    }

    #[test]
    fn replacing_a_partial_word_leaves_the_cursor_after_the_replacement() {
        let mut e = typed("ask @sto now");
        // Cursor after "@sto".
        for _ in 0..4 {
            e.left();
        }
        let at = e.text().find('@').unwrap();
        e.replace(at, e.cursor(), "@stock-guru ");
        assert_eq!(e.text(), "ask @stock-guru  now");
        assert_eq!(&e.text()[..e.cursor()], "ask @stock-guru ");
    }

    #[test]
    fn clearing_leaves_nothing_behind() {
        let mut e = typed("something");
        e.clear();
        assert!(e.is_empty());
        let (_, caret) = e.layout(40);
        assert_eq!((caret.row, caret.col), (0, 0));
    }
}
