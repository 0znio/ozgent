//! The layout: where the transcript ends and the furniture begins.
//!
//! Four regions, bottom-up:
//!
//! ```text
//!   transcript          everything said so far, scrollable
//!   prompt box          a bordered field that grows with what is typed
//!   permission bar      what tools may do — or the question being asked
//!   status bar          model, context, rate, sampler
//! ```
//!
//! The furniture has a fixed cost and the transcript gets what is left. On a
//! terminal too short to hold all of it the bars go first and the prompt is
//! kept, because a window you cannot type into is not a chat.

use ozgent_render::{Style, Theme, display_width};

/// How many rows the prompt box may grow to before it scrolls internally.
///
/// Five is a paragraph. Past that the box would be eating the conversation it
/// exists to add to.
pub const MAX_PROMPT_ROWS: usize = 5;

/// The rows each region occupies, top-down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// First row of the transcript, and how many rows it has.
    pub transcript: (usize, usize),
    /// First row of the prompt box, borders included.
    pub prompt: (usize, usize),
    /// `None` when the terminal is too short for it.
    pub permission: Option<usize>,
    pub status: Option<usize>,
    pub width: usize,
    pub height: usize,
}

impl Layout {
    /// Divide a terminal of `width` by `height`, given a prompt of `rows`.
    pub fn compute(width: usize, height: usize, prompt_rows: usize) -> Self {
        let prompt_rows = prompt_rows.clamp(1, MAX_PROMPT_ROWS);
        // Two for the box's own borders.
        let prompt_height = prompt_rows + 2;

        let mut remaining = height;
        // A bar is only taken when a row of transcript survives it: bars
        // describing a conversation you cannot see are furniture around
        // nothing. The status bar goes first because it is the one line that
        // is always true, so it is the last one given up.
        let claim = |remaining: &mut usize| {
            (*remaining > prompt_height + 1).then(|| {
                *remaining -= 1;
                *remaining
            })
        };
        let status = claim(&mut remaining);
        let permission = claim(&mut remaining);
        let prompt_start = remaining.saturating_sub(prompt_height);
        let prompt = (prompt_start, prompt_height.min(remaining));

        Self {
            transcript: (0, prompt_start),
            prompt,
            permission,
            status,
            width,
            height,
        }
    }

    /// Columns available inside the prompt box's borders.
    pub fn prompt_width(&self) -> usize {
        self.width.saturating_sub(4).max(1)
    }
}

/// Draw a bar that fills the width, in reverse video.
///
/// Reverse rather than a chosen colour: it inverts whatever palette the
/// terminal already has, so the bar is legible on a light background and a
/// dark one without ozgent guessing which it is looking at.
pub fn bar(theme: &Theme, text: &str, width: usize) -> String {
    let mut line = format!(" {text} ");
    let used = display_width(&line);
    if used < width {
        line.push_str(&" ".repeat(width - used));
    } else if used > width {
        line = truncate(&line, width);
    }
    if theme.enabled {
        format!("\x1b[7m{line}{}", Style::RESET)
    } else {
        line
    }
}

/// Cut a styled string to `width` display columns without splitting an escape.
pub fn truncate(text: &str, width: usize) -> String {
    let mut out = String::new();
    let mut used = 0;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Escapes cost no columns, so they are copied whole.
            out.push(c);
            for next in chars.by_ref() {
                out.push(next);
                if next != '[' && !next.is_ascii_digit() && next != ';' {
                    break;
                }
            }
            continue;
        }
        let w = display_width(&c.to_string());
        if used + w > width {
            break;
        }
        out.push(c);
        used += w;
    }
    out
}

/// The prompt box: a rounded border with the field inside it.
///
/// `marker` goes on the first row — the `›` that says where to type — and the
/// rows below it are indented to match, so a wrapped line reads as one field
/// rather than several.
pub fn prompt_box(
    theme: &Theme,
    lines: &[String],
    width: usize,
    marker: &str,
    accent: Style,
    hint: Option<&str>,
) -> Vec<String> {
    let inner = width.saturating_sub(2).max(1);
    let paint = |s: Style, text: &str| theme.style(s, text);
    let mut out = vec![paint(accent, &format!("╭{}╮", "─".repeat(inner)))];
    if let Some(hint) = hint.filter(|h| !h.is_empty() && h.len() + 6 <= inner) {
        // Set into the top border rather than given a row of its own. That
        // border is exactly the boundary between what has been said and where
        // you type, which is where "there is more below" belongs — and a row
        // that appears and disappears would shove the conversation about.
        let label = format!(" {hint} ");
        let lead = 2;
        let rest = inner - lead - display_width(&label);
        out[0] = format!(
            "{}{}{}",
            paint(accent, &format!("╭{}", "─".repeat(lead))),
            paint(Style::dim(), &label),
            paint(accent, &format!("{}╮", "─".repeat(rest))),
        );
    }

    for (i, line) in lines.iter().enumerate() {
        let lead = if i == 0 { marker } else { &" ".repeat(display_width(marker)) };
        let body = format!("{lead}{line}");
        let pad = inner.saturating_sub(display_width(&body) + 1);
        out.push(format!(
            "{}{}{}{}",
            paint(accent, "│"),
            format_args!(" {body}"),
            " ".repeat(pad),
            paint(accent, "│"),
        ));
    }
    out.push(paint(accent, &format!("╰{}╯", "─".repeat(inner))));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_regions_tile_the_screen_without_overlapping() {
        let l = Layout::compute(80, 30, 1);
        let (top, height) = l.transcript;
        assert_eq!(top, 0);
        assert_eq!(l.prompt.0, height, "the prompt starts where the transcript ends");
        assert_eq!(l.permission, Some(l.prompt.0 + l.prompt.1));
        assert_eq!(l.status, Some(l.permission.unwrap() + 1));
        assert_eq!(l.status.unwrap() + 1, 30, "the last row is the last row");
    }

    #[test]
    fn a_growing_prompt_takes_rows_from_the_transcript() {
        let one = Layout::compute(80, 30, 1);
        let four = Layout::compute(80, 30, 4);
        assert_eq!(four.transcript.1, one.transcript.1 - 3);
        assert_eq!(four.prompt.1, 6, "four rows plus two borders");
    }

    #[test]
    fn the_prompt_stops_growing_before_it_eats_the_conversation() {
        let huge = Layout::compute(80, 40, 50);
        assert_eq!(huge.prompt.1, MAX_PROMPT_ROWS + 2);
    }

    #[test]
    fn a_short_terminal_keeps_the_prompt_and_drops_the_bars() {
        // A window you cannot type into is not a chat.
        let tiny = Layout::compute(80, 4, 1);
        assert!(tiny.status.is_none() && tiny.permission.is_none());
        assert_eq!(tiny.prompt.1, 3, "the box keeps both its borders");
        assert_eq!(tiny.transcript.1, 1);
    }

    #[test]
    fn the_status_bar_outlives_the_permission_bar() {
        // Given one spare row, the line that is always true wins it.
        let l = Layout::compute(80, 5, 1);
        assert_eq!(l.status, Some(4));
        assert!(l.permission.is_none());
        assert_eq!(l.transcript.1, 1, "a bar must not cost the last visible line");
    }

    #[test]
    fn a_bar_never_costs_the_last_row_of_transcript() {
        // Bars describing a conversation you cannot see are furniture around
        // nothing.
        for height in 3..12 {
            let l = Layout::compute(80, height, 1);
            let bars = l.status.is_some() as usize + l.permission.is_some() as usize;
            assert!(
                bars == 0 || l.transcript.1 >= 1,
                "height {height} spent {bars} rows on bars with no transcript",
            );
        }
    }

    #[test]
    fn a_bar_fills_the_width_exactly() {
        let plain = bar(&Theme::plain(), "model · 4k/32k", 40);
        assert_eq!(display_width(&plain), 40);
    }

    #[test]
    fn a_bar_too_long_for_the_screen_is_cut_not_wrapped() {
        // Wrapping would push every row below it down by one, every frame.
        let plain = bar(&Theme::plain(), &"x".repeat(200), 40);
        assert_eq!(display_width(&plain), 40);
    }

    #[test]
    fn colour_does_not_change_a_bar_width() {
        let styled = bar(&Theme::default(), "model", 40);
        assert_eq!(display_width(&styled), 40);
    }

    #[test]
    fn truncating_keeps_escapes_whole() {
        let cut = truncate("\x1b[31mredtext\x1b[0m", 3);
        assert_eq!(display_width(&cut), 3);
        assert!(cut.starts_with("\x1b[31m"), "the colour must not be cut in half: {cut:?}");
    }

    #[test]
    fn the_prompt_box_is_rectangular() {
        let lines = prompt_box(
            &Theme::plain(),
            &["hello".to_string(), "world".to_string()],
            40,
            "› ",
            Style::default(),
            None,
        );
        let widths: Vec<usize> = lines.iter().map(|l| display_width(l)).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{widths:?}\n{lines:#?}");
        assert_eq!(widths[0], 40);
    }

    #[test]
    fn only_the_first_row_wears_the_marker() {
        let lines = prompt_box(
            &Theme::plain(),
            &["first".to_string(), "second".to_string()],
            40,
            "› ",
            Style::default(),
            None,
        );
        assert!(lines[1].contains("› first"));
        assert!(lines[2].contains("  second"), "a wrapped row is indented, not re-marked");
    }

    #[test]
    fn an_empty_prompt_still_draws_a_box() {
        let lines = prompt_box(&Theme::plain(), &[String::new()], 30, "› ", Style::default(), None);
        assert_eq!(lines.len(), 3);
        assert!(lines.iter().all(|l| display_width(l) == 30));
    }

    #[test]
    fn a_hint_sits_in_the_border_without_changing_the_shape() {
        let plain = prompt_box(&Theme::plain(), &["hi".into()], 60, "› ", Style::default(), None);
        let hinted = prompt_box(
            &Theme::plain(),
            &["hi".into()],
            60,
            "› ",
            Style::default(),
            Some("↓ 12 more"),
        );
        assert_eq!(hinted.len(), plain.len(), "a hint must not cost a row");
        assert!(hinted[0].contains("↓ 12 more"));
        let widths: Vec<usize> = hinted.iter().map(|l| display_width(l)).collect();
        assert!(widths.iter().all(|w| *w == 60), "{widths:?}\n{hinted:#?}");
    }

    #[test]
    fn a_hint_too_long_for_the_border_is_left_out_rather_than_bursting_it() {
        let narrow = prompt_box(
            &Theme::plain(),
            &["hi".into()],
            20,
            "› ",
            Style::default(),
            Some("a hint far too long for this border"),
        );
        let widths: Vec<usize> = narrow.iter().map(|l| display_width(l)).collect();
        assert!(widths.iter().all(|w| *w == 20), "{widths:?}");
    }

    #[test]
    fn a_coloured_box_is_the_same_shape_as_a_plain_one() {
        let plain = prompt_box(&Theme::plain(), &["hi".into()], 40, "› ", Style::default(), None);
        let styled = prompt_box(&Theme::default(), &["hi".into()], 40, "› ", Style::dim(), None);
        let a: Vec<usize> = plain.iter().map(|l| display_width(l)).collect();
        let b: Vec<usize> = styled.iter().map(|l| display_width(l)).collect();
        assert_eq!(a, b);
    }
}
