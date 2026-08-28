//! Asking the person at the terminal whether a tool call may run.
//!
//! The decision itself lives in `ozgent_core::permission`, which the web
//! interface consults too. What is here is only the asking: a box showing the
//! tool, what it does, and the exact arguments, and four answers.
//!
//! The arguments are the point. "Allow run_command?" is a question nobody can
//! answer, because the risk is entirely in the string being run; a prompt that
//! hides it trains people to approve without reading, which is worse than no
//! prompt at all.

use std::io::{BufRead, IsTerminal, Write};

use ozgent_core::permission::{Choice, Effect};
use ozgent_render::{Color, Style, Theme, display_width};

/// One line inside the box, as styled pieces.
///
/// Kept as pieces rather than a rendered string because the box has to be
/// able to cut a line that is too long, and cutting a string that already
/// contains escape sequences means cutting one in half.
type Row = Vec<(Style, String)>;

fn row_width(row: &Row) -> usize {
    row.iter().map(|(_, text)| display_width(text)).sum()
}

/// Cut a row to `width` columns, marking the cut.
fn truncate_row(row: &Row, width: usize) -> Row {
    if row_width(row) <= width {
        return row.clone();
    }
    let mut out: Row = Vec::new();
    let mut used = 0;
    // One column is kept back for the ellipsis, so the mark itself cannot be
    // what overflows the border.
    let budget = width.saturating_sub(1);
    for (style, text) in row {
        if used >= budget {
            break;
        }
        let room = budget - used;
        if display_width(text) <= room {
            used += display_width(text);
            out.push((*style, text.clone()));
        } else {
            let mut cut = String::new();
            for c in text.chars() {
                let w = display_width(&c.to_string());
                if used + w > room {
                    break;
                }
                used += w;
                cut.push(c);
            }
            out.push((*style, cut));
            break;
        }
    }
    out.push((Style::dim(), "…".to_string()));
    out
}

/// The question, rendered but not yet asked.
///
/// Built separately from the reading so the layout can be tested without a
/// terminal — the part that goes wrong silently is the box, not the `match`.
pub fn render(
    theme: &Theme,
    width: usize,
    tool: &str,
    effect: Effect,
    arguments: &serde_json::Value,
) -> String {
    let accent = match effect {
        // The colour carries the same information as the word, for the glance
        // before the sentence is read.
        Effect::Execute => Style::color(Color::Red),
        Effect::Write => Style::color(Color::Yellow),
        Effect::Read | Effect::Unknown => Style::color(Color::Cyan),
    };
    let bold = Style { bold: true, ..Default::default() };
    let plain = Style::default();

    let mut rows: Vec<Row> = vec![
        vec![
            (bold, tool.to_string()),
            (Style::dim(), format!(" · {}", effect.describes())),
        ],
        Vec::new(),
    ];
    // Values are elided against a generous width first; the box then cuts
    // whatever is still too wide for the terminal actually in use.
    for line in argument_lines(arguments, 88) {
        rows.push(vec![(Style::dim(), format!("  {line}"))]);
    }
    rows.push(Vec::new());
    rows.push(vec![(Style::dim(), "Run it?".to_string())]);
    for (key, label) in options(tool) {
        rows.push(vec![(accent, format!("  {key} ")), (plain, label)]);
    }

    // Wide enough for the longest line it has, narrow enough to sit inside the
    // terminal with room for the border.
    let longest = rows.iter().map(row_width).max().unwrap_or(0);
    let inner = longest.min(width.saturating_sub(4)).max(8);

    let paint = |s: Style, text: &str| theme.style(s, text);
    let mut out = String::new();
    out.push_str(&paint(accent, &format!("╭{}╮\n", "─".repeat(inner + 2))));
    for row in &rows {
        let row = truncate_row(row, inner);
        out.push_str(&paint(accent, "│ "));
        for (style, text) in &row {
            out.push_str(&paint(*style, text));
        }
        out.push_str(&" ".repeat(inner.saturating_sub(row_width(&row))));
        out.push_str(&paint(accent, " │\n"));
    }
    out.push_str(&paint(accent, &format!("╰{}╯", "─".repeat(inner + 2))));
    out
}

/// The four answers, in the order they are offered.
///
/// Four rather than Claude Code's three because ozgent's "don't ask again"
/// has two honest meanings — until you quit, and for good — and silently
/// picking one of them is how a moment of convenience becomes a setting the
/// user never chose.
fn options(tool: &str) -> [(String, String); 4] {
    [
        ("1.".into(), "Yes".into()),
        ("2.".into(), "Yes, and don't ask again this session".into()),
        ("3.".into(), format!("Yes, and always allow {tool}")),
        ("4.".into(), "No".into()),
    ]
}

/// Break the arguments into `key: value` lines that fit `width`.
///
/// JSON braces and quotes carry nothing the reader needs, and a long value —
/// a file's contents, a page of text — is elided in the middle: the beginning
/// says what it is and the end says where it stops, while the middle is what
/// scrolls the question off the screen.
fn argument_lines(arguments: &serde_json::Value, width: usize) -> Vec<String> {
    let Some(object) = arguments.as_object() else {
        return vec![elide(&arguments.to_string(), width)];
    };
    if object.is_empty() {
        return vec!["(no arguments)".to_string()];
    }
    object
        .iter()
        .map(|(key, value)| {
            let text = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let room = width.saturating_sub(display_width(key) + 2);
            format!("{key}: {}", elide(&text.replace('\n', "⏎"), room))
        })
        .collect()
}

fn elide(text: &str, width: usize) -> String {
    if display_width(text) <= width || width < 8 {
        return text.to_string();
    }
    let head: String = text.chars().take(width * 2 / 3).collect();
    let tail: String = {
        let keep = width / 3 - 1;
        let all: Vec<char> = text.chars().collect();
        all[all.len().saturating_sub(keep)..].iter().collect()
    };
    format!("{head}…{tail}")
}

/// Read one answer, mapping anything unrecognised to a refusal.
///
/// Nothing here loops asking again: a model can produce a call every second,
/// and a prompt that will not take no for an answer is one people escape with
/// Ctrl-C, which loses the conversation.
pub fn read_choice(line: &str) -> Choice {
    match line.trim().to_ascii_lowercase().as_str() {
        // Enter accepts, which is the answer being given nine times in ten and
        // the reason the box is unmissable.
        "" | "1" | "y" | "yes" => Choice::Once,
        "2" | "s" | "session" => Choice::Session,
        "3" | "a" | "always" => Choice::Always,
        _ => Choice::Deny,
    }
}

/// Draw the box and wait for an answer.
///
/// Off a terminal — a pipe, a script, `ozgent run` in a cron job — there is
/// nobody to ask, so the call is refused and the model is told. Silently
/// running it would make a permission prompt something you can defeat by
/// redirecting stdin.
pub fn ask(
    theme: &Theme,
    tool: &str,
    effect: Effect,
    arguments: &serde_json::Value,
) -> Choice {
    if !std::io::stdin().is_terminal() {
        return Choice::Deny;
    }
    let width = ozgent_render::terminal_width();
    eprintln!();
    eprintln!("{}", render(theme, width, tool, effect, arguments));
    eprint!("{}", theme.style(Style::dim(), "  [1] "));
    let _ = std::io::stderr().flush();

    let mut line = String::new();
    match std::io::stdin().lock().read_line(&mut line) {
        // End of input is not consent.
        Ok(0) | Err(_) => Choice::Deny,
        Ok(_) => read_choice(&line),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plain(args: &serde_json::Value) -> String {
        render(&Theme::plain(), 80, "run_command", Effect::Execute, args)
    }

    #[test]
    fn the_box_shows_the_arguments_not_just_the_tool() {
        // The whole risk of `run_command` is the string being run. A prompt
        // that hides it teaches people to approve without reading.
        let out = plain(&json!({ "command": "rm -rf /tmp/build" }));
        assert!(out.contains("run_command"), "{out}");
        assert!(out.contains("rm -rf /tmp/build"), "{out}");
    }

    #[test]
    fn every_line_of_the_box_is_the_same_width() {
        let out = plain(&json!({ "command": "cargo test", "cwd": "/home/me/code" }));
        let widths: Vec<usize> = out.lines().map(display_width).collect();
        assert!(
            widths.windows(2).all(|w| w[0] == w[1]),
            "ragged box: {widths:?}\n{out}",
        );
    }

    #[test]
    fn a_wide_argument_does_not_burst_the_box() {
        let out = plain(&json!({ "command": "x".repeat(400) }));
        let widths: Vec<usize> = out.lines().map(display_width).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{widths:?}");
        assert!(out.contains('…'), "a long value should be elided");
    }

    #[test]
    fn a_multiline_argument_stays_on_one_row() {
        // A newline inside a value would otherwise break out of the border.
        let out = plain(&json!({ "content": "line one\nline two" }));
        let widths: Vec<usize> = out.lines().map(display_width).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{widths:?}\n{out}");
    }

    #[test]
    fn wide_characters_do_not_skew_the_border() {
        let out = render(&Theme::plain(), 80, "write_file", Effect::Write, &json!({ "path": "日本語のファイル" }));
        let widths: Vec<usize> = out.lines().map(display_width).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{widths:?}\n{out}");
    }

    #[test]
    fn colour_does_not_change_the_geometry() {
        let coloured = render(&Theme::default(), 80, "run_command", Effect::Execute, &json!({ "command": "ls" }));
        let widths: Vec<usize> = coloured.lines().map(display_width).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{widths:?}");
    }

    #[test]
    fn a_call_with_no_arguments_says_so() {
        let out = plain(&json!({}));
        assert!(out.contains("(no arguments)"), "{out}");
    }

    #[test]
    fn the_always_option_names_the_tool_it_would_allow() {
        // "Always allow" without a name is a promise nobody can evaluate.
        let out = plain(&json!({ "command": "ls" }));
        assert!(out.contains("always allow run_command"), "{out}");
    }

    #[test]
    fn enter_accepts_and_anything_unrecognised_refuses() {
        assert_eq!(read_choice(""), Choice::Once);
        assert_eq!(read_choice("1\n"), Choice::Once);
        assert_eq!(read_choice("2"), Choice::Session);
        assert_eq!(read_choice("3"), Choice::Always);
        assert_eq!(read_choice("4"), Choice::Deny);
        assert_eq!(read_choice("no"), Choice::Deny);
        assert_eq!(read_choice("what?"), Choice::Deny, "an unclear answer is not consent");
    }

    #[test]
    fn a_narrow_terminal_still_produces_a_box() {
        let out = render(&Theme::plain(), 30, "write_file", Effect::Write, &json!({ "path": "/a/b/c.txt" }));
        let widths: Vec<usize> = out.lines().map(display_width).collect();
        assert!(widths.windows(2).all(|w| w[0] == w[1]), "{widths:?}\n{out}");
    }
}
