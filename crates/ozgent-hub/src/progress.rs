//! A download progress bar.
//!
//! Model files are gigabytes, so a download without feedback is
//! indistinguishable from a hang. The bar reports the three things that answer
//! "should I wait?": how far along, how fast, and how much longer.
//!
//! Two things decide the shape of this file.
//!
//! **The line must fit the terminal.** A bar redrawn with `\r` overwrites the
//! current row; a line wider than the terminal wraps, and `\r` then returns to
//! the start of the *last* row rather than the first. Every redraw leaves the
//! previous row behind and the screen scrolls — which looked like a broken
//! downloader and was really a line four characters too long. So the layout is
//! computed against the terminal width and drops fields, in order of how much
//! they matter, until it fits.
//!
//! **Colour is used the way the rest of ozgent uses it.** The eight ANSI names
//! rather than fixed RGB, so it follows whatever palette the terminal already
//! has; and the accent marks the thing that is *changing* — the spinner, the
//! filled part of the bar, the rate — never the chrome around it.

use std::time::{Duration, Instant};

/// How often the line is redrawn.
///
/// On a fast link a write per chunk would cost more than the transfer. Fast
/// enough that the spinner reads as motion, slow enough that the digits can be
/// read.
const REDRAW_INTERVAL: Duration = Duration::from_millis(80);

/// Length of the window used to measure speed.
const RATE_WINDOW: Duration = Duration::from_millis(500);

/// Weight given to a new rate sample. Low enough that a stalled chunk does not
/// make the estimate lurch.
const SMOOTHING: f64 = 0.3;

/// Braille frames. They occupy one cell in every terminal font that has them,
/// which the block and clock spinners do not.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Eighths, so the bar advances smoothly rather than in whole cells. On a
/// 2 GB file a whole cell is thirty seconds of apparently nothing happening.
const EIGHTHS: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
/// Yellow, which is the terminal's own accent. Reserved for state that is
/// changing, exactly as amber is in the web interface.
const ACCENT: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";

/// Tracks a single file's transfer and renders one line for it.
pub struct Bar {
    name: String,
    total: u64,
    started: Instant,
    /// Start of the current speed window.
    window_start: Instant,
    window_bytes: u64,
    /// Smoothed rate in bytes per second.
    rate: f64,
    last_draw: Option<Instant>,
    /// Terminal width. The whole layout is derived from it.
    columns: usize,
    colour: bool,
    frame: usize,
}

impl Bar {
    /// `columns` is the terminal width; `colour` whether to emit escapes.
    pub fn new(name: impl Into<String>, total: u64, columns: usize, colour: bool) -> Self {
        let now = Instant::now();
        Self {
            name: name.into(),
            total,
            started: now,
            window_start: now,
            window_bytes: 0,
            rate: 0.0,
            last_draw: None,
            columns: columns.max(20),
            colour,
            frame: 0,
        }
    }

    /// The expected size, or 0 when the server did not report one.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Follow a terminal that was resized mid-download.
    pub fn set_columns(&mut self, columns: usize) {
        self.columns = columns.max(20);
    }

    /// Record progress and return a line to draw, or `None` if it is too soon.
    pub fn update(&mut self, done: u64) -> Option<String> {
        let now = Instant::now();

        // Measured over a window rather than per chunk: chunk sizes vary
        // wildly and an instantaneous rate is unreadable.
        let elapsed = now.duration_since(self.window_start);
        if elapsed >= RATE_WINDOW {
            let sample = (done.saturating_sub(self.window_bytes)) as f64 / elapsed.as_secs_f64();
            self.rate = if self.rate == 0.0 {
                sample
            } else {
                self.rate * (1.0 - SMOOTHING) + sample * SMOOTHING
            };
            self.window_start = now;
            self.window_bytes = done;
        }

        let due = self
            .last_draw
            .is_none_or(|last| now.duration_since(last) >= REDRAW_INTERVAL);
        let finished = self.total > 0 && done >= self.total;
        if !due && !finished {
            return None;
        }
        self.last_draw = Some(now);
        self.frame = self.frame.wrapping_add(1);
        Some(self.render(done))
    }

    /// The final line, with the average rate over the whole transfer.
    pub fn finish(&self, done: u64) -> String {
        let seconds = self.started.elapsed().as_secs_f64().max(0.001);
        let tick = self.paint(GREEN, "✓");
        let line = format!(
            "  {tick} {}  {} in {} ({}/s)",
            self.paint(BOLD, &self.name),
            bytes(done),
            duration(self.started.elapsed()),
            bytes((done as f64 / seconds) as u64),
        );
        line
    }

    fn paint(&self, code: &str, text: &str) -> String {
        if !self.colour || text.is_empty() {
            return text.to_string();
        }
        format!("{code}{text}{RESET}")
    }

    /// Build the line, dropping fields until it fits the terminal.
    ///
    /// The order is deliberate: the estimate goes first because it is a guess,
    /// then the total because the percentage already says where you are, then
    /// the bar because the percentage says that too. What survives to the
    /// narrowest terminal is the name, the percentage and the rate.
    fn render(&self, done: u64) -> String {
        let spinner = if self.total > 0 && done >= self.total {
            self.paint(GREEN, "✓")
        } else {
            self.paint(ACCENT, SPINNER[self.frame % SPINNER.len()])
        };

        if self.total == 0 {
            // Length unknown: report what has arrived rather than a fake bar.
            let text = format!("{} {} ({}/s)", bytes(done), self.name, bytes(self.rate as u64));
            return format!("  {spinner} {}", clip(&text, self.columns.saturating_sub(4)));
        }

        let fraction = (done as f64 / self.total as f64).clamp(0.0, 1.0);
        let percent = format!("{:>3.0}%", fraction * 100.0);
        let rate = format!("{}/s", bytes(self.rate as u64));
        let eta = self.eta(done);
        let done_s = bytes(done);
        let total_s = bytes(self.total);

        // Measured on the plain text, then rebuilt with colour. The two must
        // agree, which is why both are driven from the same `level`.
        //
        // A bar is worth more than the estimate beside it — it is the thing
        // being looked at — so the richest layout that still leaves room for
        // one wins, and only if none does are the fields kept without it.
        let attempt = |level: usize, want_bar: bool, want_name: bool| -> Option<(String, usize, usize)> {
            let tail = plain_tail(level, &percent, &done_s, &total_s, &rate, &eta);
            // "  " + spinner + " " + name ... "  " + tail
            let base = 2 + 1 + 1 + 2 + tail.chars().count();

            // Below about six characters a name is "…uf", which identifies
            // nothing; the space is better spent on the bar.
            let room_for_name = self.columns.saturating_sub(base).min(28);
            let name =
                if room_for_name >= 6 { clip(&self.name, room_for_name) } else { String::new() };
            let used = base + name.chars().count();

            let room = self.columns.saturating_sub(used + 2);
            let width = if room >= 8 { room.min(32) } else { 0 };
            if want_bar && width == 0 {
                return None;
            }
            if want_name && name.is_empty() {
                return None;
            }
            let total_width = used + if width > 0 { width + 2 } else { 0 };
            (total_width <= self.columns).then_some((name, width, level))
        };

        // What to give up, in order: nothing; then the bar; then the name;
        // then both. Which file is downloading matters more than how much of
        // it is left, so the name outlives the total and the estimate.
        let (name, width, level) = [(true, true), (false, true), (true, false), (false, false)]
            .into_iter()
            .find_map(|(bar, named)| (0..4).find_map(|level| attempt(level, bar, named)))
            // The narrowest layout is a percentage and a rate with no name,
            // which fits any terminal this bar accepts.
            .unwrap_or((String::new(), 0, 3));

        let mut line = format!("  {spinner}");
        if !name.is_empty() {
            line.push(' ');
            line.push_str(&self.paint(BOLD, &name));
        }
        if width > 0 {
            line.push_str("  ");
            line.push_str(&self.bar(fraction, width));
        }
        line.push_str("  ");
        line.push_str(&self.tail(level, &percent, &done_s, &total_s, &rate, &eta));
        line
    }

    /// The right-hand side, coloured. Kept beside `render`'s plain-text
    /// measurement of the same thing, which is why both take the same pieces.
    fn tail(
        &self,
        level: usize,
        percent: &str,
        done: &str,
        total: &str,
        rate: &str,
        eta: &str,
    ) -> String {
        let percent = self.paint(BOLD, percent);
        let rate = self.paint(ACCENT, rate);
        match level {
            0 => format!(
                "{percent}  {done} {} {}  {rate}  {}",
                self.paint(DIM, "/"),
                self.paint(DIM, total),
                self.paint(DIM, eta)
            ),
            1 => format!("{percent}  {done} {} {}  {rate}", self.paint(DIM, "/"), self.paint(DIM, total)),
            2 => format!("{percent}  {done}  {rate}"),
            _ => format!("{percent}  {rate}"),
        }
    }

    /// A bar of `width` cells, filled to `fraction`, to an eighth of a cell.
    fn bar(&self, fraction: f64, width: usize) -> String {
        let eighths = (fraction * (width * 8) as f64).round() as usize;
        let full = eighths / 8;
        let rest = eighths % 8;

        let mut filled = "█".repeat(full.min(width));
        if full < width && rest > 0 {
            filled.push_str(EIGHTHS[rest]);
        }
        let drawn = full.min(width) + usize::from(full < width && rest > 0);
        let empty = "─".repeat(width.saturating_sub(drawn));

        format!("{}{}", self.paint(ACCENT, &filled), self.paint(DIM, &empty))
    }

    fn eta(&self, done: u64) -> String {
        if self.rate <= 0.0 || done >= self.total {
            return "--:--".into();
        }
        let remaining = (self.total - done) as f64 / self.rate;
        duration(Duration::from_secs_f64(remaining))
    }
}

/// What [`Bar::tail`] will render, without the colour. Kept next to it: if
/// the two disagree the line is measured wrong and wraps, which is the exact
/// failure this module exists to avoid.
fn plain_tail(
    level: usize,
    percent: &str,
    done: &str,
    total: &str,
    rate: &str,
    eta: &str,
) -> String {
    match level {
        0 => format!("{percent}  {done} / {total}  {rate}  {eta}"),
        1 => format!("{percent}  {done} / {total}  {rate}"),
        2 => format!("{percent}  {done}  {rate}"),
        _ => format!("{percent}  {rate}"),
    }
}

/// Shorten to `width`, with an ellipsis, keeping the end of the name.
///
/// The end is what distinguishes `…-Q4_K_M.gguf` from `…-Q8_0.gguf`; the
/// beginning is the part every file in a repository shares.
fn clip(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if width == 0 {
        return String::new();
    }
    if len <= width {
        return text.to_string();
    }
    if width == 1 {
        return "…".into();
    }
    let tail: String = text.chars().skip(len - (width - 1)).collect();
    format!("…{tail}")
}

/// Human-readable byte count.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else if v >= 100.0 {
        format!("{v:.0} {}", UNITS[u])
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// `m:ss`, or `h:mm:ss` past an hour.
pub fn duration(d: Duration) -> String {
    let total = d.as_secs();
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// The printable width of a line, ignoring SGR escapes.
pub fn visible_width(s: &str) -> usize {
    let mut width = 0;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip to the end of the escape sequence.
            for e in chars.by_ref() {
                if e.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        width += 1;
    }
    width
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(name: &str, total: u64, columns: usize) -> Bar {
        Bar::new(name, total, columns, false)
    }

    #[test]
    fn byte_counts_are_readable() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1024), "1.0 KB");
        assert_eq!(bytes(7_300_000_000), "6.8 GB");
        assert_eq!(bytes(300 * 1024 * 1024), "300 MB");
    }

    #[test]
    fn durations_grow_a_field_past_an_hour() {
        assert_eq!(duration(Duration::from_secs(9)), "0:09");
        assert_eq!(duration(Duration::from_secs(75)), "1:15");
        assert_eq!(duration(Duration::from_secs(3725)), "1:02:05");
    }

    // ------------------------------------------------- the bug this fixes

    #[test]
    fn a_line_never_exceeds_the_terminal() {
        // The whole reason this file was rewritten. A line one column too wide
        // wraps; `\r` then returns to the start of the wrapped row, the
        // previous row is left behind, and the terminal scrolls a screenful of
        // progress bars. It looks like a broken downloader.
        for columns in [20, 40, 60, 80, 100, 120, 200] {
            let mut b = plain("Qwen3.5-4B-Q4_K_M.gguf", 2_600_000_000, columns);
            for done in [0, 1, 236_000_000, 1_300_000_000, 2_599_999_999, 2_600_000_000] {
                b.last_draw = None; // force a draw
                let line = b.update(done).expect("a draw");
                assert!(
                    line.chars().count() <= columns,
                    "{} columns: {} wide: {line:?}",
                    columns,
                    line.chars().count()
                );
            }
        }
    }

    #[test]
    fn a_coloured_line_is_measured_without_its_escapes() {
        // Colour must not change the layout; only the visible width counts.
        let mut b = Bar::new("model.gguf", 1000, 60, true);
        let line = b.update(500).unwrap();
        assert!(line.contains('\x1b'), "should be coloured");
        assert!(visible_width(&line) <= 60, "{}", visible_width(&line));
    }

    #[test]
    fn a_narrow_terminal_drops_fields_rather_than_wrapping() {
        let mut wide = plain("model.gguf", 2_600_000_000, 100);
        let w = wide.update(1_000_000_000).unwrap();
        assert!(w.contains('/'), "a wide terminal shows the total: {w}");

        let mut narrow = plain("model.gguf", 2_600_000_000, 34);
        let n = narrow.update(1_000_000_000).unwrap();
        assert!(n.chars().count() <= 34, "{n}");
        // The two that always survive.
        assert!(n.contains('%'), "{n}");
        assert!(n.contains("/s"), "{n}");
    }

    #[test]
    fn a_very_long_name_is_shortened_from_the_front() {
        // The tail distinguishes Q4_K_M from Q8_0; the head is shared by every
        // file in the repository.
        let mut b = plain("some-extremely-long-model-name-that-goes-on-Q4_K_M.gguf", 1000, 80);
        let line = b.update(500).unwrap();
        assert!(line.chars().count() <= 80, "{line}");
        assert!(line.contains("Q4_K_M.gguf"), "the tail survives: {line}");
        assert!(line.contains('…'), "{line}");
    }

    // ------------------------------------------------------------ the bar

    #[test]
    fn the_bar_fills_in_proportion() {
        let mut b = plain("m.gguf", 1000, 80);
        let line = b.update(500).expect("first update always draws");
        assert!(line.contains("50%"), "{line}");
        assert!(line.contains('█'), "{line}");
        assert!(line.contains('─'), "half of it is still empty: {line}");
    }

    #[test]
    fn a_completed_bar_is_full_and_has_no_spinner() {
        let mut b = plain("m.gguf", 1000, 80);
        let line = b.update(1000).unwrap();
        assert!(line.contains("100%"), "{line}");
        assert!(!line.contains('─'), "no empty cells at 100%: {line}");
        assert!(line.contains('✓'), "finished, so no spinner: {line}");
    }

    #[test]
    fn the_bar_advances_within_a_single_cell() {
        // A whole cell is thirty seconds of apparent stillness on a large
        // file, which reads as a stall.
        let mut b = plain("m.gguf", 10_000, 80);
        b.last_draw = None;
        let a = b.update(100).unwrap();
        b.last_draw = None;
        let c = b.update(180).unwrap();
        assert_ne!(a, c, "progress under one cell must still show");
    }

    #[test]
    fn the_spinner_turns() {
        let mut b = plain("m.gguf", 1000, 80);
        b.last_draw = None;
        let first = b.update(100).unwrap();
        b.last_draw = None;
        let second = b.update(100).unwrap();
        assert_ne!(first, second, "the same bytes should still animate");
    }

    #[test]
    fn progress_beyond_the_total_does_not_overflow_the_bar() {
        // A server reporting a stale content-length must not corrupt output.
        let mut b = plain("m.gguf", 100, 80);
        let line = b.update(500).unwrap();
        assert!(line.contains("100%"), "{line}");
        assert!(line.chars().count() <= 80, "{line}");
    }

    // ------------------------------------------------------------- timing

    #[test]
    fn redraws_are_rate_limited_but_completion_always_draws() {
        let mut b = plain("m.gguf", 1000, 80);
        assert!(b.update(100).is_some(), "the first update draws");
        assert!(b.update(200).is_none(), "an immediate second update is skipped");
        assert!(b.update(1000).is_some(), "finishing always draws");
    }

    #[test]
    fn an_unknown_total_reports_bytes_rather_than_a_fake_bar() {
        let mut b = plain("m.gguf", 0, 80);
        let line = b.update(2048).unwrap();
        assert!(line.contains("2.0 KB"), "{line}");
        assert!(!line.contains('█'), "no progress bar without a total: {line}");
    }

    #[test]
    fn eta_is_unknown_until_a_rate_is_measured() {
        let b = plain("m.gguf", 1000, 80);
        assert_eq!(b.eta(100), "--:--", "no rate yet means no estimate");
    }

    #[test]
    fn the_final_line_reports_an_average_rate() {
        let b = plain("m.gguf", 1000, 80);
        let line = b.finish(1000);
        assert!(line.contains("1000 B"), "{line}");
        assert!(line.contains("/s)"), "should include a rate: {line}");
    }

    #[test]
    fn colour_is_omitted_when_it_is_not_wanted() {
        let mut b = plain("m.gguf", 1000, 80);
        let line = b.update(500).unwrap();
        assert!(!line.contains('\x1b'), "no escapes when colour is off: {line:?}");
    }

    #[test]
    fn escape_sequences_do_not_count_toward_width() {
        assert_eq!(visible_width("abc"), 3);
        assert_eq!(visible_width("\x1b[33mabc\x1b[0m"), 3);
        assert_eq!(visible_width("\x1b[1;33m─\x1b[0m"), 1);
    }
}
