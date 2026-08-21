//! A download progress bar.
//!
//! Model files are gigabytes, so a download without feedback is
//! indistinguishable from a hang. The bar reports the three things that
//! actually answer "should I wait?": how far along, how fast, and how much
//! longer.

use std::time::{Duration, Instant};

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
    width: usize,
}

/// How often the line is redrawn.
///
/// On a fast link a write per chunk would dominate the transfer itself, and
/// the numbers would flicker too quickly to read.
const REDRAW_INTERVAL: Duration = Duration::from_millis(100);

/// Length of the window used to measure speed.
const RATE_WINDOW: Duration = Duration::from_millis(500);

/// Weight given to a new rate sample. Low enough that a stalled chunk does not
/// make the estimate lurch.
const SMOOTHING: f64 = 0.3;

impl Bar {
    pub fn new(name: impl Into<String>, total: u64, width: usize) -> Self {
        let now = Instant::now();
        Self {
            name: name.into(),
            total,
            started: now,
            window_start: now,
            window_bytes: 0,
            rate: 0.0,
            last_draw: None,
            width,
        }
    }

    /// The expected size, or 0 when the server did not report one.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Record progress and return a line to draw, or `None` if it is too soon.
    pub fn update(&mut self, done: u64) -> Option<String> {
        let now = Instant::now();

        // Measure over a window rather than per chunk: chunk sizes vary wildly
        // and an instantaneous rate is unreadable.
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
        Some(self.render(done))
    }

    /// The final line, with the average rate over the whole transfer.
    pub fn finish(&self, done: u64) -> String {
        let seconds = self.started.elapsed().as_secs_f64().max(0.001);
        format!(
            "  {}  {} in {} ({}/s)",
            self.name,
            bytes(done),
            duration(self.started.elapsed()),
            bytes((done as f64 / seconds) as u64)
        )
    }

    fn render(&self, done: u64) -> String {
        if self.total == 0 {
            // Length unknown: report what has arrived rather than a fake bar.
            return format!("  {}  {} ({}/s)", self.name, bytes(done), bytes(self.rate as u64));
        }

        let fraction = (done as f64 / self.total as f64).clamp(0.0, 1.0);
        let filled = (fraction * self.width as f64).round() as usize;

        format!(
            "  {} [{}{}] {:>3.0}%  {} / {}  {}/s  {}",
            self.name,
            "━".repeat(filled),
            "─".repeat(self.width.saturating_sub(filled)),
            fraction * 100.0,
            bytes(done),
            bytes(self.total),
            bytes(self.rate as u64),
            self.eta(done),
        )
    }

    fn eta(&self, done: u64) -> String {
        if self.rate <= 0.0 || done >= self.total {
            return "--:--".into();
        }
        let remaining = (self.total - done) as f64 / self.rate;
        duration(Duration::from_secs_f64(remaining))
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_counts_are_readable() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1024), "1.0 KB");
        assert_eq!(bytes(7_300_000_000), "6.8 GB");
        // Past 100 the decimal is noise.
        assert_eq!(bytes(300 * 1024 * 1024), "300 MB");
    }

    #[test]
    fn durations_grow_a_field_past_an_hour() {
        assert_eq!(duration(Duration::from_secs(9)), "0:09");
        assert_eq!(duration(Duration::from_secs(75)), "1:15");
        assert_eq!(duration(Duration::from_secs(3725)), "1:02:05");
    }

    #[test]
    fn the_bar_fills_in_proportion() {
        let mut b = Bar::new("m.gguf", 1000, 10);
        let line = b.update(500).expect("first update always draws");
        assert!(line.contains("50%"), "{line}");
        // Five filled cells of ten.
        assert_eq!(line.matches('━').count(), 5, "{line}");
        assert_eq!(line.matches('─').count(), 5, "{line}");
    }

    #[test]
    fn a_completed_bar_is_full() {
        let mut b = Bar::new("m.gguf", 1000, 10);
        let line = b.update(1000).unwrap();
        assert!(line.contains("100%"), "{line}");
        assert_eq!(line.matches('─').count(), 0, "no empty cells at 100%: {line}");
    }

    #[test]
    fn redraws_are_rate_limited_but_completion_always_draws() {
        let mut b = Bar::new("m.gguf", 1000, 10);
        assert!(b.update(100).is_some(), "the first update draws");
        assert!(b.update(200).is_none(), "an immediate second update is skipped");
        // Completion must never be swallowed by the rate limit.
        assert!(b.update(1000).is_some(), "finishing always draws");
    }

    #[test]
    fn an_unknown_total_reports_bytes_rather_than_a_fake_bar() {
        let mut b = Bar::new("m.gguf", 0, 10);
        let line = b.update(2048).unwrap();
        assert!(line.contains("2.0 KB"), "{line}");
        assert!(!line.contains('['), "no progress bar without a total: {line}");
    }

    #[test]
    fn eta_is_unknown_until_a_rate_is_measured() {
        let b = Bar::new("m.gguf", 1000, 10);
        assert_eq!(b.eta(100), "--:--", "no rate yet means no estimate");
    }

    #[test]
    fn progress_beyond_the_total_does_not_overflow_the_bar() {
        // A server reporting a stale content-length should not corrupt output.
        let mut b = Bar::new("m.gguf", 100, 10);
        let line = b.update(500).unwrap();
        assert!(line.contains("100%"), "{line}");
        assert_eq!(line.matches('━').count(), 10, "{line}");
    }

    #[test]
    fn the_final_line_reports_an_average_rate() {
        let b = Bar::new("m.gguf", 1000, 10);
        let line = b.finish(1000);
        assert!(line.contains("1000 B"), "{line}");
        assert!(line.contains("/s)"), "should include a rate: {line}");
    }
}
