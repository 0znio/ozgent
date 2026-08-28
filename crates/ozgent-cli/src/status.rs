//! What the status bar says, and what it gives up when there is no room.
//!
//! The bar itself is drawn by [`crate::tui`]; this is only the layout. Kept
//! apart because the interesting part — which facts survive a narrow terminal
//! — is decided by ordering, and ordering is worth testing without a screen.
//!
//! A scrolling transcript has nowhere to put the facts that are true *now*:
//! how full the context is, what the last reply cost, which model is
//! answering. Printing them after every turn buries the conversation in
//! chrome. So they live on one row that the conversation never scrolls over.

/// One labelled fact, and how badly it wants to stay when space runs out.
///
/// Ordering matters: a status line that drops the model name to keep the
/// sampler settings has kept the wrong thing.
pub struct Segment {
    pub text: String,
    /// Lower is more important. Segment 0 is never dropped.
    pub priority: u8,
}

impl Segment {
    pub fn new(priority: u8, text: impl Into<String>) -> Self {
        Self { text: text.into(), priority }
    }
}

/// Fit segments into `width` columns, dropping the least important first.
///
/// Returns the plain text of the line, padded to exactly `width` so the
/// highlight runs edge to edge the way vim's does. Kept separate from drawing
/// so the layout can be tested without a terminal.
pub fn compose(segments: &[Segment], width: usize) -> String {
    const SEP: &str = "  ·  ";

    let mut keep: Vec<&Segment> = segments.iter().collect();
    loop {
        let line = join(&keep, SEP);
        // The line is padded with a leading and trailing space, so it needs
        // two columns beyond its own content.
        if line.chars().count() + 2 <= width || keep.len() <= 1 {
            let mut out = format!(" {line} ");
            let len = out.chars().count();
            if len < width {
                out.push_str(&" ".repeat(width - len));
            } else if len > width {
                // Only reachable once a single segment is left and even that
                // does not fit. Truncating beats wrapping, which would scroll
                // the transcript by a line on every redraw.
                out = out.chars().take(width).collect();
            }
            return out;
        }
        // Drop one of the least important remaining segments.
        let worst = keep.iter().enumerate().max_by_key(|(i, s)| (s.priority, *i));
        let Some((index, _)) = worst else { return String::new() };
        keep.remove(index);
    }
}

fn join(segments: &[&Segment], sep: &str) -> String {
    segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(sep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs() -> Vec<Segment> {
        vec![
            Segment::new(0, "qwen3.5:4b"),
            Segment::new(1, "ctx 3.2k/32k"),
            Segment::new(2, "48.6 tok/s"),
            Segment::new(3, "temp 0.7"),
        ]
    }

    #[test]
    fn a_wide_line_shows_everything_and_fills_the_width() {
        let line = compose(&segs(), 100);
        assert_eq!(line.chars().count(), 100, "the highlight must reach both edges");
        assert!(line.contains("qwen3.5:4b"));
        assert!(line.contains("temp 0.7"));
    }

    #[test]
    fn a_narrow_line_drops_the_least_important_first() {
        // 40 columns is a phone-sized terminal, and the model name is the one
        // fact that must survive it.
        let line = compose(&segs(), 40);
        assert_eq!(line.chars().count(), 40);
        assert!(line.contains("qwen3.5:4b"), "got {line:?}");
        assert!(!line.contains("temp 0.7"), "the sampler is the first to go: {line:?}");
    }

    #[test]
    fn dropping_stops_before_the_line_is_empty() {
        let line = compose(&segs(), 12);
        assert_eq!(line.chars().count(), 12);
        assert!(line.trim().starts_with("qwen"), "got {line:?}");
    }

    #[test]
    fn a_single_oversized_segment_is_truncated_not_wrapped() {
        // Wrapping would scroll the transcript by one line on every repaint.
        let long = Segment::new(0, "x".repeat(200));
        assert_eq!(compose(&[long], 30).chars().count(), 30);
    }

    #[test]
    fn segments_are_dropped_by_priority_not_by_position() {
        let out = compose(
            &[
                Segment::new(0, "keep"),
                Segment::new(9, "drop-me"),
                Segment::new(1, "also-keep"),
            ],
            22,
        );
        assert!(out.contains("keep") && out.contains("also-keep"), "got {out:?}");
        assert!(!out.contains("drop-me"), "got {out:?}");
    }

}
