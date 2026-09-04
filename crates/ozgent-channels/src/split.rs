//! Cutting a reply into messages a chat app will accept.
//!
//! Splitting happens on the *markdown*, before rendering, and not on the
//! rendered output. Rendering first and cutting after would put a cut in the
//! middle of `<a href="…">`, and Telegram rejects a message with an unbalanced
//! tag outright — the failure would be a lost reply, not a cosmetic one.
//!
//! The one construct that cannot simply be cut is a fenced code block: half a
//! fence in each of two messages renders as two broken blocks. So a fence that
//! is open at a cut is closed on the way out and reopened, with its language,
//! on the way in.
//!
//! [`take_one`] rather than [`split`] alone is what a live, growing reply needs.
//! Such a caller cuts the same text repeatedly as it grows, so it has to know
//! exactly how much of the source each message used — and that is *not* the
//! length of the message: leading whitespace is dropped and a closing fence is
//! added, so measuring the output undercounts the input by a character or two
//! and the next message then repeats the end of the last one.

/// Telegram's own limit is 4096, counted over the message *text* after markup
/// is parsed out. Rendering can only shrink the text — tags are not counted —
/// so measuring the markdown is conservative, and the headroom covers the
/// reopened fences a split can add.
pub const TELEGRAM_LIMIT: usize = 3800;

/// WhatsApp accepts far more than this. The limit here is about reading on a
/// phone rather than about the protocol: past a few thousand characters the
/// message becomes a wall that has to be expanded to be read at all.
pub const WHATSAPP_LIMIT: usize = 3800;

const CLOSE: &str = "```";

/// One message's worth of a reply.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// What to send.
    pub markdown: String,
    /// How many **bytes** of the source this used, including whitespace that
    /// was dropped. The caller's next cut starts here.
    pub used: usize,
    /// A fence this piece leaves open, to be reopened on the next one.
    pub open_fence: Option<String>,
}

/// Take the first message's worth of `text`.
///
/// `carried` is a fence left open by the previous piece, reopened at the top of
/// this one so the code block continues rather than restarting.
pub fn take_one(text: &str, limit: usize, carried: Option<&str>) -> Piece {
    let prefix = carried.map(|f| format!("{f}\n")).unwrap_or_default();

    // Whitespace at the front is the seam from the previous cut. It is
    // consumed but not shown.
    let lead = text.len() - text.trim_start().len();
    let body = &text[lead..];
    if body.is_empty() {
        return Piece { markdown: String::new(), used: text.len(), open_fence: None };
    }

    // Room for a closing fence has to be reserved before deciding what fits,
    // or a piece that ends inside a code block overruns the limit by exactly
    // the marker that keeps it readable.
    let fenced = body.contains("```") || body.contains("~~~") || carried.is_some();
    let reserve = if fenced { CLOSE.len() + 1 } else { 0 };
    let room = limit.saturating_sub(prefix.chars().count() + reserve).max(1);

    let cut = if body.chars().count() <= room { body.len() } else { boundary(body, room) };
    let head = body[..cut].trim_end();

    let mut markdown = format!("{prefix}{head}");
    let open = open_fence(&markdown);
    if open.is_some() {
        markdown.push('\n');
        markdown.push_str(CLOSE);
    }

    Piece { markdown, used: lead + cut, open_fence: open }
}

/// Split markdown into chunks of at most `limit` characters.
///
/// Always returns at least one chunk, so a caller never has to handle "no
/// messages to send" as a separate case.
pub fn split(markdown: &str, limit: usize) -> Vec<String> {
    let text = markdown.trim();
    if text.is_empty() {
        return vec![String::new()];
    }
    if text.chars().count() <= limit {
        return vec![text.to_string()];
    }

    let mut out = Vec::new();
    let mut at = 0;
    let mut carried: Option<String> = None;

    while at < text.len() {
        let piece = take_one(&text[at..], limit, carried.as_deref());
        // A piece that consumed nothing would loop forever. `take_one` always
        // makes progress — it hard-cuts as a last resort — but the reply
        // matters more than proving that here.
        if piece.used == 0 {
            break;
        }
        at += piece.used;
        carried = piece.open_fence;
        if !piece.markdown.trim().is_empty() {
            out.push(piece.markdown);
        }
    }

    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// The best place to cut within the first `room` characters.
///
/// Preference order is the order a reader would least notice: between
/// paragraphs, then between lines, then after a sentence, then between words.
/// A hard cut at `room` is the last resort and only happens for text with no
/// break in it at all — a long URL, or a language that does not use spaces.
fn boundary(text: &str, room: usize) -> usize {
    let cap = text.char_indices().nth(room).map(|(i, _)| i).unwrap_or(text.len());
    let window = &text[..cap];

    // Far enough in that the chunk is worth sending. Without this a break
    // near the start would produce a stream of tiny messages.
    let least = cap / 4;

    for pattern in ["\n\n", "\n"] {
        if let Some(i) = window.rfind(pattern) {
            if i >= least {
                return i + pattern.len();
            }
        }
    }
    for pattern in [". ", "! ", "? ", "; "] {
        if let Some(i) = window.rfind(pattern) {
            if i >= least {
                return i + pattern.len();
            }
        }
    }
    if let Some(i) = window.rfind(' ') {
        if i >= least {
            return i + 1;
        }
    }
    cap
}

/// The opening line of a fence left unclosed by `text`, if any.
///
/// Counted rather than searched for a final fence: nesting is not a thing in
/// markdown fences, so an odd number of fence lines means one is open, and the
/// last odd one is the one that opened it.
fn open_fence(text: &str) -> Option<String> {
    let mut open: Option<String> = None;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("```") && !trimmed.starts_with("~~~") {
            continue;
        }
        open = match open {
            // A closing fence carries no language, so any fence line while one
            // is open closes it.
            Some(_) => None,
            None => Some(trimmed.trim_end().to_string()),
        };
    }
    open
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_reply_is_one_message() {
        assert_eq!(split("hello", 100), vec!["hello".to_string()]);
    }

    #[test]
    fn an_empty_reply_still_yields_one_chunk() {
        // Callers send `chunks[0]`; returning nothing would panic there.
        assert_eq!(split("", 100), vec![String::new()]);
        assert_eq!(split("   \n ", 100), vec![String::new()]);
    }

    #[test]
    fn every_chunk_is_within_the_limit() {
        let text = "word ".repeat(4000);
        for chunk in split(&text, 200) {
            assert!(chunk.chars().count() <= 200, "chunk of {}", chunk.chars().count());
        }
    }

    #[test]
    fn a_piece_reports_the_whitespace_it_swallowed() {
        // The bug this pins: measuring consumption by the length of the
        // *output* undercounts, because the seam whitespace is dropped — so
        // the next message repeats the last character of this one.
        let piece = take_one("   hello world", 100, None);
        assert_eq!(piece.markdown, "hello world");
        assert_eq!(piece.used, "   hello world".len(), "the leading spaces were consumed");
    }

    #[test]
    fn a_piece_reports_consumption_separately_from_the_marker_it_added() {
        // The other half: a closing fence is in the output but not the input.
        let text = "```rust\nlet a = 1;";
        let piece = take_one(text, 100, None);
        assert!(piece.markdown.ends_with("```"), "{:?}", piece.markdown);
        assert_eq!(piece.used, text.len(), "the marker is not part of the source");
        assert_eq!(piece.open_fence.as_deref(), Some("```rust"));
    }

    #[test]
    fn walking_pieces_covers_the_source_exactly_once() {
        // Read back the source by consumption alone; the total must be the
        // whole text with nothing counted twice.
        let text = "alpha beta gamma. delta epsilon. ".repeat(20);
        let mut at = 0;
        let mut carried = None;
        let mut steps = 0;
        while at < text.len() {
            let piece = take_one(&text[at..], 40, carried.as_deref());
            assert!(piece.used > 0, "no progress at {at}");
            at += piece.used;
            carried = piece.open_fence;
            steps += 1;
            assert!(steps < 500, "not terminating");
        }
        assert_eq!(at, text.len());
    }

    #[test]
    fn nothing_is_lost_across_the_split() {
        let text = "alpha beta gamma. delta epsilon zeta. eta theta iota kappa lambda mu nu.";
        let joined: String = split(text, 25).join(" ");
        for word in text.split_whitespace() {
            let word = word.trim_end_matches(['.', ',']);
            assert!(joined.contains(word), "{word} went missing from {joined:?}");
        }
    }

    #[test]
    fn a_paragraph_break_is_preferred_to_a_word_break() {
        let text = format!("{}\n\n{}", "a".repeat(60), "b".repeat(60));
        let chunks = split(&text, 100);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "a".repeat(60));
        assert_eq!(chunks[1], "b".repeat(60));
    }

    #[test]
    fn a_fence_cut_in_half_is_closed_and_reopened() {
        // The failure without this is two broken code blocks, one of which
        // swallows the prose that follows it.
        let code = (0..40).map(|i| format!("line {i};")).collect::<Vec<_>>().join("\n");
        let text = format!("here:\n\n```rust\n{code}\n```\n\nafter");
        let chunks = split(&text, 200);
        assert!(chunks.len() > 1, "the fixture must actually split");

        for chunk in &chunks {
            let fences = chunk.lines().filter(|l| l.trim_start().starts_with("```")).count();
            assert_eq!(fences % 2, 0, "unbalanced fence in {chunk:?}");
            assert!(chunk.chars().count() <= 200, "{} chars", chunk.chars().count());
        }
        let resumed: Vec<&String> =
            chunks.iter().skip(1).filter(|c| c.starts_with("```rust")).collect();
        assert!(!resumed.is_empty(), "the reopened fence keeps its language: {chunks:?}");
    }

    #[test]
    fn text_with_no_break_at_all_is_still_cut() {
        // A long URL, or a script without spaces. Cutting badly beats not
        // sending the message.
        let text = "x".repeat(500);
        let chunks = split(&text, 100);
        assert!(chunks.len() >= 5);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 100);
        }
    }

    #[test]
    fn a_cut_never_lands_inside_a_character() {
        // Multi-byte text cut by byte index would panic or produce invalid
        // UTF-8; the limit is counted in characters throughout.
        let text = "日本語のテキストです。".repeat(60);
        let chunks = split(&text, 50);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 50);
        }
        assert!(chunks.join("").contains("日本語"));
    }

    #[test]
    fn an_open_fence_is_recognised_with_its_language() {
        assert_eq!(open_fence("```rust\ncode"), Some("```rust".to_string()));
        assert_eq!(open_fence("```rust\ncode\n```"), None);
        assert_eq!(open_fence("no fences here"), None);
        // A second block opened after the first closed.
        assert_eq!(open_fence("```\na\n```\n\n```py\nb"), Some("```py".to_string()));
    }
}
