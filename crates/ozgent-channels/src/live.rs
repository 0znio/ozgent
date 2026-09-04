//! One growing reply, shown as a chat message that is rewritten in place.
//!
//! Streaming into a chat app is not streaming into a terminal. There is no
//! scrollback to append to — there is one message, and rewriting it costs a
//! rate-limited API call. So updates are throttled, and identical updates are
//! skipped.
//!
//! The case that needs real care is a reply that outgrows a single message.
//! Revising cannot help there, so at that point the current message is closed
//! off at a clean boundary — it keeps exactly the text it will end with — and a
//! new one is started for the remainder, which then becomes the live one. That
//! way a long answer arrives as a sequence of complete messages, with only the
//! last still moving, and no text is ever shown twice or dropped between two.

use std::time::{Duration, Instant};

use tokio::sync::mpsc::UnboundedSender;

use crate::chat::Command;
use crate::split::take_one;

/// How often the live message may be rewritten.
///
/// Telegram allows a burst and then starts refusing; WhatsApp caps how many
/// times a message may be edited at all. Roughly once a second reads as live
/// without spending the budget that the *final* update needs — the one that
/// must land.
pub const REVISE_EVERY: Duration = Duration::from_millis(1100);

pub struct Live {
    chat: String,
    tx: UnboundedSender<Command>,
    limit: usize,
    /// The message being rewritten.
    token: u64,
    /// Whether that message has been sent yet.
    posted: bool,
    /// Bytes of the reply already closed off into finished messages.
    ///
    /// Bytes rather than characters because that is what the splitter reports
    /// consuming, and converting between the two is exactly where an off-by-one
    /// would repeat a character at every message boundary.
    committed: usize,
    /// A code fence left open by the message before this one.
    carried: Option<String>,
    /// What the live message currently says, so an unchanged update is not
    /// sent at all.
    showing: String,
    last: Instant,
}

impl Live {
    pub fn new(chat: String, tx: UnboundedSender<Command>, token: u64, limit: usize) -> Self {
        Self {
            chat,
            tx,
            limit,
            token,
            posted: false,
            committed: 0,
            carried: None,
            showing: String::new(),
            last: Instant::now() - REVISE_EVERY,
        }
    }

    /// Show `full` — the whole reply so far — throttled.
    ///
    /// `next_token` is called only when the reply outgrows a message and a new
    /// one has to be started.
    pub fn update(&mut self, full: &str, next_token: &mut dyn FnMut() -> u64) {
        if self.last.elapsed() < REVISE_EVERY {
            return;
        }
        self.write(full, next_token);
    }

    /// Show `full` whatever the throttle says. Used for the final state, and
    /// before anything that must appear after the reply — a question, say.
    pub fn flush(&mut self, full: &str, next_token: &mut dyn FnMut() -> u64) {
        self.write(full, next_token);
    }

    fn write(&mut self, full: &str, next_token: &mut dyn FnMut() -> u64) {
        self.last = Instant::now();

        loop {
            // `committed` is a byte offset the splitter handed back, so it is
            // always on a character boundary.
            let remainder = &full[self.committed.min(full.len())..];
            if remainder.trim().is_empty() {
                return;
            }

            let piece = take_one(remainder, self.limit, self.carried.as_deref());
            // A piece that consumed nothing would loop forever. `take_one`
            // always makes progress, but the reply matters more than proving
            // that here.
            if piece.used == 0 {
                return;
            }

            // Everything left fits: this is the live message, still growing.
            if piece.used >= remainder.len() {
                self.show(piece.markdown);
                return;
            }

            // It does not fit. Close this message off at a clean boundary —
            // it now says what it will always say — and start a new live one.
            self.show(piece.markdown);
            self.committed += piece.used;
            self.carried = piece.open_fence;
            self.token = next_token();
            self.posted = false;
            self.showing.clear();
        }
    }

    /// Send one message's worth, as a new message or a rewrite of the live one.
    fn show(&mut self, markdown: String) {
        if self.showing == markdown {
            return;
        }
        let command = if self.posted {
            Command::Revise { chat: self.chat.clone(), token: self.token, markdown: markdown.clone() }
        } else {
            self.posted = true;
            Command::Post { chat: self.chat.clone(), token: self.token, markdown: markdown.clone() }
        };
        self.showing = markdown;
        let _ = self.tx.send(command);
    }

    /// Whether anything has been sent to the chat yet.
    pub fn posted(&self) -> bool {
        self.posted || self.committed > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

    fn live(limit: usize) -> (Live, UnboundedReceiver<Command>) {
        let (tx, rx) = unbounded_channel();
        (Live::new("c".into(), tx, 0, limit), rx)
    }

    fn drain(rx: &mut UnboundedReceiver<Command>) -> Vec<Command> {
        let mut out = Vec::new();
        while let Ok(c) = rx.try_recv() {
            out.push(c);
        }
        out
    }

    fn tokens() -> impl FnMut() -> u64 {
        let mut n = 100;
        move || {
            n += 1;
            n
        }
    }

    #[test]
    fn the_first_update_posts_and_the_next_revises() {
        let (mut l, mut rx) = live(100);
        let mut next = tokens();
        l.flush("hello", &mut next);
        l.flush("hello there", &mut next);

        match drain(&mut rx).as_slice() {
            [Command::Post { markdown: a, token: t1, .. }, Command::Revise { markdown: b, token: t2, .. }] => {
                assert_eq!(a, "hello");
                assert_eq!(b, "hello there");
                assert_eq!(t1, t2, "the same message is rewritten");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_unchanged_update_is_not_sent() {
        // Telegram calls an edit that changes nothing an error, and every
        // provider counts it against the rate limit.
        let (mut l, mut rx) = live(100);
        let mut next = tokens();
        l.flush("same", &mut next);
        l.flush("same", &mut next);
        assert_eq!(drain(&mut rx).len(), 1);
    }

    #[test]
    fn nothing_is_sent_for_an_empty_reply() {
        let (mut l, mut rx) = live(100);
        let mut next = tokens();
        l.flush("", &mut next);
        l.flush("   \n ", &mut next);
        assert!(drain(&mut rx).is_empty());
        assert!(!l.posted());
    }

    #[test]
    fn the_throttle_holds_updates_back_but_flush_does_not() {
        let (mut l, mut rx) = live(100);
        let mut next = tokens();
        l.update("one", &mut next);
        l.update("two", &mut next);
        assert_eq!(drain(&mut rx).len(), 1, "the second was inside the throttle window");

        l.flush("three", &mut next);
        assert_eq!(drain(&mut rx).len(), 1, "a flush always lands");
    }

    #[test]
    fn a_reply_that_outgrows_a_message_continues_in_a_new_one() {
        let (mut l, mut rx) = live(40);
        let mut next = tokens();
        let text = (0..12).map(|i| format!("sentence number {i}.")).collect::<Vec<_>>().join(" ");
        l.flush(&text, &mut next);

        let sent = drain(&mut rx);
        assert!(sent.len() > 1, "expected more than one message: {sent:?}");

        // Every message is within the limit, and each new one has its own
        // token — a rewrite of an earlier message would replace text the
        // reader has already seen.
        let mut tokens_seen = Vec::new();
        for command in &sent {
            let (token, markdown) = match command {
                Command::Post { token, markdown, .. } => (token, markdown),
                Command::Revise { token, markdown, .. } => (token, markdown),
                other => panic!("{other:?}"),
            };
            assert!(markdown.chars().count() <= 40, "{markdown:?}");
            tokens_seen.push(*token);
        }
        assert!(tokens_seen.windows(2).any(|w| w[0] != w[1]), "a second message was started");
    }

    #[test]
    fn nothing_is_repeated_or_dropped_across_the_overflow() {
        // The property that matters to a reader: the messages, read in order,
        // are the reply.
        let (mut l, mut rx) = live(60);
        let mut next = tokens();
        let words: Vec<String> = (0..40).map(|i| format!("w{i}")).collect();
        let text = words.join(" ");
        l.flush(&text, &mut next);

        // Keep only the last thing said in each message, which is what the
        // chat ends up showing.
        let mut by_token: Vec<(u64, String)> = Vec::new();
        for command in drain(&mut rx) {
            let (token, markdown) = match command {
                Command::Post { token, markdown, .. } | Command::Revise { token, markdown, .. } => {
                    (token, markdown)
                }
                other => panic!("{other:?}"),
            };
            match by_token.iter_mut().find(|(t, _)| *t == token) {
                Some(slot) => slot.1 = markdown,
                None => by_token.push((token, markdown)),
            }
        }
        let read: Vec<String> = by_token
            .iter()
            .flat_map(|(_, m)| m.split_whitespace().map(str::to_string).collect::<Vec<_>>())
            .collect();
        assert_eq!(read, words, "the messages in order are the reply");
    }

    #[test]
    fn growth_after_an_overflow_only_touches_the_last_message() {
        // A late revision that rewrote an earlier message would replace text
        // the reader has already read.
        let (mut l, mut rx) = live(50);
        let mut next = tokens();
        let long: String = (0..20).map(|i| format!("word{i} ")).collect();
        l.flush(&long, &mut next);
        let first_pass = drain(&mut rx);
        let last_token = match first_pass.last().unwrap() {
            Command::Post { token, .. } | Command::Revise { token, .. } => *token,
            other => panic!("{other:?}"),
        };

        l.flush(&format!("{long} and more"), &mut next);
        for command in drain(&mut rx) {
            let token = match &command {
                Command::Post { token, .. } | Command::Revise { token, .. } => *token,
                other => panic!("{other:?}"),
            };
            assert!(token >= last_token, "an earlier message was rewritten");
        }
    }
}
