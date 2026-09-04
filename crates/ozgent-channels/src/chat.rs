//! The vocabulary every channel speaks.
//!
//! Telegram is polled over HTTPS and WhatsApp arrives from a child process, so
//! nothing about their transports is alike. What *is* alike is the shape of the
//! conversation: messages come in, replies go out, replies are revised while
//! they are being written, and some of them are questions with buttons. This
//! module is that shape, and it is all the gateway knows about.
//!
//! Outgoing messages are addressed by a `token` the gateway assigns rather than
//! by the provider's message id. The provider's id only exists *after* the
//! message has been sent, so a gateway that needed it in order to revise a
//! message would have to wait for a round trip before it could stream — and
//! would need a reply path back from every channel to learn it. A token the
//! sender made up sidesteps both: the channel keeps the mapping, and commands
//! stay one-way.

use ozgent_core::permission::{Choice, Effect};

/// A message from a person.
#[derive(Debug, Clone)]
pub struct Msg {
    /// The provider's id for the conversation, as text.
    pub chat: String,
    /// The provider's id for the person. Stable; the handle is not.
    pub sender_id: String,
    /// A second name the sender goes by, when the channel has one: the
    /// `@username` on Telegram, the full JID on WhatsApp. Carried because an
    /// operator writes down whichever one they happen to know, and an
    /// allowlist that only matched the other would look broken rather than
    /// mistyped.
    pub handle: Option<String>,
    /// Something to call them in a log line or a pairing prompt.
    pub name: String,
    pub text: String,
    pub images: Vec<ozgent_core::ImageSource>,
    /// True in a group. Kept because "who may talk to ozgent" and "where may
    /// ozgent talk" are different questions with different answers.
    pub group: bool,
}

impl Msg {
    /// Every name this sender goes by, for matching against an allowlist.
    pub fn identities(&self) -> Vec<&str> {
        let mut out = vec![self.sender_id.as_str()];
        if let Some(h) = &self.handle {
            out.push(h.as_str());
        }
        out
    }
}

/// Something that happened on a channel.
#[derive(Debug, Clone)]
pub enum Inbound {
    /// Connected, and knows who it is. Printed once per link.
    Ready { who: String },
    /// Something the operator has to see on the terminal — a QR code to scan,
    /// a reconnection. Not a failure, and not the model's business.
    Notice { text: String },
    Message(Box<Msg>),
    /// A tapped button.
    ///
    /// Carries the gateway's own token for the question rather than the tool
    /// call id: Telegram caps button data at 64 bytes, which a call id can
    /// exceed. The gateway holds the mapping and resolves it.
    Answer { chat: String, token: u64, choice: Choice },
    /// The transport is gone and will not recover on its own.
    Failed { reason: String },
}

/// Something to do on a channel.
#[derive(Debug, Clone)]
pub enum Command {
    /// Show the "typing…" hint, where the provider has one.
    Typing { chat: String },
    /// Send a new message and remember it as `token`.
    Post { chat: String, token: u64, markdown: String },
    /// Rewrite the message sent as `token`.
    ///
    /// Best-effort by definition: a provider may rate-limit edits, may refuse
    /// to edit a message that is too old, or may not support editing at all. A
    /// channel that cannot revise sends nothing rather than spamming the chat.
    Revise { chat: String, token: u64, markdown: String },
    /// Ask whether a tool call may run.
    Ask { chat: String, token: u64, question: Box<Question> },
    /// Replace a question with its outcome, taking any buttons away.
    Settle { chat: String, token: u64, markdown: String },
}

/// A permission question, as a chat has to show it.
#[derive(Debug, Clone)]
pub struct Question {
    /// The tool call id. Comes back on the [`Inbound::Answer`], and is what
    /// the inference thread is blocked waiting for.
    pub id: String,
    pub tool: String,
    pub effect: Effect,
    /// The arguments, already shortened for a phone screen.
    pub detail: String,
}

/// The four answers, in the order they are offered everywhere in ozgent.
///
/// Numbered because that ordering is already what the terminal prints and what
/// the web interface lays out, and because a channel with no buttons needs the
/// numbers to be typeable.
pub const ANSWERS: [(&str, Choice); 4] = [
    ("yes", Choice::Once),
    ("session", Choice::Session),
    ("always", Choice::Always),
    ("no", Choice::Deny),
];

/// Read a typed answer to a permission question.
///
/// Every channel needs this, not only the ones without buttons: a person can
/// always type instead of tapping, and on a phone that is often easier than
/// scrolling back to the message the buttons are on.
pub fn read_choice(text: &str) -> Option<Choice> {
    let t = text.trim().trim_end_matches(['.', '!']).to_ascii_lowercase();
    match t.as_str() {
        "1" | "y" | "yes" | "ok" | "okay" | "sure" | "go" | "allow" | "approve" => {
            Some(Choice::Once)
        }
        "2" | "session" | "this session" => Some(Choice::Session),
        "3" | "always" | "don't ask again" | "dont ask again" => Some(Choice::Always),
        "4" | "n" | "no" | "nope" | "deny" | "stop" | "cancel" => Some(Choice::Deny),
        _ => None,
    }
}

/// Hands out message tokens.
///
/// One counter for the whole process rather than one per chat: a token is only
/// ever used to look up a message the same channel sent, and a single sequence
/// makes a token unambiguous in a log line.
#[derive(Debug, Default)]
pub struct Tokens(std::sync::atomic::AtomicU64);

impl Tokens {
    pub fn next(&self) -> u64 {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tapped_number_and_the_word_it_stands_for_agree() {
        // The buttons are numbered and the numbers are typeable; the two must
        // not disagree, or tapping and typing do different things.
        for (index, (word, choice)) in ANSWERS.iter().enumerate() {
            let number = (index + 1).to_string();
            assert_eq!(read_choice(&number), Some(*choice), "answer {number}");
            assert_eq!(read_choice(word), Some(*choice), "answer {word}");
        }
    }

    #[test]
    fn an_answer_survives_the_way_people_actually_type() {
        for text in [" YES ", "Yes.", "ok", "sure"] {
            assert_eq!(read_choice(text), Some(Choice::Once), "{text:?}");
        }
        for text in ["No", "nope", "  deny "] {
            assert_eq!(read_choice(text), Some(Choice::Deny), "{text:?}");
        }
    }

    #[test]
    fn anything_else_is_a_message_and_not_an_answer() {
        // The consequence of getting this wrong is running a tool because
        // someone happened to start a sentence with a word that looked like
        // consent, so the set is deliberately small and closed.
        for text in ["yes please, and also delete the file", "not now", "1st", "y'know", ""] {
            assert_eq!(read_choice(text), None, "{text:?}");
        }
    }

    #[test]
    fn identities_include_the_handle_only_when_there_is_one() {
        let mut m = Msg {
            chat: "c".into(),
            sender_id: "42".into(),
            handle: None,
            name: "Ada".into(),
            text: String::new(),
            images: Vec::new(),
            group: false,
        };
        assert_eq!(m.identities(), vec!["42"]);
        m.handle = Some("ada".into());
        assert_eq!(m.identities(), vec!["42", "ada"]);
    }

    #[test]
    fn tokens_are_never_reused() {
        let t = Tokens::default();
        let a = t.next();
        let b = t.next();
        assert_ne!(a, b);
    }
}
