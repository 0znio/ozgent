//! What a person can type at ozgent from a chat, besides a question.
//!
//! Kept as a pure parse so the rules are visible in one place and testable
//! without a network. Two of them decide something security-relevant and are
//! worth stating plainly:
//!
//! * `/pair` is the only directive answered for someone who is **not** on the
//!   allowlist, so it is parsed before admission is checked and does nothing
//!   at all without a code that was printed on the operator's own terminal.
//! * A leading slash is required. A model's reply, a forwarded message or a
//!   pasted document cannot become a directive by accident, and a person who
//!   wants to *ask about* `/new` can do so by writing a sentence.

/// Something to do instead of answering.
#[derive(Debug, Clone, PartialEq)]
pub enum Directive {
    /// Ordinary text to answer. Carries the message with any leading
    /// whitespace removed and nothing else changed.
    Ask(String),
    Help,
    /// Forget this chat's conversation and start a new one.
    New,
    /// Show the model in use, or change it.
    Model(Option<String>),
    /// Report the ids an allowlist would need.
    Whoami,
    /// Stop the turn in flight.
    Stop,
    /// List the tools this chat may use.
    Tools,
    /// Ask to be admitted, with the code from the operator's terminal.
    Pair(String),
    /// A slash word that is not a directive.
    Unknown(String),
}

/// Read a message as a directive.
pub fn parse(text: &str) -> Directive {
    let text = text.trim();
    if !text.starts_with('/') {
        return Directive::Ask(text.to_string());
    }

    let mut parts = text[1..].splitn(2, char::is_whitespace);
    let word = parts.next().unwrap_or("").to_ascii_lowercase();
    // Telegram addresses a bot in a group as `/new@ozgent_bot`. The suffix is
    // routing, not part of the word, and a group is exactly where directives
    // are most likely to be typed.
    let word = word.split('@').next().unwrap_or("").to_string();
    let rest = parts.next().unwrap_or("").trim().to_string();

    match word.as_str() {
        "help" | "start" | "?" => Directive::Help,
        "new" | "reset" | "clear" => Directive::New,
        "model" => Directive::Model((!rest.is_empty()).then_some(rest)),
        "whoami" | "id" => Directive::Whoami,
        "stop" | "cancel" | "abort" => Directive::Stop,
        "tools" => Directive::Tools,
        "pair" => Directive::Pair(rest),
        // A message that happens to begin with a slash rather than a
        // mistyped directive: an empty word, or one with another slash in it,
        // which is a path and not a command anybody meant to type.
        "" => Directive::Ask(text.to_string()),
        w if w.contains('/') => Directive::Ask(text.to_string()),
        other => Directive::Unknown(other.to_string()),
    }
}

/// The reply to `/help`, as markdown.
pub fn help(model: &str) -> String {
    format!(
        "I am **ozgent**, running a local model on someone's own machine. \
         Send a message and I answer it; I keep the thread, so you can refer back.\n\
         \n\
         Model in use: `{model}`\n\
         \n\
         - `/new` — forget this thread and start fresh\n\
         - `/model` — which model is answering; `/model <name>` to change it\n\
         - `/tools` — what I am allowed to use here\n\
         - `/stop` — stop what I am writing\n\
         - `/whoami` — the ids an allowlist needs\n\
         \n\
         When I want to use a tool that changes something, I will ask first. \
         Answer with the buttons, or type `yes`, `session`, `always` or `no`."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_a_question() {
        assert_eq!(parse("what is a monad?"), Directive::Ask("what is a monad?".into()));
        assert_eq!(parse("  hello  "), Directive::Ask("hello".into()));
    }

    #[test]
    fn directives_are_recognised_with_their_aliases() {
        assert_eq!(parse("/new"), Directive::New);
        assert_eq!(parse("/reset"), Directive::New);
        assert_eq!(parse("/CLEAR"), Directive::New);
        assert_eq!(parse("/help"), Directive::Help);
        assert_eq!(parse("/start"), Directive::Help);
        assert_eq!(parse("/stop"), Directive::Stop);
        assert_eq!(parse("/whoami"), Directive::Whoami);
        assert_eq!(parse("/tools"), Directive::Tools);
    }

    #[test]
    fn a_bot_suffix_is_routing_and_not_part_of_the_word() {
        // How Telegram delivers a directive typed in a group.
        assert_eq!(parse("/new@ozgent_bot"), Directive::New);
        assert_eq!(parse("/model@ozgent_bot coder"), Directive::Model(Some("coder".into())));
    }

    #[test]
    fn model_takes_an_optional_argument() {
        assert_eq!(parse("/model"), Directive::Model(None));
        assert_eq!(parse("/model  coder "), Directive::Model(Some("coder".into())));
    }

    #[test]
    fn pairing_carries_its_code() {
        assert_eq!(parse("/pair 4821"), Directive::Pair("4821".into()));
        assert_eq!(parse("/pair"), Directive::Pair(String::new()));
    }

    #[test]
    fn only_a_leading_slash_makes_a_directive() {
        // Otherwise a quoted message, a pasted document or the model's own
        // reply could become a command.
        assert_eq!(parse("please run /new for me"), Directive::Ask("please run /new for me".into()));
        assert_eq!(parse("what does /stop do?"), Directive::Ask("what does /stop do?".into()));
    }

    #[test]
    fn a_path_is_not_a_mistyped_directive() {
        assert_eq!(parse("/etc/hosts"), Directive::Ask("/etc/hosts".into()));
        assert_eq!(parse("/usr/bin/env python"), Directive::Ask("/usr/bin/env python".into()));
        assert_eq!(parse("/ leading slash"), Directive::Ask("/ leading slash".into()));
    }

    #[test]
    fn an_unrecognised_slash_word_is_reported_rather_than_answered() {
        // Answering it would send a puzzled model a command it cannot run;
        // saying so lets the person try again.
        assert_eq!(parse("/deploy now"), Directive::Unknown("deploy".into()));
    }

    #[test]
    fn help_names_the_model_actually_in_use() {
        let text = help("qwythos:Q4_K_M");
        assert!(text.contains("qwythos:Q4_K_M"));
        assert!(text.contains("/new"));
    }
}
