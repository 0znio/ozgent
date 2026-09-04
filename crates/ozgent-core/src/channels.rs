//! Messaging channels: `[channels]` in `config.toml`.
//!
//! A channel is a way for a person to reach ozgent from somewhere that is not
//! this machine. That single sentence is the whole reason this module is more
//! careful than the rest of the configuration.
//!
//! Every other surface ozgent offers is reached by someone already at the
//! keyboard, or on the local network the operator chose to bind to. A bot
//! handle is reachable by anyone in the world who types it, and behind it sit
//! `run_command` and `write_file`. So the default here is not "on with sensible
//! settings" — it is **off**, and once on, **nobody is admitted**. Both have to
//! be undone deliberately, and the only paths that undo the second are the
//! operator editing this file or reading a pairing code off their own terminal.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Everything under `[channels]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChannelsConfig {
    /// Master switch. Off means `ozgent gateway` refuses to start and
    /// `ozgent web` starts no channel, whatever the per-channel settings say.
    pub enabled: bool,

    /// Model used for messages that arrive over a channel.
    ///
    /// Separate from `default_model` because the trade-off differs: a phone is
    /// a poor place to wait on a 70B, and the machine may be doing something
    /// else. Falls back to `default_model` when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    pub telegram: Telegram,
    pub whatsapp: WhatsApp,
}

impl ChannelsConfig {
    /// The channels that are both configured and switched on.
    pub fn active(&self) -> Vec<Kind> {
        if !self.enabled {
            return Vec::new();
        }
        let mut out = Vec::new();
        if self.telegram.enabled && !self.telegram.token.trim().is_empty() {
            out.push(Kind::Telegram);
        }
        if self.whatsapp.enabled {
            out.push(Kind::WhatsApp);
        }
        out
    }

    pub fn access(&self, kind: Kind) -> Access<'_> {
        match kind {
            Kind::Telegram => Access {
                allow: &self.telegram.allow,
                tools: self.telegram.tools.as_deref(),
                stream: self.telegram.stream,
            },
            Kind::WhatsApp => Access {
                allow: &self.whatsapp.allow,
                tools: self.whatsapp.tools.as_deref(),
                stream: self.whatsapp.stream,
            },
        }
    }

    /// Admit an identity to a channel. True when it was not already allowed.
    ///
    /// The write side of pairing. Kept here rather than in the gateway so that
    /// the one operation that widens who can reach this machine lives next to
    /// the rules that decide it.
    pub fn admit(&mut self, kind: Kind, identity: &str) -> bool {
        let allow = match kind {
            Kind::Telegram => &mut self.telegram.allow,
            Kind::WhatsApp => &mut self.whatsapp.allow,
        };
        let identity = identity.trim();
        if identity.is_empty() || allow.iter().any(|a| a.eq_ignore_ascii_case(identity)) {
            return false;
        }
        allow.push(identity.to_string());
        true
    }
}

/// Which channel. A closed set: each variant is a transport with its own
/// process and its own credentials, not something a config file can invent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Telegram,
    WhatsApp,
}

impl Kind {
    /// The name used as the `channel` column in the store, in log lines, and
    /// on the command line. Stable: changing it orphans every bound chat.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Telegram => "telegram",
            Self::WhatsApp => "whatsapp",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "telegram" | "tg" => Some(Self::Telegram),
            "whatsapp" | "wa" => Some(Self::WhatsApp),
            _ => None,
        }
    }
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The parts of a channel's configuration the gateway needs, whichever
/// channel it is. Borrowed rather than cloned: it is read once per message.
#[derive(Debug, Clone, Copy)]
pub struct Access<'a> {
    pub allow: &'a [String],
    pub tools: Option<&'a [String]>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Telegram {
    pub enabled: bool,

    /// The bot token from @BotFather.
    ///
    /// Whoever holds it can read every message sent to the bot, so it is
    /// treated like a password: never logged, never shown in `ozgent config`,
    /// and `$OZGENT_TELEGRAM_TOKEN` overrides it for anyone who would rather
    /// not have it on disk at all.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub token: String,

    /// Who may talk to the bot: numeric user ids, or `@username`.
    ///
    /// Empty admits nobody. `"*"` admits everyone, which hands the internet
    /// whatever your permission rules allow — the gateway says so at startup
    /// every single time.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,

    /// Tools offered to messages from this channel. `None` offers the same set
    /// every other surface gets; a list withholds everything not named.
    ///
    /// Worth having even though the permission layer already asks: consent
    /// arriving over a chat is consent from whoever holds that account, which
    /// is not necessarily the person who owns this machine.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,

    /// Edit one message as the reply is generated, instead of sending it whole
    /// when it is finished.
    pub stream: bool,
}

impl Default for Telegram {
    fn default() -> Self {
        Self {
            enabled: false,
            token: String::new(),
            allow: Vec::new(),
            tools: None,
            stream: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WhatsApp {
    pub enabled: bool,

    /// Who may talk to it: phone numbers in international form, digits only,
    /// or full JIDs. Empty admits nobody; `"*"` admits everyone.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,

    pub stream: bool,

    /// Interpreter used to run the bridge. WhatsApp has no documented protocol
    /// and no Rust client; the bridge is a small Node program driving the same
    /// library every other self-hosted WhatsApp bot uses.
    pub node: String,

    /// Where the bridge lives. Found beside the executable when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bridge: Option<PathBuf>,

    /// Answer chats you are in but were not addressed in — group chats.
    ///
    /// Off by default and deliberately awkward to turn on: a bot that replies
    /// to everything it can see in a group is both a nuisance and a way for
    /// someone who was never allowlisted to steer it through a member who was.
    pub groups: bool,
}

impl Default for WhatsApp {
    fn default() -> Self {
        Self {
            enabled: false,
            allow: Vec::new(),
            tools: None,
            stream: true,
            node: "node".into(),
            bridge: None,
            groups: false,
        }
    }
}

/// Whether an identity is allowed to talk to a channel.
///
/// `identities` is every name the sender goes by that the caller could
/// establish — a numeric id and a username on Telegram, a phone number and a
/// JID on WhatsApp — because an operator writes down whichever one they know,
/// and a rule that only matched the other one would look like a broken
/// allowlist rather than a mistyped one.
///
/// Matching is case-insensitive, and a leading `@` is optional on both sides,
/// so `@Ada`, `ada` and `Ada` are one rule and not three.
pub fn admits(allow: &[String], identities: &[&str]) -> bool {
    let norm = |s: &str| s.trim().trim_start_matches('@').to_ascii_lowercase();
    for rule in allow {
        let rule = rule.trim();
        if rule == "*" {
            return true;
        }
        // An empty entry is a typo — a stray comma, a blank line in an edited
        // list. Matching it against an empty identity would admit a sender
        // whose username simply is not set.
        if rule.is_empty() {
            continue;
        }
        let rule = norm(rule);
        if identities.iter().any(|id| !id.trim().is_empty() && norm(id) == rule) {
            return true;
        }
    }
    false
}

/// Whether a channel's allowlist admits the entire internet.
pub fn is_open_to_everyone(allow: &[String]) -> bool {
    allow.iter().any(|a| a.trim() == "*")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nobody_is_admitted_by_default() {
        // The property the whole module exists for. A channel that is on but
        // unconfigured must not be an open door.
        assert!(!admits(&[], &["4242", "ada"]));
        assert!(!admits(&Telegram::default().allow, &["4242"]));
    }

    #[test]
    fn an_identity_matches_by_id_or_by_handle() {
        let allow = vec!["4242".to_string()];
        assert!(admits(&allow, &["4242", "ada"]));

        let allow = vec!["@ada".to_string()];
        assert!(admits(&allow, &["4242", "ada"]));
    }

    #[test]
    fn the_at_sign_and_case_are_not_part_of_the_rule() {
        // Someone writes down what they see in the Telegram UI, which shows a
        // handle with an @ and whatever capitalisation the owner chose.
        for rule in ["@Ada", "ada", "ADA", " @ada "] {
            assert!(admits(&[rule.to_string()], &["ada"]), "rule {rule:?}");
            assert!(admits(&[rule.to_string()], &["@Ada"]), "rule {rule:?}");
        }
    }

    #[test]
    fn a_blank_rule_does_not_admit_a_sender_without_a_handle() {
        // Telegram users need not have a username, so the handle is empty for
        // many senders. A stray comma in the list must not become a wildcard.
        let allow = vec![String::new(), "  ".to_string()];
        assert!(!admits(&allow, &["4242", ""]));
    }

    #[test]
    fn a_star_admits_everyone_and_says_so() {
        let allow = vec!["*".to_string()];
        assert!(admits(&allow, &["anyone"]));
        assert!(is_open_to_everyone(&allow));
        assert!(!is_open_to_everyone(&["4242".to_string()]));
    }

    #[test]
    fn a_near_miss_is_not_a_match() {
        let allow = vec!["4242".to_string()];
        assert!(!admits(&allow, &["42", "424242"]));
    }

    #[test]
    fn admitting_is_idempotent_and_case_insensitive() {
        let mut c = ChannelsConfig::default();
        assert!(c.admit(Kind::Telegram, "@Ada"));
        assert!(!c.admit(Kind::Telegram, "@ada"), "already there in another case");
        assert!(!c.admit(Kind::Telegram, "   "), "nothing to add");
        assert_eq!(c.telegram.allow, vec!["@Ada".to_string()]);
        assert!(c.whatsapp.allow.is_empty(), "channels do not share a list");
    }

    #[test]
    fn nothing_is_active_until_the_master_switch_is_on() {
        let mut c = ChannelsConfig::default();
        c.telegram.enabled = true;
        c.telegram.token = "t".into();
        assert!(c.active().is_empty(), "[channels] enabled is still false");

        c.enabled = true;
        assert_eq!(c.active(), vec![Kind::Telegram]);
    }

    #[test]
    fn telegram_without_a_token_is_not_active() {
        // Enabled but tokenless is a half-finished setup, not a channel. It
        // would otherwise fail on every poll and fill the log.
        let mut c = ChannelsConfig { enabled: true, ..Default::default() };
        c.telegram.enabled = true;
        c.telegram.token = "   ".into();
        assert!(c.active().is_empty());
    }

    #[test]
    fn the_documented_configuration_parses() {
        // `Config` denies unknown fields, so a key documented under a name it
        // does not have is not a stale doc — it is a config file that refuses
        // to load, and ozgent will not start at all.
        let text = r#"
            [channels]
            enabled = true
            model = "coder"

            [channels.telegram]
            enabled = true
            token = "123456:AA"
            allow = ["@ada", "4242"]
            tools = ["web_search", "fetch_url"]
            stream = true

            [channels.whatsapp]
            enabled = true
            allow = ["15551234567"]
            stream = false
            groups = true
            node = "/opt/node/bin/node"
            bridge = "/srv/ozgent/bridge/whatsapp"
        "#;
        let config: crate::Config = toml::from_str(text).expect("the documented settings");
        assert_eq!(config.channels.model.as_deref(), Some("coder"));
        assert_eq!(config.channels.telegram.allow.len(), 2);
        assert!(!config.channels.whatsapp.stream);
        assert!(config.channels.whatsapp.groups);
        assert_eq!(config.channels.active(), vec![Kind::Telegram, Kind::WhatsApp]);
    }

    #[test]
    fn an_empty_channels_section_is_valid_and_admits_nobody() {
        let config: crate::Config = toml::from_str("[channels]
").expect("an empty section");
        assert!(!config.channels.enabled);
        assert!(config.channels.active().is_empty());
    }

    #[test]
    fn a_saved_configuration_reads_back_the_same() {
        // Round-tripped rather than only parsed: `skip_serializing_if` on a
        // field ozgent then reads is how a setting silently stops persisting.
        let mut config = crate::Config::default();
        config.channels.enabled = true;
        config.channels.telegram.enabled = true;
        config.channels.telegram.token = "123456:AA".into();
        config.channels.telegram.allow = vec!["@ada".into()];
        config.channels.telegram.tools = Some(vec!["web_search".into()]);
        config.channels.whatsapp.groups = true;

        let text = toml::to_string(&config).expect("serialising");
        let back: crate::Config = toml::from_str(&text).expect("parsing what we wrote");
        assert_eq!(back.channels.telegram.token, "123456:AA");
        assert_eq!(back.channels.telegram.allow, vec!["@ada".to_string()]);
        assert_eq!(back.channels.telegram.tools.as_deref(), Some(&["web_search".to_string()][..]));
        assert!(back.channels.whatsapp.groups);
        assert_eq!(back.channels.active(), vec![Kind::Telegram]);
    }

    #[test]
    fn a_kind_round_trips_through_its_name() {
        for kind in [Kind::Telegram, Kind::WhatsApp] {
            assert_eq!(Kind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(Kind::parse("  TELEGRAM "), Some(Kind::Telegram));
        assert_eq!(Kind::parse("signal"), None);
    }
}
