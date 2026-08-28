//! Whether a tool call runs, and who decided.
//!
//! A model choosing to call `run_command` is a request, not an instruction.
//! Something has to stand between the request and the machine, and the two
//! obvious answers are both wrong on their own: asking about everything
//! trains people to hit yes without reading, and asking about nothing means
//! the first time you learn a model wrote a file is when you find the file.
//!
//! So the question is asked once per *kind* of thing, not per call. A tool
//! declares what it does to the world — [`Effect`] — and the policy answers
//! per effect, with per-tool overrides on top. Reading runs; writing and
//! running programs ask. A user who is tired of being asked about one tool
//! answers "don't ask again" and that becomes an override, which is the same
//! data the settings page edits. There is exactly one place the rules live.
//!
//! This module decides; it never prompts. The terminal and the web interface
//! ask in their own idiom, and both call [`Permissions::verdict`] first so
//! they cannot disagree about what needed asking.
//!
//! Not to be confused with `[tools.config.permissions]`, which is the Python
//! side's sandbox — the root directory a file tool is confined to, the
//! programs a shell tool may run. That answers "what may this tool touch";
//! this answers "does this call happen at all". See [`Verdict::Allow`] for
//! how the two meet.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// What a tool does to the world, as the tool itself declares it.
///
/// Declared by the tool rather than inferred here, because ozgent cannot know
/// what a user's own Python does, and a list of dangerous-sounding names in
/// Rust would be both wrong and unmaintainable. A tool that says nothing is
/// [`Effect::Unknown`], which asks — the safe reading of silence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// Looks something up and returns it. Nothing outside ozgent changes.
    Read,
    /// Creates or changes something: a file, a record, a remote resource.
    Write,
    /// Runs a program.
    Execute,
    /// The tool did not say.
    #[default]
    Unknown,
}

impl Effect {
    /// A short phrase for a prompt: "ozgent wants to *read a page*".
    pub fn describes(self) -> &'static str {
        match self {
            Self::Read => "reads",
            Self::Write => "changes files or data",
            Self::Execute => "runs a program",
            Self::Unknown => "does not say what it does",
        }
    }
}

impl std::fmt::Display for Effect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
            Self::Unknown => "unknown",
        })
    }
}

impl std::str::FromStr for Effect {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "execute" | "exec" => Ok(Self::Execute),
            "unknown" => Ok(Self::Unknown),
            other => Err(format!("expected read, write, execute or unknown; got {other:?}")),
        }
    }
}

/// What to do when a tool of a given kind is called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    /// Run it without asking.
    Allow,
    /// Ask, every time, until told otherwise.
    Ask,
    /// Refuse, and tell the model it was refused.
    Deny,
}

impl std::fmt::Display for Rule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Allow => "allow",
            Self::Ask => "ask",
            Self::Deny => "deny",
        })
    }
}

impl std::str::FromStr for Rule {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "allow" | "always" | "yes" | "on" => Ok(Self::Allow),
            "ask" | "prompt" => Ok(Self::Ask),
            "deny" | "never" | "no" | "off" => Ok(Self::Deny),
            other => Err(format!("expected allow, ask or deny; got {other:?}")),
        }
    }
}

/// The persistent policy, from `[permissions]` in `config.toml`.
///
/// ```toml
/// [permissions]
/// read = "allow"
/// write = "ask"
/// execute = "ask"
/// unknown = "ask"
///
/// [permissions.tools]
/// web_search = "allow"
/// run_command = "deny"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Permissions {
    /// Tools that only look things up. Allowed, because a permission prompt
    /// that fires on every web search is one people switch off entirely.
    pub read: Rule,
    /// Tools that change something. The first time a model writes a file
    /// should not be a surprise.
    pub write: Rule,
    /// Tools that run programs.
    pub execute: Rule,
    /// Tools that did not declare an effect — including every tool written
    /// before effects existed, which is why silence must not mean "read".
    pub unknown: Rule,

    /// Per-tool answers, which beat the rules above.
    ///
    /// This is what "don't ask again" writes, and what the settings page and
    /// `/permissions` edit. One place, three front ends.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub tools: BTreeMap<String, Rule>,
}

impl Default for Permissions {
    fn default() -> Self {
        Self {
            read: Rule::Allow,
            write: Rule::Ask,
            execute: Rule::Ask,
            unknown: Rule::Ask,
            tools: BTreeMap::new(),
        }
    }
}

/// What should happen to one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Run it.
    ///
    /// `by_user` records whether a person chose this — an override in
    /// `[permissions.tools]`, a "don't ask again" earlier in this session, or
    /// a yes just now — as opposed to the effect defaults, which nobody was
    /// asked about. The Python sandbox is told, and treats a call a person
    /// authorised as authorised: refusing to run a command the user has just
    /// read and approved, because a config flag they never saw is off, is a
    /// permission system arguing with its own user.
    Allow { by_user: bool },
    /// Ask the person in front of the terminal or the browser.
    Ask,
    /// Refuse without asking, and tell the model.
    Deny,
}

/// What the user answered, and how long the answer lasts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Choice {
    /// Run this call. Ask again next time.
    Once,
    /// Run this and every later call to this tool until ozgent exits.
    Session,
    /// Run it, and write `allow` into the config so it never asks again.
    Always,
    /// Refuse this call.
    Deny,
    /// Refuse, and write `deny` into the config.
    DenyAlways,
}

impl Choice {
    pub fn is_allow(self) -> bool {
        matches!(self, Self::Once | Self::Session | Self::Always)
    }
}

/// Answers given during this run, which outlive a single call but are never
/// written to disk.
///
/// Separate from [`Permissions`] on purpose: "yes, for now" is a different
/// promise from "yes, always", and collapsing them means a moment of
/// convenience quietly becomes a permanent setting.
#[derive(Debug, Clone, Default)]
pub struct Grants {
    allowed: BTreeSet<String>,
    denied: BTreeSet<String>,
}

impl Grants {
    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty() && self.denied.is_empty()
    }

    /// Tools allowed for the rest of this run, for a manager to display.
    pub fn allowed(&self) -> impl Iterator<Item = &str> {
        self.allowed.iter().map(String::as_str)
    }

    pub fn clear(&mut self) {
        self.allowed.clear();
        self.denied.clear();
    }

    /// Record an answer, if it was one that outlives this call.
    ///
    /// [`Choice::Always`] and [`Choice::DenyAlways`] are *not* recorded here —
    /// they belong in the config, and remembering them in both places means a
    /// user who later edits the config finds the old answer still winning.
    pub fn remember(&mut self, tool: &str, choice: Choice) {
        match choice {
            Choice::Session => {
                self.denied.remove(tool);
                self.allowed.insert(tool.to_string());
            }
            Choice::Once | Choice::Deny | Choice::Always | Choice::DenyAlways => {}
        }
    }
}

impl Permissions {
    /// The standing rule for a tool, ignoring anything decided this session.
    pub fn rule_for(&self, tool: &str, effect: Effect) -> Rule {
        if let Some(rule) = self.tools.get(tool) {
            return *rule;
        }
        match effect {
            Effect::Read => self.read,
            Effect::Write => self.write,
            Effect::Execute => self.execute,
            Effect::Unknown => self.unknown,
        }
    }

    /// Whether the rule for a tool was set by name rather than inherited.
    pub fn is_overridden(&self, tool: &str) -> bool {
        self.tools.contains_key(tool)
    }

    /// What to do about one call, given what has already been answered.
    pub fn verdict(&self, tool: &str, effect: Effect, grants: &Grants) -> Verdict {
        // A session denial outranks a session allowance; they cannot both be
        // set, but ordering the checks makes that independent of insertion.
        if grants.denied.contains(tool) {
            return Verdict::Deny;
        }
        if grants.allowed.contains(tool) {
            return Verdict::Allow { by_user: true };
        }
        match self.rule_for(tool, effect) {
            // Named in the config by a person, so it carries their authority.
            Rule::Allow => Verdict::Allow { by_user: self.is_overridden(tool) },
            Rule::Ask => Verdict::Ask,
            Rule::Deny => Verdict::Deny,
        }
    }

    /// Apply an answer that should persist beyond this call.
    ///
    /// Returns whether the config changed and so needs saving.
    pub fn apply(&mut self, tool: &str, choice: Choice) -> bool {
        match choice {
            Choice::Always => self.tools.insert(tool.to_string(), Rule::Allow) != Some(Rule::Allow),
            Choice::DenyAlways => {
                self.tools.insert(tool.to_string(), Rule::Deny) != Some(Rule::Deny)
            }
            Choice::Once | Choice::Session | Choice::Deny => false,
        }
    }

    /// Set or clear one tool's override.
    pub fn set(&mut self, tool: &str, rule: Option<Rule>) {
        match rule {
            Some(r) => {
                self.tools.insert(tool.to_string(), r);
            }
            None => {
                self.tools.remove(tool);
            }
        }
    }
}

/// What the model is told when a call is refused.
///
/// Phrased as the user's decision rather than an error, because it is not one:
/// a model told "tool failed" retries, while a model told the user declined
/// asks what to do instead.
pub fn refusal(tool: &str) -> String {
    format!(
        "The user declined to run {tool}. Do not try it again in this turn; \
         say what you were about to do and ask how they would like to proceed."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reading_runs_and_writing_asks_out_of_the_box() {
        let p = Permissions::default();
        let g = Grants::default();
        assert_eq!(p.verdict("web_search", Effect::Read, &g), Verdict::Allow { by_user: false });
        assert_eq!(p.verdict("write_file", Effect::Write, &g), Verdict::Ask);
        assert_eq!(p.verdict("run_command", Effect::Execute, &g), Verdict::Ask);
    }

    #[test]
    fn a_tool_that_says_nothing_is_asked_about() {
        // Every tool written before effects existed lands here, so silence
        // must not be read as "harmless".
        let p = Permissions::default();
        assert_eq!(p.verdict("someones_own_tool", Effect::Unknown, &Grants::default()), Verdict::Ask);
    }

    #[test]
    fn a_named_tool_beats_its_effect() {
        let mut p = Permissions::default();
        p.set("run_command", Some(Rule::Allow));
        p.set("web_search", Some(Rule::Deny));
        let g = Grants::default();

        assert_eq!(p.verdict("run_command", Effect::Execute, &g), Verdict::Allow { by_user: true });
        assert_eq!(p.verdict("web_search", Effect::Read, &g), Verdict::Deny);
    }

    #[test]
    fn the_effect_default_does_not_claim_a_users_authority() {
        // The distinction the sandbox reads: nobody was asked about this, so
        // it must not lift a boundary the user set deliberately.
        let p = Permissions::default();
        assert_eq!(
            p.verdict("read_file", Effect::Read, &Grants::default()),
            Verdict::Allow { by_user: false },
        );
    }

    #[test]
    fn dont_ask_again_lasts_the_session_without_touching_the_config() {
        let p = Permissions::default();
        let mut g = Grants::default();
        g.remember("run_command", Choice::Session);

        assert_eq!(p.verdict("run_command", Effect::Execute, &g), Verdict::Allow { by_user: true });
        assert!(p.tools.is_empty(), "a session answer must not become a permanent one");
    }

    #[test]
    fn yes_once_is_not_remembered() {
        let p = Permissions::default();
        let mut g = Grants::default();
        g.remember("run_command", Choice::Once);
        assert_eq!(p.verdict("run_command", Effect::Execute, &g), Verdict::Ask);
    }

    #[test]
    fn always_writes_a_rule_and_reports_the_change() {
        let mut p = Permissions::default();
        assert!(p.apply("run_command", Choice::Always), "the config needs saving");
        assert_eq!(p.rule_for("run_command", Effect::Execute), Rule::Allow);
        assert!(!p.apply("run_command", Choice::Always), "saving twice is wasted work");
    }

    #[test]
    fn never_writes_a_denial() {
        let mut p = Permissions::default();
        assert!(p.apply("write_file", Choice::DenyAlways));
        assert_eq!(p.verdict("write_file", Effect::Write, &Grants::default()), Verdict::Deny);
    }

    #[test]
    fn clearing_an_override_returns_the_tool_to_its_effect() {
        let mut p = Permissions::default();
        p.set("run_command", Some(Rule::Allow));
        p.set("run_command", None);
        assert_eq!(p.verdict("run_command", Effect::Execute, &Grants::default()), Verdict::Ask);
    }

    #[test]
    fn rules_and_effects_round_trip_through_their_names() {
        for rule in [Rule::Allow, Rule::Ask, Rule::Deny] {
            assert_eq!(rule.to_string().parse(), Ok(rule));
        }
        for effect in [Effect::Read, Effect::Write, Effect::Execute, Effect::Unknown] {
            assert_eq!(effect.to_string().parse(), Ok(effect));
        }
    }

    #[test]
    fn the_policy_survives_a_trip_through_toml() {
        let mut p = Permissions::default();
        p.set("run_command", Some(Rule::Deny));
        p.write = Rule::Allow;

        let text = toml::to_string_pretty(&p).unwrap();
        let back: Permissions = toml::from_str(&text).unwrap();
        assert_eq!(back.write, Rule::Allow);
        assert_eq!(back.rule_for("run_command", Effect::Execute), Rule::Deny);
    }

    #[test]
    fn an_empty_section_is_the_default_policy() {
        let p: Permissions = toml::from_str("").unwrap();
        assert_eq!(p.read, Rule::Allow);
        assert_eq!(p.execute, Rule::Ask);
    }

    #[test]
    fn the_refusal_tells_the_model_not_to_retry() {
        // Without this a model reads a refusal as a transient failure and
        // spends the rest of its tool budget trying again.
        let text = refusal("run_command");
        assert!(text.contains("run_command"));
        assert!(text.contains("not try it again"), "{text}");
    }
}
