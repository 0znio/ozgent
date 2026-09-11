//! Agents: a named job with its own instructions and its own tools.
//!
//! Writing `@stock-guru what about NVDA?` hands the message to that agent. It
//! runs with the instructions it was written with instead of the chat's own
//! system prompt, and it is offered **only** the tools it lists. That second
//! part is the point of an agent rather than a decoration on it: a research
//! agent that cannot see `write_file` cannot be talked into writing one, which
//! is a stronger promise than an instruction asking it not to.
//!
//! Agents are TOML files in `~/ozgent/agents/<name>.toml`:
//!
//! ```toml
//! description = "Analyses a stock"            # shown in the @ panel
//! tools = ["yahoo_finance", "web_search"]     # the only tools it may call
//! max_rounds = 10                              # tool rounds before it must answer
//! thinking = "auto"                            # auto | on | off
//! temperature = 0.3
//! instructions = """What the agent is for and how it should work."""
//!
//! [permissions]                                # per tool: allow | ask | deny
//! yahoo_finance = "allow"
//! ```
//!
//! A few ship built in. A file with the same name replaces a built-in one, so
//! editing a built-in agent is copying it and changing the copy — and
//! deleting that copy brings the original back.

use crate::permission::{Effect, Grants, Permissions, Rule, Verdict};
use crate::{Paths, ThinkingMode, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The agents compiled into the binary, as `(name, toml)`.
const BUILTIN: &[(&str, &str)] = &[
    ("deep-researcher", include_str!("../agents/deep-researcher.toml")),
    ("stock-guru", include_str!("../agents/stock-guru.toml")),
    ("sentiment-analyser", include_str!("../agents/sentiment-analyser.toml")),
];

/// Tool rounds an agent gets when its file does not say.
pub const DEFAULT_ROUNDS: usize = 8;

/// The most rounds an agent may ask for.
///
/// Each round is a whole generation plus the tools it called, so this bounds
/// how long one `@mention` can hold the model. Past this a task is better
/// split into two agents than given one that runs for ten minutes.
pub const MAX_ROUNDS: usize = 32;

/// The longest name, which keeps the suggestion panel one line per agent.
pub const MAX_NAME: usize = 32;

/// How many agents one message may call. Each runs in turn with the model
/// held, so a message naming ten of them would hold it for a very long time.
pub const MAX_PER_MESSAGE: usize = 3;

/// What an agent file holds. The name is the file's, not a field.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Definition {
    /// One line, shown next to the name wherever agents are listed.
    pub description: String,
    /// What the agent is for and how it should go about it. Becomes its
    /// system prompt.
    pub instructions: String,
    /// The only tools it is offered, by name.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    /// Per-tool rules for this agent, applied on top of the global policy.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub permissions: BTreeMap<String, Rule>,
    /// Tool rounds before it must answer. Absent means [`DEFAULT_ROUNDS`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_rounds: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Longest reply, in tokens. Absent inherits the model's setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

/// Where an agent came from, which decides what deleting it means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// Compiled in, and not overridden.
    Builtin,
    /// A file in the agents directory with no built-in of the same name.
    User,
    /// A file replacing a built-in. Deleting it restores the original.
    Override,
}

/// An agent, ready to run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Agent {
    pub name: String,
    pub origin: Origin,
    #[serde(flatten)]
    pub definition: Definition,
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("{0}")]
    Invalid(String),
    #[error("{name}: {source}")]
    Parse { name: String, source: toml::de::Error },
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("no agent named {0:?}")]
    NotFound(String),
    #[error("{0} is built in; there is nothing to delete. Save your own version to change it")]
    Builtin(String),
}

/// Check a name is one a `@mention` can reach.
///
/// Lowercase letters, digits and single hyphens, starting with a letter. The
/// restriction is what makes a mention unambiguous: `@stock-guru,` ends at the
/// comma, and an email address cannot be mistaken for one because its `@` is
/// not preceded by a space.
pub fn validate_name(name: &str) -> Result<(), AgentError> {
    let bad = |why: &str| Err(AgentError::Invalid(format!("agent name {name:?} {why}")));
    if name.is_empty() {
        return bad("is empty");
    }
    if name.len() > MAX_NAME {
        return bad(&format!("is longer than {MAX_NAME} characters"));
    }
    if !name.starts_with(|c: char| c.is_ascii_lowercase()) {
        return bad("must start with a lowercase letter");
    }
    if !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
        return bad("may only use lowercase letters, digits and hyphens");
    }
    if name.ends_with('-') || name.contains("--") {
        return bad("may not end with a hyphen or contain two in a row");
    }
    Ok(())
}

impl Definition {
    /// Refuse a definition that could not run as written.
    pub fn validate(&self, name: &str) -> Result<(), AgentError> {
        let bad = |why: String| Err(AgentError::Invalid(format!("agent {name}: {why}")));
        if self.instructions.trim().is_empty() {
            return bad("instructions are empty; say what the agent is for".into());
        }
        if self.description.trim().is_empty() {
            return bad("description is empty; it is what the @ panel shows".into());
        }
        if self.description.contains('\n') {
            return bad("description must be one line".into());
        }
        if let Some(rounds) = self.max_rounds {
            if rounds == 0 || rounds > MAX_ROUNDS {
                return bad(format!("max_rounds must be between 1 and {MAX_ROUNDS}"));
            }
        }
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return bad("temperature must be between 0 and 2".into());
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for tool in &self.tools {
            if tool.trim().is_empty() {
                return bad("a tool name is empty".into());
            }
            if !seen.insert(tool.as_str()) {
                return bad(format!("{tool} is listed twice"));
            }
        }
        // A rule for a tool the agent cannot call would read as a grant that
        // does something, and it does nothing.
        if let Some(stray) = self.permissions.keys().find(|t| !self.tools.contains(t)) {
            return bad(format!(
                "has a permission for {stray}, which is not in its tools list"
            ));
        }
        Ok(())
    }

    pub fn rounds(&self) -> usize {
        self.max_rounds.unwrap_or(DEFAULT_ROUNDS).clamp(1, MAX_ROUNDS)
    }
}

impl Agent {
    /// Parse and check one agent file.
    pub fn parse(name: &str, text: &str, origin: Origin) -> Result<Self, AgentError> {
        validate_name(name)?;
        let definition: Definition = toml::from_str(text)
            .map_err(|source| AgentError::Parse { name: name.to_string(), source })?;
        definition.validate(name)?;
        Ok(Self { name: name.to_string(), origin, definition })
    }

    /// The tools this agent may be offered, out of those that exist.
    ///
    /// Also returns the names it lists that do not exist here — a tool that
    /// is switched off, or an MCP server that is not running. Those are worth
    /// saying out loud: an agent quietly missing its main tool answers from
    /// memory and looks like it worked.
    pub fn offer(&self, available: &[ToolSpec]) -> (Vec<ToolSpec>, Vec<String>) {
        let offered: Vec<ToolSpec> = self
            .definition
            .tools
            .iter()
            .filter_map(|name| available.iter().find(|s| &s.name == name).cloned())
            .collect();
        let missing = self
            .definition
            .tools
            .iter()
            .filter(|name| !available.iter().any(|s| &s.name == *name))
            .cloned()
            .collect();
        (offered, missing)
    }

    /// Whether a call this agent made may run.
    ///
    /// Layered, and the order is the policy:
    ///
    /// 1. A tool the agent does not list is refused. It was never offered, so
    ///    a call to it is the model guessing a name, and running a guess would
    ///    make the list meaningless.
    /// 2. A refusal in the global policy stands. The agent was written by
    ///    whoever wrote it; the global `deny` was written by the person who
    ///    owns this machine, and an agent must not be a way around it.
    /// 3. The agent's own rule for the tool, if it has one.
    /// 4. Otherwise the global verdict, grants included.
    ///
    /// An agent's `allow` counts as the user's authority only when the user
    /// wrote the agent. A built-in agent's `allow` is ozgent's default, which
    /// nobody was asked about, so it runs the tool inside the sandbox's
    /// standing limits rather than past them.
    pub fn verdict(
        &self,
        global: &Permissions,
        tool: &str,
        effect: Effect,
        grants: &Grants,
    ) -> Verdict {
        if !self.definition.tools.iter().any(|t| t == tool) {
            return Verdict::Deny;
        }
        let standing = global.verdict(tool, effect, grants);
        if standing == Verdict::Deny || global.rule_for(tool, effect) == Rule::Deny {
            return Verdict::Deny;
        }
        match self.definition.permissions.get(tool) {
            Some(Rule::Deny) => Verdict::Deny,
            Some(Rule::Allow) => Verdict::Allow { by_user: self.origin != Origin::Builtin },
            // Asked for by the agent, but a "yes for this session" already
            // given still counts: it was the same person, about the same tool.
            Some(Rule::Ask) => match standing {
                Verdict::Allow { by_user: true } => standing,
                _ => Verdict::Ask,
            },
            None => standing,
        }
    }

    /// The system prompt the agent runs with.
    ///
    /// Names the agent and its situation before its own instructions, so an
    /// agent written as a bare list of steps still knows it is answering one
    /// request inside a larger conversation and has a fixed set of tools.
    pub fn system_prompt(&self, date_line: Option<&str>, missing: &[String]) -> String {
        let mut out = String::new();
        if let Some(line) = date_line {
            out.push_str(line);
            out.push_str("\n\n");
        }
        out.push_str(&format!(
            "You are @{}, an agent: {}.\nThe user called you by name to handle \
             their latest message. The earlier conversation is there for context. \
             Do the job below, then give your report as your reply; it is shown \
             to the user as your answer.\n",
            self.name,
            self.definition.description.trim().trim_end_matches('.'),
        ));
        if !missing.is_empty() {
            out.push_str(&format!(
                "These tools are part of your job but are not available right now: {}. \
                 Work with what you have and say in your report what you could not check.\n",
                missing.join(", ")
            ));
        }
        out.push('\n');
        out.push_str(self.definition.instructions.trim());
        out
    }

    /// The file this agent would be saved to.
    pub fn path(paths: &Paths, name: &str) -> PathBuf {
        paths.agents_dir().join(format!("{name}.toml"))
    }
}

/// Every agent, built-in and user, with user files winning.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    agents: Vec<Agent>,
    /// Files that could not be read, as messages. Reported, not fatal: one
    /// broken agent must not take the others away.
    pub errors: Vec<String>,
}

impl Catalog {
    /// Read the built-ins and the agents directory.
    pub fn load(paths: &Paths) -> Self {
        let mut catalog = Self::builtin();
        let dir = paths.agents_dir();
        let Ok(entries) = std::fs::read_dir(&dir) else { return catalog };

        let mut files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .collect();
        files.sort();

        for path in files {
            let Some(name) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
                continue;
            };
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => {
                    catalog.errors.push(format!("{}: {e}", path.display()));
                    continue;
                }
            };
            let origin = if BUILTIN.iter().any(|(n, _)| *n == name) {
                Origin::Override
            } else {
                Origin::User
            };
            match Agent::parse(&name, &text, origin) {
                Ok(agent) => catalog.insert(agent),
                // A broken override leaves the built-in in place, which is
                // the version that is known to work.
                Err(e) => catalog.errors.push(format!("{}: {e}", path.display())),
            }
        }
        catalog
    }

    /// Only the agents that ship with ozgent.
    pub fn builtin() -> Self {
        let mut agents: Vec<Agent> = BUILTIN
            .iter()
            .map(|(name, text)| {
                Agent::parse(name, text, Origin::Builtin)
                    .unwrap_or_else(|e| panic!("built-in agent {name} is invalid: {e}"))
            })
            .collect();
        agents.sort_by(|a, b| a.name.cmp(&b.name));
        Self { agents, errors: Vec::new() }
    }

    fn insert(&mut self, agent: Agent) {
        match self.agents.iter_mut().find(|a| a.name == agent.name) {
            Some(slot) => *slot = agent,
            None => self.agents.push(agent),
        }
        self.agents.sort_by(|a, b| a.name.cmp(&b.name));
    }

    pub fn all(&self) -> &[Agent] {
        &self.agents
    }

    pub fn get(&self, name: &str) -> Option<&Agent> {
        self.agents.iter().find(|a| a.name == name)
    }

    /// Agents whose name starts with `prefix`, for a suggestion panel.
    ///
    /// Prefix matches first, then names that merely contain it, so typing
    /// `@guru` still finds `stock-guru`.
    pub fn suggest(&self, prefix: &str) -> Vec<&Agent> {
        let prefix = prefix.to_ascii_lowercase();
        let mut starts: Vec<&Agent> =
            self.agents.iter().filter(|a| a.name.starts_with(&prefix)).collect();
        let contains = self
            .agents
            .iter()
            .filter(|a| !a.name.starts_with(&prefix) && a.name.contains(&prefix));
        starts.extend(contains);
        starts
    }

    /// The agents a message calls, in the order it names them.
    pub fn mentioned(&self, text: &str) -> Vec<&Agent> {
        let mut out: Vec<&Agent> = Vec::new();
        for mention in mentions(text) {
            if let Some(agent) = self.get(&mention.name) {
                if !out.iter().any(|a| a.name == agent.name) {
                    out.push(agent);
                }
            }
            if out.len() == MAX_PER_MESSAGE {
                break;
            }
        }
        out
    }
}

/// Write an agent to the agents directory.
pub fn save(paths: &Paths, name: &str, definition: &Definition) -> Result<PathBuf, AgentError> {
    validate_name(name)?;
    definition.validate(name)?;
    let dir = paths.agents_dir();
    std::fs::create_dir_all(&dir).map_err(|source| AgentError::Io { path: dir.clone(), source })?;
    let path = Agent::path(paths, name);
    let text = toml::to_string_pretty(definition)
        .map_err(|e| AgentError::Invalid(format!("agent {name}: {e}")))?;
    // Written beside and renamed over, so an editor or a second ozgent reading
    // the directory never sees half a file.
    let partial = path.with_extension("toml.partial");
    std::fs::write(&partial, text).map_err(|source| AgentError::Io { path: partial.clone(), source })?;
    std::fs::rename(&partial, &path).map_err(|source| AgentError::Io { path: path.clone(), source })?;
    Ok(path)
}

/// Delete a user agent, or an override (which restores the built-in).
pub fn remove(paths: &Paths, name: &str) -> Result<Origin, AgentError> {
    validate_name(name)?;
    let path = Agent::path(paths, name);
    if !path.exists() {
        return Err(if BUILTIN.iter().any(|(n, _)| *n == name) {
            AgentError::Builtin(name.to_string())
        } else {
            AgentError::NotFound(name.to_string())
        });
    }
    std::fs::remove_file(&path).map_err(|source| AgentError::Io { path: path.clone(), source })?;
    Ok(if BUILTIN.iter().any(|(n, _)| *n == name) { Origin::Override } else { Origin::User })
}

/// One `@name` in a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mention {
    pub name: String,
    /// Byte range of the whole mention, `@` included.
    pub start: usize,
    pub end: usize,
}

/// Every `@name` in a text that is shaped like a mention.
///
/// A mention starts at the beginning of the text or after whitespace or an
/// opening bracket, so `me@example.com` is not one. It ends at the first
/// character a name cannot contain, so trailing punctuation is left alone.
/// Whether a name is actually an agent is the catalog's question.
pub fn mentions(text: &str) -> Vec<Mention> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'@' {
            let boundary = i == 0
                || text[..i]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_whitespace() || "([{\"'".contains(c));
            if boundary {
                let rest = &text[i + 1..];
                let len = rest
                    .find(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'))
                    .unwrap_or(rest.len());
                let name = rest[..len].trim_end_matches('-');
                if !name.is_empty() && validate_name(name).is_ok() {
                    out.push(Mention {
                        name: name.to_string(),
                        start: i,
                        end: i + 1 + name.len(),
                    });
                    i += 1 + name.len();
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// The mention being typed at the end of `before_caret`, if there is one.
///
/// Returns where it starts and what has been typed after the `@`, which may be
/// empty — a bare `@` is exactly when the panel should open with everything.
pub fn typing_mention(before_caret: &str) -> Option<(usize, &str)> {
    let at = before_caret.rfind('@')?;
    let boundary = at == 0
        || before_caret[..at]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_whitespace() || "([{\"'".contains(c));
    if !boundary {
        return None;
    }
    let typed = &before_caret[at + 1..];
    typed
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        .then_some((at, typed))
}

// ------------------------------------------------------------------ handoff

/// The tool the main model uses to pass a request to an agent itself.
///
/// Calling it is the model typing the `@mention` for the user: the rest of
/// the turn goes to that agent, with its own instructions and only its own
/// tools, and its report is the reply — labelled with its name, exactly as a
/// mention would be. It is not a way for the main model to borrow the agent's
/// tools: those stay the agent's.
pub const HANDOFF_TOOL: &str = "ask_agent";

/// The `ask_agent` tool, describing every agent the model may hand to.
/// `None` when there are none.
pub fn handoff_spec(agents: &[Agent]) -> Option<crate::ToolSpec> {
    if agents.is_empty() {
        return None;
    }
    let list: Vec<String> = agents
        .iter()
        .map(|a| format!("- {}: {}", a.name, a.definition.description))
        .collect();
    let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
    Some(crate::ToolSpec {
        name: HANDOFF_TOOL.to_string(),
        description: format!(
            "Hand the user's request to a specialist agent, who does the work with its own \
             tools and a tested method, and answers the user directly in your place. When the \
             request is one of the jobs below, hand it over instead of doing that job yourself \
             with your own tools — the agent does it more thoroughly. Answer small talk and \
             questions you can settle in one step yourself. Agents:\n{}",
            list.join("\n")
        ),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": names,
                    "description": "Which agent.",
                },
                "task": {
                    "type": "string",
                    "description": "What the agent should do, with everything it needs from the conversation — names, tickers, dates, what the user wants back.",
                },
            },
            "required": ["agent", "task"],
        }),
        output_schema: None,
        effect: crate::permission::Effect::Read,
    })
}

/// Read an `ask_agent` call: which agent, and the task. An error says what
/// was wrong, for the model to correct.
pub fn read_handoff<'a>(
    arguments: &serde_json::Value,
    agents: &'a [Agent],
) -> Result<(&'a Agent, String), String> {
    let name = arguments.get("agent").and_then(|v| v.as_str()).unwrap_or("").trim();
    let name = name.trim_start_matches('@');
    let agent = agents.iter().find(|a| a.name == name).ok_or_else(|| {
        let names: Vec<&str> = agents.iter().map(|a| a.name.as_str()).collect();
        format!("no agent called {name:?}. The agents are: {}", names.join(", "))
    })?;
    let task = arguments.get("task").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    Ok((agent, task))
}

/// The system prompt's line about handing off, when `ask_agent` is offered.
///
/// A tool description alone was not enough: a small model given both the
/// agent and the agent's tools reached for the tools every time. Naming the
/// agents where the model reads its instructions is what makes the choice one
/// it actually weighs.
pub fn handoff_prompt(agents: &[Agent]) -> String {
    let list: Vec<String> = agents
        .iter()
        .map(|a| format!("- {}: {}", a.name, a.definition.description))
        .collect();
    format!(
        "Specialist agents can take a request off your hands with the {HANDOFF_TOOL} tool:\n{}\n\
         When the user's request is one of these jobs, call {HANDOFF_TOOL} with that agent and a \
         clear task instead of doing the job yourself with your own tools. Otherwise answer \
         yourself.",
        list.join("\n")
    )
}

/// What the agent is told about why it is running, when the model handed the
/// request to it rather than the user naming it.
pub fn handoff_note(task: &str) -> String {
    if task.is_empty() {
        "The assistant handed the user's latest request to you. Do it.".to_string()
    } else {
        format!("The assistant handed the user's latest request to you, with this task:\n{task}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_handoff_tool_names_every_agent_and_reads_back() {
        let catalog = Catalog::builtin();
        let spec = handoff_spec(catalog.all()).expect("built-in agents exist");
        assert_eq!(spec.name, HANDOFF_TOOL);
        for a in catalog.all() {
            assert!(spec.description.contains(&a.name), "{}", a.name);
        }
        let (agent, task) =
            read_handoff(&serde_json::json!({"agent": "@stock-guru", "task": " NVDA "}), catalog.all()).unwrap();
        assert_eq!(agent.name, "stock-guru");
        assert_eq!(task, "NVDA");
        assert!(read_handoff(&serde_json::json!({"agent": "nobody"}), catalog.all()).is_err());
        assert!(handoff_spec(&[]).is_none());
    }

    fn spec(name: &str, effect: Effect) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: String::new(),
            input_schema: serde_json::json!({"type": "object"}),
            output_schema: None,
            effect,
        }
    }

    fn agent(tools: &[&str], rules: &[(&str, Rule)], origin: Origin) -> Agent {
        Agent {
            name: "probe".into(),
            origin,
            definition: Definition {
                description: "tests things".into(),
                instructions: "do the thing".into(),
                tools: tools.iter().map(|t| t.to_string()).collect(),
                permissions: rules.iter().map(|(t, r)| (t.to_string(), *r)).collect(),
                ..Default::default()
            },
        }
    }

    #[test]
    fn every_builtin_agent_parses_and_is_valid() {
        let catalog = Catalog::builtin();
        let names: Vec<&str> = catalog.all().iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["deep-researcher", "sentiment-analyser", "stock-guru"]);
        for a in catalog.all() {
            assert_eq!(a.origin, Origin::Builtin);
            assert!(!a.definition.tools.is_empty(), "{} has no tools", a.name);
        }
    }

    #[test]
    fn names_are_what_a_mention_can_reach() {
        for good in ["stock-guru", "a", "r2-d2", "deep-researcher"] {
            assert!(validate_name(good).is_ok(), "{good}");
        }
        for bad in ["", "Stock", "-x", "x-", "a--b", "has space", "é", "1abc", "a_b"] {
            assert!(validate_name(bad).is_err(), "{bad}");
        }
        assert!(validate_name(&"a".repeat(MAX_NAME + 1)).is_err());
    }

    #[test]
    fn a_mention_ends_at_punctuation_and_skips_email_addresses() {
        let found = mentions("hey @stock-guru, and mail me@example.com about (@deep-researcher).");
        let names: Vec<&str> = found.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["stock-guru", "deep-researcher"]);
        let text = "hey @stock-guru, ok";
        let m = &mentions(text)[0];
        assert_eq!(&text[m.start..m.end], "@stock-guru");
    }

    #[test]
    fn a_mention_at_the_very_start_counts() {
        assert_eq!(mentions("@stock-guru NVDA")[0].name, "stock-guru");
    }

    #[test]
    fn a_trailing_hyphen_is_punctuation_not_part_of_the_name() {
        assert_eq!(mentions("ask @stock-guru- now")[0].name, "stock-guru");
    }

    #[test]
    fn only_known_agents_are_called_once_each_and_capped() {
        let catalog = Catalog::builtin();
        let called = catalog.mentioned(
            "@nobody @stock-guru @stock-guru @deep-researcher @sentiment-analyser",
        );
        let names: Vec<&str> = called.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["stock-guru", "deep-researcher", "sentiment-analyser"]);
    }

    #[test]
    fn the_mention_being_typed_is_found_at_the_caret() {
        assert_eq!(typing_mention("ask @sto"), Some((4, "sto")));
        assert_eq!(typing_mention("@"), Some((0, "")));
        assert_eq!(typing_mention("ask @stock-guru now"), None);
        assert_eq!(typing_mention("me@exam"), None);
    }

    #[test]
    fn suggestions_prefer_a_prefix_but_find_a_substring() {
        let catalog = Catalog::builtin();
        let names = |p: &str| -> Vec<String> {
            catalog.suggest(p).iter().map(|a| a.name.clone()).collect()
        };
        assert_eq!(names("st"), ["stock-guru"]);
        // A prefix match ranks above a name that only contains the text.
        assert_eq!(names("se"), ["sentiment-analyser", "deep-researcher"]);
        assert_eq!(names("guru"), ["stock-guru"]);
        assert_eq!(names("").len(), 3);
        assert!(names("zzz").is_empty());
    }

    #[test]
    fn an_unlisted_tool_is_refused_even_when_the_global_policy_allows_it() {
        let a = agent(&["web_search"], &[], Origin::User);
        let global = Permissions::default();
        assert_eq!(a.verdict(&global, "read_file", Effect::Read, &Grants::default()), Verdict::Deny);
    }

    #[test]
    fn a_global_deny_beats_the_agents_allow() {
        let a = agent(&["run_command"], &[("run_command", Rule::Allow)], Origin::User);
        let mut global = Permissions::default();
        global.tools.insert("run_command".into(), Rule::Deny);
        assert_eq!(
            a.verdict(&global, "run_command", Effect::Execute, &Grants::default()),
            Verdict::Deny
        );
        let mut by_effect = Permissions::default();
        by_effect.execute = Rule::Deny;
        assert_eq!(
            a.verdict(&by_effect, "run_command", Effect::Execute, &Grants::default()),
            Verdict::Deny
        );
    }

    #[test]
    fn an_agents_allow_lifts_a_global_ask_and_carries_authority_only_if_user_written() {
        let global = Permissions::default(); // write = ask
        let mine = agent(&["write_file"], &[("write_file", Rule::Allow)], Origin::User);
        assert_eq!(
            mine.verdict(&global, "write_file", Effect::Write, &Grants::default()),
            Verdict::Allow { by_user: true }
        );
        let shipped = agent(&["web_search"], &[("web_search", Rule::Allow)], Origin::Builtin);
        assert_eq!(
            shipped.verdict(&global, "web_search", Effect::Read, &Grants::default()),
            Verdict::Allow { by_user: false }
        );
    }

    #[test]
    fn an_agents_ask_tightens_a_global_allow_unless_a_session_grant_exists() {
        let global = Permissions::default(); // read = allow
        let a = agent(&["fetch_url"], &[("fetch_url", Rule::Ask)], Origin::User);
        assert_eq!(a.verdict(&global, "fetch_url", Effect::Read, &Grants::default()), Verdict::Ask);
        let mut grants = Grants::default();
        grants.remember("fetch_url", crate::Choice::Session);
        assert_eq!(
            a.verdict(&global, "fetch_url", Effect::Read, &grants),
            Verdict::Allow { by_user: true }
        );
    }

    #[test]
    fn with_no_rule_of_its_own_the_agent_inherits_the_global_verdict() {
        let global = Permissions::default();
        let a = agent(&["write_file", "web_search"], &[], Origin::User);
        assert_eq!(a.verdict(&global, "write_file", Effect::Write, &Grants::default()), Verdict::Ask);
        assert_eq!(
            a.verdict(&global, "web_search", Effect::Read, &Grants::default()),
            Verdict::Allow { by_user: false }
        );
    }

    #[test]
    fn offering_keeps_the_agents_order_and_reports_what_is_missing() {
        let a = agent(&["yahoo_finance", "web_search", "fetch_url"], &[], Origin::User);
        let available = [spec("fetch_url", Effect::Read), spec("web_search", Effect::Read), spec("write_file", Effect::Write)];
        let (offered, missing) = a.offer(&available);
        let names: Vec<&str> = offered.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["web_search", "fetch_url"]);
        assert_eq!(missing, ["yahoo_finance"]);
    }

    #[test]
    fn a_definition_that_cannot_run_is_refused_with_the_reason() {
        let mut d = agent(&["web_search"], &[], Origin::User).definition;
        d.instructions = " ".into();
        assert!(d.validate("x").unwrap_err().to_string().contains("instructions"));

        let mut d = agent(&["web_search"], &[("fetch_url", Rule::Allow)], Origin::User).definition;
        assert!(d.validate("x").unwrap_err().to_string().contains("fetch_url"));
        d.permissions.clear();
        d.max_rounds = Some(MAX_ROUNDS + 1);
        assert!(d.validate("x").is_err());
        d.max_rounds = Some(4);
        d.tools = vec!["a".into(), "a".into()];
        assert!(d.validate("x").unwrap_err().to_string().contains("twice"));
    }

    #[test]
    fn a_saved_agent_round_trips_and_overrides_then_restores_a_builtin() {
        let dir = std::env::temp_dir().join(format!("ozgent-agents-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = Paths::with_root(&dir);

        let mut changed = Catalog::builtin().get("stock-guru").unwrap().definition.clone();
        changed.description = "my own guru".into();
        save(&paths, "stock-guru", &changed).unwrap();
        let mut custom = changed.clone();
        custom.description = "mine".into();
        save(&paths, "helper", &custom).unwrap();

        let catalog = Catalog::load(&paths);
        assert!(catalog.errors.is_empty(), "{:?}", catalog.errors);
        let guru = catalog.get("stock-guru").unwrap();
        assert_eq!(guru.origin, Origin::Override);
        assert_eq!(guru.definition, changed);
        assert_eq!(catalog.get("helper").unwrap().origin, Origin::User);

        assert_eq!(remove(&paths, "stock-guru").unwrap(), Origin::Override);
        let restored = Catalog::load(&paths);
        assert_eq!(restored.get("stock-guru").unwrap().origin, Origin::Builtin);
        assert!(matches!(remove(&paths, "stock-guru"), Err(AgentError::Builtin(_))));
        assert!(matches!(remove(&paths, "nobody"), Err(AgentError::NotFound(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_broken_file_is_reported_and_leaves_the_others_alone() {
        let dir = std::env::temp_dir().join(format!("ozgent-agents-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = Paths::with_root(&dir);
        std::fs::create_dir_all(paths.agents_dir()).unwrap();
        std::fs::write(paths.agents_dir().join("stock-guru.toml"), "description = [").unwrap();
        std::fs::write(paths.agents_dir().join("Bad Name.toml"), "description = \"x\"").unwrap();

        let catalog = Catalog::load(&paths);
        assert_eq!(catalog.errors.len(), 2, "{:?}", catalog.errors);
        // The broken override did not take the built-in with it.
        assert_eq!(catalog.get("stock-guru").unwrap().origin, Origin::Builtin);
        assert_eq!(catalog.all().len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_system_prompt_names_the_agent_and_any_missing_tools() {
        let a = Catalog::builtin().get("stock-guru").unwrap().clone();
        let prompt = a.system_prompt(Some("Today is Friday."), &["yahoo_finance".into()]);
        assert!(prompt.starts_with("Today is Friday."));
        assert!(prompt.contains("You are @stock-guru"));
        assert!(prompt.contains("not available right now: yahoo_finance"));
        assert!(prompt.contains("equity analyst"));
    }
}
