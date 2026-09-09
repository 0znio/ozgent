//! `[mcp]` — connecting to Model Context Protocol servers.
//!
//! An MCP server is someone else's program offering tools. That is the whole
//! reason this file is careful, and it is a different worry from the one
//! `[channels]` has: there the risk is *who* is talking to ozgent, here it is
//! *what ozgent has been told about a tool by the thing that wants to run it*.
//!
//! MCP tools may carry annotations — `readOnlyHint` and friends. The
//! specification is explicit that these are hints and that a client must not
//! trust them unless the server is trusted, and ozgent's own permission rules
//! already say that a tool which does not declare its effect gets asked about,
//! because silence must not be read as harmless. A *claim* by the thing being
//! asked about is not better evidence than silence, so by default annotations
//! are ignored entirely and every MCP tool is treated as "does not say" —
//! which means it asks. `trust_hints` is how an operator says otherwise, per
//! server, having decided that server is theirs.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpConfig {
    /// Master switch. Off means no server is contacted, whatever is listed.
    pub enabled: bool,

    /// Servers, by the name their tools are prefixed with.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub servers: BTreeMap<String, Server>,
}

impl McpConfig {
    /// The servers that should be started, with their names.
    pub fn active(&self) -> Vec<(&str, &Server)> {
        if !self.enabled {
            return Vec::new();
        }
        self.servers
            .iter()
            .filter(|(_, s)| s.enabled && s.transport().is_ok())
            .map(|(name, s)| (name.as_str(), s))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Server {
    pub enabled: bool,

    /// Program to run, for a server that speaks over its own stdin and stdout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Environment for the child, on top of ozgent's own.
    ///
    /// Most servers take their credentials this way, so this is where an API
    /// key ends up. It is never logged and never shown by `ozgent mcp`.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,

    /// Endpoint, for a server reached over HTTP instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,

    /// How long one call may take. Longer than the Python tools' default: an
    /// MCP server is often a network client itself.
    pub timeout_seconds: u64,

    /// Believe this server's `readOnlyHint` annotations.
    ///
    /// Off by default. With it off every tool from this server is treated as
    /// not having said what it does, and so is asked about. With it on, a tool
    /// the server calls read-only runs under the `read` rule — which normally
    /// means without asking. Turn it on for a server you run yourself.
    pub trust_hints: bool,

    /// Offer only these tools, by their name on the server. `None` offers
    /// everything it lists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            enabled: true,
            command: None,
            args: Vec::new(),
            env: BTreeMap::new(),
            cwd: None,
            url: None,
            headers: BTreeMap::new(),
            timeout_seconds: 60,
            trust_hints: false,
            tools: None,
        }
    }
}

/// How to reach a server.
#[derive(Debug, Clone, PartialEq)]
pub enum Transport<'a> {
    /// Run a program and talk over its stdin and stdout.
    Stdio { command: &'a str },
    /// POST to an endpoint.
    Http { url: &'a str },
}

/// Why a server entry cannot be used.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Invalid {
    #[error("says neither `command` nor `url`, so there is nothing to connect to")]
    Nothing,
    #[error("says both `command` and `url`; a server is reached one way or the other")]
    Both,
    #[error("`args`, `env` and `cwd` describe a program to run, but this server has a `url`")]
    StdioSettingsOnHttp,
    #[error("`headers` describes an HTTP request, but this server has a `command`")]
    HttpSettingsOnStdio,
}

impl Server {
    pub fn transport(&self) -> Result<Transport<'_>, Invalid> {
        let command = self.command.as_deref().filter(|c| !c.trim().is_empty());
        let url = self.url.as_deref().filter(|u| !u.trim().is_empty());

        match (command, url) {
            (None, None) => Err(Invalid::Nothing),
            (Some(_), Some(_)) => Err(Invalid::Both),
            // Settings that belong to the other transport are refused rather
            // than ignored: silently dropping an `env` that carries the API
            // key would look like the server rejecting the credentials.
            (Some(_), None) if !self.headers.is_empty() => Err(Invalid::HttpSettingsOnStdio),
            (None, Some(_)) if !self.args.is_empty() || !self.env.is_empty() || self.cwd.is_some() => {
                Err(Invalid::StdioSettingsOnHttp)
            }
            (Some(command), None) => Ok(Transport::Stdio { command }),
            (None, Some(url)) => Ok(Transport::Http { url }),
        }
    }
}

/// The name a server's tool is offered to the model under.
///
/// Prefixed with the server, always — not only when two servers collide.
/// A name that changed when an unrelated server was added would silently
/// invalidate any `[permissions.tools]` rule written about it, and the model
/// would be shown a tool it had never been told about mid-conversation.
///
/// The result is constrained to what every model's tool-call format accepts:
/// letters, digits, underscore and hyphen, at most 64 characters.
pub fn tool_name(server: &str, tool: &str) -> String {
    let clean = |s: &str| -> String {
        s.chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' })
            .collect()
    };
    let server = clean(server);
    let tool = clean(tool);
    let joined = format!("{server}_{tool}");
    if joined.len() <= 64 {
        return joined;
    }
    // Truncating the tail would make two long names from one server collide,
    // so the *server* half is shortened first and the tool kept whole.
    let room = 64usize.saturating_sub(tool.len() + 1);
    let server: String = server.chars().take(room).collect();
    let joined = format!("{server}_{tool}");
    joined.chars().take(64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio() -> Server {
        Server { command: Some("npx".into()), ..Default::default() }
    }

    #[test]
    fn nothing_is_contacted_until_the_switch_is_on() {
        let mut c = McpConfig { enabled: false, servers: BTreeMap::new() };
        c.servers.insert("a".into(), stdio());
        assert!(c.active().is_empty());

        c.enabled = true;
        assert_eq!(c.active().len(), 1);
    }

    #[test]
    fn a_server_is_reached_one_way_or_the_other() {
        assert_eq!(stdio().transport(), Ok(Transport::Stdio { command: "npx" }));

        let http = Server { url: Some("https://x.test/mcp".into()), ..Default::default() };
        assert_eq!(http.transport(), Ok(Transport::Http { url: "https://x.test/mcp" }));

        let both = Server { url: Some("https://x.test".into()), ..stdio() };
        assert_eq!(both.transport(), Err(Invalid::Both));

        assert_eq!(Server::default().transport(), Err(Invalid::Nothing));
    }

    #[test]
    fn an_empty_command_is_not_a_command() {
        let blank = Server { command: Some("   ".into()), ..Default::default() };
        assert_eq!(blank.transport(), Err(Invalid::Nothing));
    }

    #[test]
    fn settings_for_the_wrong_transport_are_refused_rather_than_ignored() {
        // Silently dropping an `env` that carries the API key would look like
        // the server rejecting the credentials, which is a long afternoon.
        let mut wrong = Server { url: Some("https://x.test".into()), ..Default::default() };
        wrong.env.insert("TOKEN".into(), "…".into());
        assert_eq!(wrong.transport(), Err(Invalid::StdioSettingsOnHttp));

        let mut other = stdio();
        other.headers.insert("Authorization".into(), "…".into());
        assert_eq!(other.transport(), Err(Invalid::HttpSettingsOnStdio));
    }

    #[test]
    fn a_misconfigured_server_is_not_started() {
        let mut c = McpConfig { enabled: true, servers: BTreeMap::new() };
        c.servers.insert("broken".into(), Server::default());
        c.servers.insert("fine".into(), stdio());
        assert_eq!(c.active().iter().map(|(n, _)| *n).collect::<Vec<_>>(), ["fine"]);
    }

    #[test]
    fn a_disabled_server_is_skipped() {
        let mut c = McpConfig { enabled: true, servers: BTreeMap::new() };
        c.servers.insert("off".into(), Server { enabled: false, ..stdio() });
        assert!(c.active().is_empty());
    }

    #[test]
    fn a_tool_is_named_after_its_server() {
        assert_eq!(tool_name("github", "search_issues"), "github_search_issues");
    }

    #[test]
    fn a_name_is_reduced_to_what_a_tool_call_format_accepts() {
        // Models are given a name pattern of [A-Za-z0-9_-]{1,64}; anything
        // else is rejected by the API or mangled by the template.
        let name = tool_name("my server!", "do/a thing");
        assert_eq!(name, "my_server__do_a_thing");
        assert!(name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
    }

    #[test]
    fn a_long_name_stays_within_the_limit_and_stays_distinct() {
        // Trimming the tail would make two tools from one server collide,
        // which is worse than a shortened prefix.
        let server = "a".repeat(60);
        let first = tool_name(&server, "read_the_first_thing");
        let second = tool_name(&server, "read_the_second_thing");
        assert!(first.len() <= 64 && second.len() <= 64, "{} {}", first.len(), second.len());
        assert_ne!(first, second);
        assert!(first.ends_with("read_the_first_thing"), "{first}");
    }

    #[test]
    fn the_documented_configuration_parses() {
        let text = r#"
            [mcp]
            enabled = true

            [mcp.servers.github]
            command = "npx"
            args = ["-y", "@modelcontextprotocol/server-github"]
            env = { GITHUB_TOKEN = "secret" }
            trust_hints = true

            [mcp.servers.docs]
            url = "https://example.test/mcp"
            headers = { Authorization = "Bearer secret" }
            timeout_seconds = 20
            tools = ["search"]
        "#;
        let config: crate::Config = toml::from_str(text).expect("the documented settings");
        assert_eq!(config.mcp.active().len(), 2);

        let github = &config.mcp.servers["github"];
        assert!(github.trust_hints);
        assert_eq!(github.env["GITHUB_TOKEN"], "secret");

        let docs = &config.mcp.servers["docs"];
        assert_eq!(docs.timeout_seconds, 20);
        assert_eq!(docs.tools.as_deref(), Some(&["search".to_string()][..]));
        assert!(!docs.trust_hints, "hints are not believed unless asked for");
    }

    #[test]
    fn hints_are_not_trusted_unless_asked_for() {
        // The default that matters: a server's claim about itself is not
        // evidence, so out of the box every MCP tool asks.
        assert!(!Server::default().trust_hints);
    }
}
