//! One connected server, offering its tools to ozgent.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ozgent_core::ToolSpec;
use ozgent_core::mcp::{self, Transport};
use ozgent_tools::host::ToolCallError;
use ozgent_tools::protocol::{INTERNAL_ERROR, RpcError, TOOL_ERROR};
use ozgent_tools::source::{Boxed, ToolSource};
use serde_json::{Value, json};

use crate::protocol::{self, ServerInfo};
use crate::transport::{HttpLink, Link, StdioLink, TransportError};

/// How long the handshake may take.
///
/// Longer than a call, and separately so: a stdio server is often `npx`, which
/// downloads the package the first time it is run.
const HANDSHAKE: Duration = Duration::from_secs(120);

/// Pages of `tools/list` to follow before giving up.
///
/// A server that never stops paginating would otherwise hold up start-up
/// forever; five hundred pages is far past any real tool list.
const MAX_PAGES: usize = 500;

#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error("{0}")]
    Config(#[from] mcp::Invalid),
    #[error("{0}")]
    Transport(#[from] TransportError),
    #[error("offers no tools")]
    NoTools,
}

pub struct Server {
    name: String,
    origin: String,
    link: Link,
    info: ServerInfo,
    specs: Vec<ToolSpec>,
    /// The model-facing name back to the name the server knows it by.
    real_name: HashMap<String, String>,
    timeout: Duration,
}

impl Server {
    pub fn info(&self) -> &ServerInfo {
        &self.info
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Connect, shake hands, and read the tool list.
    pub async fn connect(name: &str, config: &mcp::Server) -> Result<Self, ConnectError> {
        let origin = format!("mcp:{name}");
        let link = match config.transport()? {
            Transport::Stdio { command } => Link::Stdio(Box::new(StdioLink::start(
                command,
                &config.args,
                &config.env,
                config.cwd.as_deref(),
                &origin,
            )?)),
            Transport::Http { url } => Link::Http(Box::new(HttpLink::new(url, config.headers.clone()))),
        };

        let raw = link.request("initialize", protocol::initialize_params(), HANDSHAKE).await?;
        let info = protocol::server_info(&raw);

        // The session id arrives as a header on this exchange and is required
        // on every request afterwards by a server that issued one.
        if let Link::Http(http) = &link {
            let session = http.session().await;
            http.adopt(session, &info.protocol_version).await;
        }

        // The specification requires this before anything else is sent, and a
        // strict server will refuse `tools/list` until it arrives.
        link.notify("notifications/initialized", json!({})).await?;

        if !info.has_tools {
            link.close().await;
            return Err(ConnectError::NoTools);
        }

        let timeout = Duration::from_secs(config.timeout_seconds.max(1));
        let listed = list_tools(&link, timeout).await?;

        let wanted = |tool: &protocol::Listed| match &config.tools {
            Some(only) => only.iter().any(|t| t == &tool.name),
            None => true,
        };

        let mut specs = Vec::new();
        let mut real_name = HashMap::new();
        for tool in listed.into_iter().filter(wanted) {
            let spec = protocol::to_spec(name, &tool, config.trust_hints);
            real_name.insert(spec.name.clone(), tool.name);
            specs.push(spec);
        }

        Ok(Self { name: name.to_string(), origin, link, info, specs, real_name, timeout })
    }
}

/// Read every page of the tool list.
async fn list_tools(link: &Link, timeout: Duration) -> Result<Vec<protocol::Listed>, TransportError> {
    let mut all = Vec::new();
    let mut cursor: Option<String> = None;

    for _ in 0..MAX_PAGES {
        let params = match &cursor {
            Some(c) => json!({ "cursor": c }),
            None => json!({}),
        };
        let result = link.request("tools/list", params, timeout).await?;
        let (tools, next) = protocol::parse_tools(&result);
        all.extend(tools);
        match next {
            // A server repeating its cursor would page forever.
            Some(next) if Some(&next) != cursor.as_ref() => cursor = Some(next),
            _ => break,
        }
    }
    Ok(all)
}

impl ToolSource for Server {
    fn origin(&self) -> &str {
        &self.origin
    }

    fn tools(&self) -> &[ToolSpec] {
        &self.specs
    }

    fn call<'a>(
        &'a self,
        name: &'a str,
        arguments: Value,
        _approved: bool,
    ) -> Boxed<'a, Result<Value, ToolCallError>> {
        Box::pin(async move {
            // `approved` is not passed on, and that is deliberate rather than
            // an omission. It exists to lift boundaries ozgent's *own* Python
            // worker keeps for calls made unattended. An MCP server has its
            // own idea of what it will do and no notion of ozgent's consent,
            // so there is nothing here for the flag to lift — and inventing a
            // field to send it in would be telling a third-party program that
            // a person approved something, on no agreed meaning.
            let Some(real) = self.real_name.get(name) else {
                return Err(failed(name, INTERNAL_ERROR, format!("{} does not offer {name}", self.origin)));
            };

            let params = json!({ "name": real, "arguments": arguments });
            let result = self.link.request("tools/call", params, self.timeout).await.map_err(
                |e| match e {
                    TransportError::Timeout(after) => {
                        ToolCallError::Timeout { name: name.to_string(), after }
                    }
                    TransportError::Refused(failure) => {
                        // The server refused the call itself — a bad argument,
                        // an unknown tool. The model can correct that.
                        failed(name, TOOL_ERROR, failure.message)
                    }
                    other => ToolCallError::Transport(format!("{}: {other}", self.origin)),
                },
            )?;

            protocol::parse_call(&result).map_err(|message| failed(name, TOOL_ERROR, message))
        })
    }

    fn shutdown<'a>(&'a self) -> Boxed<'a, ()> {
        Box::pin(async move { self.link.close().await })
    }
}

fn failed(name: &str, code: i32, message: String) -> ToolCallError {
    ToolCallError::Failed {
        name: name.to_string(),
        error: RpcError { code, message, data: None },
    }
}

/// Connect to every configured server.
///
/// A server that will not start is reported and skipped, never fatal: one
/// broken entry in `config.toml` must not take away the tools that do work,
/// and it certainly must not stop ozgent starting.
pub async fn connect_all(
    config: &ozgent_core::McpConfig,
) -> (Vec<Arc<dyn ToolSource>>, Vec<String>) {
    let mut sources: Vec<Arc<dyn ToolSource>> = Vec::new();
    let mut problems = Vec::new();

    for (name, settings) in config.active() {
        match Server::connect(name, settings).await {
            Ok(server) => {
                tracing::info!(
                    target: "ozgent::mcp",
                    "{name}: {} tools from {} {}",
                    server.tools().len(),
                    server.info().name,
                    server.info().version,
                );
                sources.push(Arc::new(server));
            }
            Err(e) => problems.push(format!("{name}: {e}")),
        }
    }

    // Servers listed but not startable at all — a typo in `command`, both
    // transports given — never reach `active()`, so they are reported here or
    // nowhere.
    for (name, settings) in &config.servers {
        if settings.enabled && config.enabled {
            if let Err(e) = settings.transport() {
                problems.push(format!("{name}: {e}"));
            }
        }
    }

    // Returned rather than logged here. Every caller already reports them in
    // the way its surface calls for — a warning line in the terminal, the
    // server log, the settings page — and logging as well printed each
    // problem twice.
    (sources, problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_server_that_cannot_be_reached_is_reported_and_skipped() {
        // One broken entry must not take away the tools that work, and must
        // not stop ozgent starting.
        let mut config = ozgent_core::McpConfig { enabled: true, servers: Default::default() };
        config.servers.insert(
            "broken".into(),
            mcp::Server {
                command: Some("definitely-not-a-real-program-xyz".into()),
                ..Default::default()
            },
        );
        let (sources, problems) = connect_all(&config).await;
        assert!(sources.is_empty());
        assert_eq!(problems.len(), 1);
        assert!(problems[0].starts_with("broken:"), "{:?}", problems);
    }

    #[tokio::test]
    async fn a_misconfigured_server_is_reported_rather_than_ignored() {
        // It never reaches `active()`, so without this it would be listed in
        // config.toml, contacted by nothing, and mentioned nowhere.
        let mut config = ozgent_core::McpConfig { enabled: true, servers: Default::default() };
        config.servers.insert("empty".into(), mcp::Server::default());
        let (sources, problems) = connect_all(&config).await;
        assert!(sources.is_empty());
        assert!(problems[0].contains("nothing to connect to"), "{:?}", problems);
    }

    #[tokio::test]
    async fn nothing_is_contacted_while_the_switch_is_off() {
        let mut config = ozgent_core::McpConfig { enabled: false, servers: Default::default() };
        config.servers.insert("broken".into(), mcp::Server::default());
        let (sources, problems) = connect_all(&config).await;
        assert!(sources.is_empty());
        assert!(problems.is_empty(), "{problems:?}");
    }
}
