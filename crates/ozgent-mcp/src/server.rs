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
use crate::sandbox::Launcher;
use crate::transport::{HttpLink, Link, Log, StdioLink, TransportError};

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
    #[error("{0}")]
    Sandbox(String),
}

/// How a configured server is doing, for the admin page, `/mcp` and
/// `ozgent mcp`. Carries no settings, so nothing secret.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Status {
    pub name: String,
    pub state: State,
    /// Why it is not connected, when it is not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The last lines it wrote to stderr.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub log: Vec<String>,
    /// What it calls itself, and its version, once connected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server: Option<String>,
    /// Every tool it lists, offered or not.
    pub tools: Vec<ToolStatus>,
    pub sandboxed: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolStatus {
    /// The name the model sees: `<server>_<tool>`.
    pub name: String,
    /// The name the server knows it by.
    pub remote: String,
    pub description: String,
    pub effect: ozgent_core::permission::Effect,
    /// Offered to the model: not left out by the server's `tools` list.
    pub offered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum State {
    Connected,
    Failed,
    /// Switched off on its own.
    Disabled,
    /// `[mcp]` itself is switched off.
    Off,
}

/// Environment variables a server is given from ozgent's own, before its
/// configured `env`. What a program needs to find itself and the network —
/// never the keys and tokens ozgent was started with.
const ENV_KEEP: &[&str] = &[
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "TZ", "SHELL", "TMPDIR",
    "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
    "CARGO_HOME", "RUSTUP_HOME", "GOPATH", "GOROOT", "JAVA_HOME", "VIRTUAL_ENV",
    "NVM_DIR", "PNPM_HOME", "NODE_PATH", "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
];

const SECRETISH: &[&str] = &["KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "COOKIE", "SESSION", "AUTH"];

/// The environment a stdio server starts with: [`ENV_KEEP`] from ozgent's
/// own, then what its entry configures, which wins.
pub fn server_env(configured: &std::collections::BTreeMap<String, String>) -> std::collections::BTreeMap<String, String> {
    let mut env: std::collections::BTreeMap<String, String> = std::env::vars()
        .filter(|(k, _)| ENV_KEEP.contains(&k.as_str()) || k.starts_with("LC_"))
        .filter(|(k, _)| !SECRETISH.iter().any(|s| k.to_ascii_uppercase().contains(s)))
        .collect();
    env.insert(NPM_ALLOW_SCRIPTS.0.into(), NPM_ALLOW_SCRIPTS.1.into());
    env.extend(configured.iter().map(|(k, v)| (k.clone(), v.clone())));
    env
}

/// The install scripts `npx` may run for an MCP server, by package name.
///
/// npm 12 runs no dependency's install script unless it is named, and a
/// native add-on without its script is a package with no binary: the server
/// starts, and fails on first use ("Could not locate the bindings file" from
/// better-sqlite3, measured). These are the add-ons MCP servers pull in, and
/// only these: everything else stays as npm 12 leaves it, which is stricter
/// than npm 11, where every script ran. A server's own `env` can set it.
pub const NPM_ALLOW_SCRIPTS: (&str, &str) = (
    "npm_config_allow_scripts",
    "better-sqlite3,sqlite3,sharp,canvas,bcrypt,argon2,keytar,node-pty,re2,bufferutil,utf-8-validate,\
     onnxruntime-node,@tensorflow/tfjs-node,cpu-features,ssh2,esbuild,@swc/core,protobufjs,puppeteer",
);

pub struct Server {
    name: String,
    origin: String,
    /// Swapped for a new one when the server has to be started again.
    link: tokio::sync::RwLock<Arc<Link>>,
    /// How to start it again: a server that crashed, or whose HTTP session
    /// expired, is started afresh by the next call instead of every call
    /// failing until someone reconnects it by hand.
    revive: Revive,
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
        Self::connect_with(name, config, None, Log::default()).await.map(|(server, _)| server)
    }

    /// [`Self::connect`], with ozgent's sandbox for a server that asks for it
    /// and its stderr kept in `log`. Also returns every tool the server
    /// listed, including those its `tools` setting leaves out.
    pub async fn connect_with(
        name: &str,
        config: &mcp::Server,
        launcher: Option<&Launcher>,
        log: Log,
    ) -> Result<(Self, Vec<ToolStatus>), ConnectError> {
        let origin = format!("mcp:{name}");
        let link = open(name, config, launcher, log.clone(), &origin).await?;
        let info = handshake(&link).await?;

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
        let mut all = Vec::new();
        for tool in listed {
            let spec = protocol::to_spec(name, &tool, config.trust_hints);
            let offered = wanted(&tool);
            all.push(ToolStatus {
                name: spec.name.clone(),
                remote: tool.name.clone(),
                description: ozgent_tools::first_line(&spec.description).to_string(),
                effect: spec.effect,
                offered,
            });
            if offered {
                real_name.insert(spec.name.clone(), tool.name);
                specs.push(spec);
            }
        }

        let revive = Revive {
            config: config.clone(),
            launcher: launcher.cloned(),
            log,
            last: std::sync::Mutex::new(None),
        };
        let link = tokio::sync::RwLock::new(Arc::new(link));
        Ok((Self { name: name.to_string(), origin, link, revive, info, specs, real_name, timeout }, all))
    }

    /// Start the server again after it stopped, at most once per
    /// [`REVIVE_EVERY`]: a server that dies on every call is not restarted in
    /// a loop, and says so.
    async fn revive(&self, dead: &Arc<Link>) -> Result<Arc<Link>, String> {
        let mut current = self.link.write().await;
        // Another call may have started it again already.
        if !Arc::ptr_eq(&current, dead) {
            return Ok(Arc::clone(&current));
        }
        {
            let mut last = self.revive.last.lock().unwrap_or_else(|e| e.into_inner());
            if last.is_some_and(|at| at.elapsed() < REVIVE_EVERY) {
                return Err("it stopped, and was restarted moments ago and stopped again".into());
            }
            *last = Some(std::time::Instant::now());
        }
        tracing::info!(target: "ozgent::mcp", "{}: the server stopped; starting it again", self.name);
        let started = async {
            let link = open(&self.name, &self.revive.config, self.revive.launcher.as_ref(), self.revive.log.clone(), &self.origin).await?;
            handshake(&link).await?;
            Ok::<_, ConnectError>(link)
        };
        match started.await {
            Ok(link) => {
                let link = Arc::new(link);
                *current = Arc::clone(&link);
                Ok(link)
            }
            Err(e) => Err(format!("it stopped, and starting it again failed: {e}")),
        }
    }
}

/// How to start a server again, kept from when it was first started.
struct Revive {
    config: mcp::Server,
    launcher: Option<Launcher>,
    log: Log,
    last: std::sync::Mutex<Option<std::time::Instant>>,
}

/// The least time between two restarts of one server.
const REVIVE_EVERY: Duration = Duration::from_secs(10);

/// Start the server's process, or open its HTTP connection.
async fn open(name: &str, config: &mcp::Server, launcher: Option<&Launcher>, log: Log, origin: &str) -> Result<Link, ConnectError> {
    let link = match config.transport()? {
        Transport::Stdio { command } => {
            let env = server_env(&config.env);
            let link = if config.sandbox {
                let Some(launcher) = launcher else {
                    return Err(ConnectError::Sandbox(
                        "asks for the sandbox, but ozgent's Python runtime was not found to provide one"
                            .into(),
                    ));
                };
                let mut configured = config.env.clone();
                configured.entry(NPM_ALLOW_SCRIPTS.0.into()).or_insert_with(|| NPM_ALLOW_SCRIPTS.1.into());
                let wrapped = launcher
                    .wrap(name, command, &config.args, &configured, &config.folders, config.network)
                    .await
                    .map_err(ConnectError::Sandbox)?;
                StdioLink::start(&wrapped.command, &wrapped.args, &wrapped.env, Some(&wrapped.cwd), origin, log)?
            } else {
                // Not sandboxed, so it keeps the real `HOME`; but what
                // `npx` and `uvx` download for it still goes in its own
                // folder under ~/ozgent/mcp, so removing that removes it.
                let mut env = env;
                if let Some(home) = launcher.map(|l| l.home(name)) {
                    let cache = home.join(".cache");
                    if std::fs::create_dir_all(&cache).is_ok() {
                        for (key, dir) in [("npm_config_cache", "npm"), ("UV_CACHE_DIR", "uv")] {
                            env.entry(key.to_string()).or_insert_with(|| cache.join(dir).display().to_string());
                        }
                    }
                }
                StdioLink::start(command, &config.args, &env, config.cwd.as_deref(), origin, log)?
            };
            Link::Stdio(Box::new(link))
        }
        Transport::Http { url } => Link::Http(Box::new(HttpLink::new(url, config.headers.clone()))),
    };
    Ok(link)
}

/// `initialize`, and the notification that must follow it.
async fn handshake(link: &Link) -> Result<ServerInfo, ConnectError> {
    let raw = link.request("initialize", protocol::initialize_params(), HANDSHAKE).await?;
    let info = protocol::server_info(&raw);

    // The session id arrives as a header on this exchange and is required
    // on every request afterwards by a server that issued one.
    if let Link::Http(http) = link {
        let session = http.session().await;
        http.adopt(session, &info.protocol_version).await;
    }

    // The specification requires this before anything else is sent, and a
    // strict server will refuse `tools/list` until it arrives.
    link.notify("notifications/initialized", json!({})).await?;
    Ok(info)
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

            // Small models write `"true"` for true and `"5"` for 5 often
            // enough that a strict server's refusal cost a whole round.
            let arguments = match self.specs.iter().find(|s| s.name == name) {
                Some(spec) => coerce(&spec.input_schema, arguments),
                None => arguments,
            };
            let params = json!({ "name": real, "arguments": arguments });
            let link = Arc::clone(&*self.link.read().await);
            let first = link.request("tools/call", params.clone(), self.timeout).await;
            let result = match first {
                // Stopped since the last call, so this one never reached it:
                // started again once, and the call made to the new one. One
                // that stopped *during* the call is not repeated — the call
                // may have done its work, or be what kills it.
                Err(TransportError::Stopped) => match self.revive(&link).await {
                    Ok(fresh) => fresh.request("tools/call", params, self.timeout).await,
                    Err(why) => return Err(ToolCallError::Transport(format!("{}: {why}", self.origin))),
                },
                other => other,
            };
            let result = result.map_err(
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
        Box::pin(async move { self.link.read().await.close().await })
    }
}

/// Fix arguments whose type is plainly a model's slip, by the tool's own
/// schema: a string `"true"`/`"false"` where it asks for a boolean, a string
/// holding a number where it asks for a number or integer. Top-level only,
/// and nothing is guessed — a value that does not read cleanly as the type
/// asked for is sent as it was, for the server to refuse and the model to see.
pub fn coerce(schema: &Value, arguments: Value) -> Value {
    let Value::Object(mut args) = arguments else { return arguments };
    let Some(props) = schema.get("properties").and_then(Value::as_object) else { return Value::Object(args) };
    for (key, value) in args.iter_mut() {
        let Some(text) = value.as_str().map(str::trim) else { continue };
        let wants = |t: &str| {
            let ty = &props.get(key).map(|p| p["type"].clone()).unwrap_or(Value::Null);
            // A property that also accepts strings is left alone.
            match ty {
                Value::String(s) => s == t,
                Value::Array(list) => list.iter().any(|x| x == t) && !list.iter().any(|x| x == "string"),
                _ => false,
            }
        };
        if wants("boolean") {
            match text.to_ascii_lowercase().as_str() {
                "true" => *value = Value::Bool(true),
                "false" => *value = Value::Bool(false),
                _ => {}
            }
        } else if wants("integer") {
            if let Ok(n) = text.parse::<i64>() {
                *value = json!(n);
            }
        } else if wants("number") {
            if let Ok(n) = text.parse::<f64>() {
                if n.is_finite() {
                    *value = json!(n);
                }
            }
        }
    }
    Value::Object(args)
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
    let (sources, statuses) = connect_all_with(config, None).await;
    (sources, problems(&statuses))
}

/// One line per server that is meant to run and does not.
pub fn problems(statuses: &[Status]) -> Vec<String> {
    statuses
        .iter()
        .filter(|s| s.state == State::Failed)
        .map(|s| format!("{}: {}", s.name, s.error.as_deref().unwrap_or("failed")))
        .collect()
}

/// [`connect_all`], with ozgent's sandbox for the servers that ask for it,
/// and a status for every configured server, running or not.
///
/// Servers are connected at the same time: an `npx` server downloading its
/// package on first run takes a minute, and must not hold up the others.
pub async fn connect_all_with(
    config: &ozgent_core::McpConfig,
    launcher: Option<&Launcher>,
) -> (Vec<Arc<dyn ToolSource>>, Vec<Status>) {
    let mut sources: Vec<Arc<dyn ToolSource>> = Vec::new();
    let mut statuses = Vec::new();

    let mut connecting = tokio::task::JoinSet::new();
    for (name, settings) in &config.servers {
        let base = Status {
            name: name.clone(),
            state: State::Failed,
            error: None,
            log: Vec::new(),
            server: None,
            tools: Vec::new(),
            sandboxed: settings.sandbox,
        };
        if !config.enabled {
            statuses.push(Status { state: State::Off, ..base });
            continue;
        }
        if !settings.enabled {
            statuses.push(Status { state: State::Disabled, ..base });
            continue;
        }
        // Not startable at all — both transports given, or neither — so it
        // never reaches a connection attempt, and is reported here or nowhere.
        if let Err(e) = settings.transport() {
            statuses.push(Status { error: Some(e.to_string()), ..base });
            continue;
        }
        let (name, settings, launcher) = (name.clone(), settings.clone(), launcher.cloned());
        connecting.spawn(async move {
            let log = Log::default();
            let result = Server::connect_with(&name, &settings, launcher.as_ref(), log.clone()).await;
            let log: Vec<String> = log.lock().unwrap_or_else(|e| e.into_inner()).iter().cloned().collect();
            (name, result, log, base)
        });
    }

    while let Some(joined) = connecting.join_next().await {
        let Ok((name, result, log, base)) = joined else { continue };
        match result {
            Ok((server, tools)) => {
                tracing::info!(
                    target: "ozgent::mcp",
                    "{name}: {} tools from {} {}",
                    server.tools().len(),
                    server.info().name,
                    server.info().version,
                );
                let about = format!("{} {}", server.info().name, server.info().version).trim().to_string();
                statuses.push(Status {
                    state: State::Connected,
                    server: (!about.is_empty()).then_some(about),
                    tools,
                    log,
                    ..base
                });
                sources.push(Arc::new(server));
            }
            Err(e) => statuses.push(Status { error: Some(e.to_string()), log, ..base }),
        }
    }

    // In name order, whatever order they finished in: a tool list that
    // reordered itself between restarts would change the prompt, and with it
    // every cached prefix.
    sources.sort_by(|a, b| a.origin().cmp(b.origin()));
    statuses.sort_by(|a, b| a.name.cmp(&b.name));
    // Returned rather than logged here. Every caller already reports them in
    // the way its surface calls for — a warning line in the terminal, the
    // server log, the settings page — and logging as well printed each
    // problem twice.
    (sources, statuses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_models_type_slips_are_fixed_by_the_schema() {
        let schema = json!({ "properties": {
            "headless": { "type": "boolean" },
            "count": { "type": "integer" },
            "scale": { "type": "number" },
            "name": { "type": "string" },
            "either": { "type": ["boolean", "string"] },
        }});
        let fixed = coerce(&schema, json!({
            "headless": "True", "count": " 5", "scale": "1.5", "name": "true", "either": "true", "extra": "1",
        }));
        assert_eq!(fixed, json!({
            "headless": true, "count": 5, "scale": 1.5, "name": "true", "either": "true", "extra": "1",
        }));
        // Nothing is guessed.
        let left = coerce(&schema, json!({ "headless": "yes please", "count": "five" }));
        assert_eq!(left, json!({ "headless": "yes please", "count": "five" }));
    }

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
