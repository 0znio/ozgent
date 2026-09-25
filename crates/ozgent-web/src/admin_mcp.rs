//! MCP servers on the admin page: switch them on and off, choose their
//! tools, add them by hand or install them from the MCP Registry.
//!
//! Behind the admin password, like everything else that decides which
//! programs run on this machine: an MCP server is a program, and adding one
//! is running it. The chat page's `/api/settings` cannot reach `[mcp]` at all
//! (see `put_settings`); the chat page and the terminal get a read-only
//! status from `/api/mcp`, with no settings in it.
//!
//! Secrets stay here. The environment and headers a server is configured
//! with are listed by name, never by value; a change sends only what changes.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::extract::{Path, Query, State as AxumState};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use ozgent_core::mcp::{self, Server};
use ozgent_core::permission::Rule;
use ozgent_mcp::registry;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::admin::{fail, save};
use crate::state::State;

/// Set while the servers are being reconnected, for the page to show.
static RECONNECTING: AtomicBool = AtomicBool::new(false);

/// One reconnect at a time, each reading the configuration when it starts,
/// so two quick changes cannot finish in the wrong order and leave the older
/// set of servers running.
static RESTART: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Reconnect every server in the background. Returns at once: an `npx`
/// server downloading its package on first run can take a minute, and the
/// page polls the status instead of waiting on a request.
pub(crate) fn reconnect(state: &State) {
    let state = state.clone();
    tokio::spawn(async move {
        let _one = RESTART.lock().await;
        RECONNECTING.store(true, Ordering::SeqCst);
        let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
        crate::api::restart_tools(&state, config).await;
        RECONNECTING.store(false, Ordering::SeqCst);
    });
}

fn statuses(state: &State) -> Vec<ozgent_mcp::Status> {
    crate::worker::current_tools(&state.tools).map(|t| t.mcp.as_ref().clone()).unwrap_or_default()
}

fn transport_name(server: &Server) -> &'static str {
    match server.transport() {
        Ok(mcp::Transport::Stdio { .. }) => "stdio",
        Ok(mcp::Transport::Http { .. }) => "http",
        Err(_) => "invalid",
    }
}

/// One server as the admin page shows it: settings without secret values,
/// and how it is doing.
fn server_view(name: &str, server: &Server, status: Option<&ozgent_mcp::Status>, config: &ozgent_core::Config) -> Value {
    let tools: Vec<Value> = status
        .map(|s| {
            s.tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "remote": t.remote,
                        "description": t.description,
                        "effect": t.effect,
                        "offered": t.offered,
                        // What happens when the model calls it, all rules
                        // considered; and the per-tool rule, if one is set.
                        "rule": config.permissions.rule_for(&t.name, t.effect).to_string(),
                        "own_rule": config.permissions.tools.get(&t.name).map(|r| r.to_string()),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    json!({
        "name": name,
        // The rule for every one of its tools at once, if one is set.
        "server_rule": config
            .permissions
            .tools
            .get(&ozgent_core::permission::Permissions::server_key(name))
            .map(|r| r.to_string()),
        "enabled": server.enabled,
        "description": server.description,
        "source": server.source,
        "transport": transport_name(server),
        "invalid": server.transport().err().map(|e| e.to_string()),
        "command": server.command,
        "args": server.args,
        "url": server.url,
        "env": server.env.keys().collect::<Vec<_>>(),
        "headers": server.headers.keys().collect::<Vec<_>>(),
        "sandbox": server.sandbox,
        "network": server.network,
        "folders": server.folders,
        "trust_hints": server.trust_hints,
        "timeout_seconds": server.timeout_seconds,
        "only": server.tools,
        "load": server.load,
        "used": !config.tools.mcp_off.iter().any(|n| n == name),
        "status": status.map(|s| json!({
            "state": s.state,
            "error": s.error,
            "log": s.log,
            "server": s.server,
        })),
        "tools": tools,
    })
}

fn runtimes(state: &State, config: &ozgent_core::Config) -> Value {
    json!({
        "npx": registry::which("npx").is_some(),
        "uvx": registry::which("uvx").is_some(),
        "docker": registry::which("docker").is_some(),
        "sandbox": crate::state::mcp_launcher(&state.paths, config).is_some(),
    })
}

pub(crate) async fn view(AxumState(state): AxumState<State>) -> Response {
    let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let statuses = statuses(&state);
    let servers: Vec<Value> = config
        .mcp
        .servers
        .iter()
        .map(|(name, s)| server_view(name, s, statuses.iter().find(|st| &st.name == name), &config))
        .collect();
    Json(json!({
        "enabled": config.mcp.enabled,
        "tools_enabled": config.tools.enabled,
        "reconnecting": RECONNECTING.load(Ordering::SeqCst),
        "servers": servers,
        "runtimes": runtimes(&state, &config),
        "registry": registry::REGISTRY_URL,
    }))
    .into_response()
}

#[derive(Deserialize)]
pub(crate) struct Switch {
    enabled: bool,
}

/// `[mcp] enabled`: every server, on or off.
pub(crate) async fn switch(AxumState(state): AxumState<State>, Json(body): Json<Switch>) -> Response {
    let saved = save(&state, |c| c.mcp.enabled = body.enabled);
    if saved.status().is_success() {
        reconnect(&state);
    }
    saved
}

pub(crate) async fn reconnect_now(AxumState(state): AxumState<State>) -> Response {
    reconnect(&state);
    Json(json!({ "ok": true })).into_response()
}

/// Settings every way of adding a server shares.
#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct Placement {
    /// Defaults to on: a server added here is a program nobody has read.
    sandbox: Option<bool>,
    network: Option<bool>,
    folders: Vec<String>,
    trust_hints: bool,
}

fn place(server: &mut Server, p: &Placement) -> Result<(), String> {
    let stdio = server.url.is_none();
    // A container is Docker's to isolate; the sandbox would only stand
    // between the Docker client and its daemon.
    let container = server.command.as_deref() == Some("docker");
    server.sandbox = stdio && !container && p.sandbox.unwrap_or(true);
    server.network = p.network.unwrap_or(true);
    server.trust_hints = p.trust_hints;
    server.folders = folders(&p.folders)?;
    if server.sandbox {
        for f in registry::folders_in_args(&server.args) {
            if !server.folders.contains(&f) {
                server.folders.push(f);
            }
        }
    }
    if !stdio && !server.folders.is_empty() {
        return Err("folders are for a program run here; this server is reached by URL".into());
    }
    Ok(())
}

/// Folders a sandboxed server may use: absolute, and never ozgent's own home
/// or the root of the filesystem.
fn folders(raw: &[String]) -> Result<Vec<std::path::PathBuf>, String> {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let mut out = Vec::new();
    for f in raw.iter().map(|f| f.trim()).filter(|f| !f.is_empty()) {
        let expanded = match (f.strip_prefix("~/"), &home) {
            (Some(rest), Some(h)) => h.join(rest),
            _ => std::path::PathBuf::from(f),
        };
        if !expanded.is_absolute() {
            return Err(format!("{f} is not an absolute path"));
        }
        if expanded.parent().is_none() {
            return Err("the whole filesystem cannot be a server's folder".into());
        }
        if home.as_ref().is_some_and(|h| &expanded == h) {
            return Err("your whole home folder cannot be a server's folder; name the folders it needs".into());
        }
        out.push(expanded);
    }
    Ok(out)
}

fn check_new_name(config: &ozgent_core::Config, name: &str) -> Result<(), String> {
    mcp::valid_name(name)?;
    if config.mcp.servers.contains_key(name) {
        return Err(format!("there is already a server called {name}"));
    }
    Ok(())
}

/// Save a new server, or only describe it when `preview` is set.
fn add(state: &State, name: &str, server: Server, preview: bool) -> Response {
    if let Err(e) = server.transport() {
        return fail(StatusCode::BAD_REQUEST, e);
    }
    let command = registry::preview(&server);
    if preview {
        return Json(json!({ "name": name, "preview": command, "sandbox": server.sandbox })).into_response();
    }
    {
        let config = state.config.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = check_new_name(&config, name) {
            return fail(StatusCode::CONFLICT, e);
        }
    }
    tracing::info!("mcp: adding {name}: {command}");
    let name_owned = name.to_string();
    let saved = save(state, move |c| {
        c.mcp.servers.insert(name_owned, server);
        // Adding a server is asking for it to run.
        c.mcp.enabled = true;
    });
    if saved.status().is_success() {
        reconnect(state);
        return Json(json!({ "ok": true, "name": name, "preview": command })).into_response();
    }
    saved
}

#[derive(Deserialize)]
pub(crate) struct Manual {
    name: String,
    /// `command`, `url`, `npm` or `pypi`.
    kind: String,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    package: Option<String>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(flatten)]
    placement: Placement,
    #[serde(default)]
    preview: bool,
}

fn check_pairs(what: &str, pairs: &BTreeMap<String, String>) -> Result<(), String> {
    for (k, v) in pairs {
        let name_ok = !k.is_empty()
            && k.len() <= 128
            && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !name_ok {
            return Err(format!("{k:?} is not a valid {what} name"));
        }
        if v.contains(['\0', '\n', '\r']) || v.len() > 8192 {
            return Err(format!("the {what} {k} has a line break or is too long"));
        }
    }
    Ok(())
}

/// Add a server by hand: a command, a URL, or an npm or PyPI package.
pub(crate) async fn add_manual(AxumState(state): AxumState<State>, Json(body): Json<Manual>) -> Response {
    let built = (|| -> Result<Server, String> {
        mcp::valid_name(&body.name)?;
        check_pairs("variable", &body.env)?;
        check_pairs("header", &body.headers)?;
        let mut server = match body.kind.as_str() {
            "npm" | "pypi" => {
                let package = body.package.as_deref().map(str::trim).filter(|p| !p.is_empty()).ok_or("name the package")?;
                let version = body.version.as_deref().map(str::trim).filter(|v| !v.is_empty());
                registry::plan_package(&body.kind, package, version, &body.args)?
            }
            "command" => {
                let command = body.command.as_deref().map(str::trim).filter(|c| !c.is_empty()).ok_or("give the command to run")?;
                Server { command: Some(command.to_string()), args: body.args.clone(), ..Default::default() }
            }
            "url" => {
                let url = body.url.as_deref().map(str::trim).filter(|u| !u.is_empty()).ok_or("give the server's URL")?;
                let local = url.starts_with("http://127.0.0.1") || url.starts_with("http://localhost") || url.starts_with("http://[::1]");
                if !url.starts_with("https://") && !local {
                    return Err("a remote server's URL must be https (http only for this machine)".into());
                }
                Server { url: Some(url.to_string()), ..Default::default() }
            }
            other => return Err(format!("unknown kind {other:?}")),
        };
        server.env = body.env.clone();
        server.headers = body.headers.clone();
        if let Some(d) = body.description.as_deref().map(str::trim).filter(|d| !d.is_empty()) {
            server.description = Some(d.chars().take(200).collect());
        }
        if body.kind == "command" && server.source.is_none() {
            server.source = Some("added by hand".into());
        }
        place(&mut server, &body.placement)?;
        Ok(server)
    })();
    match built {
        Ok(server) => add(&state, &body.name, server, body.preview),
        Err(e) => fail(StatusCode::BAD_REQUEST, e),
    }
}

#[derive(Deserialize)]
pub(crate) struct Paste {
    /// The JSON as another client's settings have it: `{"mcpServers": {…}}`,
    /// `{"servers": {…}}`, or one server's object.
    text: String,
    /// For a pasted single server, which has no name of its own.
    #[serde(default)]
    name: Option<String>,
    #[serde(flatten)]
    placement: Placement,
    #[serde(default)]
    preview: bool,
}

/// Add servers from a pasted block of another client's MCP settings — the
/// form nearly every server documents itself in. Each is reviewed the same
/// way as one added by hand: `preview` lists what each would run, and saving
/// adds every entry that can be used, reporting the rest.
pub(crate) async fn import(AxumState(state): AxumState<State>, Json(body): Json<Paste>) -> Response {
    let parsed = match ozgent_mcp::import::parse(&body.text, body.name.as_deref()) {
        Ok(p) => p,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let existing: Vec<String> = state.config.lock().unwrap_or_else(|e| e.into_inner()).mcp.servers.keys().cloned().collect();
    let mut ready: Vec<(String, Server)> = Vec::new();
    let mut report = Vec::new();
    for entry in parsed {
        let mut row = json!({ "name": entry.name, "notes": entry.notes });
        match entry.server {
            Ok(mut server) => {
                let placed = place(&mut server, &body.placement);
                if let Err(e) = placed {
                    row["error"] = json!(e);
                } else if existing.contains(&entry.name) || ready.iter().any(|(n, _)| n == &entry.name) {
                    row["error"] = json!(format!("there is already a server called {}", entry.name));
                } else {
                    row["preview"] = json!(registry::preview(&server));
                    row["sandbox"] = json!(server.sandbox);
                    row["folders"] = json!(server.folders);
                    ready.push((entry.name.clone(), server));
                }
            }
            Err(e) => row["error"] = json!(e),
        }
        report.push(row);
    }
    if body.preview || ready.is_empty() {
        return Json(json!({ "servers": report, "added": 0 })).into_response();
    }
    let added = ready.len();
    for (name, server) in &ready {
        tracing::info!("mcp: adding {name} from pasted JSON: {}", registry::preview(server));
    }
    let saved = save(&state, move |c| {
        for (name, server) in ready {
            c.mcp.servers.insert(name, server);
        }
        c.mcp.enabled = true;
    });
    if !saved.status().is_success() {
        return saved;
    }
    reconnect(&state);
    Json(json!({ "servers": report, "added": added })).into_response()
}

#[derive(Deserialize)]
pub(crate) struct Search {
    #[serde(default)]
    search: String,
    #[serde(default)]
    cursor: Option<String>,
}

/// Search the MCP Registry, from the server: the page's own policy lets it
/// talk to nothing but ozgent.
pub(crate) async fn registry_search(Query(q): Query<Search>) -> Response {
    match registry::search(&q.search, q.cursor.as_deref()).await {
        Ok((servers, next)) => {
            let servers: Vec<Value> = servers
                .iter()
                .map(|l| {
                    let mut v = serde_json::to_value(l).unwrap_or_default();
                    if let Some(options) = v["options"].as_array_mut() {
                        for (o, c) in options.iter_mut().zip(&l.options) {
                            o["runner_present"] = json!(registry::runner_present(c));
                        }
                    }
                    v["suggested_name"] = json!(registry::suggested_name(&l.name));
                    v
                })
                .collect();
            Json(json!({ "servers": servers, "next": next })).into_response()
        }
        Err(e) => fail(StatusCode::BAD_GATEWAY, e),
    }
}

#[derive(Deserialize)]
pub(crate) struct Install {
    /// The server's registry name and version, which are looked up again
    /// here: the page says which entry, never what it runs.
    registry_name: String,
    #[serde(default = "latest")]
    version: String,
    #[serde(default)]
    option: usize,
    #[serde(default)]
    values: BTreeMap<String, String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(flatten)]
    placement: Placement,
    #[serde(default)]
    preview: bool,
}

fn latest() -> String {
    "latest".into()
}

pub(crate) async fn install(AxumState(state): AxumState<State>, Json(body): Json<Install>) -> Response {
    let listing = match registry::fetch(&body.registry_name, &body.version).await {
        Ok(l) => l,
        Err(e) => return fail(StatusCode::BAD_GATEWAY, e),
    };
    let name = body.name.clone().filter(|n| !n.trim().is_empty()).unwrap_or_else(|| registry::suggested_name(&listing.name));
    if let Err(e) = mcp::valid_name(&name) {
        return fail(StatusCode::BAD_REQUEST, e);
    }
    if let Some(choice) = listing.options.get(body.option) {
        if !registry::runner_present(choice) {
            let runner = choice.runner.clone().unwrap_or_default();
            return fail(
                StatusCode::BAD_REQUEST,
                format!("this server runs with {runner}, which is not installed on this machine"),
            );
        }
    }
    let mut server = match registry::plan(&listing, body.option, &body.values) {
        Ok(s) => s,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    if let Err(e) = place(&mut server, &body.placement) {
        return fail(StatusCode::BAD_REQUEST, e);
    }
    add(&state, &name, server, body.preview)
}

/// What can change about a server. Every field is optional; what is sent
/// changes. `env` and `headers` merge: a string sets, `null` removes, a name
/// not mentioned is left alone — so a secret never has to be sent back to
/// keep it.
#[derive(Deserialize, Default)]
#[serde(default)]
pub(crate) struct Update {
    enabled: Option<bool>,
    trust_hints: Option<bool>,
    sandbox: Option<bool>,
    network: Option<bool>,
    folders: Option<Vec<String>>,
    timeout_seconds: Option<u64>,
    /// `auto`, `always` or `on_request`: when its tools are described.
    load: Option<ozgent_core::mcp::Load>,
    args: Option<Vec<String>>,
    env: BTreeMap<String, Option<String>>,
    headers: BTreeMap<String, Option<String>>,
    /// Offer only these tools (by their name on the server); `null` for all.
    #[serde(default, deserialize_with = "some_option")]
    only: Option<Option<Vec<String>>>,
    /// Per-tool rules by model-facing name: `allow`, `ask`, `deny`, or `""`
    /// to go back to the rule for its kind.
    rules: BTreeMap<String, String>,
}

/// Tells `"only": null` (all tools) apart from `only` not being sent.
fn some_option<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

pub(crate) async fn update(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
    Json(body): Json<Update>,
) -> Response {
    let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let Some(current) = config.mcp.servers.get(&name) else {
        return fail(StatusCode::NOT_FOUND, format!("there is no server called {name}"));
    };
    let mut server = current.clone();
    let checked = (|| -> Result<(), String> {
        let pairs = |what: &str, target: &mut BTreeMap<String, String>, change: &BTreeMap<String, Option<String>>| {
            let sets: BTreeMap<String, String> =
                change.iter().filter_map(|(k, v)| v.clone().map(|v| (k.clone(), v))).collect();
            check_pairs(what, &sets)?;
            for (k, v) in change {
                match v {
                    Some(v) => target.insert(k.clone(), v.clone()),
                    None => target.remove(k),
                };
            }
            Ok::<(), String>(())
        };
        pairs("variable", &mut server.env, &body.env)?;
        pairs("header", &mut server.headers, &body.headers)?;
        if let Some(v) = body.enabled {
            server.enabled = v;
        }
        if let Some(v) = body.trust_hints {
            server.trust_hints = v;
        }
        if let Some(v) = body.sandbox {
            server.sandbox = v;
        }
        if let Some(v) = body.network {
            server.network = v;
        }
        if let Some(f) = &body.folders {
            server.folders = folders(f)?;
        }
        if let Some(t) = body.timeout_seconds {
            server.timeout_seconds = t.clamp(1, 3600);
        }
        if let Some(l) = body.load {
            server.load = l;
        }
        if let Some(a) = &body.args {
            if a.iter().any(|x| x.contains(['\0', '\n', '\r'])) {
                return Err("an argument has a line break".into());
            }
            server.args = a.clone();
        }
        if let Some(only) = &body.only {
            server.tools = only.clone();
        }
        server.transport().map_err(|e| e.to_string())?;
        Ok(())
    })();
    if let Err(e) = checked {
        return fail(StatusCode::BAD_REQUEST, e);
    }

    // Rules only for tools this server offers, so this route cannot set a
    // rule for `write_file` or another server's tool.
    let own_prefix = format!("{}_", mcp::tool_name(&name, "x").strip_suffix("_x").unwrap_or(&name));
    let listed: Vec<String> = statuses(&state)
        .into_iter()
        .find(|s| s.name == name)
        .map(|s| s.tools.into_iter().map(|t| t.name).collect())
        .unwrap_or_default();
    let mut rules: Vec<(String, Option<Rule>)> = Vec::new();
    for (tool, rule) in &body.rules {
        if !listed.contains(tool) && !tool.starts_with(&own_prefix) {
            return fail(StatusCode::BAD_REQUEST, format!("{tool} is not one of {name}'s tools"));
        }
        let parsed = match rule.trim() {
            "" => None,
            r => match r.parse::<Rule>() {
                Ok(r) => Some(r),
                Err(e) => return fail(StatusCode::BAD_REQUEST, e),
            },
        };
        rules.push((tool.clone(), parsed));
    }

    // A change to what runs, or how, means reconnecting; a rule does not,
    // and nor does when its tools are described, which is read per turn.
    let reconnect_needed = Server { load: current.load, ..server.clone() } != *current;
    let saved = save(&state, move |c| {
        c.mcp.servers.insert(name, server);
        for (tool, rule) in rules {
            match rule {
                Some(r) => c.permissions.tools.insert(tool, r),
                None => c.permissions.tools.remove(&tool),
            };
        }
    });
    if saved.status().is_success() && reconnect_needed {
        reconnect(&state);
    }
    saved
}

pub(crate) async fn remove(AxumState(state): AxumState<State>, Path(name): Path<String>) -> Response {
    let exists = state.config.lock().unwrap_or_else(|e| e.into_inner()).mcp.servers.contains_key(&name);
    if !exists {
        return fail(StatusCode::NOT_FOUND, format!("there is no server called {name}"));
    }
    // Its tools' rules go with it: left behind, a later server given the
    // same name would inherit an `allow` nobody gave it.
    let listed: Vec<String> = statuses(&state)
        .into_iter()
        .find(|s| s.name == name)
        .map(|s| s.tools.into_iter().map(|t| t.name).collect())
        .unwrap_or_default();
    let home = crate::state::mcp_launcher(&state.paths, &state.config.lock().unwrap_or_else(|e| e.into_inner()))
        .map(|l| l.home(&name));
    let saved = save(&state, |c| {
        c.mcp.servers.remove(&name);
        for tool in &listed {
            c.permissions.tools.remove(tool);
        }
    });
    if saved.status().is_success() {
        // Its sandbox home: downloaded packages and whatever it kept.
        if let Some(home) = home.filter(|h| h.is_dir()) {
            let _ = std::fs::remove_dir_all(home);
        }
        reconnect(&state);
    }
    saved
}

/// For the chat page and the terminal: which servers exist and how they are
/// doing. Names, states and tool names only — no commands, no settings.
pub(crate) async fn summary(AxumState(state): AxumState<State>) -> Response {
    let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let statuses = statuses(&state);
    let servers: Vec<Value> = config
        .mcp
        .servers
        .iter()
        .map(|(name, s)| {
            let st = statuses.iter().find(|x| &x.name == name);
            json!({
                "name": name,
                "description": s.description,
                "enabled": s.enabled,
                // Switched off for the chats by whoever chats; see
                // `[tools] mcp_off`. The server itself still runs.
                "used": !config.tools.mcp_off.contains(name),
                "sandboxed": s.sandbox,
                "state": st.map(|x| x.state),
                "error": st.and_then(|x| x.error.clone()),
                "tools": st.map(|x| x.tools.iter().filter(|t| t.offered).map(|t| t.name.clone()).collect::<Vec<_>>()).unwrap_or_default(),
            })
        })
        .collect();
    Json(json!({
        "enabled": config.mcp.enabled,
        "tools_enabled": config.tools.enabled,
        "reconnecting": RECONNECTING.load(Ordering::SeqCst),
        "servers": servers,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_is_absolute_and_never_everything() {
        assert!(folders(&["relative/dir".into()]).is_err());
        assert!(folders(&["/".into()]).is_err());
        if let Some(home) = std::env::var_os("HOME") {
            assert!(folders(&[home.to_string_lossy().into_owned()]).is_err());
            let notes = folders(&["~/notes".into()]).unwrap();
            assert!(notes[0].is_absolute() && notes[0].ends_with("notes"));
        }
        assert!(folders(&["/srv/data".into(), "  ".into()]).unwrap().len() == 1);
    }

    #[test]
    fn a_server_added_here_is_sandboxed_unless_told_otherwise() {
        let mut s = Server { command: Some("npx".into()), ..Default::default() };
        place(&mut s, &Placement::default()).unwrap();
        assert!(s.sandbox && s.network);
        let mut remote = Server { url: Some("https://x.test/mcp".into()), ..Default::default() };
        place(&mut remote, &Placement::default()).unwrap();
        assert!(!remote.sandbox, "a URL has no program to sandbox");
        assert!(remote.transport().is_ok());
        let mut off = Server { command: Some("npx".into()), ..Default::default() };
        place(&mut off, &Placement { sandbox: Some(false), ..Default::default() }).unwrap();
        assert!(!off.sandbox);
    }

    #[test]
    fn variable_and_header_names_cannot_carry_anything_else() {
        let ok: BTreeMap<String, String> = [("GITHUB_TOKEN".to_string(), "x".to_string())].into();
        assert!(check_pairs("variable", &ok).is_ok());
        let bad: BTreeMap<String, String> = [("A=B".to_string(), "x".to_string())].into();
        assert!(check_pairs("variable", &bad).is_err());
        let newline: BTreeMap<String, String> = [("A".to_string(), "x\nB=y".to_string())].into();
        assert!(check_pairs("variable", &newline).is_err());
    }
}
