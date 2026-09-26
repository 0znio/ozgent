//! Shared server state.

use ozgent_core::{Config, Paths};
use ozgent_memory::Store;
use std::sync::{Arc, Mutex};

use crate::worker::{Permissions, SharedConfig, SharedTools, Tools, Worker};

/// One tool as the settings page shows it.
#[derive(serde::Serialize)]
pub struct ToolSummary {
    pub name: String,
    pub description: String,
    pub enabled: bool,
    /// `mcp:<server>` for a tool from an MCP server; absent for ozgent's own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// Ask the Python worker what tools exist.
///
/// This starts an interpreter and shuts it down again, which is why it is only
/// done when the settings page asks: the chat path has no reason to pay for it.
/// Tools named in `disabled` are still listed, marked off, so the page can show
/// something to switch back on.
pub async fn discover_tools(
    paths: &Paths,
    config: &Config,
    running: Option<&Tools>,
) -> anyhow::Result<Vec<ToolSummary>> {
    let mut tools_config = config.tools.clone();
    let disabled = std::mem::take(&mut tools_config.disabled);

    let host_config = ozgent_tools::HostConfig::from_config(&tools_config, paths)?;
    let host = ozgent_tools::ToolHost::start(host_config).await?;
    let host = ozgent_tools::Toolbox::new(Some(host), Vec::new());

    let mut out: Vec<ToolSummary> = host
        .tools()
        .iter()
        .map(|t| ToolSummary {
            name: t.name.clone(),
            description: t.description.clone(),
            enabled: !disabled.contains(&t.name),
            source: None,
        })
        .collect();
    host.shutdown().await;

    // MCP servers too: the settings page lists what the model can actually
    // reach, and a tool from a server is no less a tool. Taken from the
    // servers already running, never by starting them again: that launched
    // every server a second time each time the page opened, `npx` downloads
    // and all.
    if let Some(running) = running {
        for t in running.host.tools() {
            let Some(source) = running.host.source_of(&t.name).filter(|s| s.starts_with("mcp:")) else {
                continue;
            };
            out.push(ToolSummary {
                name: t.name.clone(),
                description: t.description.clone(),
                enabled: !disabled.contains(&t.name),
                source: Some(source),
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// What MCP servers need to run in ozgent's sandbox, when the Python
/// runtime that provides it can be found.
pub fn mcp_launcher(paths: &Paths, config: &Config) -> Option<ozgent_mcp::Launcher> {
    ozgent_mcp::Launcher::from_config(config, paths)
}

/// Everything the handlers need.
///
/// The SQLite connection is not `Sync`, so it sits behind a mutex; queries are
/// short and the alternative — a pool — buys nothing for a single-user local
/// server.
pub struct App {
    pub paths: Paths,
    /// The Python tool host, shared so a settings change can replace it.
    pub tools: SharedTools,
    /// Shared with the inference thread, which re-reads it whenever it loads
    /// a model. A clone handed over at start-up meant the settings page could
    /// save a change, report it applied, and have it never reach the model.
    pub config: SharedConfig,
    /// Permission questions in flight and answers given this run. Shared with
    /// the inference thread, which asks, and the handlers, which answer.
    pub permissions: Permissions,
    pub store: Mutex<Store>,
    /// The memory layer's embedder: the embedding model when there is one.
    pub embedder: crate::memory::MemoryEmbedder,
    pub worker: Worker,
    /// Model downloads started from the browser. They belong to the server,
    /// so closing the tab that started one does not stop it.
    pub pulls: crate::hub::SharedPulls,
    /// The messaging gateway, when this process runs one. Set once, by
    /// whoever starts it; `/admin` reaches it through here.
    pub gateway: std::sync::OnceLock<Arc<dyn crate::admin::GatewayControl>>,
    /// Admin sessions and failed sign-ins.
    pub admin: crate::admin::Guard,
    /// Proof that a request comes from this user on this machine. Issued by
    /// the process that binds the port, once it has, so a second server that
    /// fails to start cannot replace the running one's.
    pub local_token: std::sync::OnceLock<String>,
    /// Rate limits, lockouts and connection counts; see [`crate::access`].
    pub shield: crate::access::Shield,
    watching: std::sync::atomic::AtomicBool,
}

pub type State = Arc<App>;

/// How often `config.toml` is checked for changes made elsewhere.
const WATCH_EVERY: std::time::Duration = std::time::Duration::from_secs(2);

/// Follow changes other programs make to `config.toml`.
///
/// `ozgent gateway telegram` and `ozgent admin reset` run as separate
/// processes and write the file; a running server holds the configuration in
/// memory and would otherwise go on using what it read at startup. Only the
/// sections that are safe to swap underneath a running server are taken —
/// `[channels]` and `[web]` — and the gateway is then brought in line.
///
/// Idempotent: both the web server and the gateway call it.
pub fn watch_config(state: &State) {
    use std::sync::atomic::Ordering;
    if state.watching.swap(true, Ordering::SeqCst) {
        return;
    }
    let state = Arc::clone(state);
    tokio::spawn(async move {
        let path = state.paths.config_file();
        let modified = |p: &std::path::Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
        let mut seen = modified(&path);
        let mut warned = false;
        let mut revived: Option<std::time::Instant> = None;
        loop {
            tokio::time::sleep(WATCH_EVERY).await;
            // ozgent's own tool worker, if it has died — a Python tool that
            // took the interpreter down, or the process killed — is started
            // again: until then every search, fetch and file tool fails.
            // Not more than once a minute, so one that dies on start does
            // not loop.
            let dead = {
                let tools = state.tools.lock().unwrap_or_else(|e| e.into_inner());
                tools.as_ref().is_some_and(|t| !t.host.python_alive())
            };
            if dead && revived.is_none_or(|at| at.elapsed() > std::time::Duration::from_secs(60)) {
                revived = Some(std::time::Instant::now());
                tracing::warn!("the tool worker stopped; starting the tools again");
                let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
                crate::api::restart_tools(&state, config).await;
            }
            let now = modified(&path);
            if now == seen {
                continue;
            }
            let fresh = match Config::load(&state.paths) {
                Ok(c) => c,
                Err(e) => {
                    // Mid-write, or a hand edit with a typo. Tried again on
                    // the next change; said once rather than every tick.
                    if !warned {
                        tracing::warn!("config.toml changed but does not load: {e}");
                        warned = true;
                    }
                    continue;
                }
            };
            warned = false;
            seen = now;
            let changed = {
                let mut config = state.config.lock().unwrap_or_else(|e| e.into_inner());
                // Everything that is read per turn, which is everything except
                // `[tools]`. The tool worker reads its configuration once when
                // the interpreter starts, so swapping that section here would
                // report a change that never reached a tool — the settings
                // page restarts the host for exactly that reason.
                //
                // This grew beyond `[channels]` and `[web]` when the terminal
                // became a client: `/config temperature` writes the file, and
                // without this it wrote a file nothing read.
                // Swapped rather than compared: most of these types have no
                // equality, the file only changes when something wrote it,
                // and assigning a section that happens to be identical costs
                // nothing.
                config.channels = fresh.channels;
                config.web = fresh.web;
                config.defaults = fresh.defaults;
                config.models = fresh.models;
                config.permissions = fresh.permissions;
                config.ui = fresh.ui;
                config.default_model = fresh.default_model;
                true
            };
            // `[mcp]` is taken too, and the servers reconnected: `ozgent mcp
            // add` and friends write the file, and a server added from the
            // terminal should not wait for a restart.
            let mcp_changed = {
                let mut config = state.config.lock().unwrap_or_else(|e| e.into_inner());
                if config.mcp != fresh.mcp {
                    config.mcp = fresh.mcp;
                    true
                } else {
                    false
                }
            };
            if mcp_changed {
                tracing::info!("[mcp] changed; reconnecting the servers");
                // Queued with the admin page's own reconnects, so the two
                // cannot finish in the wrong order.
                crate::admin_mcp::reconnect(&state);
            }
            if changed {
                tracing::info!("config.toml changed; reloaded");
            }
            if let Some(gateway) = state.gateway.get() {
                gateway.apply();
            }
        }
    });
}

impl App {
    pub async fn new(
        paths: Paths,
        config: Config,
        cli: ozgent_core::Options,
    ) -> anyhow::Result<State> {
        let store = Store::open(paths.root().join("ozgent.db"))?;
        harden_home(&paths);

        // Attachments are kept for a month; sweeping at startup avoids a timer
        // and a local server is restarted often enough for that to be enough.
        match crate::media::sweep(&paths) {
            Ok(0) => {}
            Ok(n) => tracing::info!("removed {n} attachment(s) older than {} days", crate::media::RETENTION_DAYS),
            Err(e) => tracing::warn!("sweeping old attachments: {e}"),
        }

        // Started once and kept for the life of the server: spawning an
        // interpreter per turn would add a second of latency to every message.
        let tools = if config.tools.enabled {
            match start_tools(&paths, &config).await {
                Ok(t) => Some(t),
                Err(e) => {
                    tracing::warn!("tools unavailable: {e}");
                    None
                }
            }
        } else {
            None
        };
        let config: SharedConfig = Arc::new(Mutex::new(config));
        let tools: SharedTools = Arc::new(Mutex::new(tools));
        // Shared with the handlers, not owned by the worker: a permission
        // question is asked on the inference thread and answered by an HTTP
        // request on another, and the settings page reads the same grants to
        // show what has been allowed for this run.
        let permissions = Permissions {
            pending: Arc::new(crate::permission::Pending::default()),
            grants: Arc::new(Mutex::new(ozgent_core::Grants::default())),
            config: Arc::clone(&config),
        };
        let worker = Worker::spawn(
            paths.clone(),
            Arc::clone(&config),
            Arc::clone(&tools),
            permissions.clone(),
            std::sync::Arc::new(cli),
        );
        Ok(Arc::new(App {
            paths,
            config,
            permissions,
            tools,
            store: Mutex::new(store),
            embedder: crate::memory::MemoryEmbedder::new(worker.clone()),
            worker,
            pulls: Default::default(),
            gateway: std::sync::OnceLock::new(),
            admin: Default::default(),
            local_token: std::sync::OnceLock::new(),
            shield: Default::default(),
            watching: Default::default(),
        }))
    }
}

/// Start the Python tool worker and every MCP server, for the server's life.
pub async fn start_tools(paths: &Paths, config: &Config) -> anyhow::Result<Tools> {
    let host_config = ozgent_tools::HostConfig::from_config(&config.tools, paths)?;
    let python = ozgent_tools::ToolHost::start(host_config).await?;
    // A server that will not start is reported and skipped: one bad entry in
    // config.toml must not take away the tools that do work.
    let launcher = mcp_launcher(paths, config);
    let (mut sources, mcp) = ozgent_mcp::connect_all_with(&config.mcp, launcher.as_ref()).await;
    for problem in ozgent_mcp::problems(&mcp) {
        tracing::warn!("mcp: {problem}");
    }
    // The scheduler, so the model can make and change jobs mid-conversation.
    // Offered as a source rather than a Python tool because it writes to
    // ozgent's own database, which the Python worker is sandboxed away from.
    let scheduler = match ozgent_schedule::ScheduleTools::open(paths.root()) {
        Ok(s) => {
            let s = std::sync::Arc::new(s);
            sources.push(s.clone());
            Some(s)
        }
        Err(e) => {
            tracing::warn!("the scheduler tool is unavailable: {e}");
            None
        }
    };

    let host = ozgent_tools::Toolbox::new(Some(python), sources);
    for shadowed in host.shadowed() {
        tracing::warn!(
            "{} from {} is not offered: {} already has that name",
            shadowed.name, shadowed.from, shadowed.kept
        );
    }
    tracing::info!("tools ready: {} available", host.tools().len());
    Ok(Tools {
        host: std::sync::Arc::new(host),
        runtime: tokio::runtime::Handle::current(),
        scheduler,
        mcp: std::sync::Arc::new(mcp),
    })
}

/// Make what ozgent keeps private to its user.
///
/// The database holds every conversation and was created world-readable; the
/// channel directory holds a logged-in WhatsApp session. On a machine with
/// other accounts either is someone else's to read. The models and the root
/// itself are left alone: a model directory is sometimes shared on purpose.
/// Best effort — a file that cannot be changed is not a reason not to start.
fn harden_home(paths: &Paths) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let set = |p: std::path::PathBuf, mode: u32| {
            if let Ok(meta) = std::fs::metadata(&p) {
                if meta.permissions().mode() & 0o777 != mode {
                    if let Err(e) = std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)) {
                        tracing::warn!("could not make {} private: {e}", p.display());
                    }
                }
            }
        };
        for name in ["ozgent.db", "ozgent.db-wal", "ozgent.db-shm"] {
            set(paths.root().join(name), 0o600);
        }
        set(paths.configs_dir(), 0o700);
        set(paths.channels_dir(), 0o700);
        set(paths.agents_dir(), 0o700);
    }
    #[cfg(not(unix))]
    let _ = paths;
}
