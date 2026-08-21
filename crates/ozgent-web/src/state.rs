//! Shared server state.

use ozgent_core::{Config, Paths};
use ozgent_memory::{HashingEmbedder, Store};
use std::sync::{Arc, Mutex};

use crate::worker::{Tools, Worker};

/// One tool as the settings page shows it.
#[derive(serde::Serialize)]
pub struct ToolSummary {
    pub name: String,
    pub description: String,
    pub enabled: bool,
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
) -> anyhow::Result<Vec<ToolSummary>> {
    let mut tools_config = config.tools.clone();
    let disabled = std::mem::take(&mut tools_config.disabled);

    let host_config = ozgent_tools::HostConfig::from_config(&tools_config, paths)?;
    let host = ozgent_tools::ToolHost::start(host_config).await?;

    let mut out: Vec<ToolSummary> = host
        .tools()
        .iter()
        .map(|t| ToolSummary {
            name: t.name.clone(),
            description: t.description.clone(),
            enabled: !disabled.contains(&t.name),
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    host.shutdown().await;
    Ok(out)
}

/// Everything the handlers need.
///
/// The SQLite connection is not `Sync`, so it sits behind a mutex; queries are
/// short and the alternative — a pool — buys nothing for a single-user local
/// server.
pub struct App {
    pub paths: Paths,
    pub config: Mutex<Config>,
    pub store: Mutex<Store>,
    pub embedder: HashingEmbedder,
    pub worker: Worker,
}

pub type State = Arc<App>;

impl App {
    pub async fn new(paths: Paths, config: Config) -> anyhow::Result<State> {
        let store = Store::open(paths.root().join("ozgent.db"))?;

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
        let worker = Worker::spawn(paths.clone(), config.clone(), tools);
        Ok(Arc::new(App {
            paths,
            config: Mutex::new(config),
            store: Mutex::new(store),
            embedder: HashingEmbedder::default(),
            worker,
        }))
    }
}

/// Start the Python tool worker for the server's lifetime.
async fn start_tools(paths: &Paths, config: &Config) -> anyhow::Result<Tools> {
    let host_config = ozgent_tools::HostConfig::from_config(&config.tools, paths)?;
    let host = ozgent_tools::ToolHost::start(host_config).await?;
    tracing::info!("tools ready: {} available", host.tools().len());
    Ok(Tools {
        host: std::sync::Arc::new(host),
        runtime: tokio::runtime::Handle::current(),
    })
}
