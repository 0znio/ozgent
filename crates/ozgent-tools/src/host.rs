//! Supervises the Python tool worker.
//!
//! One long-lived process handles every call, so the interpreter starts and
//! the tool modules import exactly once. Requests are multiplexed over its
//! stdio by id, and a reader task fans replies back out to the callers waiting
//! on them.
//!
//! Keeping tools out-of-process is what makes them safe to run: a tool that
//! hangs is cancelled, one that crashes takes down only itself, and neither
//! can disturb inference.

use crate::protocol::*;
use ozgent_core::ToolSpec;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, oneshot};
use tracing::{debug, error, warn};

/// Environment variable pointing at the directory containing `ozgent_tools`.
pub const RUNTIME_ENV: &str = "OZGENT_PYTHON_PATH";

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, RpcError>>>>>;

#[derive(Debug, Clone)]
pub struct HostConfig {
    /// Interpreter to run. A virtualenv's `python` works here.
    pub python: String,
    /// Directory containing the `ozgent_tools` package.
    pub runtime_path: PathBuf,
    /// Directories scanned for user-authored tools.
    pub tool_paths: Vec<PathBuf>,
    /// Tool names to refuse to load.
    pub disabled: Vec<String>,
    /// Per-tool settings from `[tools.config.<name>]`, passed through as JSON.
    pub tool_config: Value,
    /// Per-call wall-clock budget.
    pub timeout: Duration,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            python: "python3".into(),
            runtime_path: PathBuf::new(),
            tool_paths: Vec::new(),
            disabled: Vec::new(),
            tool_config: json!({}),
            timeout: Duration::from_secs(30),
        }
    }
}

impl HostConfig {
    /// Build from the user's config, resolving the runtime directory.
    pub fn from_config(
        cfg: &ozgent_core::config::ToolsConfig,
        paths: &ozgent_core::Paths,
    ) -> Result<Self, HostError> {
        let mut tool_paths = vec![paths.tools_dir()];
        tool_paths.extend(cfg.extra_paths.iter().cloned());

        Ok(Self {
            python: cfg.python.clone(),
            runtime_path: resolve_runtime()?,
            tool_paths,
            disabled: cfg.disabled.clone(),
            tool_config: toml_to_json(&toml::Value::Table(
                cfg.config.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            )),
            timeout: Duration::from_secs(cfg.timeout_seconds),
        })
    }
}

/// Locate the bundled Python runtime.
///
/// Checked in order: the override env var, a `python/` directory beside the
/// executable (installed layout), the repository's `python/` (cargo run), then
/// the ozgent home.
pub fn resolve_runtime() -> Result<PathBuf, HostError> {
    let mut tried = Vec::new();
    let check = |p: PathBuf, tried: &mut Vec<PathBuf>| -> Option<PathBuf> {
        if p.join("ozgent_tools").join("__init__.py").is_file() {
            return Some(p);
        }
        tried.push(p);
        None
    };

    if let Some(raw) = std::env::var_os(RUNTIME_ENV) {
        let p = PathBuf::from(raw);
        if let Some(found) = check(p.clone(), &mut tried) {
            return Ok(found);
        }
        // An explicit override that does not resolve is a configuration error,
        // not something to silently fall back from.
        return Err(HostError::RuntimeNotFound { tried: vec![p] });
    }

    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().skip(1).take(5) {
            if let Some(found) = check(ancestor.join("python"), &mut tried) {
                return Ok(found);
            }
        }
    }
    if let Ok(paths) = ozgent_core::Paths::discover() {
        if let Some(found) = check(paths.root().join("runtime"), &mut tried) {
            return Ok(found);
        }
    }

    Err(HostError::RuntimeNotFound { tried })
}

pub struct ToolHost {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    pending: Pending,
    next_id: AtomicU64,
    tools: Vec<ToolSpec>,
    /// Where each tool was defined, by name. Kept beside the specs rather
    /// than inside them: a `ToolSpec` is what the model is shown, and a
    /// filesystem path is neither useful nor safe to put there.
    sources: HashMap<String, String>,
    /// Tool files that failed to import, reported once at startup.
    load_errors: Vec<String>,
    timeout: Duration,
    worker_version: String,
    python_version: String,
}

impl ToolHost {
    /// Spawn the worker and complete the `initialize` handshake.
    pub async fn start(cfg: HostConfig) -> Result<Self, HostError> {
        let mut command = Command::new(&cfg.python);
        command
            .arg("-m")
            .arg("ozgent_tools.worker")
            .env("PYTHONPATH", prepend_pythonpath(&cfg.runtime_path))
            // Without this the worker's stderr arrives in unhelpful bursts.
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|e| HostError::Spawn {
            python: cfg.python.clone(),
            source: e,
        })?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        tokio::spawn(read_replies(stdout, Arc::clone(&pending)));
        tokio::spawn(forward_stderr(stderr));

        let host = Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicU64::new(1),
            tools: Vec::new(),
            sources: HashMap::new(),
            load_errors: Vec::new(),
            timeout: cfg.timeout,
            worker_version: String::new(),
            python_version: String::new(),
        };

        host.initialize(cfg).await
    }

    async fn initialize(mut self, cfg: HostConfig) -> Result<Self, HostError> {
        let params = json!({
            "tool_paths": cfg.tool_paths,
            "disabled": cfg.disabled,
            "config": cfg.tool_config,
        });

        // Startup imports every tool module, which can be slower than a call.
        let raw = self
            .request("initialize", Some(params), self.timeout.max(Duration::from_secs(60)))
            .await
            .map_err(|e| HostError::Handshake(e.to_string()))?;

        let init: InitializeResult =
            serde_json::from_value(raw).map_err(|e| HostError::Handshake(e.to_string()))?;

        if init.protocol_version != PROTOCOL_VERSION {
            return Err(HostError::ProtocolMismatch {
                ours: PROTOCOL_VERSION,
                theirs: init.protocol_version,
            });
        }

        for err in &init.errors {
            warn!(target: "ozgent::tools", "tool failed to load: {err}");
        }

        self.sources = init
            .tools
            .iter()
            .filter(|t| !t.source.is_empty())
            .map(|t| (t.name.clone(), t.source.clone()))
            .collect();
        self.tools = init.tools.into_iter().map(Into::into).collect();
        self.load_errors = init.errors;
        self.worker_version = init.worker_version;
        self.python_version = init.python;
        Ok(self)
    }

    /// Tools available to the model, in a form ready for a chat template.
    pub fn tools(&self) -> &[ToolSpec] {
        &self.tools
    }

    pub fn load_errors(&self) -> &[String] {
        &self.load_errors
    }

    pub fn python_version(&self) -> &str {
        &self.python_version
    }

    pub fn worker_version(&self) -> &str {
        &self.worker_version
    }

    /// The Python file a tool was defined in, if the worker reported one.
    pub fn source_of(&self, name: &str) -> Option<&str> {
        self.sources.get(name).map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Invoke a tool and wait for its result.
    ///
    /// On timeout the call is cancelled inside the worker rather than merely
    /// abandoned, so a runaway tool does not keep consuming resources.
    pub async fn call(&self, name: &str, arguments: Value) -> Result<Value, ToolCallError> {
        self.call_approved(name, arguments, false).await
    }

    /// Invoke a tool, saying whether a person authorised this particular call.
    ///
    /// `approved` is not a convenience: the Python side keeps its own
    /// boundaries — a root directory for file tools, an allowlist for
    /// commands — and those exist to answer "what may run with nobody
    /// looking". A call the user read and approved is past that question, so
    /// the flag lifts them for that call only.
    ///
    /// It defaults to false through [`ToolHost::call`] on purpose. Forgetting
    /// to pass it means the sandbox applies, which is the harmless mistake;
    /// the harmful one would need someone to write `true`.
    pub async fn call_approved(
        &self,
        name: &str,
        arguments: Value,
        approved: bool,
    ) -> Result<Value, ToolCallError> {
        let call_id = format!("c{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let params = json!({
            "name": name,
            "arguments": arguments,
            "call_id": call_id,
            "approved": approved,
        });

        match self.request("call", Some(params), self.timeout).await {
            Ok(v) => Ok(v),
            Err(RequestError::Timeout) => {
                let _ = self
                    .notify("cancel", Some(json!({ "call_id": call_id })))
                    .await;
                Err(ToolCallError::Timeout { name: name.into(), after: self.timeout })
            }
            Err(RequestError::Rpc(e)) => Err(ToolCallError::Failed { name: name.into(), error: e }),
            Err(RequestError::Transport(e)) => Err(ToolCallError::Transport(e)),
        }
    }

    async fn request(
        &self,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
    ) -> Result<Value, RequestError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        if let Err(e) = self.write(&Request::new(id, method, params)).await {
            self.pending.lock().await.remove(&id);
            return Err(RequestError::Transport(e));
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(Ok(value))) => Ok(value),
            Ok(Ok(Err(rpc))) => Err(RequestError::Rpc(rpc)),
            // The sender was dropped, which means the reader task exited.
            Ok(Err(_)) => Err(RequestError::Transport("worker exited unexpectedly".into())),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(RequestError::Timeout)
            }
        }
    }

    /// Fire-and-forget request; used for `cancel`, where the reply is noise.
    async fn notify(&self, method: &str, params: Option<Value>) -> Result<(), String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.write(&Request::new(id, method, params)).await
    }

    async fn write(&self, req: &Request<'_>) -> Result<(), String> {
        let mut line = serde_json::to_vec(req).map_err(|e| e.to_string())?;
        line.push(b'\n');
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(&line).await.map_err(|e| e.to_string())?;
        stdin.flush().await.map_err(|e| e.to_string())
    }

    /// Ask the worker to exit, then reap it. Falls back to a kill.
    pub async fn shutdown(&self) {
        let _ = self.notify("shutdown", None).await;
        let mut child = self.child.lock().await;
        let deadline = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
        if deadline.is_err() {
            warn!(target: "ozgent::tools", "worker did not exit; killing it");
            let _ = child.kill().await;
        }
    }
}

async fn read_replies(stdout: tokio::process::ChildStdout, pending: Pending) {
    let mut lines = BufReader::new(stdout).lines();
    loop {
        let line = match lines.next_line().await {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(e) => {
                error!(target: "ozgent::tools", "reading worker stdout: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }

        let resp: Response = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                error!(target: "ozgent::tools", "unparseable worker output: {e}: {line}");
                continue;
            }
        };

        let Some(id) = resp.id else { continue };
        let Some(tx) = pending.lock().await.remove(&id) else {
            debug!(target: "ozgent::tools", "reply for unknown id {id}");
            continue;
        };
        let _ = tx.send(match resp.error {
            Some(e) => Err(e),
            None => Ok(resp.result.unwrap_or(Value::Null)),
        });
    }

    // The worker is gone; wake everyone still waiting rather than letting them
    // block until their individual timeouts expire.
    pending.lock().await.clear();
}

async fn forward_stderr(stderr: tokio::process::ChildStderr) {
    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if !line.trim().is_empty() {
            debug!(target: "ozgent::tools::py", "{line}");
        }
    }
}

fn prepend_pythonpath(runtime: &Path) -> std::ffi::OsString {
    let mut value = runtime.as_os_str().to_owned();
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        if !existing.is_empty() {
            value.push(if cfg!(windows) { ";" } else { ":" });
            value.push(existing);
        }
    }
    value
}

/// Convert TOML to JSON so tool settings survive the trip unchanged.
pub fn toml_to_json(v: &toml::Value) -> Value {
    match v {
        toml::Value::String(s) => Value::String(s.clone()),
        toml::Value::Integer(i) => Value::Number((*i).into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        toml::Value::Boolean(b) => Value::Bool(*b),
        toml::Value::Datetime(d) => Value::String(d.to_string()),
        toml::Value::Array(a) => Value::Array(a.iter().map(toml_to_json).collect()),
        toml::Value::Table(t) => {
            Value::Object(t.iter().map(|(k, v)| (k.clone(), toml_to_json(v))).collect())
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HostError {
    #[error("could not start {python:?}: {source}. Is Python installed and on your PATH?")]
    Spawn { python: String, source: std::io::Error },
    #[error(
        "could not find the ozgent Python runtime. Set {RUNTIME_ENV} to the directory containing \
         the `ozgent_tools` package. Looked in: {}",
        tried.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )]
    RuntimeNotFound { tried: Vec<PathBuf> },
    #[error("tool worker handshake failed: {0}")]
    Handshake(String),
    #[error("tool worker speaks protocol {theirs}, this build speaks {ours}; reinstall ozgent")]
    ProtocolMismatch { ours: u32, theirs: u32 },
}

#[derive(Debug, thiserror::Error)]
enum RequestError {
    #[error("{0}")]
    Rpc(RpcError),
    #[error("timed out")]
    Timeout,
    #[error("{0}")]
    Transport(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ToolCallError {
    #[error("{name} failed: {error}")]
    Failed { name: String, error: RpcError },
    #[error("{name} timed out after {after:?}")]
    Timeout { name: String, after: Duration },
    #[error("tool worker unavailable: {0}")]
    Transport(String),
    /// The user was asked and said no. Not a failure — nothing went wrong —
    /// but it travels the same path as one, because what the model needs is a
    /// tool result either way.
    #[error("{name} was declined")]
    Declined { name: String },
    /// Called by name, but not one of the tools this turn was offered — an
    /// agent reaching past its list. Nobody refused it; it was never there.
    #[error("{name} is not available here")]
    NotOffered { name: String, offered: Vec<String> },
    /// The call was understood and could not be done as asked — handing to
    /// an agent that does not exist, say. The reason is for the model.
    #[error("{name}: {reason}")]
    Invalid { name: String, reason: String },
}

impl ToolCallError {
    /// Render the failure for the model, as the content of a tool result.
    ///
    /// A tool bug is described without inviting a retry; a bad-argument or
    /// explicit tool error tells the model it may try again.
    pub fn for_model(&self) -> String {
        match self {
            Self::Failed { name, error } if error.is_model_fault() => {
                if error.retryable() {
                    format!("Error from {name}: {error}. This may succeed if retried.")
                } else {
                    format!("Error from {name}: {error}")
                }
            }
            Self::Failed { name, error } => {
                format!("The {name} tool failed internally: {error}. Do not retry; tell the user.")
            }
            Self::Timeout { name, after } => format!(
                "The {name} tool timed out after {}s and was cancelled.",
                after.as_secs()
            ),
            Self::Transport(e) => {
                format!("The tool system is unavailable: {e}. Do not retry; tell the user.")
            }
            Self::Declined { name } => ozgent_core::permission::refusal(name),
            Self::NotOffered { name, offered } if offered.is_empty() => format!(
                "{name} is not available to you, and you have no tools here. \
                 Answer from what you already have."
            ),
            Self::NotOffered { name, offered } => format!(
                "{name} is not one of your tools. You can use: {}. Carry on with those.",
                offered.join(", ")
            ),
            Self::Invalid { name, reason } => format!("Error from {name}: {reason}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_settings_survive_conversion() {
        let t: toml::Value = toml::from_str(
            r#"
            provider = "brave"
            max_results = 5
            ratio = 0.5
            enabled = true
            tags = ["a", "b"]
            [nested]
            key = "value"
            "#,
        )
        .unwrap();

        let j = toml_to_json(&t);
        assert_eq!(j["provider"], "brave");
        assert_eq!(j["max_results"], 5);
        assert_eq!(j["ratio"], 0.5);
        assert_eq!(j["enabled"], true);
        assert_eq!(j["tags"][1], "b");
        assert_eq!(j["nested"]["key"], "value");
    }

    #[test]
    fn pythonpath_prepends_without_losing_existing_entries() {
        // SAFETY: single-threaded test; no other thread reads the environment.
        unsafe { std::env::set_var("PYTHONPATH", "/existing") };
        let joined = prepend_pythonpath(Path::new("/runtime"));
        let s = joined.to_string_lossy();
        assert!(s.starts_with("/runtime"), "ours must win: {s}");
        assert!(s.contains("/existing"), "must not discard the user's path: {s}");
        unsafe { std::env::remove_var("PYTHONPATH") };
    }

    #[test]
    fn model_facing_errors_distinguish_fault() {
        let bad_args = ToolCallError::Failed {
            name: "web_search".into(),
            error: RpcError { code: INVALID_PARAMS, message: "missing query".into(), data: None },
        };
        assert!(bad_args.for_model().contains("missing query"));
        assert!(!bad_args.for_model().contains("Do not retry"));

        let crashed = ToolCallError::Failed {
            name: "web_search".into(),
            error: RpcError { code: INTERNAL_ERROR, message: "boom".into(), data: None },
        };
        assert!(crashed.for_model().contains("Do not retry"));
    }
}
