//! Running a stdio server inside ozgent's sandbox.
//!
//! The sandbox itself is the launcher `run_command` already uses
//! (`ozgent_tools/sandbox.py`): namespaces, Landlock, a clean environment.
//! What a server may reach is decided where every other sandbox policy is
//! decided, in `ozgent_tools.permissions`, so there is one list of system
//! directories and one list of credential folders, not a second copy here
//! that drifts. This file only asks for the policy and wraps the command.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde_json::json;

/// What ozgent needs to put a server in its sandbox.
#[derive(Debug, Clone)]
pub struct Launcher {
    /// The interpreter the tool worker uses.
    pub python: String,
    /// The directory holding the `ozgent_tools` package.
    pub runtime_path: PathBuf,
    /// Where each server's home is made: `<homes>/<name>`.
    pub homes: PathBuf,
    /// Directories no server may reach: ozgent's own home, tool folders.
    pub protected: Vec<PathBuf>,
}

/// A command, rewritten to start inside the sandbox.
#[derive(Debug, Clone)]
pub struct Wrapped {
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub cwd: PathBuf,
}

/// The environment variable the launcher reads its policy from.
///
/// Not an argument: the policy carries the server's environment, and with it
/// any API key the server was configured with, and every user on the machine
/// can read another process's arguments.
const POLICY_ENV: &str = "OZGENT_SANDBOX_POLICY";

const POLICY_SCRIPT: &str = "import json, sys\n\
from ozgent_tools.permissions import mcp_sandbox\n\
a = json.loads(sys.stdin.read())\n\
print(json.dumps(mcp_sandbox(a['program'], a['home'], a['folders'], a['network'], a['env'])))\n";

impl Launcher {
    /// The launcher for this installation: the tool worker's interpreter and
    /// runtime, homes under `~/ozgent/mcp`, and the tool worker's protected
    /// directories.
    pub fn from_config(config: &ozgent_core::Config, paths: &ozgent_core::Paths) -> Option<Self> {
        let host = ozgent_tools::HostConfig::from_config(&config.tools, paths).ok()?;
        Some(Self {
            python: host.python,
            runtime_path: host.runtime_path,
            homes: paths.root().join("mcp"),
            protected: host.protected,
        })
    }

    /// This server's home, where it runs and keeps what it downloads.
    pub fn home(&self, name: &str) -> PathBuf {
        self.homes.join(name)
    }

    /// Rewrite `command args` to run inside the sandbox.
    pub async fn wrap(
        &self,
        name: &str,
        command: &str,
        args: &[String],
        env: &BTreeMap<String, String>,
        folders: &[PathBuf],
        network: bool,
    ) -> Result<Wrapped, String> {
        ozgent_core::mcp::valid_name(name)?;
        let home = self.home(name);
        let request = json!({
            "program": command,
            "home": home,
            "folders": folders,
            "network": network,
            "env": env,
        });
        let policy = self.policy(&request).await?;
        let launch_env: BTreeMap<String, String> = policy
            .get("env")
            .and_then(|e| serde_json::from_value(e.clone()).ok())
            .unwrap_or_default();

        let script = self.runtime_path.join("ozgent_tools").join("sandbox.py");
        let mut wrapped_args = vec!["-S".to_string(), "-E".to_string(), script.display().to_string(), "-".to_string(), "--".to_string()];
        wrapped_args.push(command.to_string());
        wrapped_args.extend(args.iter().cloned());

        // The launcher itself sees the server's environment (it passes it
        // on at exec) and the policy, which it removes before the exec.
        let mut env = launch_env;
        env.insert(POLICY_ENV.to_string(), policy.to_string());
        Ok(Wrapped { command: self.python.clone(), args: wrapped_args, env, cwd: home })
    }

    /// Ask `ozgent_tools.permissions` what this server may reach.
    async fn policy(&self, request: &serde_json::Value) -> Result<serde_json::Value, String> {
        use tokio::io::AsyncWriteExt;
        let protected = std::env::join_paths(&self.protected).map_err(|e| e.to_string())?;
        let mut child = tokio::process::Command::new(&self.python)
            .arg("-c")
            .arg(POLICY_SCRIPT)
            .env("PYTHONPATH", &self.runtime_path)
            .env("OZGENT_PROTECTED", protected)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("could not start {} to prepare the sandbox: {e}", self.python))?;
        // On stdin for the same reason as the policy: it holds the env.
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(request.to_string().as_bytes()).await.map_err(|e| e.to_string())?;
        }
        let out = tokio::time::timeout(Duration::from_secs(30), child.wait_with_output())
            .await
            .map_err(|_| "preparing the sandbox took more than 30 seconds".to_string())?
            .map_err(|e| e.to_string())?;
        if !out.status.success() {
            let why = String::from_utf8_lossy(&out.stderr);
            let last = why.lines().last().unwrap_or("no reason given");
            return Err(format!("could not prepare the sandbox: {last}"));
        }
        serde_json::from_slice(&out.stdout).map_err(|e| format!("the sandbox policy did not parse: {e}"))
    }
}
