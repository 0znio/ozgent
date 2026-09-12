//! `~/ozgent/configs/config.toml`.
//!
//! Per-tool settings are kept as opaque TOML and handed to the Python worker
//! untouched. Rust deliberately knows nothing about what `web_search` expects,
//! so adding a tool never requires touching the Rust side.

use crate::options::Options;
use crate::paths::Paths;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Model used when the user runs `ozgent chat` with no model argument.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,

    /// Option layer applied to every model.
    pub defaults: Options,

    /// Per-model overrides, keyed by the `name:tag` display form.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, Options>,

    pub tools: ToolsConfig,

    /// Which tool calls run, which are asked about, and which are refused.
    ///
    /// Distinct from `[tools.config.permissions]`, which the Python side reads
    /// to bound what a tool may touch once it is running. This decides whether
    /// it runs at all.
    pub permissions: crate::permission::Permissions,

    pub ui: UiConfig,

    /// Messaging channels: Telegram, WhatsApp. Off, and admitting nobody,
    /// until deliberately configured — see [`crate::channels`].
    pub channels: crate::channels::ChannelsConfig,

    /// Tools offered by Model Context Protocol servers — see [`crate::mcp`].
    pub mcp: crate::mcp::McpConfig,

    pub embedding: EmbeddingConfig,

    /// The web interface.
    pub web: WebConfig,
}

/// `[web]`: settings for `ozgent web` and `ozgent daemon`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WebConfig {
    /// An Argon2id hash of the password for `/admin`, where the messaging
    /// gateway is controlled and models are downloaded and deleted. Set it
    /// with `ozgent admin setup`; the password itself is never stored.
    ///
    /// Never set from a browser: a page that could set its own password could
    /// be claimed by whoever reached it first. Unset, `/admin` is closed and
    /// says how to open it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admin_password_hash: Option<String>,

    /// Drop the loaded model after this many idle minutes. `0` never does.
    ///
    /// A server holds its model so the next question is instant, which is
    /// right while someone is using it and wrong for the nineteen hours a day
    /// they are not — a model kept for one 9:20 brief holds several gigabytes
    /// of VRAM until midnight. The cost of getting it wrong is one reload,
    /// which is the same wait the first question of the day pays anyway.
    #[serde(default = "default_idle_unload")]
    pub idle_unload_minutes: u64,
}

fn default_idle_unload() -> u64 {
    15
}

impl Default for WebConfig {
    fn default() -> Self {
        Self { admin_password_hash: None, idle_unload_minutes: default_idle_unload() }
    }
}

impl WebConfig {
    /// The stored admin password hash, if one is set and is not blank.
    pub fn admin_hash(&self) -> Option<&str> {
        self.admin_password_hash.as_deref().map(str::trim).filter(|p| !p.is_empty())
    }

    /// How long to hold an idle model, or `None` to hold it indefinitely.
    pub fn idle_unload(&self) -> Option<std::time::Duration> {
        (self.idle_unload_minutes > 0)
            .then(|| std::time::Duration::from_secs(self.idle_unload_minutes * 60))
    }
}

/// The model used for embeddings.
///
/// Separate from the chat model on purpose: pooling a chat model's hidden
/// states produces vectors that look plausible and cluster badly, so ozgent
/// would rather have none than pretend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// An installed model, as `name:tag` or an alias. `None` disables
    /// embeddings and the memory layer falls back to lexical matching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    pub enabled: bool,

    /// Interpreter used to launch the worker. A virtualenv path works here.
    pub python: String,

    /// Extra directories scanned for user tools, on top of `~/ozgent/tools`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extra_paths: Vec<PathBuf>,

    /// Per-call wall-clock budget. A tool that exceeds it is cancelled and the
    /// model is told so, rather than the session hanging.
    pub timeout_seconds: u64,

    /// Tool calls allowed within a single assistant turn, bounding runaway
    /// tool loops.
    pub max_calls_per_turn: u32,

    /// Names to refuse to load even if present on disk.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub disabled: Vec<String>,

    /// Let the model hand a request to an `@agent` by itself, through the
    /// `ask_agent` tool, when the request is squarely that agent's job.
    pub handoff: bool,

    /// Opaque per-tool settings, keyed by tool name. Passed through verbatim.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub config: BTreeMap<String, toml::Value>,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            python: "python3".into(),
            extra_paths: Vec::new(),
            timeout_seconds: 30,
            max_calls_per_turn: 8,
            disabled: Vec::new(),
            handoff: true,
            config: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiConfig {
    /// Render markdown to ANSI. Off gives raw text, which is what you want
    /// when piping to a file.
    pub markdown: bool,
    /// Syntax-highlight fenced code blocks.
    pub highlight_code: bool,
    /// Show reasoning traces when thinking is enabled.
    pub show_thinking: bool,
    /// Print tokens/sec and timing after each response.
    pub show_stats: bool,

    /// Tell the model today's date in the system prompt.
    ///
    /// Without it, "latest" and "tomorrow" resolve against the model's
    /// training data — it guesses a year, and guesses wrong. Turn off for
    /// reproducible prompts.
    pub date_awareness: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            markdown: true,
            highlight_code: true,
            show_thinking: true,
            show_stats: false,
            date_awareness: true,
        }
    }
}

impl Config {
    /// The model to answer with when nothing names one.
    ///
    /// `[channels] model` first, then `default_model`. The order is not
    /// arbitrary: somebody who set a model for their phone chose it for
    /// answering *unattended*, which is exactly what a scheduled job is.
    ///
    /// Shared because it was not. The channels had this rule and the
    /// scheduler had its own, so a person with `[channels] model` set and no
    /// `default_model` — a perfectly ordinary setup — got working Telegram
    /// replies and every scheduled job failing with "no model is configured".
    pub fn answering_model(&self) -> Option<String> {
        self.channels.model.clone().or_else(|| self.default_model.clone())
    }

    /// Load `config.toml`, treating a missing file as an empty config so a
    /// fresh install works with no setup.
    pub fn load(paths: &Paths) -> Result<Self, ConfigError> {
        Self::load_from(&paths.config_file())
    }

    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(ConfigError::Io { path: path.to_path_buf(), source: e }),
        };
        toml::from_str(&text)
            .map_err(|e| ConfigError::Parse { path: path.to_path_buf(), source: Box::new(e) })
    }

    pub fn save(&self, paths: &Paths) -> Result<(), ConfigError> {
        let path = paths.config_file();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ConfigError::Io { path: parent.to_path_buf(), source: e })?;
        }
        let text = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::Serialize { source: Box::new(e) })?;
        // Written beside and renamed over, so a program reading it at the
        // same moment — a running server following changes — sees the old
        // file or the new one, never half of one.
        // Through a symlink to the real file, or the rename would replace a
        // dotfiles link with a plain file.
        let path = std::fs::canonicalize(&path).unwrap_or(path);
        let partial = path.with_extension("toml.partial");
        std::fs::write(&partial, text).map_err(|e| ConfigError::Io { path: partial.clone(), source: e })?;
        // Readable by its owner only: it can hold a bot token and the admin
        // password hash, and a home directory is not always private.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o600));
        }
        std::fs::rename(&partial, &path).map_err(|e| ConfigError::Io { path, source: e })?;
        Ok(())
    }

    /// The config-file half of the options stack: global defaults with any
    /// per-model block layered on top.
    pub fn options_for(&self, model: &str) -> Options {
        match self.models.get(model) {
            Some(per_model) => self.defaults.clone().merge(per_model),
            None => self.defaults.clone(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Parse { path: PathBuf, source: Box<toml::de::Error> },
    #[error("serialising config: {source}")]
    Serialize { source: Box<toml::ser::Error> },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::GpuLayers;

    #[test]
    fn missing_file_is_an_empty_config() {
        let c = Config::load_from(Path::new("/nonexistent/ozgent/config.toml")).unwrap();
        assert!(c.default_model.is_none());
        assert!(c.tools.enabled);
    }

    #[test]
    fn per_model_block_overrides_defaults() {
        let c: Config = toml::from_str(
            r#"
            default_model = "gemma4:12b"
            [defaults]
            gpu_layers = "auto"
            temperature = 0.7
            [models."gemma4:12b"]
            gpu_layers = 30
            "#,
        )
        .unwrap();

        let o = c.options_for("gemma4:12b");
        assert_eq!(o.gpu_layers, Some(GpuLayers::Count(30)));
        assert_eq!(o.temperature, Some(0.7), "unmentioned key keeps the default");

        let other = c.options_for("qwen4:8b");
        assert_eq!(other.gpu_layers, Some(GpuLayers::AUTO));
    }

    #[test]
    fn tool_config_is_opaque_passthrough() {
        let c: Config = toml::from_str(
            r#"
            [tools.config.web_search]
            provider = "brave"
            max_results = 5
            "#,
        )
        .unwrap();
        let ws = c.tools.config.get("web_search").unwrap();
        assert_eq!(ws.get("provider").unwrap().as_str(), Some("brave"));
    }

    #[test]
    fn date_awareness_is_on_by_default() {
        // A model with no clock guesses the year; the default should protect
        // against that rather than require opting in.
        assert!(UiConfig::default().date_awareness);
    }

    #[test]
    fn permissions_default_to_asking_before_anything_acts() {
        // A fresh install must not need a config file to be safe.
        let c = Config::default();
        let g = crate::permission::Grants::default();
        use crate::permission::{Effect, Verdict};
        assert_eq!(c.permissions.verdict("run_command", Effect::Execute, &g), Verdict::Ask);
        assert_eq!(
            c.permissions.verdict("web_search", Effect::Read, &g),
            Verdict::Allow { by_user: false },
        );
    }

    #[test]
    fn the_permissions_section_is_read_from_the_config_file() {
        let c: Config = toml::from_str(
            r#"
            [permissions]
            execute = "allow"
            [permissions.tools]
            write_file = "deny"
            "#,
        )
        .unwrap();
        use crate::permission::{Effect, Rule};
        assert_eq!(c.permissions.rule_for("run_command", Effect::Execute), Rule::Allow);
        assert_eq!(c.permissions.rule_for("write_file", Effect::Write), Rule::Deny);
    }

    #[test]
    fn round_trips() {
        let mut c = Config::default();
        c.default_model = Some("gemma4:12b".into());
        c.defaults.gpu_layers = Some(GpuLayers::OFF);
        let text = toml::to_string_pretty(&c).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back.defaults.gpu_layers, Some(GpuLayers::OFF));
    }
}

#[cfg(test)]
mod answering_model_tests {
    use super::*;

    #[test]
    fn a_channel_model_answers_when_nothing_else_is_set() {
        // The reported failure: `[channels] model` set, no `default_model`.
        // Telegram replied perfectly well and every scheduled job died with
        // "no model is configured to answer with", because the two surfaces
        // each had their own idea of what to fall back to.
        let mut c = Config::default();
        c.channels.model = Some("Qwen3.5-4B:Q4_K_M".into());
        assert_eq!(c.answering_model().as_deref(), Some("Qwen3.5-4B:Q4_K_M"));
    }

    #[test]
    fn a_channel_model_wins_over_the_default() {
        // Somebody who chose a model for their phone chose it for answering
        // unattended, which is what a scheduled job is.
        let mut c = Config::default();
        c.default_model = Some("big:Q8".into());
        c.channels.model = Some("small:Q4".into());
        assert_eq!(c.answering_model().as_deref(), Some("small:Q4"));
    }

    #[test]
    fn the_default_answers_when_no_channel_model_is_set() {
        let mut c = Config::default();
        c.default_model = Some("big:Q8".into());
        assert_eq!(c.answering_model().as_deref(), Some("big:Q8"));
    }

    #[test]
    fn neither_set_is_none_rather_than_a_guess() {
        assert_eq!(Config::default().answering_model(), None);
    }
}
