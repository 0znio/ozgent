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

    pub embedding: EmbeddingConfig,
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
        std::fs::write(&path, text).map_err(|e| ConfigError::Io { path, source: e })
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
