//! Command-line surface.
//!
//! Every generation setting appears twice: as a persistent flag here, and as a
//! field in `config.toml`. The flag is the highest-precedence layer, so a
//! one-off `--no-gpu` never has to be undone afterwards.

use clap::{Args, Parser, Subcommand};
use ozgent_core::accel::{CacheType, MoeOffload, Speculative};
use ozgent_core::{GpuLayers, Options, ThinkingMode};
use std::path::PathBuf;

/// Shown under `ozgent --help`.
///
/// The command list alone answers "what exists" but not "what do I type", and
/// the two that need an address — the web interface and the API — are useless
/// without one. Anyone reading `--help` is usually looking for exactly this.
const GETTING_STARTED: &str = "\
Getting started:
  ozgent pull unsloth/Qwen3.5-4B-GGUF:Q4_K_M   download a model
  ozgent list                                  see what is installed
  ozgent                                       chat with the last model used
  ozgent chat <model>                          chat with a specific one

Running as a server:
  ozgent web                                   web interface, opens a browser
                                               http://127.0.0.1:7333
  ozgent serve                                 OpenAI-compatible HTTP API
                                               http://127.0.0.1:7337/v1
  ozgent serve --host 0.0.0.0                  reachable from other machines
  ozgent serve --api-key SECRET                require a bearer token

  Point any OpenAI client at the API: set the base URL to
  http://127.0.0.1:7337/v1 and use any model name `ozgent list` shows.

When something is wrong:
  ozgent doctor                                hardware, backends, misconfiguration
  ozgent logs --follow                         watch what ozgent is doing
  ozgent logs --path                           where the log file lives
  ozgent -v ...                                more detail on the terminal

Full docs for the HTTP API are in docs/api.md.";

#[derive(Debug, Parser)]
#[command(
    name = "ozgent",
    about = "Run local models, with tools.",
    version,
    // Bare `ozgent` opens a chat, matching what users expect from the name
    // alone; every other behaviour is an explicit subcommand.
    args_conflicts_with_subcommands = true,
    subcommand_negates_reqs = true,
    after_help = GETTING_STARTED
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Model to chat with when no subcommand is given.
    pub model: Option<String>,

    #[command(flatten)]
    pub options: OptionFlags,

    /// Use a different ozgent directory.
    #[arg(long, global = true, value_name = "DIR")]
    pub root: Option<PathBuf>,

    /// Increase log detail. Repeat for more.
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Chat with a model interactively.
    Chat {
        model: Option<String>,
        #[command(flatten)]
        options: OptionFlags,
    },

    /// Run one prompt and exit. Reads stdin when no prompt is given.
    Run {
        model: String,
        /// The prompt. Omit to read from stdin.
        prompt: Vec<String>,
        #[command(flatten)]
        options: OptionFlags,
    },

    /// Serve the web interface.
    Web {
        #[arg(long, default_value_t = 7333)]
        port: u16,
        /// Address to bind. Defaults to loopback only.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Don't open a browser on start.
        #[arg(long)]
        no_open: bool,
        #[command(flatten)]
        options: OptionFlags,
    },

    /// Serve an OpenAI-compatible HTTP API.
    ///
    /// Point any OpenAI client at it: set the base URL and use any model name
    /// `ozgent list` shows. Tools, streaming and reasoning all work.
    Serve {
        #[arg(long, default_value_t = 7337)]
        port: u16,
        /// Address to bind. Loopback by default; use 0.0.0.0 to expose it.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Require this bearer token on every request.
        /// Falls back to `$OZGENT_API_KEY`.
        #[arg(long)]
        api_key: Option<String>,
        #[command(flatten)]
        options: OptionFlags,
    },

    /// Download a model from Hugging Face.
    ///
    /// Give a repository id and ozgent picks the right GGUF, finds the vision
    /// projector if there is one, and writes the manifest:
    ///
    ///   ozgent pull unsloth/gemma-3-12b-it-GGUF:Q4_K_M
    Pull {
        /// Repository, optionally `:QUANT`, e.g. `unsloth/gemma-3-12b-it-GGUF:Q4_K_M`.
        repo: String,

        /// Quantisation to fetch. Overrides any `:QUANT` in the repo argument.
        #[arg(long, short = 'q', value_name = "QUANT")]
        quant: Option<String>,

        /// Install under this name instead of one derived from the repository.
        #[arg(long = "as", value_name = "NAME:TAG")]
        as_ref: Option<String>,

        /// Short unique nickname, so `ozgent run <name>` works afterwards.
        #[arg(long, value_name = "ALIAS")]
        name: Option<String>,

        /// Git revision, branch, or tag.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Choose the largest quantisation that fits this many GB.
        #[arg(long, value_name = "GB")]
        max_size: Option<f64>,

        /// List what the repository offers and exit without downloading.
        #[arg(long)]
        list: bool,
    },

    /// Register a GGUF file already on disk.
    ///
    /// Hard-links where it can, so a 20 GB file is not duplicated and the
    /// original path keeps working.
    Import {
        #[arg(value_name = "FILE")]
        weights: PathBuf,

        /// Install under this name. Inferred from the filename when omitted.
        #[arg(long = "as", value_name = "NAME:TAG")]
        as_ref: Option<String>,

        /// Short unique nickname, so `ozgent run <name>` works afterwards.
        #[arg(long, value_name = "ALIAS")]
        name: Option<String>,

        /// Multimodal projector, for vision models.
        #[arg(long, value_name = "FILE")]
        mmproj: Option<PathBuf>,

        /// Copy instead of hard-linking. Needed across filesystems.
        #[arg(long)]
        copy: bool,
    },

    /// List installed models.
    #[command(alias = "ls")]
    List,

    /// Show a model's manifest and resolved settings.
    Show { model: String },

    /// Give a model a short unique nickname, or clear it.
    ///
    ///   ozgent alias Qwen3-Coder-30B:Q4_K_M coder
    ///   ozgent alias coder --clear
    Alias {
        /// The model, by alias or `name:tag`.
        model: String,
        /// The new nickname. Omit with --clear to remove it.
        alias: Option<String>,
        /// Remove the model's alias.
        #[arg(long, conflicts_with = "alias")]
        clear: bool,
    },

    /// Delete a model and everything in its directory.
    #[command(alias = "remove")]
    Rm {
        model: String,
        #[arg(short, long)]
        force: bool,
    },

    /// Inspect the Python tools.
    Tools {
        #[command(subcommand)]
        command: ToolsCommand,
    },

    /// Read or change configuration.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },

    /// Report hardware, backends, and what is misconfigured.
    Doctor,

    /// Show the log file every part of ozgent writes to.
    ///
    /// The same file whether ozgent was started as `chat`, `web`, `serve` or
    /// a one-shot command, which is what makes it useful when the server has
    /// been running under systemd and something went wrong hours ago.
    ///
    ///   ozgent logs --lines 200   the last 200 lines
    ///   ozgent logs --follow      keep printing as they arrive
    ///   ozgent logs --path        print the path and exit, e.g. for tail
    Logs {
        /// Lines to show from the end. (`-n` is taken by max_tokens.)
        #[arg(long, default_value_t = 50)]
        lines: usize,
        /// Keep printing new lines until interrupted.
        #[arg(short, long)]
        follow: bool,
        /// Print the path to the log file and exit.
        #[arg(long)]
        path: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum ToolsCommand {
    /// List discovered tools and their schemas.
    List {
        /// Print the full JSON Schema for each tool.
        #[arg(long)]
        schema: bool,
    },
    /// Invoke a tool directly, for debugging.
    Call {
        name: String,
        /// Arguments as a JSON object.
        #[arg(default_value = "{}")]
        arguments: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Print the config file path.
    Path,
    /// Print the effective configuration.
    Show {
        /// Resolve as it would apply to this model.
        model: Option<String>,
    },
    /// Write a default config file.
    Init {
        #[arg(long)]
        force: bool,
    },
}

/// Generation settings, shared by every command that runs a model.
#[derive(Debug, Default, Clone, Args)]
pub struct OptionFlags {
    /// Layers to offload to the GPU: a number, `auto`, or `off`.
    #[arg(long, value_name = "N", global = true)]
    pub gpu_layers: Option<GpuLayers>,

    /// Run entirely on the CPU. Shorthand for `--gpu-layers off`.
    #[arg(long, global = true, conflicts_with = "gpu_layers")]
    pub no_gpu: bool,

    /// Keep the routed experts of the first N layers in system RAM.
    #[arg(long, value_name = "N", global = true)]
    pub cpu_moe: Option<MoeOffload>,

    /// Steer generation with a control-vector GGUF.
    #[arg(long, value_name = "FILE", global = true)]
    pub control_vector: Option<std::path::PathBuf>,

    /// How hard to steer. 1.0 is the vector as trained; negative reverses it.
    #[arg(long, value_name = "F", global = true)]
    pub control_strength: Option<f32>,

    /// Physical micro-batch size, e.g. 256.
    #[arg(long, value_name = "N", global = true)]
    pub ubatch: Option<u32>,

    /// How hard a reasoning model should think: `low`, `medium`, or `high`.
    #[arg(long, value_name = "LEVEL", global = true)]
    pub effort: Option<ozgent_core::ReasoningEffort>,

    /// Context length in tokens.
    #[arg(long, short = 'c', value_name = "N", global = true)]
    pub ctx: Option<u32>,

    /// KV cache quantisation, e.g. `q8_0` or `f16`.
    #[arg(long, value_name = "TYPE", global = true)]
    pub cache_type: Option<CacheType>,

    /// Speculative decoding: `auto`, `ngram`, or `off`.
    #[arg(long, value_name = "MODE", global = true)]
    pub spec: Option<String>,

    /// Disable flash attention.
    #[arg(long, global = true)]
    pub no_flash_attn: bool,

    /// Sampling temperature.
    #[arg(long, short = 't', value_name = "T", global = true)]
    pub temperature: Option<f32>,

    #[arg(long, value_name = "P", global = true)]
    pub top_p: Option<f32>,

    #[arg(long, value_name = "K", global = true)]
    pub top_k: Option<u32>,

    /// Seed, for reproducible output.
    #[arg(long, value_name = "N", global = true)]
    pub seed: Option<u32>,

    /// Maximum tokens to generate. 0 means until the model stops.
    #[arg(long, short = 'n', value_name = "N", global = true)]
    pub max_tokens: Option<u32>,

    /// System prompt text.
    #[arg(long, short = 's', value_name = "TEXT", global = true)]
    pub system: Option<String>,

    /// Read the system prompt from a file.
    #[arg(long, value_name = "FILE", global = true, conflicts_with = "system")]
    pub system_file: Option<PathBuf>,

    /// Reasoning: `auto`, `on`, or `off`.
    #[arg(long, value_name = "MODE", global = true)]
    pub think: Option<ThinkingMode>,

    /// Suppress reasoning. Shorthand for `--think off`.
    #[arg(long, global = true, conflicts_with = "think")]
    pub no_think: bool,

    /// Disable tools for this run.
    #[arg(long, global = true)]
    pub no_tools: bool,

    /// Disable markdown rendering; emit plain text.
    #[arg(long, global = true)]
    pub plain: bool,

    /// Print timing and tokens/sec after each response.
    #[arg(long, global = true)]
    pub stats: bool,
}

impl OptionFlags {
    /// Convert flags into an option layer.
    ///
    /// Only flags the user actually passed become `Some`, so this layer
    /// overrides the config file exactly where it was asked to and nowhere
    /// else.
    pub fn to_options(&self) -> anyhow::Result<Options> {
        let system_prompt = match (&self.system, &self.system_file) {
            (Some(text), _) => Some(text.clone()),
            (None, Some(path)) => Some(std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("reading system prompt from {}: {e}", path.display())
            })?),
            _ => None,
        };

        Ok(Options {
            gpu_layers: if self.no_gpu { Some(GpuLayers::OFF) } else { self.gpu_layers },
            cpu_moe: self.cpu_moe,
            ubatch: self.ubatch,
            control_vector: self.control_vector.clone(),
            control_strength: self.control_strength,
            context_length: self.ctx,
            cache_type_k: self.cache_type,
            cache_type_v: self.cache_type,
            // A bare flag can only express one direction; absent means "defer".
            flash_attention: self.no_flash_attn.then_some(false),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            max_tokens: self.max_tokens,
            system_prompt,
            thinking: if self.no_think { Some(ThinkingMode::Off) } else { self.think },
            reasoning_effort: self.effort,
            tools: self.no_tools.then_some(false),
            speculative: match self.spec.as_deref() {
                None => None,
                Some("off" | "none") => Some(Speculative::Off),
                Some("ngram") => Some(Speculative::Ngram),
                Some("mtp") => Some(Speculative::Mtp),
                Some("auto") => Some(Speculative::Auto),
                // `draft:<model>` names a second, smaller model to propose
                // tokens. Spelled inside --spec rather than as its own flag so
                // the strategies stay mutually exclusive by construction.
                Some(other) => match other.strip_prefix("draft:") {
                    Some(model) if !model.is_empty() => Some(Speculative::Draft {
                        model: model.to_string(),
                        gpu_layers: Some(99),
                    }),
                    _ => {
                        return Err(anyhow::anyhow!(
                            "unknown --spec {other:?}; expected auto, ngram, mtp, off, \
                             or draft:<model>"
                        ));
                    }
                },
            },
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).unwrap_or_else(|e| panic!("failed to parse {args:?}: {e}"))
    }

    #[test]
    fn the_cli_definition_is_valid() {
        // Catches conflicting flags and duplicate short options at test time
        // rather than on the user's first run.
        Cli::command().debug_assert();
    }

    #[test]
    fn bare_invocation_starts_a_chat() {
        let cli = parse(&["ozgent"]);
        assert!(cli.command.is_none());
        assert!(cli.model.is_none());
    }

    #[test]
    fn a_bare_model_name_is_a_chat_target() {
        let cli = parse(&["ozgent", "gemma4:12b"]);
        assert_eq!(cli.model.as_deref(), Some("gemma4:12b"));
    }

    #[test]
    fn unset_flags_produce_an_empty_layer() {
        // Critical: an option layer that is not empty would silently override
        // the user's config file with clap's defaults.
        let opts = parse(&["ozgent"]).options.to_options().unwrap();
        assert!(opts.gpu_layers.is_none());
        assert!(opts.temperature.is_none());
        assert!(opts.thinking.is_none());
        assert!(opts.tools.is_none());
        assert!(opts.flash_attention.is_none());
        assert!(opts.system_prompt.is_none());
    }

    #[test]
    fn spec_modes_parse_and_reject_nonsense() {
        assert_eq!(
            parse(&["ozgent", "--spec", "off"]).options.to_options().unwrap().speculative,
            Some(Speculative::Off)
        );
        assert_eq!(
            parse(&["ozgent", "--spec", "ngram"]).options.to_options().unwrap().speculative,
            Some(Speculative::Ngram)
        );
        assert!(parse(&["ozgent", "--spec", "wishful"]).options.to_options().is_err());
    }

    #[test]
    fn no_gpu_maps_to_zero_offload() {
        let opts = parse(&["ozgent", "--no-gpu"]).options.to_options().unwrap();
        assert_eq!(opts.gpu_layers, Some(GpuLayers::OFF));
        assert!(opts.gpu_layers.unwrap().is_cpu_only());
    }

    #[test]
    fn gpu_layers_accepts_counts_and_keywords() {
        for (arg, expected) in [
            ("24", GpuLayers::Count(24)),
            ("auto", GpuLayers::AUTO),
            ("off", GpuLayers::OFF),
        ] {
            let opts = parse(&["ozgent", "--gpu-layers", arg]).options.to_options().unwrap();
            assert_eq!(opts.gpu_layers, Some(expected), "for --gpu-layers {arg}");
        }
    }

    #[test]
    fn no_gpu_and_gpu_layers_cannot_both_be_given() {
        assert!(Cli::try_parse_from(["ozgent", "--no-gpu", "--gpu-layers", "20"]).is_err());
    }

    #[test]
    fn no_think_maps_to_thinking_off() {
        let opts = parse(&["ozgent", "--no-think"]).options.to_options().unwrap();
        assert_eq!(opts.thinking, Some(ThinkingMode::Off));
    }

    #[test]
    fn think_accepts_explicit_modes() {
        for (arg, expected) in [
            ("on", ThinkingMode::On),
            ("off", ThinkingMode::Off),
            ("auto", ThinkingMode::Auto),
        ] {
            let opts = parse(&["ozgent", "--think", arg]).options.to_options().unwrap();
            assert_eq!(opts.thinking, Some(expected));
        }
    }

    #[test]
    fn cpu_moe_parses() {
        let opts = parse(&["ozgent", "--cpu-moe", "12"]).options.to_options().unwrap();
        assert_eq!(opts.cpu_moe, Some(MoeOffload::Layers(12)));
        let all = parse(&["ozgent", "--cpu-moe", "all"]).options.to_options().unwrap();
        assert_eq!(all.cpu_moe, Some(MoeOffload::ALL));
    }

    #[test]
    fn cache_type_sets_both_halves_of_the_kv_cache() {
        let opts = parse(&["ozgent", "--cache-type", "q4_0"]).options.to_options().unwrap();
        assert_eq!(opts.cache_type_k, Some(CacheType::Q4_0));
        assert_eq!(opts.cache_type_v, Some(CacheType::Q4_0));
    }

    #[test]
    fn a_cli_layer_overrides_the_config_layer() {
        let config = Options { temperature: Some(0.2), top_k: Some(10), ..Default::default() };
        let cli = parse(&["ozgent", "--temperature", "0.9"]).options.to_options().unwrap();
        let merged = config.merge(&cli);

        assert_eq!(merged.temperature, Some(0.9), "the flag must win");
        assert_eq!(merged.top_k, Some(10), "and must not disturb anything else");
    }

    #[test]
    fn system_prompt_can_come_from_a_file() {
        let path = std::env::temp_dir().join(format!("ozgent-sys-{}.txt", std::process::id()));
        std::fs::write(&path, "You are terse.").unwrap();

        let opts = parse(&["ozgent", "--system-file", path.to_str().unwrap()])
            .options
            .to_options()
            .unwrap();
        assert_eq!(opts.system_prompt.as_deref(), Some("You are terse."));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_system_prompt_file_reports_the_path() {
        let err = parse(&["ozgent", "--system-file", "/nonexistent/prompt.txt"])
            .options
            .to_options()
            .unwrap_err();
        assert!(err.to_string().contains("/nonexistent/prompt.txt"), "{err}");
    }

    #[test]
    fn subcommands_parse() {
        assert!(matches!(parse(&["ozgent", "list"]).command, Some(Command::List)));
        assert!(matches!(parse(&["ozgent", "pull", "gemma4:12b"]).command, Some(Command::Pull { .. })));
        assert!(matches!(parse(&["ozgent", "doctor"]).command, Some(Command::Doctor)));

        match parse(&["ozgent", "web", "--port", "9000"]).command {
            Some(Command::Web { port, host, .. }) => {
                assert_eq!(port, 9000);
                assert_eq!(host, "127.0.0.1", "must not bind publicly by default");
            }
            other => panic!("expected web, got {other:?}"),
        }
    }

    #[test]
    fn run_collects_a_multi_word_prompt() {
        match parse(&["ozgent", "run", "gemma4:12b", "explain", "MoE", "routing"]).command {
            Some(Command::Run { model, prompt, .. }) => {
                assert_eq!(model, "gemma4:12b");
                assert_eq!(prompt.join(" "), "explain MoE routing");
            }
            other => panic!("expected run, got {other:?}"),
        }
    }

    #[test]
    fn ls_is_an_alias_for_list() {
        assert!(matches!(parse(&["ozgent", "ls"]).command, Some(Command::List)));
    }

    #[test]
    fn flags_work_after_a_subcommand() {
        match parse(&["ozgent", "chat", "gemma4:12b", "--no-gpu", "--think", "off"]).command {
            Some(Command::Chat { model, options }) => {
                assert_eq!(model.as_deref(), Some("gemma4:12b"));
                let o = options.to_options().unwrap();
                assert_eq!(o.gpu_layers, Some(GpuLayers::OFF));
                assert_eq!(o.thinking, Some(ThinkingMode::Off));
            }
            other => panic!("expected chat, got {other:?}"),
        }
    }
}
