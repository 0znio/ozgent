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
  ozgent default <model>                       start there when none is named

Running as a server:
  ozgent web                                   web interface, opens a browser
                                               http://localhost:7333, and
                                               reachable from your network
  ozgent web --host 127.0.0.1                  this machine only
  ozgent serve                                 OpenAI-compatible HTTP API
                                               http://127.0.0.1:7337/v1
  ozgent serve --host 0.0.0.0                  reachable from other machines
  ozgent serve --api-key SECRET                require a bearer token

  Point any OpenAI client at the API: set the base URL to
  http://127.0.0.1:7337/v1 and use any model name `ozgent list` shows.

Tuning a run (all work on any command, and with `ozgent chat`):
  ozgent chat coder --ctx 32k                  context length, or 8192, or 128k
  ozgent chat coder --ctx 128k --inference-mode gpu_ram
                                               the whole window, cache in RAM,
                                               slower per token
  ozgent chat coder --temp 0.2 --top-p 0.9     sampling
  ozgent chat coder --top-k 40 --min-p 0.05
  ozgent chat coder --think off --effort high  reasoning
  ozgent chat coder -c 8k -t 0.7 -n 2k         short forms

  Inside a chat, /config shows the same settings and changes them for good:
  `/config ctx 32k`, `/config temp 0.2`, `/config top_p 0.9`.

When something is wrong:
  ozgent doctor                                hardware, backends, misconfiguration
  ozgent logs --follow                         watch what ozgent is doing
  ozgent logs --path                           where the log file lives
  ozgent -v ...                                more detail on the terminal

Full docs: docs/settings.md (every setting), docs/api.md (HTTP API),
           docs/tools.md (tools and permissions).";

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

    /// Serve the web interface, reachable from your local network.
    ///
    /// Binds every interface so a phone or another machine on the same network
    /// can use it. There is no password: anyone who can reach the port can
    /// chat, read past conversations and change settings. Use
    /// `--host 127.0.0.1` on a network you do not trust.
    Web {
        #[arg(long, default_value_t = 7333)]
        port: u16,
        /// Address to bind. Every interface by default; 127.0.0.1 for this
        /// machine only.
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Don't open a browser on start.
        #[arg(long)]
        no_open: bool,
        #[command(flatten)]
        options: OptionFlags,
    },

    /// Telegram and WhatsApp: set up, manage, and answer messages.
    ///
    ///   ozgent gateway telegram           set up Telegram, or change it
    ///   ozgent gateway whatsapp           link WhatsApp, or change it
    ///   ozgent gateway status             what is set up and who is allowed
    ///   ozgent gateway                    answer messages from this terminal
    ///
    /// `ozgent web` answers them too, and its /admin page does everything
    /// these commands do. Nobody can message ozgent until you allow them.
    /// See docs/channels.md before opening one up.
    #[command(alias = "channel", alias = "channels")]
    Gateway {
        #[command(subcommand)]
        command: Option<GatewayCommand>,
        /// Also serve the web interface, sharing one loaded model.
        #[arg(long)]
        web: bool,
        /// Port for `--web`.
        #[arg(long, default_value_t = 7333)]
        port: u16,
        /// Address to bind for `--web`.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[command(flatten)]
        options: OptionFlags,
    },

    /// Run ozgent in the background, so everything else can just connect.
    ///
    ///   ozgent daemon install     run it now, and at every login
    ///   ozgent daemon status      is it running, and what is it doing
    ///   ozgent daemon uninstall   stop it and remove the service
    ///   ozgent daemon             run it in this terminal instead
    ///
    /// One process owns the model, the database, the messaging channels and
    /// the scheduler, and serves the web interface and the API over them. A
    /// scheduled job runs whether or not anything is open, and nothing loads a
    /// second copy of the model. With no question in flight it holds no model
    /// at all.
    #[command(alias = "service")]
    Daemon {
        #[command(subcommand)]
        command: Option<DaemonCommand>,
        /// Address to bind. This machine only by default; 0.0.0.0 to let
        /// phones and other computers on the same network reach it.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 7333)]
        port: u16,
        #[command(flatten)]
        options: OptionFlags,
    },

    /// Things ozgent does on a timer, and where the answers go.
    ///
    ///   ozgent scheduler                      what is scheduled
    ///   ozgent scheduler add                  set one up, question by question
    ///   ozgent scheduler show <job>           its settings and recent runs
    ///   ozgent scheduler run <job>            run it now
    ///
    /// Jobs run inside `ozgent daemon` (or `ozgent web`). Ask for one in a
    /// chat — "every weekday at 9:20, send me a pre-market brief" — and it
    /// appears here. The /scheduler page does everything these commands do.
    #[command(alias = "schedule", alias = "jobs")]
    Scheduler {
        #[command(subcommand)]
        command: Option<SchedulerCommand>,
    },

    /// The password for the web interface's /admin page.
    ///
    ///   ozgent admin setup      choose the password
    ///   ozgent admin reset      forgot it? choose a new one here
    ///   ozgent admin status     is it set, and where to open it
    ///   ozgent admin disable    close /admin again
    ///
    /// Only a hash of the password is stored (Argon2id). Whoever can run
    /// commands as you on this machine can reset it; that is the way back in.
    Admin {
        #[command(subcommand)]
        command: AdminCommand,
    },

    /// List, show, create and edit agents.
    ///
    /// An agent is a named job with its own instructions and its own tools.
    /// Write `@name` in a message — terminal, browser, API or chat app — to
    /// hand that message to it.
    Agent {
        #[command(subcommand)]
        command: AgentCommand,
    },

    /// Serve an OpenAI- and Anthropic-compatible HTTP API.
    ///
    /// Point any OpenAI or Anthropic client at it: set the base URL to
    /// http://host:port/v1 and use any model name `ozgent list` shows, or an
    /// agent as `@name`. Tools, streaming and reasoning all work.
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

    /// Show or set the model used when no model is named.
    ///
    /// `ozgent chat` and the web interface both start with it, so the model
    /// you use most does not have to be typed every time.
    ///
    ///   ozgent default            what it is now
    ///   ozgent default coder      use `coder` from now on
    ///   ozgent default --clear    go back to naming one each time
    #[command(name = "default")]
    DefaultModel {
        /// The model, by alias or `name:tag`. Omit to show the current one.
        model: Option<String>,
        /// Forget the default and require a model to be named.
        #[arg(long, conflicts_with = "model")]
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

    /// Show the Model Context Protocol servers and what they offer.
    ///
    /// Connects to each one exactly as a chat would, so what it prints is what
    /// the model will actually be given — including the servers that failed,
    /// and why.
    Mcp,

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
pub enum AgentCommand {
    /// Every agent, built in and yours.
    #[command(alias = "ls")]
    List,
    /// One agent in full: its tools, rules and instructions.
    Show { name: String },
    /// Create an agent from a template and open it in $EDITOR.
    ///
    ///   ozgent agent new news-digest
    ///   ozgent agent new my-guru --from stock-guru
    New {
        name: String,
        /// Start from a copy of this agent instead of the template.
        #[arg(long)]
        from: Option<String>,
    },
    /// Open an agent in $EDITOR. A built-in one is copied to your agents
    /// folder first, and the copy is what you edit.
    Edit { name: String },
    /// Delete one of your agents. Deleting an edited built-in restores it.
    #[command(alias = "remove", alias = "delete")]
    Rm { name: String },
}

#[derive(Debug, Subcommand)]
pub enum GatewayCommand {
    /// What is set up, what is running, and who is allowed.
    #[command(alias = "list")]
    Status,
    /// Set up Telegram, or change it once it is.
    Telegram {
        #[command(subcommand)]
        action: Option<ChannelAction>,
    },
    /// Link WhatsApp, or change it once it is.
    #[command(alias = "wa")]
    Whatsapp {
        #[command(subcommand)]
        action: Option<ChannelAction>,
    },
}

/// Something to change on one channel. With none, a menu asks.
#[derive(Debug, Subcommand)]
pub enum ChannelAction {
    /// Walk through setting it up from the start.
    Setup,
    /// Who may message it.
    #[command(alias = "list")]
    Allowed,
    /// Let someone message it: a phone number with its country code, a
    /// Telegram user id, or a @username.
    Allow { who: Vec<String> },
    /// Stop someone from messaging it.
    #[command(alias = "remove")]
    Deny { who: Vec<String> },
    /// Which tools it may use: `all`, `none`, or names separated by commas.
    Tools { tools: Option<String> },
    /// Telegram: set a new bot token. Asked for when not given, so it stays out
    /// of your shell history.
    Token { token: Option<String> },
    /// WhatsApp: link this machine to an account (or to a different one).
    Link,
    /// Sign out: forget the bot token, or unlink the WhatsApp device.
    #[command(alias = "logout")]
    Signout,
    /// Answer messages on this channel.
    On,
    /// Stop answering on this channel, keeping its settings.
    Off,
}

#[derive(Debug, Subcommand)]
pub enum DaemonCommand {
    /// Install the service for whatever init system this machine runs, then
    /// start it.
    ///
    /// The address is baked into the service file, so it is asked for here
    /// rather than inherited — a service that only works when you remember to
    /// pass a flag is not a service.
    Install {
        /// Address to bind. This machine only by default; 0.0.0.0 to let
        /// phones and other computers on the same network reach it.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 7333)]
        port: u16,
    },
    /// Whether it is running, and what it is doing.
    Status,
    /// Stop it and remove the service. Your data is untouched.
    #[command(alias = "remove")]
    Uninstall,
}

#[derive(Debug, Subcommand)]
pub enum SchedulerCommand {
    /// Everything scheduled, and when each next runs.
    List,
    /// Set up a job, question by question.
    #[command(alias = "new")]
    Add {
        /// Skip the questions and give everything at once.
        #[arg(long)]
        name: Option<String>,
        /// What to ask each time it runs.
        #[arg(long)]
        prompt: Option<String>,
        /// "every weekday at 9:20", "every 2 hours", or cron "20 9 * * 1-5".
        #[arg(long)]
        when: Option<String>,
        /// An agent to ask, without the @.
        #[arg(long)]
        agent: Option<String>,
        /// Only send the answer when this is true.
        #[arg(long)]
        only_if: Option<String>,
        /// Where the answer goes: telegram, whatsapp, or none.
        #[arg(long)]
        deliver: Option<String>,
        /// One chat to send it to. Left out, it goes to everyone the channel
        /// allows — which is usually what you want.
        #[arg(long)]
        to: Option<String>,
        /// An IANA zone like Asia/Kolkata. Local time by default.
        #[arg(long)]
        timezone: Option<String>,
    },
    /// A job's settings and how its recent runs went.
    Show { job: String },
    /// Change one thing about a job. Same flags as `add`.
    #[command(alias = "edit")]
    Set {
        job: String,
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long)]
        when: Option<String>,
        #[arg(long)]
        agent: Option<String>,
        #[arg(long)]
        only_if: Option<String>,
        #[arg(long)]
        deliver: Option<String>,
        #[arg(long)]
        to: Option<String>,
        #[arg(long)]
        timezone: Option<String>,
    },
    /// Stop a job running, keeping its settings.
    #[command(alias = "disable")]
    Pause { job: String },
    /// Start it again.
    #[command(alias = "enable")]
    Resume { job: String },
    /// Run it now, without waiting for its next time.
    Run { job: String },
    /// Delete it, and its history.
    #[command(alias = "delete")]
    Rm {
        job: String,
        /// Don't ask first.
        #[arg(short, long)]
        force: bool,
    },
    /// Read a time back and say when it would actually fire.
    ///
    /// `ozgent scheduler when "every weekday at 9:20"`
    When {
        when: Vec<String>,
        #[arg(long)]
        timezone: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
pub enum AdminCommand {
    /// Choose the admin password.
    Setup,
    /// Choose a new password when the old one is forgotten. Signs every
    /// browser out and lifts any lockout.
    Reset,
    /// Whether a password is set, and where the page is.
    Status,
    /// Remove the password, which closes /admin.
    Disable,
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

    /// Where the model runs: `gpu`, `gpu_ram`, or `ram`.
    ///
    /// `gpu` puts everything on the card and trims the context window to the
    /// VRAM left over. `gpu_ram` keeps the weights on the card and moves the
    /// KV cache into system RAM, so the window is bounded by RAM instead — a
    /// much larger window, several times slower per token at that size. `ram`
    /// uses no GPU at all.
    #[arg(long, value_name = "MODE", global = true)]
    pub inference_mode: Option<ozgent_core::InferenceMode>,

    /// Keep the KV cache in system RAM instead of on the GPU.
    ///
    /// The low-level half of `--inference-mode gpu_ram`, for combining with an
    /// explicit `--gpu-layers`.
    #[arg(long, global = true)]
    pub no_kv_offload: bool,

    /// How hard a reasoning model should think: `low`, `medium`, or `high`.
    #[arg(long, value_name = "LEVEL", global = true)]
    pub effort: Option<ozgent_core::ReasoningEffort>,

    /// Context length in tokens. Accepts `8192`, `8k`, or `128k`.
    #[arg(
        long,
        short = 'c',
        visible_alias = "context",
        value_name = "N",
        global = true,
        value_parser = ozgent_core::parse_count
    )]
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

    /// Sampling temperature. Lower is more focused, higher more varied.
    #[arg(long, short = 't', visible_alias = "temp", value_name = "T", global = true)]
    pub temperature: Option<f32>,

    /// Nucleus sampling: consider tokens up to this cumulative probability.
    #[arg(long, value_name = "P", global = true)]
    pub top_p: Option<f32>,

    /// Consider only the K most likely tokens. 0 disables the cutoff.
    #[arg(long, value_name = "K", global = true)]
    pub top_k: Option<u32>,

    /// Penalise tokens by how often they have already appeared.
    #[arg(long, value_name = "P", global = true)]
    pub repeat_penalty: Option<f32>,

    /// Minimum probability, relative to the most likely token.
    #[arg(long, value_name = "P", global = true)]
    pub min_p: Option<f32>,

    /// Seed, for reproducible output.
    #[arg(long, value_name = "N", global = true)]
    pub seed: Option<u32>,

    /// Maximum tokens to generate. Accepts `2k`; 0 means until the model stops.
    #[arg(
        long,
        short = 'n',
        value_name = "N",
        global = true,
        value_parser = ozgent_core::parse_count
    )]
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
            inference_mode: self.inference_mode,
            kv_offload: self.no_kv_offload.then_some(false),
            cache_type_k: self.cache_type,
            cache_type_v: self.cache_type,
            // A bare flag can only express one direction; absent means "defer".
            flash_attention: self.no_flash_attn.then_some(false),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
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
    fn the_inference_mode_flag_reaches_the_options_layer() {
        let hybrid = Cli::parse_from(["ozgent", "chat", "m", "--inference-mode", "gpu_ram"])
            .options
            .to_options()
            .unwrap()
            .resolve();
        assert!(!hybrid.kv_offload, "the cache moves to ram");
        assert_eq!(hybrid.gpu_layers, ozgent_core::GpuLayers::AUTO, "the weights do not");

        let ram = Cli::parse_from(["ozgent", "chat", "m", "--inference-mode", "cpu"])
            .options
            .to_options()
            .unwrap()
            .resolve();
        assert!(ram.gpu_layers.is_cpu_only());
    }

    #[test]
    fn a_nonsense_mode_is_refused_at_the_command_line() {
        assert!(Cli::try_parse_from(["ozgent", "chat", "m", "--inference-mode", "quantum"]).is_err());
    }

    #[test]
    fn the_kv_offload_flag_reaches_the_options_layer() {
        // The merge list drops any field it does not name, and the failure is
        // silent: the flag parses, the config accepts it, and llama.cpp never
        // hears about it. See docs/settings.md and the merge trap it warns of.
        let on = Cli::parse_from(["ozgent", "chat", "m"]).options.to_options().unwrap();
        assert_eq!(on.kv_offload, None, "unset must not override a config file");

        let off = Cli::parse_from(["ozgent", "chat", "m", "--no-kv-offload"])
            .options
            .to_options()
            .unwrap();
        assert_eq!(off.kv_offload, Some(false));
        assert!(!off.resolve().kv_offload, "and it must survive resolve()");
        assert!(
            ozgent_core::Options::default().resolve().kv_offload,
            "the default keeps the cache on the gpu",
        );
    }

    #[test]
    fn context_length_accepts_human_sizes() {
        for (arg, expected) in [("8192", 8192), ("8k", 8192), ("128k", 131_072)] {
            let opts = parse(&["ozgent", "--ctx", arg]).options.to_options().unwrap();
            assert_eq!(opts.context_length, Some(expected), "for --ctx {arg}");
        }
    }

    #[test]
    fn max_tokens_accepts_human_sizes_too() {
        // Both counts are token counts; accepting `k` on one and not the
        // other is the kind of inconsistency users trip over once each.
        let opts = parse(&["ozgent", "--max-tokens", "2k"]).options.to_options().unwrap();
        assert_eq!(opts.max_tokens, Some(2048));
    }

    #[test]
    fn a_bad_size_is_refused_at_parse_time() {
        assert!(Cli::try_parse_from(["ozgent", "--ctx", "enormous"]).is_err());
    }

    #[test]
    fn temp_is_accepted_as_well_as_temperature() {
        let short = parse(&["ozgent", "--temp", "0.2"]).options.to_options().unwrap();
        let long = parse(&["ozgent", "--temperature", "0.2"]).options.to_options().unwrap();
        assert_eq!(short.temperature, Some(0.2));
        assert_eq!(short.temperature, long.temperature);
    }

    #[test]
    fn context_is_accepted_as_well_as_ctx() {
        let a = parse(&["ozgent", "--context", "8k"]).options.to_options().unwrap();
        let b = parse(&["ozgent", "-c", "8k"]).options.to_options().unwrap();
        assert_eq!(a.context_length, Some(8192));
        assert_eq!(a.context_length, b.context_length);
    }

    #[test]
    fn every_sampling_knob_reaches_the_options_layer() {
        let opts = parse(&[
            "ozgent", "--temp", "0.3", "--top-p", "0.85", "--top-k", "40", "--min-p", "0.02",
            "--repeat-penalty", "1.15",
        ])
        .options
        .to_options()
        .unwrap();
        assert_eq!(opts.temperature, Some(0.3));
        assert_eq!(opts.top_p, Some(0.85));
        assert_eq!(opts.top_k, Some(40));
        assert_eq!(opts.min_p, Some(0.02));
        assert_eq!(opts.repeat_penalty, Some(1.15));
    }

    #[test]
    fn the_new_flags_stay_absent_when_unused() {
        // Same trap as `unset_flags_produce_an_empty_layer`: a default here
        // would overwrite the config file for every user who never asked.
        let opts = parse(&["ozgent"]).options.to_options().unwrap();
        assert!(opts.min_p.is_none());
        assert!(opts.repeat_penalty.is_none());
        assert!(opts.context_length.is_none());
        assert!(opts.max_tokens.is_none());
    }

    #[test]
    fn help_shows_how_to_tune_a_run() {
        // The flags existed before and were undiscoverable; that is what this
        // guards, not their implementation.
        for text in ["--ctx", "--temp", "--top-p", "--top-k", "/config"] {
            assert!(GETTING_STARTED.contains(text), "{text} is missing from --help");
        }
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
                // Deliberate: the web interface is meant to be reachable from
                // a phone on the same network. The API below is not.
                assert_eq!(host, "0.0.0.0");
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
    fn the_api_still_binds_to_loopback_only() {
        // `web` is opened to the network on purpose; `serve` is an API that
        // may carry a key and must stay opt-in.
        match parse(&["ozgent", "serve"]).command {
            Some(Command::Serve { host, .. }) => assert_eq!(host, "127.0.0.1"),
            other => panic!("expected serve, got {other:?}"),
        }
    }

    #[test]
    fn the_web_interface_can_be_restricted_again() {
        match parse(&["ozgent", "web", "--host", "127.0.0.1"]).command {
            Some(Command::Web { host, .. }) => assert_eq!(host, "127.0.0.1"),
            other => panic!("expected web, got {other:?}"),
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
