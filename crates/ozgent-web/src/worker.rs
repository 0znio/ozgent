//! The inference thread.
//!
//! Everything that touches llama.cpp happens here, on one dedicated OS thread,
//! for three reasons that together rule out doing it in the request handlers:
//!
//! * `Session` borrows from `Engine`, so the pair is self-referential and
//!   cannot simply be stored in shared state.
//! * A single GPU context cannot serve two generations at once, so requests
//!   have to be serialised regardless.
//! * Keeping one session alive across turns is what makes prefix KV reuse
//!   work; building a session per request would throw the cache away every
//!   time.
//!
//! Handlers send a [`Job`] and stream [`Event`]s back, so the async side never
//! blocks on the GPU.

use ozgent_core::{Config, Message, Paths, ThinkingMode};
use ozgent_tools::Toolbox;
use std::sync::Arc;
use ozgent_llama::engine::{Engine, StopReason};
use ozgent_llama::thinking::{Chunk, ThinkingFilter};
use std::sync::mpsc::{Receiver, channel};
use tokio::sync::mpsc::UnboundedSender;

/// A unit of work for the inference thread.
pub enum Job {
    Generate(Box<Request>),
    /// Embed texts and send the vectors straight back.
    Embed {
        texts: Vec<String>,
        role: ozgent_llama::embed::Role,
        /// Refuse a text longer than the embedder takes, rather than cut it.
        strict: bool,
        reply: std::sync::mpsc::Sender<Result<Vec<Vec<f32>>, String>>,
    },
    /// Drop the loaded model and free its VRAM.
    Unload,
}

pub struct Request {
    /// Alias or `name:tag`.
    pub model: String,
    pub messages: Vec<Message>,
    pub thinking: Option<ThinkingMode>,
    pub max_tokens: Option<u32>,
    /// Whether this turn may call tools. The user can switch it off per turn.
    pub tools_enabled: bool,
    /// Which built-in tools this turn may use, by name. `None` offers all of
    /// them; `Some(list)` offers only those named, so a caller can withhold a
    /// tool rather than trusting the model not to reach for it.
    pub native_tools: Option<Vec<String>>,
    /// Tools the caller implements. Described to the model like any other, but
    /// never executed here — when one is called the turn ends and the call is
    /// handed back, which is what an OpenAI client expects.
    pub client_tools: Vec<ozgent_core::ToolSpec>,
    /// A GBNF grammar constraining the whole reply, from `response_format`.
    /// Unlike the tool gate this applies from the first token, so the model
    /// cannot produce anything the schema forbids.
    pub response_grammar: Option<String>,
    /// Sampling overrides for this request only, as the API allows. Applied on
    /// top of the server's configuration rather than replacing it.
    pub overrides: Option<ozgent_core::Options>,
    /// Images for this turn. Resolved on the inference thread, where the
    /// projector lives.
    pub images: Vec<ozgent_core::ImageSource>,
    /// Whether there is a person on the other end who can answer a permission
    /// question. True for the web interface, false for the OpenAI API, where
    /// the caller is a program.
    pub can_ask: bool,
    /// What an API caller's key allows, or `None` for this machine's owner,
    /// whom the tool policy alone governs. A key caps which effects may run at
    /// all and stands in for the question the policy would have asked about
    /// the effects it grants. See [`ozgent_core::access::Grant`].
    pub grant: Option<ozgent_core::access::Grant>,
    /// Agents the message called by name, in order. Empty for an ordinary
    /// turn. When set, each runs in turn with its own instructions, tools and
    /// rules, and their reports are the reply.
    pub agents: Vec<ozgent_core::Agent>,
    /// Tools the person switched off for the main model, by name. A
    /// preference rather than a limit: unlike `native_tools` it does not
    /// narrow an agent called by name, whose tools are part of the choice to
    /// call it.
    pub tools_off: Vec<String>,
    /// Agents the model may hand the request to itself, through the
    /// `ask_agent` tool. Empty when that is switched off or not offered here.
    pub handoff: Vec<ozgent_core::Agent>,
    pub out: UnboundedSender<Event>,
}

/// Streamed back to the HTTP handler, which forwards it as SSE.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Model load finished; generation is about to start.
    /// Loading the weights, `progress` from 0 to 1. Sent before `Ready` when
    /// a turn has to wait for a model to load.
    Loading { model: String, progress: f32 },
    Ready { model: String, context: u32 },
    Thinking { text: String },
    Answer { text: String },
    /// The model asked for a tool. Emitted before the tool runs.
    ///
    /// `id` matches the `ToolResult` that answers it. A batch is announced in
    /// full before any of it is awaited, so a client that tracks only "the
    /// current call" leaves every card but the last stuck running and lands
    /// the first result on the wrong one.
    ToolCall { id: String, name: String, arguments: serde_json::Value },
    /// The model has committed to a tool call and is still writing it.
    ///
    /// Sent as soon as the name is readable, which is long before the
    /// arguments are. Everything from the opening marker is withheld from the
    /// stream, so without this the page simply stops: a model writing a file
    /// generates the whole file before the call can be parsed, and a minute of
    /// nothing reads as a dropped connection. A `ToolCall` or a `Permission`
    /// with the same name follows once the arguments are in.
    ToolCallStarted { name: String },
    /// A tool call is waiting for the user to allow it.
    ///
    /// `id` is the same call id the matching `ToolCall` and `ToolResult`
    /// carry, so the client answers about the card it is showing. Exactly one
    /// of a `ToolCall` or a `ToolResult` follows: an approval runs the tool
    /// normally, a refusal goes straight to a result saying so.
    Permission {
        id: String,
        name: String,
        arguments: serde_json::Value,
        /// What the tool does to the world, so the page can colour the
        /// question by how much it matters.
        effect: ozgent_core::permission::Effect,
    },
    /// The model called a tool the *caller* owns. Nothing runs here; the turn
    /// ends and the caller is expected to execute it and send the result back.
    ClientToolCall { name: String, arguments: serde_json::Value },
    /// How that call turned out. `detail` is the tool's own result, so the
    /// browser can render it properly instead of showing raw JSON.
    ToolResult {
        /// The `ToolCall` this answers.
        id: String,
        name: String,
        ok: bool,
        summary: String,
        ms: u64,
        detail: serde_json::Value,
    },
    Done {
        generated: u32,
        tokens_per_second: f64,
        reused: usize,
        stop: String,
        /// Prompt tokens actually processed, for API usage accounting. Summed
        /// over every round of a turn, so not a measure of how full the
        /// context is: see `context`.
        prompt: u32,
        /// Tokens held in the context when the turn ended: what a gauge of
        /// the window should show. The sums above count each tool round's
        /// re-read again and ran past the window on a long agent turn.
        #[serde(default)]
        context: u32,
        /// Time spent on prefill. Reported so a caller can see prefix reuse
        /// working: a turn that reuses its prefix pays almost nothing here.
        prompt_ms: u64,
        /// Draft tokens proposed, and how many the model kept. Zero when
        /// nothing was drafting.
        #[serde(default, skip_serializing_if = "is_zero")]
        drafted: usize,
        #[serde(default, skip_serializing_if = "is_zero")]
        accepted: usize,
    },
    Error { message: String },
    /// An agent has taken over the turn. Everything until the matching
    /// `AgentEnd` — reasoning, tool calls, the answer — is that agent's.
    AgentStart {
        name: String,
        description: String,
        /// The tools it was actually offered.
        tools: Vec<String>,
        /// Tools it lists that are not available here, so the transcript can
        /// say why an agent answered without them.
        missing: Vec<String>,
    },
    /// The agent finished. `ok` is false when it produced no report.
    AgentEnd { name: String, ok: bool, ms: u64, calls: usize, rounds: usize },
}

/// Omit a count that carries no information. A turn with nothing drafting
/// should not report drafting nothing.
fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// Handle to the models. Routes each job to the thread holding its model,
/// starting one when nothing has it yet.
#[derive(Clone)]
pub struct Worker {
    pool: Arc<crate::pool::Pool>,
    context: Context,
}

/// What the embedding model is doing, for the admin page and the memory
/// layer. Written by the embedding thread when it loads.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct EmbedStatus {
    /// The model in use, as `name:tag`, once loaded.
    pub model: Option<String>,
    pub dimensions: usize,
    /// `gpu` or `cpu`.
    pub device: Option<&'static str>,
    /// Why the last load failed, if it did.
    pub error: Option<String>,
}

static EMBED_STATUS: std::sync::Mutex<EmbedStatus> =
    std::sync::Mutex::new(EmbedStatus { model: None, dimensions: 0, device: None, error: None });

/// The embedding model's current state.
pub fn embed_status() -> EmbedStatus {
    EMBED_STATUS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The projector type, aliased so the lifetime stays readable in signatures.
pub type LoadedProjector<'a> = ozgent_llama::mtmd::Projector<'a>;

/// What the inference thread needs to run tools.
///
/// The host is async and this thread is not, so calls are driven through a
/// runtime handle rather than blocking the executor that serves HTTP.
#[derive(Clone)]
pub struct Tools {
    /// Everything the model can call: ozgent's own Python tools and whatever
    /// the configured MCP servers offer, merged so nothing above this layer
    /// has to know which is which.
    pub host: Arc<Toolbox>,
    pub runtime: tokio::runtime::Handle,
    /// The scheduler, kept separately as well as inside the toolbox.
    ///
    /// It is the one tool that needs to know *who* is asking: a job created
    /// from a Telegram chat answers back into that chat, and may not be given
    /// tools that chat did not have. The toolbox deliberately hides which
    /// source a tool came from, so the handle is held here instead of
    /// downcasting back out of it.
    pub scheduler: Option<Arc<ozgent_schedule::ScheduleTools>>,
}

/// The Python tool host, shared so the settings page can replace it.
///
/// The host reads its configuration once, when the interpreter starts. Held
/// as a plain value it meant changing the search provider or its key was
/// saved to `config.toml`, reported as saved, and ignored until the server
/// was restarted — the same shape of bug as a stale `Config`.
pub type SharedTools = std::sync::Arc<std::sync::Mutex<Option<Tools>>>;

/// The tool host as it stands now.
pub fn current_tools(tools: &SharedTools) -> Option<Tools> {
    tools.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Everything a model thread needs, shared by all of them.
#[derive(Clone)]
struct Context {
    paths: Paths,
    config: SharedConfig,
    tools: SharedTools,
    permissions: Permissions,
    cli: CliOptions,
}

/// The key embeddings are held under.
///
/// Its own thread rather than a branch inside a chat model's loop: the
/// embedding model is a different model, and pinning it to whichever chat
/// model happened to be resident meant it was dropped and reloaded every time
/// that one changed.
const EMBED_KEY: &str = "\u{0}embeddings";

impl Worker {
    /// Start the router. Model threads come and go beneath it.
    pub fn spawn(
        paths: Paths,
        config: SharedConfig,
        tools: SharedTools,
        permissions: Permissions,
        cli: CliOptions,
    ) -> Self {
        Self {
            pool: Arc::new(crate::pool::Pool::default()),
            context: Context { paths, config, tools, permissions, cli },
        }
    }

    /// Queue a generation, loading the model if it is not already resident.
    pub fn submit(&self, request: Request) -> Result<(), &'static str> {
        // Keyed on the canonical reference, not on what the caller typed: an
        // alias and the full `name:tag` are one model, and keying on the text
        // would load it twice and hold two copies in VRAM.
        let key = match ozgent_core::resolve(&self.context.paths, &request.model) {
            Ok(found) => found.model.to_string(),
            Err(e) => {
                // Reported on the turn's own stream. The alternative — an
                // error from `submit` — reaches an HTTP handler that has
                // already started streaming and cannot say anything.
                let _ = request.out.send(Event::Error { message: e.to_string() });
                return Ok(());
            }
        };
        self.send(&key, Job::Generate(Box::new(request)))
    }

    /// Embed texts. Synchronous: there is nothing to stream, and the caller
    /// wants the vectors or an error rather than a channel.
    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        self.embed_as(ozgent_llama::embed::Role::Document, texts)
    }

    /// As [`Worker::embed`], refusing any text longer than the model takes —
    /// for API callers, who are owed an error rather than a vector of half
    /// their document.
    pub fn embed_strict(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let (tx, rx) = channel();
        let job = Job::Embed { texts, role: ozgent_llama::embed::Role::Document, strict: true, reply: tx };
        self.send(EMBED_KEY, job).map_err(str::to_string)?;
        rx.recv().map_err(|_| "the embedding thread stopped".to_string())?
    }

    /// Embed texts as queries or as documents; see
    /// [`ozgent_llama::embed::Role`]. Blocks until the vectors are back.
    pub fn embed_as(&self, role: ozgent_llama::embed::Role, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let (tx, rx) = channel();
        self.send(EMBED_KEY, Job::Embed { texts, role, strict: false, reply: tx })
            .map_err(str::to_string)?;
        rx.recv().map_err(|_| "the embedding thread stopped".to_string())?
    }

    /// The embedding model that would be used, as `name:tag`, whether or not
    /// it is loaded yet. `None` when embeddings are off or none is installed.
    pub fn embedding_model(&self) -> Option<String> {
        choose_embedding_model(&self.context.paths, &snapshot(&self.context.config)).ok().map(|(r, _)| r)
    }

    /// Drop the embedding model, so its next use loads it under the current
    /// settings — a different model, device or window.
    pub fn reload_embedder(&self) {
        if let Some(tx) = self.pool.get(EMBED_KEY) {
            let _ = tx.send(Job::Unload);
        }
        *EMBED_STATUS.lock().unwrap_or_else(|e| e.into_inner()) = EmbedStatus::default();
    }

    /// Drop every loaded model and free its VRAM.
    pub fn unload(&self) {
        self.pool.unload_all();
    }

    /// Which models are resident, most recently used first.
    pub fn loaded(&self) -> Vec<String> {
        self.pool.names().into_iter().filter(|n| n != EMBED_KEY).collect()
    }

    /// Hand a job to the thread for `key`, starting one if there is none.
    fn send(&self, key: &str, job: Job) -> Result<(), &'static str> {
        // Retried once. A model thread can time out and exit in the moment
        // between being found in the pool and being sent to, and a person
        // should not see that race as a failure. `SendError` hands the job
        // back, so the second attempt sends the same one rather than a copy.
        let mut job = job;
        for attempt in 0..2 {
            let tx = {
                // Checking and starting are one step: two callers that both
                // find nothing would otherwise each start a thread, and the
                // second would replace the first in the pool while the first
                // went on holding a model nobody could reach.
                let _starting = self.pool.starting();
                match self.pool.get(key) {
                    Some(tx) => tx,
                    None => self.start(key),
                }
            };
            match tx.send(job) {
                Ok(()) => return Ok(()),
                Err(std::sync::mpsc::SendError(returned)) if attempt == 0 => job = returned,
                Err(_) => break,
            }
        }
        Err("the model stopped before it could be given the work")
    }

    /// Start a thread for a model and register it.
    fn start(&self, key: &str) -> std::sync::mpsc::Sender<Job> {
        let (tx, rx) = channel::<Job>();
        let member = self.pool.insert(key, tx.clone());
        let context = self.context.clone();
        let key = key.to_string();
        let embedding = key == EMBED_KEY;
        let name = format!("ozgent-{}", if embedding { "embed" } else { &key });
        std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                if embedding {
                    run_embeddings(context, rx, &member);
                } else {
                    run_model(context, rx, &member);
                }
                member.leave();
                // The engine was dropped with the frame above. Freeing it is
                // not the same as giving it back; see `release_memory`.
                release_memory();
            })
            .expect("spawning a model thread");
        tx
    }
}



/// Outer loop: owns nothing but the channel, and loads a model on demand.
/// The server's configuration, shared with the HTTP handlers that edit it.
///
/// A clone taken at start-up is the bug this replaces: the settings page wrote
/// `config.toml`, asked for an unload, and the reload read the copy the thread
/// had been holding since boot — so a context-length change was saved,
/// reported back as applied, and never once reached the model.
pub type SharedConfig = std::sync::Arc<std::sync::Mutex<Config>>;

/// Read the configuration as it stands now.
fn snapshot(config: &SharedConfig) -> Config {
    config.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Everything the inference thread needs to ask about a tool call.
///
/// Grouped rather than passed as two more arguments because they are always
/// used together: the questions in flight, and the answers already given that
/// mean a question need not be asked at all.
#[derive(Clone)]
pub struct Permissions {
    pub pending: crate::permission::SharedPending,
    /// "Don't ask again this session", shared with the settings page so it can
    /// show — and clear — what has been granted.
    pub grants: SharedGrants,
    /// The standing policy. The same handle the settings page writes to, so a
    /// rule changed there applies to the very next call rather than the next
    /// restart.
    pub config: SharedConfig,
}

/// Option flags given to `ozgent web` or `ozgent serve` on the command line.
///
/// A layer of their own rather than folded into `config.defaults`, because
/// they have to beat the per-model blocks in `config.toml` — a flag typed at
/// the terminal that a config file silently overrides is worse than no flag.
/// They were previously discarded outright: `ozgent web --ctx 32k` parsed,
/// printed nothing, and changed nothing.
pub type CliOptions = std::sync::Arc<ozgent_core::Options>;

pub type SharedGrants = std::sync::Arc<std::sync::Mutex<ozgent_core::Grants>>;

/// One model's thread: load it, serve it, and stop when it goes idle.
///
/// Pinned to a single model, which is what lets several be resident at once.
/// It used to swap models in place, which meant a browser on one and a
/// terminal on another reloaded on every alternation.
fn run_model(context: Context, rx: Receiver<Job>, member: &crate::pool::Membership) {
    // Nothing is loaded until there is something to answer. A model thread
    // that started because a request arrived always has one waiting.
    let mut embedder: Option<ozgent_llama::embed::Embedder> = None;
    let first = loop {
        match rx.recv() {
            Ok(Job::Generate(r)) => break r,
            // Embeddings have their own thread; one arriving here is a caller
            // that has not been updated, and answering it is cheaper than
            // failing it.
            Ok(Job::Embed { texts, role, strict, reply }) => {
                let _ = reply.send(serve_embeddings(
                    &context.paths,
                    &snapshot(&context.config),
                    &mut embedder,
                    texts,
                    role,
                    strict,
                    Some(member),
                ));
            }
            Ok(Job::Unload) => return,
            Err(_) => return,
        }
    };

    // Loading and the session that borrows it both live in this scope, so the
    // borrow checker is satisfied without any self-referential trick — and
    // that is exactly why each model gets a thread rather than a slot in a
    // collection.
    // Kept so a failure can still be told to the request that caused it.
    // Only logged, a model that loaded and then could not open a context left
    // the page at "loading 100%" for good, with the reason in a log file.
    // Weak: a strong clone kept the channel open for the thread's whole life,
    // and a non-streaming API call, which ends when its channel closes, then
    // never returned.
    let asked = first.out.downgrade();
    if let Err(e) = serve_model(
        &context.paths,
        &context.config,
        &context.tools,
        &context.permissions,
        &context.cli,
        &mut embedder,
        first,
        &rx,
        member,
    ) {
        tracing::error!("{e}");
        if let Some(out) = asked.upgrade() {
            let _ = out.send(Event::Error { message: e.to_string() });
        }
    }
}

/// The embedding model's thread.
///
/// Separate because it is a separate model: held inside a chat model's loop it
/// was dropped and reloaded every time that model changed, which is a load of
/// its own for every switch.
fn run_embeddings(context: Context, rx: Receiver<Job>, member: &crate::pool::Membership) {
    let mut embedder: Option<ozgent_llama::embed::Embedder> = None;
    while let Ok(job) = rx.recv() {
        match job {
            Job::Embed { texts, role, strict, reply } => {
                // Marked busy while it works and recently used afterwards, so
                // eviction takes it in its proper turn rather than always
                // first — and never in the middle of a batch.
                member.working(true);
                let result = serve_embeddings(
                    &context.paths,
                    &snapshot(&context.config),
                    &mut embedder,
                    texts,
                    role,
                    strict,
                    Some(member),
                );
                member.working(false);
                let _ = reply.send(result);
            }
            Job::Unload => return,
            // Not this thread's work. Answered rather than dropped so the
            // turn ends with a reason instead of a closed stream.
            Job::Generate(request) => {
                let _ = request.out.send(Event::Error {
                    message: "that request reached the embedding model".into(),
                });
            }
        }
    }
}

/// Return freed heap to the operating system.
///
/// `free()` hands memory back to the allocator, not to the kernel — glibc
/// keeps it in its arenas ready for the next allocation, so a process that
/// loads and unloads a model looks, to `ps` and to anyone watching a systemd
/// unit, as though it never released anything. For a daemon that idles for
/// twenty hours between two briefs, that difference is the whole point of
/// unloading, so it is asked for explicitly.
///
/// glibc-only and advisory: every other allocator either does this already or
/// has no equivalent, and there is nothing to do on a failure.
pub fn release_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: malloc_trim takes a byte count and only ever returns unused
        // arena pages to the kernel. It cannot invalidate a live pointer.
        unsafe {
            unsafe extern "C" {
                fn malloc_trim(pad: usize) -> i32;
            }
            malloc_trim(0);
        }
    }
}

/// Load one model and serve jobs against it until a different one is asked
/// for, returning that job so the caller can load its model.
/// A fingerprint of each prefix of a conversation.
///
/// `marks[i]` identifies `messages[..=i]`, so two turns of the same
/// conversation agree on every mark up to where they diverge. That is what
/// lets a follow-up be sent back to the slot whose cache already holds its
/// history: the slot records the fingerprint of the messages it answered, and
/// the next turn finds that same value among its own marks.
fn prefix_marks(messages: &[Message]) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    let mut running = std::collections::hash_map::DefaultHasher::new();
    let mut marks = Vec::with_capacity(messages.len());
    for message in messages {
        // Hashed field by field rather than through `Hash`, because what
        // matters is what reaches the model: the role, the text, and the tool
        // traffic that is rendered into the prompt beside it.
        (message.role as u8).hash(&mut running);
        for part in &message.content {
            if let ozgent_core::Part::Text { text } = part {
                text.hash(&mut running);
            }
        }
        message.thinking.hash(&mut running);
        for call in &message.tool_calls {
            call.name.hash(&mut running);
            call.arguments.to_string().hash(&mut running);
        }
        message.tool_call_id.hash(&mut running);
        marks.push(running.clone().finish());
    }
    marks
}

/// Which slot should answer a turn whose prefixes are `marks`.
///
/// A conversation wants the slot whose cache already holds its history:
/// sending a follow-up anywhere else costs a cold prefill, measured at 1.32 s
/// a turn against 0.33 s when it goes back where it came from.
///
/// But a cache match must not beat an idle slot. Waiting for the matching
/// slot to finish what it is doing costs a whole turn, where re-prefilling
/// somewhere free costs a fraction of one — and preferring the match
/// regardless piled two requests onto one slot while another sat empty, worth
/// 1.56x against 2.19x with four callers. So free comes first, and the cache
/// match decides between the slots that are free.
fn route(marks: &[u64], served: &[u64], depth: &[usize]) -> usize {
    let reach = |slot: usize| marks.iter().rposition(|m| *m == served[slot]);
    (0..served.len())
        .min_by_key(|&slot| {
            (
                // Free before busy, but no finer than that: a slot with three
                // turns queued is not meaningfully worse than one with two.
                depth[slot].min(1),
                // Then the deepest cache match.
                std::cmp::Reverse(reach(slot)),
                // Then the shortest queue, then a stable choice.
                depth[slot],
                slot,
            )
        })
        .unwrap_or(0)
}

fn serve_model(
    paths: &Paths,
    shared: &SharedConfig,
    tools: &SharedTools,
    permissions: &Permissions,
    cli: &ozgent_core::Options,
    // Threaded through rather than rebuilt: an embedding request that arrives
    // mid-conversation should not reload the model it already has.
    embedder: &mut Option<ozgent_llama::embed::Embedder>,
    first: Box<Request>,
    rx: &Receiver<Job>,
    member: &crate::pool::Membership,
) -> anyhow::Result<Option<Box<Request>>> {
    // The name this thread answers to, canonically. Comparing the raw strings
    // callers send would treat an alias and its own `name:tag` as different
    // models — the pool keys on the resolved reference, so the check here must
    // too or every alias request would bounce straight back out.
    let wanted = first.model.clone();
    let found = match ozgent_core::resolve(paths, &wanted) {
        Ok(f) => f,
        Err(e) => {
            let _ = first.out.send(Event::Error { message: e.to_string() });
            return Ok(None);
        }
    };

    // The layers under every request for this model. The first request's
    // overrides are folded in too, because loading needs concrete numbers for
    // context length and layer placement and this is the only request in hand.
    // Read now, not at start-up: the settings page may have written the file
    // since, and this load is usually the direct consequence of that.
    let config = snapshot(shared);
    let base = config
        .options_for(&found.model.to_string())
        .merge(&found.manifest.defaults)
        .merge(cli);
    let resolved = base
        .clone()
        .merge(first.overrides.as_ref().unwrap_or(&Default::default()))
        .resolve();

    let weights = found.manifest.primary_weights(&found.dir);
    // Reported in whole percents: llama.cpp calls back per tensor, hundreds
    // of times, and a page only needs to see it move.
    let out = first.out.clone();
    let name = found.model.to_string();
    let _ = out.send(Event::Loading { model: name.clone(), progress: 0.0 });
    let mut last = 0u32;

    // Planning and loading under one lock. Two models loading at once would
    // each read the same free memory, each conclude they fit, and together
    // not: the driver's figure only falls once the weights are actually
    // resident. Generation is not serialised — only this is.
    //
    // If what is left would squeeze this model badly, idle models are dropped
    // first and the plan is made again. A model half on the GPU is fine; one
    // pushed almost entirely onto the CPU because something nobody is using
    // still holds VRAM is not.
    // The context holds one sequence per conversation plus the shared prefix,
    // and on a hybrid model every one of them keeps a recurrent state beside
    // each block on the card: placement has to know how many.
    //
    // For a model that does not fit, those states are paid for in blocks on
    // the CPU. Ternary Bonsai 27B with four conversations' worth kept 50 of 64
    // blocks on the card and decoded at 5.4 tok/s, for concurrency almost
    // nothing uses. When fewer slots buy blocks, the model serves one
    // conversation at a time and the rest wait their turn; a model that fits
    // either way keeps them all.
    let mut cap = snapshot(shared).web.parallel();
    if cap > 1 {
        let many = ozgent_llama::backend::Plan::for_model_with(&weights, &resolved, cap + 1);
        let one = ozgent_llama::backend::Plan::for_model_with(&weights, &resolved, 2);
        if !many.is_full() && one.layers > many.layers {
            tracing::info!(
                "{name}: one conversation at a time, which keeps {} blocks on the GPU instead of {}",
                one.layers,
                many.layers
            );
            cap = 1;
        }
    }
    let sequences = cap + 1;
    let (admitted, plan) = {
        let weights = weights.clone();
        let resolved = resolved.clone();
        member.admit(crate::pool::PlanFor(move || {
            ozgent_llama::backend::Plan::for_model_with(&weights, &resolved, sequences)
        }))
    };
    if !plan.is_full() {
        tracing::info!(
            "{name}: {} of {} layers on the GPU, the rest on the CPU",
            plan.layers,
            plan.total_layers
        );
    }

    // What was asked for, kept apart from what the load settles on. The load
    // may move more experts to the host than asked, and comparing each turn's
    // settings against *that* read as a changed setting on every message:
    // the model reloaded, moved them again, and reloaded again.
    let requested = resolved.clone();
    let mut resolved = resolved;
    let engine = match Engine::load_for(&weights, &resolved, sequences, move |p| {
        let pct = (p * 100.0) as u32;
        if pct > last {
            last = pct;
            let _ = out.send(Event::Loading { model: name.clone(), progress: p });
        }
    }) {
        Ok(e) => e,
        Err(e) => {
            let _ = first.out.send(Event::Error {
                message: format!("loading {}: {e}", found.model),
            });
            return Ok(None);
        }
    };
    // Said on every load, not only a partial one. "How much of this is on the
    // GPU" is the first question when something is slow, and a line that only
    // appears when the answer is bad means silence has to be read as good
    // news — which nobody does.
    tracing::info!(
        "{}: {} of {} layers on the GPU{}",
        found.model,
        engine.gpu_layers_used(),
        engine.n_layer(),
        if engine.gpu_layers_used() >= engine.n_layer() {
            String::new()
        } else {
            format!(", {} on the CPU", engine.n_layer() - engine.gpu_layers_used())
        }
    );

    // The weights are resident, so the next model may now plan against a
    // figure that includes them. Held any longer — to the end of this
    // function, which is the whole life of the thread — and the second model
    // blocks for ever on a lock the first never lets go of. Not hypothetical:
    // that is what the first version of this did, and it hung on the 9B.
    drop(admitted);

    // One session per slot, sharing a context and its forward passes.
    //
    // How many is decided by what the memory holds at the asked-for window,
    // not by the number in the config: `UpTo` opens a second slot only when a
    // second full-length cache fits beside the first. Concurrency that
    // silently halves somebody's context is not worth having.

    // A placement is made against an estimate of llama.cpp's scratch, and on
    // an architecture the estimate has not met it can be badly low: a 35B MoE
    // was planned as though its compute buffer were a few hundred megabytes,
    // asked for 1137 MiB, and the only thing left to give was the window --
    // 32768 tokens became 960, which cannot hold the system prompt, and every
    // turn failed. Trading the conversation away to keep a few more experts on
    // the card is the wrong trade: experts on the host cost tokens per second,
    // a window nothing fits in costs the model.
    //
    // So a load that offloaded experts is tried once before it is kept. If the
    // window comes up short, the shortfall is exactly known, as is what one
    // layer's experts weigh, so the reload moves precisely that many more
    // layers to the host -- no guess, and no waiting for a learned estimate to
    // converge over several failed loads.
    //
    // The attempt is dropped rather than kept because the sessions borrow the
    // engine, and an engine that is still borrowed cannot be replaced.
    let mut engine = engine;
    // Two reasons to move more experts to system RAM and load again, each
    // taken at most once. Both are about a model already split across the
    // card and the host, where one more layer of experts on the host costs
    // little and the alternatives cost a great deal.
    let auto_experts = matches!(
        resolved.cpu_moe,
        ozgent_core::accel::MoeOffload::Keyword(ozgent_core::accel::MoeKeyword::Auto)
    );
    let auto_layers = matches!(
        resolved.gpu_layers,
        ozgent_core::options::GpuLayers::Keyword(ozgent_core::options::GpuKeyword::Auto)
    );
    let mut floor_done = false;
    let mut wide_done = false;
    // Only a model already split between the card and the host is probed: one
    // wholly on the card has nothing to move, and its load pays nothing extra.
    while engine.cpu_moe_layers() > 0 || (auto_layers && engine.gpu_layers_used() < engine.n_layer()) {
        // Reduced to plain data at once: the sessions borrow the engine,
        // which may be about to be replaced.
        let probe = match engine.sessions(&resolved, ozgent_llama::engine::Slots::UpTo(cap)) {
            Ok(_) => Ok(()),
            Err(ozgent_llama::engine::EngineError::WindowBelowFloor { opened, floor, short_by }) => {
                Err(Some((opened, floor, short_by)))
            }
            Err(_) => Err(None),
        };
        let extra = match probe {
            // The window that fitted is below what the model needs: nothing
            // works until more room is found.
            Err(Some((opened, floor, short_by))) if !floor_done =>
            {
                floor_done = true;
                // Experts first: a layer's experts cost a MoE model far less
                // on the host than a whole block does. A dense model has only
                // whole blocks to move.
                if let Some(extra) = engine.expert_layers_for(short_by) {
                    tracing::warn!(
                        "{}: only {opened} tokens of context fitted beside the weights, below {floor}; \
                         moving the experts of {extra} more layer{} to system RAM and loading again",
                        found.model,
                        if extra == 1 { "" } else { "s" },
                    );
                    extra
                } else if let Some(extra) = engine.block_layers_for(short_by).filter(|_| auto_layers) {
                    let keep = engine.gpu_layers_used().saturating_sub(extra);
                    tracing::warn!(
                        "{}: only {opened} tokens of context fitted beside the weights, below {floor}; \
                         keeping {keep} of {} layers on the GPU and loading again",
                        found.model,
                        engine.n_layer(),
                    );
                    drop(engine);
                    resolved.gpu_layers = ozgent_core::options::GpuLayers::Count(keep);
                    let (admitted, _) = {
                        let weights = weights.clone();
                        let resolved = resolved.clone();
                        member.admit(crate::pool::PlanFor(move || {
                            ozgent_llama::backend::Plan::for_model_with(&weights, &resolved, sequences)
                        }))
                    };
                    engine = Engine::load_for(&weights, &resolved, sequences, |_| {})
                        .map_err(|e| anyhow::anyhow!("reloading {}: {e}", found.model))?;
                    drop(admitted);
                    continue;
                } else {
                    anyhow::bail!("{}: only {opened} tokens of context fit, below {floor}", found.model);
                }
            }
            // The context opened, but on the narrow micro-batch. With experts
            // on the host every micro-batch of a prefill uploads all of them
            // across the bus, so halving how many there are nearly halves
            // prefill: on a 35B MoE with 28 of 40 layers' experts on the host,
            // 1024 against 512 measured 589 tok/s against 378, for 26.2 tok/s
            // of decoding against 26.4 with two more layers moved. Worth it
            // for a few layers; not for many, where decoding starts to pay —
            // four more and a 2048 batch cost twelve percent — so it is
            // bounded to a twentieth of the model.
            Ok(()) if auto_experts && !wide_done => {
                wide_done = true;
                let Some(extra) = engine
                    .wide_batch_shortfall()
                    .and_then(|bytes| engine.expert_layers_for(bytes))
                    .filter(|extra| extra * 20 <= engine.n_layer())
                else {
                    break;
                };
                tracing::info!(
                    "{}: moving the experts of {extra} more layer{} to system RAM for the wider \
                     micro-batch, which roughly halves how often a prefill re-uploads them",
                    found.model,
                    if extra == 1 { "" } else { "s" },
                );
                extra
            }
            _ => break,
        };
        let layers = (engine.cpu_moe_layers() + extra).min(engine.n_layer());
        drop(engine);
        resolved.cpu_moe = ozgent_core::accel::MoeOffload::Layers(layers);
        let (admitted, _) = {
            let weights = weights.clone();
            let resolved = resolved.clone();
            member.admit(crate::pool::PlanFor(move || {
                ozgent_llama::backend::Plan::for_model_with(&weights, &resolved, sequences)
            }))
        };
        engine = Engine::load_for(&weights, &resolved, sequences, |_| {})
            .map_err(|e| anyhow::anyhow!("reloading {}: {e}", found.model))?;
        drop(admitted);
    }
    let sessions = engine.sessions(&resolved, ozgent_llama::engine::Slots::UpTo(cap))?;
    let slots = sessions.len();
    tracing::info!(
        "{}: {slots} conversation{} at a time, {} tokens each",
        found.model,
        if slots == 1 { "" } else { "s" },
        sessions.first().map(|s| s.n_ctx()).unwrap_or(0),
    );
    // The system prompt and tool schemas are the same on every turn, and they
    // are most of the prompt: 2822 tokens against a user message of twenty.
    // Held once here, every conversation borrows them for the cost of a cache
    // copy instead of prefilling them itself.
    //
    // Done now rather than on the first turn because the model is loading
    // anyway and nobody is waiting on a token yet. Left to discover itself,
    // the shared prefix cannot exist until a second prompt has arrived to be
    // compared against the first, so the first turn of every conversation
    // prefills it in full and the second turn prefills it *again* to fill the
    // commons -- 1665 ms and 1595 ms, measured, on a 4B that fits on the card.
    if let Some(first) = sessions.first() {
        prewarm(&engine, first, tools, &resolved, paths, &snapshot(shared));
    }

    // Registered busy so nothing evicts it mid-load; it is loaded now, and the
    // turn loop below takes the flag again for each answer. Without this a
    // model would stay marked busy for ever and never become evictable.
    member.working(false);
    let mmproj = found.manifest.projector_path(&found.dir);

    // A queue per slot rather than one shared one, so a conversation can be
    // sent back to the slot holding its cache. See `route`.
    let mut work_tx = Vec::with_capacity(slots);
    let mut work_rx = Vec::with_capacity(slots);
    for _ in 0..slots {
        let (tx, rx) = channel::<Box<Request>>();
        work_tx.push(tx);
        work_rx.push(std::sync::Mutex::new(Some(rx)));
    }
    // What each slot last answered, and how much is waiting for it.
    let served: Vec<std::sync::atomic::AtomicU64> =
        (0..slots).map(|_| std::sync::atomic::AtomicU64::new(0)).collect();
    let depth: Vec<std::sync::atomic::AtomicUsize> =
        (0..slots).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect();
    // How many slots are mid-answer. The pool's busy flag is one bit for the
    // whole model, so it is set when the first slot starts and cleared when
    // the last finishes — clearing it per slot would offer the model up for
    // eviction while another slot was still writing.
    let busy = std::sync::atomic::AtomicUsize::new(0);
    // A request that cannot be served by this load, handed back to be served
    // against the next one.
    let handback = std::sync::Mutex::new(None::<Box<Request>>);
    let stopping = std::sync::atomic::AtomicBool::new(false);

    std::thread::scope(|scope| {
        for (i, mut session) in sessions.into_iter().enumerate() {
            let rx_mine = work_rx[i].lock().unwrap().take().expect("one receiver per slot");
            let (busy, handback, stopping) = (&busy, &handback, &stopping);
            let (served, depth) = (&served, &depth);
            let (engine, resolved, requested, found, mmproj) = (&engine, &resolved, &requested, &found, &mmproj);
            std::thread::Builder::new()
                .name(format!("ozgent-slot-{i}"))
                .spawn_scoped(scope, move || {
                    // Loaded on the first turn that needs it and kept: it costs
                    // VRAM, but a conversation with one image usually has more.
                    // One per slot, because a projector writes embeddings into
                    // the cache of whichever session is using it.
                    let mut projector: Option<crate::worker::LoadedProjector<'_>> = None;
                    loop {
                        let Ok(request) = rx_mine.recv() else { return };

                        // Each request brings its own sampling. Re-resolving per
                        // turn is what keeps two clients on one model from
                        // inheriting each other's temperature, seed and
                        // reasoning budget.
                        let live = snapshot(shared);
                        let base = live
                            .options_for(&found.model.to_string())
                            .merge(&found.manifest.defaults)
                            .merge(cli);
                        let per_turn = base
                            .merge(request.overrides.as_ref().unwrap_or(&Default::default()))
                            .resolve();

                        // Context length, layer placement and the rest are fixed
                        // when the weights load. Handing the request back sends
                        // it to a freshly loaded model, which is what "applies on
                        // your next message" has to mean if it is to be true.
                        if requested.needs_reload(&per_turn) {
                            tracing::info!(
                                "a load-time setting changed; reloading {}",
                                found.model
                            );
                            *handback.lock().unwrap_or_else(|e| e.into_inner()) = Some(request);
                            stopping.store(true, std::sync::atomic::Ordering::SeqCst);
                            return;
                        }
                        session.set_options(&per_turn);
                        let thinking = request.thinking.unwrap_or(per_turn.thinking);
                        let _ = request.out.send(Event::Ready {
                            model: found.model.to_string(),
                            context: session.n_ctx(),
                        });

                        if !request.images.is_empty() && projector.is_none() {
                            match mmproj {
                                Some(path) => match engine.projector(path, resolved) {
                                    Ok(p) => projector = Some(p),
                                    Err(e) => {
                                        let _ = request.out.send(Event::Error {
                                            message: format!(
                                                "loading the vision projector: {e}"
                                            ),
                                        });
                                    }
                                },
                                None => {
                                    let _ = request.out.send(Event::Error {
                                        message: format!(
                                            "{} cannot see images: no vision projector installed",
                                            found.model
                                        ),
                                    });
                                }
                            }
                        }

                        // Held across the turn so this model cannot be chosen as
                        // the one to evict while somebody is reading its answer.
                        if busy.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                            member.working(true);
                        }
                        let outcome = turn(
                            engine,
                            &mut session,
                            &per_turn,
                            thinking,
                            tools,
                            permissions,
                            projector.as_ref(),
                            &request,
                        );
                        if busy.fetch_sub(1, std::sync::atomic::Ordering::SeqCst) == 1 {
                            member.working(false);
                        }
                        // Generation parks the slot on its way out, but a turn
                        // that ended through an error may not have reached
                        // that. An unparked slot is one the others wait for.
                        session.park();
                        if let Err(e) = outcome {
                            let _ = request.out.send(Event::Error { message: e.to_string() });
                        }
                        // Counted down here rather than on receipt, because
                        // what `route` needs to know is how much work a slot
                        // still has — not how much is sitting in its channel.
                        // Decrementing on `recv` made every slot look idle the
                        // instant it was handed something, so four concurrent
                        // requests all went to slot zero and queued: 1.24x
                        // where spreading them gives 2.19x.
                        depth[i].fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                        // Remembered so the next turn of this conversation
                        // comes back here, where its history is already cached.
                        served[i].store(
                            prefix_marks(&request.messages).last().copied().unwrap_or(0),
                            std::sync::atomic::Ordering::SeqCst,
                        );
                        // A turn that ended while a question was outstanding
                        // leaves nobody to answer it; the card is gone from the
                        // page with the stream.
                        permissions.pending.abandon_all();
                        // Dropping the sender ends the SSE stream.
                        drop(request);
                    }
                })
                .expect("spawning a slot");
        }

        // The dispatcher. It never generates, so embedding requests and the
        // idle clock are answered while every slot is busy.
        let mut queued = Some(first);
        loop {
            if let Some(request) = queued.take() {
                let marks = prefix_marks(&request.messages);
                let seen: Vec<u64> =
                    served.iter().map(|s| s.load(std::sync::atomic::Ordering::SeqCst)).collect();
                let waiting: Vec<usize> =
                    depth.iter().map(|d| d.load(std::sync::atomic::Ordering::SeqCst)).collect();
                let slot = route(&marks, &seen, &waiting);
                depth[slot].fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if work_tx[slot].send(request).is_err() {
                    break;
                }
            }
            if stopping.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            let idle_for = snapshot(shared).web.idle_unload();
            let job = match idle_for {
                Some(limit) => match rx.recv_timeout(limit) {
                    Ok(job) => job,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        // Idle means nobody is being answered, not merely that
                        // nothing new arrived. A long turn must not unload the
                        // model out from under itself.
                        if busy.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                            continue;
                        }
                        tracing::info!(
                            "idle for {} minutes; unloading {wanted}",
                            limit.as_secs() / 60
                        );
                        break;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                },
                None => match rx.recv() {
                    Ok(job) => job,
                    Err(_) => break,
                },
            };
            match job {
                Job::Generate(request) => {
                    // The pool routes by resolved reference, so this should
                    // never differ. Checked anyway, and compared the way the
                    // pool compares: a request that did somehow reach the wrong
                    // model must not be answered by it.
                    let same = request.model == wanted
                        || ozgent_core::resolve(paths, &request.model)
                            .map(|f| f.model.to_string() == found.model.to_string())
                            .unwrap_or(false);
                    if !same {
                        *handback.lock().unwrap_or_else(|e| e.into_inner()) = Some(request);
                        break;
                    }
                    queued = Some(request);
                }
                Job::Embed { texts, role, strict, reply } => {
                    let _ = reply.send(serve_embeddings(paths, &snapshot(shared), embedder, texts, role, strict, None));
                }
                // Unloading means returning so the engine is dropped with the
                // scope.
                Job::Unload => break,
            }
        }
        // Ends every slot's `recv`, so the scope can join them.
        work_tx.clear();
    });

    Ok(handback.into_inner().unwrap_or_else(|e| e.into_inner()))
}

#[cfg(test)]
mod routing {
    use super::route;

    #[test]
    fn a_follow_up_goes_back_to_the_slot_that_answered_it() {
        // Slot 1 answered a conversation whose last mark was 77; this turn
        // extends it, so 77 is one of its marks.
        let marks = [11, 77, 99];
        assert_eq!(route(&marks, &[5, 77, 6], &[0, 0, 0]), 1);
    }

    #[test]
    fn an_idle_slot_beats_a_busy_match() {
        // Slot 0 holds the history but is mid-turn. Waiting for it costs a
        // whole turn; re-prefilling on slot 2 costs part of one.
        let marks = [11, 77];
        assert_eq!(route(&marks, &[77, 5, 6], &[1, 1, 0]), 2);
    }

    #[test]
    fn the_deepest_match_wins_among_free_slots() {
        // Both slots know this conversation; slot 1 knows more of it.
        let marks = [11, 77, 99];
        assert_eq!(route(&marks, &[11, 99, 5], &[0, 0, 0]), 1);
    }

    #[test]
    fn with_nothing_to_match_the_work_is_spread() {
        let marks = [1, 2];
        assert_eq!(route(&marks, &[0, 0, 0], &[1, 0, 0]), 1);
        assert_eq!(route(&marks, &[0, 0, 0], &[1, 1, 0]), 2);
        assert_eq!(route(&marks, &[0, 0, 0], &[2, 1, 3]), 1);
    }

    #[test]
    fn everything_busy_falls_back_to_the_match() {
        let marks = [11, 77];
        assert_eq!(route(&marks, &[5, 77, 6], &[1, 2, 1]), 1);
    }

    #[test]
    fn one_slot_is_always_a_valid_answer() {
        assert_eq!(route(&[1], &[0], &[3]), 0);
    }
}

/// Why an embedding request cannot be served.
///
/// Deliberately an error rather than a fallback. Pooling a chat model's hidden
/// states yields vectors that look plausible, cluster badly, and give the
/// caller no way to tell — the same trap the memory layer fell into by shipping
/// a lexical stand-in behind a semantic-sounding interface.
/// Answer an embedding request, loading the model on first use.
fn serve_embeddings(
    paths: &Paths,
    config: &Config,
    slot: &mut Option<ozgent_llama::embed::Embedder>,
    texts: Vec<String>,
    role: ozgent_llama::embed::Role,
    strict: bool,
    member: Option<&crate::pool::Membership>,
) -> Result<Vec<Vec<f32>>, String> {
    if !config.embedding.enabled {
        return Err("embeddings are switched off ([embedding] enabled = false)".into());
    }
    if slot.is_none() {
        // Planned and loaded under the pool's loading lock, so it cannot
        // read the same free memory a chat model is being placed into.
        let _guard = member.map(|m| m.loading());
        let beside_a_chat_model = member.is_some_and(|m| m.others_resident());
        match load_embedder(paths, config, beside_a_chat_model) {
            Ok((embedder, reference, device)) => {
                *EMBED_STATUS.lock().unwrap_or_else(|e| e.into_inner()) = EmbedStatus {
                    model: Some(reference),
                    dimensions: embedder.dimensions(),
                    device: Some(device),
                    error: None,
                };
                *slot = Some(embedder);
            }
            Err(e) => {
                EMBED_STATUS.lock().unwrap_or_else(|e| e.into_inner()).error = Some(e.clone());
                return Err(e);
            }
        }
    }
    let embedder = slot.as_ref().expect("just loaded");
    if strict {
        if let Err(t) = embedder.check_lengths(role, &texts) {
            return Err(format!(
                "input {} is {} tokens; this embedding model can take at most {} per input right now \
                 ([embedding] max_tokens caps it; on the GPU, free memory does too), so split the text",
                t.index, t.tokens, t.limit
            ));
        }
    }
    embedder.embed_as(role, &texts).map_err(|e| e.to_string())
}

/// The embedding model to use: the configured one, or the first installed.
pub(crate) fn choose_embedding_model(
    paths: &Paths,
    config: &Config,
) -> Result<(String, ozgent_core::Installed), String> {
    if !config.embedding.enabled {
        return Err("embeddings are switched off".into());
    }
    if let Some(name) = config.embedding.model.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        let found = ozgent_core::resolve(paths, name).map_err(|e| format!("embedding model {name:?}: {e}"))?;
        return Ok((found.model.to_string(), found));
    }
    ozgent_core::installed(paths)
        .into_iter()
        .find(|m| ozgent_llama::layout::is_embedding(&m.manifest.primary_weights(&m.dir)))
        .map(|m| (m.model.to_string(), m))
        .ok_or_else(embedding_unavailable)
}

/// Load the embedding model, once, where it fits. Returns it, its name, and
/// where it went.
///
/// Lazily, because a session that never recalls anything never pays for it.
/// On the GPU only when it fits above everything a resident chat model holds
/// *and* what that model still needs for its next decode — the memory
/// llama.cpp allocates lazily and aborts the whole process without.
fn load_embedder(
    paths: &Paths,
    config: &Config,
    beside_a_chat_model: bool,
) -> Result<(ozgent_llama::embed::Embedder, String, &'static str), String> {
    use ozgent_core::config::EmbedDevice;
    let (reference, found) = choose_embedding_model(paths, config)?;
    let weights = found.manifest.primary_weights(&found.dir);
    let size = std::fs::metadata(&weights).map(|m| m.len()).unwrap_or(0);
    let gpu = match config.embedding.device {
        EmbedDevice::Gpu => true,
        EmbedDevice::Cpu => false,
        // Only into room a chat model has already left. Loaded first, it
        // would take memory the next chat model is planned against, and a
        // chat model losing a few layers is not a squeeze bad enough for the
        // pool to evict anything to get them back.
        EmbedDevice::Auto if !beside_a_chat_model => false,
        EmbedDevice::Auto => {
            // Weights, a context of eight 512-token lanes and its scratch,
            // and the decode reserve left untouched for whoever else is here.
            // The working context's cache (8k tokens, about 60 KB a token at
            // 8 bits for a 0.6B model); longer contexts are checked against
            // free memory when they are needed.
            let cache = ozgent_llama::embed::WORKING_TOKENS as u64 * (64 << 10);
            let need = size + size / 10 + cache + (256 << 20) + ozgent_llama::backend::decode_reserve();
            ozgent_llama::backend::best_gpu()
                .filter(|d| d.is_gpu())
                .is_some_and(|d| d.memory_free as u64 >= need)
        }
    };
    let embedder = ozgent_llama::embed::Embedder::load_with(&weights, if gpu { 99 } else { 0 }, config.embedding.max_tokens)
        .map_err(|e| format!("loading embedding model {reference:?}: {e}"))?;
    let device = if gpu { "gpu" } else { "cpu" };
    tracing::info!("embedding model {reference}: {} dimensions, on the {device}", embedder.dimensions());
    Ok((embedder, reference, device))
}

fn embedding_unavailable() -> String {
    "no embedding model is installed. Get one on the admin page (Models), for example \
     Qwen3-Embedding-0.6B"
        .to_string()
}

/// One user turn: generate, run any tools the model asks for, generate again.
///
/// Decide whether one call may run, asking the browser if the policy says to.
///
/// Returns `None` for a refusal, or `Some(by_user)` to run it, where `by_user`
/// records that a person authorised this call rather than a default doing it
/// for them — the distinction the Python sandbox reads.
///
/// Blocking is deliberate and safe here: this is the inference thread, which
/// has nothing else to do while a tool would have been running, and the wait
/// is bounded so a closed tab cannot strand it.
fn permit(
    permissions: &Permissions,
    spec: Option<&ozgent_core::ToolSpec>,
    call: &ozgent_core::ToolCall,
    request: &Request,
    agent: Option<&ozgent_core::Agent>,
) -> Option<bool> {
    use ozgent_core::permission::Verdict;
    let config = &permissions.config;

    // A call naming a tool that does not exist fails in the host a moment
    // later with a far better message than anything here could produce.
    let effect = spec.map(|s| s.effect).unwrap_or_default();

    let verdict = {
        let policy = config.lock().unwrap_or_else(|e| e.into_inner());
        let grants = permissions.grants.lock().unwrap_or_else(|e| e.into_inner());
        match agent {
            // An agent's own rules sit on top of the policy; see
            // `Agent::verdict` for the order they apply in.
            Some(agent) => agent.verdict(&policy.permissions, &call.name, effect, &grants),
            None => policy.permissions.verdict(&call.name, effect, &grants),
        }
    };
    // A key's scopes come first: a caller without the scope for this effect
    // is refused whatever the policy would allow. With it, a question the
    // policy would ask is answered by the key — the operator decided when
    // issuing it. Never as `by_user`: that flag tells the tool a person saw
    // this exact call, which lifts limits (a command allowlist, a root) that
    // a standing grant must not.
    if let Some(grant) = &request.grant {
        if !grant.permits(effect) {
            tracing::info!("key {}: refused {} ({effect:?} is not in its scopes)", grant.key, call.name);
            return None;
        }
        match verdict {
            Verdict::Deny => return None,
            Verdict::Allow { .. } | Verdict::Ask => return Some(false),
        }
    }
    match verdict {
        Verdict::Allow { by_user } => return Some(by_user),
        Verdict::Deny => return None,
        // Nobody to ask. An OpenAI client is a program, and a program cannot
        // consent on a person's behalf; blocking for five minutes and then
        // refusing would be the same answer, arrived at slowly. An operator
        // who wants these tools available to the API says so in the policy.
        Verdict::Ask if !request.can_ask => return None,
        Verdict::Ask => {}
    }

    let rx = permissions.pending.ask(&call.id);
    if request
        .out
        .send(Event::Permission {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            effect,
        })
        .is_err()
    {
        // Nobody is listening to this turn any more, so nobody will answer.
        permissions.pending.forget(&call.id);
        return None;
    }

    let choice = crate::permission::wait(rx);
    permissions.pending.forget(&call.id);
    permissions
        .grants
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remember(&call.name, choice);

    // "Always" is the answer that outlives the process, so it is the one that
    // touches the file. Saving is best-effort: a config that could not be
    // written must not turn an approval into a refusal.
    let mut policy = config.lock().unwrap_or_else(|e| e.into_inner());
    if policy.permissions.apply(&call.name, choice) {
        let saved = ozgent_core::Paths::discover()
            .map_err(|e| e.to_string())
            .and_then(|p| policy.save(&p).map_err(|e| e.to_string()));
        if let Err(e) = saved {
            tracing::warn!("could not save the permission for {}: {e}", call.name);
        }
    }
    choice.is_allow().then_some(true)
}

/// Mirrors the terminal client's loop so a conversation behaves the same in
/// both front ends. Bounded, because a model that keeps calling tools would
/// otherwise never produce an answer.
///
/// A turn is either the model answering in its own voice, or one or more
/// agents answering in theirs. Both run the same rounds; an agent differs in
/// what it is told, which tools it is shown, what its rules are, and how many
/// rounds it gets.
fn turn(
    engine: &Engine,
    session: &mut ozgent_llama::engine::Session<'_>,
    resolved: &ozgent_core::options::Resolved,
    thinking: ThinkingMode,
    tools: &SharedTools,
    permissions: &Permissions,
    projector: Option<&LoadedProjector<'_>>,
    request: &Request,
) -> anyhow::Result<()> {
    // Images are read here rather than in the handler: failures belong in the
    // stream the user is watching, next to the turn they broke.
    let images = match projector {
        Some(_) if !request.images.is_empty() => {
            match ozgent_llama::mtmd::load_media(&request.images) {
                Ok(images) => images,
                Err(e) => {
                    let _ = request.out.send(Event::Error { message: e.to_string() });
                    Vec::new()
                }
            }
        }
        _ => Vec::new(),
    };

    // Read per turn, so switching the search provider or disabling a tool in
    // Settings reaches the very next message rather than the next restart.
    let tools = current_tools(tools);
    // A template with its own tools block tells the model the call format it
    // was trained on; ozgent's generic description would be a second one, in
    // a different syntax, and the model splits the difference.
    let native_tools = engine.template_handles_tools();

    // Ground the model in the picture before it is allowed to act on it.
    //
    // Asked "what can you tell me about this image", the model would search the
    // web for terms lifted from the picture and then answer from the results —
    // getting colours wrong that it had described correctly with no tools at
    // all. Instructions not to did not stop it.
    //
    // Removing tools for the turn would fix that and break the opposite case:
    // an image plus "what does this cost now?" genuinely needs a search. So
    // instead of taking the capability away, the model is made to look first. A
    // short, tool-free pass describes what is actually there; that description
    // joins the prompt, and the real turn proceeds with every tool available.
    //
    // The question "what is in this image" is then already answered, so there
    // is nothing to search for — and when a search *is* wanted, it is issued
    // from an accurate reading of the image rather than a guess at it. This is
    // the grounding-before-response pattern the vision-language literature
    // settles on for the same failure.
    let observation = if images.is_empty() {
        None
    } else {
        ground(engine, session, &request.messages, media_for(projector, &images), request)
    };

    let mut totals = Outcome::default();
    if request.agents.is_empty() {
        let Some(offered) = offer(tools.as_ref(), request) else { return Ok(()) };
        // Caller-supplied tools are additive: a request may use the server's
        // Python tools, its own, or both. A name collision resolves in favour
        // of the server's, because that is the one this process can actually
        // run.
        let server_names: std::collections::HashSet<String> =
            offered.iter().map(|s| s.name.clone()).collect();
        let client_owned: std::collections::HashSet<String> = request
            .client_tools
            .iter()
            .map(|s| s.name.clone())
            .filter(|n| !server_names.contains(n))
            .collect();
        let mut offered = offered;
        for spec in &request.client_tools {
            if client_owned.contains(&spec.name) {
                offered.push(spec.clone());
            }
        }

        let mut messages = request.messages.clone();
        prepare(&mut messages, &offered, native_tools, projector, &images, observation.as_deref());
        if offered.iter().any(|s| s.name == ozgent_core::agents::HANDOFF_TOOL) {
            append_system(&mut messages, &ozgent_core::agents::handoff_prompt(&request.handoff));
        }
        if !install(engine, session, &offered, request.response_grammar.as_deref(), request) {
            return Ok(());
        }
        totals = rounds(
            engine, session, resolved, thinking, tools.as_ref(), permissions, projector, &images,
            request, messages, &offered, &client_owned, MAX_TOOL_ROUNDS, None,
        )?;

        // The model passed the request to an agent: it answers from here,
        // exactly as if the user had named it. The images were already read
        // into this turn, so the agent gets the observation instead.
        if let Some((agent, task)) = totals.handoff.take() {
            let note = ozgent_core::agents::handoff_note(&task);
            let outcome = run_agents(
                engine, session, resolved, thinking, tools.as_ref(), permissions, projector,
                &[], observation.as_deref(), request, std::slice::from_ref(&agent), Some(&note),
            )?;
            let before = std::mem::take(&mut totals.text);
            totals.absorb(outcome);
            if !before.trim().is_empty() {
                totals.text = format!("{before}\n\n{}", totals.text);
            }
        }
    } else {
        let outcome = run_agents(
            engine, session, resolved, thinking, tools.as_ref(), permissions, projector,
            &images, observation.as_deref(), request, &request.agents, None,
        )?;
        totals.absorb(outcome);
    }

    if request.response_grammar.is_some() {
        let _ = session.set_grammar(None);
    }

    let seconds = totals.elapsed_ms as f64 / 1000.0;
    let _ = request.out.send(Event::Done {
        generated: totals.generated,
        tokens_per_second: if seconds > 0.0 { totals.generated as f64 / seconds } else { 0.0 },
        reused: totals.reused,
        stop: if totals.handed_back { "ToolCalls".to_string() } else { format!("{:?}", totals.stop) },
        prompt: totals.prompt_tokens,
        context: session.used(),
        prompt_ms: totals.prompt_ms as u64,
        drafted: totals.proposed,
        accepted: totals.accepted,
    });
    Ok(())
}

/// Run agents one after another, each with its own instructions and tools.
/// Each sees the conversation, plus `note` when there is one, plus the
/// reports of the agents before it.
#[allow(clippy::too_many_arguments)]
fn run_agents(
    engine: &Engine,
    session: &mut ozgent_llama::engine::Session<'_>,
    resolved: &ozgent_core::options::Resolved,
    thinking: ThinkingMode,
    tools: Option<&Tools>,
    permissions: &Permissions,
    projector: Option<&LoadedProjector<'_>>,
    images: &[ozgent_llama::mtmd::Media],
    observation: Option<&str>,
    request: &Request,
    agents: &[ozgent_core::Agent],
    note: Option<&str>,
) -> anyhow::Result<Outcome> {
    let native_tools = engine.template_handles_tools();
    let mut totals = Outcome::default();
    // What each agent may be shown. A channel's allowlist narrows agents
    // too: consent arriving over a chat is consent from whoever holds that
    // account, and an agent is not a way round the list.
    let available: Vec<ozgent_core::ToolSpec> = tools
        .map(|t| t.host.tools().to_vec())
        .unwrap_or_default()
        .into_iter()
        .filter(|s| request.native_tools.as_ref().is_none_or(|list| list.contains(&s.name)))
        .collect();
    let date = snapshot(&permissions.config)
        .ui
        .date_awareness
        .then(|| crate::turn::date_line());

    let mut conversation = request.messages.clone();
    for (index, agent) in agents.iter().enumerate() {
        let (offered, missing) = agent.offer(&available);
        let _ = request.out.send(Event::AgentStart {
            name: agent.name.clone(),
            description: agent.definition.description.clone(),
            tools: offered.iter().map(|s| s.name.clone()).collect(),
            missing: missing.clone(),
        });

        // Sampling is the agent's where it says, the turn's otherwise.
        let mut tuned = resolved.clone();
        if let Some(t) = agent.definition.temperature {
            tuned.temperature = t;
        }
        if let Some(m) = agent.definition.max_tokens {
            tuned.max_tokens = m;
        }
        session.set_options(&tuned);
        if !install(engine, session, &offered, None, request) {
            break;
        }

        let mut messages = conversation.clone();
        let system = agent.system_prompt(date.as_deref(), &missing);
        // The agent's instructions replace the chat's system prompt rather
        // than joining it: a persona written for the chat, or a client's
        // description of tools the agent does not have, would contradict the
        // job it was called to do. A handoff note joins the instructions: it
        // is the job. Joined rather than sent as a message of its own after
        // the conversation, which templates that take a system message only
        // at the start refuse — and the fallback has no tools to offer.
        messages.retain(|m| m.role != ozgent_core::Role::System);
        messages.insert(0, Message::system(system));
        if let Some(note) = note {
            append_system(&mut messages, note);
        }
        // Images belong to the first agent: they are evaluated into the cache
        // once, with that agent's prompt around them.
        let (media_images, media_note) = if index == 0 { (images, observation) } else { (&[][..], None) };
        prepare(&mut messages, &offered, native_tools, projector, media_images, media_note);

        let started = std::time::Instant::now();
        let outcome = rounds(
            engine,
            session,
            &tuned,
            agent.definition.thinking.unwrap_or(thinking),
            tools,
            permissions,
            projector,
            media_images,
            request,
            messages,
            &offered,
            &Default::default(),
            agent.definition.rounds(),
            Some(agent),
        )?;
        let _ = request.out.send(Event::AgentEnd {
            name: agent.name.clone(),
            ok: !outcome.text.trim().is_empty(),
            ms: started.elapsed().as_millis() as u64,
            calls: outcome.calls,
            rounds: outcome.rounds,
        });
        conversation.push(Message::assistant(outcome.text.clone()));
        totals.absorb(outcome);
        if request.out.is_closed() {
            break;
        }
    }
    // The next request on this session must not inherit the last agent's
    // temperature.
    session.set_options(resolved);
    Ok(totals)
}

/// The built-in tools a plain turn offers, or `None` after reporting that the
/// request named one that does not exist.
/// Prefill the part of the prompt every turn repeats, before any turn asks.
///
/// The stable head is not guessed at: two prompts are rendered that differ
/// only in the user's words, and their common run of tokens *is* the head, by
/// construction. Anything that varies -- the message, the date, a per-turn
/// instruction -- differs between the two and falls outside it. That matters
/// more than saving the render: a prefix that is merely *probably* shared is a
/// cache other conversations would borrow while describing something else.
///
/// A conversation offered a different set of tools simply will not match this
/// prefix and prefills its own, which is what happens today for every
/// conversation.
fn prewarm(
    engine: &Engine,
    session: &ozgent_llama::engine::Session<'_>,
    tools: &SharedTools,
    resolved: &ozgent_core::options::Resolved,
    paths: &Paths,
    config: &Config,
) {
    let Some(t) = current_tools(tools) else { return };
    let native_tools = engine.template_handles_tools();
    // The head of a real turn, built the way `turn::start` and `run` build it:
    // the same system prompt, the same tools, the handoff tool and its note.
    // Rendering without them produced a prefix that no real prompt shared —
    // a template that puts the system text ahead of the tools diverged three
    // tokens in, and the held prefix was never once lent.
    let system = crate::turn::system_prompt(config.ui.date_awareness, None);
    let handoff = if config.tools.handoff {
        ozgent_core::AgentCatalog::load(paths).all().to_vec()
    } else {
        Vec::new()
    };
    let mut specs = t.host.tools().to_vec();
    specs.extend(ozgent_core::agents::handoff_spec(&handoff));
    if specs.is_empty() && system.is_none() {
        return;
    }
    let render = |text: &str| {
        let mut messages: Vec<Message> = system.iter().map(|s| Message::system(s.as_str())).collect();
        messages.push(Message::user(text));
        prepare(&mut messages, &specs, native_tools, None, &[], None);
        if specs.iter().any(|s| s.name == ozgent_core::agents::HANDOFF_TOOL) {
            append_system(&mut messages, &ozgent_core::agents::handoff_prompt(&handoff));
        }
        let prompt = engine
            .render_prompt_full(&messages, resolved.thinking, resolved.reasoning_effort, &specs)
            .ok()?;
        engine.tokenize(&prompt).ok()
    };
    // Deliberately different lengths as well as different words, so a template
    // that pads or counts cannot make the two agree past the head.
    let (Some(a), Some(b)) = (render("a"), render("something else entirely")) else { return };
    let shared = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    match session.prewarm_commons(&a[..shared]) {
        Ok(0) => {}
        Ok(n) => tracing::info!("holding the {n} tokens every conversation starts with"),
        Err(e) => tracing::warn!("the shared prefix could not be held: {e}"),
    }
}

fn offer(tools: Option<&Tools>, request: &Request) -> Option<Vec<ozgent_core::ToolSpec>> {
    let Some(t) = tools.filter(|_| request.tools_enabled) else { return Some(Vec::new()) };
    let mut out: Vec<ozgent_core::ToolSpec> = match &request.native_tools {
        None => t.host.tools().to_vec(),
        Some(allowed) => {
            // A name that matches nothing is a caller mistake worth reporting:
            // silently dropping it would leave them believing a tool is
            // available when the model was never told about it.
            let available: Vec<&str> = t.host.tools().iter().map(|s| s.name.as_str()).collect();
            if let Some(unknown) = allowed.iter().find(|a| !available.iter().any(|n| n == &a.as_str())) {
                let _ = request.out.send(Event::Error {
                    message: format!(
                        "no built-in tool named {unknown:?}. Available: {}",
                        available.join(", ")
                    ),
                });
                return None;
            }
            t.host.tools().iter().filter(|s| allowed.iter().any(|a| a == &s.name)).cloned().collect()
        }
    };
    out.retain(|s| !request.tools_off.contains(&s.name));
    if !request.tools_off.iter().any(|n| n == ozgent_core::agents::HANDOFF_TOOL) {
        out.extend(ozgent_core::agents::handoff_spec(&request.handoff));
    }
    Some(out)
}

/// Put the tools and the grammar for this run on the session.
///
/// Returns false after reporting a grammar the model rejected.
fn install(
    engine: &Engine,
    session: &mut ozgent_llama::engine::Session<'_>,
    offered: &[ozgent_core::ToolSpec],
    grammar: Option<&str>,
    request: &Request,
) -> bool {
    // Constrain the body of a tool call the moment one starts, so a malformed
    // call is unreachable rather than emitted and then rejected.
    // Same reasoning as the terminal client: the gate's grammar is for the
    // JSON body ozgent's preamble describes, so it is withheld from a model
    // that was given its own format instead.
    if engine.template_handles_tools() {
        session.set_tools(&[]);
    } else {
        session.set_tools(offered);
    }
    // The session outlives the request, so a grammar left installed by an
    // earlier one would silently shape this reply. Cleared unconditionally
    // before anything else decides to set it.
    if let Err(e) = session.set_grammar(None) {
        let _ = request.out.send(Event::Error { message: e.to_string() });
        return false;
    }
    if let Some(grammar) = grammar {
        if let Err(e) = session.set_grammar(Some(grammar)) {
            let _ = request.out.send(Event::Error {
                message: format!("response_format produced a grammar the model rejected: {e}"),
            });
            return false;
        }
    }
    true
}

/// Join what this run needs to the system prompt, and mark the images.
///
/// Everything is appended to the existing system message rather than
/// replacing it, so a user's persona — or an agent's instructions — survive.
fn prepare(
    messages: &mut Vec<Message>,
    offered: &[ozgent_core::ToolSpec],
    native_tools: bool,
    projector: Option<&LoadedProjector<'_>>,
    images: &[ozgent_llama::mtmd::Media],
    observation: Option<&str>,
) {
    let append = |messages: &mut Vec<Message>, text: &str| match messages.first_mut() {
        Some(m) if m.role == ozgent_core::Role::System => {
            let existing = m.text_content();
            *m = Message::system(format!("{existing}\n\n{text}"));
        }
        _ => messages.insert(0, Message::system(text)),
    };
    if !images.is_empty() {
        append(messages, ozgent_tools::MEDIA_RULE);
    }
    if let (Some(p), false) = (projector, images.is_empty()) {
        if let Some(last) = messages.iter_mut().rev().find(|m| m.role == ozgent_core::Role::User) {
            let text = ozgent_llama::mtmd::with_markers(p.marker(), &last.text_content(), images.len());
            *last = Message::user(text);
        }
    }
    if !offered.is_empty() && !native_tools {
        append(messages, &ozgent_tools::tool_preamble(offered));
    }
    if let Some(observation) = observation {
        // The observation alone was not enough: the model read the image
        // correctly, then searched anyway. Naming the *only* reason a tool is
        // still warranted turns "should I search?" from an open question into
        // a test it can apply. Appended after the tool preamble, which would
        // otherwise be the last thing read.
        let note = format!(
            "You have already looked at the attached media. This is what is \
             actually in it:\n{observation}\n\nAnswer the user from that \
             observation. It is a complete and accurate record of the media, \
             so questions about what the media contains, shows, or looks like \
             are already answered — do not use a tool for them, and do not ask \
             the user to describe it.\n\nUse a tool only if the user asked for \
             something the media cannot contain: a current price, recent news, \
             today's weather, or another fact from the outside world."
        );
        append(messages, &note);
    }
}

/// Add to the system message, or start one.
fn append_system(messages: &mut Vec<Message>, text: &str) {
    match messages.first_mut() {
        Some(m) if m.role == ozgent_core::Role::System => {
            let existing = m.text_content();
            *m = Message::system(format!("{existing}\n\n{text}"));
        }
        _ => messages.insert(0, Message::system(text)),
    }
}

/// What one run of rounds produced, and what it cost.
struct Outcome {
    /// The visible reply of the last round.
    text: String,
    generated: u32,
    prompt_tokens: u32,
    elapsed_ms: u128,
    prompt_ms: u128,
    reused: usize,
    /// Tokens a drafter proposed, and how many of those the model kept.
    ///
    /// Collected by the engine since speculation was written and surfaced
    /// nowhere, which made the one setting that trades correctness-preserving
    /// work for latency impossible to tune: a drafter landing 10% of its
    /// guesses is costing time, and looked exactly like one landing 80%.
    proposed: usize,
    accepted: usize,
    stop: StopReason,
    /// The turn ended because the caller has a tool to run, which is a
    /// different thing from the model choosing to stop.
    handed_back: bool,
    /// Tool calls that actually ran.
    calls: usize,
    rounds: usize,
    /// The model handed the request to this agent, with this task.
    handoff: Option<(ozgent_core::Agent, String)>,
}

impl Default for Outcome {
    fn default() -> Self {
        Self {
            text: String::new(),
            generated: 0,
            prompt_tokens: 0,
            elapsed_ms: 0,
            prompt_ms: 0,
            reused: 0,
            proposed: 0,
            accepted: 0,
            stop: StopReason::EndOfText,
            handed_back: false,
            calls: 0,
            rounds: 0,
            handoff: None,
        }
    }
}

impl Outcome {
    fn absorb(&mut self, other: Outcome) {
        self.generated += other.generated;
        self.prompt_tokens += other.prompt_tokens;
        self.elapsed_ms += other.elapsed_ms;
        self.prompt_ms += other.prompt_ms;
        self.reused += other.reused;
        self.proposed += other.proposed;
        self.accepted += other.accepted;
        self.stop = other.stop;
        self.handed_back |= other.handed_back;
        self.calls += other.calls;
        self.rounds += other.rounds;
        self.text = other.text;
    }
}

/// Generate, run the tools asked for, and generate again, until the model
/// answers or its rounds run out.
#[allow(clippy::too_many_arguments)]
fn rounds(
    engine: &Engine,
    session: &mut ozgent_llama::engine::Session<'_>,
    resolved: &ozgent_core::options::Resolved,
    thinking: ThinkingMode,
    tools: Option<&Tools>,
    permissions: &Permissions,
    projector: Option<&LoadedProjector<'_>>,
    images: &[ozgent_llama::mtmd::Media],
    request: &Request,
    mut messages: Vec<Message>,
    offered: &[ozgent_core::ToolSpec],
    client_owned: &std::collections::HashSet<String>,
    max_rounds: usize,
    agent: Option<&ozgent_core::Agent>,
) -> anyhow::Result<Outcome> {
    let native_tools = engine.template_handles_tools();
    let mut out = Outcome::default();
    // One retry only, so a model that answers with silence twice cannot spin.
    let mut nudged = false;
    // Counts every call made in this run, so no two share an id.
    let mut next_call_id = 0usize;
    // Ids are prefixed per agent: two agents in one message each number from
    // zero, and the transcript pairs results with cards by id.
    let id_prefix = agent.map(|a| format!("{}_", a.name)).unwrap_or_default();
    let offered_names: Vec<String> = offered.iter().map(|s| s.name.clone()).collect();

    for round in 0..=max_rounds {
        out.rounds = round + 1;
        // Room is checked before every round after the first, because every
        // round adds tool results and nothing else bounds their sum: an agent
        // reading eight pages reached 30,899 tokens of a 32,768 window and
        // the whole answer was lost to a full cache.
        let cramped = round > 0 && !make_room_for_answer(engine, session, resolved, thinking, &mut messages, offered);
        let last = round == max_rounds || offered.is_empty() || cramped;
        // Out of rounds, with tools still on the table. Left there, the model
        // spends this turn on one more call, and the visible text before that
        // call — nothing — becomes the answer. Taking them away and asking for
        // an answer is what turns an exhausted loop into a reply.
        if last && round > 0 && !offered.is_empty() {
            session.set_tools(&[]);
            nudge(&mut messages, if cramped { OUT_OF_ROOM } else { OUT_OF_ROUNDS });
        }
        let media = (round == 0 && !images.is_empty())
            .then(|| projector.map(|p| (p, images, &request.images[..])))
            .flatten();
        // The tools stay in the rendered prompt even on the last round: taking
        // them out changes the system block, and the whole conversation would
        // be read again from the top. The instruction to answer does the job.
        let (reply, stats, reason, early) = generate(
            engine, session, resolved, thinking, &messages, media, request, offered, permissions, agent,
        )?;
        // A turn can span several generations; the client is told once, at the
        // end, with the totals. Sending `Done` per round ended the SSE stream
        // before the first tool had even run.
        out.generated += stats.generated_tokens as u32;
        out.prompt_tokens += stats.prompt_tokens as u32;
        out.elapsed_ms += stats.generation_ms;
        out.prompt_ms += stats.prompt_ms;
        out.reused += stats.reused_tokens;
        out.proposed += stats.proposed_drafts;
        out.accepted += stats.accepted_drafts;
        out.stop = reason;

        let mut parsed = ozgent_llama::extract_tool_calls(&reply);
        // The parser numbers calls from zero each time it runs, so round two
        // reissues `call_0`. Within a turn that has to be unique: it is how a
        // client pairs a result with the card that asked for it, and how the
        // conversation pairs a result with its call.
        for call in parsed.calls.iter_mut() {
            call.id = format!("{id_prefix}call_{next_call_id}");
            next_call_id += 1;
        }
        out.text = parsed.text.clone();
        tracing::debug!(calls = parsed.calls.len(), last, cramped, text = parsed.text.len(), "tool round {round}");

        // Reasoning, then nothing. A model can close its thought and stop
        // without either answering or calling anything, and the turn then
        // returns an empty string — which reads to the user as ozgent being
        // broken. Ask once for the answer it was about to give; if it stays
        // silent the second time, that is its reply and the loop ends.
        if !parsed.has_calls() && parsed.text.trim().is_empty() && !nudged {
            nudged = true;
            nudge(&mut messages, ANSWER_NOW);
            continue;
        }

        // Told to answer and called a tool anyway. Whatever came before the
        // call is a preamble — "I will now search…" — not an answer, and
        // ending here returns it after all that work. So it is asked once more
        // with the tools gone from the prompt altogether: the only thing left
        // to write is the answer. That costs reading the prompt again, once,
        // and only when the instruction was ignored.
        if last && parsed.has_calls() && !offered.is_empty() {
            tracing::info!("the model called a tool on its last round; asking again without tools");
            session.forbid_tool_calls(true);
            let retried = generate(
                engine, session, resolved, thinking, &messages, None, request, &[], permissions, agent,
            );
            session.forbid_tool_calls(false);
            let (reply, stats, reason, _) = retried?;
            out.generated += stats.generated_tokens as u32;
            out.prompt_tokens += stats.prompt_tokens as u32;
            out.elapsed_ms += stats.generation_ms;
            out.prompt_ms += stats.prompt_ms;
            out.reused += stats.reused_tokens;
            out.stop = reason;
            out.text = ozgent_llama::extract_tool_calls(&reply).text;
            break;
        }
        if last || !parsed.has_calls() {
            break;
        }
        // The visible part of the reply is whatever preceded the call — and
        // the call itself, which used to be dropped. The model was handed a
        // tool result with no record of having asked for it.
        messages.push(if native_tools {
            Message {
                role: ozgent_core::Role::Assistant,
                content: vec![ozgent_core::Part::Text { text: parsed.text.clone() }],
                thinking: None,
                tool_calls: parsed.calls.clone(),
                tool_call_id: None,
            }
        } else {
            // The fallback renderers read only `content`, so the call has to
            // be spelt out — in the format the preamble described.
            let calls: String = parsed
                .calls
                .iter()
                .map(|c| {
                    format!(
                        "<tool_call>{{\"name\": \"{}\", \"arguments\": {}}}</tool_call>",
                        c.name, c.arguments
                    )
                })
                .collect();
            Message::assistant(format!("{}{calls}", parsed.text))
        });

        // The model handing the request to an agent ends its part of the
        // turn: the agent answers from here, as if the user had named it.
        // A call naming an agent that does not exist is answered with the
        // list, so the model can correct it or answer itself.
        if agent.is_none() {
            if let Some(call) = parsed.calls.iter().find(|c| c.name == ozgent_core::agents::HANDOFF_TOOL) {
                match ozgent_core::agents::read_handoff(&call.arguments, &request.handoff) {
                    Ok((to, task)) => {
                        out.handoff = Some((to.clone(), task));
                        // Only the text before the call is the main model's.
                        messages.pop();
                        break;
                    }
                    Err(e) => {
                        let err = ozgent_tools::ToolCallError::Invalid { name: call.name.clone(), reason: e };
                        record(request, session, &mut messages, call, Err(err), 0);
                        continue;
                    }
                }
            }
        }

        // A call the caller owns ends the turn: this process has no code for
        // it, so the call is handed back to be run there. Checked before the
        // guard below, which would otherwise discard the call whenever the
        // server has no Python tools of its own — the common case for a client
        // that brought only its own.
        //
        // A batch mixing caller and server tools hands back only the caller's
        // and runs none of the server's: the caller replies with its results,
        // and the model reissues whatever it still wants.
        let handing_back: Vec<_> =
            parsed.calls.iter().filter(|c| client_owned.contains(&c.name)).collect();
        if !handing_back.is_empty() {
            for call in handing_back {
                let _ = request.out.send(Event::ClientToolCall {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
            }
            out.handed_back = true;
            break;
        }

        // A call to a tool this run was not offered is answered without
        // running anything. The host has the tool — it simply was not on the
        // table — and running it anyway would make an agent's list, or a
        // channel's allowlist, a suggestion rather than a limit.
        let reachable: Vec<bool> =
            parsed.calls.iter().map(|c| offered_names.contains(&c.name)).collect();

        // Nothing can run without a host. The calls still get results, so the
        // conversation stays well formed and the model learns why.
        let Some(t) = tools else {
            for call in &parsed.calls {
                let err = ozgent_tools::ToolCallError::NotOffered {
                    name: call.name.clone(),
                    offered: Vec::new(),
                };
                record(request, session, &mut messages, call, Err(err), 0);
            }
            continue;
        };

        // Independent calls run together. In series, a turn asking for three
        // searches paid three network round trips end to end, and the tool
        // timeout applied to each in turn rather than to the set.
        //
        // Permission first, one question at a time. Asking about a batch all
        // at once would put four modal cards on the page and make the user
        // answer them in whatever order they happened to be announced; asking
        // in the model's own order means the first refusal is about the first
        // call, which is the one the user is reading.
        let mut approved: Vec<Option<bool>> = Vec::with_capacity(parsed.calls.len());
        for (call, reachable) in parsed.calls.iter().zip(&reachable) {
            if !reachable {
                approved.push(None);
                continue;
            }
            // Already answered while the call was still being written. Asking
            // again would make the early prompt look like it did nothing.
            match early.as_ref().filter(|(name, _)| *name == call.name) {
                Some((_, allowed)) => approved.push(allowed.then_some(true)),
                None => approved.push(permit(permissions, t.host.get(&call.name), call, request, agent)),
            }
        }

        // Only the calls that survived are announced as running. A card that
        // appears and then turns into a refusal reads as a tool that failed.
        for (call, allowed) in parsed.calls.iter().zip(&approved) {
            if allowed.is_some() {
                let _ = request.out.send(Event::ToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
            }
        }
        out.calls += approved.iter().filter(|a| a.is_some()).count();
        // Captured before the futures borrow the request.
        let can_ask = request.can_ask;
        let outcomes = t.runtime.block_on(futures_util::future::join_all(
            parsed.calls.iter().zip(&approved).zip(&reachable).map(|((call, allowed), reachable)| {
                let offered_names = offered_names.clone();
                async move {
                    if !reachable {
                        let outcome = Err(ozgent_tools::ToolCallError::NotOffered {
                            name: call.name.clone(),
                            offered: offered_names,
                        });
                        report(request, call, &outcome, 0);
                        return (outcome, 0);
                    }
                    let Some(by_user) = *allowed else {
                        // "Declined" and "nobody was there" are different
                        // facts, and a model told the first apologises to
                        // somebody who never spoke. On a scheduled run there
                        // is no user in the conversation at all.
                        let name = call.name.clone();
                        let refused = if can_ask {
                            ozgent_tools::ToolCallError::Declined { name }
                        } else {
                            ozgent_tools::ToolCallError::Unattended { name }
                        };
                        let outcome = Err(refused);
                        report(request, call, &outcome, 0);
                        return (outcome, 0);
                    };
                    let started = std::time::Instant::now();
                    let outcome =
                        t.host.call_approved(&call.name, call.arguments.clone(), by_user).await;
                    let ms = started.elapsed().as_millis() as u64;
                    report(request, call, &outcome, ms);
                    (outcome, ms)
                }
            }),
        ));

        // Consumed in the model's original order, not completion order: the
        // results become the next prompt, so letting a race decide their order
        // would make the same turn produce different continuations.
        // Each was already reported to the client as it finished.
        for (call, (outcome, _)) in parsed.calls.iter().zip(outcomes) {
            feed(session, &mut messages, call, outcome);
        }
    }
    Ok(out)
}

/// Report one call's result to the client and hand it back to the model.
fn record(
    request: &Request,
    session: &ozgent_llama::engine::Session<'_>,
    messages: &mut Vec<Message>,
    call: &ozgent_core::ToolCall,
    outcome: Result<serde_json::Value, ozgent_tools::ToolCallError>,
    ms: u64,
) {
    report(request, call, &outcome, ms);
    feed(session, messages, call, outcome);
}

/// Tell the client how one call went.
///
/// Sent as each call finishes, not when the batch does. Held until the
/// slowest had finished, a one-second search sat behind a thirty-second
/// timeout with nothing on screen, and then every card resolved at once —
/// which is exactly what a frozen interface looks like.
fn report(
    request: &Request,
    call: &ozgent_core::ToolCall,
    outcome: &Result<serde_json::Value, ozgent_tools::ToolCallError>,
    ms: u64,
) {
    let (ok, summary, detail) = match outcome {
        Ok(value) => (true, summarise(value), value.clone()),
        // A tool failure is information the model can act on, not an error
        // for the user: it is fed back so the model can retry or explain,
        // exactly as the terminal client does.
        Err(e) => {
            let text = e.for_model();
            let summary = ozgent_tools::first_line(&text).to_string();
            (false, summary, serde_json::json!({ "error": text }))
        }
    };
    let _ = request.out.send(Event::ToolResult {
        id: call.id.clone(),
        name: call.name.clone(),
        ok,
        summary,
        ms,
        detail,
    });
}

/// Hand one call's result back to the model.
fn feed(
    session: &ozgent_llama::engine::Session<'_>,
    messages: &mut Vec<Message>,
    call: &ozgent_core::ToolCall,
    outcome: Result<serde_json::Value, ozgent_tools::ToolCallError>,
) {
    let payload = match outcome {
        Ok(value) => serde_json::to_string(&value).unwrap_or_default(),
        Err(e) => e.for_model(),
    };
    // A search asked for 20 results can return more text than the whole
    // context window holds. Unbounded, it crowds out the room the model needs
    // to answer and the reply stops mid-sentence — which reads like a crash
    // but is simply no space left.
    // The window the session actually opened, not the one that was asked for:
    // a request for more context than memory holds is granted at a smaller
    // size, and budgeting against the request would size tool results to a
    // window that does not exist.
    let budget = fit_budget(session.n_ctx());
    messages.push(Message::tool_result(call.id.clone(), fit(&payload, budget)));
}

/// Tokens a call is given to finish its arguments before it is asked about.
///
/// Long enough that a compact call — a command, a search — arrives whole and
/// is asked about in full; short enough that a file's `content` has barely
/// started, which is the point of asking early at all.
const ARGUMENT_GRACE: usize = 24;

/// How many times a turn may call tools before it must answer.
///
/// Each round is a full generation, so this is a latency budget as much as a
/// capability one. Four was not enough for an ordinary three-step request —
/// read a file, pick something out of it, read what that pointed at — which
/// spent every round and answered nothing.
pub const MAX_TOOL_ROUNDS: usize = 8;

/// Said to the model when it stops without answering.
///
/// Its own reasoning is the best prompt available here: it usually ends
/// mid-plan, and repeating that back is more likely to produce the answer
/// than a generic instruction to try again.
const ANSWER_NOW: &str = "\
You stopped without answering. Give the user your answer now, based on what \
you already know and whatever the tools have returned. If you cannot answer, \
say so plainly and say what is missing.";

/// Said to the model once its tool rounds are spent.
///
/// The instruction to admit a shortfall is deliberate. A model told only to
/// answer will invent the part it never managed to look up, which is worse
/// than the loop running out.
pub const OUT_OF_ROUNDS: &str = "\
You have no tool calls left. Answer now, using only what the tool results \
above actually contain. If they did not give you enough, say what you found \
and what is still missing — do not fill the gap with a guess.";

/// Put an instruction from ozgent where the model will read it next.
///
/// Onto the end of the last message rather than as a system message of its
/// own. Most templates take a system message only at the start, so a late one
/// is folded into the first — which changes the top of the prompt and makes
/// the whole conversation be read again, five seconds at ten thousand tokens,
/// to deliver one sentence.
fn nudge(messages: &mut Vec<Message>, text: &str) {
    match messages.last_mut() {
        Some(m) if matches!(m.role, ozgent_core::Role::Tool | ozgent_core::Role::User) => {
            let joined = format!("{}\n\n[{text}]", m.text_content());
            let mut replaced = m.clone();
            replaced.content = vec![ozgent_core::Part::Text { text: joined }];
            *m = replaced;
        }
        _ => messages.push(Message::user(format!("[{text}]"))),
    }
}

pub const OUT_OF_ROOM: &str = "\
The conversation is close to the limit of what you can hold, so there is no \
room for more tool calls. Answer now, using only what the tool results above \
actually contain, and say what is still missing.";

/// Where compaction starts, and where it stops, as shares of the window.
///
/// Three quarters leaves room for a round's reasoning, a call and a result of
/// the size [`fit_budget`] allows; stopping at a half keeps compaction from
/// running again on the very next round.
const COMPACT_ABOVE: (usize, usize) = (3, 4);
const COMPACT_TO: (usize, usize) = (1, 2);

/// The most of the window a prompt may take and still leave room to answer.
const FITS: (usize, usize) = (17, 20);

/// How much of an older tool result survives compaction, in characters.
const COMPACTED_CHARS: usize = 1200;

/// Keep the conversation inside the window before another round.
///
/// Older tool results are cut to their opening, oldest first: the model has
/// already read them and acted on them, and what it concluded is in its own
/// messages. The newest results are left whole — they are what this round is
/// about to use. Returns false when even that cannot make room, which is the
/// point to stop calling tools and answer.
fn make_room_for_answer(
    engine: &Engine,
    session: &ozgent_llama::engine::Session<'_>,
    resolved: &ozgent_core::options::Resolved,
    thinking: ThinkingMode,
    messages: &mut [Message],
    offered: &[ozgent_core::ToolSpec],
) -> bool {
    let window = session.n_ctx() as usize;
    let size = |messages: &[Message]| -> Option<usize> {
        let prompt = engine
            .render_prompt_full(messages, thinking, resolved.reasoning_effort, offered)
            .ok()?;
        engine.tokenize(&prompt).ok().map(|t| t.len())
    };
    let Some(mut used) = size(messages) else { return true };
    tracing::debug!("room check: {used} of {window} tokens");
    if used * COMPACT_ABOVE.1 <= window * COMPACT_ABOVE.0 {
        return true;
    }
    // The results of the latest round are the ones after the last assistant
    // message; everything before that is fair game.
    let newest = messages
        .iter()
        .rposition(|m| m.role == ozgent_core::Role::Assistant)
        .unwrap_or(messages.len());
    let mut compacted = 0;
    for i in 0..newest {
        if used * COMPACT_TO.1 <= window * COMPACT_TO.0 {
            break;
        }
        let m = &messages[i];
        if m.role != ozgent_core::Role::Tool {
            continue;
        }
        let text = m.text_content();
        if text.len() <= COMPACTED_CHARS {
            continue;
        }
        let cut = text.floor_char_boundary(COMPACTED_CHARS);
        let short = format!(
            "{}\n[… {} more characters, cut to make room. You have already read this result.]",
            &text[..cut],
            text.len() - cut
        );
        let id = m.tool_call_id.clone().unwrap_or_default();
        messages[i] = Message::tool_result(id, short);
        compacted += 1;
        if let Some(now) = size(messages) {
            used = now;
        }
    }
    // Still too big to answer in, which happens when one round's results
    // alone are: four parallel searches each within their own budget came to
    // more than a 12k window. Then the longest result is halved, whatever
    // round it came from, until the prompt fits with room for a reply. Every
    // result keeps its opening; nothing is dropped outright.
    while used * FITS.1 > window * FITS.0 {
        let longest = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == ozgent_core::Role::Tool)
            .map(|(i, m)| (i, m.text_content().len()))
            .filter(|(_, len)| *len > COMPACTED_CHARS / 2)
            .max_by_key(|(_, len)| *len);
        let Some((i, len)) = longest else { break };
        let text = messages[i].text_content();
        let cut = text.floor_char_boundary(len / 2);
        let short = format!("{}\n[… cut to make room.]", &text[..cut]);
        let id = messages[i].tool_call_id.clone().unwrap_or_default();
        messages[i] = Message::tool_result(id, short);
        compacted += 1;
        match size(messages) {
            Some(now) => used = now,
            None => break,
        }
    }
    if compacted > 0 {
        tracing::info!("the conversation neared its window; cut tool results {compacted} time(s), now {used} of {window} tokens");
    }
    used * COMPACT_ABOVE.1 <= window * COMPACT_ABOVE.0
}

/// Tokens the grounding pass may spend. Enough for a faithful description,
/// short enough that it costs a fraction of a second.
const GROUNDING_LIMIT: u32 = 200;

fn media_for<'a>(
    projector: Option<&'a LoadedProjector<'a>>,
    images: &'a [ozgent_llama::mtmd::Media],
) -> Option<(&'a LoadedProjector<'a>, &'a [ozgent_llama::mtmd::Media])> {
    projector.filter(|_| !images.is_empty()).map(|p| (p, images))
}

/// Ask the model, with no tools available, what is actually in the media.
///
/// Streamed to the client as reasoning: it is the model's own observation, not
/// its answer, and showing it makes the extra pass visible rather than a
/// mysterious pause. A failure here is not fatal — the turn simply proceeds
/// without the grounding.
fn ground(
    engine: &Engine,
    session: &mut ozgent_llama::engine::Session<'_>,
    // Deliberately not given the turn's options: this pass describes what is
    // in the image, so it runs with reasoning off and no effort budget
    // whatever the user asked of the answer itself.
    messages: &[Message],
    media: Option<(&LoadedProjector<'_>, &[ozgent_llama::mtmd::Media])>,
    request: &Request,
) -> Option<String> {
    let (projector, images) = media?;
    let question = messages
        .iter()
        .rev()
        .find(|m| m.role == ozgent_core::Role::User)
        .map(|m| m.text_content())
        .unwrap_or_default();

    let prompt = engine
        .render_prompt_with(
            &[
                Message::system(
                    "Describe exactly what is in the attached media: subjects, text,                      colours, layout. State only what you can actually see. Do not                      speculate about what it might be, and do not answer the user's                      question yet.",
                ),
                Message::user(question),
            ],
            ThinkingMode::Off,
            Default::default(),
        )
        .ok()?;

    let out = request.out.clone();
    let mut observed = String::new();
    let sources: Vec<ozgent_core::ImageSource> = request.images.clone();
    let result = session.generate_with_media(
        &prompt,
        Some((projector, images, &sources)),
        GROUNDING_LIMIT,
        |piece| {
            observed.push_str(piece);
            out.send(Event::Thinking { text: piece.to_string() }).is_ok()
        },
    );

    match result {
        Ok(_) if !observed.trim().is_empty() => Some(observed.trim().to_string()),
        Ok(_) => None,
        Err(e) => {
            tracing::warn!("grounding pass failed, continuing without it: {e}");
            None
        }
    }
}

/// A human sentence about a tool result, for the collapsed row.
///
/// Shape-aware rather than a JSON prefix: `{"category":"news","provider":...`
/// tells a reader nothing, while "15 news results from brave" tells them
/// whether the call did what they wanted. Falls back to a trimmed first line
/// for tools whose output has no shape worth naming.
fn summarise(value: &serde_json::Value) -> String {
    if let Some(known) = ozgent_tools::summary::describe(value) {
        return known;
    }
    if let Some(results) = value.get("results").and_then(|r| r.as_array()) {
        let mut out = format!(
            "{} result{}",
            results.len(),
            if results.len() == 1 { "" } else { "s" }
        );
        if let Some(category) = value.get("category").and_then(|c| c.as_str()) {
            out = format!("{} {out}", category);
        }
        if let Some(provider) = value.get("provider").and_then(|p| p.as_str()) {
            out.push_str(&format!(" from {provider}"));
        }
        return out;
    }
    if let Some(path) = value.get("path").and_then(|p| p.as_str()) {
        if let Some(total) = value.get("total_lines").and_then(|l| l.as_u64()) {
            let shown = total - value.get("omitted").and_then(|o| o.as_u64()).unwrap_or(0);
            let name = path.rsplit('/').next().unwrap_or(path);
            return format!("{name}: {shown} of {total} lines");
        }
        if let Some(written) = value.get("lines_written").and_then(|l| l.as_u64()) {
            let name = path.rsplit('/').next().unwrap_or(path);
            return format!("wrote {written} lines to {name}");
        }
    }
    if let Some(text) = value.as_str() {
        return trim_to(text, 110);
    }
    trim_to(&serde_json::to_string(value).unwrap_or_default(), 110)
}

/// Characters of tool output a single call may contribute to the prompt.
///
/// A quarter of the window, at roughly four characters per token. The model
/// still needs room for the conversation and for its own answer.
fn fit_budget(context_length: u32) -> usize {
    (context_length as usize / 4) * 4
}

/// Trim a tool payload to `budget` characters without producing invalid JSON.
///
/// Whole results are dropped from the end where the shape allows it, because a
/// list of five complete results is far more useful to a model than eight
/// results with the last one cut in half.
fn fit(payload: &str, budget: usize) -> String {
    if payload.len() <= budget {
        return payload.to_string();
    }
    if let Ok(mut value) = serde_json::from_str::<serde_json::Value>(payload) {
        let mut count = value
            .get("results")
            .and_then(|r| r.as_array())
            .map(Vec::len)
            .unwrap_or(0);
        while count > 1 {
            match value.get_mut("results").and_then(|r| r.as_array_mut()) {
                Some(results) => {
                    results.pop();
                    count = results.len();
                }
                None => break,
            }
            let candidate = serde_json::to_string(&value).unwrap_or_default();
            if candidate.len() <= budget {
                return candidate;
            }
        }
    }
    let head: String = payload.chars().take(budget.saturating_sub(20)).collect();
    format!("{head} …[truncated]")
}

fn trim_to(text: &str, max: usize) -> String {
    let line = text.lines().next().unwrap_or("");
    if line.chars().count() <= max {
        return line.to_string();
    }
    let head: String = line.chars().take(max.saturating_sub(3)).collect();
    format!("{head}...")
}

fn generate(
    engine: &Engine,
    session: &mut ozgent_llama::engine::Session<'_>,
    resolved: &ozgent_core::options::Resolved,
    thinking: ThinkingMode,
    messages: &[Message],
    media: Option<(&LoadedProjector<'_>, &[ozgent_llama::mtmd::Media], &[ozgent_core::ImageSource])>,
    request: &Request,
    tools: &[ozgent_core::ToolSpec],
    permissions: &Permissions,
    agent: Option<&ozgent_core::Agent>,
) -> anyhow::Result<(String, ozgent_llama::engine::Stats, StopReason, Option<(String, bool)>)> {
    let prompt =
        engine.render_prompt_full(messages, thinking, resolved.reasoning_effort, tools)?;
    let mut filter = ThinkingFilter::new(thinking);
    if let Some(close) = Engine::stream_starts_inside(&prompt) {
        filter = filter.starting_inside(close);
    }
    // One gate per stream, not one shared between them. The gate latches shut
    // the moment a tool call begins — correct for the answer, where the call is
    // the last thing emitted, but fatal if shared: reasoning produced after the
    // call would be withheld for the rest of the round and never shown.
    let mut answer_gate = ozgent_llama::toolcall::StreamGate::new();
    let mut thinking_gate = ozgent_llama::toolcall::StreamGate::new();
    // Kept so an unclosed reasoning block can be handed back as the reply.
    let mut reasoning = String::new();
    let mut answered = false;
    let out = request.out.clone();
    let mut raw = String::new();
    // Set once the call being written has been named, so it is announced once
    // rather than on every token.
    let mut announced = false;
    let mut since_named = 0usize;
    // The answer given before the content existed, so the tool loop does not
    // ask a second time about the same call.
    let mut early: Option<(String, bool)> = None;

    let limit = request.max_tokens.unwrap_or(resolved.max_tokens);
    let (stats, reason) = session.generate_with_media(&prompt, media, limit, |piece| {
        raw.push_str(piece);
        for chunk in filter.push(piece) {
            let event = match chunk {
                Chunk::Thinking(text) => {
                    let text = thinking_gate.push(&text);
                    reasoning.push_str(&text);
                    Event::Thinking { text }
                }
                Chunk::Answer(text) => {
                    let text = answer_gate.push(&text);
                    if !text.is_empty() {
                        answered = true;
                    }
                    Event::Answer { text }
                }
            };
            let empty = match &event {
                Event::Thinking { text } | Event::Answer { text } => text.is_empty(),
                _ => false,
            };
            if empty {
                continue;
            }
            // A closed receiver means the browser navigated away; stopping
            // frees the GPU instead of generating into nothing.
            if out.send(event).is_err() {
                return false;
            }
        }

        // The name is readable from the call's first few tokens. Announcing it
        // here is what fills the silence while the arguments are generated.
        if !announced && answer_gate.suppressing() {
            if let Some(name) = ozgent_llama::toolcall::pending_name(&raw) {
                announced = true;
                if out.send(Event::ToolCallStarted { name: name.to_string() }).is_err() {
                    return false;
                }
            }
        }

        // Ask before the content is generated, not after. A model writing a
        // file spends the whole call on `content`, so waiting for the finished
        // call means asking a minute after the decision was made — and a
        // refusal then has already paid for every token of it. Asked here, a
        // refusal stops generation on the spot.
        if announced {
            since_named += 1;
            if early.is_none() && since_named >= ARGUMENT_GRACE {
                let name =
                    ozgent_llama::toolcall::pending_name(&raw).unwrap_or_default().to_string();
                let mut arguments = ozgent_llama::toolcall::pending_arguments(&raw);
                // Said plainly, because the question is being asked before the
                // answer to "what exactly" exists.
                if !ozgent_llama::toolcall::call_is_closed(&raw) {
                    arguments.insert(
                        "…".into(),
                        serde_json::Value::String("still being written".into()),
                    );
                }
                let call = ozgent_core::ToolCall {
                    id: format!("early-{name}"),
                    name: name.clone(),
                    arguments: serde_json::Value::Object(arguments),
                };
                let spec = tools.iter().find(|t| t.name == name);
                // A call to something this run was not offered is not asked
                // about: there is nothing to allow. Generation stops, and the
                // round reports it as unavailable.
                let verdict = match spec {
                    Some(_) => permit(permissions, spec, &call, request, agent),
                    None => None,
                };
                match verdict {
                    Some(_) => early = Some((name, true)),
                    None => {
                        early = Some((name, false));
                        return false;
                    }
                }
            }
        }
        true
    })?;

    for chunk in filter.finish() {
        if let Chunk::Answer(text) = chunk {
            let text = answer_gate.push(&text);
            if !text.is_empty() {
                answered = true;
                let _ = out.send(Event::Answer { text });
            }
        }
    }

    // A reasoning block the model never closed was not a reasoning block: it
    // ended its turn still inside `<think>`, so what it wrote there is its
    // reply. Left as reasoning, an OpenAI-compatible client gets a response
    // with no content at all, and the browser shows an empty message.
    if !answered && filter.is_thinking() && !reasoning.trim().is_empty() {
        let _ = out.send(Event::Answer { text: std::mem::take(&mut reasoning) });
    }

    Ok((raw, stats, reason, early))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The browser switches on `type` and reads these field names directly, so
    /// the wire shape is a contract with `assets/app.js`, not an implementation
    /// detail. Renaming a variant without updating the client would fail
    /// silently as a chat that streams nothing.
    #[test]
    fn events_serialise_with_the_tags_the_client_switches_on() {
        let cases = vec![
            (
                Event::Ready { model: "m:Q4".into(), context: 4096 },
                serde_json::json!({"type": "ready", "model": "m:Q4", "context": 4096}),
            ),
            (
                Event::Thinking { text: "hm".into() },
                serde_json::json!({"type": "thinking", "text": "hm"}),
            ),
            (
                Event::Answer { text: "hi".into() },
                serde_json::json!({"type": "answer", "text": "hi"}),
            ),
            (
                Event::Error { message: "boom".into() },
                serde_json::json!({"type": "error", "message": "boom"}),
            ),
        ];
        for (event, expected) in cases {
            assert_eq!(serde_json::to_value(&event).unwrap(), expected);
        }
    }

    #[test]
    fn tool_events_carry_what_the_transcript_row_shows() {
        let call = serde_json::to_value(Event::ToolCall {
            id: "call_1".into(),
            name: "web_search".into(),
            arguments: serde_json::json!({ "query": "nse" }),
        })
        .unwrap();
        assert_eq!(call["type"], "tool_call");
        assert_eq!(call["name"], "web_search");
        assert_eq!(call["arguments"]["query"], "nse");

        let result = serde_json::to_value(Event::ToolResult {
            id: "call_1".into(),
            name: "web_search".into(),
            ok: false,
            summary: "rate limited".into(),
            ms: 42,
            detail: serde_json::json!({ "error": "rate limited" }),
        })
        .unwrap();
        assert_eq!(result["type"], "tool_result");
        assert_eq!(result["ok"], false);
        assert_eq!(result["ms"], 42);

        // A batch is announced in full before any of it is awaited, so the
        // client can only pair a result with its card by id. Without this the
        // first result closed the last card and every other card stayed
        // running for good — which is what a model issuing several calls per
        // round, as Ling does, produces on every turn.
        assert_eq!(call["id"], "call_1");
        assert_eq!(result["id"], call["id"]);
    }

    #[test]
    fn the_done_event_carries_what_the_status_line_shows() {
        let json = serde_json::to_value(Event::Done {
            generated: 120,
            tokens_per_second: 57.5,
            reused: 64,
            stop: "EndOfText".into(),
            prompt: 40,
            context: 0,
                        prompt_ms: 210,
            drafted: 30,
            accepted: 21,
        })
        .unwrap();
        assert_eq!(json["type"], "done");
        assert_eq!(json["generated"], 120);
        assert_eq!(json["reused"], 64);
        assert_eq!(json["prompt"], 40, "the API reports prompt tokens from this");
        assert_eq!(json["drafted"], 30);
        assert_eq!(json["accepted"], 21);
        // The client calls .toFixed(1) on this, so it must be a number.
        assert!(json["tokens_per_second"].is_f64());
    }

    #[test]
    fn a_turn_with_nothing_drafting_does_not_report_drafting_nothing() {
        // Zeroes here would read as "speculation ran and landed none of it",
        // which is the opposite of what an absent drafter means.
        let json = serde_json::to_value(Event::Done {
            generated: 10,
            tokens_per_second: 20.0,
            reused: 0,
            stop: "EndOfText".into(),
            prompt: 5,
            context: 0,
                        prompt_ms: 10,
            drafted: 0,
            accepted: 0,
        })
        .unwrap();
        assert!(json.get("drafted").is_none(), "{json}");
        assert!(json.get("accepted").is_none(), "{json}");
    }

    #[test]
    fn a_large_tool_result_drops_whole_results_rather_than_cutting_one_in_half() {
        // Five complete results beat eight with the last one truncated, and a
        // half-written JSON object is worse than either.
        let results: Vec<serde_json::Value> = (0..20)
            .map(|i| serde_json::json!({ "title": format!("result {i}"), "snippet": "x".repeat(300) }))
            .collect();
        let payload = serde_json::json!({ "provider": "brave", "results": results }).to_string();

        let out = fit(&payload, 2000);
        assert!(out.len() <= 2000, "still {} bytes", out.len());
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("what is fed back must stay valid JSON");
        let kept = parsed["results"].as_array().unwrap().len();
        assert!(kept > 0 && kept < 20, "kept {kept} of 20");
        assert_eq!(parsed["results"][0]["title"], "result 0", "the best results are kept");
    }

    #[test]
    fn a_payload_within_budget_is_untouched() {
        let small = serde_json::json!({ "celsius": -3 }).to_string();
        assert_eq!(fit(&small, 2000), small);
    }

    #[test]
    fn an_unshaped_payload_is_truncated_with_a_marker() {
        let blob = "y".repeat(5000);
        let out = fit(&blob, 500);
        assert!(out.len() <= 500, "got {}", out.len());
        assert!(out.ends_with("…[truncated]"));
    }

    #[test]
    fn the_budget_leaves_room_for_the_conversation_and_the_answer() {
        // A tool must never be allowed to fill the window on its own.
        assert!(fit_budget(4096) < 4096 * 4 / 2);
        assert!(fit_budget(4096) > 0);
    }

    #[test]
    fn a_search_result_is_summarised_by_what_it_found() {
        // `{"category":"news","provider":"brave",...` tells a reader nothing.
        let value = serde_json::json!({
            "provider": "brave",
            "category": "news",
            "query": "nse bse",
            "results": [{"title": "a"}, {"title": "b"}, {"title": "c"}],
        });
        assert_eq!(summarise(&value), "news 3 results from brave");
    }

    #[test]
    fn one_result_is_not_pluralised() {
        let value = serde_json::json!({ "provider": "tavily", "results": [{"title": "a"}] });
        assert_eq!(summarise(&value), "1 result from tavily");
    }

    #[test]
    fn a_file_read_is_summarised_by_how_much_of_it_came_back() {
        let value = serde_json::json!({
            "path": "/home/u/project/engine.rs",
            "total_lines": 1091,
            "omitted": 700,
        });
        assert_eq!(summarise(&value), "engine.rs: 391 of 1091 lines");

        let written = serde_json::json!({ "path": "/tmp/notes.md", "lines_written": 12 });
        assert_eq!(summarise(&written), "wrote 12 lines to notes.md");
    }

    #[test]
    fn an_unrecognised_shape_falls_back_to_a_bounded_first_line() {
        let long = serde_json::Value::String("x".repeat(400));
        let out = summarise(&long);
        assert!(out.chars().count() <= 110, "got {} chars", out.chars().count());
        assert!(out.ends_with("..."));
        assert_eq!(summarise(&serde_json::json!("first\nsecond")), "first");
    }

    /// A worker over an empty ozgent directory: no models are installed, so
    /// nothing can ever be loaded and no thread is ever started.
    fn empty_worker(root: &std::path::Path) -> Worker {
        let config: SharedConfig = Arc::new(std::sync::Mutex::new(Config::default()));
        Worker::spawn(
            Paths::with_root(root),
            Arc::clone(&config),
            Arc::new(std::sync::Mutex::new(None)),
            Permissions {
                pending: Arc::new(crate::permission::Pending::default()),
                grants: Arc::new(std::sync::Mutex::new(ozgent_core::Grants::default())),
                config,
            },
            Arc::new(ozgent_core::Options::default()),
        )
    }

    fn request(model: &str, out: tokio::sync::mpsc::UnboundedSender<Event>) -> Request {
        Request {
            // No stream is being watched in a test.
            can_ask: false,
            grant: None,
            model: model.into(),
            messages: Vec::new(),
            thinking: None,
            max_tokens: None,
            tools_enabled: true,
            native_tools: None,
            client_tools: Vec::new(),
            response_grammar: None,
            overrides: None,
            images: Vec::new(),
            agents: Vec::new(),
            tools_off: Vec::new(),
            handoff: Vec::new(),
            out,
        }
    }

    #[test]
    fn a_model_that_is_not_installed_is_reported_on_the_turns_own_stream() {
        // Not as an error from `submit`: by the time a turn is submitted the
        // handler has already started streaming and has no way to say
        // anything except through the stream itself.
        let dir = std::env::temp_dir().join(format!("ozgent-worker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let worker = empty_worker(&dir);

        let (out, mut events) = tokio::sync::mpsc::unbounded_channel();
        assert!(worker.submit(request("not-installed", out)).is_ok());

        match events.try_recv() {
            Ok(Event::Error { message }) => assert!(!message.is_empty(), "must say why"),
            other => panic!("expected an error event, got {other:?}"),
        }
        assert_eq!(worker.loaded(), Vec::<String>::new(), "and nothing was loaded");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_worker_starts_with_nothing_loaded() {
        // The pool is empty until something is asked for. A daemon that has
        // answered nothing holds no model at all.
        let dir = std::env::temp_dir().join(format!("ozgent-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let worker = empty_worker(&dir);
        assert!(worker.loaded().is_empty());
        // Unloading nothing is not an error.
        worker.unload();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
