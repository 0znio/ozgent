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
use ozgent_tools::ToolHost;
use std::sync::Arc;
use ozgent_llama::engine::{Engine, StopReason};
use ozgent_llama::thinking::{Chunk, ThinkingFilter};
use std::sync::mpsc::{Receiver, Sender, channel};
use tokio::sync::mpsc::UnboundedSender;

/// A unit of work for the inference thread.
pub enum Job {
    Generate(Box<Request>),
    /// Embed texts and send the vectors straight back.
    Embed {
        texts: Vec<String>,
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
    pub out: UnboundedSender<Event>,
}

/// Streamed back to the HTTP handler, which forwards it as SSE.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    /// Model load finished; generation is about to start.
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
        /// Prompt tokens actually processed, for API usage accounting.
        prompt: u32,
        /// Time spent on prefill. Reported so a caller can see prefix reuse
        /// working: a turn that reuses its prefix pays almost nothing here.
        prompt_ms: u64,
    },
    Error { message: String },
}

/// Handle to the inference thread.
#[derive(Clone)]
pub struct Worker {
    tx: Sender<Job>,
}

/// The projector type, aliased so the lifetime stays readable in signatures.
pub type LoadedProjector<'a> = ozgent_llama::mtmd::Projector<'a>;

/// What the inference thread needs to run tools.
///
/// The host is async and this thread is not, so calls are driven through a
/// runtime handle rather than blocking the executor that serves HTTP.
#[derive(Clone)]
pub struct Tools {
    pub host: Arc<ToolHost>,
    pub runtime: tokio::runtime::Handle,
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

impl Worker {
    /// Start the thread. It lives for the process.
    pub fn spawn(paths: Paths, config: SharedConfig, tools: SharedTools) -> Self {
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("ozgent-inference".into())
            .spawn(move || run(paths, config, tools, rx))
            .expect("spawning the inference thread");
        Self { tx }
    }

    /// Queue a generation. Returns an error only if the thread has died.
    pub fn submit(&self, request: Request) -> Result<(), &'static str> {
        self.tx
            .send(Job::Generate(Box::new(request)))
            .map_err(|_| "the inference thread is not running")
    }

    /// Embed texts on the inference thread.
    ///
    /// Synchronous by design: an embedding request has nothing to stream, and
    /// the caller wants the vectors or an error, not a channel.
    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.tx
            .send(Job::Embed { texts, reply: tx })
            .map_err(|_| "the inference thread is not running".to_string())?;
        rx.recv().map_err(|_| "the inference thread stopped".to_string())?
    }

    pub fn unload(&self) {
        let _ = self.tx.send(Job::Unload);
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

fn run(paths: Paths, config: SharedConfig, tools: SharedTools, rx: Receiver<Job>) {
    let mut pending: Option<Box<Request>> = None;
    // Held across model loads: the embedding model is independent of whichever
    // chat model happens to be resident.
    let mut embedder: Option<ozgent_llama::embed::Embedder> = None;
    loop {
        // Either carry over the job that forced a model switch, or wait.
        let request = match pending.take() {
            Some(r) => r,
            None => match rx.recv() {
                Ok(Job::Generate(r)) => r,
                Ok(Job::Embed { texts, reply }) => {
                    let _ = reply.send(serve_embeddings(&paths, &snapshot(&config), &mut embedder, texts));
                    continue;
                }
                Ok(Job::Unload) => continue, // nothing loaded
                Err(_) => return,            // all senders dropped
            },
        };

        // Loading and the session that borrows it both live in this scope, so
        // the borrow checker is satisfied without any self-referential trick.
        match serve_model(&paths, &config, &tools, &mut embedder, request, &rx) {
            Ok(next) => pending = next,
            Err(e) => tracing::error!("inference thread: {e}"),
        }
    }
}

/// Load one model and serve jobs against it until a different one is asked
/// for, returning that job so the caller can load its model.
fn serve_model(
    paths: &Paths,
    shared: &SharedConfig,
    tools: &SharedTools,
    // Threaded through rather than rebuilt: an embedding request that arrives
    // mid-conversation should not reload the model it already has.
    embedder: &mut Option<ozgent_llama::embed::Embedder>,
    first: Box<Request>,
    rx: &Receiver<Job>,
) -> anyhow::Result<Option<Box<Request>>> {
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
    let base = config.options_for(&found.model.to_string()).merge(&found.manifest.defaults);
    let resolved = base
        .clone()
        .merge(first.overrides.as_ref().unwrap_or(&Default::default()))
        .resolve();

    let weights = found.manifest.primary_weights(&found.dir);
    let engine = match Engine::load(&weights, &resolved) {
        Ok(e) => e,
        Err(e) => {
            let _ = first.out.send(Event::Error {
                message: format!("loading {}: {e}", found.model),
            });
            return Ok(None);
        }
    };
    let mut session = engine.session(&resolved)?;
    // Loaded on the first turn that needs it and kept: it costs VRAM, but a
    // conversation with one image usually has more.
    let mut projector: Option<crate::worker::LoadedProjector<'_>> = None;
    let mmproj = found.manifest.projector_path(&found.dir);

    let mut request = first;
    loop {
        // Each request brings its own sampling. Re-resolving per turn is what
        // keeps two clients on one model from inheriting each other's
        // temperature, seed and reasoning budget.
        let live = snapshot(shared);
        let base = live.options_for(&found.model.to_string()).merge(&found.manifest.defaults);
        let per_turn = base
            .clone()
            .merge(request.overrides.as_ref().unwrap_or(&Default::default()))
            .resolve();

        // Context length, layer placement and the rest are fixed when the
        // weights load. Returning the request sends it back to be served
        // against a freshly loaded model, which is what "applies on your next
        // message" has to mean if it is to be true.
        if resolved.needs_reload(&per_turn) {
            tracing::info!("a load-time setting changed; reloading {}", found.model);
            return Ok(Some(request));
        }
        session.set_options(&per_turn);
        let thinking = request.thinking.unwrap_or(per_turn.thinking);
        let _ = request.out.send(Event::Ready {
            model: found.model.to_string(),
            context: session.n_ctx(),
        });

        if !request.images.is_empty() && projector.is_none() {
            match &mmproj {
                Some(path) => match engine.projector(path, &resolved) {
                    Ok(p) => projector = Some(p),
                    Err(e) => {
                        let _ = request.out.send(Event::Error {
                            message: format!("loading the vision projector: {e}"),
                        });
                    }
                },
                None => {
                    let _ = request.out.send(Event::Error {
                        message: format!("{} cannot see images: no vision projector installed", found.model),
                    });
                }
            }
        }

        if let Err(e) = turn(
            &engine,
            &mut session,
            &per_turn,
            thinking,
            tools,
            projector.as_ref(),
            &request,
        ) {
            let _ = request.out.send(Event::Error { message: e.to_string() });
        }
        // Dropping the sender ends the SSE stream for this request.
        drop(request);

        // Keep waiting until something to generate arrives: an embedding
        // request is answered here and does not end the turn loop.
        request = loop {
            match rx.recv() {
                Ok(Job::Generate(r)) => break r,
                Ok(Job::Embed { texts, reply }) => {
                    let _ = reply.send(serve_embeddings(paths, &snapshot(shared), embedder, texts));
                }
                // Unloading means returning so the engine is dropped with the scope.
                Ok(Job::Unload) => return Ok(None),
                Err(_) => return Ok(None),
            }
        };
        if request.model != wanted {
            return Ok(Some(request));
        }
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
) -> Result<Vec<Vec<f32>>, String> {
    if slot.is_none() {
        *slot = Some(load_embedder(paths, config)?);
    }
    let embedder = slot.as_ref().expect("just loaded");
    embedder.embed_batch(&texts).map_err(|e| e.to_string())
}

/// Load the configured embedding model, once.
///
/// Lazily, because most sessions never ask for an embedding and the model is
/// another half-gigabyte of VRAM that a chat has better uses for.
fn load_embedder(
    paths: &Paths,
    config: &Config,
) -> Result<ozgent_llama::embed::Embedder, String> {
    let Some(name) = config.embedding.model.as_deref() else {
        return Err(embedding_unavailable());
    };
    let found = ozgent_core::resolve(paths, name)
        .map_err(|e| format!("embedding model {name:?}: {e}"))?;
    let weights = found.manifest.primary_weights(&found.dir);
    ozgent_llama::embed::Embedder::load(&weights, 99)
        .map_err(|e| format!("loading embedding model {name:?}: {e}"))
}

fn embedding_unavailable() -> String {
    "no embedding model is configured. Install one and set \
     [embedding] model = \"<name>\" in config.toml"
        .to_string()
}

/// One user turn: generate, run any tools the model asks for, generate again.
///
/// Mirrors the terminal client's loop so a conversation behaves the same in
/// both front ends. Bounded, because a model that keeps calling tools would
/// otherwise never produce an answer.
fn turn(
    engine: &Engine,
    session: &mut ozgent_llama::engine::Session<'_>,
    resolved: &ozgent_core::options::Resolved,
    thinking: ThinkingMode,
    tools: &SharedTools,
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

    let media_turn = !images.is_empty();
    // Read per turn, so switching the search provider or disabling a tool in
    // Settings reaches the very next message rather than the next restart.
    let tools = current_tools(tools);
    let mut offered: Vec<ozgent_core::ToolSpec> = match &tools {
        Some(t) if request.tools_enabled => match &request.native_tools {
            None => t.host.tools().to_vec(),
            Some(allowed) => {
                // A name that matches nothing is a caller mistake worth
                // reporting: silently dropping it would leave them believing a
                // tool is available when the model was never told about it.
                let available: Vec<&str> =
                    t.host.tools().iter().map(|s| s.name.as_str()).collect();
                if let Some(unknown) =
                    allowed.iter().find(|a| !available.iter().any(|n| n == &a.as_str()))
                {
                    let _ = request.out.send(Event::Error {
                        message: format!(
                            "no built-in tool named {unknown:?}. Available: {}",
                            available.join(", ")
                        ),
                    });
                    return Ok(());
                }
                t.host
                    .tools()
                    .iter()
                    .filter(|s| allowed.iter().any(|a| a == &s.name))
                    .cloned()
                    .collect()
            }
        },
        _ => Vec::new(),
    };
    // Caller-supplied tools are additive: a request may use the server's
    // Python tools, its own, or both. A name collision resolves in favour of
    // the server's, because that is the one this process can actually run.
    let server_names: std::collections::HashSet<String> =
        offered.iter().map(|s| s.name.clone()).collect();
    let client_owned: std::collections::HashSet<String> = request
        .client_tools
        .iter()
        .map(|s| s.name.clone())
        .filter(|n| !server_names.contains(n))
        .collect();
    for spec in &request.client_tools {
        if client_owned.contains(&spec.name) {
            offered.push(spec.clone());
        }
    }

    // Constrain the body of a tool call the moment one starts, so a malformed
    // call is unreachable rather than emitted and then rejected.
    // Same reasoning as the terminal client: the gate's grammar is for the
    // JSON body ozgent's preamble describes, so it is withheld from a model
    // that was given its own format instead.
    if engine.template_handles_tools() {
        session.set_tools(&[]);
    } else {
        session.set_tools(&offered);
    }

    // The session outlives the request, so a grammar left installed by an
    // earlier one would silently shape this reply. Cleared unconditionally
    // before anything else decides to set it.
    if let Err(e) = session.set_grammar(None) {
        let _ = request.out.send(Event::Error { message: e.to_string() });
        return Ok(());
    }
    if let Some(grammar) = &request.response_grammar {
        if let Err(e) = session.set_grammar(Some(grammar)) {
            let _ = request.out.send(Event::Error {
                message: format!("response_format produced a grammar the model rejected: {e}"),
            });
            return Ok(());
        }
    }

    let mut messages = request.messages.clone();
    if !images.is_empty() {
        // Joined to the system prompt rather than replacing it, so a user's
        // persona survives.
        match messages.first_mut() {
            Some(m) if m.role == ozgent_core::Role::System => {
                let existing = m.text_content();
                *m = Message::system(format!("{existing}\n\n{}", ozgent_tools::MEDIA_RULE));
            }
            _ => messages.insert(0, Message::system(ozgent_tools::MEDIA_RULE)),
        }
    }
    if let (Some(p), false) = (projector, images.is_empty()) {
        if let Some(last) = messages.iter_mut().rev().find(|m| m.role == ozgent_core::Role::User) {
            let text = ozgent_llama::mtmd::with_markers(p.marker(), &last.text_content(), images.len());
            *last = Message::user(text);
        }
    }
    // A template with its own tools block tells the model the call format it
    // was trained on; ozgent's generic description would be a second one, in
    // a different syntax, and the model splits the difference.
    let native_tools = engine.template_handles_tools();

    if !offered.is_empty() && !native_tools {
        // The tool description joins the system prompt rather than replacing
        // it, so a user-set persona survives.
        let preamble = ozgent_tools::tool_preamble(&offered);
        match messages.first_mut() {
            Some(m) if m.role == ozgent_core::Role::System => {
                let existing = m.text_content();
                *m = Message::system(format!("{existing}\n\n{preamble}"));
            }
            _ => messages.insert(0, Message::system(preamble)),
        }
    }

    if media_turn {
        if let Some(observation) = ground(engine, session, &messages, media_for(projector, &images), request) {
            // The observation alone was not enough: the model read the image
            // correctly, then searched anyway. Naming the *only* reason a tool
            // is still warranted turns "should I search?" from an open question
            // into a test it can apply.
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
            match messages.first_mut() {
                Some(m) if m.role == ozgent_core::Role::System => {
                    let existing = m.text_content();
                    *m = Message::system(format!("{existing}\n\n{note}"));
                }
                _ => messages.insert(0, Message::system(note)),
            }
            let _ = &observation;
        }
    }


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
    let max_rounds = MAX_TOOL_ROUNDS;
    let mut generated = 0u32;
    let mut prompt_tokens = 0u32;
    let mut elapsed_ms = 0u128;
    let mut prompt_ms = 0u128;
    let mut reused = 0usize;
    let mut stop = StopReason::EndOfText;
    // The turn ended because the caller has a tool to run, which is a
    // different thing from the model choosing to stop.
    let mut handed_back = false;
    // One retry only, so a model that answers with silence twice cannot spin.
    let mut nudged = false;
    // Counts every call made in this turn, so no two share an id.
    let mut next_call_id = 0usize;

    for round in 0..=max_rounds {
        let last = round == max_rounds || offered.is_empty();
        // Out of rounds, with tools still on the table. Left there, the model
        // spends this turn on one more call, and the visible text before that
        // call — nothing — becomes the answer. Taking them away and asking for
        // an answer is what turns an exhausted loop into a reply.
        if last && round > 0 && !offered.is_empty() {
            session.set_tools(&[]);
            messages.push(Message::system(OUT_OF_ROUNDS));
        }
        let media = (round == 0 && !images.is_empty())
            .then(|| projector.map(|p| (p, &images[..], &request.images[..])))
            .flatten();
        let (reply, stats, reason) =
            generate(engine, session, resolved, thinking, &messages, media, request, &offered)?;
        // A turn can span several generations; the client is told once, at the
        // end, with the totals. Sending `Done` per round ended the SSE stream
        // before the first tool had even run.
        generated += stats.generated_tokens as u32;
        prompt_tokens += stats.prompt_tokens as u32;
        elapsed_ms += stats.generation_ms;
        prompt_ms += stats.prompt_ms;
        reused += stats.reused_tokens;
        stop = reason;

        let mut parsed = ozgent_llama::extract_tool_calls(&reply);
        // The parser numbers calls from zero each time it runs, so round two
        // reissues `call_0`. Within a turn that has to be unique: it is how a
        // client pairs a result with the card that asked for it, and how the
        // conversation pairs a result with its call.
        for call in parsed.calls.iter_mut() {
            call.id = format!("call_{next_call_id}");
            next_call_id += 1;
        }
        tracing::debug!(calls = parsed.calls.len(), "tool round {round}");

        // Reasoning, then nothing. A model can close its thought and stop
        // without either answering or calling anything, and the turn then
        // returns an empty string — which reads to the user as ozgent being
        // broken. Ask once for the answer it was about to give; if it stays
        // silent the second time, that is its reply and the loop ends.
        if !parsed.has_calls() && parsed.text.trim().is_empty() && !nudged {
            nudged = true;
            messages.push(Message::system(ANSWER_NOW));
            continue;
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
            handed_back = true;
            break;
        }

        let Some(t) = tools.as_ref() else { break };

        // Independent calls run together. In series, a turn asking for three
        // searches paid three network round trips end to end, and the tool
        // timeout applied to each in turn rather than to the set.
        //
        // Every call is announced before any is awaited, so the transcript
        // shows the whole batch as pending rather than appearing to work
        // through them one at a time.
        for call in &parsed.calls {
            let _ = request.out.send(Event::ToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            });
        }
        let outcomes = t.runtime.block_on(futures_util::future::join_all(
            parsed.calls.iter().map(|call| async {
                let started = std::time::Instant::now();
                let outcome = t.host.call(&call.name, call.arguments.clone()).await;
                (outcome, started.elapsed().as_millis() as u64)
            }),
        ));

        // Consumed in the model's original order, not completion order: the
        // results become the next prompt, so letting a race decide their order
        // would make the same turn produce different continuations.
        for (call, (outcome, ms)) in parsed.calls.iter().zip(outcomes) {
            let (ok, summary, detail, payload) = match outcome {
                Ok(value) => {
                    let text = serde_json::to_string(&value).unwrap_or_default();
                    (true, summarise(&value), value, text)
                }
                // A tool failure is information the model can act on, not an
                // error for the user: it is fed back so the model can retry or
                // explain, exactly as the terminal client does.
                Err(e) => {
                    let text = e.for_model();
                    let summary = ozgent_tools::first_line(&text).to_string();
                    (false, summary, serde_json::json!({ "error": text.clone() }), text)
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
            // A search asked for 20 results can return more text than the whole
            // context window holds. Unbounded, it crowds out the room the model
            // needs to answer and the reply stops mid-sentence — which reads
            // like a crash but is simply no space left.
            let budget = fit_budget(resolved.context_length);
            messages.push(Message::tool_result(call.id.clone(), fit(&payload, budget)));
        }
    }

    if request.response_grammar.is_some() {
        let _ = session.set_grammar(None);
    }

    let seconds = elapsed_ms as f64 / 1000.0;
    let _ = request.out.send(Event::Done {
        generated,
        tokens_per_second: if seconds > 0.0 { generated as f64 / seconds } else { 0.0 },
        reused,
        stop: if handed_back { "ToolCalls".to_string() } else { format!("{stop:?}") },
        prompt: prompt_tokens,
        prompt_ms: prompt_ms as u64,
    });
    Ok(())
}

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
const OUT_OF_ROUNDS: &str = "\
You have no tool calls left. Answer now, using only what the tool results \
above actually contain. If they did not give you enough, say what you found \
and what is still missing — do not fill the gap with a guess.";

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
) -> anyhow::Result<(String, ozgent_llama::engine::Stats, StopReason)> {
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

    Ok((raw, stats, reason))
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
            prompt_ms: 210,
        })
        .unwrap();
        assert_eq!(json["type"], "done");
        assert_eq!(json["generated"], 120);
        assert_eq!(json["reused"], 64);
        assert_eq!(json["prompt"], 40, "the API reports prompt tokens from this");
        // The client calls .toFixed(1) on this, so it must be a number.
        assert!(json["tokens_per_second"].is_f64());
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

    #[test]
    fn a_dead_worker_is_reported_rather_than_panicking() {
        // Dropping the receiver simulates the thread having exited; submitting
        // must surface an error so the handler can return a 500 instead of
        // hanging the request forever.
        let (tx, rx) = channel::<Job>();
        drop(rx);
        let worker = Worker { tx };
        let (out, _keep) = tokio::sync::mpsc::unbounded_channel();
        let result = worker.submit(Request {
            model: "any".into(),
            messages: Vec::new(),
            thinking: None,
            max_tokens: None,
            tools_enabled: true,
            native_tools: None,
            client_tools: Vec::new(),
            response_grammar: None,
            overrides: None,
            images: Vec::new(),
            out,
        });
        assert!(result.is_err());
    }
}
