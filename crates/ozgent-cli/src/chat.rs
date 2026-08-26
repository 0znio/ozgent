//! The interactive chat loop.
//!
//! This is where the finished pieces meet: the engine generates, memory
//! decides what the model is allowed to remember, the Python worker runs
//! tools, and the renderer draws it. Each of those is exercised elsewhere in
//! isolation; this module is the wiring.

use anyhow::{Context, Result};
use ozgent_core::{Config, Manifest, Message, ModelRef, Paths, Resolved, Role, ThinkingMode};
use ozgent_llama::engine::{Engine, Session, StopReason};
use ozgent_llama::thinking::{Chunk, ThinkingFilter};
use ozgent_llama::toolcall;
use ozgent_memory::{Budget, ContextBuilder, HashingEmbedder, OwnerKind, Store};
use ozgent_render::{MarkdownRenderer, StreamRenderer, Style, Theme};
use ozgent_tools::ToolHost;
use crate::input::{Input, Prompt};
use std::io::Write;

/// Everything one chat session needs.
/// The engine's embedder, adapted to the memory layer's trait.
///
/// A newtype because both the trait and the type are foreign here; the
/// alternative is a dependency from ozgent-llama on ozgent-memory, which would
/// pull SQLite into the inference crate for one interface.
struct ModelEmbedder(ozgent_llama::embed::Embedder);

impl ozgent_memory::Embedder for ModelEmbedder {
    fn dimensions(&self) -> usize {
        self.0.dimensions()
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        // A failure here must not take the turn down with it: retrieval
        // degrades to the keyword half rather than the conversation ending.
        self.0.embed(text).unwrap_or_else(|e| {
            tracing::warn!("embedding failed, falling back to no vector: {e}");
            vec![0.0; self.0.dimensions()]
        })
    }

    fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        self.0.embed_batch(texts).unwrap_or_else(|e| {
            tracing::warn!("batch embedding failed: {e}");
            texts.iter().map(|_| vec![0.0; self.0.dimensions()]).collect()
        })
    }
}

/// The configured embedding model, or the lexical fallback.
///
/// Falling back is deliberate and logged: memory still works without an
/// embedding model, it just loses the half of retrieval that finds a note
/// whose words differ from the query's.
fn build_embedder(paths: &Paths, config: &Config) -> Box<dyn ozgent_memory::Embedder> {
    let Some(name) = config.embedding.model.as_deref() else {
        return Box::new(HashingEmbedder::default());
    };
    let loaded = ozgent_core::resolve(paths, name)
        .map_err(|e| e.to_string())
        .and_then(|found| {
            let weights = found.manifest.primary_weights(&found.dir);
            ozgent_llama::embed::Embedder::load(&weights, 99).map_err(|e| e.to_string())
        });
    match loaded {
        Ok(e) => Box::new(ModelEmbedder(e)),
        Err(e) => {
            tracing::warn!("embedding model {name:?} unavailable; memory falls back to lexical matching: {e}");
            Box::new(HashingEmbedder::default())
        }
    }
}

pub struct Chat<'a> {
    engine: &'a Engine,
    session: Session<'a>,
    store: Store,
    embedder: Box<dyn ozgent_memory::Embedder>,
    tools: Option<ToolHost>,
    conversation: i64,
    opts: Resolved,
    theme: Theme,
    width: usize,
    model: ModelRef,
    manifest: Manifest,
    /// Where this model's files live, for finding its projector.
    model_dir: std::path::PathBuf,
    /// Loaded on the first turn that carries an image, then kept.
    projector: Option<ozgent_llama::mtmd::Projector<'a>>,
    /// Images for the turn being built; cleared once evaluated.
    pending_images: Vec<ozgent_llama::mtmd::Media>,
    /// The sources those bytes came from, for capability reporting.
    pending_sources: Vec<ozgent_core::ImageSource>,
    /// Set when this turn carries media, so the model is told it can see.
    media_turn: bool,
    /// What the grounding pass saw, consumed when the turn's prompt is built.
    media_observation: Option<String>,
    /// Turns are numbered for the `/memory` display.
    turn: usize,
    /// Owned copies so `/config` and `/tools` can persist changes without
    /// borrowing from the caller across an await.
    paths: Paths,
    config: Config,
}

/// Start a chat.
pub async fn run(
    paths: &Paths,
    config: &Config,
    model: Option<String>,
    options: &crate::cli::OptionFlags,
) -> Result<()> {
    let mut name = model
        .or_else(|| config.default_model.clone())
        .context(
            "no model given and no default_model set in config.toml.\n\
             Try: ozgent pull <huggingface-repo> --name <short-name>, then ozgent chat <short-name>",
        )?;

    paths.ensure()?;
    let mut prompt = Prompt::new(Some(paths.root().join("history")));
    // The conversation survives a model switch: it is the user's thread, not
    // the model's.
    let mut conversation: Option<i64> = None;

    // Each pass loads one model. `/models` unwinds here to load another;
    // rebuilding the store and tool worker costs far less than the model load
    // that is happening anyway.
    loop {
        let next = run_one(paths, config, &name, options, &mut prompt, &mut conversation).await?;
        match next {
            Some(other) => {
                eprintln!();
                name = other;
            }
            None => return Ok(()),
        }
    }
}

/// Run a chat against one model. Returns the model to switch to, if any.
async fn run_one(
    paths: &Paths,
    config: &Config,
    name: &str,
    options: &crate::cli::OptionFlags,
    prompt: &mut Prompt,
    conversation: &mut Option<i64>,
) -> Result<Option<String>> {
    let found = ozgent_core::resolve(paths, name)?;
    let (model_ref, dir, manifest) = (found.model, found.dir, found.manifest);

    let opts = config
        .options_for(&model_ref.to_string())
        .merge(&manifest.defaults)
        .merge(&options.to_options()?)
        .resolve();

    eprint!("loading {model_ref}… ");
    std::io::stderr().flush().ok();
    let started = std::time::Instant::now();
    let engine = Engine::load(&manifest.primary_weights(&dir), &opts)?;
    eprintln!(
        "{} layers ({} on gpu) in {:.1}s",
        engine.n_layer(),
        engine.gpu_layers_used(),
        started.elapsed().as_secs_f32()
    );

    let session = engine.session(&opts)?;

    // One database for every front end, so the web UI sees the same history.
    let store = Store::open(paths.root().join("ozgent.db"))?;
    let conversation_id = match *conversation {
        Some(id) => id,
        None => {
            let id = store.create_conversation("", Some(&model_ref.to_string()))?;
            *conversation = Some(id);
            id
        }
    };

    let tools = if opts.tools && config.tools.enabled {
        match crate::start_tools(paths, config).await {
            Ok(host) => {
                eprintln!("{} tools loaded", host.tools().len());
                Some(host)
            }
            Err(e) => {
                eprintln!("tools unavailable: {e}");
                None
            }
        }
    } else {
        None
    };

    let plain = options.plain || !config.ui.markdown;
    let turn = store.message_count(conversation_id).unwrap_or(0) as usize;

    let mut chat = Chat {
        engine: &engine,
        session,
        store,
        embedder: build_embedder(paths, config),
        tools,
        conversation: conversation_id,
        theme: if plain { Theme::plain() } else { Theme::default() },
        width: ozgent_render::terminal_width(),
        model: model_ref,
        manifest,
        model_dir: dir.clone(),
        projector: None,
        pending_images: Vec::new(),
        pending_sources: Vec::new(),
        media_turn: false,
        media_observation: None,
        opts,
        turn,
        paths: paths.clone(),
        config: config.clone(),
    };

    // Constrain the body of a tool call once one starts. This is the cheap
    // half of the pair below: the retry path in `turn` fixes a malformed call
    // after paying for it, while gating stops it being generated at all.
    if let Some(host) = &chat.tools {
        let specs = host.tools().to_vec();
        chat.session.set_tools(&specs);
    }

    chat.banner();
    let outcome = chat.repl(prompt).await;
    prompt.save();

    if let Some(host) = &chat.tools {
        host.shutdown().await;
    }
    *conversation = Some(chat.conversation);

    match outcome? {
        Flow::Switch(other) => Ok(Some(other)),
        _ => Ok(None),
    }
}

impl<'a> Chat<'a> {
    fn banner(&self) {
        let dim = |s: &str| self.theme.style(Style::dim(), s);
        eprintln!();
        eprintln!("{} {}", self.model, dim(&format!("· {} ctx", self.session.n_ctx())));
        eprintln!("{}", dim("/help for commands, /exit to quit"));
        eprintln!();
    }

    async fn repl(&mut self, prompt: &mut Prompt) -> Result<Flow> {
        crate::input::install_interrupt_handler();
        let marker = self.theme.style(Style::color(ozgent_render::Color::Cyan), "› ");

        loop {
            let line = match prompt.read(&marker) {
                Input::Line(l) => l,
                // Ctrl-C at the prompt abandons the line, not the session.
                Input::Cancelled => continue,
                Input::Eof => break,
            };
            let input = line.trim();

            if input.is_empty() {
                continue;
            }
            if looks_like_command(input) {
                match self.command(input).await? {
                    Flow::Continue => continue,
                    Flow::Exit => return Ok(Flow::Exit),
                    // The engine is owned by `run`, so switching unwinds to
                    // there rather than swapping it underneath us.
                    Flow::Switch(next) => {
                        prompt.save();
                        return Ok(Flow::Switch(next));
                    }
                }
            }

            if let Err(e) = self.turn(input).await {
                eprintln!(
                    "{}",
                    self.theme.style(Style::color(ozgent_render::Color::Red), &format!("error: {e}"))
                );
            }
            prompt.save();
        }
        eprintln!();
        Ok(Flow::Exit)
    }

    /// One user turn: remember it, build context, generate, run any tools.
    async fn turn(&mut self, input: &str) -> Result<()> {
        self.turn += 1;

        // Images referenced by path or URL are picked up automatically.
        let mut extracted = ozgent_llama::vision::extract(input);
        if extracted.has_images() {
            match self.load_media(&extracted.images) {
                Ok(count) if count > 0 => {
                    let marker = self.projector.as_ref().expect("loaded above").marker();
                    extracted.text =
                        ozgent_llama::mtmd::with_markers(marker, &extracted.text, count);
                    self.media_turn = true;
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("{}", self.theme.style(Style::dim(), &format!("note: {e}")));
                    self.pending_images.clear();
                }
            }
        }

        let user_id = self.store.append_message(
            self.conversation,
            "user",
            &extracted.text,
            0,
        )?;
        self.store.put_embedding(
            OwnerKind::Message,
            user_id,
            &self.embedder.embed(&extracted.text),
        )?;

        // Name the conversation after its first line, so `ozgent web` has
        // something readable in its sidebar.
        if self.turn == 1 {
            let title: String = extracted.text.chars().take(60).collect();
            self.store.rename_conversation(self.conversation, &title)?;
        }

        if self.media_turn {
            self.media_observation = self.ground(&extracted.text);
        }
        let mut messages = self.build_context(&extracted.text)?;
        let mut reply = self.generate(&messages)?;

        // A tool call is a request, not an answer: run it, hand the result
        // back, and let the model continue. Bounded so a model that keeps
        // calling cannot loop forever.
        let max_calls = 4;
        for round in 0..max_calls {
            let mut parsed = toolcall::extract(&reply.text);

            // The model tried to call a tool and produced something
            // unparseable. Constrained decoding cannot produce malformed JSON,
            // an unknown tool name, or a misspelled parameter, so retrying
            // under the grammar turns a failed attempt into a valid one.
            if !parsed.has_calls() && reply.attempted_call {
                if let Some(host) = &self.tools {
                    if let Some(grammar) = ozgent_llama::grammar::tool_call_grammar(host.tools()) {
                        eprintln!(
                            "{}",
                            self.theme.style(Style::dim(), "· malformed tool call; retrying under grammar")
                        );
                        reply = self.generate_with(&messages, Some(&grammar))?;
                        parsed = toolcall::extract(&reply.text);
                    }
                }
            }

            if !parsed.has_calls() || self.tools.is_none() {
                break;
            }
            reply.text = parsed.text.clone();

            for call in &parsed.calls {
                self.show_call(call);
                let started = std::time::Instant::now();

                let host = self.tools.as_ref().expect("checked above");
                let outcome = host.call(&call.name, call.arguments.clone()).await;
                self.show_result(&outcome, started.elapsed());

                let result = match outcome {
                    Ok(value) => serde_json::to_string(&value).unwrap_or_default(),
                    Err(e) => e.for_model(),
                };

                messages.push(Message {
                    role: Role::Assistant,
                    content: vec![ozgent_core::Part::Text {
                        text: format!(
                            "<tool_call>{{\"name\": \"{}\", \"arguments\": {}}}</tool_call>",
                            call.name, call.arguments
                        ),
                    }],
                    thinking: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                });
                messages.push(Message::tool_result(call.id.clone(), truncate_result(&result)));
            }

            if round + 1 == max_calls {
                eprintln!(
                    "{}",
                    self.theme.style(Style::dim(), "· tool call limit reached")
                );
            }
            reply = self.generate(&messages)?;
        }

        let assistant_id = self.store.append_message_full(
            self.conversation,
            "assistant",
            &reply.text,
            reply.thinking.as_deref(),
            None,
            None,
            0,
        )?;
        self.store.put_embedding(
            OwnerKind::Message,
            assistant_id,
            &self.embedder.embed(&reply.text),
        )?;
        Ok(())
    }

    /// Assemble the prompt from pinned facts, recent turns, and recall.
    fn build_context(&self, query: &str) -> Result<Vec<Message>> {
        // Leave room for the answer, and keep the window inside the context.
        let budget = Budget {
            total: self.session.n_ctx() as usize,
            reserve_for_reply: (self.session.n_ctx() as usize / 4).max(256),
            recent_messages: 12,
            max_retrieved: 6,
        };

        let ctx = ContextBuilder::new(&self.store, &self.embedder)
            .with_budget(budget)
            .build(self.conversation, query)?;

        if ctx.used_recall() {
            eprintln!(
                "{}",
                self.theme.style(
                    Style::dim(),
                    &format!("· recalled {} earlier item(s)", ctx.retrieved.len())
                )
            );
        }
        // The date goes first: it is context about the world, not instruction,
        // and a model reads the opening of a system prompt most reliably.
        let dated = if self.config.ui.date_awareness {
            let line = ozgent_core::DateTime::now().prompt_line();
            Some(match &self.opts.system_prompt {
                Some(base) if !base.trim().is_empty() => format!("{line}\n\n{base}"),
                _ => line,
            })
        } else {
            self.opts.system_prompt.clone()
        };

        // The tool description joins the system prompt rather than replacing
        // it, so a user-set persona survives.
        // What the model saw when it looked, so the turn starts from an
        // accurate reading rather than a guess at one. Appended *after* the
        // tool preamble below, not before: placed first, the tool instructions
        // were the last thing read and the model reached for a search anyway.
        let media_note = self.media_observation.as_deref().map(grounded_note);

        let system = match (&dated, &self.tools) {
            (base, Some(host)) if !host.tools().is_empty() => {
                let mut text = base.clone().unwrap_or_default();
                if !text.is_empty() {
                    text.push_str("\n\n");
                }
                text.push_str(&ozgent_tools::tool_preamble(host.tools()));
                if let Some(note) = &media_note {
                    text.push_str("\n\n");
                    text.push_str(note);
                }
                Some(text)
            }
            (base, _) => match (&media_note, base) {
                (Some(note), Some(b)) => Some(format!("{b}\n\n{note}")),
                (Some(note), None) => Some(note.clone()),
                (None, b) => b.clone(),
            },
        };
        Ok(ctx.to_messages(system.as_deref()))
    }

    /// Load a projector if needed and read this turn's images.
    ///
    /// The projector is kept for the life of the session: loading it costs time
    /// and VRAM, and a conversation with one image usually has more.
    fn load_media(&mut self, sources: &[ozgent_core::ImageSource]) -> Result<usize> {
        if self.projector.is_none() {
            let Some(mmproj) = self.manifest.projector_path(&self.model_dir) else {
                anyhow::bail!("this model has no vision projector; the image is ignored");
            };
            self.projector = Some(self.engine.projector(&mmproj, &self.opts)?);
        }
        self.pending_images = ozgent_llama::mtmd::load_media(sources)?;
        self.pending_sources = sources.to_vec();
        Ok(self.pending_images.len())
    }

    /// Look at the media before doing anything else with it.
    ///
    /// A short, tool-free pass that describes what is actually in the
    /// attachment. The observation joins the system prompt, so by the time the
    /// real turn runs, "what is in this image" is already answered and there is
    /// nothing to search for — while a question the media genuinely cannot
    /// answer still has every tool available. See the note in the web worker
    /// for what this replaced and why a rule in the prompt was not enough.
    fn ground(&mut self, question: &str) -> Option<String> {
        let projector = self.projector.as_ref()?;
        if self.pending_images.is_empty() {
            return None;
        }

        let prompt = self
            .engine
            .render_prompt_with(
                &[
                    Message::system(
                        "Describe exactly what is in the attached media: subjects, text, \
                         colours, layout. State only what you can actually see. Do not \
                         speculate about what it might be, and do not answer the user's \
                         question yet.",
                    ),
                    Message::user(question),
                ],
                ozgent_core::ThinkingMode::Off,
                Default::default(),
            )
            .ok()?;

        eprint!("{}", self.theme.style(Style::dim(), "· looking"));
        let _ = std::io::stderr().flush();

        let mut observed = String::new();
        let result = self.session.generate_with_media(
            &prompt,
            Some((projector, &self.pending_images[..], &self.pending_sources[..])),
            GROUNDING_LIMIT,
            |piece| {
                observed.push_str(piece);
                true
            },
        );
        eprintln!("{}", self.theme.style(Style::dim(), " ✓"));

        match result {
            Ok(_) if !observed.trim().is_empty() => Some(observed.trim().to_string()),
            Ok(_) => None,
            Err(e) => {
                tracing::warn!("grounding pass failed, continuing without it: {e}");
                None
            }
        }
    }

    /// Generate one reply, streaming it to the terminal.
    fn generate(&mut self, messages: &[Message]) -> Result<Reply> {
        self.generate_with(messages, None)
    }

    /// Generate, optionally constrained to a GBNF grammar.
    fn generate_with(&mut self, messages: &[Message], grammar: Option<&str>) -> Result<Reply> {
        self.session.set_grammar(grammar)?;
        let result = self.generate_inner(messages);
        // The constraint must not leak into the next turn, or every reply
        // would be forced into the shape of a tool call.
        self.session.set_grammar(None)?;
        result
    }

    fn generate_inner(&mut self, messages: &[Message]) -> Result<Reply> {
        let prompt = self.engine.render_prompt_with(messages, self.opts.thinking, self.opts.reasoning_effort)?;
        // Images belong to this turn only: once evaluated they are resident in
        // the cache, and re-sending them would duplicate them in the context.
        let pending_images = std::mem::take(&mut self.pending_images);
        let pending_sources = std::mem::take(&mut self.pending_sources);
        self.media_turn = false;
        self.media_observation = None;
        tracing::debug!(
            "prompt ({} messages, {} chars):\n{prompt}",
            messages.len(),
            prompt.len()
        );

        let mut markdown =
            StreamRenderer::new(MarkdownRenderer::new(self.theme.clone(), self.width));
        let mut filter = ThinkingFilter::new(self.opts.thinking);
        if let Some(close) = Engine::stream_starts_inside(&prompt) {
            filter = filter.starting_inside(close);
        }
        let mut out = std::io::stdout();

        // Tool-call syntax is a request to the runtime, not output for the
        // user, so it is withheld from the terminal while still being captured
        // for parsing.
        let mut gate = toolcall::StreamGate::new();
        // Reasoning needs its own gate: a model that never closes `</think>`
        // emits its tool call inside the reasoning stream, and an ungated
        // thinking display would print raw JSON at the user.
        let mut think_gate = toolcall::StreamGate::new();
        crate::input::arm_interrupt();
        let mut answer = String::new();
        let mut thinking = String::new();
        let mut in_thinking = false;
        let show_thinking = self.opts.thinking != ThinkingMode::Off;
        let theme = self.theme.clone();

        let media = self
            .projector
            .as_ref()
            .filter(|_| !pending_images.is_empty())
            .map(|p| (p, &pending_images[..], &pending_sources[..]));
        let (stats, reason) = self.session.generate_with_media(&prompt, media, self.opts.max_tokens, |piece| {
            for chunk in filter.push(piece) {
                match chunk {
                    Chunk::Thinking(text) => {
                        thinking.push_str(&text);
                        if show_thinking {
                            let visible = think_gate.push(&text);
                            // A model that decides not to reason still emits an
                            // empty `<think></think>`. Waiting for real content
                            // before printing the label means those produce no
                            // output at all, rather than a bare "thinking".
                            if !in_thinking {
                                if !visible.trim().is_empty() {
                                    in_thinking = true;
                                    eprint!("{}", theme.style(theme.thinking, "thinking "));
                                    eprint!(
                                        "{}",
                                        theme.style(theme.thinking, visible.trim_start())
                                    );
                                    let _ = std::io::stderr().flush();
                                }
                            } else if !visible.is_empty() {
                                eprint!("{}", theme.style(theme.thinking, &visible));
                                let _ = std::io::stderr().flush();
                            }
                        }
                    }
                    Chunk::Answer(text) => {
                        if in_thinking {
                            in_thinking = false;
                            eprintln!();
                        }
                        answer.push_str(&text);
                        // Gated: tool-call syntax is a request to the runtime,
                        // not prose, and must never reach the terminal.
                        let visible = gate.push(&text);
                        if !visible.is_empty() {
                            let _ = markdown.push(&visible, &mut out);
                        }
                    }
                }
            }
            // Polled between tokens, so Ctrl-C stops the answer not the process.
            !crate::input::interrupted()
        })?;

        for chunk in filter.finish() {
            if let Chunk::Answer(text) = chunk {
                answer.push_str(&text);
                let visible = gate.push(&text);
                if !visible.is_empty() {
                    let _ = markdown.push(&visible, &mut out);
                }
            }
        }
        let tail = gate.finish();
        if !tail.is_empty() {
            let _ = markdown.push(&tail, &mut out);
        }
        markdown.finish(&mut out)?;
        println!();

        if crate::input::interrupted() {
            eprintln!("{}", self.theme.style(Style::dim(), "· interrupted"));
        }
        if answer.trim().is_empty() && stats.generated_tokens > 0 && !crate::input::interrupted() {
            eprintln!(
                "{}",
                self.theme.style(
                    Style::dim(),
                    "· no answer produced; the budget went on reasoning. Try /think off",
                )
            );
        }
        if reason == StopReason::ContextFull {
            eprintln!("{}", self.theme.style(Style::dim(), "· context full"));
        }
        if self.opts_show_stats() {
            eprintln!(
                "{}",
                self.theme.style(
                    Style::dim(),
                    &format!(
                        "· {} in ({:.0}/s), {} reused · {} out ({:.1}/s)",
                        stats.prompt_tokens,
                        stats.prompt_tokens_per_second(),
                        stats.reused_tokens,
                        stats.generated_tokens,
                        stats.tokens_per_second()
                    )
                )
            );
        }

        Ok(Reply {
            // An unclosed `</think>` leaves the call in the reasoning stream;
            // the parser has to see it either way.
            text: if toolcall::extract(&answer).has_calls() || !think_gate.suppressing() {
                answer
            } else {
                format!("{answer}{thinking}")
            },
            // An empty reasoning block is not a reasoning trace.
            thinking: (!thinking.trim().is_empty()).then_some(thinking),
            attempted_call: gate.suppressing() || think_gate.suppressing(),
        })
    }

    fn opts_show_stats(&self) -> bool {
        std::env::var("OZGENT_STATS").is_ok()
    }

    /// Announce a tool call: a green marker, the name, then its arguments.
    ///
    /// Arguments are shown as `key: value` rather than raw JSON, since the
    /// braces and quotes carry no information the user needs.
    fn show_call(&self, call: &ozgent_core::ToolCall) {
        let marker = self.theme.style(Style::color(ozgent_render::Color::Green), "●");
        let name = self.theme.style(
            ozgent_render::Style { bold: true, ..Default::default() },
            &call.name,
        );
        eprintln!("{marker} {name}{}", self.theme.style(Style::dim(), &pretty_args(&call.arguments)));
    }

    /// Report what a tool returned, indented under its call.
    fn show_result(
        &self,
        outcome: &Result<serde_json::Value, ozgent_tools::ToolCallError>,
        elapsed: std::time::Duration,
    ) {
        let (colour, text) = match outcome {
            Ok(value) => (ozgent_render::Color::Green, summarise_result(value)),
            Err(e) => (ozgent_render::Color::Red, ozgent_tools::first_line(&e.for_model()).to_string()),
        };
        let arrow = self.theme.style(Style::color(colour), "  ↳");
        let timing = self.theme.style(Style::dim(), &format!(" · {}ms", elapsed.as_millis()));
        eprintln!("{arrow} {}{timing}", self.theme.style(Style::dim(), &text));
    }

    /// Show installed models and pick one to switch to.
    fn pick_model(&mut self, arg: &str) -> Result<Option<String>> {
        let models = ozgent_core::installed(&self.paths);
        let dim = |t: &str| self.theme.style(Style::dim(), t);

        if models.is_empty() {
            eprintln!("{}", dim("no models installed. Try: ozgent pull <repo> --name <short>"));
            return Ok(None);
        }

        // A direct argument skips the picker: `/models coder`, or `/models 2`.
        if !arg.is_empty() {
            if let Ok(n) = arg.parse::<usize>() {
                let Some(chosen) = models.get(n.wrapping_sub(1)) else {
                    eprintln!("{}", dim(&format!("no model {n}; there are {}", models.len())));
                    return Ok(None);
                };
                if chosen.model == self.model {
                    eprintln!("{}", dim("already using that model"));
                    return Ok(None);
                }
                return Ok(Some(chosen.short_name()));
            }
            let found = ozgent_core::resolve(&self.paths, arg)?;
            if found.model == self.model {
                eprintln!("{}", dim(&format!("already using {}", found.model)));
                return Ok(None);
            }
            return Ok(Some(found.short_name()));
        }

        eprintln!("{}", dim("installed models:"));
        for (i, m) in models.iter().enumerate() {
            let marker = if m.model == self.model { "*" } else { " " };
            let alias = m.manifest.alias.as_ref().map(|a| format!("  ({a})")).unwrap_or_default();
            eprintln!("{}", dim(&format!("{marker} {:>2}. {}{alias}", i + 1, m.model)));
        }
        eprintln!("{}", dim("switch with /models <number|alias|name:tag>"));
        Ok(None)
    }

    /// Show or change settings for the current model, persisting them.
    fn configure(&mut self, arg: &str) -> Result<()> {
        let dim = |t: &str| self.theme.style(Style::dim(), t);
        let mut parts = arg.split_whitespace();
        let key = parts.next().unwrap_or("");
        let value = parts.collect::<Vec<_>>().join(" ");

        if key.is_empty() {
            eprintln!("{}", dim(&format!("settings for {}:", self.model)));
            eprintln!("{}", dim(&format!("  thinking     {:?}", self.opts.thinking)));
            eprintln!("{}", dim(&format!("  temperature  {}", self.opts.temperature)));
            eprintln!("{}", dim(&format!("  context      {}", self.opts.context_length)));
            eprintln!("{}", dim(&format!("  gpu layers   {}", self.opts.gpu_layers)));
            eprintln!("{}", dim(&format!("  tools        {}", self.opts.tools)));
            eprintln!("{}", dim("set with /config <key> <value>; keys: thinking, temperature, tools"));
            return Ok(());
        }
        if value.is_empty() {
            eprintln!("{}", dim(&format!("usage: /config {key} <value>")));
            return Ok(());
        }

        // Apply to the live session, then persist for next time.
        let mut layer = ozgent_core::Options::default();
        match key {
            "thinking" | "think" => {
                let mode: ThinkingMode = value.parse().map_err(|e: String| anyhow::anyhow!(e))?;
                self.opts.thinking = mode;
                layer.thinking = Some(mode);
            }
            "effort" | "reasoning_effort" => {
                let level: ozgent_core::ReasoningEffort =
                    value.parse().map_err(|e: String| anyhow::anyhow!(e))?;
                self.opts.reasoning_effort = level;
                layer.reasoning_effort = Some(level);
            }
            "temperature" | "temp" => {
                let t: f32 = value.parse().context("temperature must be a number")?;
                self.opts.temperature = t;
                layer.temperature = Some(t);
            }
            "tools" => {
                let on = matches!(value.as_str(), "on" | "true" | "yes" | "1");
                self.opts.tools = on;
                layer.tools = Some(on);
            }
            other => {
                eprintln!("{}", dim(&format!("unknown setting {other:?}; try thinking, effort, temperature, tools")));
                return Ok(());
            }
        }

        let entry = self.config.models.entry(self.model.to_string()).or_default();
        *entry = entry.clone().merge(&layer);
        self.config.save(&self.paths)?;
        eprintln!("{}", dim(&format!("· {key} = {value} (saved)")));
        Ok(())
    }

    /// Point a tool at a provider, e.g. `/tools web_search brave`.
    fn configure_tool(&mut self, arg: &str) -> Result<()> {
        let dim = |t: &str| self.theme.style(Style::dim(), t);
        let mut parts = arg.split_whitespace();
        let tool = parts.next().unwrap_or("").to_string();
        let provider = parts.next().map(str::to_string);

        let Some(provider) = provider else {
            eprintln!("{}", dim(&format!("usage: /tools {tool} <provider>")));
            if tool == "web_search" {
                eprintln!("{}", dim("  providers: brave, tavily, duckduckgo"));
            }
            return Ok(());
        };

        // Only these need a key; duckduckgo works without an account.
        let env_var = match provider.as_str() {
            "brave" => Some("BRAVE_API_KEY"),
            "tavily" => Some("TAVILY_API_KEY"),
            _ => None,
        };

        let mut key: Option<String> = None;
        if let Some(var) = env_var {
            let already = std::env::var(var).is_ok()
                || self
                    .config
                    .tools
                    .config
                    .get(&tool)
                    .and_then(|v| v.get(&provider))
                    .and_then(|v| v.get("api_key"))
                    .is_some();
            if !already {
                eprintln!("{}", dim(&format!("{provider} needs an API key (or set ${var}).")));
                eprint!("{}", dim("api key (blank to skip): "));
                std::io::stderr().flush().ok();
                let mut line = String::new();
                std::io::stdin().read_line(&mut line)?;
                let line = line.trim().to_string();
                if !line.is_empty() {
                    key = Some(line);
                }
            }
        }

        set_tool_provider(&mut self.config, &tool, &provider, key.as_deref());
        self.config.save(&self.paths)?;
        harden_config_permissions(&self.paths.config_file());

        eprintln!("{}", dim(&format!("· {tool} now uses {provider} (saved)")));
        eprintln!("{}", dim("restart the chat for the tool worker to pick it up"));
        Ok(())
    }

    /// Force the next reply to be a tool call, for `/call`.
    async fn forced_call(&mut self, query: &str) -> Result<()> {
        let Some(host) = &self.tools else {
            eprintln!("{}", self.theme.style(Style::dim(), "tools are disabled"));
            return Ok(());
        };
        let Some(grammar) = ozgent_llama::grammar::tool_call_grammar(host.tools()) else {
            eprintln!("{}", self.theme.style(Style::dim(), "no tools available"));
            return Ok(());
        };

        let mut messages = self.build_context(query)?;
        messages.push(Message::user(format!(
            "Call the most appropriate tool to answer: {query}"
        )));
        let reply = self.generate_with(&messages, Some(&grammar))?;

        let parsed = toolcall::extract(&reply.text);
        if parsed.calls.is_empty() {
            eprintln!("{}", self.theme.style(Style::dim(), "· the model produced no call"));
            return Ok(());
        }
        for call in &parsed.calls {
            eprintln!(
                "{}",
                self.theme.style(Style::dim(), &format!("· {}({})", call.name, compact(&call.arguments)))
            );
            let host = self.tools.as_ref().expect("checked above");
            match host.call(&call.name, call.arguments.clone()).await {
                Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default()),
                Err(e) => eprintln!("{}", self.theme.style(Style::dim(), &e.for_model())),
            }
        }
        Ok(())
    }

    /// `/clear` must drop the cache, since the next prompt shares no history.

    /// Handle a slash command.
    async fn command(&mut self, input: &str) -> Result<Flow> {
        let mut parts = input.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();
        let dim = |s: &str| self.theme.style(Style::dim(), s);

        match cmd {
            "/exit" | "/quit" | "/q" => return Ok(Flow::Exit),

            "/help" | "/h" => {
                eprintln!("{}", dim(HELP));
            }

            "/clear" | "/new" => {
                self.conversation = self
                    .store
                    .create_conversation("", Some(&self.model.to_string()))?;
                self.session.reset();
                self.turn = 0;
                eprintln!("{}", dim("· started a new conversation"));
            }

            "/think" => match arg {
                "" => eprintln!("{}", dim(&format!("thinking: {:?}", self.opts.thinking))),
                other => match other.parse::<ThinkingMode>() {
                    Ok(mode) => {
                        self.opts.thinking = mode;
                        eprintln!("{}", dim(&format!("· thinking {other}")));
                    }
                    Err(e) => eprintln!("{}", dim(&e)),
                },
            },

            "/effort" => match arg {
                "" => eprintln!(
                    "{}",
                    dim(&format!("effort: {:?}", self.opts.reasoning_effort))
                ),
                other => match other.parse::<ozgent_core::ReasoningEffort>() {
                    Ok(level) => {
                        self.opts.reasoning_effort = level;
                        eprintln!("{}", dim(&format!("· effort {other}")));
                    }
                    Err(e) => eprintln!("{}", dim(&e)),
                },
            },

            "/system" => {
                if arg.is_empty() {
                    match &self.opts.system_prompt {
                        Some(s) => eprintln!("{}", dim(s)),
                        None => eprintln!("{}", dim("no system prompt set")),
                    }
                } else {
                    self.opts.system_prompt = Some(arg.to_string());
                    eprintln!("{}", dim("· system prompt updated"));
                }
            }

            "/remember" => {
                if arg.is_empty() {
                    eprintln!("{}", dim("usage: /remember <fact>"));
                } else {
                    let id = self.store.add_fact(
                        Some(self.conversation),
                        ozgent_memory::Scope::User,
                        arg,
                        None,
                    )?;
                    // Explicitly-stated facts are pinned: the user asked for
                    // them to be remembered, so they should not depend on
                    // retrieval finding them again.
                    self.store.set_pinned(id, true)?;
                    self.store.put_embedding(
                        OwnerKind::Fact,
                        id,
                        &self.embedder.embed(arg),
                    )?;
                    eprintln!("{}", dim("· remembered"));
                }
            }

            "/memory" => {
                let facts = self.store.facts_for(self.conversation)?;
                let count = self.store.message_count(self.conversation)?;
                eprintln!("{}", dim(&format!("{count} messages, {} facts", facts.len())));
                for f in facts.iter().take(20) {
                    let mark = if f.pinned { "*" } else { " " };
                    eprintln!("{}", dim(&format!("  {mark} {}", f.text)));
                }
            }

            "/call" => {
                if arg.is_empty() {
                    eprintln!("{}", dim("usage: /call <what you want done>"));
                } else {
                    self.forced_call(arg).await?;
                }
            }

            "/models" | "/model" => {
                if let Some(next) = self.pick_model(arg)? {
                    return Ok(Flow::Switch(next));
                }
            }

            "/config" => self.configure(arg)?,

            "/tools" if !arg.is_empty() => self.configure_tool(arg)?,

            "/tools" => match &self.tools {
                Some(host) => {
                    eprintln!("{}", dim(&format!("{} tools", host.tools().len())));
                    for t in host.tools() {
                        eprintln!("{}", dim(&format!("  {}  {}", t.name, ozgent_tools::first_line(&t.description))));
                    }
                }
                None => eprintln!("{}", dim("tools are disabled")),
            },

            "/stats" => {
                eprintln!(
                    "{}",
                    dim(&format!(
                        "{} · {} layers ({} gpu) · {} ctx, {} used, {} reused last turn · thinking {:?}",
                        self.model,
                        self.engine.n_layer(),
                        self.engine.gpu_layers_used(),
                        self.session.n_ctx(),
                        self.session.used(),
                        self.session.last_reused(),
                        self.opts.thinking
                    ))
                );
            }

            other => eprintln!("{}", dim(&format!("unknown command {other}; try /help"))),
        }
        Ok(Flow::Continue)
    }
}

/// Describe the available tools in the system prompt.
///
/// Most local GGUF chat templates have no tool slot, so the description goes
/// in as plain text. The output format is stated exactly, because a model that
/// invents its own wrapper produces a call the parser will not recognise.


struct Reply {
    text: String,
    thinking: Option<String>,
    /// The model began a tool call, whether or not it parsed. Used to decide
    /// when a grammar-constrained retry is worth attempting.
    attempted_call: bool,
}

enum Flow {
    Continue,
    Exit,
    /// `/models` picked a different model; `run` reloads and re-enters.
    Switch(String),
}

/// Record a tool's provider choice, and its key if one was supplied.
///
/// Tool settings are opaque TOML that the Python worker reads verbatim, so
/// this only has to place values at the right path.
fn set_tool_provider(config: &mut Config, tool: &str, provider: &str, api_key: Option<&str>) {
    use toml::Value;

    let entry = config
        .tools
        .config
        .entry(tool.to_string())
        .or_insert_with(|| Value::Table(Default::default()));
    if !entry.is_table() {
        *entry = Value::Table(Default::default());
    }
    let table = entry.as_table_mut().expect("just ensured a table");
    table.insert("provider".into(), Value::String(provider.to_string()));

    // Keys live under the provider they belong to, so switching provider and
    // back does not lose a key that was already entered.
    if let Some(key) = api_key {
        let sub = table
            .entry(provider.to_string())
            .or_insert_with(|| Value::Table(Default::default()));
        if !sub.is_table() {
            *sub = Value::Table(Default::default());
        }
        sub.as_table_mut()
            .expect("just ensured a table")
            .insert("api_key".into(), Value::String(key.to_string()));
    }
}

/// Restrict the config file to its owner.
///
/// It can hold API keys, and a world-readable file in a shared home is a real
/// leak. Failure is not fatal — the setting is still saved.
fn harden_config_permissions(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// Render arguments as `key: value` pairs rather than JSON.
fn pretty_args(args: &serde_json::Value) -> String {
    let Some(obj) = args.as_object() else {
        return String::new();
    };
    if obj.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = obj
        .iter()
        .map(|(k, v)| {
            let shown = match v {
                // Strings print bare; quotes add nothing at a glance.
                serde_json::Value::String(s) => truncate_middle(s, 48),
                other => truncate_middle(&other.to_string(), 48),
            };
            format!("{k}: {shown}")
        })
        .collect();
    format!("  {}", parts.join("  "))
}

/// One line describing what a tool returned.
///
/// A search response is thousands of characters; the user wants to know it
/// worked and roughly what came back, not to read it.
fn summarise_result(value: &serde_json::Value) -> String {
    if let Some(obj) = value.as_object() {
        // The shape every web_search provider normalises to.
        if let Some(results) = obj.get("results").and_then(|r| r.as_array()) {
            let provider = obj.get("provider").and_then(|p| p.as_str()).unwrap_or("");
            let n = results.len();
            let plural = if n == 1 { "result" } else { "results" };
            return if provider.is_empty() {
                format!("{n} {plural}")
            } else {
                format!("{n} {plural} from {provider}")
            };
        }
        let keys: Vec<&str> = obj.keys().take(4).map(String::as_str).collect();
        if !keys.is_empty() {
            return keys.join(", ");
        }
    }
    if let Some(a) = value.as_array() {
        return format!("{} items", a.len());
    }
    truncate_middle(&value.to_string(), 72)
}

/// Shorten from the middle, so both ends stay readable.
fn truncate_middle(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let head: String = chars[..max / 2].iter().collect();
    let tail: String = chars[chars.len() - (max / 2 - 2)..].iter().collect();
    format!("{head}…{tail}")
}

/// Compact JSON for a one-line trace of a tool call.
fn compact(v: &serde_json::Value) -> String {
    let s = v.to_string();
    if s.len() > 80 { format!("{}…", &s[..77]) } else { s }
}

/// Tool output can be large; a whole search response would crowd out the
/// conversation, so it is capped before entering the context.
fn truncate_result(s: &str) -> String {
    const MAX: usize = 4000;
    if s.len() <= MAX {
        return s.to_string();
    }
    let mut end = MAX;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…\n[truncated, {} bytes total]", &s[..end], s.len())
}



const HELP: &str = "\
/help              this list
/exit              quit
/clear             start a new conversation
/think on|off|auto show or suppress reasoning
/effort low|med|high how long the model may reason
/system <text>     set the system prompt
/remember <fact>   pin a fact for this and future chats
/memory            what is remembered
/models [name]     list models, or switch to one
/config [k] [v]    show or change this model's settings
/tools             list available tools
/tools <t> <prov>  point a tool at a provider, e.g. /tools web_search brave
/call <request>    force a tool call, constrained by grammar
/stats             model and context state

Paste an image path or URL in a message and it is picked up automatically.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_lists_every_command_the_parser_accepts() {
        // A command that exists but is undocumented is invisible to the user.
        for cmd in ["/help", "/exit", "/clear", "/think", "/effort", "/system", "/remember", "/memory", "/tools", "/stats"] {
            assert!(HELP.contains(cmd), "{cmd} is missing from /help");
        }
    }

    #[test]
    fn arguments_render_as_readable_pairs() {
        let args = serde_json::json!({"query": "latest news on NSE", "count": 5});
        let out = pretty_args(&args);
        assert!(out.contains("query: latest news on NSE"), "{out}");
        assert!(out.contains("count: 5"), "{out}");
        assert!(!out.contains('{'), "no raw JSON braces: {out}");
        assert!(!out.contains('\"'), "no quoting noise: {out}");
    }

    #[test]
    fn empty_arguments_render_as_nothing() {
        assert_eq!(pretty_args(&serde_json::json!({})), "");
        assert_eq!(pretty_args(&serde_json::json!(null)), "");
    }

    #[test]
    fn a_search_response_summarises_by_count_and_provider() {
        let v = serde_json::json!({
            "provider": "brave",
            "query": "x",
            "results": [{"title": "a"}, {"title": "b"}, {"title": "c"}]
        });
        assert_eq!(summarise_result(&v), "3 results from brave");
    }

    #[test]
    fn a_single_result_is_not_pluralised() {
        let v = serde_json::json!({"provider": "tavily", "results": [{"title": "a"}]});
        assert_eq!(summarise_result(&v), "1 result from tavily");
    }

    #[test]
    fn other_shapes_fall_back_to_their_keys() {
        let v = serde_json::json!({"celsius": 19, "city": "Lima"});
        assert_eq!(summarise_result(&v), "celsius, city");
    }

    #[test]
    fn long_values_are_shortened_from_the_middle() {
        let s = "start".to_string() + &"x".repeat(200) + "end";
        let out = truncate_middle(&s, 40);
        assert!(out.len() < 80, "should be short: {out}");
        assert!(out.starts_with("start"), "the beginning must survive: {out}");
        assert!(out.ends_with("end"), "and so must the end: {out}");
        assert!(out.contains('…'));
    }

    #[test]
    fn short_values_are_left_alone() {
        assert_eq!(truncate_middle("brief", 40), "brief");
    }

    #[test]
    fn setting_a_tool_provider_writes_the_expected_shape() {
        let mut config = Config::default();
        set_tool_provider(&mut config, "web_search", "brave", Some("secret123"));

        let ws = config.tools.config.get("web_search").expect("tool entry");
        assert_eq!(ws.get("provider").and_then(|v| v.as_str()), Some("brave"));
        assert_eq!(
            ws.get("brave").and_then(|v| v.get("api_key")).and_then(|v| v.as_str()),
            Some("secret123"),
            "the key must sit under the provider it belongs to"
        );
    }

    #[test]
    fn changing_provider_keeps_a_previously_stored_key() {
        let mut config = Config::default();
        set_tool_provider(&mut config, "web_search", "brave", Some("k1"));
        set_tool_provider(&mut config, "web_search", "duckduckgo", None);

        let ws = config.tools.config.get("web_search").unwrap();
        assert_eq!(ws.get("provider").and_then(|v| v.as_str()), Some("duckduckgo"));
        assert_eq!(
            ws.get("brave").and_then(|v| v.get("api_key")).and_then(|v| v.as_str()),
            Some("k1"),
            "switching provider must not discard an existing key"
        );
    }

    #[test]
    fn a_provider_without_a_key_stores_only_the_choice() {
        let mut config = Config::default();
        set_tool_provider(&mut config, "web_search", "duckduckgo", None);
        let ws = config.tools.config.get("web_search").unwrap();
        assert_eq!(ws.get("provider").and_then(|v| v.as_str()), Some("duckduckgo"));
        assert!(ws.get("duckduckgo").is_none(), "no key means no sub-table");
    }

    #[test]
    fn the_tool_preamble_states_the_exact_format() {
        let spec = ozgent_core::ToolSpec {
            name: "web_search".into(),
            description: "Search the web.\nMore detail.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"query": {"type": "string", "description": "what to find"}},
                "required": ["query"]
            }),
            output_schema: None,
        };
        let p = ozgent_tools::tool_preamble(&[spec]);
        assert!(p.contains("<tool_call>"), "the wrapper must be shown: {p}");
        assert!(p.contains("web_search"), "{p}");
        assert!(p.contains("query: string (required)"), "types and requiredness: {p}");
        assert!(p.contains("what to find"), "parameter docs should reach the model: {p}");
        assert!(!p.contains("More detail."), "only the first description line");
    }

    #[test]
    fn tool_results_are_capped_before_entering_context() {
        let big = "x".repeat(10_000);
        let out = truncate_result(&big);
        assert!(out.len() < 5000, "must be capped, got {}", out.len());
        assert!(out.contains("truncated"), "and must say so: {}", &out[out.len() - 40..]);
        assert_eq!(truncate_result("small"), "small");
    }

    #[test]
    fn truncation_never_splits_a_character() {
        let s = "é".repeat(5000);
        let out = truncate_result(&s);
        assert!(out.is_char_boundary(out.len()), "must remain valid UTF-8");
    }

    #[test]
    fn compact_shortens_long_arguments() {
        let v = serde_json::json!({"q": "y".repeat(200)});
        assert!(compact(&v).len() <= 81, "should be one line");
        assert!(compact(&serde_json::json!({"a": 1})).contains("\"a\""));
    }

    #[test]
    fn first_line_truncates_multiline_descriptions() {
        assert_eq!(ozgent_tools::first_line("one\ntwo\nthree"), "one");
        assert_eq!(ozgent_tools::first_line(""), "");
    }
}

/// Tokens the grounding pass may spend before the real turn begins.
const GROUNDING_LIMIT: u32 = 200;

/// The system-prompt note built from what the model saw.
///
/// Naming the only reason a tool is still warranted is load-bearing: given the
/// observation alone the model read the image correctly and then searched the
/// web for it anyway.
fn grounded_note(observation: &str) -> String {
    format!(
        "You have already looked at the attached media. This is what is actually \
         in it:\n{observation}\n\nAnswer the user from that observation. It is a \
         complete and accurate record of the media, so questions about what the \
         media contains, shows, or looks like are already answered — do not use a \
         tool for them, and do not ask the user to describe it.\n\nUse a tool only \
         if the user asked for something the media cannot contain: a current \
         price, recent news, today's weather, or another fact from the outside \
         world."
    )
}

/// Whether a line is a slash command rather than a message.
///
/// An absolute path is not a command. Dropping `/home/me/photo.png` into the
/// prompt — the documented way to attach an image — was answered with
/// "unknown command", because the leading slash was all that was being checked.
fn looks_like_command(input: &str) -> bool {
    let Some(rest) = input.strip_prefix('/') else { return false };
    let word = rest.split_whitespace().next().unwrap_or("");
    // Commands are one bare word. A path has separators or a file extension,
    // and either is enough to tell them apart without a list to keep in sync.
    !word.is_empty()
        && word.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[cfg(test)]
mod command_tests {
    use super::looks_like_command;

    #[test]
    fn slash_commands_are_recognised() {
        for line in ["/help", "/exit", "/models 2", "/tools web_search brave", "/config coder"] {
            assert!(looks_like_command(line), "{line} should be a command");
        }
    }

    #[test]
    fn an_absolute_path_is_a_message_not_a_command() {
        // The regression: pasting an image path is how attachments work.
        for line in [
            "/home/me/photo.png what is this?",
            "/tmp/a/b.jpg",
            "/usr/share/doc/readme.md summarise this",
        ] {
            assert!(!looks_like_command(line), "{line} should be a message");
        }
    }

    #[test]
    fn a_bare_slash_is_not_a_command() {
        assert!(!looks_like_command("/"));
        assert!(!looks_like_command("/ "));
    }

    #[test]
    fn ordinary_text_is_never_a_command() {
        assert!(!looks_like_command("what is 2 + 2"));
        assert!(!looks_like_command(""));
    }
}
