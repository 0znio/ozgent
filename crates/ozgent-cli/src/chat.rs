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
use ozgent_render::{Style, Theme};
use ozgent_tools::ToolHost;
use crate::tui::{Submission, Ui};
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
    /// The conversation being written to, once there is one.
    ///
    /// `None` until the first message. Opening a chat and closing it again
    /// used to leave a titleless empty row behind every time, which filled
    /// the picker with nothing.
    conversation: Option<i64>,
    opts: Resolved,
    theme: Theme,
    /// The screen. Borrowed rather than owned because switching models tears
    /// this struct down and rebuilds it, and the conversation on screen must
    /// survive that.
    ui: &'a mut Ui,
    /// Tools the user has approved for the rest of this run. Never written
    /// to disk: "yes, for now" is a different promise from "yes, always".
    grants: ozgent_core::Grants,
    /// Generation rate of the last reply, for the status line. `None`
    /// until this session has produced one — a rate carried over from a
    /// previous model would be a lie about this one.
    last_rate: Option<f64>,
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
    // The conversation survives a model switch: it is the user's thread, not
    // the model's.
    let mut conversation: Option<i64> = None;

    // Built once, outside the loop. Switching models tears down the engine and
    // the tool worker; taking over the terminal again as well would blank the
    // screen and lose everything said so far.
    let plain = options.plain || !config.ui.markdown;
    let theme = if plain { Theme::plain() } else { Theme::default() };
    let mut ui = Ui::new(theme, Some(paths.root().join("history")));

    // Each pass loads one model. `/models` unwinds here to load another;
    // rebuilding the store and tool worker costs far less than the model load
    // that is happening anyway.
    let outcome = loop {
        let next = match run_one(paths, config, &name, options, &mut ui, &mut conversation).await {
            Ok(next) => next,
            Err(e) => break Err(e),
        };
        match next {
            Some(other) => {
                ui.blank();
                name = other;
            }
            None => break Ok(()),
        }
    };

    ui.save_history();
    // Before the error is printed, or it lands on the alternate screen and
    // disappears with it.
    ui.close();
    outcome
}

/// Run a chat against one model. Returns the model to switch to, if any.
async fn run_one(
    paths: &Paths,
    config: &Config,
    name: &str,
    options: &crate::cli::OptionFlags,
    ui: &mut Ui,
    conversation: &mut Option<i64>,
) -> Result<Option<String>> {
    let found = ozgent_core::resolve(paths, name)?;
    let (model_ref, dir, manifest) = (found.model, found.dir, found.manifest);

    let opts = config
        .options_for(&model_ref.to_string())
        .merge(&manifest.defaults)
        .merge(&options.to_options()?)
        .resolve();

    // Painted before the load rather than after, because the load is the long
    // part and a blank screen for twenty seconds looks like a hang.
    ui.say(format!("loading {model_ref}…"));
    ui.render();
    let started = std::time::Instant::now();
    let engine = Engine::load(&manifest.primary_weights(&dir), &opts)?;
    ui.say(format!(
        "{} layers ({} on gpu) in {:.1}s",
        engine.n_layer(),
        engine.gpu_layers_used(),
        started.elapsed().as_secs_f32()
    ));

    let session = engine.session(&opts)?;

    // One database for every front end, so the web UI sees the same history.
    let store = Store::open(paths.root().join("ozgent.db"))?;
    // Deliberately not created here. A conversation begins when the user says
    // something, not when a model finishes loading.
    let conversation_id = *conversation;

    let tools = if opts.tools && config.tools.enabled {
        match crate::start_tools(paths, config).await {
            Ok(host) => {
                ui.say(format!("{} tools loaded", host.tools().len()));
                Some(host)
            }
            Err(e) => {
                ui.say(format!("tools unavailable: {e}"));
                None
            }
        }
    } else {
        None
    };

    // The same decision `run` made when it built the screen, so the two
    // cannot disagree about whether this session is in colour.
    let plain = options.plain || !config.ui.markdown;
    let theme = if plain { Theme::plain() } else { Theme::default() };
    let turn = conversation_id
        .and_then(|id| store.message_count(id).ok())
        .unwrap_or(0) as usize;

    let mut chat = Chat {
        engine: &engine,
        session,
        store,
        embedder: build_embedder(paths, config),
        tools,
        conversation: conversation_id,
        ui,
        theme,
        grants: ozgent_core::Grants::default(),
        last_rate: None,
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
        // Withheld from a model whose template writes calls in its own
        // syntax: the grammar describes the JSON body ozgent's preamble asks
        // for, and that preamble was not sent.
        if !engine.template_handles_tools() {
            chat.session.set_tools(&specs);
        }
    }

    chat.banner();
    let outcome = chat.repl().await;

    if let Some(host) = &chat.tools {
        host.shutdown().await;
    }
    *conversation = chat.conversation;

    match outcome? {
        Flow::Switch(other) => Ok(Some(other)),
        _ => Ok(None),
    }
}

impl<'a> Chat<'a> {
    fn banner(&mut self) {
        // A clone, not a borrow of `self.theme`: writing to the screen
        // takes `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |s: &str| theme.style(Style::dim(), s);
        self.ui.blank();
        let window = self.session.n_ctx();
        self.ui.say(format!(
            "{} {}",
            self.model,
            dim(&format!("· {} ctx", ozgent_core::format_count(window)))
        ));
        // A window smaller than the one asked for is not a detail to leave in
        // the log. The model was loaded, answers will be correct, and the only
        // visible symptom is that a long conversation runs out sooner than the
        // user planned for — which is impossible to work out after the fact.
        if window < self.opts.context_length {
            let asked = ozgent_core::format_count(self.opts.context_length);
            let trained = self.engine.n_ctx_train();
            // Two different reasons land here and the fix for each is
            // different: one is answered by freeing memory or quantising the
            // cache, the other by picking a model that was trained longer.
            let why = if trained > 0 && window >= trained {
                format!("this model was trained for {}", ozgent_core::format_count(trained))
            } else {
                "that much KV cache does not fit in this machine's memory".to_string()
            };
            self.ui.say(dim(&format!("· asked for {asked}; {why}")));
        }
        self.ui.say(dim("/help for commands, /exit to quit"));
        self.ui.blank();
    }

    /// What the bottom row says: the facts that change as the session runs.
    ///
    /// Priorities decide what survives a narrow terminal. The model name goes
    /// first because a status line that has dropped it no longer says which
    /// machine's answer you are reading; the sampler settings go last because
    /// they are the ones `/config` will tell you on demand.
    fn status_segments(&self) -> Vec<crate::status::Segment> {
        use crate::status::Segment;
        use ozgent_core::format_count;

        let used = self.session.used();
        let window = self.session.n_ctx();
        let percent = if window > 0 { used * 100 / window } else { 0 };

        let mut segments = vec![
            Segment::new(0, self.model.to_string()),
            Segment::new(
                1,
                format!("ctx {}/{} {percent}%", format_count(used), format_count(window)),
            ),
        ];
        // Absent rather than zero before the first reply: "0.0 tok/s" reads as
        // a measurement, and there has not been one yet.
        if let Some(rate) = self.last_rate {
            segments.push(Segment::new(2, format!("{rate:.1} tok/s")));
        }
        segments.push(Segment::new(
            3,
            match self.opts.thinking {
                ThinkingMode::Off => "think off".to_string(),
                _ => format!("think {}", self.opts.reasoning_effort),
            },
        ));
        // Tools, and how many of them can act without being asked. A count
        // alone says nothing about the thing worth knowing.
        segments.push(Segment::new(
            4,
            match &self.tools {
                Some(host) => {
                    let asking = host
                        .tools()
                        .iter()
                        .filter(|spec| {
                            matches!(
                                self.config.permissions.verdict(
                                    &spec.name,
                                    spec.effect,
                                    &self.grants,
                                ),
                                ozgent_core::Verdict::Ask,
                            )
                        })
                        .count();
                    match asking {
                        0 => format!("{} tools", host.tools().len()),
                        n => format!("{} tools · {n} ask", host.tools().len()),
                    }
                }
                None => "no tools".to_string(),
            },
        ));
        segments.push(Segment::new(
            5,
            format!("temp {:.2} top-p {:.2}", self.opts.temperature, self.opts.top_p),
        ));
        segments
    }

    /// Refresh the two bottom bars from the session's current state.
    fn update_status(&mut self) {
        let segments = self.status_segments();
        self.ui.set_status(segments);
        let posture = self.permission_posture();
        self.ui.set_posture(posture);
    }

    /// What the permission bar says when nothing is being asked.
    ///
    /// The standing policy, in the same words the prompt will use when a tool
    /// does ask. Someone who has been reading "runs programs: ask" all week
    /// knows what the question means the moment it appears.
    fn permission_posture(&self) -> String {
        let p = &self.config.permissions;
        let session: Vec<&str> = self.grants.allowed().collect();
        let mut text = format!(
            "tools · read {} · write {} · run {}",
            p.read, p.write, p.execute
        );
        if !session.is_empty() {
            text.push_str(&format!("  ·  allowed this session: {}", session.join(", ")));
        }
        text
    }

    async fn repl(&mut self) -> Result<Flow> {
        crate::input::install_interrupt_handler();
        self.update_status();

        loop {
            let line = match self.ui.read("› ") {
                Submission::Line(l) => l,
                // Ctrl-C at the prompt abandons the line, not the session.
                Submission::Cancelled => continue,
                Submission::Eof => break,
            };
            let input = line.trim();

            if input.is_empty() {
                continue;
            }
            if looks_like_command(input) {
                // A command can change the model's settings, empty the
                // context, or load tools, so the row is stale afterwards.
                let flow = self.command(input).await?;
                self.update_status();
                match flow {
                    Flow::Continue => continue,
                    Flow::Exit => return Ok(Flow::Exit),
                    // The engine is owned by `run`, so switching unwinds to
                    // there rather than swapping it underneath us.
                    Flow::Switch(next) => {
                        self.ui.save_history();
                        return Ok(Flow::Switch(next));
                    }
                }
            }

            // Echoed into the transcript. In the scrolling REPL the terminal
            // did this for free; a full-screen application draws its own
            // rows, so without it a conversation is a column of answers to
            // questions nobody can see.
            self.echo(input);

            if let Err(e) = self.turn(input).await {
                self.ui.say(self.theme.style(Style::color(ozgent_render::Color::Red), &format!("error: {e}")));
            }
            self.update_status();
            self.ui.save_history();
        }
        self.ui.blank();
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
                    self.ui.say(self.theme.style(Style::dim(), &format!("note: {e}")));
                    self.pending_images.clear();
                }
            }
        }

        // The first message is what brings a conversation into existence.
        let conversation = self.ensure_conversation()?;
        let user_id = self.store.append_message(
            conversation,
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
            self.store.rename_conversation(conversation, &title)?;
        }

        if self.media_turn {
            self.media_observation = self.ground(&extracted.text);
        }
        let mut messages = self.build_context(Some(conversation), &extracted.text)?;
        let mut reply = self.generate(&messages)?;

        // A tool call is a request, not an answer: run it, hand the result
        // back, and let the model continue. Bounded so a model that keeps
        // calling cannot loop forever.
        // The same budget the server gives a turn. Four was enough for a
        // lookup and not for a question worth asking an agent: two searches
        // and two pages, which a research prompt spends before it has read
        // anything, and the turn then answers from what it half-saw.
        let max_calls = ozgent_web::worker::MAX_TOOL_ROUNDS;
        for round in 0..max_calls {
            let mut parsed = toolcall::extract(&reply.text);

            // The model tried to call a tool and produced something
            // unparseable. Constrained decoding cannot produce malformed JSON,
            // an unknown tool name, or a misspelled parameter, so retrying
            // under the grammar turns a failed attempt into a valid one.
            if !parsed.has_calls() && reply.attempted_call {
                if let Some(host) = &self.tools {
                    if let Some(grammar) = ozgent_llama::grammar::tool_call_grammar(host.tools()) {
                        self.ui.say(self.theme.style(Style::dim(), "· malformed tool call; retrying under grammar"));
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

                // Asked before the call runs, so the transcript never shows
                // a tool working that the user is about to refuse. A refusal
                // still goes into the history as a result: the assistant
                // message below records the call, and a call with no answer
                // leaves the conversation malformed for every later turn.
                let result = match self.permit(call).await? {
                    Some(approved) => {
                        let started = std::time::Instant::now();
                        let outcome = self.run_tool(call, approved).await;
                        self.ui.settle();
                        self.show_result(&outcome, started.elapsed());
                        match outcome {
                            Ok(value) => serde_json::to_string(&value).unwrap_or_default(),
                            Err(e) => e.for_model(),
                        }
                    }
                    None => {
                        self.ui.settle();
                        self.show_refusal(&call.name);
                        ozgent_core::permission::refusal(&call.name)
                    }
                };

                // Structured when the template can render a call itself, so
                // it writes the syntax this model was trained on. Spelt out
                // as text when it cannot: the fallback renderers read only
                // `content`, and a structured call would vanish from the
                // history entirely — and the text form is the one ozgent's
                // preamble described to the model in that case anyway.
                let (content, tool_calls) = if self.engine.template_handles_tools() {
                    (Vec::new(), vec![call.clone()])
                } else {
                    (
                        vec![ozgent_core::Part::Text {
                            text: format!(
                                "<tool_call>{{\"name\": \"{}\", \"arguments\": {}}}</tool_call>",
                                call.name, call.arguments
                            ),
                        }],
                        Vec::new(),
                    )
                };
                messages.push(Message {
                    role: Role::Assistant,
                    content,
                    // The reasoning that led to this call, so the next round
                    // sees why it was made. Dropped, the template writes an
                    // empty `<think></think>` for the turn and the model
                    // continues from a history saying it never reasoned.
                    thinking: reply.thinking.clone(),
                    tool_calls,
                    tool_call_id: None,
                });
                messages.push(Message::tool_result(call.id.clone(), truncate_result(&result)));
            }

            if round + 1 == max_calls {
                self.ui.say(self.theme.style(Style::dim(), "· tool call limit reached"));
            }
            reply = self.generate(&messages)?;
        }

        let assistant_id = self.store.append_message_full(
            conversation,
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
    fn build_context(&mut self, conversation: Option<i64>, query: &str) -> Result<Vec<Message>> {
        // Leave room for the answer, and keep the window inside the context.
        let budget = Budget {
            total: self.session.n_ctx() as usize,
            reserve_for_reply: (self.session.n_ctx() as usize / 4).max(256),
            recent_messages: 12,
            max_retrieved: 6,
        };

        // Before the first message there is no conversation and so no history
        // to assemble — the system prompt and the query are the whole context.
        let ctx = match conversation {
            Some(id) => ContextBuilder::new(&self.store, &self.embedder)
                .with_budget(budget)
                .build(id, query)?,
            None => Default::default(),
        };

        if ctx.used_recall() {
            self.ui.say(self.theme.style(
                    Style::dim(),
                    &format!("· recalled {} earlier item(s)", ctx.retrieved.len())
                ));
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

        // A template with its own tools block already tells the model the
        // format it was trained on. Adding ozgent's generic description then
        // gives it two, in different syntaxes, and it splits the difference.
        let native_tools = self.engine.template_handles_tools();

        let system = match (&dated, &self.tools) {
            (base, Some(host)) if !host.tools().is_empty() && !native_tools => {
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

        self.ui.say(self.theme.style(Style::dim(), "· looking"));
        self.ui.render();

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
        self.ui.say(self.theme.style(Style::dim(), " ✓"));

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
        let offered: &[ozgent_core::ToolSpec] =
            self.tools.as_ref().map(|h| h.tools()).unwrap_or(&[]);
        let prompt = self.engine.render_prompt_full(
            messages,
            self.opts.thinking,
            self.opts.reasoning_effort,
            offered,
        )?;
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

        let mut filter = ThinkingFilter::new(self.opts.thinking);
        if let Some(close) = Engine::stream_starts_inside(&prompt) {
            filter = filter.starting_inside(close);
        }

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
        // What has actually been shown, as opposed to what the model has
        // produced. The gates withhold a partial tool call until they know
        // whether it is one, and the screen must not flicker a half-written
        // call into view and then take it back.
        let mut shown_thinking = String::new();
        let mut shown_answer = String::new();
        let show_thinking = self.opts.thinking != ThinkingMode::Off;

        let media = self
            .projector
            .as_ref()
            .filter(|_| !pending_images.is_empty())
            .map(|p| (p, &pending_images[..], &pending_sources[..]));

        // The screen is repainted from the whole reply on every token rather
        // than appended to. Markdown cannot be appended: a closing fence
        // changes how every line since the opening one is drawn, so a renderer
        // that had already committed those lines would have to take them back.
        // `Ui::stream` throttles the repaint, so this is cheap.
        let ui = &mut *self.ui;
        let session = &mut self.session;
        let (stats, reason) = session.generate_with_media(&prompt, media, self.opts.max_tokens, |piece| {
            for chunk in filter.push(piece) {
                match chunk {
                    Chunk::Thinking(text) => {
                        thinking.push_str(&text);
                        if show_thinking {
                            shown_thinking.push_str(&think_gate.push(&text));
                        }
                    }
                    Chunk::Answer(text) => {
                        answer.push_str(&text);
                        // Gated: tool-call syntax is a request to the runtime,
                        // not prose, and must never reach the terminal.
                        shown_answer.push_str(&gate.push(&text));
                    }
                }
            }
            ui.stream(Some(shown_thinking.as_str()), &shown_answer, false);
            // Polled between tokens, so Ctrl-C stops the answer not the
            // process — and so a resize or a page-up is noticed mid-reply.
            !ui.poll_interrupt()
        })?;

        for chunk in filter.finish() {
            if let Chunk::Answer(text) = chunk {
                answer.push_str(&text);
                shown_answer.push_str(&gate.push(&text));
            }
        }
        shown_answer.push_str(&gate.finish());
        // Forced, because the last token would otherwise sit unpainted behind
        // the throttle until something else happened to redraw.
        self.ui.stream(Some(shown_thinking.as_str()), &shown_answer, true);
        self.ui.commit();

        if crate::input::interrupted() {
            self.ui.say(self.theme.style(Style::dim(), "· interrupted"));
        }
        // Three different situations used to share one message, and it was
        // wrong for two of them.
        if stats.generated_tokens > 0 && !crate::input::interrupted() {
            let note = if filter.is_thinking() && !thinking.trim().is_empty() {
                // The text is above, styled as reasoning, because that is how
                // it arrived. Say why rather than leave it looking unanswered.
                Some("· the model never closed its reasoning tag; the text above is its reply")
            } else if answer.trim().is_empty() && !thinking.trim().is_empty() {
                Some("· reasoning finished with no reply. Try /think off, or a longer /effort")
            } else if answer.trim().is_empty() {
                Some("· the model produced nothing")
            } else {
                None
            };
            if let Some(note) = note {
                self.ui.say(self.theme.style(Style::dim(), note));
            }
        }
        // Kept for the status line. Only a real generation updates it: an
        // interrupted or empty turn would otherwise report a rate measured
        // over a handful of tokens.
        if stats.generated_tokens > 0 {
            self.last_rate = Some(stats.tokens_per_second());
        }
        if reason == StopReason::ContextFull {
            self.ui.say(self.theme.style(Style::dim(), "· context full"));
        }
        if self.opts_show_stats() {
            self.ui.say(self.theme.style(
                    Style::dim(),
                    &format!(
                        "· {} in ({:.0}/s), {} reused · {} out ({:.1}/s)",
                        stats.prompt_tokens,
                        stats.prompt_tokens_per_second(),
                        stats.reused_tokens,
                        stats.generated_tokens,
                        stats.tokens_per_second()
                    )
                ));
        }

        // A reasoning block the model never closed was not a reasoning block.
        // It reached the end of its turn still inside `<think>`, so the words
        // in there are its reply, mislabelled by a tag it did not emit.
        // Treating them as reasoning and returning an empty answer loses a
        // complete response and writes an empty assistant message into the
        // conversation, which then poisons every later turn.
        let unclosed = filter.is_thinking() && !thinking.trim().is_empty();

        let (text, reasoning) = if toolcall::extract(&answer).has_calls() {
            (answer, Some(thinking))
        } else if unclosed {
            (format!("{answer}{thinking}"), None)
        } else if think_gate.suppressing() {
            // Closed its reasoning, but the call was inside it. The parser has
            // to see the call; the reasoning is still reasoning.
            (format!("{answer}{thinking}"), Some(thinking))
        } else {
            (answer, Some(thinking))
        };

        Ok(Reply {
            text,
            // An empty reasoning block is not a reasoning trace.
            thinking: reasoning.filter(|t| !t.trim().is_empty()),
            attempted_call: gate.suppressing() || think_gate.suppressing(),
        })
    }

    fn opts_show_stats(&self) -> bool {
        std::env::var("OZGENT_STATS").is_ok()
    }

    /// Decide whether a call may run, asking the user if the policy says to.
    ///
    /// Returns `None` for a refusal, or `Some(by_user)` to run it — where
    /// `by_user` says a person authorised this call, which is what lets the
    /// Python side treat it as past its own standing boundaries.
    async fn permit(&mut self, call: &ozgent_core::ToolCall) -> Result<Option<bool>> {
        use ozgent_core::permission::Verdict;

        let effect = self
            .tools
            .as_ref()
            .and_then(|h| h.get(&call.name))
            .map(|spec| spec.effect)
            // A call to a tool that does not exist fails in the host a moment
            // later with a much better message than anything here could give.
            .unwrap_or_default();

        match self.config.permissions.verdict(&call.name, effect, &self.grants) {
            Verdict::Allow { by_user } => return Ok(Some(by_user)),
            Verdict::Deny => return Ok(None),
            Verdict::Ask => {}
        }

        let choice = self.ui.ask_permission(&call.name, effect, &call.arguments);
        self.grants.remember(&call.name, choice);
        // "Always" is the one answer that outlives the process, so it is the
        // one that touches the file.
        if self.config.permissions.apply(&call.name, choice) {
            self.config.save(&self.paths)?;
        }
        Ok(choice.is_allow().then_some(true))
    }

    /// Put what the user typed into the transcript.
    ///
    /// Marked with the same `›` the prompt box wears, so the eye can find
    /// where each exchange started when scrolling back through a long thread.
    fn echo(&mut self, text: &str) {
        let marker = self.theme.style(Style::color(ozgent_render::Color::Cyan), "› ");
        let body = self.theme.style(Style { bold: true, ..Default::default() }, text);
        self.ui.blank();
        self.ui.say(format!("{marker}{body}"));
        self.ui.blank();
    }

    /// Run one tool, keeping the screen alive while it works.
    ///
    /// A web search takes seconds and a page fetch can take most of the tool
    /// timeout. Awaiting it plainly leaves a still screen for that whole time,
    /// which is indistinguishable from a hang — so the spinner on the call's
    /// own line is advanced until the call returns.
    ///
    /// The two borrows are of different fields, which is what makes this
    /// legal: the host is read while the screen is written.
    async fn run_tool(
        &mut self,
        call: &ozgent_core::ToolCall,
        approved: bool,
    ) -> Result<serde_json::Value, ozgent_tools::ToolCallError> {
        let host = self.tools.as_ref().expect("checked above");
        let ui = &mut *self.ui;

        let running = host.call_approved(&call.name, call.arguments.clone(), approved);
        tokio::pin!(running);
        // The first tick fires immediately, which would advance the spinner
        // before any time had passed; starting a period late keeps the line
        // still for calls that return at once.
        let mut spin = tokio::time::interval_at(
            tokio::time::Instant::now() + crate::tui::ui::SPIN,
            crate::tui::ui::SPIN,
        );
        loop {
            tokio::select! {
                outcome = &mut running => return outcome,
                _ = spin.tick() => ui.tick(),
            }
        }
    }

    /// Say that a call did not run, in the same shape as a result.
    ///
    /// Shaped like `show_result` on purpose: a refusal is an outcome of the
    /// call, and putting it anywhere else makes the transcript look like the
    /// tool is still running.
    fn show_refusal(&mut self, name: &str) {
        let arrow = self.theme.style(Style::color(ozgent_render::Color::Red), "  ⎿");
        self.ui.say(format!(
            "{arrow} {}",
            self.theme.style(Style::dim(), &format!("declined · {name} did not run"))
        ));
    }

    /// Announce a tool call: a green marker, the name, then its arguments.
    ///
    /// Arguments are shown as `key: value` rather than raw JSON, since the
    /// braces and quotes carry no information the user needs.
    fn show_call(&mut self, call: &ozgent_core::ToolCall) {
        let name = self.theme.style(
            ozgent_render::Style { bold: true, ..Default::default() },
            &call.name,
        );
        let args = self.theme.style(Style::dim(), &pretty_args(&call.arguments));
        // The marker column belongs to the screen, which turns it into a
        // spinner while the call runs and back into a dot when it returns.
        self.ui.begin_activity(format!("{name}{args}"));
    }

    /// Report what a tool returned, indented under its call.
    fn show_result(
        &mut self,
        outcome: &Result<serde_json::Value, ozgent_tools::ToolCallError>,
        elapsed: std::time::Duration,
    ) {
        let (colour, text) = match outcome {
            Ok(value) => (ozgent_render::Color::Green, summarise_result(value)),
            Err(e) => (ozgent_render::Color::Red, ozgent_tools::first_line(&e.for_model()).to_string()),
        };
        let arrow = self.theme.style(Style::color(colour), "  ⎿");
        let timing = self.theme.style(Style::dim(), &format!(" · {}ms", elapsed.as_millis()));
        self.ui.say(format!("{arrow} {}{timing}", self.theme.style(Style::dim(), &text)));
    }

    /// The conversation being written to, creating it if this is the first
    /// message.
    fn ensure_conversation(&mut self) -> Result<i64> {
        if let Some(id) = self.conversation {
            return Ok(id);
        }
        let id = self
            .store
            .create_conversation("", Some(&self.model.to_string()))?;
        self.conversation = Some(id);
        Ok(id)
    }

    /// List past conversations, reopen one, or delete one.
    ///
    ///   /conv           list
    ///   /conv 3         reopen the third
    ///   /conv rm 3      delete it
    ///   /conv prune     drop every empty one left by older versions
    ///
    /// Reopening is only a change of id: the prompt is rebuilt from the store
    /// on every turn, so the history comes back on its own and the engine's
    /// prefix check refuses the stale checkpoint by itself.
    fn conversations(&mut self, arg: &str) -> Result<Flow> {
        // A clone, not a borrow of `self.theme`: writing to the screen
        // takes `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |t: &str| theme.style(Style::dim(), t);
        let mut parts = arg.split_whitespace();
        let verb = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("");

        let listed = self.store.list_active_conversations(50)?;

        match verb {
            "" => {
                if listed.is_empty() {
                    self.ui.say(dim("no conversations yet; send a message to start one"));
                    return Ok(Flow::Continue);
                }
                self.ui.say(dim("conversations:"));
                for (i, c) in listed.iter().enumerate() {
                    let marker = if Some(c.id) == self.conversation { "*" } else { " " };
                    self.ui.say(dim(&format!("{marker} {:>2}. {}", i + 1, describe(c))));
                }
                let empties = self.store.empty_conversation_count()?;
                // Only worth mentioning when there is a mess to clean up; one
                // empty row is the live conversation and is not news.
                if empties > 1 {
                    self.ui.say(dim(&format!("{empties} empty conversations · /conv prune to remove them")));
                }
                self.ui.say(dim("/conv <number> to reopen · /conv rm <number> to delete"));
            }

            "rm" | "delete" | "del" => {
                let Some(target) = pick(&listed, rest) else {
                    self.ui.say(dim(&format!("no conversation {rest:?}; /conv to list")));
                    return Ok(Flow::Continue);
                };
                let (id, label) = (target.id, describe(target));
                self.store.delete_conversation(id)?;
                // Deleting the one being written to leaves the chat with no
                // conversation rather than a dangling id; the next message
                // starts a fresh one.
                if self.conversation == Some(id) {
                    self.conversation = None;
                    self.session.reset();
                    self.turn = 0;
                    self.ui.say(dim(&format!("· deleted {label} (was the current one)")));
                } else {
                    self.ui.say(dim(&format!("· deleted {label}")));
                }
            }

            "prune" => {
                let removed = self.store.delete_empty_conversations(self.conversation)?;
                self.ui.say(dim(&format!("· removed {removed} empty conversation(s)")));
            }

            "new" => {
                self.conversation = None;
                self.session.reset();
                self.turn = 0;
                self.ui.clear();
                self.banner();
                self.ui.say(dim("· new conversation"));
            }

            number => {
                let Some(target) = pick(&listed, number) else {
                    self.ui.say(dim(&format!("no conversation {number:?}; /conv to list")));
                    return Ok(Flow::Continue);
                };
                if self.conversation == Some(target.id) {
                    self.ui.say(dim("already in that conversation"));
                    return Ok(Flow::Continue);
                }
                let (id, model) = (target.id, target.model.clone());
                self.conversation = Some(id);
                self.turn = self.store.message_count(id)? as usize;
                // The KV cache holds the previous conversation's prompt, and
                // none of it is a prefix of this one.
                self.session.reset();
                // Same reasoning as /new: the thread being left behind must
                // not stay on screen above the one being opened.
                clear_screen();
                self.banner();
                self.ui.say(dim(&format!("· {} ({} messages)", describe(target), self.turn)));
                self.recap(id)?;

                // Not switched automatically: reloading weights is a ten-second
                // surprise for someone who only asked to look at a thread.
                if let Some(model) = model.filter(|m| *m != self.model.to_string()) {
                    self.ui.say(dim(&format!("  was {model} · /models {model} to switch back")));
                }
            }
        }
        Ok(Flow::Continue)
    }

    /// Print the tail of a conversation, so reopening one lands somewhere
    /// recognisable rather than at a bare prompt.
    fn recap(&mut self, conversation: i64) -> Result<()> {
        // A clone, not a borrow of `self.theme`: writing to the screen
        // takes `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |t: &str| theme.style(Style::dim(), t);
        let recent = self.store.recent_messages(conversation, 4)?;
        for m in &recent {
            let who = match m.role.as_str() {
                "user" => "you",
                "assistant" => self.model.name.as_str(),
                other => other,
            };
            self.ui.say(dim(&format!("  {who}: {}", one_line(&m.content, 72))));
        }
        Ok(())
    }

    /// Show installed models and pick one to switch to.
    fn pick_model(&mut self, arg: &str) -> Result<Option<String>> {
        let models = ozgent_core::installed(&self.paths);
        // A clone, not a borrow of `self.theme`: writing to the screen
        // takes `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |t: &str| theme.style(Style::dim(), t);

        if models.is_empty() {
            self.ui.say(dim("no models installed. Try: ozgent pull <repo> --name <short>"));
            return Ok(None);
        }

        // A direct argument skips the picker: `/models coder`, or `/models 2`.
        if !arg.is_empty() {
            if let Ok(n) = arg.parse::<usize>() {
                let Some(chosen) = models.get(n.wrapping_sub(1)) else {
                    self.ui.say(dim(&format!("no model {n}; there are {}", models.len())));
                    return Ok(None);
                };
                if chosen.model == self.model {
                    self.ui.say(dim("already using that model"));
                    return Ok(None);
                }
                return Ok(Some(chosen.short_name()));
            }
            let found = ozgent_core::resolve(&self.paths, arg)?;
            if found.model == self.model {
                self.ui.say(dim(&format!("already using {}", found.model)));
                return Ok(None);
            }
            return Ok(Some(found.short_name()));
        }

        self.ui.say(dim("installed models:"));
        for (i, m) in models.iter().enumerate() {
            let marker = if m.model == self.model { "*" } else { " " };
            let alias = m.manifest.alias.as_ref().map(|a| format!("  ({a})")).unwrap_or_default();
            self.ui.say(dim(&format!("{marker} {:>2}. {}{alias}", i + 1, m.model)));
        }
        self.ui.say(dim("switch with /models <number|alias|name:tag>"));
        Ok(None)
    }

    /// Show or change settings for the current model, persisting them.
    ///
    /// The same names as the command-line flags, minus their dashes, so that
    /// `--top-p 0.9` and `/config top_p 0.9` are one thing to learn.
    fn configure(&mut self, arg: &str) -> Result<()> {
        // A clone, not a borrow of `self.theme`: writing to the screen
        // takes `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |t: &str| theme.style(Style::dim(), t);
        let mut parts = arg.split_whitespace();
        let key = parts.next().unwrap_or("");
        let value = parts.collect::<Vec<_>>().join(" ");

        if key.is_empty() {
            self.ui.say(dim(&format!("settings for {}:", self.model)));
            self.ui.say(dim(&format!("  thinking        {:?}", self.opts.thinking)));
            self.ui.say(dim(&format!("  effort          {:?}", self.opts.reasoning_effort)));
            self.ui.say(dim(&format!("  temperature     {}", self.opts.temperature)));
            self.ui.say(dim(&format!("  top_p           {}", self.opts.top_p)));
            self.ui.say(dim(&format!("  top_k           {}", self.opts.top_k)));
            self.ui.say(dim(&format!("  min_p           {}", self.opts.min_p)));
            self.ui.say(dim(&format!("  repeat_penalty  {}", self.opts.repeat_penalty)));
            self.ui.say(dim(&format!("  max_tokens      {}", self.opts.max_tokens)));
            let seed = self.opts.seed.map_or("random".to_string(), |s| s.to_string());
            self.ui.say(dim(&format!("  seed            {seed}")));
            // Both numbers, because they can differ: what was asked for is
            // what `/config ctx` set, and what the session runs on is what
            // the machine's memory allowed.
            let asked = ozgent_core::format_count(self.opts.context_length);
            let window = self.session.n_ctx();
            let ctx = if window == self.opts.context_length {
                asked
            } else {
                format!("{asked} (running at {})", ozgent_core::format_count(window))
            };
            self.ui.say(dim(&format!("  ctx             {ctx}")));
            self.ui.say(dim(&format!("  gpu layers      {}", self.opts.gpu_layers)));
            self.ui.say(dim(&format!("  tools           {}", self.opts.tools)));
            self.ui.say(dim(&format!("set with /config <key> <value>; keys: {SETTABLE}")));
            return Ok(());
        }
        if value.is_empty() {
            self.ui.say(dim(&format!("usage: /config {key} <value>")));
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
            "top_p" | "top-p" => {
                let p: f32 = value.parse().context("top_p must be a number")?;
                self.opts.top_p = p;
                layer.top_p = Some(p);
            }
            "top_k" | "top-k" => {
                let k: u32 = value.parse().context("top_k must be a whole number")?;
                self.opts.top_k = k;
                layer.top_k = Some(k);
            }
            "min_p" | "min-p" => {
                let p: f32 = value.parse().context("min_p must be a number")?;
                self.opts.min_p = p;
                layer.min_p = Some(p);
            }
            "repeat_penalty" | "repeat-penalty" => {
                let p: f32 = value.parse().context("repeat_penalty must be a number")?;
                self.opts.repeat_penalty = p;
                layer.repeat_penalty = Some(p);
            }
            "max_tokens" | "max-tokens" => {
                let n = ozgent_core::parse_count(&value).map_err(|e| anyhow::anyhow!(e))?;
                self.opts.max_tokens = n;
                layer.max_tokens = Some(n);
            }
            "seed" => {
                let n: u32 = value.parse().context("seed must be a whole number")?;
                self.opts.seed = Some(n);
                layer.seed = Some(n);
            }
            // Context length is fixed when the weights are loaded, so this
            // one is saved but deliberately not applied to the live session:
            // claiming a change the KV cache never saw would be a lie the
            // user only discovers when the model runs out of room.
            "ctx" | "context" | "context_length" => {
                let n = ozgent_core::parse_count(&value).map_err(|e| anyhow::anyhow!(e))?;
                layer.context_length = Some(n);
                let entry = self.config.models.entry(self.model.to_string()).or_default();
                *entry = entry.clone().merge(&layer);
                self.config.save(&self.paths)?;
                self.ui.say(dim(&format!(
                        "· ctx = {n} (saved; applies when the model is next loaded, \
                         currently {})",
                        ozgent_core::format_count(self.session.n_ctx())
                    )));
                return Ok(());
            }
            "tools" => {
                let on = matches!(value.as_str(), "on" | "true" | "yes" | "1");
                self.opts.tools = on;
                layer.tools = Some(on);
            }
            other => {
                self.ui.say(dim(&format!("unknown setting {other:?}; try {SETTABLE}")));
                return Ok(());
            }
        }

        // Sampling changes only reach generation through the sampler, which is
        // rebuilt from these options; without this the value is printed as set
        // and every following turn still uses the old one.
        self.session.set_options(&self.opts);

        let entry = self.config.models.entry(self.model.to_string()).or_default();
        *entry = entry.clone().merge(&layer);
        self.config.save(&self.paths)?;
        self.ui.say(dim(&format!("· {key} = {value} (saved)")));
        Ok(())
    }

    /// Show or change what tools may do without asking.
    ///
    /// The listing shows the rule in force for every tool, whether or not it
    /// was set by name — "what happens if this is called" is the question, and
    /// an inherited rule answers it just as much as an override does.
    fn permissions(&mut self, arg: &str) -> Result<()> {
        use ozgent_core::permission::Rule;

        // A clone, not a borrow of `self.theme`: writing to the screen
        // takes `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |t: &str| theme.style(Style::dim(), t);
        let mut parts = arg.split_whitespace();
        let target = parts.next().unwrap_or("");
        let value = parts.next().unwrap_or("");

        if target.is_empty() {
            let policy = &self.config.permissions;
            self.ui.say(dim("by what a tool does:"));
            for (label, rule) in [
                ("reads", policy.read),
                ("writes", policy.write),
                ("runs programs", policy.execute),
                ("does not say", policy.unknown),
            ] {
                self.ui.say(dim(&format!("  {label:<14}  {rule}")));
            }

            match &self.tools {
                Some(host) if !host.tools().is_empty() => {
                    self.ui.say(dim("by tool:"));
                    for spec in host.tools() {
                        let rule = policy.rule_for(&spec.name, spec.effect);
                        // Marked so that clearing an override is a visible
                        // change rather than something that appears to do
                        // nothing.
                        let how = if policy.is_overridden(&spec.name) {
                            "set here"
                        } else {
                            "from its kind"
                        };
                        let session = if self.grants.allowed().any(|t| t == spec.name) {
                            "  · allowed for this session"
                        } else {
                            ""
                        };
                        self.ui.say(dim(&format!(
                                "  {:<16} {:<8} {:<8} ({how}){session}",
                                spec.name,
                                spec.effect.to_string(),
                                rule.to_string(),
                            )));
                    }
                }
                _ => self.ui.say(dim("no tools are loaded")),
            }
            self.ui.say(dim("/permissions <tool> allow|ask|deny · /permissions <tool> clear · \
                     /permissions <kind> <rule> where kind is read|write|execute|unknown"));
            return Ok(());
        }

        if value.is_empty() {
            self.ui.say(dim(&format!("usage: /permissions {target} allow|ask|deny|clear")));
            return Ok(());
        }

        // A kind before a tool: the four kinds are fixed names, and a tool
        // called "write" would otherwise be unreachable in the other form.
        if let Ok(effect) = target.parse::<ozgent_core::Effect>() {
            let rule: Rule = value.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            let policy = &mut self.config.permissions;
            match effect {
                ozgent_core::Effect::Read => policy.read = rule,
                ozgent_core::Effect::Write => policy.write = rule,
                ozgent_core::Effect::Execute => policy.execute = rule,
                ozgent_core::Effect::Unknown => policy.unknown = rule,
            }
            self.config.save(&self.paths)?;
            self.ui.say(dim(&format!("· tools that {effect}: {rule} (saved)")));
            return Ok(());
        }

        let known = self.tools.as_ref().is_some_and(|h| h.get(target).is_some());
        if !known {
            // Not an error: a rule can be set for a tool that is not loaded
            // right now, and refusing would make the manager useless whenever
            // tools are switched off.
            self.ui.say(dim(&format!("· {target} is not loaded; setting it anyway")));
        }
        if value == "clear" || value == "default" {
            self.config.permissions.set(target, None);
            self.config.save(&self.paths)?;
            self.ui.say(dim(&format!("· {target} follows its kind again (saved)")));
            return Ok(());
        }
        let rule: Rule = value.parse().map_err(|e: String| anyhow::anyhow!(e))?;
        self.config.permissions.set(target, Some(rule));
        self.config.save(&self.paths)?;
        self.ui.say(dim(&format!("· {target}: {rule} (saved)")));
        Ok(())
    }

    /// Point a tool at a provider, e.g. `/tools web_search brave`.
    fn configure_tool(&mut self, arg: &str) -> Result<()> {
        // A clone, not a borrow of `self.theme`: writing to the screen
        // takes `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |t: &str| theme.style(Style::dim(), t);
        let mut parts = arg.split_whitespace();
        let tool = parts.next().unwrap_or("").to_string();
        let provider = parts.next().map(str::to_string);

        let Some(provider) = provider else {
            self.ui.say(dim(&format!("usage: /tools {tool} <provider>")));
            if tool == "web_search" {
                self.ui.say(dim("  providers: brave, tavily, duckduckgo"));
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
                self.ui.say(dim(&format!("{provider} needs an API key (or set ${var}).")));
                let line = self.ui.ask_text("api key (blank to skip): ").unwrap_or_default();
                if !line.is_empty() {
                    key = Some(line);
                }
            }
        }

        set_tool_provider(&mut self.config, &tool, &provider, key.as_deref());
        self.config.save(&self.paths)?;
        harden_config_permissions(&self.paths.config_file());

        self.ui.say(dim(&format!("· {tool} now uses {provider} (saved)")));
        self.ui.say(dim("restart the chat for the tool worker to pick it up"));
        Ok(())
    }

    /// Force the next reply to be a tool call, for `/call`.
    async fn forced_call(&mut self, query: &str) -> Result<()> {
        let Some(host) = &self.tools else {
            self.ui.say(self.theme.style(Style::dim(), "tools are disabled"));
            return Ok(());
        };
        let Some(grammar) = ozgent_llama::grammar::tool_call_grammar(host.tools()) else {
            self.ui.say(self.theme.style(Style::dim(), "no tools available"));
            return Ok(());
        };

        // `/call` is a probe: it uses the conversation for context when there
        // is one, but must not bring one into being.
        let mut messages = self.build_context(self.conversation, query)?;
        messages.push(Message::user(format!(
            "Call the most appropriate tool to answer: {query}"
        )));
        let reply = self.generate_with(&messages, Some(&grammar))?;

        let parsed = toolcall::extract(&reply.text);
        if parsed.calls.is_empty() {
            self.ui.say(self.theme.style(Style::dim(), "· the model produced no call"));
            return Ok(());
        }
        for call in &parsed.calls {
            self.ui.say(self.theme.style(Style::dim(), &format!("· {}({})", call.name, compact(&call.arguments))));
            // `/call` is the user asking directly, which is consent for this
            // one call — but the standing policy still decides, so a tool set
            // to `deny` stays denied rather than being reachable by typing a
            // different command.
            let approved = match self.permit(call).await? {
                Some(by_user) => by_user,
                None => {
                    self.show_refusal(&call.name);
                    return Ok(());
                }
            };
            // The same spinner as the model's own calls: `/call` runs the
            // identical tool and can take just as long.
            self.ui.begin_activity(self.theme.style(Style::dim(), &call.name.clone()));
            let outcome = self.run_tool(call, approved).await;
            self.ui.settle();
            match outcome {
                Ok(v) => {
                    // Into the transcript, not stdout: in full-screen mode
                    // stdout is the screen, and writing to it directly would
                    // scroll the layout apart.
                    let text = serde_json::to_string_pretty(&v).unwrap_or_default();
                    self.ui.markdown(format!("```json\n{text}\n```"));
                }
                Err(e) => self.ui.say(self.theme.style(Style::dim(), &e.for_model())),
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
        // A clone, not a borrow of `self.theme`: writing to the screen
        // takes `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |s: &str| theme.style(Style::dim(), s);

        match cmd {
            "/exit" | "/quit" | "/q" => return Ok(Flow::Exit),

            "/help" | "/h" => {
                self.ui.say(dim(HELP));
            }

            "/clear" | "/new" => {
                // Nothing is created here either: the next message does it.
                // Otherwise `/clear` typed twice leaves an orphan behind.
                self.conversation = None;
                self.session.reset();
                self.turn = 0;
                // Wipe the screen too. The old thread staying on screen under
                // a one-line notice reads as "still in that conversation",
                // which is the opposite of what just happened.
                self.ui.clear();
                self.banner();
                self.ui.say(dim("· new conversation"));
            }

            "/conv" | "/convs" | "/conversations" => return self.conversations(arg),

            "/think" => match arg {
                "" => self.ui.say(dim(&format!("thinking: {:?}", self.opts.thinking))),
                other => match other.parse::<ThinkingMode>() {
                    Ok(mode) => {
                        self.opts.thinking = mode;
                        self.ui.say(dim(&format!("· thinking {other}")));
                    }
                    Err(e) => self.ui.say(dim(&e)),
                },
            },

            "/effort" => match arg {
                "" => self.ui.say(dim(&format!("effort: {:?}", self.opts.reasoning_effort))),
                other => match other.parse::<ozgent_core::ReasoningEffort>() {
                    Ok(level) => {
                        self.opts.reasoning_effort = level;
                        self.ui.say(dim(&format!("· effort {other}")));
                    }
                    Err(e) => self.ui.say(dim(&e)),
                },
            },

            "/system" => {
                if arg.is_empty() {
                    match &self.opts.system_prompt {
                        Some(s) => self.ui.say(dim(s)),
                        None => self.ui.say(dim("no system prompt set")),
                    }
                } else {
                    self.opts.system_prompt = Some(arg.to_string());
                    self.ui.say(dim("· system prompt updated"));
                }
            }

            "/remember" => {
                if arg.is_empty() {
                    self.ui.say(dim("usage: /remember <fact>"));
                } else {
                    // Scoped to the conversation when there is one; before
                    // the first message there is nothing to scope it to, and
                    // an unscoped fact is the right reading of "remember this
                    // from now on".
                    let id = self.store.add_fact(
                        self.conversation,
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
                    self.ui.say(dim("· remembered"));
                }
            }

            "/memory" => {
                let (facts, count) = match self.conversation {
                    Some(id) => {
                        (self.store.facts_for(id)?, self.store.message_count(id)?)
                    }
                    None => (Vec::new(), 0),
                };
                self.ui.say(dim(&format!("{count} messages, {} facts", facts.len())));
                for f in facts.iter().take(20) {
                    let mark = if f.pinned { "*" } else { " " };
                    self.ui.say(dim(&format!("  {mark} {}", f.text)));
                }
            }

            "/call" => {
                if arg.is_empty() {
                    self.ui.say(dim("usage: /call <what you want done>"));
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

            "/permissions" | "/perms" => self.permissions(arg)?,

            "/default" => {
                // The alias when there is one: it is the name the user chose,
                // it is what `ozgent list` shows, and it survives re-pulling
                // the model at another quantisation.
                let name = self
                    .manifest
                    .alias
                    .clone()
                    .unwrap_or_else(|| self.model.to_string());
                if arg == "clear" || arg == "off" {
                    self.config.default_model = None;
                    self.config.save(&self.paths)?;
                    self.ui.say(dim("· no default model; name one each time"));
                } else {
                    self.config.default_model = Some(name.clone());
                    self.config.save(&self.paths)?;
                    self.ui.say(dim(&format!("· {name} is now the default for `ozgent chat` and the web interface")));
                }
            }

            "/tools" if !arg.is_empty() => self.configure_tool(arg)?,

            "/tools" => match &self.tools {
                Some(host) => {
                    self.ui.say(dim(&format!("{} tools", host.tools().len())));
                    for t in host.tools() {
                        self.ui.say(dim(&format!("  {}  {}", t.name, ozgent_tools::first_line(&t.description))));
                    }
                }
                None => self.ui.say(dim("tools are disabled")),
            },

            "/stats" => {
                self.ui.say(dim(&format!(
                        "{} · {} layers ({} gpu) · {} ctx, {} used, {} reused last turn · thinking {:?}",
                        self.model,
                        self.engine.n_layer(),
                        self.engine.gpu_layers_used(),
                        self.session.n_ctx(),
                        self.session.used(),
                        self.session.last_reused(),
                        self.opts.thinking
                    )));
            }

            other => self.ui.say(dim(&format!("unknown command {other}; try /help"))),
        }
        Ok(Flow::Continue)
    }
}

/// What one generation produced, after the reasoning and tool-call streams
/// have been told apart.
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



/// Clear the terminal and put the cursor at the top.
///
/// Written straight to the terminal rather than through the theme: this is a
/// cursor movement, not styling, and a plain-output run still wants a clean
/// screen. Skipped when stderr is not a terminal, where the escapes would end
/// up in whatever is capturing the output.
pub fn clear_screen() {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() {
        return;
    }
    // Erase the scrollback as well as the screen: without `3J` the old
    // conversation is one scroll away and still looks current.
    eprint!("\x1b[H\x1b[2J\x1b[3J");
    std::io::stderr().flush().ok();
}

/// One line describing a conversation, for the `/conv` listing.
fn describe(c: &ozgent_memory::Conversation) -> String {
    let title = if c.title.trim().is_empty() {
        "(untitled)".to_string()
    } else {
        one_line(&c.title, 48)
    };
    format!("{title}  ·  {} msg  ·  {}", c.message_count, ago(c.updated_at))
}

/// Resolve a `/conv` argument — currently a 1-based position in the listing.
fn pick<'c>(listed: &'c [ozgent_memory::Conversation], arg: &str) -> Option<&'c ozgent_memory::Conversation> {
    let n: usize = arg.trim().parse().ok()?;
    listed.get(n.checked_sub(1)?)
}

/// Collapse to a single line and cap it, so a pasted paragraph stays one row.
fn one_line(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        return flat;
    }
    // Counted in characters, not bytes: slicing a multi-byte character in
    // half panics, and titles are whatever the user typed.
    let cut: String = flat.chars().take(max.saturating_sub(1)).collect();
    format!("{cut}…")
}

/// A coarse "when", accurate enough to tell threads apart in a list.
fn ago(timestamp: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(timestamp);
    let seconds = now.saturating_sub(timestamp);
    match seconds {
        // A clock that disagrees with the database is not worth a negative
        // duration; `saturating_sub` lands it here.
        s if s < 60 => "just now".to_string(),
        s if s < 3600 => format!("{}m ago", s / 60),
        s if s < 86_400 => format!("{}h ago", s / 3600),
        s if s < 86_400 * 30 => format!("{}d ago", s / 86_400),
        s => format!("{}mo ago", s / (86_400 * 30)),
    }
}

/// Setting names `/config` accepts, listed for the user in one place so the
/// listing, the error and `/help` cannot drift apart.
const SETTABLE: &str =
    "thinking, effort, temperature, top_p, top_k, min_p, repeat_penalty, max_tokens, seed, \
     ctx, tools";

const HELP: &str = "\
/help              this list
/exit              quit
/new, /clear       start a new conversation and clear the screen
/conv              list past conversations
/conv <n>          reopen one · /conv rm <n> delete · /conv prune drop empties
/think on|off|auto show or suppress reasoning
/effort low|med|high how long the model may reason
/system <text>     set the system prompt
/remember <fact>   pin a fact for this and future chats
/memory            what is remembered
/models [name]     list models, or switch to one
/config [k] [v]    show or change this model's settings, saved for next time
                   temperature, top_p, top_k, min_p, repeat_penalty, max_tokens,
                   seed, thinking, effort, tools, and ctx (applies on next load)
                   sizes take k/m: /config ctx 32k
/tools             list available tools
/default           use this model when none is named · /default clear to unset
/permissions       what tools may do without asking
/permissions <tool> allow|ask|deny|clear   ·  or read|write|execute <rule>
/tools <t> <prov>  point a tool at a provider, e.g. /tools web_search brave
/call <request>    force a tool call, constrained by grammar
/stats             model and context state

Scrolling    wheel, PageUp/PageDown, or Shift-Up/Shift-Down
             Esc returns to the newest message
             Shift-drag to select text, since the wheel belongs to ozgent
Editing      Alt-Enter for a new line · Ctrl-A/E/K/U/W as in any shell
             Up/Down walk what you typed before

Paste an image path or URL in a message and it is picked up automatically.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_lists_every_command_the_parser_accepts() {
        // A command that exists but is undocumented is invisible to the user.
        for cmd in ["/help", "/exit", "/clear", "/new", "/conv", "/think", "/effort", "/system", "/remember", "/memory", "/tools", "/stats", "/default", "/permissions"] {
            assert!(HELP.contains(cmd), "{cmd} is missing from /help");
        }
    }

    #[test]
    fn one_line_flattens_and_caps() {
        assert_eq!(one_line("a\n  b   c", 40), "a b c");
        assert_eq!(one_line("hello", 5), "hello", "an exact fit is not truncated");
        assert_eq!(one_line("hello there", 6), "hello…");
    }

    #[test]
    fn one_line_counts_characters_not_bytes() {
        // Slicing a multi-byte character in half panics, and a title is
        // whatever the user typed.
        let text = "héllo wörld ünd mehr";
        let out = one_line(text, 8);
        assert_eq!(out.chars().count(), 8, "{out}");
    }

    #[test]
    fn pick_is_one_based_and_refuses_nonsense() {
        let listed = vec![conversation(1, "first"), conversation(2, "second")];
        assert_eq!(pick(&listed, "1").map(|c| c.id), Some(1));
        assert_eq!(pick(&listed, "2").map(|c| c.id), Some(2));
        for bad in ["0", "3", "", "rm", "-1", "1.5"] {
            assert!(pick(&listed, bad).is_none(), "{bad:?} must not resolve");
        }
    }

    #[test]
    fn an_untitled_conversation_still_reads_as_something() {
        let c = conversation(1, "");
        assert!(describe(&c).contains("(untitled)"), "{}", describe(&c));
    }

    #[test]
    fn ago_reads_in_the_largest_useful_unit() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert_eq!(ago(now), "just now");
        assert_eq!(ago(now - 300), "5m ago");
        assert_eq!(ago(now - 7200), "2h ago");
        assert_eq!(ago(now - 86_400 * 3), "3d ago");
    }

    #[test]
    fn a_timestamp_from_the_future_does_not_wrap_around() {
        // A database written on a machine whose clock ran ahead must not
        // produce a duration of eighteen quintillion seconds.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert_eq!(ago(now + 10_000), "just now");
    }

    fn conversation(id: i64, title: &str) -> ozgent_memory::Conversation {
        ozgent_memory::Conversation {
            id,
            uuid: String::new(),
            title: title.to_string(),
            model: None,
            created_at: 0,
            updated_at: 0,
            message_count: 2,
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
            effect: ozgent_core::Effect::Read,
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
