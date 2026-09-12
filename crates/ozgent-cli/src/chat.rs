//! The interactive chat loop.
//!
//! This is where the finished pieces meet: the engine generates, memory
//! decides what the model is allowed to remember, the Python worker runs
//! tools, and the renderer draws it. Each of those is exercised elsewhere in
//! isolation; this module is the wiring.

use anyhow::{Context, Result};
use ozgent_core::{Config, Paths, ThinkingMode};
use crate::backend::Backend;
use ozgent_memory::Store;
use ozgent_render::{Style, Theme};
use crate::tui::{Submission, Ui};
use std::io::Write;

/// Everything one chat session needs.
///
/// No engine, no session, no tool host: the daemon holds all of it. What is
/// left is a screen, a conversation to write into, and the name of the model
/// to ask for.
pub struct Chat<'a> {
    /// The daemon. The terminal loads no model of its own: it asks whichever
    /// ozgent is already running, which is usually holding the model already.
    backend: Backend,
    /// Read directly for the conversation picker and `/memory`. One SQLite
    /// file, shared with the daemon that writes to it.
    store: Store,
    /// The conversation being written to, once there is one.
    ///
    /// `None` until the first message. Opening a chat and closing it again
    /// used to leave a titleless empty row behind every time.
    conversation: Option<i64>,
    theme: Theme,
    /// The screen.
    ui: &'a mut Ui,
    /// Generation rate of the last reply, for the status line.
    last_rate: Option<f64>,
    /// The model to ask for. A name, not a loaded thing — the daemon resolves
    /// it, and answers from the copy it already has if it has one.
    model: String,
    /// Context used and the window it came out of, as the daemon reported
    /// them. Its numbers rather than a second guess at them.
    used: Option<u32>,
    window: Option<u32>,
    /// Turns are numbered for the `/memory` display.
    turn: usize,
    /// Owned copies so `/config` and `/tools` can persist changes without
    /// borrowing from the caller across an await.
    paths: Paths,
    config: Config,
    /// The last reply's text, for `/copy`.
    last_reply: String,
    /// Reasoning mode for the next turn. Sent with each request rather than
    /// held on a session, because there is no session here to hold it.
    thinking: Option<ThinkingMode>,
}

/// Start a chat.
///
/// No model is loaded here. The terminal is a client of the daemon — see
/// [`crate::backend`] — so this connects, starting one if nothing answers,
/// and everything after that is a request.
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

    let plain = options.plain || !config.ui.markdown;
    let theme = if plain { Theme::plain() } else { Theme::default() };
    let mut ui = Ui::new(theme.clone(), Some(paths.root().join("history")));

    // Connect before taking over the screen, so "starting a backend" is an
    // ordinary line on the terminal rather than a message on a blank
    // alternate screen that vanishes when it is handed back.
    let backend = match crate::backend::Backend::connect().await {
        Some(backend) => backend,
        None => {
            println!("no backend at {} — starting one", crate::backend::Backend::address());
            crate::backend::Backend::connect_or_start(|_| {}).await?
        }
    };

    let store = Store::open(paths.root().join("ozgent.db"))?;
    let conversation: Option<i64> = None;

    let mut chat = Chat {
        backend,
        store,
        conversation,
        theme,
        ui: &mut ui,
        last_rate: None,
        model: name,
        used: None,
        window: None,
        turn: 0,
        paths: paths.clone(),
        config: config.clone(),
        last_reply: String::new(),
        thinking: None,
    };
    chat.banner();
    let outcome = chat.repl().await.map(|_| ());

    ui.save_history();
    // Before the error is printed, or it lands on the alternate screen and
    // disappears with it.
    ui.close();
    outcome
}

impl<'a> Chat<'a> {
    fn banner(&mut self) {
        // A clone, not a borrow of `self.theme`: writing to the screen takes
        // `&mut self`, and a closure holding the theme would block it.
        let theme = self.theme.clone();
        let dim = |s: &str| theme.style(Style::dim(), s);
        self.ui.blank();
        self.ui.say(format!("{} {}", self.model, dim("· asking the ozgent daemon")));
        // The window is the daemon's to report, and it does on the first
        // reply. Claiming a number before then would be a guess at settings
        // this process no longer resolves.
        self.ui.say(dim("/help for commands, /exit to quit"));
        self.ui.blank();
    }

    /// What the bottom row says: the facts that change as the session runs.
    fn status_segments(&self) -> Vec<crate::status::Segment> {
        use crate::status::Segment;
        use ozgent_core::format_count;

        // The model first: a status line that has dropped it no longer says
        // which machine's answer you are reading.
        let mut out = vec![Segment::new(100, self.model.clone())];
        if let (Some(used), Some(window)) = (self.used, self.window) {
            out.push(Segment::new(
                80,
                format!("{}/{}", format_count(used), format_count(window)),
            ));
        }
        if let Some(rate) = self.last_rate {
            out.push(Segment::new(60, format!("{rate:.0} tok/s")));
        }
        if self.config.tools.enabled {
            out.push(Segment::new(40, "tools"));
        }
        out
    }

    fn update_status(&mut self) {
        let segments = self.status_segments();
        self.ui.set_status(segments);
        let posture = self.permission_posture();
        self.ui.set_posture(posture);
    }

    /// One line about what tool calls will do, so the rules are visible
    /// without asking for them.
    fn permission_posture(&self) -> String {
        let p = &self.config.permissions;
        // Session grants live in the daemon now, so only the standing rules
        // are shown. Claiming to know what was allowed "this session" from
        // here would be a guess at somebody else's state.
        format!("tools · read {} · write {} · run {}", p.read, p.write, p.execute)
    }

    /// Echo the question into the transcript.
    ///
    /// In a scrolling REPL the terminal did this for free; a full-screen
    /// application draws its own rows, so without it a conversation is a
    /// column of answers to questions nobody can see.
    fn echo(&mut self, text: &str) {
        let marker = self.theme.style(Style::color(ozgent_render::Color::Cyan), "› ");
        // Agents called by name are picked out, so it is clear before any
        // frame appears that this message went to one.
        let catalog = ozgent_core::AgentCatalog::load(&self.paths);
        let mut body = String::new();
        let mut from = 0;
        for m in ozgent_core::agents::mentions(text) {
            if catalog.get(&m.name).is_none() {
                continue;
            }
            body.push_str(
                &self.theme.style(Style { bold: true, ..Default::default() }, &text[from..m.start]),
            );
            body.push_str(&self.theme.style(
                Style { bold: true, color: Some(ozgent_render::Color::Cyan), ..Default::default() },
                &text[m.start..m.end],
            ));
            from = m.end;
        }
        body.push_str(&self.theme.style(Style { bold: true, ..Default::default() }, &text[from..]));
        self.ui.blank();
        self.ui.say(format!("{marker}{body}"));
        self.ui.blank();
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
    /// One user turn: hand it to the daemon and draw what comes back.
    ///
    /// Everything that used to be here — assembling context, the tool loop,
    /// agents, persisting the reply — is the daemon's now, and was always
    /// also implemented there. What is left is the part that is genuinely
    /// about a terminal.
    async fn turn(&mut self, input: &str) -> Result<()> {
        self.turn += 1;
        let conversation = self.ensure_conversation()?;

        // Images named by path or URL are read here, because the daemon
        // cannot open files on this machine — it may not even be on it.
        let extracted = ozgent_llama::vision::extract(input);
        let images = self.read_images(&extracted.images);

        let request = serde_json::json!({
            "conversation": conversation,
            "model": self.model,
            "message": extracted.text,
            "thinking": self.thinking.map(|m| match m {
                ThinkingMode::On => "on",
                ThinkingMode::Off => "off",
                ThinkingMode::Auto => "auto",
            }),
            "tools": self.config.tools.enabled,
            "images": images,
        });

        let events = self.backend.chat(request).await?;
        let show_thinking = self.config.ui.show_thinking;
        let theme = self.theme.clone();
        let backend = self.backend.clone();
        let ui = &mut *self.ui;

        // Asking is the one part that genuinely needs the whole terminal — the
        // prompt, one-key answers, the redraw underneath — so it stays here
        // and the renderer calls back into it.
        let finished = crate::stream::render(
            &backend,
            events,
            ui,
            &theme,
            show_thinking,
            answer_permission,
        )
        .await?;

        self.last_reply = finished.answer.trim().to_string();
        if let Some(rate) = finished.rate {
            self.last_rate = Some(rate);
        }
        if finished.used.is_some() {
            self.used = finished.used;
        }
        if finished.window.is_some() {
            self.window = finished.window;
        }
        Ok(())
    }

    /// Attachments as data URLs, which is how the API takes them.
    ///
    /// Read here rather than passed as paths: the daemon may be on another
    /// machine, and a path that means something here means nothing there.
    fn read_images(&mut self, sources: &[ozgent_core::ImageSource]) -> Vec<String> {
        let mut out = Vec::new();
        for source in sources {
            let bytes = match source {
                ozgent_core::ImageSource::Path { path } => std::fs::read(path)
                    .map_err(|e| format!("{}: {e}", path.display())),
                ozgent_core::ImageSource::Bytes { bytes, .. } => Ok(bytes.clone()),
                // A URL is left for the daemon to fetch; it has the network
                // access and the caching for it.
                ozgent_core::ImageSource::Url { .. } => continue,
            };
            match bytes {
                Ok(bytes) => {
                    let mime = mime_of(&bytes);
                    out.push(format!("data:{mime};base64,{}", base64(&bytes)));
                }
                Err(e) => {
                    self.ui.say(self.theme.style(Style::dim(), &format!("note: {e}")));
                }
            }
        }
        out
    }


    fn opts_show_stats(&self) -> bool {
        std::env::var("OZGENT_STATS").is_ok()
    }

    /// Decide whether a call may run, asking the user if the policy says to.
    ///
    /// Returns `None` for a refusal, or `Some(by_user)` to run it — where
    /// `by_user` says a person authorised this call, which is what lets the
    /// Python side treat it as past its own standing boundaries.
    fn agents(&mut self, arg: &str) {
        let theme = self.theme.clone();
        let dim = |s: &str| theme.style(Style::dim(), s);
        let catalog = ozgent_core::AgentCatalog::load(&self.paths);
        self.ui.set_agents(catalog.clone());
        if arg.is_empty() {
            for a in catalog.all() {
                let name = theme.style(
                    Style { bold: true, color: Some(ozgent_render::Color::Cyan), ..Default::default() },
                    &format!("@{}", a.name),
                );
                let origin = match a.origin {
                    ozgent_core::agents::Origin::Builtin => "built in",
                    ozgent_core::agents::Origin::User => "yours",
                    ozgent_core::agents::Origin::Override => "edited",
                };
                self.ui.say(format!("{name}  {}  {}", a.definition.description, dim(origin)));
                self.ui.say(dim(&format!("    tools: {}", a.definition.tools.join(", "))));
            }
            for e in &catalog.errors {
                self.ui.say(theme.style(Style::color(ozgent_render::Color::Red), e));
            }
            self.ui.say(dim(
                "write @name in a message to call one · /agents <name> for its instructions · \
                 `ozgent agent new <name>` to make your own",
            ));
            return;
        }
        let name = arg.trim_start_matches('@');
        match catalog.get(name) {
            Some(a) => {
                let d = &a.definition;
                self.ui.say(format!("@{}  {}", a.name, d.description));
                let rules: Vec<String> = d.permissions.iter().map(|(t, r)| format!("{t}={r}")).collect();
                self.ui.say(dim(&format!(
                    "tools: {}  ·  rules: {}  ·  rounds: {}  ·  thinking: {}",
                    d.tools.join(", "),
                    if rules.is_empty() { "global".into() } else { rules.join(", ") },
                    d.rounds(),
                    d.thinking.map(|t| format!("{t:?}").to_lowercase()).unwrap_or_else(|| "as the chat".into()),
                )));
                self.ui.markdown(d.instructions.clone());
            }
            None => self.ui.say(dim(&format!("no agent named {name}; /agents lists them"))),
        }
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
                "assistant" => self.model.as_str(),
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
                if chosen.model.to_string() == self.model {
                    self.ui.say(dim("already using that model"));
                    return Ok(None);
                }
                return Ok(Some(chosen.short_name()));
            }
            let found = ozgent_core::resolve(&self.paths, arg)?;
            if found.model.to_string() == self.model {
                self.ui.say(dim(&format!("already using {}", found.model)));
                return Ok(None);
            }
            return Ok(Some(found.short_name()));
        }

        self.ui.say(dim("installed models:"));
        for (i, m) in models.iter().enumerate() {
            let marker = if m.model.to_string() == self.model { "*" } else { " " };
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

        // Resolved here from config.toml rather than read off a session:
        // there is no session, and the file is what the daemon reads too.
        let opts = self.config.options_for(&self.model).resolve();

        if key.is_empty() {
            self.ui.say(dim(&format!("settings for {}:", self.model)));
            self.ui.say(dim(&format!("  thinking        {:?}", opts.thinking)));
            self.ui.say(dim(&format!("  effort          {:?}", opts.reasoning_effort)));
            self.ui.say(dim(&format!("  temperature     {}", opts.temperature)));
            self.ui.say(dim(&format!("  top_p           {}", opts.top_p)));
            self.ui.say(dim(&format!("  top_k           {}", opts.top_k)));
            self.ui.say(dim(&format!("  min_p           {}", opts.min_p)));
            self.ui.say(dim(&format!("  repeat_penalty  {}", opts.repeat_penalty)));
            self.ui.say(dim(&format!("  max_tokens      {}", opts.max_tokens)));
            let seed = opts.seed.map_or("random".to_string(), |s| s.to_string());
            self.ui.say(dim(&format!("  seed            {seed}")));
            // Both numbers, because they can differ: what was asked for is
            // what `/config ctx` set, and what the session runs on is what
            // the machine's memory allowed.
            let asked = ozgent_core::format_count(opts.context_length);
            let ctx = match self.window {
                Some(window) if window != opts.context_length => {
                    format!("{asked} (the daemon is running at {})", ozgent_core::format_count(window))
                }
                _ => asked,
            };
            self.ui.say(dim(&format!("  ctx             {ctx}")));
            self.ui.say(dim(&format!("  gpu layers      {}", opts.gpu_layers)));
            let kv = if opts.kv_offload { "gpu" } else { "system ram" };
            self.ui.say(dim(&format!("  kv cache        {kv}")));
            self.ui.say(dim(&format!(
                "  mode            {} ({})",
                opts.inference_mode,
                opts.inference_mode.describes(),
            )));
            self.ui.say(dim(&format!("  tools           {}", opts.tools)));
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
                layer.thinking = Some(mode);
            }
            "effort" | "reasoning_effort" => {
                let level: ozgent_core::ReasoningEffort =
                    value.parse().map_err(|e: String| anyhow::anyhow!(e))?;
                layer.reasoning_effort = Some(level);
            }
            "temperature" | "temp" => {
                let t: f32 = value.parse().context("temperature must be a number")?;
                layer.temperature = Some(t);
            }
            "top_p" | "top-p" => {
                let p: f32 = value.parse().context("top_p must be a number")?;
                layer.top_p = Some(p);
            }
            "top_k" | "top-k" => {
                let k: u32 = value.parse().context("top_k must be a whole number")?;
                layer.top_k = Some(k);
            }
            "min_p" | "min-p" => {
                let p: f32 = value.parse().context("min_p must be a number")?;
                layer.min_p = Some(p);
            }
            "repeat_penalty" | "repeat-penalty" => {
                let p: f32 = value.parse().context("repeat_penalty must be a number")?;
                layer.repeat_penalty = Some(p);
            }
            "max_tokens" | "max-tokens" => {
                let n = ozgent_core::parse_count(&value).map_err(|e| anyhow::anyhow!(e))?;
                layer.max_tokens = Some(n);
            }
            "seed" => {
                let n: u32 = value.parse().context("seed must be a whole number")?;
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
                        self.window
                            .map(ozgent_core::format_count)
                            .unwrap_or_else(|| "unknown".into())
                    )));
                return Ok(());
            }
            "tools" => {
                let on = matches!(value.as_str(), "on" | "true" | "yes" | "1");
                layer.tools = Some(on);
            }
            "mode" | "inference_mode" => {
                let mode: ozgent_core::InferenceMode =
                    value.parse().map_err(|e: String| anyhow::anyhow!(e))?;
                layer.inference_mode = Some(mode);
                // Load-time: where the weights and the cache live is fixed
                // when the context opens.
                let entry = self.config.models.entry(self.model.to_string()).or_default();
                *entry = entry.clone().merge(&layer);
                self.config.save(&self.paths)?;
                self.ui.say(dim(&format!(
                    "· mode = {mode} — {} (saved; applies when the model is next loaded)",
                    mode.describes(),
                )));
                return Ok(());
            }
            "kv_offload" | "kv" => {
                // Load-time, like ctx: where the cache lives is fixed when the
                // context opens and cannot move under a live one.
                let on = matches!(value.as_str(), "on" | "true" | "yes" | "1" | "gpu");
                layer.kv_offload = Some(on);
                let entry = self.config.models.entry(self.model.to_string()).or_default();
                *entry = entry.clone().merge(&layer);
                self.config.save(&self.paths)?;
                let note = if on {
                    "kv cache on the gpu (saved; applies when the model is next loaded)"
                } else {
                    "kv cache in system ram — a larger window, more slowly \
                     (saved; applies when the model is next loaded)"
                };
                self.ui.say(dim(&format!("· {note}")));
                return Ok(());
            }
            other => {
                self.ui.say(dim(&format!("unknown setting {other:?}; try {SETTABLE}")));
                return Ok(());
            }
        }

        // Saved to config.toml, which is what the daemon reads. It re-reads
        // the file as it changes, so a sampling change here reaches the next
        // turn without anything being restarted — see `state::watch_config`.
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

            match None::<&ozgent_tools::Toolbox> {
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
                        let session = if false {
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

        // A rule can be set for a tool that is not loaded right now, and the
        // list lives in the daemon anyway, so nothing is checked here.
        if false {
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
                self.turn = 0;
                // Wipe the screen too. The old thread staying on screen under
                // a one-line notice reads as "still in that conversation",
                // which is the opposite of what just happened.
                self.ui.clear();
                self.banner();
                self.ui.say(dim("· new conversation"));
            }

            "/conv" | "/convs" | "/conversations" => return self.conversations(arg),

            // Sent with each turn rather than held on a session: there is no
            // session here to hold it, and the daemon takes it per request.
            "/think" => match arg {
                "" => {
                    let now = self.thinking.map(|m| format!("{m:?}")).unwrap_or("auto".into());
                    self.ui.say(dim(&format!("thinking: {now}")));
                }
                other => match other.parse::<ThinkingMode>() {
                    Ok(mode) => {
                        self.thinking = Some(mode);
                        self.ui.say(dim(&format!("· thinking {other}")));
                    }
                    Err(e) => self.ui.say(dim(&e)),
                },
            },

            // These are the model's settings, and the model is the daemon's.
            // Writing them here would change a copy nothing reads.
            "/effort" | "/system" => {
                self.ui.say(dim(
                    "that is a model setting now that the terminal is a client. \
                     Set it in config.toml, or on the web interface's settings page.",
                ));
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

            "/call" => self.ui.say(dim(
                "ask for it in a sentence instead — the model chooses the tool, \
                 and the daemon runs it.",
            )),

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
                let name = self.model.clone();
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

            "/agents" | "/agent" => self.agents(arg),

            // The whole reply, as the model wrote it — markdown and all, and
            // including the parts scrolled off screen that a drag cannot reach.
            "/copy" => {
                if self.last_reply.is_empty() {
                    self.ui.say(dim("nothing to copy yet"));
                } else {
                    let text = self.last_reply.clone();
                    self.ui.copy(&text);
                    self.ui.render();
                }
            }

            "/tools" if !arg.is_empty() => self.configure_tool(arg)?,

            // Asked of the daemon, which is the only thing that knows what
            // actually loaded — including tools from MCP servers this process
            // has never spoken to.
            "/tools" => match self.backend.get("/api/tools").await {
                Ok(listed) => {
                    let tools = listed["available"].as_array().cloned().unwrap_or_default();
                    if listed["enabled"].as_bool() == Some(false) {
                        self.ui.say(dim("tools are switched off"));
                    }
                    self.ui.say(dim(&format!("{} tools", tools.len())));
                    for t in tools {
                        let name = t["name"].as_str().unwrap_or("?");
                        let about = t["description"].as_str().unwrap_or("");
                        self.ui.say(dim(&format!(
                            "  {name}  {}",
                            ozgent_tools::first_line(about)
                        )));
                    }
                }
                Err(e) => self.ui.say(dim(&format!("could not ask the daemon: {e}"))),
            },

            "/stats" => {
                let window = self
                    .window
                    .map(ozgent_core::format_count)
                    .unwrap_or_else(|| "?".into());
                let used = self
                    .used
                    .map(ozgent_core::format_count)
                    .unwrap_or_else(|| "?".into());
                let rate = self
                    .last_rate
                    .map(|r| format!("{r:.0} tok/s"))
                    .unwrap_or_else(|| "no reply yet".into());
                self.ui.say(dim(&format!(
                    "{} · {used}/{window} ctx · {rate} · via {}",
                    self.model,
                    self.backend.base()
                )));
            }

            other => self.ui.say(dim(&format!("unknown command {other}; try /help"))),
        }
        Ok(Flow::Continue)
    }
}

/// Tokens a call is given to finish its arguments before it is asked about.
///
/// The whole point of asking early is to ask before a file's `content` has
/// been generated, so this cannot be large. It only has to be long enough
/// that a compact call — a command, a search — arrives complete and is asked
/// about in full, which takes a couple of dozen tokens.
const ARGUMENT_GRACE: usize = 24;

/// What one generation produced, after the reasoning and tool-call streams
/// have been told apart.
struct Reply {
    text: String,
    thinking: Option<String>,
    /// A permission answered while the call was still being written, so the
    /// tool loop does not ask a second time about the same call.
    early_permission: Option<(String, ozgent_core::Choice)>,
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
pub(crate) fn pretty_args(args: &serde_json::Value) -> String {
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
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            // Flattened before truncating, not after: a whole file's contents
            // clipped to 48 characters can still contain three newlines, and
            // each one breaks the row it is supposed to be sharing.
            let shown = truncate_middle(&shown.replace('\n', "⏎"), 48);
            format!("{k}: {shown}")
        })
        .collect();
    format!("  {}", parts.join("  "))
}

/// One line describing what a tool returned.
///
/// A search response is thousands of characters; the user wants to know it
/// worked and roughly what came back, not to read it.
pub(crate) fn summarise_result(value: &serde_json::Value) -> String {
    if let Some(known) = ozgent_tools::summary::describe(value) {
        return known;
    }
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
     ctx, mode, kv, tools";

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
                   /config mode gpu|gpu_ram|ram — where the model runs.
                   gpu_ram keeps the weights on the card and the KV cache in
                   RAM: the full window, several times slower per token
/tools             list available tools
/agents [name]     list agents, or show one · write @name in a message to call it
/copy              copy the last reply to the clipboard
/default           use this model when none is named · /default clear to unset
/permissions       what tools may do without asking
/permissions <tool> allow|ask|deny|clear   ·  or read|write|execute <rule>
/tools <t> <prov>  point a tool at a provider, e.g. /tools web_search brave
/call <request>    force a tool call, constrained by grammar
/stats             model and context state

Scrolling    wheel, PageUp/PageDown, or Shift-Up/Shift-Down
             Esc returns to the newest message
Copying      drag over text to select it; it is copied when you let go
             /copy copies the whole last reply, markdown included
Editing      Shift-Enter for a new line (Alt-Enter or Ctrl-J in terminals
             that cannot report Shift) · Ctrl-A/E/K/U/W as in any shell
             Up/Down walk what you typed before

Paste an image path or URL in a message and it is picked up automatically.
Type @ for the agents; Tab or Enter picks one.";

/// Every agent, for the model to hand a turn to. Read per turn, so one saved a
/// moment ago is included.
fn catalog_agents(paths: &Paths) -> Vec<ozgent_core::Agent> {
    ozgent_core::AgentCatalog::load(paths).all().to_vec()
}


/// `█████░░░░░  42%`: a bar a terminal can draw in any font.
pub fn progress_bar(fraction: f32, width: usize) -> String {
    let f = fraction.clamp(0.0, 1.0);
    let full = (f * width as f32).round() as usize;
    format!("{}{} {:>3}%", "█".repeat(full), "░".repeat(width - full), (f * 100.0).floor() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_progress_bar_fills_and_says_how_far() {
        assert_eq!(progress_bar(0.0, 4), "░░░░   0%");
        assert_eq!(progress_bar(0.5, 4), "██░░  50%");
        assert_eq!(progress_bar(1.0, 4), "████ 100%");
        assert_eq!(progress_bar(2.0, 4), "████ 100%");
    }

    #[test]
    fn a_multiline_argument_stays_on_one_row() {
        // A file's contents clipped to 48 characters can still contain three
        // newlines, and each one breaks the row it is meant to share.
        let out = pretty_args(&serde_json::json!({
            "path": "a.py",
            "content": "line one\nline two\nline three\n",
        }));
        assert!(!out.contains('\n'), "{out:?}");
    }

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

/// A picture's type, from its first bytes.
///
/// Sniffed rather than taken from the file name: people paste screenshots with
/// no extension, and a wrong content type is refused by the API with a message
/// about the data URL rather than about the picture.
fn mime_of(bytes: &[u8]) -> &'static str {
    match bytes {
        [0x89, b'P', b'N', b'G', ..] => "image/png",
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg",
        [b'G', b'I', b'F', ..] => "image/gif",
        [b'R', b'I', b'F', b'F', _, _, _, _, b'W', b'E', b'B', b'P', ..] => "image/webp",
        _ => "application/octet-stream",
    }
}

/// Base64, without a dependency for sixty lines of table lookup.
fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        let take = chunk.len() + 1;
        for i in 0..4 {
            if i < take {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod client_tests {
    use super::*;

    #[test]
    fn base64_matches_the_known_examples() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_handles_bytes_that_are_not_text() {
        assert_eq!(base64(&[0x00, 0xFF, 0x80]), "AP+A");
        assert_eq!(base64(&[0xFF; 3]), "////");
    }

    #[test]
    fn a_picture_is_recognised_by_its_first_bytes_not_its_name() {
        assert_eq!(mime_of(&[0x89, b'P', b'N', b'G', 13, 10, 26, 10]), "image/png");
        assert_eq!(mime_of(&[0xFF, 0xD8, 0xFF, 0xE0]), "image/jpeg");
        assert_eq!(mime_of(b"GIF89a"), "image/gif");
        assert_eq!(mime_of(b"RIFF____WEBPVP8 "), "image/webp");
        assert_eq!(mime_of(b"not a picture"), "application/octet-stream");
        assert_eq!(mime_of(&[]), "application/octet-stream");
    }
}

/// Answer a permission question the daemon asked, using the terminal's own
/// prompt.
///
/// A free function taking the screen rather than a closure capturing it: the
/// renderer already holds `&mut Ui`, and a closure that captured it too would
/// be a second mutable borrow of the same thing.
fn answer_permission(
    ui: &mut Ui,
    tool: &str,
    arguments: &serde_json::Value,
    effect: &str,
) -> Option<String> {
    let effect: ozgent_core::Effect = effect.parse().unwrap_or_default();
    Some(
        match ui.ask_permission(tool, effect, arguments) {
            ozgent_core::Choice::Once => "once",
            ozgent_core::Choice::Session => "session",
            ozgent_core::Choice::Always => "always",
            ozgent_core::Choice::Deny => "deny",
            // The terminal never offers this one, but the type carries it.
            ozgent_core::Choice::DenyAlways => "deny_always",
        }
        .to_string(),
    )
}
