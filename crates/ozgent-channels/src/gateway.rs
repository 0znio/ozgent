//! The gateway: messages in, turns run, replies out.
//!
//! Everything transport-specific lives in the channel modules; this decides
//! *whether* a message is answered and *how* the answer is shown. Three rules
//! shape it, and each exists because getting it wrong is a real failure rather
//! than an untidy one:
//!
//! * **Admission is checked before anything else happens.** Not after the
//!   conversation is looked up, not after the model is chosen — before. The
//!   only thing an unadmitted sender can do is offer a pairing code.
//! * **A chat is answered one turn at a time.** Two turns in one chat would
//!   interleave into the same conversation and produce a history neither
//!   question can be read against.
//! * **A question blocks its turn, so it is answered off the turn's path.** The
//!   inference thread is waiting inside `permit`, so the answer arrives on the
//!   receiving side and is delivered straight to the pending map. Routing it
//!   through the chat's queue would put it behind the very turn it unblocks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ozgent_core::channels::{self, Kind};
use ozgent_core::permission::{Choice, Effect};
use ozgent_core::{Config, Paths};
use ozgent_web::state::State;
use ozgent_web::turn::{self, Turn};
use ozgent_web::worker::Event;
use tokio::sync::mpsc::{Sender, UnboundedSender, channel, unbounded_channel};

use crate::chat::{Command, Inbound, Msg, Question, Tokens, read_choice};
use crate::command::{Directive, help, parse};
use crate::compose::Composer;
use crate::live::Live;
use crate::split::{TELEGRAM_LIMIT, WHATSAPP_LIMIT};

/// A chat, on a channel. The store keys bound conversations the same way.
type Key = (Kind, String);

/// A question posted to a chat and not yet answered.
struct Ask {
    /// The tool call id the inference thread is blocked on.
    call_id: String,
    chat: String,
    kind: Kind,
    tool: String,
}

struct Shared {
    app: State,
    paths: Paths,
    tokens: Tokens,
    senders: HashMap<Kind, UnboundedSender<Command>>,
    /// Questions in flight, by the token their buttons carry.
    asks: Mutex<HashMap<u64, Ask>>,
    /// The question a chat is waiting on, so a *typed* answer is read as an
    /// answer rather than as a new question. Needed on every channel: a person
    /// can always type instead of tapping.
    waiting: Mutex<HashMap<Key, u64>>,
    /// The turn in flight per chat, so `/stop` can end it.
    running: Mutex<HashMap<Key, tokio::task::AbortHandle>>,
    /// The current pairing code. Single use: replaced the moment it works.
    pairing: Mutex<String>,
}

/// Run every configured channel until the process is stopped.
pub async fn run(app: State, paths: Paths) -> anyhow::Result<()> {
    let config = snapshot(&app);
    let active = config.channels.active();
    if active.is_empty() {
        anyhow::bail!(
            "no channel is switched on. Set `[channels] enabled = true` and turn one on — \
             see `ozgent channel --help`."
        );
    }

    let (inbound_tx, mut inbound_rx) = channel::<(Kind, Inbound)>(64);
    let mut senders = HashMap::new();

    for kind in &active {
        let (command_tx, command_rx) = unbounded_channel::<Command>();
        senders.insert(*kind, command_tx);

        // Each channel speaks plain `Inbound`; the kind is added here so a
        // channel never has to know what else is running.
        let (tagged_tx, mut tagged_rx) = channel::<Inbound>(64);
        let to_gateway = inbound_tx.clone();
        let kind = *kind;
        tokio::spawn(async move {
            while let Some(event) = tagged_rx.recv().await {
                if to_gateway.send((kind, event)).await.is_err() {
                    break;
                }
            }
        });

        start(kind, &config, &paths, tagged_tx, command_rx, inbound_tx.clone());
    }
    drop(inbound_tx);

    let shared = Arc::new(Shared {
        app,
        paths,
        tokens: Tokens::default(),
        senders,
        asks: Mutex::new(HashMap::new()),
        waiting: Mutex::new(HashMap::new()),
        running: Mutex::new(HashMap::new()),
        pairing: Mutex::new(pairing_code()),
    });

    announce(&shared, &config, &active);

    let mut chats: HashMap<Key, UnboundedSender<Msg>> = HashMap::new();
    // A channel that has reported a terminal failure. When the last one goes,
    // so does the gateway: a process that stays up answering nothing looks
    // like it is working, and is the worst of the possible outcomes.
    let mut dead: Vec<Kind> = Vec::new();

    while let Some((kind, event)) = inbound_rx.recv().await {
        if matches!(event, Inbound::Failed { .. }) && !dead.contains(&kind) {
            dead.push(kind);
        }
        route(&shared, &mut chats, kind, event).await;
        if dead.len() == active.len() {
            anyhow::bail!("every channel stopped; nothing is being answered");
        }
    }
    Ok(())
}

/// Start one channel's own task.
fn start(
    kind: Kind,
    config: &Config,
    paths: &Paths,
    tx: Sender<Inbound>,
    rx: tokio::sync::mpsc::UnboundedReceiver<Command>,
    failures: Sender<(Kind, Inbound)>,
) {
    // A channel that cannot start must say so on the same path everything else
    // is reported on, or `ozgent gateway` prints a banner and then sits there
    // silently doing nothing.
    let report = move |reason: String| {
        tokio::spawn(async move {
            let _ = failures.send((kind, Inbound::Failed { reason })).await;
        });
    };

    match kind {
        Kind::Telegram => {
            // The environment wins, for anyone who would rather not have a
            // password in a file that gets copied around.
            let token = std::env::var("OZGENT_TELEGRAM_TOKEN")
                .unwrap_or_else(|_| config.channels.telegram.token.clone());
            tokio::spawn(async move {
                if let Err(e) = crate::telegram::run(token, tx, rx).await {
                    report(e.to_string());
                }
            });
        }
        Kind::WhatsApp => {
            let wa = config.channels.whatsapp.clone();
            let paths = paths.clone();
            tokio::spawn(async move {
                let bridge = match crate::whatsapp::locate(wa.bridge.as_deref(), &paths) {
                    Ok(b) => b,
                    Err(e) => return report(e.to_string()),
                };
                let state = paths.channel_dir("whatsapp").join("auth");
                if let Err(e) = crate::whatsapp::run(wa.node, bridge, state, tx, rx).await {
                    report(e.to_string());
                }
            });
        }
    }
}

/// What the operator sees when the gateway starts.
///
/// Long, because every line of it is something that will otherwise be found
/// out the hard way: which channels are live, who can reach them, and that a
/// chat can reach this machine's tools.
fn announce(shared: &Arc<Shared>, config: &Config, active: &[Kind]) {
    println!("ozgent gateway");
    for kind in active {
        let access = config.channels.access(*kind);
        let who = if channels::is_open_to_everyone(access.allow) {
            "ANYONE".to_string()
        } else if access.allow.is_empty() {
            "nobody yet".to_string()
        } else {
            format!("{} allowed", access.allow.len())
        };
        println!("  {kind:<9}  {who}");
    }
    println!();

    let open = active
        .iter()
        .any(|k| channels::is_open_to_everyone(config.channels.access(*k).allow));
    let empty = active.iter().all(|k| config.channels.access(*k).allow.is_empty());

    if open {
        println!("  WARNING: a channel admits everyone. Anyone who finds it can use this");
        println!("  machine's tools, subject only to your permission rules. Replace `*`");
        println!("  in [channels] with the ids that should be allowed.");
        println!();
    }
    if empty {
        println!("  Nobody is allowed yet, so nothing will be answered.");
    }
    println!("  To allow someone, have them send:   /pair {}", shared.pairing.lock().unwrap());
    println!("  The code works once, and changes after it is used.");
    println!();
    println!("press ctrl-c to stop");
}

/// A short code, from the clock and an allocation address.
///
/// Not a secret worth attacking — it is read off a terminal and typed within
/// the minute — but it must not be guessable from the outside, which rules out
/// anything derived from the time alone.
fn pairing_code() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let entropy = {
        let boxed = Box::new(0u8);
        let addr = Box::into_raw(boxed) as usize;
        // SAFETY: reclaimed immediately; only its address was wanted.
        unsafe { drop(Box::from_raw(addr as *mut u8)) };
        addr
    };
    let mut n = (nanos as usize) ^ entropy.rotate_left(17);
    // No vowels and no look-alikes, so a code read aloud or off a screen is
    // typed back correctly.
    const ALPHABET: &[u8] = b"3479CDFHJKMNPRTWXY";
    let mut out = String::new();
    for _ in 0..6 {
        out.push(ALPHABET[n % ALPHABET.len()] as char);
        n /= ALPHABET.len();
    }
    out
}

fn snapshot(app: &State) -> Config {
    app.config.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn limit(kind: Kind) -> usize {
    match kind {
        Kind::Telegram => TELEGRAM_LIMIT,
        Kind::WhatsApp => WHATSAPP_LIMIT,
    }
}

// ------------------------------------------------------------------ routing

async fn route(
    shared: &Arc<Shared>,
    chats: &mut HashMap<Key, UnboundedSender<Msg>>,
    kind: Kind,
    event: Inbound,
) {
    match event {
        Inbound::Ready { who } => println!("  {kind}: connected as {who}"),
        Inbound::Notice { text } => println!("  {kind}: {text}"),
        Inbound::Failed { reason } => {
            // Printed as well as logged: the operator is looking at a terminal
            // that has just stopped doing anything.
            eprintln!("  {kind}: {reason}");
            tracing::error!("{kind}: {reason}");
        }
        Inbound::Answer { token, choice, .. } => answer(shared, token, choice).await,
        Inbound::Message(msg) => {
            let key = (kind, msg.chat.clone());

            if !admitted(shared, kind, &msg) {
                offer_pairing(shared, kind, &msg).await;
                return;
            }

            // A typed answer to an outstanding question. Checked before the
            // directive parse and before the queue, because the turn it
            // answers is blocking that queue.
            let outstanding = shared.waiting.lock().unwrap().get(&key).copied();
            if let Some(token) = outstanding {
                if let Some(choice) = read_choice(&msg.text) {
                    answer(shared, token, choice).await;
                    return;
                }
            }

            // `/stop` is the other thing that cannot wait its turn: the point
            // of it is to end what is currently running.
            if matches!(parse(&msg.text), Directive::Stop) {
                let aborted = shared.running.lock().unwrap().remove(&key);
                match aborted {
                    Some(handle) => {
                        handle.abort();
                        // A turn abandoned mid-question leaves the inference
                        // thread waiting; refusing is the safe direction.
                        clear_question(shared, &key).await;
                        say(shared, kind, &msg.chat, "Stopped.").await;
                    }
                    None => say(shared, kind, &msg.chat, "Nothing to stop.").await,
                }
                return;
            }

            let queue = chats.entry(key.clone()).or_insert_with(|| {
                let (tx, rx) = unbounded_channel::<Msg>();
                tokio::spawn(chat_task(shared.clone(), kind, msg.chat.clone(), rx));
                tx
            });
            if queue.send(*msg).is_err() {
                // The task is gone; a fresh one takes over on the next message.
                chats.remove(&key);
            }
        }
    }
}

/// Whether this sender may talk to this channel.
fn admitted(shared: &Arc<Shared>, kind: Kind, msg: &Msg) -> bool {
    let config = snapshot(&shared.app);
    if msg.group && kind == Kind::WhatsApp && !config.channels.whatsapp.groups {
        return false;
    }
    channels::admits(config.channels.access(kind).allow, &msg.identities())
}

/// Answer an unadmitted sender: only ever about pairing.
async fn offer_pairing(shared: &Arc<Shared>, kind: Kind, msg: &Msg) {
    let Directive::Pair(code) = parse(&msg.text) else {
        // Deliberately quiet. Anyone can message a bot handle, and a reply to
        // every stranger both confirms the bot is live and makes it a way to
        // send mail from someone else's machine.
        tracing::info!(
            "{kind}: ignoring a message from {} ({}), who is not allowed",
            msg.name,
            msg.sender_id
        );
        return;
    };

    let expected = shared.pairing.lock().unwrap().clone();
    if code.trim().to_ascii_uppercase() != expected {
        say(shared, kind, &msg.chat, "That code is not right.").await;
        return;
    }

    // The identity written to the allowlist is the provider's id, never the
    // display name: a name can be changed to anything by the person holding
    // the account, and matching on one would make the allowlist meaningless.
    let identity = msg.sender_id.clone();
    let saved = {
        let mut config = shared.app.config.lock().unwrap_or_else(|e| e.into_inner());
        let added = config.channels.admit(kind, &identity);
        if added {
            config.save(&shared.paths).map_err(|e| e.to_string())
        } else {
            Ok(())
        }
    };
    // Replaced whether or not the write worked: it has been used.
    *shared.pairing.lock().unwrap() = pairing_code();

    match saved {
        Ok(()) => {
            println!("  {kind}: allowed {} ({identity})", msg.name);
            println!("  next pairing code: /pair {}", shared.pairing.lock().unwrap());
            say(
                shared,
                kind,
                &msg.chat,
                "You're in. Send me a message, or `/help` to see what I can do.",
            )
            .await;
        }
        Err(e) => {
            tracing::error!("saving the allowlist: {e}");
            say(shared, kind, &msg.chat, "I could not save that. Ask whoever runs me.").await;
        }
    }
}

// ---------------------------------------------------------------- questions

/// Deliver an answer to the call waiting on it.
///
/// The same path whether the answer was tapped on a button or typed as a word,
/// so the two can never come to mean different things.
async fn answer(shared: &Arc<Shared>, token: u64, choice: Choice) {
    let Some(ask) = shared.asks.lock().unwrap().remove(&token) else {
        // Answered twice, or after the turn gave up waiting. Not an error.
        return;
    };
    let key = (ask.kind, ask.chat.clone());
    shared.waiting.lock().unwrap().remove(&key);

    // This is what unblocks the inference thread inside `permit`.
    shared.app.permissions.pending.answer(&ask.call_id, choice);

    let outcome = match choice {
        Choice::Once => format!("**{}** — allowed, this once.", ask.tool),
        Choice::Session => format!("**{}** — allowed for the rest of this session.", ask.tool),
        Choice::Always => format!("**{}** — always allowed from now on.", ask.tool),
        Choice::Deny => format!("**{}** — not allowed.", ask.tool),
        Choice::DenyAlways => format!("**{}** — never allowed from now on.", ask.tool),
    };
    settle(shared, ask.kind, &ask.chat, token, &outcome).await;
}

/// Take a question's buttons away and say how it ended.
async fn settle(shared: &Arc<Shared>, kind: Kind, chat: &str, token: u64, markdown: &str) {
    if let Some(tx) = shared.senders.get(&kind) {
        let _ = tx.send(Command::Settle {
            chat: chat.to_string(),
            token,
            markdown: markdown.to_string(),
        });
    }
}

/// Refuse whatever this chat was being asked, and clear it.
async fn clear_question(shared: &Arc<Shared>, key: &Key) {
    let token = shared.waiting.lock().unwrap().remove(key);
    let Some(token) = token else { return };
    let Some(ask) = shared.asks.lock().unwrap().remove(&token) else { return };
    shared.app.permissions.pending.answer(&ask.call_id, Choice::Deny);
    settle(shared, ask.kind, &ask.chat, token, &format!("**{}** — not allowed.", ask.tool)).await;
}

// -------------------------------------------------------------------- turns

/// One chat's queue. Messages are answered in order, one at a time.
async fn chat_task(
    shared: Arc<Shared>,
    kind: Kind,
    chat: String,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Msg>,
) {
    let key = (kind, chat.clone());
    while let Some(msg) = rx.recv().await {
        match parse(&msg.text) {
            Directive::Ask(text) if !text.is_empty() || !msg.images.is_empty() => {
                let mut msg = msg;
                msg.text = text;
                // Spawned rather than awaited inline so `/stop` has something
                // to abort; the queue still waits for it, so the chat stays
                // one turn at a time.
                let task = tokio::spawn(turn_for(shared.clone(), kind, chat.clone(), msg));
                shared.running.lock().unwrap().insert(key.clone(), task.abort_handle());
                let _ = task.await;
                shared.running.lock().unwrap().remove(&key);
                // A turn that ended with a question outstanding — because it
                // was aborted, or the model gave up — must not leave the chat
                // believing it still owes an answer.
                clear_question(&shared, &key).await;
            }
            Directive::Ask(_) => {}
            Directive::Help => {
                let model = model_for(&snapshot(&shared.app)).unwrap_or_else(|| "none".into());
                say(&shared, kind, &chat, &help(&model)).await;
            }
            Directive::New => {
                // Scoped tightly: the guard is not `Send`, and holding it
                // across the reply would make this whole task unspawnable.
                let existed = {
                    let store = shared.app.store.lock().unwrap();
                    store.unbind_channel_chat(kind.as_str(), &chat).unwrap_or(false)
                };
                let text = if existed {
                    "Starting fresh. The old thread is still in the web interface."
                } else {
                    "Nothing to forget — this is already a new thread."
                };
                say(&shared, kind, &chat, text).await;
            }
            Directive::Model(None) => {
                let text = match model_for(&snapshot(&shared.app)) {
                    Some(m) => format!("Answering with `{m}`."),
                    None => "No model is set. Whoever runs me needs to pick one.".into(),
                };
                say(&shared, kind, &chat, &text).await;
            }
            Directive::Model(Some(name)) => set_model(&shared, kind, &chat, &name).await,
            Directive::Tools => say(&shared, kind, &chat, &tool_list(&shared, kind)).await,
            Directive::Whoami => {
                let handle = msg.handle.clone().unwrap_or_else(|| "none".into());
                let text = format!(
                    "On {kind} you are:\n\n- id: `{}`\n- handle: `{}`\n- this chat: `{}`",
                    msg.sender_id, handle, chat
                );
                say(&shared, kind, &chat, &text).await;
            }
            // Handled before the queue, since the point is not to wait.
            Directive::Stop => {}
            Directive::Pair(_) => {
                say(&shared, kind, &chat, "You are already allowed to talk to me.").await;
            }
            Directive::Unknown(word) => {
                say(&shared, kind, &chat, &format!("I don't know `/{word}`. Try `/help`.")).await;
            }
        }
    }
}

/// The model channels answer with.
fn model_for(config: &Config) -> Option<String> {
    config.channels.model.clone().or_else(|| config.default_model.clone())
}

async fn set_model(shared: &Arc<Shared>, kind: Kind, chat: &str, name: &str) {
    // Resolved before it is saved, so a typo is a message rather than a
    // channel that stops answering.
    let found = match ozgent_core::resolve(&shared.paths, name) {
        Ok(f) => f.model.to_string(),
        Err(e) => {
            say(shared, kind, chat, &format!("I don't have `{name}`: {e}")).await;
            return;
        }
    };
    let saved = {
        let mut config = shared.app.config.lock().unwrap_or_else(|e| e.into_inner());
        config.channels.model = Some(found.clone());
        config.save(&shared.paths).map_err(|e| e.to_string())
    };
    match saved {
        Ok(()) => say(shared, kind, chat, &format!("Answering with `{found}` from now on.")).await,
        Err(e) => say(shared, kind, chat, &format!("I could not save that: {e}")).await,
    }
}

fn tool_list(shared: &Arc<Shared>, kind: Kind) -> String {
    let config = snapshot(&shared.app);
    if !config.tools.enabled {
        return "Tools are switched off, so I can only answer from what I know.".into();
    }
    let Some(tools) = ozgent_web::worker::current_tools(&shared.app.tools) else {
        return "Tools are configured but not running.".into();
    };
    let allowed = config.channels.access(kind).tools;
    let mut lines = Vec::new();
    for spec in tools.host.tools() {
        if let Some(list) = allowed {
            if !list.iter().any(|t| t == &spec.name) {
                continue;
            }
        }
        if config.tools.disabled.contains(&spec.name) {
            continue;
        }
        let rule = config.permissions.rule_for(&spec.name, spec.effect);
        lines.push(format!("- `{}` — {rule}", spec.name));
    }
    if lines.is_empty() {
        return "I have no tools here.".into();
    }
    format!("Here I can use:\n\n{}", lines.join("\n"))
}

/// What each tool does, so activity lines can say what is being generated.
fn effects(shared: &Arc<Shared>) -> HashMap<String, Effect> {
    match ozgent_web::worker::current_tools(&shared.app.tools) {
        Some(t) => t.host.tools().iter().map(|s| (s.name.clone(), s.effect)).collect(),
        None => HashMap::new(),
    }
}

/// The conversation this chat continues, creating one if there is none.
fn conversation(shared: &Arc<Shared>, kind: Kind, chat: &str, msg: &Msg) -> anyhow::Result<i64> {
    let store = shared.app.store.lock().unwrap();
    if let Some(id) = store.channel_conversation(kind.as_str(), chat)? {
        return Ok(id);
    }
    let title = format!("{kind} · {}", msg.name);
    let id = store.create_conversation(&title, None)?;
    store.bind_channel_chat(kind.as_str(), chat, id, &msg.name)?;
    Ok(id)
}

/// Answer one message.
async fn turn_for(shared: Arc<Shared>, kind: Kind, chat: String, msg: Msg) {
    let config = snapshot(&shared.app);
    let Some(model) = model_for(&config) else {
        say(&shared, kind, &chat, "No model is set. Whoever runs me needs to pick one.").await;
        return;
    };
    let conversation = match conversation(&shared, kind, &chat, &msg) {
        Ok(id) => id,
        Err(e) => {
            tracing::error!("{kind}: opening a conversation: {e}");
            say(&shared, kind, &chat, "I could not open the conversation. Try again.").await;
            return;
        }
    };

    let Some(tx) = shared.senders.get(&kind).cloned() else { return };
    let access = config.channels.access(kind);
    let _ = tx.send(Command::Typing { chat: chat.clone() });

    let started = turn::start(
        &shared.app,
        Turn {
            conversation,
            model,
            message: msg.text.clone(),
            thinking: None,
            tools: config.tools.enabled,
            // Whatever the channel is allowed, which may be less than the
            // browser gets: consent arriving over a chat is consent from
            // whoever holds that account, not from whoever owns this machine.
            native_tools: access.tools.map(<[String]>::to_vec),
            images: msg.images.clone(),
            can_ask: true,
        },
    );
    let mut events = match started {
        Ok(rx) => rx,
        Err(e) => {
            tracing::error!("{kind}: starting a turn: {e}");
            say(&shared, kind, &chat, &format!("I could not start: {e}")).await;
            return;
        }
    };

    let mut composer = Composer::new(effects(&shared));
    let first = shared.tokens.next();
    let mut live = Live::new(chat.clone(), tx.clone(), first, limit(kind));
    let handed = shared.clone();
    let mut next = move || handed.tokens.next();

    while let Some(event) = events.recv().await {
        if let Event::Permission { id, name, arguments, effect } = &event {
            composer.absorb(&event);
            // The reply so far goes out before the question, so the question
            // is the last thing in the chat when it arrives.
            live.flush(&composer.render(), &mut next);
            ask(&shared, kind, &chat, &tx, id, name, arguments, *effect).await;
            continue;
        }
        composer.absorb(&event);
        if access.stream {
            live.update(&composer.render(), &mut next);
        }
    }

    live.flush(&composer.render(), &mut next);
    if !live.posted() {
        // Nothing at all came back. Silence would read as a machine that had
        // stopped working.
        say(&shared, kind, &chat, "I have nothing to say to that.").await;
    }
}

/// Post a permission question and remember what it is about.
#[allow(clippy::too_many_arguments)]
async fn ask(
    shared: &Arc<Shared>,
    kind: Kind,
    chat: &str,
    tx: &UnboundedSender<Command>,
    call_id: &str,
    tool: &str,
    arguments: &serde_json::Value,
    effect: Effect,
) {
    let token = shared.tokens.next();
    shared.asks.lock().unwrap().insert(
        token,
        Ask {
            call_id: call_id.to_string(),
            chat: chat.to_string(),
            kind,
            tool: tool.to_string(),
        },
    );
    shared.waiting.lock().unwrap().insert((kind, chat.to_string()), token);

    let _ = tx.send(Command::Ask {
        chat: chat.to_string(),
        token,
        question: Box::new(Question {
            id: call_id.to_string(),
            tool: tool.to_string(),
            effect,
            detail: describe(arguments),
        }),
    });
}

/// A tool call's arguments, short enough to read on a phone.
pub fn describe(arguments: &serde_json::Value) -> String {
    const MAX_VALUE: usize = 160;
    let Some(object) = arguments.as_object() else {
        return String::new();
    };

    let mut lines = Vec::new();
    for (key, value) in object {
        // The marker `permit` adds when it asks before the call is finished.
        if key == "…" {
            lines.push("(still being written)".to_string());
            continue;
        }
        let rendered = match value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let rendered = rendered.replace('\n', " ");
        let rendered = if rendered.chars().count() > MAX_VALUE {
            let cut: String = rendered.chars().take(MAX_VALUE).collect();
            format!("{cut}…")
        } else {
            rendered
        };
        lines.push(format!("{key}: {rendered}"));
    }
    lines.join("\n")
}

/// Say one thing, outside any turn.
async fn say(shared: &Arc<Shared>, kind: Kind, chat: &str, markdown: &str) {
    if let Some(tx) = shared.senders.get(&kind) {
        let _ = tx.send(Command::Post {
            chat: chat.to_string(),
            token: shared.tokens.next(),
            markdown: markdown.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_are_shown_as_lines_a_phone_can_hold() {
        let args = serde_json::json!({ "path": "poem.txt", "content": "x".repeat(500) });
        let text = describe(&args);
        assert!(text.contains("path: poem.txt"), "{text}");
        for line in text.lines() {
            assert!(line.chars().count() <= 200, "{line}");
        }
    }

    #[test]
    fn an_unfinished_call_says_so_rather_than_showing_a_stray_key() {
        // `permit` asks before the arguments exist and marks the gap; showing
        // the marker raw would read as a corrupted message.
        let args = serde_json::json!({ "path": "poem.txt", "…": "still being written" });
        let text = describe(&args);
        assert!(text.contains("(still being written)"), "{text}");
        assert!(!text.contains('…') || text.contains("(still"), "{text}");
    }

    #[test]
    fn newlines_in_an_argument_do_not_become_extra_lines() {
        // Otherwise a multi-line command reads as several separate arguments.
        let args = serde_json::json!({ "command": "git add .\ngit commit" });
        assert_eq!(describe(&args).lines().count(), 1);
    }

    #[test]
    fn arguments_that_are_not_an_object_produce_nothing() {
        assert_eq!(describe(&serde_json::json!("just a string")), "");
        assert_eq!(describe(&serde_json::Value::Null), "");
    }

    #[test]
    fn a_pairing_code_is_readable_and_not_the_same_twice() {
        // It gets read off a terminal and typed into a phone, so the alphabet
        // avoids characters that look alike; and a code that repeated would
        // stop being single-use.
        let a = pairing_code();
        assert_eq!(a.len(), 6);
        assert!(a.chars().all(|c| "3479CDFHJKMNPRTWXY".contains(c)), "{a}");
        assert!(!a.chars().any(|c| "01OIL".contains(c)), "{a}");

        let many: std::collections::HashSet<String> = (0..50).map(|_| pairing_code()).collect();
        assert!(many.len() > 40, "codes repeat: {} distinct of 50", many.len());
    }

    #[test]
    fn the_channel_model_beats_the_general_default() {
        // A phone is a poor place to wait on the biggest model installed.
        let mut config = Config::default();
        config.default_model = Some("big".into());
        assert_eq!(model_for(&config).as_deref(), Some("big"));

        config.channels.model = Some("small".into());
        assert_eq!(model_for(&config).as_deref(), Some("small"));
    }

    #[test]
    fn with_no_model_anywhere_there_is_nothing_to_answer_with() {
        assert_eq!(model_for(&Config::default()), None);
    }
}
