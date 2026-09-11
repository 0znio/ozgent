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
//!
//! It is also a supervisor. Channels are started, stopped and restarted while
//! it runs — from `/admin`, or because `config.toml` changed under it — so
//! setting up a channel never means restarting ozgent. [`Gateway::apply`] is
//! the one place that compares what is configured with what is running.
//!
//! Only one process may answer the channels: two would fight over the
//! Telegram token (it allows one reader) and the WhatsApp session (a second
//! connection replaces the first). A lock file decides which; the other is
//! told who has it, and takes over if that process goes away.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ozgent_core::channels::{self, Kind};
use ozgent_core::permission::{Choice, Effect};
use ozgent_core::{Config, Paths};
use ozgent_web::admin::{Fut, GatewayControl, GatewayView, Phase, Runtime};
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

const KINDS: [Kind; 2] = [Kind::Telegram, Kind::WhatsApp];

/// How long a WhatsApp linking code is shown before giving up. WhatsApp
/// replaces the code every twenty seconds and stops after a few minutes.
const LINK_FOR: Duration = Duration::from_secs(180);

/// How often a process that does not hold the channels checks whether it can
/// take them over, and a linking code is checked for expiry.
const TICK: Duration = Duration::from_secs(5);

/// A question posted to a chat and not yet answered.
struct Ask {
    /// The tool call id the inference thread is blocked on.
    call_id: String,
    chat: String,
    kind: Kind,
    tool: String,
}

/// A channel that is running.
struct Instance {
    /// Which start this is. Events from an instance that has since been
    /// stopped carry an older number and are dropped.
    generation: u64,
    /// The settings it was started with that need a restart to change.
    fingerprint: String,
    handle: tokio::task::AbortHandle,
}

/// Where the terminal output of the gateway goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `ozgent gateway`: the terminal is the only place to see anything, so
    /// connections, failures and the pairing code are printed.
    Terminal,
    /// Inside `ozgent web`: connections and failures are printed, the rest is
    /// on `/admin`.
    Web,
}

struct Shared {
    app: State,
    paths: Paths,
    mode: Mode,
    tokens: Tokens,
    senders: Mutex<HashMap<Kind, UnboundedSender<Command>>>,
    instances: Mutex<HashMap<Kind, Instance>>,
    runtime: Mutex<HashMap<Kind, Runtime>>,
    /// The settings a channel failed with. It is not started again with the
    /// same ones — that would fail the same way, forever — until they change
    /// or someone presses restart.
    failed_with: Mutex<HashMap<Kind, String>>,
    /// When WhatsApp linking was asked for; `None` when it was not.
    linking: Mutex<Option<Instant>>,
    generation: AtomicU64,
    inbound: Sender<(Kind, u64, Inbound)>,
    /// Held for as long as this process answers the channels.
    lock: Mutex<Option<std::fs::File>>,
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

/// The running gateway. Cheap to clone; every clone is the same gateway.
#[derive(Clone)]
pub struct Gateway {
    shared: Arc<Shared>,
}

/// Start the gateway inside this process and register it with the web app,
/// so `/admin` can control it. Channels that are set up start straight away.
pub fn start(app: State, paths: Paths, mode: Mode) -> Gateway {
    let (inbound_tx, inbound_rx) = channel::<(Kind, u64, Inbound)>(64);
    let shared = Arc::new(Shared {
        app: app.clone(),
        paths,
        mode,
        tokens: Tokens::default(),
        senders: Mutex::new(HashMap::new()),
        instances: Mutex::new(HashMap::new()),
        runtime: Mutex::new(HashMap::new()),
        failed_with: Mutex::new(HashMap::new()),
        linking: Mutex::new(None),
        generation: AtomicU64::new(1),
        inbound: inbound_tx,
        lock: Mutex::new(None),
        asks: Mutex::new(HashMap::new()),
        waiting: Mutex::new(HashMap::new()),
        running: Mutex::new(HashMap::new()),
        pairing: Mutex::new(pairing_code()),
    });
    let gateway = Gateway { shared };

    tokio::spawn(receive(gateway.shared.clone(), inbound_rx));
    // Changes other programs make to config.toml arrive through here, and
    // each one ends in `apply`.
    ozgent_web::state::watch_config(&app);
    tokio::spawn({
        let gateway = gateway.clone();
        async move {
            loop {
                tokio::time::sleep(TICK).await;
                gateway.tick();
            }
        }
    });

    let _ = app.gateway.set(Arc::new(gateway.clone()));
    gateway.apply();
    gateway
}

/// `ozgent gateway`: answer the channels from this terminal until stopped.
pub async fn run(app: State, paths: Paths) -> anyhow::Result<()> {
    let config = snapshot(&app);
    let linked = whatsapp_linked(&paths);
    let ready: Vec<Kind> = KINDS
        .into_iter()
        .filter(|k| wanted(&config, *k, linked, false, &paths))
        .collect();
    if ready.is_empty() {
        anyhow::bail!(
            "no channel is set up yet. Set one up with:\n\n  \
             ozgent gateway telegram\n  ozgent gateway whatsapp\n\n\
             Or run `ozgent web` and use http://localhost:7333/admin."
        );
    }
    let gateway = start(app, paths, Mode::Terminal);
    if let Some(other) = gateway.holder() {
        anyhow::bail!(
            "the channels are already being answered by {other}. Stop that first — only one \
             program can hold a bot token or a WhatsApp session."
        );
    }
    announce(&gateway.shared, &config, &ready);
    std::future::pending::<()>().await;
    Ok(())
}

impl Gateway {
    fn note(&self, text: impl std::fmt::Display) {
        note(&self.shared, text);
    }

    /// Who holds the channels, when it is not this process.
    fn holder(&self) -> Option<String> {
        if self.shared.lock.lock().unwrap().is_some() {
            return None;
        }
        let path = lock_path(&self.shared.paths);
        let pid = std::fs::read_to_string(&path).unwrap_or_default();
        let pid = pid.trim();
        Some(if pid.is_empty() {
            "another ozgent".to_string()
        } else {
            format!("another ozgent (process {pid})")
        })
    }

    /// Take the lock if nobody has it. True when this process holds it.
    fn acquire(&self) -> bool {
        let mut held = self.shared.lock.lock().unwrap();
        if held.is_some() {
            return true;
        }
        let path = lock_path(&self.shared.paths);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let Ok(file) = std::fs::OpenOptions::new().create(true).truncate(false).write(true).read(true).open(&path)
        else {
            return false;
        };
        if file.try_lock().is_err() {
            return false;
        }
        use std::io::{Seek, Write};
        let mut f = &file;
        let _ = f.set_len(0);
        let _ = f.seek(std::io::SeekFrom::Start(0));
        let _ = write!(f, "{}", std::process::id());
        *held = Some(file);
        true
    }

    fn set_runtime(&self, kind: Kind, change: impl FnOnce(&mut Runtime)) {
        let mut all = self.shared.runtime.lock().unwrap();
        change(all.entry(kind).or_default());
    }

    fn phase(&self, kind: Kind) -> Phase {
        self.shared.runtime.lock().unwrap().get(&kind).map(|r| r.phase).unwrap_or_default()
    }

    /// Periodic housekeeping: take over the channels when the process that had
    /// them has gone, and give up on a linking code nobody scanned.
    fn tick(&self) {
        if self.shared.lock.lock().unwrap().is_none() {
            self.apply();
        }
        let started = *self.shared.linking.lock().unwrap();
        if let Some(started) = started {
            if started.elapsed() > LINK_FOR && self.phase(Kind::WhatsApp) != Phase::Connected {
                *self.shared.linking.lock().unwrap() = None;
                self.stop(Kind::WhatsApp);
                self.set_runtime(Kind::WhatsApp, |r| {
                    r.phase = Phase::Failed;
                    r.qr = None;
                    r.detail = Some("the code was not scanned in time. Link again for a new one.".into());
                });
                self.note("whatsapp: the linking code was not scanned in time");
            }
        }
    }

    fn start_channel(&self, kind: Kind, config: &Config, fingerprint: String) {
        let generation = self.shared.generation.fetch_add(1, Ordering::SeqCst);
        let (command_tx, command_rx) = unbounded_channel::<Command>();
        let (tagged_tx, mut tagged_rx) = channel::<Inbound>(64);

        // Each channel speaks plain `Inbound`; kind and generation are added
        // here so a channel never has to know what else is running.
        let to_gateway = self.shared.inbound.clone();
        tokio::spawn(async move {
            while let Some(event) = tagged_rx.recv().await {
                if to_gateway.send((kind, generation, event)).await.is_err() {
                    break;
                }
            }
        });

        // A channel that cannot start must say so on the same path everything
        // else is reported on, or it would sit there silently doing nothing.
        let failures = self.shared.inbound.clone();
        let report = move |reason: String| async move {
            let _ = failures.send((kind, generation, Inbound::Failed { reason })).await;
        };

        let task = match kind {
            Kind::Telegram => {
                // The environment wins, for anyone who would rather not have a
                // password in a file that gets copied around.
                let token = telegram_token(config);
                tokio::spawn(async move {
                    if let Err(e) = crate::telegram::run(token, tagged_tx, command_rx).await {
                        report(e.to_string()).await;
                    }
                })
            }
            Kind::WhatsApp => {
                let wa = config.channels.whatsapp.clone();
                let paths = self.shared.paths.clone();
                tokio::spawn(async move {
                    let bridge = match crate::whatsapp::locate(wa.bridge.as_deref(), &paths) {
                        Ok(b) => b,
                        Err(e) => return report(e.to_string()).await,
                    };
                    let state = paths.channel_dir("whatsapp").join("auth");
                    if let Err(e) =
                        crate::whatsapp::run(wa.node, bridge, state, wa.self_chat, tagged_tx, command_rx)
                            .await
                    {
                        report(e.to_string()).await;
                    }
                })
            }
        };

        self.shared.senders.lock().unwrap().insert(kind, command_tx);
        self.shared.instances.lock().unwrap().insert(
            kind,
            Instance { generation, fingerprint, handle: task.abort_handle() },
        );
        self.set_runtime(kind, |r| {
            r.phase = Phase::Starting;
            r.detail = None;
            r.qr = None;
        });
    }

    /// Stop a channel. Aborting the task drops the bridge process with it
    /// (`kill_on_drop`) and ends a Telegram long poll mid-wait.
    fn stop(&self, kind: Kind) {
        self.shared.senders.lock().unwrap().remove(&kind);
        if let Some(instance) = self.shared.instances.lock().unwrap().remove(&kind) {
            instance.handle.abort();
        }
        self.set_runtime(kind, |r| {
            r.phase = Phase::Off;
            r.qr = None;
            r.who = None;
            r.detail = None;
        });
    }

    fn current_generation(&self, kind: Kind) -> Option<u64> {
        self.shared.instances.lock().unwrap().get(&kind).map(|i| i.generation)
    }
}

impl GatewayControl for Gateway {
    fn view(&self) -> GatewayView {
        let hosted = self.shared.lock.lock().unwrap().is_some();
        let runtime = self.shared.runtime.lock().unwrap().clone();
        let config = snapshot(&self.shared.app);
        let mut telegram = runtime.get(&Kind::Telegram).cloned().unwrap_or_default();
        let mut whatsapp = runtime.get(&Kind::WhatsApp).cloned().unwrap_or_default();
        whatsapp.linked = whatsapp_linked(&self.shared.paths);
        whatsapp.link_left = self
            .shared
            .linking
            .lock()
            .unwrap()
            .map(|started| LINK_FOR.saturating_sub(started.elapsed()).as_secs());
        whatsapp.installed = crate::whatsapp::locate(config.channels.whatsapp.bridge.as_deref(), &self.shared.paths)
            .ok()
            .map(|b| crate::whatsapp::is_installed(&b));
        if !hosted {
            for r in [&mut telegram, &mut whatsapp] {
                r.phase = Phase::Elsewhere;
            }
        }
        GatewayView {
            hosted,
            elsewhere: self.holder(),
            pairing: hosted.then(|| self.shared.pairing.lock().unwrap().clone()),
            telegram,
            whatsapp,
        }
    }

    fn apply(&self) {
        if !self.acquire() {
            return;
        }
        let config = snapshot(&self.shared.app);
        let linking = self.shared.linking.lock().unwrap().is_some();
        let linked = whatsapp_linked(&self.shared.paths);

        for kind in KINDS {
            let want = wanted(&config, kind, linked, linking, &self.shared.paths);
            let fingerprint = fingerprint(kind, &config);
            let running = self
                .shared
                .instances
                .lock()
                .unwrap()
                .get(&kind)
                .map(|i| i.fingerprint.clone());

            match (want, running) {
                (true, None) => {
                    let failed = self.shared.failed_with.lock().unwrap().get(&kind).cloned();
                    if failed.as_deref() != Some(fingerprint.as_str()) {
                        self.shared.failed_with.lock().unwrap().remove(&kind);
                        self.start_channel(kind, &config, fingerprint);
                    }
                }
                (true, Some(was)) if was != fingerprint => {
                    self.note(format!("{kind}: settings changed, restarting"));
                    self.stop(kind);
                    self.shared.failed_with.lock().unwrap().remove(&kind);
                    self.start_channel(kind, &config, fingerprint);
                }
                (true, Some(_)) => {}
                (false, Some(_)) => {
                    self.stop(kind);
                    self.note(format!("{kind}: stopped"));
                }
                (false, None) => {
                    // Switched off clears a failure; merely not ready yet (no
                    // token, not linked) keeps it on show.
                    if !config.channels.enabled || !config.channels.enabled(kind) {
                        self.shared.failed_with.lock().unwrap().remove(&kind);
                        self.set_runtime(kind, |r| {
                            if r.phase != Phase::Installing {
                                *r = Runtime::default();
                            }
                        });
                    }
                }
            }
        }
    }

    fn restart(&self, kind: Kind) {
        self.stop(kind);
        self.shared.failed_with.lock().unwrap().remove(&kind);
        self.apply();
    }

    fn new_pairing_code(&self) -> String {
        let code = pairing_code();
        *self.shared.pairing.lock().unwrap() = code.clone();
        code
    }

    fn check_telegram<'a>(&'a self, token: &'a str) -> Fut<'a, Result<String, String>> {
        Box::pin(async move {
            crate::telegram::Telegram::new(token.to_string()).identify().await.map_err(|e| e.to_string())
        })
    }

    fn link_whatsapp(&self) -> Fut<'_, Result<(), String>> {
        Box::pin(async move {
            if !self.acquire() {
                return Err(format!(
                    "{} is answering the channels. Link from that one, or stop it first.",
                    self.holder().unwrap_or_default()
                ));
            }
            let config = snapshot(&self.shared.app);
            let wa = config.channels.whatsapp.clone();
            let bridge = crate::whatsapp::locate(wa.bridge.as_deref(), &self.shared.paths)
                .map_err(|e| e.to_string())?;

            // Installing takes a minute; the page watches it happen rather
            // than waiting on one long request.
            let gateway = self.clone();
            tokio::spawn(async move {
                if !crate::whatsapp::is_installed(&bridge) {
                    gateway.set_runtime(Kind::WhatsApp, |r| {
                        r.phase = Phase::Installing;
                        r.detail = Some("installing the WhatsApp bridge (npm install)".into());
                    });
                    gateway.note("whatsapp: installing the bridge");
                    if let Err(e) = crate::whatsapp::install_quietly(&wa.node, &bridge).await {
                        gateway.set_runtime(Kind::WhatsApp, |r| {
                            r.phase = Phase::Failed;
                            r.detail = Some(e.to_string());
                        });
                        return;
                    }
                }
                *gateway.shared.linking.lock().unwrap() = Some(Instant::now());
                let saved = {
                    let mut c = gateway.shared.app.config.lock().unwrap_or_else(|e| e.into_inner());
                    c.channels.set_enabled(Kind::WhatsApp, true);
                    c.save(&gateway.shared.paths)
                };
                if let Err(e) = saved {
                    gateway.set_runtime(Kind::WhatsApp, |r| {
                        r.phase = Phase::Failed;
                        r.detail = Some(format!("saving the settings: {e}"));
                    });
                    return;
                }
                // A stopped or failed bridge is started fresh, so it asks for
                // a code rather than reusing a dead session.
                gateway.stop(Kind::WhatsApp);
                gateway.shared.failed_with.lock().unwrap().remove(&Kind::WhatsApp);
                gateway.apply();
            });
            Ok(())
        })
    }

    fn unlink_whatsapp(&self) -> Fut<'_, Result<(), String>> {
        Box::pin(async move {
            if !self.acquire() {
                return Err(format!(
                    "{} is answering the channels. Sign out from that one, or stop it first.",
                    self.holder().unwrap_or_default()
                ));
            }
            *self.shared.linking.lock().unwrap() = None;
            let config = snapshot(&self.shared.app);
            let state = self.shared.paths.channel_dir("whatsapp").join("auth");

            // A running bridge is already connected: ask it to sign out, and
            // give it a moment to be told yes.
            let sender = self.shared.senders.lock().unwrap().get(&Kind::WhatsApp).cloned();
            let result = if let Some(tx) = sender.filter(|_| self.phase(Kind::WhatsApp) == Phase::Connected) {
                let _ = tx.send(Command::Logout);
                for _ in 0..40 {
                    let done = self
                        .shared
                        .instances
                        .lock()
                        .unwrap()
                        .get(&Kind::WhatsApp)
                        .is_none_or(|i| i.handle.is_finished());
                    if done {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                self.stop(Kind::WhatsApp);
                crate::whatsapp::logout(&state).map_err(|e| e.to_string())
            } else {
                self.stop(Kind::WhatsApp);
                match crate::whatsapp::locate(config.channels.whatsapp.bridge.as_deref(), &self.shared.paths) {
                    Ok(bridge) => crate::whatsapp::unlink(&config.channels.whatsapp.node, &bridge, &state)
                        .await
                        .map_err(|e| e.to_string()),
                    Err(_) => crate::whatsapp::logout(&state).map_err(|e| e.to_string()),
                }
            };
            self.set_runtime(Kind::WhatsApp, |r| *r = Runtime::default());
            self.note("whatsapp: signed out");
            result
        })
    }
}

/// Whether a channel should be running, as far as the settings go.
fn wanted(config: &Config, kind: Kind, linked: bool, linking: bool, _paths: &Paths) -> bool {
    if !config.channels.enabled || !config.channels.enabled(kind) {
        return false;
    }
    match kind {
        Kind::Telegram => !telegram_token(config).trim().is_empty(),
        // An unlinked bridge would ask for a code nobody is looking at, again
        // and again; it runs unlinked only while someone asked to link.
        Kind::WhatsApp => linked || linking,
    }
}

/// The settings a running channel cannot pick up without a restart.
fn fingerprint(kind: Kind, config: &Config) -> String {
    match kind {
        Kind::Telegram => telegram_token(config),
        Kind::WhatsApp => {
            let wa = &config.channels.whatsapp;
            format!("{}|{:?}|{}", wa.node, wa.bridge, wa.self_chat)
        }
    }
}

fn telegram_token(config: &Config) -> String {
    std::env::var("OZGENT_TELEGRAM_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
        .unwrap_or_else(|| config.channels.telegram.token.clone())
}

/// Whether WhatsApp credentials are saved on this machine.
pub fn whatsapp_linked(paths: &Paths) -> bool {
    paths.channel_dir("whatsapp").join("auth").join("creds.json").is_file()
}

/// The lock that says which process answers the channels.
pub fn lock_path(paths: &Paths) -> std::path::PathBuf {
    paths.channels_dir().join("gateway.lock")
}

/// Who holds the channels right now, if anyone does: for the terminal setup,
/// which must not link a WhatsApp session a running server is using.
pub fn held_elsewhere(paths: &Paths) -> Option<String> {
    let path = lock_path(paths);
    let file = std::fs::OpenOptions::new().read(true).write(true).open(&path).ok()?;
    if file.try_lock().is_ok() {
        let _ = file.unlock();
        return None;
    }
    let pid = std::fs::read_to_string(&path).unwrap_or_default();
    Some(match pid.trim() {
        "" => "another ozgent".to_string(),
        pid => format!("another ozgent (process {pid})"),
    })
}

fn note(_shared: &Shared, text: impl std::fmt::Display) {
    tracing::info!("gateway: {text}");
    println!("  {text}");
}

/// What the operator sees when `ozgent gateway` starts.
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
        } else if access.allow.is_empty() && !(*kind == Kind::WhatsApp && config.channels.whatsapp.self_chat) {
            "nobody yet".to_string()
        } else {
            let mut n = access.allow.len();
            if *kind == Kind::WhatsApp && config.channels.whatsapp.self_chat {
                n += 1;
            }
            format!("{n} allowed")
        };
        println!("  {kind:<9}  {who}");
    }
    println!();

    let open = active
        .iter()
        .any(|k| channels::is_open_to_everyone(config.channels.access(*k).allow));
    if open {
        println!("  WARNING: a channel admits everyone. Anyone who finds it can use this");
        println!("  machine's tools, subject only to your permission rules. Change it with");
        println!("  `ozgent gateway <channel>`.");
        println!();
    }
    println!("  To allow someone else, have them send:   /pair {}", shared.pairing.lock().unwrap());
    println!("  The code works once, and changes after it is used.");
    println!("  Changes made with `ozgent gateway <channel>` or /admin apply without a restart.");
    println!();
    println!("press ctrl-c to stop");
}

/// A short code from the operating system's random source.
///
/// Read off a terminal or the admin page and typed within the minute, so it
/// only needs to be unguessable from outside, not long.
fn pairing_code() -> String {
    // No vowels and no look-alikes, so a code read aloud or off a screen is
    // typed back correctly.
    const ALPHABET: &[u8] = b"3479CDFHJKMNPRTWXY";
    let hex = ozgent_core::secret::random_token(8);
    let mut n = u64::from_str_radix(&hex, 16).unwrap_or(0);
    let mut out = String::new();
    for _ in 0..6 {
        out.push(ALPHABET[(n % ALPHABET.len() as u64) as usize] as char);
        n /= ALPHABET.len() as u64;
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

/// Every channel's events, one at a time.
async fn receive(shared: Arc<Shared>, mut rx: tokio::sync::mpsc::Receiver<(Kind, u64, Inbound)>) {
    let gateway = Gateway { shared: shared.clone() };
    let mut chats: HashMap<Key, UnboundedSender<Msg>> = HashMap::new();
    while let Some((kind, generation, event)) = rx.recv().await {
        // From an instance that has since been stopped or replaced.
        if gateway.current_generation(kind) != Some(generation) {
            continue;
        }
        route(&gateway, &mut chats, kind, event).await;
    }
}

async fn route(
    gateway: &Gateway,
    chats: &mut HashMap<Key, UnboundedSender<Msg>>,
    kind: Kind,
    event: Inbound,
) {
    let shared = &gateway.shared;
    match event {
        Inbound::Ready { who } => {
            gateway.note(format!("{kind}: connected as {who}"));
            if kind == Kind::WhatsApp {
                *shared.linking.lock().unwrap() = None;
            }
            gateway.set_runtime(kind, |r| {
                r.phase = Phase::Connected;
                r.who = Some(who);
                r.qr = None;
                r.detail = None;
            });
        }
        Inbound::Notice { text } => {
            gateway.note(format!("{kind}: {text}"));
            gateway.set_runtime(kind, |r| r.detail = Some(text));
        }
        Inbound::Qr { data } => {
            // Only while someone asked to link; otherwise a lost session
            // would sit there offering codes to nobody.
            if shared.linking.lock().unwrap().is_none() {
                gateway.stop(kind);
                gateway.set_runtime(kind, |r| {
                    r.phase = Phase::Failed;
                    r.detail = Some("WhatsApp is not linked any more. Link it again.".into());
                });
                gateway.note("whatsapp: not linked any more; link it again with `ozgent gateway whatsapp` or /admin");
                return;
            }
            gateway.set_runtime(kind, |r| {
                r.phase = Phase::Linking;
                r.qr = Some(data);
                r.detail = Some("scan the code with WhatsApp ▸ Linked devices ▸ Link a device".into());
            });
        }
        Inbound::Failed { reason } => {
            // Printed as well as logged: the operator may be looking at a
            // terminal that has just stopped doing anything.
            tracing::error!("{kind}: {reason}");
            if shared.mode == Mode::Terminal {
                eprintln!("  {kind}: {reason}");
            }
            let fingerprint = shared.instances.lock().unwrap().remove(&kind).map(|i| i.fingerprint);
            shared.senders.lock().unwrap().remove(&kind);
            if let Some(f) = fingerprint {
                shared.failed_with.lock().unwrap().insert(kind, f);
            }
            if kind == Kind::WhatsApp {
                *shared.linking.lock().unwrap() = None;
            }
            gateway.set_runtime(kind, |r| {
                r.phase = Phase::Failed;
                r.qr = None;
                r.detail = Some(reason);
            });
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
    admits_message(&snapshot(&shared.app), kind, msg)
}

/// The whole admission decision, as a function of the configuration alone.
///
/// Separated from the shared state so it can be tested directly: this is the
/// rule that decides whether a stranger on the internet reaches this machine's
/// tools, and it should not be reachable only through a running gateway.
pub fn admits_message(config: &Config, kind: Kind, msg: &Msg) -> bool {
    if msg.group && kind == Kind::WhatsApp && !config.channels.whatsapp.groups {
        return false;
    }
    // Your own chat with yourself. The bridge only reports one when the
    // setting is on, and the sender is the account that scanned the QR — so
    // requiring them to also write their own number in `allow` would be a rule
    // with nobody on the other side of it, and a confusing silence for anyone
    // who turned the setting on and expected it to work.
    if msg.own {
        return true;
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
            note(shared, format!("{kind}: allowed {} ({identity})", msg.name));
            if shared.mode == Mode::Terminal {
                println!("  next pairing code: /pair {}", shared.pairing.lock().unwrap());
            }
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
    let tx = shared.senders.lock().unwrap().get(&kind).cloned();
    if let Some(tx) = tx {
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

    let Some(tx) = shared.senders.lock().unwrap().get(&kind).cloned() else { return };
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
            tools_off: Vec::new(),
            images: msg.images.clone(),
            // The operator can switch approvals off for a channel: then only
            // what the rules allow outright runs, and anything that would ask
            // is refused.
            can_ask: access.approve,
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
    let tx = shared.senders.lock().unwrap().get(&kind).cloned();
    if let Some(tx) = tx {
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

    fn message(over: fn(&mut Msg)) -> Msg {
        let mut m = Msg {
            chat: "c".into(),
            sender_id: "1555".into(),
            handle: None,
            name: "someone".into(),
            text: "hello".into(),
            images: Vec::new(),
            group: false,
            own: false,
        };
        over(&mut m);
        m
    }

    #[test]
    fn a_stranger_is_not_admitted() {
        let config = Config::default();
        assert!(!admits_message(&config, Kind::WhatsApp, &message(|_| {})));
        assert!(!admits_message(&config, Kind::Telegram, &message(|_| {})));
    }

    #[test]
    fn your_own_chat_with_yourself_needs_no_allowlist_entry() {
        // The sender is the account that scanned the QR code. Requiring them
        // to also write their own number down would be a rule with nobody on
        // the other side of it, and reads as ozgent silently ignoring you.
        let config = Config::default();
        assert!(admits_message(&config, Kind::WhatsApp, &message(|m| m.own = true)));
    }

    #[test]
    fn a_group_is_never_admitted_by_the_self_chat_rule() {
        // `own` skips the allowlist, so it must not combine with a group —
        // that would be a way for someone never allowlisted to steer ozgent
        // through a group the operator is also in.
        let config = Config::default();
        let in_a_group = message(|m| {
            m.own = true;
            m.group = true;
        });
        assert!(!admits_message(&config, Kind::WhatsApp, &in_a_group));
    }

    #[test]
    fn an_allowlisted_number_is_admitted_on_its_own_channel_only() {
        let mut config = Config::default();
        config.channels.whatsapp.allow = vec!["1555".into()];
        assert!(admits_message(&config, Kind::WhatsApp, &message(|_| {})));
        assert!(!admits_message(&config, Kind::Telegram, &message(|_| {})));
    }

    #[test]
    fn a_group_is_answered_only_when_groups_are_switched_on() {
        let mut config = Config::default();
        config.channels.whatsapp.allow = vec!["1555".into()];
        let group = message(|m| m.group = true);
        assert!(!admits_message(&config, Kind::WhatsApp, &group));

        config.channels.whatsapp.groups = true;
        assert!(admits_message(&config, Kind::WhatsApp, &group));
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
