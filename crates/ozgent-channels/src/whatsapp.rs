//! WhatsApp, through a Node bridge.
//!
//! WhatsApp publishes no protocol and there is no Rust client, so this talks to
//! a small Node program — `bridge/whatsapp` — over stdin and stdout, exactly
//! the way ozgent already talks to the Python tool worker: a child process,
//! newline-delimited JSON, and nothing shared but the pipe. The bridge links
//! the account as a second device, the way WhatsApp Web does.
//!
//! That design is a deliberate trade, and the costs belong here in the open:
//! ozgent stops being a single binary for anyone who turns this on, Node has to
//! be installed, and automating a personal account is against WhatsApp's terms
//! of service — accounts have been banned for it. The alternative, Meta's Cloud
//! API, is a business product needing a public webhook and approved templates,
//! and cannot talk to the number you already have.
//!
//! Unlike Telegram there are no buttons: a permission question is posted as
//! numbered options and answered by typing, which is why [`crate::chat`] reads
//! typed answers on every channel rather than only this one.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use ozgent_core::ImageSource;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command as Process};
use tokio::sync::mpsc::{Sender, UnboundedReceiver};

use crate::chat::{ANSWERS, Command, Inbound, Msg, Question};
use crate::markup::{Flavour, render};
use crate::split::{WHATSAPP_LIMIT, split};

/// Where the bridge is, searched the way the Python runtime is.
///
/// Checked in order: an explicit setting, `bridge/whatsapp` beside the
/// executable (the installed layout), the repository's own copy (`cargo run`),
/// then the ozgent home.
pub fn locate(configured: Option<&Path>, paths: &ozgent_core::Paths) -> anyhow::Result<PathBuf> {
    let mut tried = Vec::new();
    let check = |p: PathBuf, tried: &mut Vec<PathBuf>| -> Option<PathBuf> {
        if p.join("index.mjs").is_file() {
            return Some(p);
        }
        tried.push(p);
        None
    };

    if let Some(dir) = configured {
        // An explicit setting that does not resolve is a mistake to report,
        // not something to quietly fall back from.
        return check(dir.to_path_buf(), &mut tried)
            .ok_or_else(|| anyhow::anyhow!("no index.mjs in {}", dir.display()));
    }

    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors().skip(1).take(5) {
            if let Some(found) = check(ancestor.join("bridge").join("whatsapp"), &mut tried) {
                return Ok(found);
            }
        }
    }
    if let Some(found) = check(paths.root().join("bridges").join("whatsapp"), &mut tried) {
        return Ok(found);
    }

    let list = tried.iter().map(|p| format!("\n  {}", p.display())).collect::<String>();
    Err(anyhow::anyhow!("the WhatsApp bridge was not found. Looked in:{list}"))
}

/// Whether the bridge's dependencies are installed.
pub fn is_installed(bridge: &Path) -> bool {
    bridge.join("node_modules").join("@whiskeysockets").join("baileys").is_dir()
}

/// Install the bridge's dependencies with npm.
pub async fn install(node: &str, bridge: &Path) -> anyhow::Result<()> {
    // npm sits beside node, which matters when node is a version-manager shim
    // that is not the one on `PATH`.
    let npm = which_npm(node);
    let status = Process::new(&npm)
        .arg("install")
        .arg("--no-audit")
        .arg("--no-fund")
        .current_dir(bridge)
        .status()
        .await
        .map_err(|e| anyhow::anyhow!("could not run {npm}: {e}. Is Node installed?"))?;
    if !status.success() {
        anyhow::bail!("npm install failed in {}", bridge.display());
    }
    Ok(())
}

/// Install the bridge's dependencies without writing to the terminal: for a
/// server, where npm's progress output would land in the middle of its own.
/// A failure carries the end of npm's output, which is where npm says why.
pub async fn install_quietly(node: &str, bridge: &Path) -> anyhow::Result<()> {
    if Process::new(node).arg("--version").output().await.is_err() {
        anyhow::bail!(
            "Node.js is not installed, and the WhatsApp bridge needs it. Install Node 18 or \
             newer with your package manager, then try again."
        );
    }
    let npm = which_npm(node);
    let out = Process::new(&npm)
        .args(["install", "--no-audit", "--no-fund", "--loglevel=error"])
        .current_dir(bridge)
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("could not run {npm}: {e}. Is npm installed?"))?;
    if !out.status.success() {
        let text = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = text.lines().rev().take(6).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        anyhow::bail!("npm install failed in {}:\n{}", bridge.display(), tail.join("\n"));
    }
    Ok(())
}

fn which_npm(node: &str) -> String {
    let path = Path::new(node);
    match path.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(dir) => dir.join("npm").to_string_lossy().into_owned(),
        None => "npm".to_string(),
    }
}

/// Start the bridge process.
fn spawn(
    node: &str,
    bridge: &Path,
    state: &Path,
    login: bool,
    self_chat: bool,
) -> anyhow::Result<Child> {
    std::fs::create_dir_all(state)?;
    let mut cmd = Process::new(node);
    cmd.arg(bridge.join("index.mjs"))
        .arg("--state")
        .arg(state)
        .current_dir(bridge)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // The child must not outlive a killed gateway holding the WhatsApp
        // session open.
        .kill_on_drop(true);
    if login {
        cmd.arg("--login").arg("1");
    }
    // Passed rather than read by the bridge, so the one place that decides
    // whether your own notes become prompts is `config.toml`.
    cmd.arg("--self-chat").arg(if self_chat { "1" } else { "0" });
    cmd.spawn()
        .map_err(|e| anyhow::anyhow!("could not run {node}: {e}. Is Node installed?"))
}

/// Link this machine to a WhatsApp account, printing the QR to the terminal.
///
/// Returns the account it linked. Runs to completion rather than in the
/// background: linking is something a person does once, watching.
pub async fn login(node: &str, bridge: &Path, state: &Path) -> anyhow::Result<String> {
    let mut child = spawn(node, bridge, state, true, false)?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let mut lines = BufReader::new(stdout).lines();
    drain_stderr(&mut child);

    println!("Linking WhatsApp. This connects your own account as a second device.");
    println!();

    while let Some(line) = lines.next_line().await? {
        match read_event(&line) {
            Some(Event::Qr { ascii, .. }) => {
                println!("{ascii}");
                println!("  On your phone: WhatsApp ▸ Settings ▸ Linked devices ▸ Link a device");
                println!("  The code refreshes every 20 seconds; a new one will appear.");
                println!();
            }
            Some(Event::Ready { who }) => {
                let _ = child.kill().await;
                return Ok(who);
            }
            Some(Event::Fatal { reason }) => {
                let _ = child.kill().await;
                anyhow::bail!("{reason}");
            }
            Some(Event::Notice { text }) => println!("  {text}"),
            _ => {}
        }
    }
    anyhow::bail!("the bridge stopped before the device was linked")
}

/// Sign the linked device out on WhatsApp's side, then forget it here.
///
/// Connects with the saved credentials only to say goodbye, so the device
/// disappears from the phone's Linked devices list instead of lingering there
/// as a session nobody holds. If that cannot be done — no network, or the
/// session is already dead — the local credentials are still removed and the
/// error says what is left to do by hand.
pub async fn unlink(node: &str, bridge: &Path, state: &Path) -> anyhow::Result<()> {
    let signed_out = if state.join("creds.json").is_file() && is_installed(bridge) {
        tokio::time::timeout(std::time::Duration::from_secs(25), say_goodbye(node, bridge, state))
            .await
            .unwrap_or_else(|_| Err(anyhow::anyhow!("WhatsApp did not answer in time")))
    } else {
        Ok(())
    };
    logout(state)?;
    signed_out.map_err(|e| {
        anyhow::anyhow!(
            "forgot the device here, but could not sign it out on WhatsApp ({e}). \
             Remove it on your phone: WhatsApp ▸ Linked devices"
        )
    })
}

async fn say_goodbye(node: &str, bridge: &Path, state: &Path) -> anyhow::Result<()> {
    let mut child = spawn(node, bridge, state, false, false)?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let mut stdin = child.stdin.take().expect("stdin was piped");
    drain_stderr(&mut child);
    let mut lines = BufReader::new(stdout).lines();
    while let Some(line) = lines.next_line().await? {
        match read_event(&line) {
            Some(Event::Ready { .. }) => {
                stdin.write_all(b"{\"type\":\"logout\"}\n").await?;
                stdin.flush().await?;
                // The bridge exits once WhatsApp has confirmed.
                let _ = child.wait().await;
                return Ok(());
            }
            // Asking for a QR code means there was no session left to end.
            Some(Event::Qr { .. }) => {
                let _ = child.kill().await;
                return Ok(());
            }
            Some(Event::Fatal { reason }) => {
                let _ = child.kill().await;
                anyhow::bail!("{reason}");
            }
            _ => {}
        }
    }
    anyhow::bail!("the bridge stopped before it could sign out")
}

/// Forget the credentials on this machine.
pub fn logout(state: &Path) -> anyhow::Result<()> {
    if state.exists() {
        std::fs::remove_dir_all(state)?;
    }
    Ok(())
}

/// Run the channel until the command side is dropped or the bridge dies.
pub async fn run(
    node: String,
    bridge: PathBuf,
    state: PathBuf,
    self_chat: bool,
    tx: Sender<Inbound>,
    mut rx: UnboundedReceiver<Command>,
) -> anyhow::Result<()> {
    if !is_installed(&bridge) {
        anyhow::bail!(
            "the WhatsApp bridge is not installed. Run `ozgent gateway whatsapp`, which installs it."
        );
    }
    let mut child = spawn(&node, &bridge, &state, false, self_chat)?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let mut stdin = child.stdin.take().expect("stdin was piped");
    drain_stderr(&mut child);

    let reader = tokio::spawn({
        let tx = tx.clone();
        async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Some(event) = read_event(&line) else { continue };
                let inbound = match event {
                    Event::Qr { data, .. } => Inbound::Qr { data },
                    Event::Ready { who } => Inbound::Ready { who },
                    Event::Notice { text } => Inbound::Notice { text },
                    Event::Fatal { reason } => Inbound::Failed { reason },
                    Event::Message(msg) => Inbound::Message(Box::new(msg)),
                };
                if tx.send(inbound).await.is_err() {
                    break;
                }
            }
        }
    });

    // Only the last chunk keeps the token: it is the one a revision rewrites.
    while let Some(command) = rx.recv().await {
        let lines: Vec<serde_json::Value> = match command {
            Command::Typing { chat } => {
                vec![serde_json::json!({ "type": "typing", "chat": chat })]
            }
            Command::Post { chat, token, markdown } => {
                let text = render(&markdown, Flavour::WhatsApp);
                split(&text, WHATSAPP_LIMIT)
                    .into_iter()
                    .map(|chunk| {
                        serde_json::json!({
                            "type": "post", "chat": chat, "token": token, "text": chunk,
                        })
                    })
                    .collect()
            }
            Command::Revise { token, markdown, .. } => {
                let text = render(&markdown, Flavour::WhatsApp);
                // Past one message's worth there is nothing useful to edit —
                // the finished reply is posted whole when the turn ends.
                if text.chars().count() > WHATSAPP_LIMIT {
                    continue;
                }
                vec![serde_json::json!({ "type": "revise", "token": token, "text": text })]
            }
            Command::Ask { chat, token, question } => {
                let text = render(&question_text(&question), Flavour::WhatsApp);
                vec![serde_json::json!({
                    "type": "post", "chat": chat, "token": token, "text": text,
                })]
            }
            Command::Settle { token, markdown, .. } => {
                let text = render(&markdown, Flavour::WhatsApp);
                vec![serde_json::json!({ "type": "settle", "token": token, "text": text })]
            }
            Command::Logout => vec![serde_json::json!({ "type": "logout" })],
        };

        for line in lines {
            let mut bytes = serde_json::to_vec(&line)?;
            bytes.push(b'\n');
            if stdin.write_all(&bytes).await.is_err() {
                break;
            }
        }
        let _ = stdin.flush().await;
    }

    reader.abort();
    let _ = child.kill().await;
    Ok(())
}

/// Forward the bridge's own logging to ozgent's, so a Node stack trace ends up
/// somewhere it can be read rather than on a pipe nobody drains — which would
/// eventually block the bridge.
fn drain_stderr(child: &mut Child) {
    let Some(stderr) = child.stderr.take() else { return };
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!("whatsapp bridge: {line}");
        }
    });
}

/// What the bridge says.
#[derive(Debug)]
enum Event {
    Qr { ascii: String, data: String },
    Ready { who: String },
    Notice { text: String },
    Message(Msg),
    Fatal { reason: String },
}

/// Parse one line of the bridge's output.
///
/// A line that does not parse is dropped rather than fatal: a dependency that
/// writes to stdout despite the bridge's guard would otherwise take the whole
/// channel down, and the next line is very likely fine.
fn read_event(line: &str) -> Option<Event> {
    let value: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    let text_at = |key: &str| value.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string();

    match value.get("type").and_then(|v| v.as_str())? {
        "qr" => Some(Event::Qr { ascii: text_at("ascii"), data: text_at("data") }),
        "ready" => Some(Event::Ready { who: text_at("who") }),
        "notice" => Some(Event::Notice { text: text_at("text") }),
        "fatal" => Some(Event::Fatal { reason: text_at("reason") }),
        "message" => {
            let chat = text_at("chat");
            if chat.is_empty() {
                return None;
            }
            let jid = text_at("jid");
            let images = value
                .get("image")
                .and_then(read_image)
                .map(|i| vec![i])
                .unwrap_or_default();
            let text = text_at("text");
            if text.trim().is_empty() && images.is_empty() {
                return None;
            }
            Some(Event::Message(Msg {
                chat,
                sender_id: text_at("sender"),
                // The JID as well as the number, so an allowlist written
                // either way matches.
                handle: (!jid.is_empty()).then_some(jid),
                name: {
                    let n = text_at("name");
                    if n.is_empty() { "someone".to_string() } else { n }
                },
                text,
                images,
                group: value.get("group").and_then(|v| v.as_bool()).unwrap_or(false),
                own: value.get("own").and_then(|v| v.as_bool()).unwrap_or(false),
            }))
        }
        _ => None,
    }
}

fn read_image(image: &serde_json::Value) -> Option<ImageSource> {
    let data = image.get("data")?.as_str()?;
    let mime = image.get("mime").and_then(|v| v.as_str()).unwrap_or("image/jpeg");
    match ozgent_core::chat::b64::decode(data) {
        Ok(bytes) => Some(ImageSource::Bytes { bytes, mime: Some(mime.to_string()) }),
        Err(e) => {
            tracing::warn!("whatsapp: undecodable image: {e}");
            None
        }
    }
}

/// A permission question, written for a channel with no buttons.
fn question_text(q: &Question) -> String {
    let what = match q.effect {
        ozgent_core::permission::Effect::Write => "wants to change something",
        ozgent_core::permission::Effect::Execute => "wants to run a program",
        ozgent_core::permission::Effect::Read => "wants to look something up",
        _ => "wants to do something it has not described",
    };
    let mut out = format!("**{}** {what}.", q.tool);
    if !q.detail.trim().is_empty() {
        out.push_str(&format!("\n\n```\n{}\n```", q.detail.trim()));
    }
    out.push_str("\n\nReply with a number:\n");
    for (i, (word, _)) in ANSWERS.iter().enumerate() {
        out.push_str(&format!("{}. {}\n", i + 1, describe(word)));
    }
    out.trim_end().to_string()
}

fn describe(word: &str) -> &'static str {
    match word {
        "yes" => "yes, this once",
        "session" => "yes, for the rest of this session",
        "always" => "always, don't ask again",
        _ => "no",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bridge_never_answers_itself() {
        // The filter lives in JavaScript because the bridge does, and it is
        // the one piece where a mistake is unbounded rather than untidy: in
        // the self-chat every message is `fromMe`, so a wrong test means
        // ozgent answers its own answer forever on a real account. Run here so
        // `cargo test` covers it; skipped when node is missing, since the
        // bridge is optional.
        let paths = ozgent_core::Paths::with_root("/nowhere/ozgent");
        let bridge = locate(None, &paths).expect("the repository's bridge");
        let run = std::process::Command::new("node")
            .arg("--test")
            .arg("filter.test.mjs")
            .current_dir(&bridge)
            .output();

        let Ok(out) = run else {
            eprintln!("skipping: node is not installed");
            return;
        };
        assert!(
            out.status.success(),
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    #[test]
    fn the_self_chat_is_reported_so_the_gateway_can_admit_it() {
        let line = r#"{"type":"message","chat":"1555@s.whatsapp.net","sender":"1555","jid":"1555@s.whatsapp.net","name":"me","text":"remind me to call","own":true}"#;
        let Some(Event::Message(m)) = read_event(line) else { panic!("not a message") };
        assert!(m.own);
    }

    #[test]
    fn a_message_without_the_flag_is_not_treated_as_your_own() {
        // `own` skips the allowlist entirely, so its absence must never be
        // read as true — an older bridge, or a field that failed to parse,
        // must fail closed.
        for line in [
            r#"{"type":"message","chat":"c","sender":"1","name":"A","text":"hi"}"#,
            r#"{"type":"message","chat":"c","sender":"1","name":"A","text":"hi","own":null}"#,
            r#"{"type":"message","chat":"c","sender":"1","name":"A","text":"hi","own":"yes"}"#,
        ] {
            let Some(Event::Message(m)) = read_event(line) else { panic!("not a message") };
            assert!(!m.own, "{line}");
        }
    }

    #[test]
    fn a_message_carries_both_names_the_sender_goes_by() {
        // An operator writes down a phone number or a JID; an allowlist that
        // matched only the other would look broken rather than mistyped.
        let line = r#"{"type":"message","chat":"1555@s.whatsapp.net","sender":"1555","jid":"1555@s.whatsapp.net","name":"Ada","text":"hello","group":false}"#;
        let Some(Event::Message(m)) = read_event(line) else { panic!("not a message") };
        assert_eq!(m.sender_id, "1555");
        assert_eq!(m.identities(), vec!["1555", "1555@s.whatsapp.net"]);
        assert_eq!(m.name, "Ada");
        assert!(!m.group);
    }

    #[test]
    fn an_image_arrives_as_bytes_the_model_can_see() {
        let data = ozgent_core::chat::b64::encode(b"\xff\xd8\xffnot-really-a-jpeg");
        let line = format!(
            r#"{{"type":"message","chat":"c","sender":"1","name":"A","text":"what is this?","image":{{"data":"{data}","mime":"image/png"}}}}"#
        );
        let Some(Event::Message(m)) = read_event(&line) else { panic!("not a message") };
        match &m.images[..] {
            [ImageSource::Bytes { bytes, mime }] => {
                assert_eq!(bytes, b"\xff\xd8\xffnot-really-a-jpeg");
                assert_eq!(mime.as_deref(), Some("image/png"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_message_with_neither_text_nor_image_is_not_a_turn() {
        // A sticker, a location, a reaction. Answering it would send the model
        // an empty question.
        let line = r#"{"type":"message","chat":"c","sender":"1","name":"A","text":"  "}"#;
        assert!(matches!(read_event(line), None));
    }

    #[test]
    fn a_line_that_is_not_the_protocol_is_dropped_rather_than_fatal() {
        // Some dependency writing to stdout would otherwise take the channel
        // down; the next line is very likely fine.
        for line in ["", "not json", "{}", r#"{"type":"unheard-of"}"#, "null"] {
            assert!(read_event(line).is_none(), "{line:?}");
        }
    }

    #[test]
    fn the_bridges_own_events_map_onto_the_channel_vocabulary() {
        assert!(matches!(read_event(r#"{"type":"qr","ascii":"block"}"#), Some(Event::Qr { .. })));
        assert!(matches!(read_event(r#"{"type":"ready","who":"+1"}"#), Some(Event::Ready { .. })));
        assert!(matches!(read_event(r#"{"type":"fatal","reason":"x"}"#), Some(Event::Fatal { .. })));
    }

    #[test]
    fn a_question_offers_numbers_because_there_are_no_buttons() {
        let q = Question {
            id: "c1".into(),
            tool: "write_file".into(),
            effect: ozgent_core::permission::Effect::Write,
            detail: "poem.txt".into(),
        };
        let text = question_text(&q);
        for n in 1..=4 {
            assert!(text.contains(&format!("{n}.")), "missing option {n} in {text}");
        }
        // And each number is one the typed-answer reader accepts.
        for (i, (_, choice)) in ANSWERS.iter().enumerate() {
            assert_eq!(crate::chat::read_choice(&(i + 1).to_string()), Some(*choice));
        }
    }

    #[test]
    fn npm_is_looked_for_beside_the_node_that_was_named() {
        // A version manager puts node somewhere that is not on PATH; the npm
        // next to it is the one that matches.
        assert_eq!(which_npm("/opt/node/bin/node"), "/opt/node/bin/npm");
        assert_eq!(which_npm("node"), "npm");
    }

    #[test]
    fn the_bridge_is_found_beside_the_executable() {
        // Under `cargo test` the executable is in `target/…`, so this walks up
        // to the repository's own copy — the same walk that finds the
        // installed layout beside a real binary.
        let paths = ozgent_core::Paths::with_root("/nowhere/ozgent");
        let found = locate(None, &paths).expect("the repository's bridge");
        assert!(found.join("index.mjs").is_file(), "{}", found.display());
    }

    #[test]
    fn a_configured_bridge_that_is_wrong_is_reported_rather_than_ignored() {
        // Falling back would leave the setting looking like it had been
        // honoured while a different bridge ran.
        let paths = ozgent_core::Paths::with_root("/nowhere/ozgent");
        let err = locate(Some(Path::new("/nowhere/else")), &paths).unwrap_err().to_string();
        assert!(err.contains("/nowhere/else"), "{err}");
    }

    #[test]
    fn the_repository_bridge_is_the_one_this_crate_speaks_to() {
        // The protocol is split across two languages, so the one thing that
        // can silently drift is the set of message types. Both halves are
        // checked against each other here rather than trusted.
        let paths = ozgent_core::Paths::with_root("/nowhere/ozgent");
        let bridge = locate(None, &paths).expect("the repository's bridge");
        let source = std::fs::read_to_string(bridge.join("index.mjs")).unwrap();
        for kind in ["'qr'", "'ready'", "'notice'", "'fatal'", "'message'"] {
            assert!(source.contains(kind), "the bridge never emits {kind}");
        }
        for command in ["case 'typing'", "case 'post'", "case 'revise'", "case 'settle'"] {
            assert!(source.contains(command), "the bridge does not handle {command}");
        }
    }
}
