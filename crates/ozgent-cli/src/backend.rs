//! Talking to the ozgent daemon.
//!
//! The terminal used to load its own model. That meant a copy of the weights
//! in VRAM per open terminal, a CUDA context beside it, ten to forty seconds
//! before the first prompt — and, worse than any of that, a *second*
//! implementation of a turn. `chat.rs` had its own prompt assembly, its own
//! tool-call loop, its own permission handling, all of it a parallel version
//! of what the server does. Two implementations of the same thing drift, and
//! the drift is always discovered by a user.
//!
//! So the terminal is a client now, exactly as the browser already was. There
//! is one place inference happens.
//!
//! # There is always a backend
//!
//! Not "use the daemon if one is running, otherwise load a model here" —
//! that keeps both paths alive and gains nothing. If nothing is listening,
//! one is started. After that there is a single code path, and the terminal
//! never touches llama.cpp at all.
//!
//! The daemon that gets started is a real one: it outlives the terminal that
//! started it, drops its model when idle, and answers the channels and the
//! scheduler while it is up. Leaving it running is the point — the next
//! `ozgent chat` opens instantly instead of paying for a model load.

use anyhow::{Context, Result};
use std::time::Duration;

/// How long to wait for a daemon we started to come up.
///
/// Generous because starting one does real work — the Python tool worker, the
/// database, the channels — on a cold page cache. It does *not* include
/// loading a model, which happens later and is reported as progress.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// How often to check whether it is up yet.
const POLL: Duration = Duration::from_millis(150);

/// A connection to the daemon.
#[derive(Clone)]
pub struct Backend {
    base: String,
    http: reqwest::Client,
}

impl Backend {
    /// The address to talk to, honouring `$OZGENT_HOST`.
    ///
    /// One variable rather than a flag on every command: a person who moved
    /// their daemon has moved it for every terminal they open.
    pub fn address() -> String {
        std::env::var("OZGENT_HOST")
            .ok()
            .map(|h| {
                if h.starts_with("http://") || h.starts_with("https://") {
                    h
                } else {
                    format!("http://{h}")
                }
            })
            .unwrap_or_else(|| "http://127.0.0.1:7333".into())
    }

    fn at(base: &str) -> Self {
        Self {
            base: base.trim_end_matches('/').to_string(),
            // No global timeout: a turn is a long-lived stream, and a model
            // load inside it can legitimately take a minute. Connect timeouts
            // are set per request where they belong.
            http: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(5))
                .build()
                .expect("building an HTTP client"),
        }
    }

    /// Connect to a running daemon, or `None` if nothing answers.
    pub async fn connect() -> Option<Self> {
        let backend = Self::at(&Self::address());
        backend.alive().await.then_some(backend)
    }

    /// Connect, starting a daemon if nothing is listening.
    ///
    /// `announce` is called with a line to print when a daemon has to be
    /// started, so the caller decides how it looks — the full-screen terminal
    /// draws it differently from a one-shot `ozgent run`.
    pub async fn connect_or_start(announce: impl FnOnce(&str)) -> Result<Self> {
        if let Some(backend) = Self::connect().await {
            return Ok(backend);
        }
        let address = Self::address();
        if !is_local(&address) {
            // A remote daemon is somebody else's machine. Starting one here
            // would silently answer a different address from the one asked
            // for, which is worse than saying it is not there.
            anyhow::bail!(
                "nothing is answering at {address}, and it is not this machine, \
                 so there is nothing to start here"
            );
        }
        announce(&format!("no backend at {address} — starting one"));
        spawn_daemon(&address)?;

        let deadline = std::time::Instant::now() + START_TIMEOUT;
        let backend = Self::at(&address);
        while std::time::Instant::now() < deadline {
            if backend.alive().await {
                return Ok(backend);
            }
            tokio::time::sleep(POLL).await;
        }
        anyhow::bail!(
            "started a daemon but it did not come up within {}s.\n\
             Run `ozgent daemon` in another terminal to see why.",
            START_TIMEOUT.as_secs()
        )
    }

    /// Whether something is answering.
    pub async fn alive(&self) -> bool {
        self.get("/health").await.is_ok()
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    // ------------------------------------------------------------ requests

    pub async fn get(&self, path: &str) -> Result<serde_json::Value> {
        let res = self
            .http
            .get(format!("{}{path}", self.base))
            .timeout(Duration::from_secs(30))
            .send()
            .await?;
        read(res).await
    }

    pub async fn post(&self, path: &str, body: serde_json::Value) -> Result<serde_json::Value> {
        let res = self
            .http
            .post(format!("{}{path}", self.base))
            .timeout(Duration::from_secs(30))
            .json(&body)
            .send()
            .await?;
        read(res).await
    }

    pub async fn put(&self, path: &str, body: serde_json::Value) -> Result<serde_json::Value> {
        let res = self
            .http
            .put(format!("{}{path}", self.base))
            .timeout(Duration::from_secs(30))
            .json(&body)
            .send()
            .await?;
        read(res).await
    }

    pub async fn delete(&self, path: &str) -> Result<serde_json::Value> {
        let res = self
            .http
            .delete(format!("{}{path}", self.base))
            .timeout(Duration::from_secs(30))
            .send()
            .await?;
        read(res).await
    }

    /// Start a turn. Yields every event until `done` or `error`.
    ///
    /// The events are the server's own, unchanged: the browser renders this
    /// exact stream, and giving the terminal a translated version of it would
    /// be the seam where the two surfaces start to differ again.
    pub async fn chat(&self, request: serde_json::Value) -> Result<Stream> {
        let res = self
            .http
            .post(format!("{}/api/chat", self.base))
            .json(&request)
            .send()
            .await
            .context("asking the daemon")?;
        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            anyhow::bail!("{status}: {}", message_in(&body));
        }
        Ok(Stream { res: Some(res), buffer: String::new(), done: false })
    }

    /// Answer a permission question the daemon asked.
    pub async fn decide(&self, id: &str, choice: &str) -> Result<()> {
        self.post(
            "/api/permissions/decide",
            serde_json::json!({ "id": id, "choice": choice }),
        )
        .await
        .map(|_| ())
    }
}

/// Read a JSON body, turning the server's own error text into the error.
async fn read(res: reqwest::Response) -> Result<serde_json::Value> {
    let status = res.status();
    let body = res.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("{}", message_in(&body));
    }
    if body.trim().is_empty() {
        return Ok(serde_json::Value::Null);
    }
    serde_json::from_str(&body).context("the daemon sent something that is not JSON")
}

/// The `error` field out of a JSON body, or the body itself.
///
/// Both shapes are in use — `{"error": "..."}` from the internal API and
/// `{"error": {"message": "..."}}` from the OpenAI-compatible one — and a
/// person reading a terminal wants the sentence either way.
fn message_in(body: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return body.trim().to_string();
    };
    let error = &value["error"];
    if let Some(text) = error.as_str() {
        return text.to_string();
    }
    if let Some(text) = error["message"].as_str() {
        return text.to_string();
    }
    body.trim().to_string()
}

/// A server-sent event stream, read one event at a time.
pub struct Stream {
    res: Option<reqwest::Response>,
    buffer: String,
    done: bool,
}

impl Stream {
    /// The next event, or `None` when the turn has ended.
    ///
    /// Events are separated by a blank line and each carries one `data:` line,
    /// which is what axum's SSE writes. Anything else in the stream — comments
    /// a proxy might inject to keep the connection alive — is skipped rather
    /// than treated as an event.
    pub async fn next(&mut self) -> Result<Option<serde_json::Value>> {
        loop {
            if let Some(event) = self.take() {
                return Ok(Some(event));
            }
            if self.done {
                return Ok(None);
            }
            let Some(res) = self.res.as_mut() else { return Ok(None) };
            match res.chunk().await? {
                Some(bytes) => self.buffer.push_str(&String::from_utf8_lossy(&bytes)),
                None => {
                    self.done = true;
                    // One more pass: the last event may have arrived without
                    // the trailing blank line that usually ends one.
                    if let Some(event) = self.take_rest() {
                        return Ok(Some(event));
                    }
                    return Ok(None);
                }
            }
        }
    }

    /// A complete event out of the buffer, if there is one.
    fn take(&mut self) -> Option<serde_json::Value> {
        while let Some(end) = self.buffer.find("\n\n") {
            let block: String = self.buffer.drain(..end + 2).collect();
            if let Some(event) = parse(&block) {
                return Some(event);
            }
        }
        None
    }

    fn take_rest(&mut self) -> Option<serde_json::Value> {
        let block = std::mem::take(&mut self.buffer);
        parse(&block)
    }
}

/// One SSE block as JSON, or `None` if it carries no data.
fn parse(block: &str) -> Option<serde_json::Value> {
    // A single event may spread its payload over several `data:` lines, which
    // are joined with newlines. Rare here, but it is the specification and
    // handling it is two lines.
    let data: Vec<&str> = block
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|rest| rest.strip_prefix(' ').unwrap_or(rest))
        .collect();
    if data.is_empty() {
        return None;
    }
    serde_json::from_str(&data.join("\n")).ok()
}

/// Whether an address is this machine, and so something we may start.
fn is_local(address: &str) -> bool {
    let host = address
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .split('/')
        .next()
        .unwrap_or("")
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or_else(|| address.trim_start_matches("http://"));
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]" | "0.0.0.0" | "")
}

/// The port out of an address, for the daemon we start.
fn port_of(address: &str) -> u16 {
    address
        .rsplit(':')
        .next()
        .and_then(|p| p.trim_end_matches('/').parse().ok())
        .unwrap_or(7333)
}

/// Start a daemon that outlives this process.
///
/// Detached on purpose. It is a real daemon — it answers the channels and runs
/// the scheduler — and killing it when one terminal closes would make those
/// depend on a window being open, which is the thing this whole arrangement
/// exists to stop.
fn spawn_daemon(address: &str) -> Result<()> {
    let exe = std::env::current_exe().context("cannot find where ozgent is installed")?;
    let mut command = std::process::Command::new(exe);
    command
        .arg("daemon")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port_of(address).to_string())
        // Its output belongs in its own log, not interleaved with the
        // terminal that happened to start it.
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    #[cfg(unix)]
    {
        // A new session, so Ctrl-C in this terminal does not reach it and
        // closing the terminal does not hang it up.
        use std::os::unix::process::CommandExt;
        unsafe {
            command.pre_exec(|| {
                libc_setsid();
                Ok(())
            });
        }
    }
    command.spawn().context("starting a daemon")?;
    Ok(())
}

#[cfg(unix)]
fn libc_setsid() {
    // SAFETY: setsid takes no arguments and only detaches this child from the
    // controlling terminal. It is called between fork and exec, where only
    // async-signal-safe calls are allowed, and setsid is one of them.
    unsafe {
        unsafe extern "C" {
            fn setsid() -> i32;
        }
        setsid();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_address_is_this_machine_on_the_usual_port() {
        // Checked without the variable set, which is how it usually runs.
        if std::env::var_os("OZGENT_HOST").is_none() {
            assert_eq!(Backend::address(), "http://127.0.0.1:7333");
        }
    }

    #[test]
    fn a_bare_host_and_port_gains_a_scheme() {
        // People type `OZGENT_HOST=box:7333`, not a URL.
        // SAFETY: single-threaded test, and the variable is removed after.
        unsafe { std::env::set_var("OZGENT_HOST", "box:7333") };
        assert_eq!(Backend::address(), "http://box:7333");
        unsafe { std::env::set_var("OZGENT_HOST", "https://box:7333") };
        assert_eq!(Backend::address(), "https://box:7333");
        unsafe { std::env::remove_var("OZGENT_HOST") };
    }

    #[test]
    fn only_this_machine_is_something_we_may_start() {
        // Starting a daemon here when the address is somebody else's machine
        // would answer a different address from the one asked for.
        for local in [
            "http://127.0.0.1:7333",
            "http://localhost:7333",
            "http://[::1]:7333",
            "http://0.0.0.0:7333",
        ] {
            assert!(is_local(local), "{local} is this machine");
        }
        for remote in ["http://192.168.1.10:7333", "http://box.local:7333", "https://example.com"] {
            assert!(!is_local(remote), "{remote} is not");
        }
    }

    #[test]
    fn the_port_is_taken_from_the_address() {
        assert_eq!(port_of("http://127.0.0.1:7333"), 7333);
        assert_eq!(port_of("http://127.0.0.1:9000"), 9000);
        // No port, or an unreadable one, falls back to the usual.
        assert_eq!(port_of("http://127.0.0.1"), 7333);
        assert_eq!(port_of(""), 7333);
    }

    // --------------------------------------------------------- the stream

    fn events(text: &str) -> Vec<serde_json::Value> {
        let mut stream = Stream { res: None, buffer: text.to_string(), done: true };
        let mut out = Vec::new();
        while let Some(event) = stream.take().or_else(|| stream.take_rest()) {
            out.push(event);
        }
        out
    }

    #[test]
    fn events_are_read_one_per_block() {
        let got = events("data: {\"type\":\"answer\",\"text\":\"hi\"}\n\ndata: {\"type\":\"done\"}\n\n");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0]["type"], "answer");
        assert_eq!(got[1]["type"], "done");
    }

    #[test]
    fn a_partial_event_is_not_read_until_it_is_complete() {
        // The failure this prevents: half a JSON object arriving in one TCP
        // chunk and being parsed as a broken event rather than waited on.
        let mut stream = Stream {
            res: None,
            buffer: "data: {\"type\":\"answ".to_string(),
            done: false,
        };
        assert!(stream.take().is_none(), "half an event is not an event");
        stream.buffer.push_str("er\",\"text\":\"hi\"}\n\n");
        let event = stream.take().expect("now it is complete");
        assert_eq!(event["text"], "hi");
    }

    #[test]
    fn the_last_event_is_read_even_without_a_trailing_blank_line() {
        // A stream that ends cleanly after its final event would otherwise
        // lose the `done` that tells the terminal the turn is over.
        let got = events("data: {\"type\":\"done\",\"generated\":4}");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["generated"], 4);
    }

    #[test]
    fn keep_alive_comments_are_skipped_rather_than_parsed() {
        let got = events(": keep-alive\n\ndata: {\"type\":\"ready\"}\n\n");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["type"], "ready");
    }

    #[test]
    fn an_event_split_over_several_data_lines_is_joined() {
        let got = events("data: {\"type\":\"answer\",\ndata: \"text\":\"hi\"}\n\n");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["text"], "hi");
    }

    #[test]
    fn text_with_blank_lines_in_it_survives_the_round_trip() {
        // A reply containing a paragraph break is JSON-escaped, so the blank
        // line inside it must not be mistaken for an event boundary.
        let text = serde_json::json!({ "type": "answer", "text": "one\n\ntwo" }).to_string();
        let got = events(&format!("data: {text}\n\n"));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["text"], "one\n\ntwo");
    }

    #[test]
    fn rubbish_in_the_stream_is_skipped_rather_than_ending_the_turn() {
        let got = events("data: not json\n\ndata: {\"type\":\"done\"}\n\n");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["type"], "done");
    }

    // ----------------------------------------------------------- messages

    #[test]
    fn an_error_is_reported_in_the_servers_own_words() {
        // Both shapes are in use, and a person reading a terminal wants the
        // sentence rather than the envelope.
        assert_eq!(message_in(r#"{"error":"no such model"}"#), "no such model");
        assert_eq!(message_in(r#"{"error":{"message":"bad request"}}"#), "bad request");
        assert_eq!(message_in("plain text"), "plain text");
        assert_eq!(message_in(""), "");
    }
}
