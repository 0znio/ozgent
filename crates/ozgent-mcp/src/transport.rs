//! Getting JSON-RPC to a server and the answer back.
//!
//! Two ways, and they have almost nothing in common beyond the messages they
//! carry. A stdio server is a child process this machine owns; an HTTP server
//! is somewhere else entirely, reached over a connection that may be reused,
//! may return a stream, and may hand back a session id that every later
//! request has to quote.

use std::collections::{HashMap, VecDeque};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, oneshot};

use crate::protocol::{self, Failure};

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("could not start {program}: {source}. Is it installed?")]
    Spawn { program: String, source: std::io::Error },
    #[error("{0}")]
    Io(String),
    #[error("the server did not answer within {0:?}")]
    Timeout(Duration),
    #[error("the server stopped")]
    Gone,
    #[error("{0}")]
    Refused(#[from] Failure),
}

/// A live connection to one server.
pub enum Link {
    Stdio(Box<StdioLink>),
    Http(Box<HttpLink>),
}

impl Link {
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, TransportError> {
        match self {
            Self::Stdio(l) => l.request(method, params, timeout).await,
            Self::Http(l) => l.request(method, params, timeout).await,
        }
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<(), TransportError> {
        match self {
            Self::Stdio(l) => l.notify(method, params).await,
            Self::Http(l) => l.notify(method, params).await,
        }
    }

    pub async fn close(&self) {
        if let Self::Stdio(l) = self {
            l.close().await;
        }
    }
}

// ------------------------------------------------------------------- stdio

type Waiting = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;

/// The last lines a server wrote to stderr: what is wanted first when it will
/// not start, and shown beside it on the admin page.
pub type Log = Arc<std::sync::Mutex<VecDeque<String>>>;

/// Lines of stderr kept per server.
const LOG_LINES: usize = 40;

/// A child process speaking JSON-RPC over its own stdin and stdout.
pub struct StdioLink {
    child: Mutex<Child>,
    /// Taken on close: dropping it is how a stdio server is told to stop.
    stdin: Mutex<Option<ChildStdin>>,
    waiting: Waiting,
    /// Set once its stdout has ended. Every request after that fails at once
    /// instead of waiting out its timeout for a reply that cannot come.
    gone: Arc<AtomicBool>,
    next_id: AtomicU64,
}

impl StdioLink {
    /// Start `command` with exactly `env` as its environment: nothing of
    /// ozgent's own is passed on unless the caller put it there.
    pub fn start(
        command: &str,
        args: &[String],
        env: &std::collections::BTreeMap<String, String>,
        cwd: Option<&std::path::Path>,
        label: &str,
        log: Log,
    ) -> Result<Self, TransportError> {
        let mut process = Command::new(command);
        process
            .args(args)
            .env_clear()
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // A server left running after ozgent exits would hold whatever it
            // had open — a database, a port — with nothing to stop it.
            .kill_on_drop(true);
        if let Some(dir) = cwd {
            process.current_dir(dir);
        }

        let mut child = process.spawn().map_err(|e| TransportError::Spawn {
            program: command.to_string(),
            source: e,
        })?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        let waiting: Waiting = Arc::new(Mutex::new(HashMap::new()));
        let gone = Arc::new(AtomicBool::new(false));
        tokio::spawn(read_replies(stdout, Arc::clone(&waiting), Arc::clone(&gone)));
        // Drained, not discarded: an unread pipe eventually blocks the child,
        // and a server's own diagnostics are the first thing wanted when it
        // will not start.
        let label = label.to_string();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "ozgent::mcp", "{label}: {line}");
                let mut kept = log.lock().unwrap_or_else(|e| e.into_inner());
                if kept.len() == LOG_LINES {
                    kept.pop_front();
                }
                kept.push_back(line.chars().take(400).collect());
            }
        });

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(Some(stdin)),
            waiting,
            gone,
            next_id: AtomicU64::new(1),
        })
    }

    async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, TransportError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        {
            let mut waiting = self.waiting.lock().await;
            // Checked under the lock the reader clears the map under, so a
            // request can never be left waiting on a reader that has finished.
            if self.gone.load(Ordering::SeqCst) {
                return Err(TransportError::Gone);
            }
            waiting.insert(id, tx);
        }

        if let Err(e) = self.write(&protocol::request(id, method, params)).await {
            self.waiting.lock().await.remove(&id);
            return Err(e);
        }

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(response)) => protocol::result_of(&response).map_err(TransportError::from),
            // The reader dropped the sender: the process is gone.
            Ok(Err(_)) => Err(TransportError::Gone),
            Err(_) => {
                // Forgotten, or a late reply would sit in the map forever.
                self.waiting.lock().await.remove(&id);
                Err(TransportError::Timeout(timeout))
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), TransportError> {
        self.write(&protocol::notification(method, params)).await
    }

    async fn write(&self, message: &Value) -> Result<(), TransportError> {
        let mut line = serde_json::to_vec(message).map_err(|e| TransportError::Io(e.to_string()))?;
        line.push(b'\n');
        let mut stdin = self.stdin.lock().await;
        let Some(stdin) = stdin.as_mut() else { return Err(TransportError::Gone) };
        stdin.write_all(&line).await.map_err(|e| TransportError::Io(e.to_string()))?;
        stdin.flush().await.map_err(|e| TransportError::Io(e.to_string()))
    }

    async fn close(&self) {
        let mut child = self.child.lock().await;
        // MCP has no shutdown message: closing stdin is how a stdio server is
        // told to stop, and the kill is for one that does not. Taken, not
        // just locked: dropping a guard closes nothing.
        drop(self.stdin.lock().await.take());
        if tokio::time::timeout(Duration::from_secs(3), child.wait()).await.is_err() {
            let _ = child.kill().await;
        }
    }
}

/// Match replies to the requests waiting for them.
async fn read_replies(stdout: tokio::process::ChildStdout, waiting: Waiting, gone: Arc<AtomicBool>) {
    read_until_closed(stdout, &waiting).await;
    // The process has ended, or closed its stdout. Dropping every waiting
    // sender fails those requests now; left in place, a server that died on
    // start-up held its handshake for the full two minutes.
    let mut waiting = waiting.lock().await;
    gone.store(true, Ordering::SeqCst);
    waiting.clear();
}

async fn read_until_closed(stdout: tokio::process::ChildStdout, waiting: &Waiting) {
    let mut lines = BufReader::new(stdout).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(message) = serde_json::from_str::<Value>(line.trim()) else {
            // Servers do print things to stdout despite the protocol living
            // there. Skipping the line beats killing the connection.
            tracing::debug!(target: "ozgent::mcp", "not protocol: {line}");
            continue;
        };
        // A request *from* the server. ozgent claims no capabilities that
        // would cause one, so there is nothing to answer and nothing waiting.
        let Some(id) = message.get("id").and_then(|i| i.as_u64()) else { continue };
        if let Some(tx) = waiting.lock().await.remove(&id) {
            let _ = tx.send(message);
        }
    }
}

// -------------------------------------------------------------------- http

/// A server reached over HTTP, using the Streamable HTTP transport.
pub struct HttpLink {
    url: String,
    http: reqwest::Client,
    headers: std::collections::BTreeMap<String, String>,
    /// Handed out by the server on `initialize`, and required on every request
    /// afterwards by a server that issued one.
    session: Mutex<Option<String>>,
    /// The revision the server settled on, quoted back per the specification.
    version: Mutex<String>,
    next_id: AtomicU64,
}

impl HttpLink {
    pub fn new(url: &str, headers: std::collections::BTreeMap<String, String>) -> Self {
        Self {
            url: url.to_string(),
            http: reqwest::Client::builder()
                .build()
                .unwrap_or_default(),
            headers,
            session: Mutex::new(None),
            version: Mutex::new(protocol::PROTOCOL_VERSION.to_string()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Remember the session and version an `initialize` established.
    pub async fn adopt(&self, session: Option<String>, version: &str) {
        if let Some(id) = session {
            *self.session.lock().await = Some(id);
        }
        *self.version.lock().await = version.to_string();
    }

    async fn send(&self, body: &Value, timeout: Duration) -> Result<reqwest::Response, TransportError> {
        let mut request = self
            .http
            .post(&self.url)
            .timeout(timeout)
            // Either is allowed back: a lone JSON response, or a stream.
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", self.version.lock().await.clone())
            .json(body);
        if let Some(session) = self.session.lock().await.as_deref() {
            request = request.header("mcp-session-id", session);
        }
        for (name, value) in &self.headers {
            request = request.header(name.as_str(), value.as_str());
        }
        request.send().await.map_err(|e| {
            if e.is_timeout() { TransportError::Timeout(timeout) } else { TransportError::Io(e.to_string()) }
        })
    }

    async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, TransportError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let response = self.send(&protocol::request(id, method, params), timeout).await?;

        // A session that expired: the server says so with 404, and the fix is
        // a new handshake rather than a retry of this call.
        if response.status() == reqwest::StatusCode::NOT_FOUND
            && self.session.lock().await.is_some()
        {
            return Err(TransportError::Gone);
        }
        if let Some(session) = response.headers().get("mcp-session-id") {
            if let Ok(value) = session.to_str() {
                *self.session.lock().await = Some(value.to_string());
            }
        }
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(TransportError::Io(format!("{status}: {}", body.trim())));
        }

        let stream = response
            .headers()
            .get("content-type")
            .and_then(|c| c.to_str().ok())
            .is_some_and(|c| c.starts_with("text/event-stream"));
        let body = response.text().await.map_err(|e| TransportError::Io(e.to_string()))?;

        let message = if stream {
            find_reply(&body, id).ok_or_else(|| {
                TransportError::Io("the stream ended without answering".into())
            })?
        } else {
            serde_json::from_str::<Value>(&body)
                .map_err(|e| TransportError::Io(format!("unreadable reply: {e}")))?
        };
        protocol::result_of(&message).map_err(TransportError::from)
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), TransportError> {
        // A notification is answered with 202 and no body; nothing is waiting
        // on it, so a failure here is logged rather than raised.
        let _ = self
            .send(&protocol::notification(method, params), Duration::from_secs(10))
            .await?;
        Ok(())
    }

    pub async fn session(&self) -> Option<String> {
        self.session.lock().await.clone()
    }
}

/// Find the reply to `id` in an SSE body.
///
/// A stream may carry notifications and server-to-client requests alongside
/// the answer, so the id is what identifies it — not its position.
pub fn find_reply(body: &str, id: u64) -> Option<Value> {
    for event in body.split("\n\n") {
        for line in event.lines() {
            let Some(data) = line.strip_prefix("data:") else { continue };
            let Ok(value) = serde_json::from_str::<Value>(data.trim()) else { continue };
            if value.get("id").and_then(|i| i.as_u64()) == Some(id) {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_answer_is_found_by_its_id_and_not_its_position() {
        // A stream may carry notifications before the reply; taking the first
        // message would return progress instead of the result.
        let body = concat!(
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n",
            "\n",
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":4,\"result\":{\"ok\":true}}\n",
            "\n",
        );
        let found = find_reply(body, 4).expect("the reply");
        assert_eq!(found["result"]["ok"], true);
    }

    #[test]
    fn a_reply_to_someone_elses_request_is_not_taken() {
        let body = "data: {\"jsonrpc\":\"2.0\",\"id\":9,\"result\":{}}\n\n";
        assert_eq!(find_reply(body, 4), None);
    }

    #[test]
    fn a_stream_with_nothing_readable_in_it_yields_nothing() {
        for body in ["", "event: ping\n\n", "data: not json\n\n", ": a comment\n\n"] {
            assert_eq!(find_reply(body, 1), None, "{body:?}");
        }
    }

    #[test]
    fn a_multi_line_event_is_still_read() {
        let body = "id: 7\nevent: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":1}\n\n";
        assert_eq!(find_reply(body, 1).unwrap()["result"], json!(1));
    }

    #[tokio::test]
    async fn a_session_id_is_remembered_and_quoted_back() {
        let link = HttpLink::new("https://x.test/mcp", Default::default());
        assert_eq!(link.session().await, None);
        link.adopt(Some("abc123".into()), "2024-11-05").await;
        assert_eq!(link.session().await.as_deref(), Some("abc123"));
        assert_eq!(*link.version.lock().await, "2024-11-05");
    }

    #[tokio::test]
    async fn a_server_that_issues_no_session_is_not_given_one() {
        let link = HttpLink::new("https://x.test/mcp", Default::default());
        link.adopt(None, protocol::PROTOCOL_VERSION).await;
        assert_eq!(link.session().await, None);
    }

    #[tokio::test]
    async fn a_program_that_does_not_exist_says_so_usefully() {
        let started = StdioLink::start(
            "definitely-not-a-real-program-xyz",
            &[],
            &Default::default(),
            None,
            "test",
            Log::default(),
        );
        let Err(err) = started else { panic!("that program should not exist") };
        let text = err.to_string();
        assert!(text.contains("definitely-not-a-real-program-xyz"), "{text}");
        assert!(text.contains("installed"), "{text}");
    }

    #[tokio::test]
    async fn a_server_that_dies_at_once_fails_at_once_with_its_last_words() {
        // It used to hold the handshake for the full two minutes.
        let log = Log::default();
        let env: std::collections::BTreeMap<String, String> =
            std::env::vars().filter(|(k, _)| k == "PATH").collect();
        let link = StdioLink::start(
            "sh",
            &["-c".into(), "echo 'npm error 404 Not Found' >&2; exit 1".into()],
            &env,
            None,
            "test",
            log.clone(),
        )
        .unwrap();
        let started = std::time::Instant::now();
        let result = link.request("initialize", serde_json::json!({}), Duration::from_secs(60)).await;
        assert!(matches!(result, Err(TransportError::Gone)), "{result:?}");
        assert!(started.elapsed() < Duration::from_secs(5), "took {:?}", started.elapsed());
        tokio::time::sleep(Duration::from_millis(200)).await;
        let kept: Vec<String> = log.lock().unwrap().iter().cloned().collect();
        assert_eq!(kept, ["npm error 404 Not Found"]);
        link.close().await;
    }

    #[tokio::test]
    async fn only_the_given_environment_reaches_the_server() {
        // SAFETY: test-only; nothing else reads this variable.
        unsafe { std::env::set_var("OZGENT_TEST_LEAK_TOKEN", "must-not-leak") };
        let env = crate::server::server_env(&[("MINE".to_string(), "yes".to_string())].into());
        assert_eq!(env.get("MINE").map(String::as_str), Some("yes"));
        assert!(!env.contains_key("OZGENT_TEST_LEAK_TOKEN"));
        assert!(env.contains_key("PATH"));
        let log = Log::default();
        let link = StdioLink::start("sh", &["-c".into(), "env >&2".into()], &env, None, "test", log.clone()).unwrap();
        let _ = link.request("x", serde_json::json!({}), Duration::from_secs(5)).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let seen = log.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n");
        assert!(seen.contains("MINE=yes"), "{seen}");
        assert!(!seen.contains("must-not-leak"), "{seen}");
    }
}
