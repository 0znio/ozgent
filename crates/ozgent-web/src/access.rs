//! Who may reach this server, and as whom.
//!
//! Everything the port serves passes through here first — the chat page, its
//! API, the admin page, the scheduler and the OpenAI/Anthropic API — because
//! the port is one door and each of those used to trust whoever came through
//! it. Anyone who could reach it could read every conversation, change which
//! tools run without asking, point the tool host at another interpreter, and
//! so run anything. On `127.0.0.1` that was still true for any web page: a
//! page on a domain its owner re-points at 127.0.0.1 is, to the browser,
//! same-origin with this server.
//!
//! The checks, in order:
//!
//! 1. **The address** ([`GuardedListener`], then [`guard`]): the `[web.access]`
//!    deny list, and in `allowlist` mode the allow list; refused at `accept`,
//!    before a byte of HTTP is parsed. Connections per address are capped, and
//!    one that does not finish its headers in time, or goes quiet, is closed.
//! 2. **The rate**: requests per minute per address, and a lockout after
//!    repeated bad keys or tokens.
//! 3. **The name**: `Host` must be this machine, a bare IP address, or a name
//!    listed in `[web.access] hosts`. This is what defeats DNS rebinding.
//! 4. **The origin**: a browser request that changes something must come from
//!    this server's own pages.
//! 5. **The caller** ([`Caller`]): this machine's own page (the local token),
//!    a signed-in admin, or an API key and its scopes.
//!
//! Every response also carries headers that keep the pages out of frames, stop
//! content sniffing, and allow scripts from this server only.

use axum::extract::{ConnectInfo, Request};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{Html, IntoResponse, Response};
use ozgent_core::access::{AccessConfig, Grant};
use std::collections::HashMap;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use crate::state::State;

/// Header the local page and the terminal client send the local token in.
pub const TOKEN_HEADER: &str = "x-ozgent-token";
/// Cookie the same token travels in, for what a page loads without a header
/// of its own: images in a conversation, an export link.
const TOKEN_COOKIE: &str = "ozgent_local";

/// A connection must finish its first request's headers within this.
const HEADER_DEADLINE: Duration = Duration::from_secs(20);
/// And may then sit with nothing read or written for this long.
const IDLE: Duration = Duration::from_secs(180);
/// Connections open at once, from everyone. Past this, only this machine.
const MAX_CONNECTIONS: u32 = 1024;

// ------------------------------------------------------------ the token

/// Create the local token, owner-readable only, replacing any old one.
pub fn issue_token(path: &std::path::Path) -> std::io::Result<String> {
    let token = ozgent_core::secret::random_token(32);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
        }
    }
    let partial = path.with_extension("partial");
    {
        use std::io::Write;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&partial)?;
        file.write_all(token.as_bytes())?;
    }
    std::fs::rename(&partial, path)?;
    Ok(token)
}

/// The local token as a client on this machine reads it, if a server wrote one.
pub fn read_token(paths: &ozgent_core::Paths) -> Option<String> {
    std::fs::read_to_string(paths.local_token_file()).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

// ------------------------------------------------------------ the shield

/// Counters the checks keep between requests. In memory: a restart forgets
/// them, which only ever errs towards letting someone back in.
#[derive(Default)]
pub struct Shield {
    rates: Mutex<HashMap<IpAddr, Bucket>>,
    failures: Mutex<HashMap<IpAddr, Failures>>,
    connections: Arc<Mutex<Connections>>,
    /// The single key `ozgent serve --api-key` was started with, if any.
    legacy_key: std::sync::OnceLock<String>,
}

#[derive(Default)]
struct Connections {
    by_address: HashMap<IpAddr, u32>,
    total: u32,
}

struct Bucket {
    tokens: f64,
    at: Instant,
}

struct Failures {
    count: u32,
    first: Instant,
    locked_until: Option<Instant>,
}

impl Shield {
    /// Accept `key` as a key with every scope, as `ozgent serve --api-key` has
    /// always meant.
    pub fn set_legacy_key(&self, key: String) {
        let _ = self.legacy_key.set(key);
    }

    /// Take one request from `ip`'s allowance. `Err` carries how long to wait.
    fn spend(&self, ip: IpAddr, per_minute: u32) -> Result<(), Duration> {
        if per_minute == 0 {
            return Ok(());
        }
        let rate = per_minute as f64 / 60.0;
        // A burst of a tenth of a minute's allowance: a page load fetches a
        // dozen things at once and should not be throttled for it.
        let burst = (per_minute as f64 / 10.0).max(20.0);
        let mut rates = self.rates.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if rates.len() > 50_000 {
            rates.retain(|_, b| now.duration_since(b.at) < Duration::from_secs(120));
        }
        let b = rates.entry(ip).or_insert(Bucket { tokens: burst, at: now });
        b.tokens = (b.tokens + now.duration_since(b.at).as_secs_f64() * rate).min(burst);
        b.at = now;
        if b.tokens >= 1.0 {
            b.tokens -= 1.0;
            Ok(())
        } else {
            Err(Duration::from_secs_f64(((1.0 - b.tokens) / rate).max(1.0)))
        }
    }

    /// How long `ip` is still locked out, if it is.
    fn locked(&self, ip: IpAddr) -> Option<Duration> {
        let mut failures = self.failures.lock().unwrap_or_else(|e| e.into_inner());
        let f = failures.get_mut(&ip)?;
        let until = f.locked_until?;
        let now = Instant::now();
        if until <= now {
            failures.remove(&ip);
            return None;
        }
        Some(until - now)
    }

    /// Count a bad key or token from `ip`.
    fn failed(&self, ip: IpAddr, access: &AccessConfig) {
        if access.max_auth_failures == 0 {
            return;
        }
        let window = Duration::from_secs(access.lockout_minutes.max(1) as u64 * 60);
        let now = Instant::now();
        let mut failures = self.failures.lock().unwrap_or_else(|e| e.into_inner());
        if failures.len() > 50_000 {
            failures.retain(|_, f| now.duration_since(f.first) < window);
        }
        let f = failures.entry(ip).or_insert(Failures { count: 0, first: now, locked_until: None });
        if now.duration_since(f.first) >= window {
            *f = Failures { count: 0, first: now, locked_until: None };
        }
        f.count += 1;
        if f.count >= access.max_auth_failures {
            f.locked_until = Some(now + window);
            tracing::warn!("{ip}: {} bad keys or tokens; refused for {} minutes", f.count, access.lockout_minutes.max(1));
        }
    }
}

// ------------------------------------------------------------ the listener

/// A TCP listener that refuses addresses before HTTP begins.
///
/// Refusing in a handler still lets a refused address hold connections open
/// and cost a parse per request. Here a refused connection is closed on
/// accept, and a permitted one is counted, timed and closed if it stalls —
/// axum gives hyper no timer, so hyper's own header timeout never fires.
pub struct GuardedListener {
    inner: tokio::net::TcpListener,
    state: State,
}

impl GuardedListener {
    pub fn new(inner: tokio::net::TcpListener, state: State) -> Self {
        Self { inner, state }
    }
}

impl axum::serve::Listener for GuardedListener {
    type Io = Guarded;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let (stream, addr) = match self.inner.accept().await {
                Ok(pair) => pair,
                Err(e) => {
                    // Out of descriptors, usually. Accepting again at once
                    // would spin; a moment's pause lets connections close.
                    tracing::debug!("accept failed: {e}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            };
            let ip = ozgent_core::access::canonical(addr.ip());
            let access = access_config(&self.state);
            // A trusted proxy is judged by the address it forwards, per
            // request, in `guard`; the proxy itself is not refused here.
            if !access.trusts_proxy(ip) {
                if let Err(why) = access.admits(ip) {
                    tracing::info!("{ip}: connection refused ({why:?})");
                    drop(stream);
                    continue;
                }
            }
            let connections = self.state.shield.connections.clone();
            {
                let mut c = connections.lock().unwrap_or_else(|e| e.into_inner());
                let mine = c.by_address.get(&ip).copied().unwrap_or(0);
                let per = access.max_connections_per_address;
                let over_mine = per > 0 && mine >= per && !access.trusts_proxy(ip);
                let over_all = c.total >= MAX_CONNECTIONS && !ip.is_loopback();
                if over_mine || over_all {
                    tracing::info!("{ip}: connection refused ({mine} open)");
                    drop(stream);
                    continue;
                }
                *c.by_address.entry(ip).or_insert(0) += 1;
                c.total += 1;
            }
            let _ = stream.set_nodelay(true);
            let io = Guarded {
                stream,
                ip,
                connections,
                started: false,
                seen: [0; 3],
                deadline: Box::pin(tokio::time::sleep(HEADER_DEADLINE)),
            };
            return (io, addr);
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

/// A connection with a deadline and a place in the count.
pub struct Guarded {
    stream: tokio::net::TcpStream,
    ip: IpAddr,
    connections: Arc<Mutex<Connections>>,
    /// Whether the first request's headers have arrived.
    started: bool,
    /// The last three bytes read, to find `\r\n\r\n` across reads.
    seen: [u8; 3],
    deadline: Pin<Box<tokio::time::Sleep>>,
}

impl Guarded {
    fn touch(&mut self) {
        if self.started {
            self.deadline.as_mut().reset(tokio::time::Instant::now() + IDLE);
        }
    }

    /// Whether the deadline has passed; registers for a wake-up if not.
    fn expired(&mut self, cx: &mut Context<'_>) -> bool {
        self.deadline.as_mut().poll(cx).is_ready()
    }
}

impl Drop for Guarded {
    fn drop(&mut self) {
        let mut c = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = c.by_address.get_mut(&self.ip) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                c.by_address.remove(&self.ip);
            }
        }
        c.total = c.total.saturating_sub(1);
    }
}

impl AsyncRead for Guarded {
    fn poll_read(mut self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        match Pin::new(&mut self.stream).poll_read(cx, buf) {
            Poll::Ready(Ok(())) => {
                let read = &buf.filled()[before..];
                if !self.started && !read.is_empty() {
                    // The previous read's last three bytes in front, so a
                    // blank line split across two reads is still found.
                    let mut window = Vec::with_capacity(3 + read.len());
                    window.extend_from_slice(&self.seen);
                    window.extend_from_slice(read);
                    if window.windows(4).any(|w| w == b"\r\n\r\n") {
                        self.started = true;
                    }
                    let n = window.len();
                    self.seen.copy_from_slice(&window[n - 3..]);
                }
                if !read.is_empty() {
                    self.touch();
                }
                Poll::Ready(Ok(()))
            }
            Poll::Pending => {
                if self.expired(cx) {
                    let why = if self.started { "idle" } else { "headers not finished" };
                    tracing::debug!("{}: closing a connection ({why})", self.ip);
                    return Poll::Ready(Err(std::io::Error::new(std::io::ErrorKind::TimedOut, why)));
                }
                Poll::Pending
            }
            other => other,
        }
    }
}

impl AsyncWrite for Guarded {
    fn poll_write(mut self: Pin<&mut Self>, cx: &mut Context<'_>, data: &[u8]) -> Poll<std::io::Result<usize>> {
        let out = Pin::new(&mut self.stream).poll_write(cx, data);
        if matches!(out, Poll::Ready(Ok(n)) if n > 0) {
            self.touch();
        }
        out
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        let out = Pin::new(&mut self.stream).poll_write_vectored(cx, bufs);
        if matches!(out, Poll::Ready(Ok(n)) if n > 0) {
            self.touch();
        }
        out
    }
    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }
}

// ------------------------------------------------------------ requests

/// Who a request is from, as far as the rest of the server is concerned.
#[derive(Debug, Clone)]
pub enum Caller {
    /// This machine's own user: the local page, the terminal client, a local
    /// program calling `/v1` without a key, or a signed-in admin elsewhere.
    /// Governed by the tool policy exactly as before.
    Owner,
    /// A program with an API key, limited to that key's scopes.
    Key(Grant),
}

impl Caller {
    /// The grant to hand a worker request: `None` for the owner.
    pub fn grant(&self) -> Option<Grant> {
        match self {
            Caller::Owner => None,
            Caller::Key(g) => Some(g.clone()),
        }
    }
}

/// The address a request is really from, after any trusted proxy.
#[derive(Debug, Clone, Copy)]
pub struct ClientIp(pub IpAddr);

fn access_config(state: &State) -> AccessConfig {
    state.config.lock().unwrap_or_else(|e| e.into_inner()).web.access.clone()
}

fn peer_of(request: &Request) -> IpAddr {
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| ozgent_core::access::canonical(c.0.ip()))
        .unwrap_or(IpAddr::from([127, 0, 0, 1]))
}

/// Whether a request names another hop: a proxy saying who it is for.
fn forwarded(headers: &HeaderMap) -> bool {
    ["x-forwarded-for", "forwarded", "x-real-ip"].iter().any(|h| headers.contains_key(*h))
}

/// The client address: the peer, or — only when the peer is a trusted proxy —
/// the right-most address in `X-Forwarded-For` that is not itself trusted.
fn client_ip(access: &AccessConfig, peer: IpAddr, headers: &HeaderMap) -> IpAddr {
    if !access.trusts_proxy(peer) {
        return peer;
    }
    let chain: Vec<IpAddr> = headers
        .get_all("x-forwarded-for")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .filter_map(|s| s.trim().parse::<IpAddr>().ok())
        .map(ozgent_core::access::canonical)
        .collect();
    chain.into_iter().rev().find(|ip| !access.trusts_proxy(*ip)).unwrap_or(peer)
}

/// Whether the request came straight from this machine: a loopback peer that
/// is not relaying for someone else. A reverse proxy on this machine makes
/// every visitor a loopback peer, which is why forwarding headers disqualify.
fn from_this_machine(peer: IpAddr, headers: &HeaderMap) -> bool {
    peer.is_loopback() && !forwarded(headers)
}

/// The host a request names, without its port.
fn host_of(headers: &HeaderMap) -> Option<String> {
    let raw = headers.get(header::HOST)?.to_str().ok()?.trim().to_ascii_lowercase();
    let host = if let Some(rest) = raw.strip_prefix('[') {
        rest.split(']').next()?.to_string()
    } else {
        raw.rsplit_once(':').filter(|(_, p)| p.chars().all(|c| c.is_ascii_digit())).map_or(raw.clone(), |(h, _)| h.to_string())
    };
    Some(host)
}

/// Whether this server answers to `host`.
///
/// This machine's names and any bare address are fine: an attacker's page can
/// only reach this server through a *name* it controls, never through an
/// address literal, so refusing unknown names is the whole defence against
/// DNS rebinding. Names a reverse proxy or a LAN uses are listed by the
/// operator.
fn host_allowed(host: &str, access: &AccessConfig) -> bool {
    if host == "localhost" || host.ends_with(".localhost") || host.parse::<IpAddr>().is_ok() {
        return true;
    }
    access.hosts.iter().any(|h| h.trim().eq_ignore_ascii_case(host))
}

/// Whether a browser sent this request from somewhere other than this
/// server's own pages. Programs send no `Origin`, so they pass.
fn cross_site(headers: &HeaderMap, host: &str) -> bool {
    if headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()) == Some("cross-site") {
        return true;
    }
    let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) else { return false };
    if origin == "null" {
        return true;
    }
    let after_scheme = origin.split_once("://").map_or(origin, |(_, rest)| rest);
    let origin_host = if let Some(rest) = after_scheme.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        after_scheme.rsplit_once(':').map_or(after_scheme, |(h, _)| h)
    };
    !origin_host.eq_ignore_ascii_case(host)
}

fn presented_key(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()))
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
}

fn token_from(headers: &HeaderMap) -> Option<String> {
    if let Some(t) = headers.get(TOKEN_HEADER).and_then(|v| v.to_str().ok()) {
        return Some(t.trim().to_string());
    }
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|kv| kv.trim().split_once('='))
        .find(|(k, _)| *k == TOKEN_COOKIE)
        .map(|(_, v)| v.to_string())
}

/// Which part of the server a path belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Area {
    /// The pages and their assets. Nothing in them is private.
    Public,
    /// Signing in to the admin page; admin.rs counts its own failures.
    AdminLogin,
    /// Behind the admin password, which admin.rs checks.
    Admin,
    /// The OpenAI- and Anthropic-compatible API.
    Api,
    /// Everything else: the chat page's own API, attachments, history.
    Owner,
}

fn area(method: &Method, path: &str) -> Area {
    match path {
        "/" | "/new" | "/chat" | "/app.css" | "/app.js" | "/theme.js" | "/favicon.ico" | "/logo.png"
        | "/admin" | "/admin.js" | "/scheduler" | "/scheduler.js" | "/health" => Area::Public,
        "/api/admin/login" | "/api/admin/session" => Area::AdminLogin,
        _ if path.starts_with("/api/admin/") || path.starts_with("/api/hub/") => Area::Admin,
        _ if path.starts_with("/api/models/") && method == Method::DELETE && !path.ends_with("/options") => Area::Admin,
        _ if path.starts_with("/v1/") || path == "/v1" => Area::Api,
        _ => Area::Owner,
    }
}

fn refuse(status: StatusCode, message: &str, api: bool) -> Response {
    let body = if api {
        // OpenAI's envelope, which Anthropic clients also read well enough.
        serde_json::json!({ "error": { "message": message, "type": "invalid_request_error", "code": serde_json::Value::Null } })
    } else {
        serde_json::json!({ "error": message })
    };
    (status, axum::Json(body)).into_response()
}

/// The checks, for every request.
pub async fn guard(axum::extract::State(state): axum::extract::State<State>, mut request: Request, next: Next) -> Response {
    let access = access_config(&state);
    let peer = peer_of(&request);
    let headers = request.headers().clone();
    let ip = client_ip(&access, peer, &headers);
    let path = request.uri().path().to_string();
    let method = request.method().clone();
    let area = area(&method, &path);
    let api = area == Area::Api;

    // 1. The address, now that a proxy's client is known.
    if let Err(why) = access.admits(ip) {
        tracing::info!("{ip}: refused {path} ({why:?})");
        return refuse(StatusCode::FORBIDDEN, "this address is not allowed to use this server", api);
    }

    // 2. The rate, and any lockout.
    if let Some(left) = state.shield.locked(ip) {
        let mut r = refuse(StatusCode::TOO_MANY_REQUESTS, "too many bad keys or tokens; try again later", api);
        r.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(left.as_secs().max(1)));
        return r;
    }
    let local = from_this_machine(peer, &headers);
    if !local {
        if let Err(wait) = state.shield.spend(ip, access.requests_per_minute) {
            let mut r = refuse(StatusCode::TOO_MANY_REQUESTS, "too many requests; slow down", api);
            r.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(wait.as_secs().max(1)));
            return r;
        }
    }

    // 3. The name.
    let Some(host) = host_of(&headers) else {
        return refuse(StatusCode::BAD_REQUEST, "no Host header", api);
    };
    if !host_allowed(&host, &access) {
        tracing::info!("{ip}: refused {path} for unknown host {host:?}");
        return refuse(
            StatusCode::MISDIRECTED_REQUEST,
            "this server does not answer to that name; add it to [web.access] hosts on the admin page",
            api,
        );
    }

    // 4. The origin, for anything that changes something.
    let changes = !matches!(method, Method::GET | Method::HEAD | Method::OPTIONS);
    if changes && cross_site(&headers, &host) {
        tracing::info!("{ip}: refused a cross-site {method} {path}");
        return refuse(StatusCode::FORBIDDEN, "cross-site requests are not accepted", api);
    }
    request.extensions_mut().insert(ClientIp(ip));

    // 5. The caller.
    let token_ok = token_from(&headers)
        .zip(state.local_token.get())
        .is_some_and(|(given, expected)| ozgent_core::secret::same(&given, expected));
    let admin = crate::admin::signed_in(&state, &headers);
    // A signed-in admin's changes must carry the header a cross-site form
    // cannot add; the local token already arrives in one.
    let admin_owner = admin && (!changes || headers.contains_key("x-ozgent-admin"));
    // A header token proves this machine's page or client; the cookie copy
    // only counts for reading, since a cookie rides along on requests a page
    // did not mean to make.
    let header_token = headers.contains_key(TOKEN_HEADER);
    let owner = (token_ok && (header_token || !changes)) || admin_owner;

    let caller = match area {
        Area::Public | Area::AdminLogin | Area::Admin => None,
        Area::Owner => {
            if !owner {
                if token_from(&headers).is_some() && !token_ok {
                    state.shield.failed(ip, &access);
                }
                return refuse(StatusCode::UNAUTHORIZED, "sign in at /admin, or open this page on the machine it runs on", false);
            }
            Some(Caller::Owner)
        }
        Area::Api => match presented_key(&headers) {
            Some(key) => {
                let digest = ozgent_core::secret::key_digest(&key);
                if let Some(entry) = access.key_by_digest(&digest) {
                    Some(Caller::Key(Grant::from_scopes(&entry.id, &entry.scopes)))
                } else if state.shield.legacy_key.get().is_some_and(|k| ozgent_core::secret::same(k, &key)) {
                    Some(Caller::Key(Grant::full("serve --api-key")))
                } else if state.local_token.get().is_some_and(|t| ozgent_core::secret::same(t, &key)) {
                    // The terminal client may present the local token as its key.
                    Some(Caller::Owner)
                } else {
                    state.shield.failed(ip, &access);
                    tracing::info!("{ip}: bad API key for {path}");
                    return refuse(StatusCode::UNAUTHORIZED, "missing or invalid API key", true);
                }
            }
            None if owner => Some(Caller::Owner),
            // A program on this machine, as it always could — unless the
            // operator has said every API caller needs a key, or the server
            // was started with one.
            None if local && access.local_api_open && state.shield.legacy_key.get().is_none() => Some(Caller::Owner),
            None => return refuse(StatusCode::UNAUTHORIZED, "missing or invalid API key", true),
        },
    };
    if let Some(Caller::Key(g)) = &caller {
        if !g.inference && !g.tools && !g.agents {
            return refuse(StatusCode::FORBIDDEN, "this key has no scopes", true);
        }
        if !g.inference && path != "/v1/models" && !path.starts_with("/v1/models/") && path != "/v1/agents" {
            // Tools and agents both need the model; a key without inference
            // scope may only list what exists.
            return refuse(StatusCode::FORBIDDEN, "this key may not use the models", true);
        }
    }
    if let Some(c) = caller {
        request.extensions_mut().insert(c);
    }

    let mut response = next.run(request).await;
    secure_headers(response.headers_mut());
    response
}

/// Headers every response carries.
fn secure_headers(h: &mut HeaderMap) {
    h.insert("x-content-type-options", HeaderValue::from_static("nosniff"));
    h.insert("x-frame-options", HeaderValue::from_static("DENY"));
    h.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    h.insert("cross-origin-opener-policy", HeaderValue::from_static("same-origin"));
    h.insert("cross-origin-resource-policy", HeaderValue::from_static("same-origin"));
    let html = h
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|t| t.starts_with("text/html"));
    if html {
        // Scripts from this server only, no inline script or handler: the
        // page renders model output, and a model steered by a page it read is
        // exactly who would try to put a script in it. Inline styles stay —
        // they cannot run anything.
        h.insert(
            "content-security-policy",
            HeaderValue::from_static(
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; \
                 img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; \
                 object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
            ),
        );
    }
}

/// Serve a page, giving this machine's own browser the local token.
///
/// Only a browser on this machine, asking by a name this server answers to,
/// gets it. Anywhere else the page arrives without one and signs in through
/// `/admin` instead.
pub fn page(state: &State, request: &Request, html: &str) -> Response {
    let peer = peer_of(request);
    let headers = request.headers();
    let access = access_config(state);
    let local = from_this_machine(peer, headers)
        && host_of(headers).is_some_and(|h| host_allowed(&h, &access));
    let token = state.local_token.get().filter(|_| local);
    let body = match token {
        Some(t) => html.replacen(
            r#"<meta name="ozgent-token" content="">"#,
            &format!(r#"<meta name="ozgent-token" content="{t}">"#),
            1,
        ),
        None => html.to_string(),
    };
    let mut response = Html(body).into_response();
    let h = response.headers_mut();
    // The page itself must not be cached: it carries a token that changes
    // with every server start.
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    if let Some(t) = token {
        if let Ok(v) = HeaderValue::from_str(&format!("{TOKEN_COOKIE}={t}; Path=/; HttpOnly; SameSite=Strict")) {
            h.insert(header::SET_COOKIE, v);
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(), v.parse().unwrap());
        }
        h
    }

    #[test]
    fn a_rebinding_name_is_refused_and_addresses_are_not() {
        let a = AccessConfig::default();
        assert!(host_allowed("localhost", &a));
        assert!(host_allowed("127.0.0.1", &a));
        assert!(host_allowed("::1", &a));
        assert!(host_allowed("192.168.1.20", &a));
        assert!(!host_allowed("attacker.example", &a));
        let named = AccessConfig { hosts: vec!["box.lan".into()], ..Default::default() };
        assert!(host_allowed("box.lan", &named));
        assert_eq!(host_of(&headers(&[("host", "localhost:7333")])).as_deref(), Some("localhost"));
        assert_eq!(host_of(&headers(&[("host", "[::1]:7333")])).as_deref(), Some("::1"));
        assert_eq!(host_of(&headers(&[("host", "Evil.Example")])).as_deref(), Some("evil.example"));
    }

    #[test]
    fn cross_site_is_judged_by_origin_host() {
        assert!(!cross_site(&headers(&[]), "localhost"));
        assert!(!cross_site(&headers(&[("origin", "http://localhost:7333")]), "localhost"));
        assert!(cross_site(&headers(&[("origin", "https://evil.example")]), "localhost"));
        assert!(cross_site(&headers(&[("origin", "null")]), "localhost"));
        assert!(!cross_site(&headers(&[("origin", "http://[::1]:7333")]), "::1"));
        assert!(cross_site(&headers(&[("sec-fetch-site", "cross-site")]), "localhost"));
    }

    #[test]
    fn forwarded_for_is_believed_only_from_a_trusted_proxy() {
        let proxy: IpAddr = "10.0.0.2".parse().unwrap();
        let h = headers(&[("x-forwarded-for", "6.6.6.6, 203.0.113.5")]);
        let untrusted = AccessConfig::default();
        assert_eq!(client_ip(&untrusted, proxy, &h), proxy);
        let trusted = AccessConfig { trusted_proxies: vec!["10.0.0.0/8".into()], ..Default::default() };
        // Right-most untrusted: the client's own claim to be 6.6.6.6 is not believed.
        assert_eq!(client_ip(&trusted, proxy, &h), "203.0.113.5".parse::<IpAddr>().unwrap());
        // A local reverse proxy does not make its visitors local.
        assert!(!from_this_machine("127.0.0.1".parse().unwrap(), &h));
        assert!(from_this_machine("127.0.0.1".parse().unwrap(), &headers(&[])));
    }

    #[test]
    fn the_areas_are_where_they_should_be() {
        assert_eq!(area(&Method::GET, "/"), Area::Public);
        assert_eq!(area(&Method::PUT, "/api/settings"), Area::Owner);
        assert_eq!(area(&Method::GET, "/media/abc.png"), Area::Owner);
        assert_eq!(area(&Method::POST, "/v1/chat/completions"), Area::Api);
        assert_eq!(area(&Method::POST, "/api/admin/login"), Area::AdminLogin);
        assert_eq!(area(&Method::PUT, "/api/admin/access"), Area::Admin);
        assert_eq!(area(&Method::DELETE, "/api/models/x:y"), Area::Admin);
        assert_eq!(area(&Method::PUT, "/api/models/x:y/options"), Area::Owner);
    }

    #[test]
    fn the_rate_allows_a_burst_then_throttles() {
        let s = Shield::default();
        let ip: IpAddr = "203.0.113.1".parse().unwrap();
        let allowed = (0..100).filter(|_| s.spend(ip, 60).is_ok()).count();
        assert_eq!(allowed, 20, "burst of 20 at 60/min");
        assert!(s.spend("203.0.113.2".parse().unwrap(), 60).is_ok(), "per address");
        assert!(s.spend(ip, 0).is_ok(), "0 is unlimited");
    }

    #[test]
    fn repeated_bad_keys_lock_the_address_out() {
        let s = Shield::default();
        let ip: IpAddr = "203.0.113.1".parse().unwrap();
        let a = AccessConfig { max_auth_failures: 3, ..Default::default() };
        s.failed(ip, &a);
        s.failed(ip, &a);
        assert!(s.locked(ip).is_none());
        s.failed(ip, &a);
        assert!(s.locked(ip).is_some());
        assert!(s.locked("203.0.113.9".parse().unwrap()).is_none());
    }

    #[test]
    fn the_token_file_is_private() {
        let dir = std::env::temp_dir().join(format!("ozgent-token-{}", std::process::id()));
        let path = dir.join("run").join("local-token");
        let t = issue_token(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), t);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(std::fs::metadata(path.parent().unwrap()).unwrap().permissions().mode() & 0o777, 0o700);
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
