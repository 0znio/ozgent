//! `/admin`: the messaging gateway and model downloads, behind a password.
//!
//! The rest of the web interface is open to whoever can reach it — that was
//! the user's call, and it is printed at startup. These two things are not:
//! the gateway decides who in the world can reach this machine's tools, and
//! the model pages download and delete gigabytes. So they sit behind a
//! password that only someone with a shell on this machine can set
//! (`ozgent admin setup`), stored as an Argon2id hash, never in the clear.
//!
//! The basics, and why each is here:
//!
//! * **A session cookie, not the password, on every request.** `HttpOnly` so
//!   a script on the page cannot read it, `SameSite=Strict` so another site
//!   cannot ride it. Mutating requests also need a custom header, which a
//!   cross-origin form cannot send.
//! * **Sessions are bound to the hash they were issued under.** Changing or
//!   resetting the password signs every browser out, with nothing to sweep.
//! * **Wrong guesses are slowed and then locked out,** per address and in
//!   total. The way back from a lockout — or a forgotten password — is
//!   `ozgent admin reset` on the machine itself, which also clears it.
//!
//! The gateway runs in `ozgent-channels`, which depends on this crate rather
//! than the other way round, so it is reached through [`GatewayControl`].

use std::collections::HashMap;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, Path, Request, State as AxumState};
use axum::http::{HeaderMap, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use ozgent_core::channels::{self, Kind};
use ozgent_core::secret;
use serde::{Deserialize, Serialize};

use crate::state::State;

// ------------------------------------------------------------------ gateway

pub type Fut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What the admin page can ask of the running gateway.
pub trait GatewayControl: Send + Sync {
    /// What every channel is doing right now.
    fn view(&self) -> GatewayView;
    /// Bring the running channels in line with the configuration: start what
    /// is newly on, stop what is off, restart what has new credentials.
    fn apply(&self);
    /// Stop a channel and start it again, clearing a failure.
    fn restart(&self, kind: Kind);
    /// Replace the pairing code and return the new one.
    fn new_pairing_code(&self) -> String;
    /// Check a bot token with Telegram. The bot's @name when it works.
    fn check_telegram<'a>(&'a self, token: &'a str) -> Fut<'a, Result<String, String>>;
    /// Install the bridge if needed and start linking; the QR code appears
    /// in [`GatewayControl::view`].
    fn link_whatsapp(&self) -> Fut<'_, Result<(), String>>;
    /// Log the linked device out on WhatsApp's side and forget it here.
    fn unlink_whatsapp(&self) -> Fut<'_, Result<(), String>>;

    /// Send a message to a chat that nobody asked a question in.
    ///
    /// Every other message a channel sends is a reply, written while a turn is
    /// running and addressed to whoever spoke. A scheduled job has neither: it
    /// starts on a timer and has to reach a chat that may have been quiet for
    /// a week. Failing here is ordinary — the channel may be off, or another
    /// process may hold it — so the reason comes back to be recorded against
    /// the run rather than logged and lost.
    fn deliver(&self, kind: Kind, chat: &str, markdown: &str) -> Result<(), String>;
}

/// The gateway as a whole.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GatewayView {
    /// Whether this process answers the channels. Only one process may: two
    /// would fight over the Telegram token and the WhatsApp session.
    pub hosted: bool,
    /// Who does, when it is not this process.
    pub elsewhere: Option<String>,
    /// The code an unlisted person can send to be allowed in.
    pub pairing: Option<String>,
    pub telegram: Runtime,
    pub whatsapp: Runtime,
}

/// One channel, as it is running.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Runtime {
    pub phase: Phase,
    /// The account it is connected as: a bot's @name, a phone number.
    pub who: Option<String>,
    /// Why it failed, or what it is doing.
    pub detail: Option<String>,
    /// A WhatsApp linking code, raw; drawn as an image for the page.
    #[serde(skip)]
    pub qr: Option<String>,
    /// WhatsApp: credentials are saved, so it can start without a QR code.
    pub linked: bool,
    /// WhatsApp: whether the bridge's Node packages are installed.
    pub installed: Option<bool>,
    /// WhatsApp: seconds left to scan the code, while linking.
    pub link_left: Option<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    #[default]
    Off,
    Installing,
    Starting,
    Linking,
    Connected,
    Failed,
    /// Another ozgent process is answering this channel.
    Elsewhere,
}

// ------------------------------------------------------------------ sessions

/// How long a session lasts without being used.
const IDLE: Duration = Duration::from_secs(12 * 60 * 60);
/// Failed logins from one address before it is locked out.
const PER_ADDRESS: u32 = 5;
/// Failed logins from anywhere before logins stop altogether. Guards against
/// guesses spread over many addresses; `ozgent admin reset` clears it.
const IN_TOTAL: u32 = 30;
/// How far back failures are counted, and how long a lockout lasts.
const WINDOW: Duration = Duration::from_secs(15 * 60);
const COOKIE: &str = "ozgent_admin";

struct Session {
    /// The password hash this session was issued under.
    hash: String,
    last: Instant,
}

#[derive(Default)]
struct Failures {
    by_address: HashMap<IpAddr, Vec<Instant>>,
    all: Vec<Instant>,
    /// Locks in force, until when: per address, and for everyone.
    locked_until: HashMap<IpAddr, Instant>,
    all_locked_until: Option<Instant>,
}

impl Failures {
    fn prune(&mut self, now: Instant) {
        self.all.retain(|t| now.duration_since(*t) < WINDOW);
        for v in self.by_address.values_mut() {
            v.retain(|t| now.duration_since(*t) < WINDOW);
        }
        self.by_address.retain(|_, v| !v.is_empty());
    }

    /// How long a login from this address is still refused, if it is.
    fn locked(&mut self, ip: IpAddr) -> Option<Duration> {
        let now = Instant::now();
        self.prune(now);
        self.locked_until.retain(|_, t| *t > now);
        if self.all_locked_until.is_some_and(|t| t <= now) {
            self.all_locked_until = None;
        }
        let left = |t: Instant| t.saturating_duration_since(now);
        self.all_locked_until.map(left).or_else(|| self.locked_until.get(&ip).map(|t| left(*t)))
    }

    /// Count a miss. Reaching the limit locks for a full [`WINDOW`] from
    /// this miss — what the message then says.
    fn record(&mut self, ip: IpAddr) -> u32 {
        let now = Instant::now();
        self.all.push(now);
        let list = self.by_address.entry(ip).or_default();
        list.push(now);
        let n = list.len() as u32;
        if n >= PER_ADDRESS {
            self.locked_until.insert(ip, now + WINDOW);
        }
        if self.all.len() as u32 >= IN_TOTAL {
            self.all_locked_until = Some(now + WINDOW);
        }
        n
    }
}

/// Signed-in browsers and failed attempts. In memory: a restart signs
/// everyone out, which is the right direction to fail in.
#[derive(Default)]
pub struct Guard {
    sessions: Mutex<HashMap<String, Session>>,
    failures: Mutex<Failures>,
    /// The hash failures were counted against; a new password clears them.
    counted_for: Mutex<Option<String>>,
}

impl Guard {
    fn issue(&self, hash: &str) -> String {
        let token = secret::random_token(32);
        let mut sessions = self.sessions.lock().unwrap();
        let now = Instant::now();
        sessions.retain(|_, s| now.duration_since(s.last) < IDLE);
        sessions.insert(token.clone(), Session { hash: hash.to_string(), last: now });
        token
    }

    fn check(&self, token: &str, hash: &str) -> bool {
        let mut sessions = self.sessions.lock().unwrap();
        let Some(session) = sessions.get_mut(token) else { return false };
        if session.hash != hash || session.last.elapsed() >= IDLE {
            sessions.remove(token);
            return false;
        }
        session.last = Instant::now();
        true
    }

    fn end(&self, token: &str) {
        self.sessions.lock().unwrap().remove(token);
    }

    /// Forget failures counted against an older password.
    fn sync(&self, hash: &str) {
        let mut counted = self.counted_for.lock().unwrap();
        if counted.as_deref() != Some(hash) {
            *counted = Some(hash.to_string());
            *self.failures.lock().unwrap() = Failures::default();
        }
    }
}

fn current_hash(state: &State) -> Option<String> {
    state.config.lock().unwrap_or_else(|e| e.into_inner()).web.admin_hash().map(str::to_string)
}

fn cookie_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|kv| kv.trim().split_once('='))
        .find(|(k, _)| *k == COOKIE)
        .map(|(_, v)| v.to_string())
}

fn session_cookie(token: &str) -> String {
    format!("{COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={}", IDLE.as_secs())
}

fn clear_cookie() -> String {
    format!("{COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
}

/// The address a request came from; loopback when the server was built
/// without connection info (tests).
fn peer(request: &Request) -> IpAddr {
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|c| c.0.ip())
        .unwrap_or(IpAddr::from([127, 0, 0, 1]))
}

/// Whether a request carries a live session. Used by the gate and by the
/// page, which shows the sign-in form otherwise.
fn signed_in(state: &State, headers: &HeaderMap) -> bool {
    let (Some(hash), Some(token)) = (current_hash(state), cookie_token(headers)) else {
        return false;
    };
    state.admin.check(&token, &hash)
}

/// Everything behind the password goes through here.
async fn gate(AxumState(state): AxumState<State>, request: Request, next: Next) -> Response {
    if current_hash(&state).is_none() {
        return fail(StatusCode::FORBIDDEN, NOT_SET_UP);
    }
    if !signed_in(&state, request.headers()) {
        return fail(StatusCode::UNAUTHORIZED, "sign in at /admin first");
    }
    // A cross-site form can POST with the cookie attached but cannot add a
    // header; `SameSite=Strict` already stops it, and this does not depend
    // on the browser getting that right.
    if request.method() != Method::GET && request.headers().get("x-ozgent-admin").is_none() {
        return fail(StatusCode::FORBIDDEN, "missing the x-ozgent-admin header");
    }
    next.run(request).await
}

const NOT_SET_UP: &str =
    "the admin page has no password yet. On this machine, run: ozgent admin setup";

fn fail(status: StatusCode, message: impl std::fmt::Display) -> Response {
    (status, Json(serde_json::json!({ "error": message.to_string() }))).into_response()
}

// ------------------------------------------------------------------ routes

pub fn router(state: State) -> Router {
    let guarded = Router::new()
        .route("/api/admin/gateway", get(gateway_view).put(gateway_settings))
        .route("/api/admin/gateway/qr", get(qr_image))
        .route("/api/admin/gateway/pairing", post(new_pairing))
        .route("/api/admin/gateway/{kind}", put(channel_settings))
        .route("/api/admin/gateway/{kind}/restart", post(restart))
        .route("/api/admin/gateway/{kind}/signout", post(sign_out))
        .route("/api/admin/gateway/telegram/token", post(telegram_token))
        .route("/api/admin/gateway/whatsapp/link", post(whatsapp_link))
        .route("/api/admin/password", post(change_password))
        .route("/api/hub/search", get(crate::hub::search))
        .route("/api/hub/repo", get(crate::hub::repo))
        .route("/api/hub/pull", post(crate::hub::pull))
        .route("/api/hub/pulls", get(crate::hub::pulls))
        .route("/api/hub/pulls/{id}", delete(crate::hub::cancel))
        .route("/api/models/{model}", delete(crate::hub::remove))
        .route("/api/admin/models/upload", put(upload))
        .route("/api/admin/models/import", post(import_upload))
        .route("/api/admin/models/default", post(make_default))
        .route_layer(middleware::from_fn_with_state(state.clone(), gate));

    Router::new()
        .route("/admin", get(page))
        .route("/admin.js", get(script))
        .route("/api/admin/session", get(session).delete(logout))
        .route("/api/admin/login", post(login))
        .merge(guarded)
        .with_state(state)
}

async fn page() -> Html<&'static str> {
    Html(include_str!("../assets/admin.html"))
}

async fn script() -> impl IntoResponse {
    ([("content-type", "text/javascript; charset=utf-8")], include_str!("../assets/admin.js"))
}

#[derive(Serialize)]
struct SessionState {
    /// A password has been set with `ozgent admin setup`.
    configured: bool,
    signed_in: bool,
    /// The config holds something that is not a hash, such as a password
    /// typed in by hand. Refused, and said so.
    invalid: bool,
}

async fn session(AxumState(state): AxumState<State>, headers: HeaderMap) -> Json<SessionState> {
    let hash = current_hash(&state);
    let invalid = hash.as_deref().is_some_and(|h| !secret::is_hash(h));
    Json(SessionState {
        configured: hash.is_some() && !invalid,
        signed_in: !invalid && signed_in(&state, &headers),
        invalid,
    })
}

#[derive(Deserialize)]
struct Login {
    password: String,
}

async fn login(AxumState(state): AxumState<State>, request: Request) -> Response {
    let ip = peer(&request);
    let Some(hash) = current_hash(&state) else {
        return fail(StatusCode::FORBIDDEN, NOT_SET_UP);
    };
    if !secret::is_hash(&hash) {
        return fail(
            StatusCode::FORBIDDEN,
            "[web] admin_password_hash is not a password hash. Run: ozgent admin setup",
        );
    }
    state.admin.sync(&hash);
    if let Some(wait) = state.admin.failures.lock().unwrap().locked(ip) {
        return fail(
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "too many wrong passwords. Try again in {} min, or run `ozgent admin reset` on this machine",
                wait.as_secs().div_ceil(60).max(1)
            ),
        );
    }
    let body = match axum::body::to_bytes(request.into_body(), 64 * 1024).await {
        Ok(b) => b,
        Err(_) => return fail(StatusCode::BAD_REQUEST, "unreadable request"),
    };
    let Ok(Login { password }) = serde_json::from_slice::<Login>(&body) else {
        return fail(StatusCode::BAD_REQUEST, "expected {\"password\": \"…\"}");
    };

    let check = hash.clone();
    let ok = tokio::task::spawn_blocking(move || secret::verify_password(&password, &check))
        .await
        .unwrap_or(false);
    if !ok {
        let n = state.admin.failures.lock().unwrap().record(ip);
        tracing::warn!("admin: wrong password from {ip} ({n} in the last 15 min)");
        // A pause on every miss, growing with the count: cheap for a person
        // who mistyped, expensive for a script.
        tokio::time::sleep(Duration::from_millis(400 * u64::from(n.min(10)))).await;
        let left = PER_ADDRESS.saturating_sub(n);
        let message = if left == 0 {
            "wrong password. Locked for 15 minutes; `ozgent admin reset` on this machine sets a new one".to_string()
        } else {
            format!("wrong password ({left} more {} before a 15 minute lock)", if left == 1 { "try" } else { "tries" })
        };
        return fail(StatusCode::UNAUTHORIZED, message);
    }

    tracing::info!("admin: signed in from {ip}");
    let token = state.admin.issue(&hash);
    (
        [(header::SET_COOKIE, session_cookie(&token))],
        Json(serde_json::json!({ "ok": true })),
    )
        .into_response()
}

async fn logout(AxumState(state): AxumState<State>, headers: HeaderMap) -> Response {
    if let Some(token) = cookie_token(&headers) {
        state.admin.end(&token);
    }
    ([(header::SET_COOKIE, clear_cookie())], Json(serde_json::json!({ "ok": true }))).into_response()
}

#[derive(Deserialize)]
struct NewPassword {
    current: String,
    new: String,
}

/// Change the password from the page. Needs the current one: a session left
/// open on someone else's screen must not be enough to lock the owner out.
async fn change_password(
    AxumState(state): AxumState<State>,
    Json(body): Json<NewPassword>,
) -> Response {
    let Some(hash) = current_hash(&state) else { return fail(StatusCode::FORBIDDEN, NOT_SET_UP) };
    if let Err(e) = secret::check_strength(&body.new) {
        return fail(StatusCode::BAD_REQUEST, format!("new password: {e}"));
    }
    let check = hash.clone();
    let current = body.current.clone();
    let ok = tokio::task::spawn_blocking(move || secret::verify_password(&current, &check))
        .await
        .unwrap_or(false);
    if !ok {
        return fail(StatusCode::UNAUTHORIZED, "the current password is not right");
    }
    let new = body.new;
    let Ok(Ok(fresh)) = tokio::task::spawn_blocking(move || secret::hash_password(&new)).await else {
        return fail(StatusCode::INTERNAL_SERVER_ERROR, "could not hash the new password");
    };
    let saved = {
        let mut config = state.config.lock().unwrap_or_else(|e| e.into_inner());
        config.web.admin_password_hash = Some(fresh.clone());
        config.save(&state.paths)
    };
    if let Err(e) = saved {
        return fail(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    // Every other session was issued under the old hash and is now dead;
    // this browser gets a new one so the person changing it stays in.
    let token = state.admin.issue(&fresh);
    tracing::info!("admin: password changed from the admin page");
    ([(header::SET_COOKIE, session_cookie(&token))], Json(serde_json::json!({ "ok": true })))
        .into_response()
}

// ------------------------------------------------------------ gateway routes

fn gateway(state: &State) -> Result<&dyn GatewayControl, Response> {
    state.gateway.get().map(|g| g.as_ref()).ok_or_else(|| {
        fail(StatusCode::SERVICE_UNAVAILABLE, "the gateway is not running in this process")
    })
}

fn parse_kind(name: &str) -> Result<Kind, Response> {
    Kind::parse(name).ok_or_else(|| fail(StatusCode::NOT_FOUND, format!("no channel {name:?}")))
}

#[derive(Serialize)]
struct ToolRow {
    name: String,
    effect: String,
    /// What the global rules say: allow, ask or deny.
    rule: String,
}

/// Everything the gateway section of the page shows.
async fn gateway_view(AxumState(state): AxumState<State>) -> Response {
    let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let view = state.gateway.get().map(|g| g.view()).unwrap_or_default();

    let tools: Vec<ToolRow> = crate::worker::current_tools(&state.tools)
        .map(|t| {
            let mut rows: Vec<ToolRow> = t
                .host
                .tools()
                .iter()
                .filter(|s| !config.tools.disabled.contains(&s.name))
                .map(|s| ToolRow {
                    name: s.name.clone(),
                    effect: format!("{:?}", s.effect).to_lowercase(),
                    rule: config.permissions.rule_for(&s.name, s.effect).to_string(),
                })
                .collect();
            rows.sort_by(|a, b| a.name.cmp(&b.name));
            rows
        })
        .unwrap_or_default();

    // Chat models only: an embedding model cannot answer anyone.
    let models: Vec<String> = ozgent_core::installed(&state.paths)
        .into_iter()
        .filter(|m| !ozgent_llama::layout::is_embedding(&m.manifest.primary_weights(&m.dir)))
        .map(|m| m.manifest.alias.clone().unwrap_or_else(|| m.model.to_string()))
        .collect();

    let c = &config.channels;
    let token_from_env = std::env::var("OZGENT_TELEGRAM_TOKEN").is_ok_and(|t| !t.trim().is_empty());
    Json(serde_json::json!({
        "running_here": state.gateway.get().is_some(),
        "hosted": view.hosted,
        "elsewhere": view.elsewhere,
        "pairing": view.pairing,
        "enabled": c.enabled,
        "model": c.model,
        "default_model": config.default_model,
        "models": models,
        "tools": tools,
        "tools_enabled": config.tools.enabled,
        "telegram": {
            "runtime": view.telegram,
            "enabled": c.telegram.enabled,
            "token_set": !c.telegram.token.trim().is_empty() || token_from_env,
            "token_from_env": token_from_env,
            "allow": c.telegram.allow,
            "tools": c.telegram.tools,
            "approve": c.telegram.approve,
            "stream": c.telegram.stream,
        },
        "whatsapp": {
            "runtime": view.whatsapp,
            "enabled": c.whatsapp.enabled,
            "allow": c.whatsapp.allow,
            "tools": c.whatsapp.tools,
            "approve": c.whatsapp.approve,
            "stream": c.whatsapp.stream,
            "self_chat": c.whatsapp.self_chat,
            "groups": c.whatsapp.groups,
        },
    }))
    .into_response()
}

/// The WhatsApp linking code as an SVG, while there is one.
async fn qr_image(AxumState(state): AxumState<State>) -> Response {
    let Some(data) = state.gateway.get().and_then(|g| g.view().whatsapp.qr) else {
        return fail(StatusCode::NOT_FOUND, "no linking code right now");
    };
    match qrcode::QrCode::new(data.as_bytes()) {
        Ok(code) => {
            let svg = code
                .render::<qrcode::render::svg::Color>()
                .min_dimensions(264, 264)
                .quiet_zone(true)
                .dark_color(qrcode::render::svg::Color("#000000"))
                .light_color(qrcode::render::svg::Color("#ffffff"))
                .build();
            ([(header::CONTENT_TYPE, "image/svg+xml"), (header::CACHE_CONTROL, "no-store")], svg)
                .into_response()
        }
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(Deserialize)]
struct GatewaySettings {
    /// The model channels answer with. Empty to use the default model.
    model: Option<String>,
}

async fn gateway_settings(
    AxumState(state): AxumState<State>,
    Json(body): Json<GatewaySettings>,
) -> Response {
    let model = match body.model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        None => None,
        Some(name) => match ozgent_core::resolve(&state.paths, name) {
            Ok(_) => Some(name.to_string()),
            Err(e) => return fail(StatusCode::BAD_REQUEST, e),
        },
    };
    save(&state, |c| c.channels.model = model)
}

/// A channel's settings. Every field is optional; what is sent is changed.
#[derive(Deserialize)]
struct ChannelSettings {
    enabled: Option<bool>,
    /// The whole list, replacing the old one. Each entry is checked.
    allow: Option<Vec<String>>,
    /// `null` for every tool; a list for only those.
    #[serde(default, with = "double_option")]
    tools: Option<Option<Vec<String>>>,
    approve: Option<bool>,
    stream: Option<bool>,
    self_chat: Option<bool>,
    groups: Option<bool>,
}

/// Distinguishes "leave the tools alone" (absent) from "every tool" (`null`).
mod double_option {
    use serde::{Deserialize, Deserializer};
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Option<Vec<String>>>, D::Error> {
        Ok(Some(Option::deserialize(d)?))
    }
}

async fn channel_settings(
    AxumState(state): AxumState<State>,
    Path(kind): Path<String>,
    Json(body): Json<ChannelSettings>,
) -> Response {
    let kind = match parse_kind(&kind) {
        Ok(k) => k,
        Err(r) => return r,
    };
    let allow = match body.allow {
        None => None,
        Some(list) => {
            let mut out: Vec<String> = Vec::new();
            for entry in list.iter().filter(|e| !e.trim().is_empty()) {
                match channels::normalise_identity(kind, entry) {
                    Ok(id) if !out.contains(&id) => out.push(id),
                    Ok(_) => {}
                    Err(e) => return fail(StatusCode::BAD_REQUEST, e),
                }
            }
            Some(out)
        }
    };
    if kind == Kind::Telegram && (body.self_chat.is_some() || body.groups.is_some()) {
        return fail(StatusCode::BAD_REQUEST, "self_chat and groups are WhatsApp settings");
    }
    save(&state, |c| {
        let ch = &mut c.channels;
        if let Some(on) = body.enabled {
            ch.set_enabled(kind, on);
        }
        if let Some(list) = allow {
            *ch.allow_mut(kind) = list;
        }
        if let Some(tools) = body.tools {
            *ch.tools_mut(kind) = tools;
        }
        if let Some(on) = body.approve {
            ch.set_approve(kind, on);
        }
        if let Some(on) = body.stream {
            match kind {
                Kind::Telegram => ch.telegram.stream = on,
                Kind::WhatsApp => ch.whatsapp.stream = on,
            }
        }
        if let Some(on) = body.self_chat {
            ch.whatsapp.self_chat = on;
        }
        if let Some(on) = body.groups {
            ch.whatsapp.groups = on;
        }
    })
}

/// Change the configuration, save it, and let the gateway catch up.
fn save(state: &State, change: impl FnOnce(&mut ozgent_core::Config)) -> Response {
    let saved = {
        let mut config = state.config.lock().unwrap_or_else(|e| e.into_inner());
        change(&mut config);
        config.save(&state.paths)
    };
    if let Err(e) = saved {
        return fail(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    if let Some(g) = state.gateway.get() {
        g.apply();
    }
    Json(serde_json::json!({ "ok": true })).into_response()
}

#[derive(Deserialize)]
struct Token {
    token: String,
}

/// Set or change the bot token. Checked with Telegram before it is saved, so
/// a typo is an error here rather than a channel that fails to start.
async fn telegram_token(AxumState(state): AxumState<State>, Json(body): Json<Token>) -> Response {
    let g = match gateway(&state) {
        Ok(g) => g,
        Err(r) => return r,
    };
    let token = body.token.trim().to_string();
    if !looks_like_token(&token) {
        return fail(
            StatusCode::BAD_REQUEST,
            "that does not look like a bot token. It is two parts with a colon, like 123456789:AA…",
        );
    }
    let who = match g.check_telegram(&token).await {
        Ok(who) => who,
        Err(e) => return fail(StatusCode::BAD_REQUEST, format!("Telegram refused it: {e}")),
    };
    let response = save(&state, |c| {
        c.channels.telegram.token = token;
        c.channels.set_enabled(Kind::Telegram, true);
    });
    if response.status() != StatusCode::OK {
        return response;
    }
    Json(serde_json::json!({ "ok": true, "bot": who })).into_response()
}

/// A bot token's shape: digits, a colon, then letters, digits, `_` and `-`.
pub fn looks_like_token(t: &str) -> bool {
    let Some((id, secret)) = t.split_once(':') else { return false };
    !id.is_empty()
        && id.chars().all(|c| c.is_ascii_digit())
        && secret.len() >= 20
        && secret.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

async fn whatsapp_link(AxumState(state): AxumState<State>) -> Response {
    let g = match gateway(&state) {
        Ok(g) => g,
        Err(r) => return r,
    };
    match g.link_whatsapp().await {
        Ok(()) => Json(serde_json::json!({ "ok": true })).into_response(),
        Err(e) => fail(StatusCode::BAD_REQUEST, e),
    }
}

async fn sign_out(AxumState(state): AxumState<State>, Path(kind): Path<String>) -> Response {
    let kind = match parse_kind(&kind) {
        Ok(k) => k,
        Err(r) => return r,
    };
    match kind {
        Kind::Telegram => save(&state, |c| {
            c.channels.telegram.token.clear();
            c.channels.telegram.enabled = false;
        }),
        Kind::WhatsApp => {
            let g = match gateway(&state) {
                Ok(g) => g,
                Err(r) => return r,
            };
            match g.unlink_whatsapp().await {
                Ok(()) => save(&state, |c| c.channels.whatsapp.enabled = false),
                Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e),
            }
        }
    }
}

async fn restart(AxumState(state): AxumState<State>, Path(kind): Path<String>) -> Response {
    let kind = match parse_kind(&kind) {
        Ok(k) => k,
        Err(r) => return r,
    };
    match gateway(&state) {
        Ok(g) => {
            g.restart(kind);
            Json(serde_json::json!({ "ok": true })).into_response()
        }
        Err(r) => r,
    }
}

async fn new_pairing(AxumState(state): AxumState<State>) -> Response {
    match gateway(&state) {
        Ok(g) => Json(serde_json::json!({ "pairing": g.new_pairing_code() })).into_response(),
        Err(r) => r,
    }
}

// ------------------------------------------------------------ local models

/// Where uploads wait to be installed. Inside the models directory, so the
/// install is a hard link on the same filesystem rather than a second copy.
fn uploads_dir(state: &State) -> std::path::PathBuf {
    state.paths.models_dir().join(".uploads")
}

/// An upload id: what the page sends back to install it. Never a path.
fn valid_id(id: &str) -> bool {
    id.len() == 32 && id.chars().all(|c| c.is_ascii_hexdigit())
}

/// A file name as given by the browser, reduced to something safe to create.
fn clean_name(name: &str) -> Option<String> {
    let base = name.rsplit(['/', '\\']).next()?.trim();
    let base: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || "._-+".contains(c) { c } else { '_' })
        .collect();
    let lower = base.to_ascii_lowercase();
    (lower.ends_with(".gguf") && base.len() > 5 && !base.starts_with('.')).then_some(base)
}

/// Uploads older than this were abandoned: a closed tab, a cancel whose
/// cleanup never ran. Swept whenever a new upload starts.
const ABANDONED: Duration = Duration::from_secs(6 * 60 * 60);

fn sweep_uploads(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok());
        if old.is_some_and(|age| age > ABANDONED) {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

#[derive(Deserialize)]
struct UploadQuery {
    name: String,
}

/// Receive one GGUF file, streamed to disk as it arrives.
///
/// Refused at the first four bytes if it is not a GGUF, so a wrong file does
/// not cost a multi-gigabyte upload before saying so. A connection that drops
/// halfway leaves nothing behind.
async fn upload(
    AxumState(state): AxumState<State>,
    axum::extract::Query(query): axum::extract::Query<UploadQuery>,
    request: Request,
) -> Response {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    let Some(name) = clean_name(&query.name) else {
        return fail(StatusCode::BAD_REQUEST, "only .gguf files can be installed");
    };
    let root = uploads_dir(&state);
    sweep_uploads(&root);
    let id = secret::random_token(16);
    let dir = root.join(&id);
    if let Err(e) = tokio::fs::create_dir_all(&dir).await {
        return fail(StatusCode::INTERNAL_SERVER_ERROR, format!("creating {}: {e}", dir.display()));
    }
    let path = dir.join(&name);
    // Buffered: the body arrives in small chunks, and a file write per chunk
    // held a localhost upload to 50 MB/s.
    let mut file = match tokio::fs::File::create(&path).await {
        Ok(f) => tokio::io::BufWriter::with_capacity(8 << 20, f),
        Err(e) => return fail(StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    let mut stream = request.into_body().into_data_stream();
    let mut head: Vec<u8> = Vec::with_capacity(4);
    let mut bytes: u64 = 0;
    let outcome: Result<(), (StatusCode, String)> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| (StatusCode::BAD_REQUEST, format!("the upload was interrupted: {e}")))?;
            if head.len() < 4 {
                head.extend(chunk.iter().take(4 - head.len()));
                if head.len() == 4 && &head[..] != b"GGUF" {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("{name} is not a GGUF file. Safetensors and PyTorch checkpoints must be converted first."),
                    ));
                }
            }
            file.write_all(&chunk)
                .await
                .map_err(|e| (StatusCode::INSUFFICIENT_STORAGE, format!("writing the file: {e}")))?;
            bytes += chunk.len() as u64;
        }
        if head.len() < 4 {
            return Err((StatusCode::BAD_REQUEST, "the file is empty".to_string()));
        }
        file.flush().await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        Ok(())
    }
    .await;

    match outcome {
        Ok(()) => {
            tracing::info!("admin: received {name} ({bytes} bytes)");
            Json(serde_json::json!({ "id": id, "name": name, "bytes": bytes })).into_response()
        }
        Err((status, message)) => {
            drop(file);
            let _ = tokio::fs::remove_dir_all(&dir).await;
            fail(status, message)
        }
    }
}

#[derive(Deserialize)]
struct ImportBody {
    weights: String,
    mmproj: Option<String>,
    alias: Option<String>,
}

/// The one file an upload id holds.
fn uploaded(state: &State, id: &str) -> Result<std::path::PathBuf, Response> {
    if !valid_id(id) {
        return Err(fail(StatusCode::BAD_REQUEST, "not an upload id"));
    }
    let dir = uploads_dir(state).join(id);
    std::fs::read_dir(&dir)
        .ok()
        .and_then(|mut d| d.find_map(|e| e.ok().map(|e| e.path())))
        .ok_or_else(|| fail(StatusCode::NOT_FOUND, "that upload is gone; send the file again"))
}

/// Install uploaded files as a model.
async fn import_upload(AxumState(state): AxumState<State>, Json(body): Json<ImportBody>) -> Response {
    let weights = match uploaded(&state, &body.weights) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let mmproj = match body.mmproj.as_deref() {
        None => None,
        Some(id) => match uploaded(&state, id) {
            Ok(p) => Some(p),
            Err(r) => return r,
        },
    };
    let alias = body.alias.map(|a| a.trim().to_string()).filter(|a| !a.is_empty());
    if let Some(a) = &alias {
        if let Err(e) = ozgent_core::validate_alias(&state.paths, a, None) {
            return fail(StatusCode::BAD_REQUEST, e);
        }
    }

    let paths = state.paths.clone();
    let cleanup: Vec<std::path::PathBuf> =
        std::iter::once(&weights).chain(mmproj.as_ref()).filter_map(|p| p.parent().map(|d| d.to_path_buf())).collect();
    let result = tokio::task::spawn_blocking(move || {
        let reference = ozgent_hub::suggest_reference(&weights);
        let out = ozgent_hub::import(
            &paths,
            &ozgent_hub::ImportRequest { reference, weights, mmproj, copy: false },
        )?;
        if let Some(a) = &alias {
            ozgent_core::set_alias(&paths, &out.model, Some(a)).map_err(|e| ozgent_hub::HubError::Other(e.to_string()))?;
        }
        Ok::<_, ozgent_hub::HubError>(out)
    })
    .await;
    // The model directory holds its own link to the bytes now; the upload's
    // copy of the name is not needed either way.
    for dir in cleanup {
        let _ = tokio::fs::remove_dir_all(dir).await;
    }
    match result {
        Ok(Ok(out)) => Json(serde_json::json!({
            "model": out.model.to_string(),
            "vision": out.manifest.supports_vision(),
        }))
        .into_response(),
        Ok(Err(e)) => fail(StatusCode::BAD_REQUEST, e),
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(Deserialize)]
struct DefaultBody {
    model: String,
}

async fn make_default(AxumState(state): AxumState<State>, Json(body): Json<DefaultBody>) -> Response {
    if let Err(e) = ozgent_core::resolve(&state.paths, &body.model) {
        return fail(StatusCode::BAD_REQUEST, e);
    }
    save(&state, |c| c.default_model = Some(body.model.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_uploaded_name_cannot_escape_or_be_something_else() {
        assert_eq!(clean_name("Qwen3-8B-Q4_K_M.gguf").as_deref(), Some("Qwen3-8B-Q4_K_M.gguf"));
        assert_eq!(clean_name("../../etc/passwd.gguf").as_deref(), Some("passwd.gguf"));
        assert_eq!(clean_name("C:\\models\\x y.GGUF").as_deref(), Some("x_y.GGUF"));
        assert!(clean_name("model.safetensors").is_none());
        assert!(clean_name(".gguf").is_none());
        assert!(valid_id(&secret::random_token(16)));
        assert!(!valid_id("../x"));
    }

    #[test]
    fn a_bot_token_is_recognised_by_its_shape() {
        assert!(looks_like_token("123456789:AAHfK3-_abcdefghijklmnopqrstu"));
        assert!(!looks_like_token("123456789"));
        assert!(!looks_like_token("abc:AAHfK3abcdefghijklmnopqrstu"));
        assert!(!looks_like_token("123:short"));
        assert!(!looks_like_token("123:has space in it and is long"));
    }

    #[test]
    fn five_misses_lock_an_address_and_not_its_neighbour() {
        let mut f = Failures::default();
        let a = IpAddr::from([10, 0, 0, 2]);
        let b = IpAddr::from([10, 0, 0, 3]);
        for _ in 0..PER_ADDRESS {
            assert!(f.locked(a).is_none());
            f.record(a);
        }
        assert!(f.locked(a).is_some());
        assert!(f.locked(b).is_none());
    }

    #[test]
    fn many_misses_from_everywhere_lock_everyone() {
        let mut f = Failures::default();
        for i in 0..IN_TOTAL {
            f.record(IpAddr::from([10, 0, (i / 250) as u8, (i % 250) as u8 + 1]));
        }
        assert!(f.locked(IpAddr::from([192, 168, 1, 1])).is_some());
    }

    #[test]
    fn a_session_dies_with_the_password_it_was_issued_under() {
        let g = Guard::default();
        let t = g.issue("hash-1");
        assert!(g.check(&t, "hash-1"));
        assert!(!g.check(&t, "hash-2"), "a new password signs everyone out");
        assert!(!g.check(&t, "hash-1"), "and the session is gone for good");
    }

    #[test]
    fn a_new_password_clears_the_lockout() {
        let g = Guard::default();
        g.sync("old");
        let ip = IpAddr::from([10, 0, 0, 9]);
        for _ in 0..PER_ADDRESS {
            g.failures.lock().unwrap().record(ip);
        }
        assert!(g.failures.lock().unwrap().locked(ip).is_some());
        g.sync("new");
        assert!(g.failures.lock().unwrap().locked(ip).is_none());
    }

    #[test]
    fn the_cookie_is_found_among_others() {
        let mut h = HeaderMap::new();
        h.insert(header::COOKIE, "a=1; ozgent_admin=abc123; b=2".parse().unwrap());
        assert_eq!(cookie_token(&h).as_deref(), Some("abc123"));
        let c = session_cookie("t");
        assert!(c.contains("HttpOnly") && c.contains("SameSite=Strict"));
    }
}
