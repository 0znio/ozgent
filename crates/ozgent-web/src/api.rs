//! HTTP routes.

use axum::extract::{Path, State as AxumState};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, patch, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use ozgent_core::ThinkingMode;
use ozgent_memory::{Budget, ContextBuilder, Embedder, OwnerKind};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

use crate::state::State;

pub fn router(state: State) -> Router {
    Router::new()
        .route("/", get(index))
        // Client-side routes: the browser owns them, but a reload or a shared
        // link asks the server for them, so the shell must be served.
        .route("/new", get(index))
        .route("/chat", get(index))
        .route("/api/conversations/by-uuid/{uuid}", get(conversation_by_uuid))
        .route("/media/{name}", get(media_file))
        .route("/app.css", get(css))
        .route("/app.js", get(js))
        .route("/theme.js", get(theme))
        .route("/favicon.ico", get(favicon))
        .route("/logo.png", get(logo))
        .route("/api/models", get(models))
        .route("/api/conversations", get(list_conversations).post(new_conversation))
        .route("/api/conversations/{id}", delete(drop_conversation))
        .route("/api/conversations/{id}", patch(rename_conversation))
        .route("/api/conversations/{id}/messages", get(messages))
        .route("/api/settings", get(get_settings).put(put_settings))
        .route("/api/models/{model}/options", get(model_options).put(set_model_options).patch(patch_model_options))
        .route("/api/styles", get(list_styles))
        .route("/api/styles/{name}", axum::routing::put(save_style).delete(delete_style))
        .route("/api/tools", get(tools).put(set_tool_config))
        .route("/api/tools/active", get(active_tools))
        .route("/api/permissions", get(get_permissions).put(put_permissions))
        .route("/api/permissions/decide", post(decide_permission))
        .route("/api/conversations/{id}/facts", get(facts).post(add_fact))
        .route("/api/conversations/{id}/recall", post(preview_recall))
        .route("/api/facts/{id}", patch(pin_fact).delete(forget_fact))
        .route("/api/agents", get(list_agents))
        .route("/api/agents/{name}", axum::routing::put(save_agent).delete(delete_agent))
        .route("/api/chat", post(chat))
        .route("/api/unload", post(unload))
        .with_state(state)
}

// ------------------------------------------------------------------ assets

async fn index(AxumState(state): AxumState<State>, request: axum::extract::Request) -> Response {
    crate::access::page(&state, &request, include_str!("../assets/index.html"))
}

async fn theme() -> impl IntoResponse {
    ([("content-type", "text/javascript; charset=utf-8")], include_str!("../assets/theme.js"))
}

async fn css() -> impl IntoResponse {
    ([("content-type", "text/css; charset=utf-8")], include_str!("../assets/app.css"))
}

async fn js() -> impl IntoResponse {
    (
        [("content-type", "text/javascript; charset=utf-8")],
        include_str!("../assets/app.js"),
    )
}

/// Cached for a day: both ship inside the binary and change only with it.
async fn favicon() -> impl IntoResponse {
    (
        [("content-type", "image/x-icon"), ("cache-control", "public, max-age=86400")],
        include_bytes!("../assets/favicon.ico").as_slice(),
    )
}

async fn logo() -> impl IntoResponse {
    (
        [("content-type", "image/png"), ("cache-control", "public, max-age=86400")],
        include_bytes!("../assets/logo.png").as_slice(),
    )
}

// ------------------------------------------------------------------- errors

/// Any handler failure, rendered as JSON so the frontend can show it.
#[derive(Debug)]
pub struct ApiError {
    error: anyhow::Error,
    /// Which status to answer with.
    ///
    /// Worth carrying rather than defaulting everything to 500: a client that
    /// retries on 500 would keep resending a request that can never succeed, a
    /// log full of 500s hides the ones that are actually ozgent's fault, and a
    /// page asking for something that is gone should be told it is gone rather
    /// than that the server broke.
    status: StatusCode,
}

impl ApiError {
    pub fn bad_request(message: impl std::fmt::Display) -> Self {
        Self { error: anyhow::anyhow!("{message}"), status: StatusCode::BAD_REQUEST }
    }

    pub fn not_found(message: impl std::fmt::Display) -> Self {
        Self { error: anyhow::anyhow!("{message}"), status: StatusCode::NOT_FOUND }
    }

    pub fn internal(message: impl std::fmt::Display) -> Self {
        Self { error: anyhow::anyhow!("{message}"), status: StatusCode::INTERNAL_SERVER_ERROR }
    }

    #[cfg(test)]
    pub fn status(&self) -> u16 {
        self.status.as_u16()
    }
}

impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        Self { error: e.into(), status: StatusCode::INTERNAL_SERVER_ERROR }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!("{:#}", self.error);
        } else {
            // Not logged as an error: the caller made a mistake, and ozgent's
            // log is for ozgent's mistakes.
            tracing::debug!("rejected: {:#}", self.error);
        }
        (self.status, Json(serde_json::json!({ "error": self.error.to_string() }))).into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

// ------------------------------------------------------------------- agents

/// Every agent, and any file that could not be read.
///
/// The errors are part of the answer rather than a log line: an agent the user
/// saved by hand that does not appear in the `@` panel is a mystery unless the
/// page says why.
async fn list_agents(AxumState(state): AxumState<State>) -> Json<serde_json::Value> {
    let catalog = ozgent_core::AgentCatalog::load(&state.paths);
    Json(serde_json::json!({
        "agents": catalog.all(),
        "errors": catalog.errors,
        "limits": {
            "max_rounds": ozgent_core::agents::MAX_ROUNDS,
            "default_rounds": ozgent_core::agents::DEFAULT_ROUNDS,
            "max_name": ozgent_core::agents::MAX_NAME,
        },
    }))
}

/// Create or replace an agent. Saving a built-in's name makes an override.
async fn save_agent(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
    Json(definition): Json<ozgent_core::agents::Definition>,
) -> ApiResult<Json<serde_json::Value>> {
    use ozgent_core::agents::AgentError;
    match ozgent_core::agents::save(&state.paths, &name, &definition) {
        Ok(path) => {
            let catalog = ozgent_core::AgentCatalog::load(&state.paths);
            Ok(Json(serde_json::json!({
                "saved": path.display().to_string(),
                "agent": catalog.get(&name),
            })))
        }
        Err(e @ AgentError::Invalid(_)) => Err(ApiError::bad_request(e)),
        Err(e) => Err(e.into()),
    }
}

/// Delete an agent. Deleting an override brings the built-in back.
async fn delete_agent(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    use ozgent_core::agents::{AgentError, Origin};
    match ozgent_core::agents::remove(&state.paths, &name) {
        Ok(origin) => Ok(Json(serde_json::json!({ "restored_builtin": origin == Origin::Override }))),
        Err(e @ (AgentError::Builtin(_) | AgentError::NotFound(_) | AgentError::Invalid(_))) => {
            Err(ApiError::bad_request(e))
        }
        Err(e) => Err(e.into()),
    }
}

// -------------------------------------------------------------- permissions

/// One tool as the permissions manager shows it.
#[derive(Serialize)]
struct PermissionRow {
    name: String,
    description: String,
    /// What the tool says it does. Decides the rule when nothing names it.
    effect: String,
    /// The rule in force: the override if there is one, otherwise the rule the
    /// effect inherits.
    rule: String,
    /// Whether that rule was set for this tool by name, as opposed to being
    /// inherited. The page shows the difference so "clear" has a meaning.
    overridden: bool,
    /// Allowed for the rest of this run by a "don't ask again" answer.
    granted: bool,
}

#[derive(Serialize)]
struct PermissionsView {
    /// The effect defaults, as `{ read, write, execute, unknown }`.
    defaults: serde_json::Value,
    tools: Vec<PermissionRow>,
}

/// The policy, joined against the tools actually installed.
///
/// Joined rather than returned raw because a list of rules is unreadable
/// without the tools they apply to, and a tool with no rule of its own still
/// has an answer — the one its effect gives it — which is the thing a user
/// most needs to see before deciding to override it.
async fn get_permissions(AxumState(state): AxumState<State>) -> ApiResult<Json<PermissionsView>> {
    let policy = state.config.lock().unwrap().permissions.clone();
    let grants = state.permissions.grants.lock().unwrap().clone();
    let granted: std::collections::BTreeSet<&str> = grants.allowed().collect();

    let installed = match crate::worker::current_tools(&state.tools) {
        Some(t) => t.host.tools().to_vec(),
        None => Vec::new(),
    };
    let tools = installed
        .iter()
        .map(|spec| PermissionRow {
            rule: policy.rule_for(&spec.name, spec.effect).to_string(),
            overridden: policy.is_overridden(&spec.name),
            granted: granted.contains(spec.name.as_str()),
            effect: spec.effect.to_string(),
            description: ozgent_tools::first_line(&spec.description).to_string(),
            name: spec.name.clone(),
        })
        .collect();

    Ok(Json(PermissionsView {
        defaults: serde_json::json!({
            "read": policy.read.to_string(),
            "write": policy.write.to_string(),
            "execute": policy.execute.to_string(),
            "unknown": policy.unknown.to_string(),
        }),
        tools,
    }))
}

#[derive(Deserialize)]
struct PermissionsUpdate {
    #[serde(default)]
    read: Option<String>,
    #[serde(default)]
    write: Option<String>,
    #[serde(default)]
    execute: Option<String>,
    #[serde(default)]
    unknown: Option<String>,
    /// Per-tool rules. A tool mapped to `null` has its override cleared and
    /// goes back to inheriting from its effect — which is a different state
    /// from being set to the same value the effect would have given it.
    #[serde(default)]
    tools: std::collections::BTreeMap<String, Option<String>>,
    /// Drop every "don't ask again this session" answer.
    #[serde(default)]
    clear_session: bool,
}

async fn put_permissions(
    AxumState(state): AxumState<State>,
    Json(body): Json<PermissionsUpdate>,
) -> ApiResult<StatusCode> {
    let parse = |name: &str, value: &Option<String>| -> ApiResult<Option<ozgent_core::Rule>> {
        match value {
            None => Ok(None),
            Some(text) => text
                .parse::<ozgent_core::Rule>()
                .map(Some)
                .map_err(|e| ApiError::bad_request(format!("{name}: {e}"))),
        }
    };
    let read = parse("read", &body.read)?;
    let write = parse("write", &body.write)?;
    let execute = parse("execute", &body.execute)?;
    let unknown = parse("unknown", &body.unknown)?;

    let mut per_tool = Vec::new();
    for (tool, value) in &body.tools {
        per_tool.push((tool.clone(), parse(tool, value)?));
    }

    let config = {
        let mut config = state.config.lock().unwrap();
        let p = &mut config.permissions;
        if let Some(r) = read {
            p.read = r;
        }
        if let Some(r) = write {
            p.write = r;
        }
        if let Some(r) = execute {
            p.execute = r;
        }
        if let Some(r) = unknown {
            p.unknown = r;
        }
        for (tool, rule) in per_tool {
            p.set(&tool, rule);
        }
        config.clone()
    };
    config.save(&state.paths)?;

    if body.clear_session {
        state.permissions.grants.lock().unwrap().clear();
    }
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct Decision {
    /// The call id from the `permission` event.
    id: String,
    choice: ozgent_core::Choice,
}

/// Answer one outstanding permission question.
///
/// A `404` means nothing was waiting: the turn ended, the wait ran out, or the
/// button was clicked twice. The page treats that as "already decided" rather
/// than an error, because from the user's side it is.
async fn decide_permission(
    AxumState(state): AxumState<State>,
    Json(body): Json<Decision>,
) -> StatusCode {
    if state.permissions.pending.answer(&body.id, body.choice) {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

// ------------------------------------------------------------------- models

#[derive(Serialize)]
struct ModelInfo {
    reference: String,
    alias: Option<String>,
    quantization: Option<String>,
    size_bytes: Option<u64>,
    vision: bool,
    /// Whether `default_model` names this one. The page opens on it, so the
    /// model someone uses most is not one they re-pick on every visit.
    is_default: bool,
    /// An embedding model: it turns text into vectors and cannot chat. Kept
    /// out of the chat picker, where it used to be the first entry and so the
    /// one the page opened on.
    embedding: bool,
    /// The context it was trained for, from its own header.
    context_train: Option<u32>,
}

async fn models(AxumState(state): AxumState<State>) -> Json<Vec<ModelInfo>> {
    // The config, not the start-up clone: the default can be changed from the
    // settings page, and a stale copy would keep opening the old model.
    let default = state.config.lock().unwrap().default_model.clone();
    let models = ozgent_core::installed(&state.paths)
        .into_iter()
        .map(|m| {
            let reference = m.model.to_string();
            // `default_model` is stored as the user wrote it, which may be
            // either spelling, so both have to be checked.
            let is_default = default
                .as_deref()
                .is_some_and(|d| d == reference || Some(d) == m.manifest.alias.as_deref());
            let weights = m.manifest.primary_weights(&m.dir);
            ModelInfo {
                reference,
                alias: m.manifest.alias.clone(),
                quantization: m.manifest.quantization.clone(),
                size_bytes: m.manifest.size_bytes,
                vision: m.manifest.supports_vision(),
                is_default,
                embedding: ozgent_llama::layout::is_embedding(&weights),
                context_train: ozgent_llama::layout::read(&weights)
                    .map(|l| l.context_train)
                    .filter(|c| *c > 0),
            }
        })
        .collect();
    Json(models)
}

// ------------------------------------------------------------ conversations

#[derive(Serialize)]
struct ConversationInfo {
    id: i64,
    /// What goes in the URL.
    uuid: String,
    title: String,
    model: Option<String>,
    created_at: i64,
    messages: i64,
}

async fn list_conversations(
    AxumState(state): AxumState<State>,
) -> ApiResult<Json<Vec<ConversationInfo>>> {
    let store = state.store.lock().unwrap();
    let mut out = Vec::new();
    // Only conversations that hold something. One with no messages is a
    // placeholder nobody filled, and offering to reopen nothing is what filled
    // this sidebar with rows called "New chat".
    for c in store.list_active_conversations(200)? {
        let messages = store.message_count(c.id)?;
        out.push(ConversationInfo {
            id: c.id,
            uuid: c.uuid,
            title: c.title,
            model: c.model,
            created_at: c.created_at,
            messages,
        });
    }
    Ok(Json(out))
}

#[derive(Deserialize)]
struct NewConversation {
    title: Option<String>,
    model: Option<String>,
}

async fn new_conversation(
    AxumState(state): AxumState<State>,
    Json(body): Json<NewConversation>,
) -> ApiResult<Json<serde_json::Value>> {
    let store = state.store.lock().unwrap();
    // Untitled by default, not "New chat": the first message names it, and a
    // placeholder title outlives its usefulness the moment that happens.
    let id = store.create_conversation(
        body.title.as_deref().unwrap_or(""),
        body.model.as_deref(),
    )?;
    let uuid = store.get_conversation(id)?.map(|c| c.uuid).unwrap_or_default();
    Ok(Json(serde_json::json!({ "id": id, "uuid": uuid })))
}

/// Resolve a public id to the conversation behind it.
async fn conversation_by_uuid(
    AxumState(state): AxumState<State>,
    Path(uuid): Path<String>,
) -> ApiResult<Json<ConversationInfo>> {
    let store = state.store.lock().unwrap();
    let found = store
        .conversation_by_uuid(&uuid)?
        .ok_or_else(|| ApiError::bad_request(format!("no conversation {uuid}")))?;
    let messages = store.message_count(found.id)?;
    Ok(Json(ConversationInfo {
        id: found.id,
        uuid: found.uuid,
        title: found.title,
        model: found.model,
        created_at: found.created_at,
        messages,
    }))
}

async fn drop_conversation(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
) -> ApiResult<StatusCode> {
    state.store.lock().unwrap().delete_conversation(id)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct Rename {
    title: String,
}

async fn rename_conversation(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
    Json(body): Json<Rename>,
) -> ApiResult<StatusCode> {
    state.store.lock().unwrap().rename_conversation(id, &body.title)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Serialize)]
struct MessageInfo {
    id: i64,
    /// Position in the conversation. What a rewind is addressed by, so
    /// "regenerate" and "edit and resend" have something to name.
    seq: i64,
    role: String,
    text: String,
    created_at: i64,
    /// The reasoning trace, so reloading a conversation shows what the model
    /// worked through rather than losing it.
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    /// Tool activity for this turn, in the order it happened.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
    /// What the reply cost: speed, tokens, how long it reasoned.
    #[serde(skip_serializing_if = "Option::is_none")]
    stats: Option<serde_json::Value>,
    /// URLs of the attachments this message carried.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    media: Vec<String>,
}

async fn messages(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
) -> ApiResult<Json<Vec<MessageInfo>>> {
    let store = state.store.lock().unwrap();
    let out = store
        .messages(id)?
        .into_iter()
        .map(|m| MessageInfo {
            id: m.id,
            seq: m.seq,
            role: m.role,
            text: m.content,
            created_at: m.created_at,
            thinking: m.thinking.filter(|t| !t.trim().is_empty()),
            stats: m.stats.as_deref().and_then(|raw| serde_json::from_str(raw).ok()),
            tool_calls: m
                .tool_calls
                .as_deref()
                .and_then(|raw| serde_json::from_str(raw).ok()),
            media: m
                .media
                .as_deref()
                .and_then(|raw| serde_json::from_str::<Vec<String>>(raw).ok())
                .unwrap_or_default()
                .into_iter()
                .filter(|n| crate::media::safe_name(n))
                .map(|n| format!("/media/{n}"))
                .collect(),
        })
        .collect();
    Ok(Json(out))
}

// ----------------------------------------------------------------- settings

#[cfg(test)]
mod public_settings_tests {
    #[test]
    fn the_open_settings_page_never_sees_admin_or_channel_secrets() {
        let mut c = ozgent_core::Config::default();
        c.channels.telegram.token = "123:secret".into();
        c.channels.telegram.allow = vec!["4242".into()];
        c.web.admin_password_hash = Some("$argon2id$…".into());
        let shown = serde_json::to_string(&super::public(&c)).unwrap();
        assert!(!shown.contains("secret") && !shown.contains("argon2") && !shown.contains("4242"), "{shown}");
    }
}

/// The settings page's view of the configuration: everything except what
/// belongs to `/admin`.
///
/// This page has no password, so it must neither show nor change `[channels]`
/// (a bot token, and who in the world may reach this machine's tools) or
/// `[web]` (the admin password hash). Sending them would hand out the token;
/// accepting them back would let anyone who can load this page replace the
/// admin password or open the allowlist, and skip `/admin` altogether.
fn public(config: &ozgent_core::Config) -> ozgent_core::Config {
    let mut c = config.clone();
    c.channels = Default::default();
    c.web = Default::default();
    c
}

/// What a secret reads as over the API. Sent back unchanged, it means "keep
/// what is there".
const REDACTED: &str = "••••••••";

/// Whether a config key names a secret.
fn secret_key(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    ["key", "token", "secret", "password"].iter().any(|s| n.contains(s))
}

/// Mask every secret in a config, however deeply a tool's settings nest it.
///
/// A tool's own section is opaque here — `[tools.config.web_search.tavily]
/// api_key` is only a key because of its name — so they are found by name.
/// Before this, `GET /api/settings` returned the search provider's key to
/// anyone who could reach the port.
fn redact(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if secret_key(k) && v.as_str().is_some_and(|s| !s.is_empty()) {
                    *v = REDACTED.into();
                } else {
                    redact(v);
                }
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(redact),
        _ => {}
    }
}

/// Put back every secret the page returned masked, from the config in force.
fn unredact(value: &mut serde_json::Value, current: &serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                let here = current.get(k).unwrap_or(&serde_json::Value::Null);
                if v.as_str() == Some(REDACTED) {
                    *v = here.clone();
                } else {
                    unredact(v, here);
                }
            }
        }
        serde_json::Value::Array(items) => {
            for (i, v) in items.iter_mut().enumerate() {
                unredact(v, current.get(i).unwrap_or(&serde_json::Value::Null));
            }
        }
        _ => {}
    }
}

async fn get_settings(AxumState(state): AxumState<State>) -> ApiResult<Json<serde_json::Value>> {
    let mut value = serde_json::to_value(public(&state.config.lock().unwrap()))
        .map_err(|e| ApiError::internal(e.to_string()))?;
    redact(&mut value);
    Ok(Json(value))
}

async fn put_settings(
    AxumState(state): AxumState<State>,
    Json(mut raw): Json<serde_json::Value>,
) -> ApiResult<StatusCode> {
    let current = state.config.lock().unwrap().clone();
    let current_value = serde_json::to_value(&current).map_err(|e| ApiError::internal(e.to_string()))?;
    unredact(&mut raw, &current_value);
    let mut body: ozgent_core::Config =
        serde_json::from_value(raw).map_err(|e| ApiError::bad_request(e.to_string()))?;
    // Whatever the page sent for these is ignored; the server's own stand.
    body.channels = current.channels.clone();
    body.web = current.web.clone();
    // And for everything that decides what code runs on this machine, or
    // where from: the interpreter, the directories tools load from, the MCP
    // servers it starts, and what a tool may touch once running. Writable
    // here, any of them turned a request into a program of the caller's
    // choosing. They change on the admin page or in the file.
    body.mcp = current.mcp.clone();
    body.tools.python = current.tools.python.clone();
    body.tools.extra_paths = current.tools.extra_paths.clone();
    match current.tools.config.get("permissions") {
        Some(p) => {
            body.tools.config.insert("permissions".into(), p.clone());
        }
        None => {
            body.tools.config.remove("permissions");
        }
    }
    body.save(&state.paths)?;
    *state.config.lock().unwrap() = body;
    // Nothing to unload: the worker reads the live config on every turn and
    // reloads by itself only when a load-time setting changed. Unloading here
    // made every press of Save cost a full model load on the next message.
    Ok(StatusCode::NO_CONTENT)
}

async fn unload(AxumState(state): AxumState<State>) -> StatusCode {
    state.worker.unload();
    StatusCode::NO_CONTENT
}

// --------------------------------------------------------------------- chat

#[derive(Deserialize)]
struct ChatRequest {
    conversation: i64,
    model: String,
    message: String,
    #[serde(default)]
    thinking: Option<String>,
    /// Off lets the user ask a question without the model reaching for a tool.
    #[serde(default = "yes")]
    tools: bool,
    /// Attached images, as `data:` URLs or bare base64.
    #[serde(default)]
    images: Vec<String>,
    /// A persona for this conversation only; see `turn::Turn::system`.
    #[serde(default)]
    system: Option<String>,
    /// Tools switched off in the composer's tools tray, by name.
    #[serde(default)]
    tools_off: Vec<String>,
}

fn yes() -> bool {
    true
}

async fn chat(
    AxumState(state): AxumState<State>,
    Json(body): Json<ChatRequest>,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    let thinking = body.thinking.as_deref().and_then(|t| match t {
        "on" => Some(ThinkingMode::On),
        "off" => Some(ThinkingMode::Off),
        "auto" => Some(ThinkingMode::Auto),
        _ => None,
    });

    let images = body
        .images
        .iter()
        .map(|d| decode_data_url(d))
        .collect::<Result<Vec<_>, _>>()
        // An unreadable data URL came from the browser, not from here.
        .map_err(ApiError::bad_request)?;

    let events = crate::turn::start(
        &state,
        crate::turn::Turn {
            system: body.system.clone().filter(|s| !s.trim().is_empty()),
            conversation: body.conversation,
            model: body.model,
            message: body.message,
            thinking,
            tools: body.tools,
            native_tools: None,
            tools_off: body.tools_off,
            images,
            // The browser can show a permission card and answer it.
            can_ask: true,
            // Someone at this machine, or on a network its owner chose.
            caller: None,
        },
    )?;

    let sse = futures_util::stream::unfold(events, |mut rx| async move {
        let event = rx.recv().await?;
        let json = serde_json::to_string(&event).unwrap_or_default();
        Some((Ok(SseEvent::default().data(json)), rx))
    });

    Ok(Sse::new(sse))
}

// -------------------------------------------------------- model options

/// Every knob, with the value in force and whether this model overrides it.
///
/// The distinction matters: showing only the effective value makes it
/// impossible to tell a deliberate per-model setting from an inherited
/// default, and clearing an override then looks the same as setting it.
#[derive(Serialize)]
struct ModelOptions {
    model: String,
    /// What this model actually runs with, after defaults and manifest merge.
    effective: serde_json::Value,
    /// Only the keys set for this model in `config.toml`.
    overrides: serde_json::Value,
    /// What each setting would be with none of this model's own overrides:
    /// the value a per-setting reset goes back to.
    inherited: serde_json::Value,
    /// Bounds the page should not let the user exceed, read from the model
    /// itself. Without these the settings page invents its own, and a model
    /// trained for 256k gets a slider that stops at 32k.
    limits: serde_json::Value,
}

async fn model_options(
    AxumState(state): AxumState<State>,
    Path(model): Path<String>,
) -> ApiResult<Json<ModelOptions>> {
    let found = ozgent_core::resolve(&state.paths, &model)?;
    let key = found.model.to_string();
    let config = state.config.lock().unwrap();

    let merged = config.options_for(&key).merge(&found.manifest.defaults);
    let resolved = merged.resolve();
    let overrides = config.models.get(&key).cloned().unwrap_or_default();

    // Read from the GGUF header only — the weights are never mapped, so this
    // costs no more than opening the file.
    let layout = ozgent_llama::layout::read(&found.manifest.primary_weights(&found.dir));
    let context_max = layout.map(|l| l.context_train).filter(|c| *c > 0);

    Ok(Json(ModelOptions {
        model: key,
        effective: serde_json::to_value(resolved_view(&resolved))?,
        overrides: tidy(serde_json::to_value(&overrides)?),
        inherited: resolved_view(&config.defaults.clone().merge(&found.manifest.defaults).resolve()),
        limits: serde_json::json!({
            "context_length": context_max,
            // Output cannot exceed the window it has to fit inside.
            "max_tokens": context_max.unwrap_or(resolved.context_length),
        }),
    }))
}

/// `body` without the settings that say exactly what would be inherited.
///
/// Compared as JSON rather than field by field, so a setting added later is
/// covered without anybody remembering to add it here. A field the inherited
/// layer says nothing about is kept: it is an override over ozgent's own
/// default, which is the one thing a person may well want held still.
fn without_inherited(
    body: ozgent_core::Options,
    inherited: &ozgent_core::Options,
) -> Result<ozgent_core::Options, serde_json::Error> {
    let mut mine = serde_json::to_value(&body)?;
    let theirs = serde_json::to_value(inherited)?;
    if let (Some(mine_map), Some(theirs_map)) = (mine.as_object_mut(), theirs.as_object()) {
        mine_map.retain(|key, value| {
            !matches!(theirs_map.get(key), Some(other) if other == value && !value.is_null())
        });
    }
    serde_json::from_value(mine)
}

/// A settings object as a person would read it: no null entries, and every
/// number the shortest decimal that means the same `f32` — the type every
/// fractional setting is stored as.
fn tidy(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.into_iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| (k, tidy(v))).collect(),
        ),
        serde_json::Value::Number(n) if n.is_f64() => {
            n.as_f64().map(|f| serde_json::json!(decimal(f as f32))).unwrap_or(serde_json::Value::Number(n))
        }
        other => other,
    }
}

/// A setting stored as `f32`, as the decimal it was written as.
///
/// JSON numbers are `f64`, and widening `0.8f32` gives 0.800000011920929 —
/// which the settings page then showed. The shortest decimal that reads back
/// as the same `f32` is the number somebody actually typed.
fn decimal(x: f32) -> f64 {
    format!("{x}").parse().unwrap_or(x as f64)
}

/// A flat, JSON-friendly view of the resolved options for display.
fn resolved_view(r: &ozgent_core::options::Resolved) -> serde_json::Value {
    serde_json::json!({
        "gpu_layers": r.gpu_layers.to_string(),
        "cpu_moe": r.cpu_moe.to_string(),
        "context_length": r.context_length,
        "batch_size": r.batch_size,
        "flash_attention": r.flash_attention,
        "cache_type_k": format!("{:?}", r.cache_type_k).to_lowercase(),
        "cache_type_v": format!("{:?}", r.cache_type_v).to_lowercase(),
        "temperature": decimal(r.temperature),
        "top_p": decimal(r.top_p),
        "top_k": r.top_k,
        "min_p": decimal(r.min_p),
        "repeat_penalty": decimal(r.repeat_penalty),
        "repeat_last_n": r.repeat_last_n,
        "max_tokens": r.max_tokens,
        "thinking": format!("{:?}", r.thinking).to_lowercase(),
        "reasoning_effort": r.reasoning_effort.to_string(),
        "prefix_reuse": format!("{:?}", r.prefix_reuse).to_lowercase(),
        "tools": r.tools,
        "system_prompt": r.system_prompt,
        "style": r.style,
    })
}

async fn set_model_options(
    AxumState(state): AxumState<State>,
    Path(model): Path<String>,
    Json(body): Json<ozgent_core::Options>,
) -> ApiResult<StatusCode> {
    store_model_options(&state, &model, body)
}

/// Change some of a model's settings and leave the rest: `{"style":
/// "concise"}` sets one, `{"system_prompt": null}` clears one. What a slash
/// command needs, where the settings page sends the whole set.
async fn patch_model_options(
    AxumState(state): AxumState<State>,
    Path(model): Path<String>,
    Json(patch): Json<serde_json::Map<String, serde_json::Value>>,
) -> ApiResult<StatusCode> {
    let found = ozgent_core::resolve(&state.paths, &model)?;
    let current = state.config.lock().unwrap().models.get(&found.model.to_string()).cloned().unwrap_or_default();
    let mut value = serde_json::to_value(current)?;
    let object = value.as_object_mut().ok_or_else(|| ApiError::internal("options are not an object"))?;
    for (k, v) in patch {
        if v.is_null() {
            object.remove(&k);
        } else {
            object.insert(k, v);
        }
    }
    if let Some(style) = object.get("style").and_then(|v| v.as_str()) {
        let custom = state.config.lock().unwrap().styles.clone();
        if ozgent_core::styles::find(&custom, style).is_none() {
            return Err(ApiError::bad_request(format!("no style called {style:?}")));
        }
        object.insert("style".into(), ozgent_core::styles::normalise_name(style).into());
    }
    let body: ozgent_core::Options = serde_json::from_value(value).map_err(|e| ApiError::bad_request(e.to_string()))?;
    store_model_options(&state, &model, body)
}

fn store_model_options(state: &State, model: &str, body: ozgent_core::Options) -> ApiResult<StatusCode> {
    let found = ozgent_core::resolve(&state.paths, model)?;
    let key = found.model.to_string();

    let mut config = state.config.lock().unwrap();
    // A setting that merely repeats what it would inherit is not an override,
    // and storing it as one pins the value for good: the slider a person
    // nudged and put back would go on holding that model to a number long
    // after the default it came from had moved. This is how a 4B ended up
    // pinned to a 107,008-token window written by an earlier load, which then
    // outlived every improvement to how the window is chosen.
    let inherited = config.defaults.clone().merge(&found.manifest.defaults);
    let body = without_inherited(body, &inherited)?;
    // An empty override layer is removed rather than stored, so the file does
    // not accumulate sections that say nothing. Unset fields serialise as
    // null, so "empty" means nothing but nulls.
    if tidy(serde_json::to_value(&body)?).as_object().is_some_and(|o| o.is_empty()) {
        config.models.remove(&key);
    } else {
        config.models.insert(key, body);
    }
    config.save(&state.paths)?;
    drop(config);

    // No unload here. Every turn re-resolves its options against the live
    // config, and the worker reloads by itself when a load-time setting —
    // context, layers, cache types — actually differs from what is loaded.
    // Unloading on every save made a temperature change cost a full model
    // load on the next message.
    Ok(StatusCode::NO_CONTENT)
}

// ---------------------------------------------------------------- tools

#[derive(Serialize)]
struct ToolsView {
    enabled: bool,
    python: String,
    timeout_seconds: u64,
    max_calls_per_turn: u32,
    available: Vec<crate::state::ToolSummary>,
    /// Search providers ozgent ships, and whether each is usable.
    search_providers: Vec<ProviderInfo>,
    search_provider: String,
}

#[derive(Serialize)]
struct ProviderInfo {
    name: String,
    /// Whether a key is present. The key itself is never sent to the browser.
    configured: bool,
    needs_key: bool,
}

/// Providers the shipped `web_search` tool supports.
///
/// Kept here rather than discovered, because the list is part of the settings
/// UI's contract; `python/ozgent_tools/builtin/web_search.py` is the source of
/// truth for what each one does.
const SEARCH_PROVIDERS: [(&str, bool); 3] =
    [("brave", true), ("tavily", true), ("duckduckgo", false)];

/// One tool the model can be offered right now, for the composer's tray.
#[derive(Serialize)]
struct ActiveTool {
    name: String,
    /// A friendlier name, where the tool's own is an identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    /// The first sentence of what it does.
    description: String,
    effect: String,
    /// What the permission rules say: allow, ask or deny.
    rule: String,
}

/// The tools that are running, read from the live host — unlike `/api/tools`,
/// which starts an interpreter to find everything installed. Includes
/// `ask_agent` when the model may hand requests to agents.
async fn active_tools(AxumState(state): AxumState<State>) -> Json<Vec<ActiveTool>> {
    let config = state.config.lock().unwrap().clone();
    if !config.tools.enabled {
        return Json(Vec::new());
    }
    let mut out: Vec<ActiveTool> = crate::worker::current_tools(&state.tools)
        .map(|t| {
            t.host
                .tools()
                .iter()
                .filter(|s| !config.tools.disabled.contains(&s.name))
                .map(|s| ActiveTool {
                    name: s.name.clone(),
                    label: None,
                    description: ozgent_tools::first_line(&s.description).to_string(),
                    effect: format!("{:?}", s.effect).to_lowercase(),
                    rule: config.permissions.rule_for(&s.name, s.effect).to_string(),
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    if config.tools.handoff && !ozgent_core::AgentCatalog::load(&state.paths).all().is_empty() {
        out.push(ActiveTool {
            name: ozgent_core::agents::HANDOFF_TOOL.to_string(),
            label: Some("hand off to @agents".into()),
            description: "Let the model pass a question to an agent when it fits one".into(),
            effect: "read".into(),
            rule: "allow".into(),
        });
    }
    Json(out)
}

async fn tools(AxumState(state): AxumState<State>) -> ApiResult<Json<ToolsView>> {
    // Cloned rather than borrowed: discovering tools is async, and holding a
    // std mutex guard across an await makes the whole handler future !Send.
    let config = state.config.lock().unwrap().clone();
    let ws = config.tools.config.get("web_search");
    let current = ws
        .and_then(|v| v.get("provider"))
        .and_then(|v| v.as_str())
        .unwrap_or("duckduckgo")
        .to_string();

    let search_providers = SEARCH_PROVIDERS
        .iter()
        .map(|(name, needs_key)| ProviderInfo {
            name: (*name).to_string(),
            needs_key: *needs_key,
            configured: !needs_key
                || ws
                    .and_then(|v| v.get(name))
                    .and_then(|v| v.get("api_key"))
                    .and_then(|v| v.as_str())
                    .is_some_and(|k| !k.trim().is_empty()),
        })
        .collect();

    // Listing tools means starting the Python worker, so a failure here is
    // reported as an empty list with the reason rather than a dead page.
    let available = match crate::state::discover_tools(&state.paths, &config).await {
        Ok(list) => list,
        Err(e) => {
            tracing::warn!("listing tools: {e}");
            Vec::new()
        }
    };

    Ok(Json(ToolsView {
        enabled: config.tools.enabled,
        python: config.tools.python.clone(),
        timeout_seconds: config.tools.timeout_seconds,
        max_calls_per_turn: config.tools.max_calls_per_turn,
        available,
        search_providers,
        search_provider: current,
    }))
}

#[derive(Deserialize)]
struct ToolUpdate {
    enabled: Option<bool>,
    search_provider: Option<String>,
    /// Set a key for `search_provider`. Absent leaves the stored key alone;
    /// an empty string clears it.
    api_key: Option<String>,
    disabled: Option<Vec<String>>,
}

async fn set_tool_config(
    AxumState(state): AxumState<State>,
    Json(body): Json<ToolUpdate>,
) -> ApiResult<StatusCode> {
    // Scoped so the guard is provably gone before the await below: held
    // across one, the handler's future is not `Send` and axum rejects it.
    let snapshot = {
        let mut config = state.config.lock().unwrap();

        if let Some(enabled) = body.enabled {
            config.tools.enabled = enabled;
        }
        if let Some(disabled) = body.disabled {
            config.tools.disabled = disabled;
        }
        if let Some(provider) = &body.search_provider {
            if !SEARCH_PROVIDERS.iter().any(|(n, _)| n == provider) {
                return Err(ApiError::bad_request(format!(
                    "unknown search provider {provider:?}"
                )));
            }
            set_search_provider(&mut config, provider, body.api_key.as_deref());
        }

        config.save(&state.paths)?;
        // The key lives in this file; it must not be world-readable.
        harden(&state.paths.config_file());
        config.clone()
    };

    // The host read its configuration when its interpreter started, so a new
    // provider or key reaches it only through a new interpreter. Without this
    // the page said "saved" and the old provider went on answering until the
    // server was restarted.
    restart_tools(&state, snapshot).await;
    Ok(StatusCode::NO_CONTENT)
}

/// Replace the running tool host with one built from `config`.
///
/// A failure leaves the previous host in place rather than none at all: tools
/// that were working should not stop because a new setting was rejected.
pub(crate) async fn restart_tools(state: &State, config: ozgent_core::Config) {
    // Each guard is bound and dropped before the next await: held across one,
    // the handler's future stops being `Send` and axum will not take it.
    let previous = if config.tools.enabled {
        match crate::state::start_tools(&state.paths, &config).await {
            Ok(fresh) => state.tools.lock().unwrap().replace(fresh),
            Err(e) => {
                tracing::warn!("keeping the running tools: restarting them failed: {e}");
                return;
            }
        }
    } else {
        let taken = state.tools.lock().unwrap().take();
        taken
    };

    // Shut the old interpreter down in the background: waiting for it to exit
    // would add that delay to the request that asked for the change.
    if let Some(old) = previous {
        tokio::spawn(async move { old.host.shutdown().await });
    }
}

/// Write the provider choice, and its key when one was supplied.
fn set_search_provider(config: &mut ozgent_core::Config, provider: &str, api_key: Option<&str>) {
    let entry = config
        .tools
        .config
        .entry("web_search".to_string())
        .or_insert_with(|| toml::Value::Table(Default::default()));
    let Some(table) = entry.as_table_mut() else { return };

    table.insert("provider".into(), toml::Value::String(provider.into()));

    if let Some(key) = api_key {
        let slot = table
            .entry(provider.to_string())
            .or_insert_with(|| toml::Value::Table(Default::default()));
        if let Some(inner) = slot.as_table_mut() {
            if key.trim().is_empty() {
                inner.remove("api_key");
            } else {
                inner.insert("api_key".into(), toml::Value::String(key.trim().into()));
            }
        }
    }
}

#[cfg(unix)]
fn harden(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn harden(_path: &std::path::Path) {}

// --------------------------------------------------------------- memory

#[derive(Serialize)]
struct FactInfo {
    id: i64,
    text: String,
    scope: String,
    pinned: bool,
    created_at: i64,
}

async fn facts(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
) -> ApiResult<Json<Vec<FactInfo>>> {
    let store = state.store.lock().unwrap();
    let out = store
        .facts_for(id)?
        .into_iter()
        .map(|f| FactInfo {
            id: f.id,
            text: f.text,
            scope: match f.scope {
                ozgent_memory::Scope::User => "user".into(),
                ozgent_memory::Scope::Conversation => "conversation".into(),
            },
            pinned: f.pinned,
            created_at: f.created_at,
        })
        .collect();
    Ok(Json(out))
}

#[derive(Deserialize)]
struct NewFact {
    text: String,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    pinned: bool,
}

async fn add_fact(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
    Json(body): Json<NewFact>,
) -> ApiResult<Json<serde_json::Value>> {
    if body.text.trim().is_empty() {
        return Err(ApiError::bad_request("a fact needs some text"));
    }
    let scope = match body.scope.as_deref() {
        Some("user") => ozgent_memory::Scope::User,
        _ => ozgent_memory::Scope::Conversation,
    };
    let store = state.store.lock().unwrap();
    let fact_id = store.add_fact(Some(id), scope, body.text.trim(), None)?;
    if body.pinned {
        store.set_pinned(fact_id, true)?;
    }
    // Indexed like anything else, so retrieval can surface it later.
    store.put_embedding(
        OwnerKind::Fact,
        fact_id,
        &state.embedder.embed(body.text.trim()),
    )?;
    Ok(Json(serde_json::json!({ "id": fact_id })))
}

#[derive(Deserialize)]
struct PinUpdate {
    pinned: bool,
}

async fn pin_fact(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
    Json(body): Json<PinUpdate>,
) -> ApiResult<StatusCode> {
    state.store.lock().unwrap().set_pinned(id, body.pinned)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn forget_fact(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
) -> ApiResult<StatusCode> {
    state.store.lock().unwrap().delete_fact(id)?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct RecallQuery {
    query: String,
}

/// Show what the memory layer *would* put in front of the model.
///
/// Retrieval is otherwise invisible — the point of this endpoint is to make it
/// inspectable, so a surprising answer can be traced to what was recalled.
async fn preview_recall(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
    Json(body): Json<RecallQuery>,
) -> ApiResult<Json<serde_json::Value>> {
    let store = state.store.lock().unwrap();
    let assembled = ContextBuilder::new(&store, &state.embedder)
        .with_budget(Budget {
            total: 4096,
            reserve_for_reply: 1024,
            recent_messages: 12,
            max_retrieved: 6,
        })
        .build(id, &body.query)?;

    Ok(Json(serde_json::json!({
        "pinned": assembled.pinned.iter().map(|f| &f.text).collect::<Vec<_>>(),
        "retrieved": assembled.retrieved.iter().map(|h| serde_json::json!({
            "text": h.text,
            "seq": h.seq,
        })).collect::<Vec<_>>(),
        "recent": assembled.recent.len(),
        "elided": assembled.messages_elided,
        "tokens_used": assembled.tokens_used,
    })))
}

/// A tool result trimmed to something worth storing.
///
/// A search can return kilobytes per result; the conversation only needs
/// enough to redraw the card, and an unbounded copy would bloat every row.
pub(crate) fn bounded(detail: &serde_json::Value) -> serde_json::Value {
    const MAX_RESULTS: usize = 10;
    const MAX_FIELD: usize = 400;

    let clip = |v: &serde_json::Value| -> serde_json::Value {
        match v.as_str() {
            Some(s) if s.chars().count() > MAX_FIELD => {
                serde_json::Value::String(s.chars().take(MAX_FIELD).collect())
            }
            _ => v.clone(),
        }
    };

    if let Some(results) = detail.get("results").and_then(|r| r.as_array()) {
        let trimmed: Vec<serde_json::Value> = results
            .iter()
            .take(MAX_RESULTS)
            .map(|r| {
                let mut out = serde_json::Map::new();
                for key in ["title", "url", "snippet", "description"] {
                    if let Some(v) = r.get(key) {
                        out.insert(key.to_string(), clip(v));
                    }
                }
                serde_json::Value::Object(out)
            })
            .collect();
        return serde_json::json!({ "results": trimmed });
    }

    let text = serde_json::to_string(detail).unwrap_or_default();
    if text.len() > 4000 {
        return serde_json::json!({ "truncated": true });
    }
    detail.clone()
}

/// Serve one stored attachment.
async fn media_file(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
) -> Response {
    match crate::media::read(&state.paths, &name) {
        Some((bytes, mime)) => (
            [
                (axum::http::header::CONTENT_TYPE, mime),
                // Immutable: a stored attachment is never rewritten under the
                // same name, so the browser can keep it for the session.
                (axum::http::header::CACHE_CONTROL, "private, max-age=86400"),
            ],
            bytes,
        )
            .into_response(),
        None => (StatusCode::NOT_FOUND, "no such attachment").into_response(),
    }
}

/// Decode a browser `data:` URL, or bare base64, into image bytes.
///
/// Browsers hand back `data:image/png;base64,...` from a file read, so the
/// prefix is stripped rather than demanded — a caller posting raw base64
/// should work too.
pub fn decode_data_url(value: &str) -> Result<ozgent_core::ImageSource, String> {
    let (mime, payload) = match value.strip_prefix("data:") {
        Some(rest) => {
            let (meta, data) = rest
                .split_once(',')
                .ok_or_else(|| "malformed data URL: no comma".to_string())?;
            let mime = meta.split(';').next().unwrap_or("").to_string();
            (Some(mime).filter(|m| !m.is_empty()), data)
        }
        None => (None, value),
    };

    let bytes = ozgent_core::chat::b64::decode(payload.trim())
        .map_err(|_| "an attached image was not valid base64".to_string())?;
    if bytes.is_empty() {
        return Err("an attached image was empty".into());
    }
    Ok(ozgent_core::ImageSource::Bytes { bytes, mime })
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn a_callers_mistake_is_not_reported_as_the_servers() {
        // A client that retries on 500 would keep resending a request that
        // can never succeed, and a log full of 500s hides the ones that are
        // actually ozgent's fault.
        let refused = ApiError::bad_request("execute: expected allow, ask or deny");
        assert_eq!(refused.into_response().status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn an_internal_failure_still_reports_itself_as_one() {
        let broke: ApiError = anyhow::anyhow!("the database is on fire").into();
        assert_eq!(broke.into_response().status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn an_unparseable_rule_is_a_bad_request() {
        // The path a settings page hits by sending a value ozgent does not
        // know, which is a typo in a request rather than a fault here.
        let err: ApiError = "maybe"
            .parse::<ozgent_core::Rule>()
            .map(|_| ())
            .map_err(|e| ApiError::bad_request(format!("execute: {e}")))
            .unwrap_err();
        assert_eq!(err.into_response().status(), StatusCode::BAD_REQUEST);
    }
}

#[cfg(test)]
mod asset_tests {
    /// Run the client's own markdown tests.
    ///
    /// The renderer is JavaScript, so its tests are too; this puts them in
    /// `cargo test` where the rest of the guards live. Skipped rather than
    /// failed when node is absent, since node is not a build requirement —
    /// the assets ship as source.
    #[test]
    fn markdown_renders_what_models_write() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");
        let run = std::process::Command::new("node")
            .arg("markdown.test.mjs")
            .current_dir(dir)
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
}

#[cfg(test)]
mod decimal_tests {
    #[test]
    fn a_stored_setting_reads_back_as_it_was_typed() {
        assert_eq!(super::decimal(0.8).to_string(), "0.8");
        assert_eq!(super::decimal(0.95).to_string(), "0.95");
        assert_eq!(super::decimal(1.1).to_string(), "1.1");
    }

    #[test]
    fn a_settings_object_drops_nulls_and_float_noise() {
        let v = serde_json::json!({ "temperature": 0.20000000298023224f64, "seed": null, "top_k": 20 });
        assert_eq!(super::tidy(v), serde_json::json!({ "temperature": 0.2, "top_k": 20 }));
    }
}

#[cfg(test)]
mod override_tests {
    use ozgent_core::Options;

    #[test]
    fn a_setting_that_repeats_the_default_is_not_an_override() {
        // What pinned a 4B to a 107,008-token window: the value was written
        // as an override, and from then on every improvement to how the
        // window is chosen passed that model by.
        let inherited = Options { context_length: Some(65_536), ..Default::default() };
        let asked = Options {
            context_length: Some(65_536),
            temperature: Some(0.2),
            ..Default::default()
        };
        let kept = super::without_inherited(asked, &inherited).unwrap();
        assert_eq!(kept.context_length, None, "the window follows the default again");
        assert_eq!(kept.temperature, Some(0.2), "a real change is still an override");
    }

    #[test]
    fn a_setting_the_defaults_say_nothing_about_is_kept() {
        let kept = super::without_inherited(
            Options { context_length: Some(8192), ..Default::default() },
            &Options::default(),
        )
        .unwrap();
        assert_eq!(kept.context_length, Some(8192));
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    fn secrets_are_masked_going_out_and_kept_coming_back() {
        let current = serde_json::json!({
            "tools": { "config": { "web_search": { "tavily": { "api_key": "tvly-real" } } } },
            "ui": { "markdown": true },
        });
        let mut shown = current.clone();
        redact(&mut shown);
        assert_eq!(shown["tools"]["config"]["web_search"]["tavily"]["api_key"], REDACTED);
        assert!(!shown.to_string().contains("tvly-real"));
        // The page sends the mask back unchanged, with its own edit.
        shown["ui"]["markdown"] = false.into();
        unredact(&mut shown, &current);
        assert_eq!(shown["tools"]["config"]["web_search"]["tavily"]["api_key"], "tvly-real");
        assert_eq!(shown["ui"]["markdown"], false);
        // A new value is a new value.
        let mut replaced = current.clone();
        replaced["tools"]["config"]["web_search"]["tavily"]["api_key"] = "tvly-new".into();
        unredact(&mut replaced, &current);
        assert_eq!(replaced["tools"]["config"]["web_search"]["tavily"]["api_key"], "tvly-new");
    }
}

// ---------------------------------------------------------------- styles

/// Every response style, built-in and custom.
async fn list_styles(AxumState(state): AxumState<State>) -> Json<serde_json::Value> {
    let custom = state.config.lock().unwrap().styles.clone();
    Json(serde_json::json!({ "styles": ozgent_core::styles::all(&custom) }))
}

#[derive(Deserialize)]
struct StyleBody {
    #[serde(default)]
    title: String,
    prompt: String,
}

/// Create or change a custom style.
async fn save_style(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
    Json(body): Json<StyleBody>,
) -> ApiResult<StatusCode> {
    let name = ozgent_core::styles::normalise_name(&name);
    ozgent_core::styles::valid_custom_name(&name).map_err(ApiError::bad_request)?;
    let prompt = body.prompt.trim().to_string();
    if prompt.is_empty() || prompt.chars().count() > 4000 {
        return Err(ApiError::bad_request("a style's instruction is 1 to 4000 characters"));
    }
    let mut config = state.config.lock().unwrap();
    config.styles.insert(name, ozgent_core::styles::CustomStyle { title: body.title.trim().to_string(), prompt });
    config.save(&state.paths)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Remove a custom style, and stop any model using it.
async fn delete_style(AxumState(state): AxumState<State>, Path(name): Path<String>) -> ApiResult<StatusCode> {
    let name = ozgent_core::styles::normalise_name(&name);
    let mut config = state.config.lock().unwrap();
    if config.styles.remove(&name).is_none() {
        return Err(ApiError::not_found(format!("no custom style called {name:?}")));
    }
    if config.defaults.style.as_deref() == Some(name.as_str()) {
        config.defaults.style = None;
    }
    for options in config.models.values_mut() {
        if options.style.as_deref() == Some(name.as_str()) {
            options.style = None;
        }
    }
    config.save(&state.paths)?;
    Ok(StatusCode::NO_CONTENT)
}
