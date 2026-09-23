//! The admin page's security controls: who may connect, API keys, and what
//! tools may touch.
//!
//! All of it lives behind the admin password (the routes are added to the
//! gated router in `admin.rs`), because each setting here can open the
//! machine wider. None of it is reachable through `/api/settings`.

use axum::extract::{Path, Request, State as AxumState};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use ozgent_core::access::{AccessConfig, ApiKeyEntry, NetMode, Scope};
use serde::Deserialize;

use crate::admin::{fail, save};
use crate::state::State;

fn client_of(request: &Request) -> Option<std::net::IpAddr> {
    request.extensions().get::<crate::access::ClientIp>().map(|c| c.0)
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// What the page shows. Keys without their digests: a digest is not the key,
/// but there is no reason for it to leave the server.
fn view(access: &AccessConfig, you: Option<std::net::IpAddr>) -> serde_json::Value {
    serde_json::json!({
        "mode": access.mode,
        "allow": access.allow,
        "deny": access.deny,
        "trusted_proxies": access.trusted_proxies,
        "hosts": access.hosts,
        "local_api_open": access.local_api_open,
        "requests_per_minute": access.requests_per_minute,
        "max_connections_per_address": access.max_connections_per_address,
        "max_body_mb": access.max_body_mb,
        "max_auth_failures": access.max_auth_failures,
        "lockout_minutes": access.lockout_minutes,
        "keys": access.keys.iter().map(|k| serde_json::json!({
            "id": k.id, "name": k.name, "scopes": k.scopes, "created": k.created, "disabled": k.disabled,
        })).collect::<Vec<_>>(),
        "scopes": Scope::ALL.iter().map(|s| serde_json::json!({ "name": s, "describes": s.describe() })).collect::<Vec<_>>(),
        "you": you.map(|ip| ip.to_string()),
    })
}

pub(crate) async fn access_view(AxumState(state): AxumState<State>, request: Request) -> Response {
    let access = state.config.lock().unwrap_or_else(|e| e.into_inner()).web.access.clone();
    Json(view(&access, client_of(&request))).into_response()
}

/// Everything but the keys, which have routes of their own. Every field is
/// optional; what is sent is changed.
#[derive(Deserialize)]
pub(crate) struct AccessUpdate {
    mode: Option<NetMode>,
    allow: Option<Vec<String>>,
    deny: Option<Vec<String>>,
    trusted_proxies: Option<Vec<String>>,
    hosts: Option<Vec<String>>,
    local_api_open: Option<bool>,
    requests_per_minute: Option<u32>,
    max_connections_per_address: Option<u32>,
    max_body_mb: Option<u32>,
    max_auth_failures: Option<u32>,
    lockout_minutes: Option<u32>,
    /// Save even if it shuts out the address saving it.
    #[serde(default)]
    force: bool,
}

fn clean(list: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = list.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    out.dedup();
    out
}

pub(crate) async fn access_update(AxumState(state): AxumState<State>, request: Request) -> Response {
    let you = client_of(&request);
    let bytes = match axum::body::to_bytes(request.into_body(), 1 << 20).await {
        Ok(b) => b,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let body: AccessUpdate = match serde_json::from_slice(&bytes) {
        Ok(b) => b,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    let mut next = state.config.lock().unwrap_or_else(|e| e.into_inner()).web.access.clone();
    if let Some(v) = body.mode { next.mode = v }
    if let Some(v) = body.allow { next.allow = clean(v) }
    if let Some(v) = body.deny { next.deny = clean(v) }
    if let Some(v) = body.trusted_proxies { next.trusted_proxies = clean(v) }
    if let Some(v) = body.hosts {
        next.hosts = clean(v).into_iter().map(|h| h.to_ascii_lowercase()).collect();
    }
    if let Some(v) = body.local_api_open { next.local_api_open = v }
    if let Some(v) = body.requests_per_minute { next.requests_per_minute = v }
    if let Some(v) = body.max_connections_per_address { next.max_connections_per_address = v }
    if let Some(v) = body.max_body_mb { next.max_body_mb = v.clamp(1, 4096) }
    if let Some(v) = body.max_auth_failures { next.max_auth_failures = v }
    if let Some(v) = body.lockout_minutes { next.lockout_minutes = v.max(1) }

    // A rule that does not parse would be silently ignored when matching —
    // a deny list with a typo protecting nothing — so it is refused here.
    let bad = next.invalid_rules();
    if !bad.is_empty() {
        return fail(StatusCode::BAD_REQUEST, format!("not an address or range: {}", bad.join("; ")));
    }
    for h in &next.hosts {
        if h.contains(['/', ':', ' ']) || h.starts_with('.') {
            return fail(StatusCode::BAD_REQUEST, format!("{h:?} is not a host name"));
        }
    }
    if let Some(ip) = you {
        if next.admits(ip).is_err() && !body.force {
            return fail(
                StatusCode::CONFLICT,
                format!("this would refuse your own address ({ip}) and end this session. Send again with force to save it anyway"),
            );
        }
    }
    let rpm = next.requests_per_minute;
    let body_changed = body.max_body_mb.is_some();
    let saved = save(&state, move |c| c.web.access = next);
    if body_changed {
        tracing::info!("the request size limit takes effect when the server restarts");
    }
    tracing::info!("access rules changed ({rpm} requests/minute per address)");
    saved
}

#[derive(Deserialize)]
pub(crate) struct NewKey {
    name: String,
    scopes: Vec<Scope>,
}

/// Make a key. The key itself is in this response and nowhere else, ever.
pub(crate) async fn key_create(AxumState(state): AxumState<State>, Json(body): Json<NewKey>) -> Response {
    let name = body.name.trim().to_string();
    if name.is_empty() || name.chars().count() > 80 {
        return fail(StatusCode::BAD_REQUEST, "give the key a name of 1 to 80 characters");
    }
    let mut scopes = body.scopes;
    scopes.sort();
    scopes.dedup();
    if scopes.is_empty() {
        return fail(StatusCode::BAD_REQUEST, "a key needs at least one scope");
    }
    let (key, id, hash) = ozgent_core::secret::new_api_key();
    let entry = ApiKeyEntry { id: id.clone(), name: name.clone(), hash, scopes: scopes.clone(), created: now(), disabled: false };
    let saved = save(&state, move |c| c.web.access.keys.push(entry));
    if saved.status() != StatusCode::OK {
        return saved;
    }
    tracing::info!("API key {id} ({name}) created with {scopes:?}");
    Json(serde_json::json!({ "key": key, "id": id, "name": name, "scopes": scopes })).into_response()
}

#[derive(Deserialize)]
pub(crate) struct KeyChange {
    name: Option<String>,
    scopes: Option<Vec<Scope>>,
    disabled: Option<bool>,
}

pub(crate) async fn key_update(
    AxumState(state): AxumState<State>,
    Path(id): Path<String>,
    Json(body): Json<KeyChange>,
) -> Response {
    let known = state.config.lock().unwrap_or_else(|e| e.into_inner()).web.access.keys.iter().any(|k| k.id == id);
    if !known {
        return fail(StatusCode::NOT_FOUND, format!("no key {id}"));
    }
    if body.scopes.as_ref().is_some_and(|s| s.is_empty()) {
        return fail(StatusCode::BAD_REQUEST, "a key needs at least one scope; disable it instead");
    }
    save(&state, move |c| {
        if let Some(k) = c.web.access.keys.iter_mut().find(|k| k.id == id) {
            if let Some(n) = body.name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()) {
                k.name = n;
            }
            if let Some(mut s) = body.scopes {
                s.sort();
                s.dedup();
                k.scopes = s;
            }
            if let Some(d) = body.disabled {
                k.disabled = d;
            }
        }
    })
}

pub(crate) async fn key_delete(AxumState(state): AxumState<State>, Path(id): Path<String>) -> Response {
    let known = state.config.lock().unwrap_or_else(|e| e.into_inner()).web.access.keys.iter().any(|k| k.id == id);
    if !known {
        return fail(StatusCode::NOT_FOUND, format!("no key {id}"));
    }
    tracing::info!("API key {id} revoked");
    save(&state, move |c| c.web.access.keys.retain(|k| k.id != id))
}

// ------------------------------------------------------------ the sandbox

/// What `[tools.config.permissions]` says, with defaults filled in, so the
/// page shows what is actually in force rather than what happens to be
/// written down.
fn sandbox_of(config: &ozgent_core::Config) -> serde_json::Value {
    let p = config
        .tools
        .config
        .get("permissions")
        .and_then(|v| serde_json::to_value(v).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let flag = |k: &str, d: bool| p.get(k).and_then(|v| v.as_bool()).unwrap_or(d);
    let list = |k: &str| p.get(k).and_then(|v| v.as_array()).cloned().unwrap_or_default();
    serde_json::json!({
        "root": p.get("root").and_then(|v| v.as_str()),
        "write": flag("write", false),
        "shell": flag("shell", false),
        "shell_allow": list("shell_allow"),
        "shell_network": flag("shell_network", false),
        "network": flag("network", false),
        "network_allow": list("network_allow"),
        "network_private": flag("network_private", false),
        "allow_sensitive": flag("allow_sensitive", false),
        // Shown, not editable: each names a program that runs with this
        // user's rights. Changed in the file, deliberately.
        "python": config.tools.python,
        "extra_paths": config.tools.extra_paths,
        "mcp_servers": config.mcp.servers.keys().collect::<Vec<_>>(),
    })
}

pub(crate) async fn sandbox_view(AxumState(state): AxumState<State>) -> Response {
    let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
    Json(sandbox_of(&config)).into_response()
}

#[derive(Deserialize)]
pub(crate) struct SandboxUpdate {
    #[serde(default, with = "crate::admin_access::double_option_string")]
    root: Option<Option<String>>,
    write: Option<bool>,
    shell: Option<bool>,
    shell_allow: Option<Vec<String>>,
    shell_network: Option<bool>,
    network: Option<bool>,
    network_allow: Option<Vec<String>>,
    network_private: Option<bool>,
    allow_sensitive: Option<bool>,
}

pub(crate) mod double_option_string {
    use serde::{Deserialize, Deserializer};
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Option<String>>, D::Error> {
        Ok(Some(Option::deserialize(d)?))
    }
}

pub(crate) async fn sandbox_update(AxumState(state): AxumState<State>, Json(body): Json<SandboxUpdate>) -> Response {
    let response = sandbox_save(&state, body);
    if response.status() == StatusCode::OK {
        // The worker reads these when its interpreter starts.
        let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
        crate::api::restart_tools(&state, config).await;
    }
    response
}

fn sandbox_save(state: &State, body: SandboxUpdate) -> Response {
    // A program name, not a command: the allowlist matches the first word.
    if let Some(list) = &body.shell_allow {
        for p in list {
            let p = p.trim();
            if p.is_empty() || p.contains(char::is_whitespace) || p.contains('/') {
                return fail(StatusCode::BAD_REQUEST, format!("{p:?} is not a program name (no paths, no arguments)"));
            }
        }
    }
    if let Some(Some(root)) = &body.root {
        let path = std::path::Path::new(root.trim());
        if !path.is_absolute() || !path.is_dir() {
            return fail(StatusCode::BAD_REQUEST, format!("{root:?} is not an existing absolute directory"));
        }
        let home = state.paths.root();
        if path.starts_with(home) {
            return fail(StatusCode::BAD_REQUEST, "tools may not be rooted inside ozgent's own directory");
        }
    }
    save(state, move |c| {
        let entry = c.tools.config.entry("permissions".into()).or_insert_with(|| toml::Value::Table(Default::default()));
        if !entry.is_table() {
            *entry = toml::Value::Table(Default::default());
        }
        let t = entry.as_table_mut().expect("a table");
        let set_bool = |t: &mut toml::Table, k: &str, v: Option<bool>| {
            if let Some(v) = v {
                t.insert(k.into(), toml::Value::Boolean(v));
            }
        };
        set_bool(t, "write", body.write);
        set_bool(t, "shell", body.shell);
        set_bool(t, "shell_network", body.shell_network);
        set_bool(t, "network", body.network);
        set_bool(t, "network_private", body.network_private);
        set_bool(t, "allow_sensitive", body.allow_sensitive);
        let list = |v: Vec<String>| {
            toml::Value::Array(
                v.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).map(toml::Value::String).collect(),
            )
        };
        if let Some(v) = body.shell_allow {
            t.insert("shell_allow".into(), list(v));
        }
        if let Some(v) = body.network_allow {
            t.insert("network_allow".into(), list(v.into_iter().map(|h| h.to_ascii_lowercase()).collect()));
        }
        match body.root {
            Some(Some(r)) => {
                t.insert("root".into(), toml::Value::String(r.trim().to_string()));
            }
            Some(None) => {
                t.remove("root");
            }
            None => {}
        }
    })
}

// ------------------------------------------------------------ embeddings

/// The embedding settings and what the embedding model is doing.
pub(crate) async fn embedding_view(AxumState(state): AxumState<State>) -> Response {
    let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let installed: Vec<String> = ozgent_core::installed(&state.paths)
        .into_iter()
        .filter(|m| ozgent_llama::layout::is_embedding(&m.manifest.primary_weights(&m.dir)))
        .map(|m| m.model.to_string())
        .collect();
    let status = crate::worker::embed_status();
    let chosen = state.worker.embedding_model();
    let coverage = Some(status.dimensions)
        .filter(|d| *d > 0)
        .and_then(|d| state.store.lock().ok()?.embedding_coverage(d).ok());
    Json(serde_json::json!({
        "enabled": config.embedding.enabled,
        "model": config.embedding.model,
        "device": config.embedding.device,
        "max_tokens": config.embedding.max_tokens,
        "installed": installed,
        "chosen": chosen,
        "status": status,
        "coverage": coverage.map(|(done, all)| serde_json::json!({ "done": done, "all": all })),
        "backfilling": crate::memory::backfilling(),
    }))
    .into_response()
}

#[derive(Deserialize)]
pub(crate) struct EmbeddingUpdate {
    enabled: Option<bool>,
    /// `null` for automatic.
    #[serde(default, with = "crate::admin_access::double_option_string")]
    model: Option<Option<String>>,
    device: Option<ozgent_core::config::EmbedDevice>,
    max_tokens: Option<u32>,
}

pub(crate) async fn embedding_update(AxumState(state): AxumState<State>, Json(body): Json<EmbeddingUpdate>) -> Response {
    if let Some(Some(name)) = &body.model {
        match ozgent_core::resolve(&state.paths, name) {
            Ok(found) if ozgent_llama::layout::is_embedding(&found.manifest.primary_weights(&found.dir)) => {}
            Ok(_) => return fail(StatusCode::BAD_REQUEST, format!("{name} is a chat model, not an embedding model")),
            Err(e) => return fail(StatusCode::BAD_REQUEST, e),
        }
    }
    if body.max_tokens.is_some_and(|n| n != 0 && !(64..=1_048_576).contains(&n)) {
        return fail(StatusCode::BAD_REQUEST, "max_tokens is 0 (the model's own window) or at least 64");
    }
    let saved = save(&state, move |c| {
        if let Some(v) = body.enabled { c.embedding.enabled = v }
        if let Some(v) = body.model { c.embedding.model = v.map(|m| m.trim().to_string()).filter(|m| !m.is_empty()) }
        if let Some(v) = body.device { c.embedding.device = v }
        if let Some(v) = body.max_tokens { c.embedding.max_tokens = v }
    });
    if saved.status() == StatusCode::OK {
        // Loaded again under the new settings on its next use, and every
        // stored message brought up to the model now chosen.
        state.worker.reload_embedder();
        crate::memory::backfill(&state);
    }
    saved
}

pub(crate) async fn embedding_backfill(AxumState(state): AxumState<State>) -> Response {
    crate::memory::backfill(&state);
    Json(serde_json::json!({ "ok": true })).into_response()
}

// ------------------------------------------------------------ server

/// Settings for how the server holds models and runs tools.
pub(crate) async fn server_view(AxumState(state): AxumState<State>) -> Response {
    let c = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
    Json(serde_json::json!({
        "idle_unload_minutes": c.web.idle_unload_minutes,
        "parallel": c.web.parallel,
        "tool_timeout_seconds": c.tools.timeout_seconds,
        "max_calls_per_turn": c.tools.max_calls_per_turn,
        "handoff": c.tools.handoff,
    }))
    .into_response()
}

#[derive(Deserialize)]
pub(crate) struct ServerUpdate {
    idle_unload_minutes: Option<u64>,
    parallel: Option<u32>,
    tool_timeout_seconds: Option<u64>,
    max_calls_per_turn: Option<u32>,
    handoff: Option<bool>,
}

pub(crate) async fn server_update(AxumState(state): AxumState<State>, Json(body): Json<ServerUpdate>) -> Response {
    if body.parallel.is_some_and(|n| !(1..=16).contains(&n)) {
        return fail(StatusCode::BAD_REQUEST, "conversations at once must be 1 to 16");
    }
    if body.tool_timeout_seconds.is_some_and(|n| !(1..=3600).contains(&n)) {
        return fail(StatusCode::BAD_REQUEST, "a tool timeout must be 1 to 3600 seconds");
    }
    if body.max_calls_per_turn.is_some_and(|n| !(1..=64).contains(&n)) {
        return fail(StatusCode::BAD_REQUEST, "tool calls per turn must be 1 to 64");
    }
    let timeout_changed = body.tool_timeout_seconds.is_some();
    let response = save(&state, move |c| {
        if let Some(v) = body.idle_unload_minutes { c.web.idle_unload_minutes = v }
        if let Some(v) = body.parallel { c.web.parallel = v }
        if let Some(v) = body.tool_timeout_seconds { c.tools.timeout_seconds = v }
        if let Some(v) = body.max_calls_per_turn { c.tools.max_calls_per_turn = v }
        if let Some(v) = body.handoff { c.tools.handoff = v }
    });
    if timeout_changed && response.status() == StatusCode::OK {
        // The tool host reads its timeout when it starts.
        let config = state.config.lock().unwrap_or_else(|e| e.into_inner()).clone();
        crate::api::restart_tools(&state, config).await;
    }
    response
}
