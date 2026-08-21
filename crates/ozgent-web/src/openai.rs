//! An OpenAI-compatible HTTP API.
//!
//! The point is that existing clients work unchanged: set the base URL, use any
//! name `ozgent list` shows as the model, and the SDK you already have does the
//! rest. So the shapes here follow OpenAI's, including the parts that are
//! awkward — `choices` as an array of one, `finish_reason` spelled their way,
//! and `[DONE]` terminating a stream.
//!
//! Where ozgent knows more than the OpenAI schema carries, it is added rather
//! than substituted: `usage` gains a `timings` block with real tokens/second,
//! and reasoning is exposed as `reasoning_content`, the field name the
//! ecosystem settled on for models that think. A client that ignores both still
//! works.

use axum::extract::State as AxumState;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use ozgent_core::{Message, ThinkingMode};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

use crate::state::State;
use crate::worker::{Event, Request};

/// Bearer token required on every request, when one is configured.
#[derive(Clone, Default)]
pub struct ApiKey(pub Option<String>);

pub fn router(state: State, key: ApiKey) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/models/{model}", get(model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/health", get(health))
        .layer(axum::Extension(key))
        .with_state(state)
}

// ------------------------------------------------------------------ errors

/// OpenAI's error envelope, so a client's own error handling still fires.
struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, kind: "invalid_request_error", message: message.into() }
    }
    fn not_found(message: impl Into<String>) -> Self {
        Self { status: StatusCode::NOT_FOUND, kind: "not_found_error", message: message.into() }
    }
    fn internal(message: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, kind: "api_error", message: message.into() }
    }
    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            kind: "invalid_request_error",
            message: "missing or invalid API key".into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": { "message": self.message, "type": self.kind, "code": serde_json::Value::Null }
            })),
        )
            .into_response()
    }
}

type ApiResult<T> = Result<T, ApiError>;

fn authorise(headers: &HeaderMap, key: &ApiKey) -> Result<(), ApiError> {
    let Some(expected) = key.0.as_deref() else { return Ok(()) };
    let given = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    // Constant-time enough for a local key: compare lengths first, then bytes.
    if given.len() == expected.len() && given.bytes().zip(expected.bytes()).all(|(a, b)| a == b) {
        Ok(())
    } else {
        Err(ApiError::unauthorized())
    }
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn id(prefix: &str) -> String {
    // Enough entropy to correlate a request in a log without a uuid dependency.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 + d.as_secs() * 1_000_000_000)
        .unwrap_or(0);
    format!("{prefix}-{nanos:x}")
}

// ------------------------------------------------------------------ models

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: i64,
    owned_by: &'static str,
    /// Not in OpenAI's schema, but the first thing anyone actually wants.
    #[serde(skip_serializing_if = "Option::is_none")]
    quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
    context_length: u32,
    capabilities: Vec<&'static str>,
}

fn describe(state: &State, m: &ozgent_core::registry::Installed) -> ModelObject {
    let resolved = state
        .config
        .lock()
        .unwrap()
        .options_for(&m.model.to_string())
        .merge(&m.manifest.defaults)
        .resolve();

    let mut capabilities = vec!["completion", "tools"];
    if m.manifest.supports_vision() {
        capabilities.push("vision");
    }
    ModelObject {
        id: m.manifest.alias.clone().unwrap_or_else(|| m.model.to_string()),
        object: "model",
        created: 0,
        owned_by: "ozgent",
        quantization: m.manifest.quantization.clone(),
        size_bytes: m.manifest.size_bytes,
        context_length: resolved.context_length,
        capabilities,
    }
}

async fn models(
    AxumState(state): AxumState<State>,
    axum::Extension(key): axum::Extension<ApiKey>,
    headers: HeaderMap,
) -> ApiResult<Json<serde_json::Value>> {
    authorise(&headers, &key)?;
    let data: Vec<ModelObject> = ozgent_core::installed(&state.paths)
        .iter()
        .map(|m| describe(&state, m))
        .collect();
    Ok(Json(serde_json::json!({ "object": "list", "data": data })))
}

async fn model(
    AxumState(state): AxumState<State>,
    axum::Extension(key): axum::Extension<ApiKey>,
    headers: HeaderMap,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> ApiResult<Json<ModelObject>> {
    authorise(&headers, &key)?;
    let found = ozgent_core::resolve(&state.paths, &name)
        .map_err(|_| ApiError::not_found(format!("no model {name:?}")))?;
    Ok(Json(describe(&state, &found)))
}

async fn health(AxumState(state): AxumState<State>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "models": ozgent_core::installed(&state.paths).len(),
    }))
}

// -------------------------------------------------------------- the request

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// OpenAI's newer name for the same thing.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub stop: Option<StopField>,
    /// `true`/`false`, or `"auto"`/`"on"`/`"off"` for models that reason.
    #[serde(default)]
    pub reasoning: Option<serde_json::Value>,
    /// Off by default so a plain OpenAI client never gets a surprise tool call.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// ozgent extension: use the server's own Python tools.
    #[serde(default)]
    pub ozgent_tools: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// A string, or OpenAI's content-part array.
    #[serde(default)]
    pub content: Option<serde_json::Value>,
}

impl ChatMessage {
    /// Flatten OpenAI's content shapes into plain text.
    ///
    /// Both a bare string and the `[{type:"text",text:...}]` array are valid,
    /// and clients differ, so both must work.
    pub fn text(&self) -> String {
        match &self.content {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| {
                    p.get("text")
                        .and_then(|t| t.as_str())
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }

    /// Images referenced by this message, in OpenAI's `image_url` form.
    ///
    /// Both a `data:` URL and a plain http(s) URL are legal there; only the
    /// former can be served without fetching, so a remote URL is reported
    /// rather than quietly ignored.
    pub fn images(&self) -> Result<Vec<ozgent_core::ImageSource>, String> {
        let Some(serde_json::Value::Array(parts)) = &self.content else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for part in parts {
            if part.get("type").and_then(|t| t.as_str()) != Some("image_url") {
                continue;
            }
            let url = part
                .get("image_url")
                .and_then(|u| u.get("url"))
                .and_then(|u| u.as_str())
                .ok_or_else(|| "image_url is missing its url".to_string())?;
            if url.starts_with("data:") || !url.contains("://") {
                out.push(crate::api::decode_data_url(url)?);
            } else {
                return Err(format!(
                    "{url} must be inlined as a data: URL; this server does not fetch images"
                ));
            }
        }
        Ok(out)
    }
}

/// Turn a request's sampling fields into an ozgent option layer.
///
/// Only what the caller actually set: an absent field must inherit the server's
/// configuration rather than silently reset it to an OpenAI default.
pub fn options_from(request: &ChatRequest) -> ozgent_core::Options {
    ozgent_core::Options {
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        min_p: request.min_p,
        seed: request.seed,
        // OpenAI's presence/frequency penalties are a different formulation to
        // llama.cpp's repeat penalty; `repeat_penalty` is accepted directly and
        // the OpenAI ones map on as the closest equivalent.
        repeat_penalty: request.repeat_penalty.or_else(|| {
            request
                .presence_penalty
                .or(request.frequency_penalty)
                .map(|p| 1.0 + p.clamp(-1.0, 2.0) / 2.0)
        }),
        max_tokens: request.max_tokens.or(request.max_completion_tokens),
        ..Default::default()
    }
}

pub fn thinking_from(value: Option<&serde_json::Value>) -> Option<ThinkingMode> {
    match value {
        Some(serde_json::Value::Bool(true)) => Some(ThinkingMode::On),
        Some(serde_json::Value::Bool(false)) => Some(ThinkingMode::Off),
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "on" | "always" | "high" | "medium" | "low" => Some(ThinkingMode::On),
            "off" | "none" | "never" => Some(ThinkingMode::Off),
            "auto" => Some(ThinkingMode::Auto),
            _ => None,
        },
        Some(serde_json::Value::Object(o)) => {
            thinking_from(o.get("effort").or_else(|| o.get("enabled")))
        }
        _ => None,
    }
}

pub fn to_messages(request: &ChatRequest) -> Vec<Message> {
    request
        .messages
        .iter()
        .map(|m| match m.role.as_str() {
            "system" | "developer" => Message::system(m.text()),
            "assistant" => Message::assistant(m.text()),
            "tool" => Message::tool_result(String::new(), m.text()),
            _ => Message::user(m.text()),
        })
        .collect()
}

/// OpenAI's spelling of why generation ended.
pub fn finish_reason(stop: &str) -> &'static str {
    match stop {
        "EndOfText" => "stop",
        "TokenLimit" | "ContextFull" => "length",
        "Cancelled" => "stop",
        _ => "stop",
    }
}

// ------------------------------------------------------------- completions

async fn chat_completions(
    AxumState(state): AxumState<State>,
    axum::Extension(key): axum::Extension<ApiKey>,
    headers: HeaderMap,
    Json(request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    authorise(&headers, &key)?;
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let found = ozgent_core::resolve(&state.paths, &request.model)
        .map_err(|_| ApiError::not_found(format!("no model {:?}", request.model)))?;

    let mut images = Vec::new();
    for message in &request.messages {
        images.extend(message.images().map_err(ApiError::bad_request)?);
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    state
        .worker
        .submit(Request {
            model: found.model.to_string(),
            messages: to_messages(&request),
            thinking: thinking_from(request.reasoning.as_ref()),
            max_tokens: request.max_tokens.or(request.max_completion_tokens),
            tools_enabled: request.ozgent_tools.unwrap_or(false),
            overrides: Some(options_from(&request)),
            images,
            out: tx,
        })
        .map_err(ApiError::internal)?;

    let model_id = found.model.to_string();
    if request.stream {
        Ok(Sse::new(stream_chunks(rx, model_id)).into_response())
    } else {
        Ok(Json(collect(rx, model_id).await?).into_response())
    }
}

/// `/v1/completions`, for clients that predate the chat API.
async fn completions(
    state: AxumState<State>,
    key: axum::Extension<ApiKey>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let prompt = body
        .get("prompt")
        .and_then(|p| p.as_str())
        .ok_or_else(|| ApiError::bad_request("prompt is required"))?;
    let mut as_chat = body.clone();
    as_chat["messages"] = serde_json::json!([{ "role": "user", "content": prompt }]);
    let request: ChatRequest = serde_json::from_value(as_chat)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    chat_completions(state, key, headers, Json(request)).await
}

/// Everything ozgent measured about a turn.
#[derive(Serialize, Default)]
pub struct Timings {
    pub prompt_tokens: u32,
    pub prompt_ms: u64,
    pub prompt_tokens_per_second: f64,
    pub completion_tokens: u32,
    pub completion_ms: u64,
    pub tokens_per_second: f64,
    pub cached_prompt_tokens: usize,
}

/// Collect a whole turn into one OpenAI response object.
async fn collect(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    model: String,
) -> Result<serde_json::Value, ApiError> {
    let mut answer = String::new();
    let mut reasoning = String::new();
    let mut calls: Vec<serde_json::Value> = Vec::new();
    let mut timings = Timings::default();
    let mut stop = "stop";

    while let Some(event) = rx.recv().await {
        match event {
            Event::Answer { text } => answer.push_str(&text),
            Event::Thinking { text } => reasoning.push_str(&text),
            Event::ToolCall { name, arguments } => calls.push(serde_json::json!({
                "id": id("call"),
                "type": "function",
                "function": { "name": name, "arguments": arguments.to_string() },
            })),
            Event::Done { generated, tokens_per_second, reused, stop: reason, prompt } => {
                timings.completion_tokens = generated;
                timings.tokens_per_second = tokens_per_second;
                timings.cached_prompt_tokens = reused;
                timings.prompt_tokens = prompt;
                timings.completion_ms = if tokens_per_second > 0.0 {
                    (generated as f64 / tokens_per_second * 1000.0) as u64
                } else {
                    0
                };
                stop = finish_reason(&reason);
            }
            Event::Error { message } => return Err(ApiError::internal(message)),
            Event::Ready { .. } | Event::ToolResult { .. } => {}
        }
    }

    let mut message = serde_json::json!({ "role": "assistant", "content": answer });
    if !reasoning.trim().is_empty() {
        message["reasoning_content"] = reasoning.trim().into();
    }
    if !calls.is_empty() {
        message["tool_calls"] = calls.into();
    }

    Ok(serde_json::json!({
        "id": id("chatcmpl"),
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{ "index": 0, "message": message, "finish_reason": stop }],
        "usage": {
            "prompt_tokens": timings.prompt_tokens,
            "completion_tokens": timings.completion_tokens,
            "total_tokens": timings.prompt_tokens + timings.completion_tokens,
            "timings": timings,
        },
    }))
}

/// Stream a turn as `chat.completion.chunk` events.
fn stream_chunks(
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    model: String,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let completion = id("chatcmpl");
    futures_util::stream::unfold(
        (rx, model, completion, false, false),
        |(mut rx, model, completion, opened, finished)| async move {
            if finished {
                return None;
            }
            let chunk = |delta: serde_json::Value, reason: Option<&str>| {
                serde_json::json!({
                    "id": completion,
                    "object": "chat.completion.chunk",
                    "created": now(),
                    "model": model,
                    "choices": [{ "index": 0, "delta": delta, "finish_reason": reason }],
                })
            };

            let Some(event) = rx.recv().await else {
                // The stream ended without a Done — close it properly anyway,
                // or a client waits forever.
                return Some((
                    Ok(SseEvent::default().data("[DONE]")),
                    (rx, model, completion, opened, true),
                ));
            };

            // OpenAI's first chunk announces the role and nothing else.
            let role = (!opened).then(|| serde_json::json!("assistant"));
            let mut delta = serde_json::Map::new();
            if let Some(role) = role {
                delta.insert("role".into(), role);
            }

            match event {
                Event::Answer { text } => {
                    delta.insert("content".into(), text.into());
                }
                Event::Thinking { text } => {
                    delta.insert("reasoning_content".into(), text.into());
                }
                Event::ToolCall { name, arguments } => {
                    delta.insert(
                        "tool_calls".into(),
                        serde_json::json!([{
                            "index": 0,
                            "id": id("call"),
                            "type": "function",
                            "function": { "name": name, "arguments": arguments.to_string() },
                        }]),
                    );
                }
                Event::Done { generated, tokens_per_second, reused, stop, prompt } => {
                    let usage = serde_json::json!({
                        "prompt_tokens": prompt,
                        "completion_tokens": generated,
                        "total_tokens": prompt + generated,
                        "timings": {
                            "completion_tokens": generated,
                            "tokens_per_second": tokens_per_second,
                            "cached_prompt_tokens": reused,
                            "prompt_tokens": prompt,
                        },
                    });
                    let mut final_chunk = chunk(serde_json::json!({}), Some(finish_reason(&stop)));
                    final_chunk["usage"] = usage;
                    let text = serde_json::to_string(&final_chunk).unwrap_or_default();
                    return Some((
                        Ok(SseEvent::default().data(text)),
                        (rx, model, completion, true, false),
                    ));
                }
                Event::Error { message } => {
                    let text = serde_json::to_string(&serde_json::json!({
                        "error": { "message": message, "type": "api_error" }
                    }))
                    .unwrap_or_default();
                    return Some((
                        Ok(SseEvent::default().data(text)),
                        (rx, model, completion, true, false),
                    ));
                }
                Event::Ready { .. } | Event::ToolResult { .. } => {}
            }

            if delta.is_empty() {
                // Nothing to say this time; keep the stream open.
                return Some((
                    Ok(SseEvent::default().comment("")),
                    (rx, model, completion, opened, false),
                ));
            }
            let text = serde_json::to_string(&chunk(serde_json::Value::Object(delta), None))
                .unwrap_or_default();
            Some((Ok(SseEvent::default().data(text)), (rx, model, completion, true, false)))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: serde_json::Value) -> ChatRequest {
        serde_json::from_value(json).expect("should deserialise")
    }

    #[test]
    fn both_openai_content_shapes_are_accepted() {
        // Clients disagree: some send a string, some the parts array.
        let r = request(serde_json::json!({
            "model": "m", "messages": [
                { "role": "user", "content": "plain" },
                { "role": "user", "content": [
                    { "type": "text", "text": "part one" },
                    { "type": "text", "text": "part two" }
                ]}
            ]
        }));
        assert_eq!(r.messages[0].text(), "plain");
        assert_eq!(r.messages[1].text(), "part one\npart two");
    }

    #[test]
    fn unset_sampling_fields_inherit_rather_than_reset() {
        // A request that names only temperature must not silently drop the
        // server's configured top_p back to a default.
        let r = request(serde_json::json!({
            "model": "m", "messages": [{"role":"user","content":"hi"}], "temperature": 0.2
        }));
        let opts = options_from(&r);
        assert_eq!(opts.temperature, Some(0.2));
        assert_eq!(opts.top_p, None, "unset means inherit");
        assert_eq!(opts.top_k, None);
    }

    #[test]
    fn either_spelling_of_the_token_limit_works() {
        let old = request(serde_json::json!({
            "model": "m", "messages": [{"role":"user","content":"hi"}], "max_tokens": 64
        }));
        let new = request(serde_json::json!({
            "model": "m", "messages": [{"role":"user","content":"hi"}], "max_completion_tokens": 64
        }));
        assert_eq!(options_from(&old).max_tokens, Some(64));
        assert_eq!(options_from(&new).max_tokens, Some(64));
    }

    #[test]
    fn reasoning_accepts_the_spellings_clients_actually_send() {
        assert_eq!(thinking_from(Some(&serde_json::json!(true))), Some(ThinkingMode::On));
        assert_eq!(thinking_from(Some(&serde_json::json!(false))), Some(ThinkingMode::Off));
        assert_eq!(thinking_from(Some(&serde_json::json!("auto"))), Some(ThinkingMode::Auto));
        assert_eq!(thinking_from(Some(&serde_json::json!("none"))), Some(ThinkingMode::Off));
        // OpenAI's own nested shape.
        assert_eq!(
            thinking_from(Some(&serde_json::json!({ "effort": "medium" }))),
            Some(ThinkingMode::On)
        );
        assert_eq!(thinking_from(None), None, "absent means inherit");
    }

    #[test]
    fn an_inlined_image_part_is_accepted() {
        let r = request(serde_json::json!({
            "model": "m", "messages": [{ "role": "user", "content": [
                { "type": "text", "text": "what is this?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,aGVsbG8=" } }
            ]}]
        }));
        assert_eq!(r.messages[0].text(), "what is this?", "text parts still flatten");
        let images = r.messages[0].images().expect("should decode");
        assert_eq!(images.len(), 1);
        match &images[0] {
            ozgent_core::ImageSource::Bytes { bytes, mime } => {
                assert_eq!(bytes, b"hello");
                assert_eq!(mime.as_deref(), Some("image/png"));
            }
            other => panic!("expected inline bytes, got {other:?}"),
        }
    }

    #[test]
    fn a_remote_image_url_is_refused_rather_than_ignored() {
        // Silently dropping it would leave the model answering about an image
        // it was never shown.
        let r = request(serde_json::json!({
            "model": "m", "messages": [{ "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
            ]}]
        }));
        let err = r.messages[0].images().expect_err("must not silently pass");
        assert!(err.contains("data:"), "{err}");
    }

    #[test]
    fn stop_reasons_use_openais_vocabulary() {
        // A client switches on these exact strings.
        assert_eq!(finish_reason("EndOfText"), "stop");
        assert_eq!(finish_reason("TokenLimit"), "length");
        assert_eq!(finish_reason("ContextFull"), "length");
    }

    #[test]
    fn tools_are_off_unless_asked_for() {
        // A plain OpenAI client must never receive a surprise tool call from
        // the server's own Python tools.
        let r = request(serde_json::json!({
            "model": "m", "messages": [{"role":"user","content":"hi"}]
        }));
        assert_eq!(r.ozgent_tools, None);
    }

    #[test]
    fn roles_map_onto_ozgents_own() {
        let r = request(serde_json::json!({
            "model": "m", "messages": [
                {"role":"system","content":"s"},
                {"role":"developer","content":"d"},
                {"role":"assistant","content":"a"},
                {"role":"user","content":"u"}
            ]
        }));
        let msgs = to_messages(&r);
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, ozgent_core::Role::System);
        assert_eq!(msgs[1].role, ozgent_core::Role::System, "developer is a system role");
        assert_eq!(msgs[2].role, ozgent_core::Role::Assistant);
        assert_eq!(msgs[3].role, ozgent_core::Role::User);
    }
}
