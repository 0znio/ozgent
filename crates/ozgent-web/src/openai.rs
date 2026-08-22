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
use ozgent_core::{Message, ThinkingMode, ToolSpec};
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
        .route("/v1/embeddings", post(embeddings))
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
    /// `"low"`, `"medium"` or `"high"`, spelled as OpenAI spells it.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Off by default so a plain OpenAI client never gets a surprise tool call.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// ozgent extension: use the server's own Python tools.
    #[serde(default)]
    pub ozgent_tools: Option<bool>,
    /// `{"type":"json_object"}`, or `{"type":"json_schema","json_schema":{...}}`.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// ozgent extension: which built-in tools this request may use, by name.
    ///
    /// Absent offers all of them. Naming a subset is how a caller keeps, say,
    /// `write_file` out of reach for a request that has no business writing —
    /// the model is never told the tool exists, so it cannot ask for it.
    /// An empty list offers none.
    #[serde(default)]
    pub native_tools: Option<Vec<String>>,
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
    /// Present on an assistant turn that called tools.
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    /// Present on a `tool` turn, naming the call it answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// The calls this assistant turn made, as ozgent records them.
    pub fn tool_calls(&self) -> Vec<ozgent_core::ToolCall> {
        let Some(serde_json::Value::Array(items)) = &self.tool_calls else {
            return Vec::new();
        };
        items
            .iter()
            .filter_map(|item| {
                let function = item.get("function")?;
                let name = function.get("name")?.as_str()?.to_string();
                // OpenAI sends arguments as a *string* of JSON, not an object.
                // Passing the string through would render the call with quoted
                // arguments, which the model reads as a different call than the
                // one it made.
                let arguments = match function.get("arguments") {
                    Some(serde_json::Value::String(raw)) => {
                        serde_json::from_str(raw).unwrap_or(serde_json::Value::Null)
                    }
                    Some(other) => other.clone(),
                    None => serde_json::Value::Null,
                };
                Some(ozgent_core::ToolCall {
                    id: item.get("id").and_then(|i| i.as_str()).unwrap_or_default().to_string(),
                    name,
                    arguments,
                })
            })
            .collect()
    }

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
/// The effort level a request asked for, if it named a valid one.
///
/// An unrecognised level is ignored rather than refused: it narrows how long
/// the model thinks and nothing else, so a client sending a level ozgent does
/// not know should still get an answer.
pub fn effort_from(value: Option<&String>) -> Option<ozgent_core::ReasoningEffort> {
    value.and_then(|v| v.parse().ok())
}

pub fn options_from(request: &ChatRequest) -> ozgent_core::Options {
    ozgent_core::Options {
        reasoning_effort: effort_from(request.reasoning_effort.as_ref()),
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
            "assistant" => {
                let mut message = Message::assistant(m.text());
                // An assistant turn that called a tool has little or no text —
                // the call *is* the content. Dropping it leaves the following
                // tool result answering a question the model cannot see it
                // asked, so the calls are carried back into the transcript.
                message.tool_calls = m.tool_calls();
                message
            }
            // The id ties the result to the call that asked for it. Without it
            // a turn with two outstanding calls cannot say which is which.
            "tool" => Message::tool_result(m.tool_call_id.clone().unwrap_or_default(), m.text()),
            _ => Message::user(m.text()),
        })
        .collect()
}

/// The grammar `response_format` asks for, if any.
///
/// `Err` carries a message for the caller: a `response_format` that cannot be
/// honoured has to be refused, because the alternative is returning prose to a
/// client that will try to parse it as JSON.
pub fn response_grammar(value: Option<&serde_json::Value>) -> Result<Option<String>, String> {
    let Some(format) = value else { return Ok(None) };
    match format.get("type").and_then(|t| t.as_str()) {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(ozgent_llama::grammar::json_object_grammar())),
        Some("json_schema") => {
            // OpenAI nests the schema one level down; some clients put it at the
            // top. Accepting only the documented spelling would reject requests
            // that are otherwise perfectly clear.
            let schema = format
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .or_else(|| format.get("schema"))
                .ok_or_else(|| {
                    "response_format json_schema needs a `schema`".to_string()
                })?;
            Ok(Some(ozgent_llama::grammar::schema_grammar(schema)))
        }
        Some(other) => Err(format!("unsupported response_format type {other:?}")),
    }
}

/// Tools the *caller* implements, taken from the request's `tools` array.
///
/// These are not ozgent's Python tools: the server has no code for them. They
/// are described to the model, and when the model calls one the turn stops and
/// the call is handed back for the caller to run — which is what OpenAI's
/// `finish_reason: "tool_calls"` means.
///
/// Entries that are not `type: "function"` name a capability this server
/// cannot provide (OpenAI's hosted tools, say). Offering them would let the
/// model call something that can never run, so they are skipped.
pub fn client_tools(value: Option<&serde_json::Value>) -> Vec<ToolSpec> {
    let Some(serde_json::Value::Array(items)) = value else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            // The nesting under `function` is the documented shape, but enough
            // clients send the inner object bare that both are worth accepting.
            let body = match item.get("function") {
                Some(f) => f,
                None => item,
            };
            if let Some(kind) = item.get("type").and_then(|t| t.as_str()) {
                if kind != "function" && item.get("function").is_some() {
                    return None;
                }
            }
            let name = body.get("name")?.as_str()?.trim();
            if name.is_empty() {
                return None;
            }
            // An absent `parameters` means a function that takes none; the
            // empty object schema says exactly that, and keeps the grammar
            // builder on its normal path.
            let input_schema = body
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} }));
            Some(ToolSpec {
                name: name.to_string(),
                description: body
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or_default()
                    .to_string(),
                input_schema,
                output_schema: None,
            })
        })
        .collect()
}

/// OpenAI's spelling of why generation ended.
pub fn finish_reason(stop: &str) -> &'static str {
    match stop {
        "EndOfText" => "stop",
        // The caller owns a tool the model called; it must run it and come
        // back, so this is not an ordinary stop.
        "ToolCalls" => "tool_calls",
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

    let response_grammar = response_grammar(request.response_format.as_ref())
        .map_err(ApiError::bad_request)?;

    // A schema grammar masks out every token that would break it, so the model
    // physically cannot emit a tool-call marker. Honouring both would leave the
    // caller's tools silently dead — the failure this whole area keeps
    // producing — so the conflict is refused instead.
    if response_grammar.is_some()
        && (request.tools.is_some() || request.ozgent_tools == Some(true)
            || request.native_tools.is_some())
    {
        return Err(ApiError::bad_request(
            "response_format and tools cannot be combined: a schema grammar makes a tool call \
             unrepresentable, so the tools would never be used",
        ));
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    state
        .worker
        .submit(Request {
            model: found.model.to_string(),
            messages: to_messages(&request),
            // A reasoning model opens with `<think>`, which no schema admits —
            // the grammar would mask the very first token it wants. Structured
            // output and visible reasoning are not compatible, and discovering
            // that as a stalled generation would be far worse than this.
            thinking: if response_grammar.is_some() {
                Some(ThinkingMode::Off)
            } else {
                thinking_from(request.reasoning.as_ref())
            },
            max_tokens: request.max_tokens.or(request.max_completion_tokens),
            // Naming the built-ins is itself a request to use them, so a
            // caller does not have to set two fields that mean one thing.
            tools_enabled: request.ozgent_tools.unwrap_or(request.native_tools.is_some()),
            native_tools: request.native_tools.clone(),
            client_tools: client_tools(request.tools.as_ref()),
            response_grammar,
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

/// `/v1/embeddings`.
///
/// Reports rather than improvises. Serving embeddings from a chat model by
/// pooling its hidden states produces vectors that look plausible and cluster
/// badly, and a caller has no way to tell — so with no embedding model
/// configured this refuses instead.
async fn embeddings(
    state: AxumState<State>,
    key: axum::Extension<ApiKey>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    authorise(&headers, &key)?;

    let inputs = match body.get("input") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|i| i.as_str().unwrap_or_default().to_string())
            .collect(),
        _ => return Err(ApiError::bad_request("input is required")),
    };
    if inputs.iter().all(|i| i.trim().is_empty()) {
        return Err(ApiError::bad_request("input is empty"));
    }

    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or_default()
        .to_string();

    let vectors = state
        .worker
        .embed(inputs.clone())
        .map_err(|e| ApiError::bad_request(e.to_string()))?;

    let data: Vec<serde_json::Value> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| serde_json::json!({ "object": "embedding", "index": i, "embedding": v }))
        .collect();
    // Token accounting is approximate here: embedding happens on the inference
    // thread and the tokeniser is not reachable from this side, so a rough
    // count is better than a fabricated exact one.
    let approx_tokens: usize = inputs.iter().map(|i| i.split_whitespace().count()).sum();

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": { "prompt_tokens": approx_tokens, "total_tokens": approx_tokens },
    }))
    .into_response())
}

/// Everything ozgent measured about a turn./// Everything ozgent measured about a turn.
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
            // Only calls the *caller* must run belong in `tool_calls`. A
            // server-side tool has already executed by the time the turn ends,
            // and reporting it here would tell a compliant client to run
            // something it has no code for — and to send back a result the
            // model already has.
            Event::ClientToolCall { name, arguments } => calls.push(serde_json::json!({
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
            Event::Ready { .. } | Event::ToolCall { .. } | Event::ToolResult { .. } => {}
        }
    }

    let mut message = serde_json::json!({ "role": "assistant", "content": answer });
    if !reasoning.trim().is_empty() {
        message["reasoning_content"] = reasoning.trim().into();
    }
    if !calls.is_empty() {
        message["tool_calls"] = calls.into();
        // The turn stopped because the caller has work to do, which OpenAI
        // spells this way; the generation's own stop reason is not the reason
        // the client should act on.
        stop = "tool_calls";
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
                Event::ClientToolCall { name, arguments } => {
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
                // A server-side tool has already run; telling the client about
                // it here would invite it to run the same call again.
                Event::ToolCall { .. } => {}
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
    fn a_json_schema_becomes_a_grammar_naming_its_fields() {
        let r = request(serde_json::json!({
            "model": "m", "messages": [],
            "response_format": { "type": "json_schema", "json_schema": { "name": "person",
                "schema": { "type": "object",
                    "properties": { "name": { "type": "string" }, "age": { "type": "integer" } },
                    "required": ["name", "age"] } } }
        }));
        let g = response_grammar(r.response_format.as_ref()).expect("valid").expect("some");
        assert!(g.starts_with("root ::="), "must have a root rule:\n{g}");
        assert!(g.contains("name"), "the schema's fields drive the grammar:\n{g}");
        assert!(g.contains("integer"), "an integer field needs the integer rule:\n{g}");
    }

    #[test]
    fn the_schema_may_sit_at_the_top_level_too() {
        // OpenAI nests it; several clients do not. Rejecting the flat spelling
        // would refuse a request whose intent is unambiguous.
        let r = request(serde_json::json!({
            "model": "m", "messages": [],
            "response_format": { "type": "json_schema",
                "schema": { "type": "object", "properties": { "ok": { "type": "boolean" } } } }
        }));
        assert!(response_grammar(r.response_format.as_ref()).unwrap().is_some());
    }

    #[test]
    fn json_object_mode_allows_nesting() {
        // Tool arguments do not nest arbitrarily, so the shared `value` rule is
        // flat. A bare json_object does nest, and reusing that rule would
        // forbid perfectly ordinary output.
        let r = request(serde_json::json!({
            "model": "m", "messages": [], "response_format": { "type": "json_object" }
        }));
        let g = response_grammar(r.response_format.as_ref()).unwrap().unwrap();
        assert!(g.contains("value ::= object | array"), "value must recurse:\n{g}");
    }

    #[test]
    fn plain_text_asks_for_no_constraint() {
        let r = request(serde_json::json!({
            "model": "m", "messages": [], "response_format": { "type": "text" }
        }));
        assert!(response_grammar(r.response_format.as_ref()).unwrap().is_none());
        let none = request(serde_json::json!({ "model": "m", "messages": [] }));
        assert!(response_grammar(none.response_format.as_ref()).unwrap().is_none());
    }

    #[test]
    fn an_unhonourable_response_format_is_refused_not_ignored() {
        // Returning prose to a client that will parse it as JSON is worse than
        // an error it can read.
        let r = request(serde_json::json!({
            "model": "m", "messages": [], "response_format": { "type": "yaml" }
        }));
        assert!(response_grammar(r.response_format.as_ref()).is_err());

        let missing = request(serde_json::json!({
            "model": "m", "messages": [], "response_format": { "type": "json_schema" }
        }));
        assert!(response_grammar(missing.response_format.as_ref()).is_err());
    }

    #[test]
    fn naming_native_tools_is_enough_to_enable_them() {
        // Requiring `ozgent_tools: true` as well would be two fields for one
        // intention, and the omission would look like the tools being ignored.
        let r = request(serde_json::json!({
            "model": "m", "messages": [], "native_tools": ["read_file"]
        }));
        assert_eq!(r.native_tools.as_deref(), Some(&["read_file".to_string()][..]));
        assert!(r.ozgent_tools.unwrap_or(r.native_tools.is_some()));
    }

    #[test]
    fn an_empty_native_tools_list_is_not_the_same_as_absent() {
        // `[]` withholds every built-in; absent offers all of them. Collapsing
        // the two would silently hand back the tools a caller just excluded.
        let none_named = request(serde_json::json!({
            "model": "m", "messages": [], "native_tools": []
        }));
        assert_eq!(none_named.native_tools.as_deref(), Some(&[][..]));

        let unset = request(serde_json::json!({ "model": "m", "messages": [] }));
        assert!(unset.native_tools.is_none());
        assert!(!unset.ozgent_tools.unwrap_or(unset.native_tools.is_some()));
    }

    #[test]
    fn caller_tools_are_read_from_the_documented_shape() {
        let r = request(serde_json::json!({
            "model": "m", "messages": [],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                }
            }]
        }));
        let tools = client_tools(r.tools.as_ref());
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description, "Look up the weather");
        assert_eq!(tools[0].input_schema["properties"]["city"]["type"], "string");
    }

    #[test]
    fn a_bare_function_object_is_also_accepted() {
        // Enough clients omit the `function` nesting that refusing it would
        // look like ozgent silently ignoring their tools — the exact failure
        // this replaces.
        let r = request(serde_json::json!({
            "model": "m", "messages": [],
            "tools": [{ "name": "flat", "parameters": { "type": "object", "properties": {} } }]
        }));
        assert_eq!(client_tools(r.tools.as_ref())[0].name, "flat");
    }

    #[test]
    fn a_function_without_parameters_gets_an_empty_object_schema() {
        // `None` would reach the grammar builder as an absent schema; the empty
        // object is what "takes no arguments" actually means.
        let r = request(serde_json::json!({
            "model": "m", "messages": [],
            "tools": [{ "type": "function", "function": { "name": "ping" } }]
        }));
        let tools = client_tools(r.tools.as_ref());
        assert_eq!(tools[0].input_schema["type"], "object");
    }

    #[test]
    fn a_tool_this_server_cannot_run_is_not_offered() {
        // Offering a hosted tool would let the model call something that can
        // never execute, and the turn would stall waiting for a result.
        let r = request(serde_json::json!({
            "model": "m", "messages": [],
            "tools": [
                { "type": "web_search_preview" },
                { "type": "function", "function": { "name": "real" } }
            ]
        }));
        let tools = client_tools(r.tools.as_ref());
        assert_eq!(tools.len(), 1, "only the function survives");
        assert_eq!(tools[0].name, "real");
    }

    #[test]
    fn no_tools_field_means_no_caller_tools() {
        let r = request(serde_json::json!({ "model": "m", "messages": [] }));
        assert!(client_tools(r.tools.as_ref()).is_empty());
    }

    #[test]
    fn a_calls_arguments_come_back_as_an_object_not_a_string() {
        // OpenAI sends arguments as a JSON *string*. Left as one, the call
        // renders with quoted arguments and the model reads it as a different
        // call than the one it made.
        let r = request(serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}" }
                }]
            }]
        }));
        let calls = r.messages[0].tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].arguments["city"], "Oslo");
    }

    #[test]
    fn a_tool_result_keeps_the_id_of_the_call_it_answers() {
        // With two calls outstanding, an empty id makes the results
        // indistinguishable.
        let r = request(serde_json::json!({
            "model": "m",
            "messages": [{ "role": "tool", "tool_call_id": "call_7", "content": "18C" }]
        }));
        let messages = to_messages(&r);
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_7"));
    }

    #[test]
    fn handing_a_call_back_is_not_an_ordinary_stop() {
        // A client that reads "stop" here will treat an unfinished turn as
        // finished and never run the tool.
        assert_eq!(finish_reason("ToolCalls"), "tool_calls");
        assert_eq!(finish_reason("EndOfText"), "stop");
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
