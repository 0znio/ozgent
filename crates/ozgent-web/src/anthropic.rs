//! Anthropic's Messages API, at `/v1/messages`.
//!
//! Beside the OpenAI routes so a client written for either works unchanged:
//! point it at `http://host:port/v1`, pick a model from `/v1/models`, and it
//! speaks its own protocol to the same engine, tools and agents.
//!
//! The parts of the protocol that are awkward are kept, because clients depend
//! on them: content is a list of typed blocks; tool results come back as
//! blocks inside a *user* message; a stream is a fixed sequence of named events
//! (`message_start`, then blocks opened, filled and closed by index, then
//! `message_delta` with the stop reason, then `message_stop`).
//!
//! What ozgent adds travels in blocks the protocol already has. An agent's
//! account of its work is a `thinking` block — where every Anthropic client
//! already shows a model working — and its report is the `text`.

use axum::extract::State as AxumState;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::Stream;
use ozgent_core::{Message, ThinkingMode, ToolSpec};
use serde::Deserialize;
use std::collections::VecDeque;
use std::convert::Infallible;

use crate::openai::{ApiKey, id};
use crate::state::State;
use crate::worker::{Event, Request};

// ------------------------------------------------------------------ errors

/// Anthropic's error envelope, so a client's own error handling fires.
pub struct AnthropicError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl AnthropicError {
    fn invalid(message: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, kind: "invalid_request_error", message: message.into() }
    }
    fn not_found(message: impl Into<String>) -> Self {
        Self { status: StatusCode::NOT_FOUND, kind: "not_found_error", message: message.into() }
    }
    fn internal(message: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, kind: "api_error", message: message.into() }
    }
    /// A prompt longer than the context is the caller's to fix, not a fault.
    fn from_worker(message: String) -> Self {
        if message.contains("but the context holds") {
            Self::invalid(message)
        } else {
            Self::internal(message)
        }
    }
}

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "type": "error",
                "error": { "type": self.kind, "message": self.message },
            })),
        )
            .into_response()
    }
}

// -------------------------------------------------------------- the request

#[derive(Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<InMessage>,
    /// A string, or a list of text blocks.
    #[serde(default)]
    pub system: Option<serde_json::Value>,
    /// Required by Anthropic; optional here, where absent means the model's
    /// own limit rather than an error.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// `{"type":"enabled","budget_tokens":N}`, `{"type":"disabled"}` or
    /// `{"type":"adaptive"}`.
    #[serde(default)]
    pub thinking: Option<serde_json::Value>,
    /// ozgent extension, as on the OpenAI route: use the server's own tools.
    #[serde(default)]
    pub ozgent_tools: Option<bool>,
    /// ozgent extension: which built-in tools to offer, by name.
    #[serde(default)]
    pub native_tools: Option<Vec<String>>,
    /// ozgent extension: `false` stops `@name` in a message calling an agent.
    #[serde(default)]
    pub ozgent_agents: Option<bool>,
}

#[derive(Deserialize)]
pub struct InMessage {
    pub role: String,
    /// A string, or a list of content blocks.
    pub content: serde_json::Value,
}

/// A request's messages in ozgent's terms, with the images they carry.
pub fn to_messages(
    request: &MessagesRequest,
) -> Result<(Vec<Message>, Vec<ozgent_core::ImageSource>), String> {
    let mut out = Vec::new();
    let mut images = Vec::new();

    match &request.system {
        Some(serde_json::Value::String(s)) if !s.trim().is_empty() => out.push(Message::system(s.clone())),
        Some(serde_json::Value::Array(blocks)) => {
            let text = texts(blocks);
            if !text.trim().is_empty() {
                out.push(Message::system(text));
            }
        }
        _ => {}
    }

    for message in &request.messages {
        let blocks: Vec<serde_json::Value> = match &message.content {
            serde_json::Value::String(s) => vec![serde_json::json!({"type": "text", "text": s})],
            serde_json::Value::Array(blocks) => blocks.clone(),
            other => return Err(format!("message content must be a string or a list, got {other}")),
        };
        let mut text = String::new();
        let mut calls = Vec::new();
        for block in &blocks {
            match block.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "text" => {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(block.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                }
                "image" => images.push(image(block)?),
                "document" => {
                    // Plain-text documents are text; a PDF would need a
                    // renderer this server does not have, and dropping it
                    // silently would answer a question about a file unseen.
                    let source = block.get("source").cloned().unwrap_or_default();
                    match source.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            text.push_str(source.get("data").and_then(|d| d.as_str()).unwrap_or(""));
                        }
                        other => {
                            return Err(format!(
                                "document blocks of source type {:?} are not supported; send the text",
                                other.unwrap_or("none")
                            ));
                        }
                    }
                }
                // Tool results are their own messages in ozgent, in the order
                // they arrive, ahead of any text the user added after them.
                "tool_result" => {
                    let id = block.get("tool_use_id").and_then(|i| i.as_str()).unwrap_or_default();
                    let mut body = match block.get("content") {
                        Some(serde_json::Value::String(s)) => s.clone(),
                        Some(serde_json::Value::Array(parts)) => texts(parts),
                        _ => String::new(),
                    };
                    if block.get("is_error").and_then(|e| e.as_bool()) == Some(true) {
                        body = format!("Error: {body}");
                    }
                    out.push(Message::tool_result(id, body));
                }
                "tool_use" => calls.push(ozgent_core::ToolCall {
                    id: block.get("id").and_then(|i| i.as_str()).unwrap_or_default().to_string(),
                    name: block.get("name").and_then(|n| n.as_str()).unwrap_or_default().to_string(),
                    arguments: block.get("input").cloned().unwrap_or_else(|| serde_json::json!({})),
                }),
                // Earlier reasoning is not replayed: ozgent's templates render
                // the model's own history, and a signed block from another
                // model means nothing to this one.
                "thinking" | "redacted_thinking" => {}
                // Server-side tool blocks from Anthropic's hosted tools have
                // no meaning here.
                _ => {}
            }
        }
        match message.role.as_str() {
            "assistant" => {
                let mut m = Message::assistant(text);
                m.tool_calls = calls;
                out.push(m);
            }
            "user" => {
                if !text.is_empty() {
                    out.push(Message::user(text));
                }
            }
            other => return Err(format!("role must be user or assistant, got {other:?}")),
        }
    }
    Ok((out, images))
}

fn texts(blocks: &[serde_json::Value]) -> String {
    blocks
        .iter()
        .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn image(block: &serde_json::Value) -> Result<ozgent_core::ImageSource, String> {
    let source = block.get("source").ok_or("image block has no source")?;
    match source.get("type").and_then(|t| t.as_str()) {
        Some("base64") => {
            let mime = source.get("media_type").and_then(|m| m.as_str()).unwrap_or("image/png");
            let data = source.get("data").and_then(|d| d.as_str()).ok_or("image has no data")?;
            crate::api::decode_data_url(&format!("data:{mime};base64,{data}"))
        }
        Some("url") => Err("image URLs are not fetched by this server; send the image as base64".into()),
        other => Err(format!("unsupported image source {:?}", other.unwrap_or("none"))),
    }
}

/// Tools the caller implements, from Anthropic's `tools` list.
///
/// Anthropic's own hosted tools (`web_search_20250305` and the rest) carry a
/// `type` and no schema; this server cannot run them, so they are not
/// offered — a model that called one would wait for a result that never comes.
pub fn client_tools(value: Option<&serde_json::Value>) -> Vec<ToolSpec> {
    let Some(serde_json::Value::Array(items)) = value else { return Vec::new() };
    items
        .iter()
        .filter(|t| matches!(t.get("type").and_then(|k| k.as_str()), None | Some("custom")))
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?.trim();
            (!name.is_empty()).then(|| ToolSpec {
                name: name.to_string(),
                description: t.get("description").and_then(|d| d.as_str()).unwrap_or_default().to_string(),
                input_schema: t
                    .get("input_schema")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
                output_schema: None,
                // Run by the caller, never here.
                effect: ozgent_core::permission::Effect::Read,
            })
        })
        .collect()
}

/// The thinking mode and effort a request's `thinking` field asks for.
///
/// Absent means off, as it does on Anthropic's own API: a client that never
/// mentions thinking is not expecting a model that spends a minute on it.
pub fn thinking(value: Option<&serde_json::Value>) -> (ThinkingMode, Option<ozgent_core::ReasoningEffort>) {
    use ozgent_core::ReasoningEffort as Effort;
    let Some(v) = value else { return (ThinkingMode::Off, None) };
    match v.get("type").and_then(|t| t.as_str()) {
        Some("enabled") => {
            let budget = v.get("budget_tokens").and_then(|b| b.as_u64()).unwrap_or(4096);
            let effort = if budget < 2048 {
                Effort::Low
            } else if budget < 16384 {
                Effort::Medium
            } else {
                Effort::High
            };
            (ThinkingMode::On, Some(effort))
        }
        Some("adaptive") => (ThinkingMode::Auto, None),
        _ => (ThinkingMode::Off, None),
    }
}

/// Anthropic's spelling of why generation ended.
pub fn stop_reason(stop: &str) -> &'static str {
    match stop {
        "ToolCalls" => "tool_use",
        "TokenLimit" | "ContextFull" => "max_tokens",
        _ => "end_turn",
    }
}

// ----------------------------------------------------------------- handler

pub async fn messages(
    AxumState(state): AxumState<State>,
    axum::Extension(key): axum::Extension<ApiKey>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, AnthropicError> {
    if !crate::openai::authorise_key(&headers, &key) {
        return Err(AnthropicError {
            status: StatusCode::UNAUTHORIZED,
            kind: "authentication_error",
            message: "missing or invalid API key".into(),
        });
    }
    // Parsed by hand so a malformed body gets Anthropic's error shape rather
    // than axum's plain-text rejection, which no client can read.
    let request: MessagesRequest = serde_json::from_slice(&body)
        .map_err(|e| AnthropicError::invalid(format!("invalid request body: {e}")))?;
    if request.messages.is_empty() {
        return Err(AnthropicError::invalid("messages: at least one message is required"));
    }

    let (messages, images) = to_messages(&request).map_err(AnthropicError::invalid)?;
    let latest = request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| match &m.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(blocks) => texts(blocks),
            _ => String::new(),
        })
        .unwrap_or_default();
    let resolved = crate::agents::resolve(&state, &request.model, &latest, request.ozgent_agents.unwrap_or(true))
        .map_err(|r| match r {
            crate::agents::Refusal::NotFound(m) => AnthropicError::not_found(m),
            crate::agents::Refusal::BadRequest(m) => AnthropicError::invalid(m),
        })?;

    let (mode, effort) = thinking(request.thinking.as_ref());
    // Shown when the caller asked to see reasoning. An agent's account of its
    // work is shown regardless: it is the answer to "what did it do".
    let show_thinking = mode != ThinkingMode::Off;
    let no_tools = request
        .tool_choice
        .as_ref()
        .and_then(|c| c.get("type"))
        .and_then(|t| t.as_str())
        == Some("none");

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    state
        .worker
        .submit(Request {
            can_ask: false,
            model: resolved.model.model.to_string(),
            messages,
            thinking: Some(mode),
            max_tokens: request.max_tokens,
            tools_enabled: !no_tools
                && request.ozgent_tools.unwrap_or(request.native_tools.is_some()),
            native_tools: request.native_tools.clone(),
            client_tools: if no_tools { Vec::new() } else { client_tools(request.tools.as_ref()) },
            response_grammar: None,
            overrides: Some(ozgent_core::Options {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
                max_tokens: request.max_tokens,
                reasoning_effort: effort,
                ..Default::default()
            }),
            images,
            agents: resolved.agents,
            tools_off: Vec::new(),
            handoff: Vec::new(),
            out: tx,
        })
        .map_err(AnthropicError::internal)?;

    let model = resolved.model.model.to_string();
    if request.stream {
        Ok(Sse::new(stream(rx, model, show_thinking)).into_response())
    } else {
        Ok(Json(collect(rx, model, show_thinking).await?).into_response())
    }
}

// ----------------------------------------------------------------- replies

/// Content blocks built up from a turn's events, merging runs of one kind.
#[derive(Default)]
struct Blocks {
    blocks: Vec<serde_json::Value>,
    trace: crate::agents::Trace,
    tool_calls: usize,
}

impl Blocks {
    fn push_text(&mut self, kind: &str, text: &str) {
        if text.is_empty() {
            return;
        }
        let field = if kind == "thinking" { "thinking" } else { "text" };
        if let Some(last) = self.blocks.last_mut().filter(|b| b["type"] == kind) {
            let joined = format!("{}{text}", last[field].as_str().unwrap_or_default());
            last[field] = joined.into();
            return;
        }
        let mut block = serde_json::json!({ "type": kind, field: text });
        if kind == "thinking" {
            // Anthropic signs its thinking blocks so they can be replayed.
            // Nothing here verifies one, so the field is present and empty.
            block["signature"] = "".into();
        }
        self.blocks.push(block);
    }
}

async fn collect(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    model: String,
    show_thinking: bool,
) -> Result<serde_json::Value, AnthropicError> {
    let mut out = Blocks::default();
    let mut usage = (0u32, 0u32);
    let mut stop = "end_turn";

    while let Some(event) = rx.recv().await {
        if let Some(line) = out.trace.line(&event) {
            out.push_text("thinking", &line);
        }
        match event {
            Event::Answer { text } => out.push_text("text", &text),
            Event::Thinking { text } if show_thinking || out.trace.in_agent() => {
                out.push_text("thinking", &text)
            }
            Event::ClientToolCall { name, arguments } => {
                out.tool_calls += 1;
                out.blocks.push(serde_json::json!({
                    "type": "tool_use",
                    "id": id("toolu").replacen('-', "_", 1),
                    "name": name,
                    "input": arguments,
                }));
            }
            Event::Done { generated, prompt, stop: reason, .. } => {
                usage = (prompt, generated);
                stop = stop_reason(&reason);
            }
            Event::Error { message } => return Err(AnthropicError::from_worker(message)),
            _ => {}
        }
    }
    if out.tool_calls > 0 {
        stop = "tool_use";
    }
    // A reply must have content; an empty text block is how Anthropic says
    // "nothing", and clients index into the list without checking.
    if out.blocks.is_empty() {
        out.blocks.push(serde_json::json!({ "type": "text", "text": "" }));
    }
    Ok(serde_json::json!({
        "id": id("msg").replacen('-', "_", 1),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": out.blocks,
        "stop_reason": stop,
        "stop_sequence": null,
        "usage": { "input_tokens": usage.0, "output_tokens": usage.1 },
    }))
}

/// The state of a stream between events.
struct Streaming {
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    model: String,
    show_thinking: bool,
    started: bool,
    finished: bool,
    /// The block being filled, by kind, and the next index to open.
    open: Option<&'static str>,
    next_index: usize,
    tool_calls: usize,
    trace: crate::agents::Trace,
    /// Events produced but not yet sent, as `(name, data)`. Kept structured
    /// until the last moment so the sequence can be tested as data.
    queued: VecDeque<(&'static str, serde_json::Value)>,
}

fn sse(name: &str, data: serde_json::Value) -> SseEvent {
    SseEvent::default().event(name).data(serde_json::to_string(&data).unwrap_or_default())
}

impl Streaming {
    fn start(&mut self) {
        if self.started {
            return;
        }
        self.started = true;
        self.queued.push_back((
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": id("msg").replacen('-', "_", 1),
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    // Not known until the prompt has been processed; the
                    // final `message_delta` carries both counts.
                    "usage": { "input_tokens": 0, "output_tokens": 0 },
                },
            }),
        ));
    }

    fn close(&mut self) {
        if self.open.take().is_some() {
            self.queued.push_back((
                "content_block_stop",
                serde_json::json!({ "type": "content_block_stop", "index": self.next_index - 1 }),
            ));
        }
    }

    /// Append text to a block of `kind`, opening one if another kind is open.
    fn text(&mut self, kind: &'static str, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.open != Some(kind) {
            self.close();
            let block = if kind == "thinking" {
                serde_json::json!({ "type": "thinking", "thinking": "", "signature": "" })
            } else {
                serde_json::json!({ "type": "text", "text": "" })
            };
            self.queued.push_back((
                "content_block_start",
                serde_json::json!({ "type": "content_block_start", "index": self.next_index, "content_block": block }),
            ));
            self.open = Some(kind);
            self.next_index += 1;
        }
        let delta = if kind == "thinking" {
            serde_json::json!({ "type": "thinking_delta", "thinking": text })
        } else {
            serde_json::json!({ "type": "text_delta", "text": text })
        };
        self.queued.push_back((
            "content_block_delta",
            serde_json::json!({ "type": "content_block_delta", "index": self.next_index - 1, "delta": delta }),
        ));
    }

    fn tool_use(&mut self, name: &str, arguments: &serde_json::Value) {
        self.close();
        let index = self.next_index;
        self.next_index += 1;
        self.tool_calls += 1;
        self.queued.push_back((
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": index,
                "content_block": { "type": "tool_use", "id": id("toolu").replacen('-', "_", 1), "name": name, "input": {} },
            }),
        ));
        self.queued.push_back((
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": index,
                "delta": { "type": "input_json_delta", "partial_json": arguments.to_string() },
            }),
        ));
        self.queued.push_back((
            "content_block_stop",
            serde_json::json!({ "type": "content_block_stop", "index": index }),
        ));
    }

    fn handle(&mut self, event: Event) {
        self.start();
        if let Some(line) = self.trace.line(&event) {
            self.text("thinking", &line);
        }
        match event {
            Event::Answer { text } => self.text("text", &text),
            Event::Thinking { text } if self.show_thinking || self.trace.in_agent() => {
                self.text("thinking", &text)
            }
            Event::ClientToolCall { name, arguments } => self.tool_use(&name, &arguments),
            Event::Done { generated, prompt, stop, .. } => {
                // Every message has at least one block; a client that indexes
                // `content[0]` must not fall off an empty list.
                if self.next_index == 0 {
                    self.text("text", " ");
                }
                self.close();
                let reason = if self.tool_calls > 0 { "tool_use" } else { stop_reason(&stop) };
                self.queued.push_back((
                    "message_delta",
                    serde_json::json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": reason, "stop_sequence": null },
                        "usage": { "input_tokens": prompt, "output_tokens": generated },
                    }),
                ));
                self.queued.push_back(("message_stop", serde_json::json!({ "type": "message_stop" })));
                self.finished = true;
            }
            Event::Error { message } => {
                self.queued.push_back((
                    "error",
                    serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": message },
                    }),
                ));
                self.finished = true;
            }
            _ => {}
        }
    }
}

fn stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    model: String,
    show_thinking: bool,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let state = Streaming {
        rx,
        model,
        show_thinking,
        started: false,
        finished: false,
        open: None,
        next_index: 0,
        tool_calls: 0,
        trace: Default::default(),
        queued: VecDeque::new(),
    };
    futures_util::stream::unfold(state, |mut st| async move {
        loop {
            if let Some((name, data)) = st.queued.pop_front() {
                return Some((Ok(sse(name, data)), st));
            }
            if st.finished {
                return None;
            }
            match st.rx.recv().await {
                Some(event) => {
                    st.handle(event);
                    // An event that produced nothing — a model loading, a
                    // server tool running — still says the stream is alive,
                    // the way Anthropic's own server does.
                    if st.queued.is_empty() {
                        return Some((Ok(sse("ping", serde_json::json!({ "type": "ping" }))), st));
                    }
                }
                None => {
                    // Ended without a Done: finish the message properly, or a
                    // client waits for a `message_stop` that never comes.
                    st.handle(Event::Done {
                        generated: 0,
                        tokens_per_second: 0.0,
                        reused: 0,
                        stop: "EndOfText".into(),
                        prompt: 0,
                        prompt_ms: 0,
                    });
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(json).expect("should deserialise")
    }

    #[test]
    fn a_tool_round_trip_keeps_calls_and_results_paired() {
        let r = request(serde_json::json!({
            "model": "m",
            "system": [{"type": "text", "text": "be brief"}],
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hm", "signature": "x"},
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Pune"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "31C"}]},
                    {"type": "text", "text": "and tomorrow?"},
                ]},
            ],
        }));
        let (messages, images) = to_messages(&r).unwrap();
        assert!(images.is_empty());
        let roles: Vec<_> = messages.iter().map(|m| m.role).collect();
        use ozgent_core::Role::*;
        assert_eq!(roles, [System, User, Assistant, Tool, User]);
        assert_eq!(messages[0].text_content(), "be brief");
        assert_eq!(messages[2].text_content(), "checking");
        assert_eq!(messages[2].tool_calls[0].id, "toolu_1");
        assert_eq!(messages[2].tool_calls[0].arguments["city"], "Pune");
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(messages[3].text_content(), "31C");
    }

    #[test]
    fn an_error_result_is_marked_as_one() {
        let r = request(serde_json::json!({"model": "m", "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "boom", "is_error": true}]},
        ]}));
        let (messages, _) = to_messages(&r).unwrap();
        assert_eq!(messages[0].text_content(), "Error: boom");
    }

    #[test]
    fn a_base64_image_is_accepted_and_a_url_is_refused() {
        let png = "iVBORw0KGgo=";
        let ok = request(serde_json::json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
            {"type": "text", "text": "what is this"},
        ]}]}));
        let (_, images) = to_messages(&ok).unwrap();
        assert_eq!(images.len(), 1);

        let url = request(serde_json::json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
        ]}]}));
        assert!(to_messages(&url).unwrap_err().contains("base64"));
    }

    #[test]
    fn hosted_tools_are_not_offered_and_custom_ones_are() {
        let tools = client_tools(Some(&serde_json::json!([
            {"name": "get_weather", "description": "d", "input_schema": {"type": "object"}},
            {"type": "custom", "name": "lookup", "input_schema": {"type": "object"}},
            {"type": "web_search_20250305", "name": "web_search"},
        ])));
        let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["get_weather", "lookup"]);
    }

    #[test]
    fn thinking_follows_anthropics_meaning_of_absent() {
        assert_eq!(thinking(None).0, ThinkingMode::Off);
        assert_eq!(thinking(Some(&serde_json::json!({"type": "disabled"}))).0, ThinkingMode::Off);
        let (mode, effort) = thinking(Some(&serde_json::json!({"type": "enabled", "budget_tokens": 1024})));
        assert_eq!(mode, ThinkingMode::On);
        assert_eq!(effort, Some(ozgent_core::ReasoningEffort::Low));
        assert_eq!(thinking(Some(&serde_json::json!({"type": "adaptive"}))).0, ThinkingMode::Auto);
    }

    #[test]
    fn stop_reasons_use_anthropics_vocabulary() {
        assert_eq!(stop_reason("EndOfText"), "end_turn");
        assert_eq!(stop_reason("ToolCalls"), "tool_use");
        assert_eq!(stop_reason("TokenLimit"), "max_tokens");
        assert_eq!(stop_reason("ContextFull"), "max_tokens");
    }

    /// Drive the stream state machine and read back what it would send.
    fn run(events: Vec<Event>, show_thinking: bool) -> Vec<(&'static str, serde_json::Value)> {
        let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut st = Streaming {
            rx,
            model: "m".into(),
            show_thinking,
            started: false,
            finished: false,
            open: None,
            next_index: 0,
            tool_calls: 0,
            trace: Default::default(),
            queued: VecDeque::new(),
        };
        for e in events {
            st.handle(e);
        }
        st.queued.into_iter().collect()
    }

    fn done() -> Event {
        Event::Done {
            generated: 5,
            tokens_per_second: 1.0,
            reused: 0,
            stop: "EndOfText".into(),
            prompt: 9,
            prompt_ms: 1,
        }
    }

    #[test]
    fn a_stream_opens_closes_and_numbers_its_blocks_in_order() {
        let out = run(
            vec![
                Event::Thinking { text: "hidden".into() },
                Event::Answer { text: "Hel".into() },
                Event::Answer { text: "lo".into() },
                Event::ClientToolCall { name: "f".into(), arguments: serde_json::json!({"a": 1}) },
                done(),
            ],
            false,
        );
        let names: Vec<&str> = out.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        // Every event carries its own name as `type`, which is what the SDKs
        // actually switch on.
        for (name, data) in &out {
            assert_eq!(data["type"], *name);
        }
        // Reasoning the caller did not ask for is not sent.
        assert!(!serde_json::to_string(&out.iter().map(|(_, d)| d).collect::<Vec<_>>()).unwrap().contains("hidden"));
        // The text block is index 0 and the tool block index 1.
        assert_eq!(out[1].1["index"], 0);
        assert_eq!(out[1].1["content_block"]["type"], "text");
        assert_eq!(out[5].1["index"], 1);
        assert_eq!(out[5].1["content_block"]["type"], "tool_use");
        assert_eq!(out[6].1["delta"]["partial_json"], r#"{"a":1}"#);
        // The call makes the stop reason tool_use, and usage is reported.
        assert_eq!(out[8].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(out[8].1["usage"]["input_tokens"], 9);
        assert_eq!(out[8].1["usage"]["output_tokens"], 5);
    }

    #[test]
    fn asked_for_reasoning_is_a_thinking_block_before_the_text() {
        let out = run(
            vec![Event::Thinking { text: "hmm".into() }, Event::Answer { text: "hi".into() }, done()],
            true,
        );
        assert_eq!(out[1].1["content_block"]["type"], "thinking");
        assert_eq!(out[2].1["delta"]["thinking"], "hmm");
        assert_eq!(out[4].1["content_block"]["type"], "text");
        assert_eq!(out[4].1["index"], 1);
    }

    #[test]
    fn an_empty_reply_still_has_one_block() {
        let out = run(vec![done()], false);
        assert!(out.iter().any(|(n, _)| *n == "content_block_start"));
    }

    #[test]
    fn an_agents_account_is_streamed_as_thinking_even_when_not_asked_for() {
        let out = run(
            vec![
                Event::AgentStart {
                    name: "stock-guru".into(),
                    description: "d".into(),
                    tools: vec!["yahoo_finance".into()],
                    missing: vec![],
                },
                Event::Thinking { text: "let me look".into() },
                Event::AgentEnd { name: "stock-guru".into(), ok: true, ms: 1000, calls: 0, rounds: 1 },
                Event::Answer { text: "NVDA is up".into() },
                done(),
            ],
            false,
        );
        let thinking: String = out
            .iter()
            .filter_map(|(_, d)| d["delta"]["thinking"].as_str())
            .collect();
        assert!(thinking.starts_with("@stock-guru is working on this"));
        assert!(thinking.contains("let me look"));
        assert!(thinking.contains("@stock-guru finished"));
        let text: String = out.iter().filter_map(|(_, d)| d["delta"]["text"].as_str()).collect();
        assert_eq!(text, "NVDA is up");
    }
}
