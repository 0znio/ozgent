//! Running one turn against the inference thread.
//!
//! Extracted from the HTTP handler because it stopped being about HTTP. A turn
//! is: record the question, assemble context from pinned facts, the recent
//! window and retrieval, submit it, relay what comes back, and persist the
//! reply. None of that changes because the question arrived over Telegram
//! instead of over `fetch`, and every part of it that *did* differ per surface
//! would be a way for two surfaces to drift apart — different context budgets,
//! a title rule applied in one place and not the other, a reply persisted from
//! the browser path and lost from the channel one.
//!
//! The one thing callers still own is what to *do* with the events, which is
//! why this hands back a receiver rather than rendering anything.

use ozgent_core::{ImageSource, ThinkingMode};
use ozgent_memory::{Budget, ContextBuilder, Embedder, OwnerKind};
use tokio::sync::mpsc::UnboundedReceiver;

use crate::state::State;
use crate::worker::{Event, Request};

/// How much of the window memory may spend on context.
///
/// Shared rather than per-surface: the same conversation continued from a
/// phone and from the browser has to be assembled the same way, or the model
/// sees a different history depending on which one you happened to pick up.
pub const BUDGET: Budget = Budget {
    total: 4096,
    reserve_for_reply: 1024,
    recent_messages: 12,
    max_retrieved: 6,
};

/// One question, from whichever surface asked it.
pub struct Turn {
    pub conversation: i64,
    pub model: String,
    pub message: String,
    pub thinking: Option<ThinkingMode>,
    /// Whether the model may call tools at all this turn.
    pub tools: bool,
    /// Which tools to offer. `None` offers all of them; a list withholds
    /// everything not named, which is how a channel keeps the shell away from
    /// a conversation happening on someone's phone.
    pub native_tools: Option<Vec<String>>,
    pub images: Vec<ImageSource>,
    /// Whether there is a person on the other end who can answer a permission
    /// question.
    pub can_ask: bool,
}

/// Start a turn. The receiver yields every event until `Done` or `Error`.
///
/// Persistence does not depend on the caller draining the receiver to the end:
/// a relay task owns the reply, keeps accumulating after the caller has gone,
/// and writes what it has. That is what makes a closed browser tab — or a
/// phone that went into a tunnel — lose the delivery but not the answer.
pub fn start(state: &State, turn: Turn) -> anyhow::Result<UnboundedReceiver<Event>> {
    let messages = {
        let store = state.store.lock().unwrap();

        // Name the conversation after its first line, so every list of
        // conversations shows something readable instead of "New chat".
        if store.message_count(turn.conversation)? == 0 {
            let title: String = turn
                .message
                .trim()
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(60)
                .collect();
            if !title.is_empty() {
                store.rename_conversation(turn.conversation, &title)?;
            }
        }

        let id = store.append_message(turn.conversation, "user", &turn.message, 0)?;

        // Kept so a reload still shows what the question was about. A failure
        // here must not lose the message, so it is logged rather than raised.
        let stored: Vec<String> = turn
            .images
            .iter()
            .filter_map(|src| match src {
                ImageSource::Bytes { bytes, mime } => {
                    crate::media::store(&state.paths, bytes.as_slice(), mime.as_deref())
                        .map_err(|e| tracing::warn!("storing an attachment: {e}"))
                        .ok()
                }
                _ => None,
            })
            .collect();
        if !stored.is_empty() {
            let _ = store.set_message_media(id, &stored);
        }

        store.put_embedding(OwnerKind::Message, id, &state.embedder.embed(&turn.message))?;

        let assembled = ContextBuilder::new(&store, &state.embedder)
            .with_budget(BUDGET)
            .build(turn.conversation, &turn.message)?;

        let config = state.config.lock().unwrap();
        let system = config
            .ui
            .date_awareness
            .then(|| ozgent_core::DateTime::now().prompt_line());
        assembled.to_messages(system.as_deref())
    };

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    state
        .worker
        .submit(Request {
            can_ask: turn.can_ask,
            model: turn.model,
            messages,
            thinking: turn.thinking,
            max_tokens: None,
            tools_enabled: turn.tools,
            native_tools: turn.native_tools,
            client_tools: Vec::new(),
            response_grammar: None,
            overrides: None,
            images: turn.images,
            out: tx,
        })
        // The inference thread is gone, which is ozgent's problem, not the
        // caller's.
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(relay(state.clone(), turn.conversation, rx))
}

/// Forward events to the caller while accumulating the reply, and write it
/// when the turn ends however it ends.
fn relay(
    state: State,
    conversation: i64,
    mut rx: UnboundedReceiver<Event>,
) -> UnboundedReceiver<Event> {
    let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();

    tokio::spawn(async move {
        let mut answer = String::new();
        let mut thinking = String::new();
        let mut activity: Vec<serde_json::Value> = Vec::new();

        while let Some(event) = rx.recv().await {
            match &event {
                Event::Answer { text } => answer.push_str(text),
                Event::Thinking { text } => thinking.push_str(text),
                Event::ToolCall { name, arguments, .. } => activity.push(serde_json::json!({
                    "name": name,
                    "arguments": arguments,
                })),
                Event::ToolResult { name, ok, summary, ms, detail, .. } => {
                    // Attach to the call this answers, so a reload replays the
                    // pair rather than two loose halves.
                    let slot = activity
                        .iter_mut()
                        .rev()
                        .find(|c| c["name"] == name.as_str() && c.get("ok").is_none());
                    if let Some(call) = slot {
                        call["ok"] = (*ok).into();
                        call["ms"] = (*ms).into();
                        call["summary"] = summary.clone().into();
                        call["detail"] = crate::api::bounded(detail);
                    }
                }
                _ => {}
            }
            let finished = matches!(event, Event::Done { .. } | Event::Error { .. });
            // A send failure means the caller went away. Stop relaying, but
            // fall through to persist whatever arrived first.
            let gone = out_tx.send(event).is_err();
            if finished || gone {
                break;
            }
        }
        // Dropping this end is what tells the inference thread to stop, so a
        // caller that disappeared mid-generation still frees the GPU.
        drop(rx);

        if answer.trim().is_empty() {
            return;
        }
        let store = state.store.lock().unwrap();
        let trace = (!thinking.trim().is_empty()).then(|| thinking.trim().to_string());
        let calls =
            (!activity.is_empty()).then(|| serde_json::to_string(&activity).unwrap_or_default());
        match store.append_message_full(
            conversation,
            "assistant",
            answer.trim(),
            trace.as_deref(),
            calls.as_deref(),
            None,
            0,
        ) {
            Ok(id) => {
                let _ =
                    store.put_embedding(OwnerKind::Message, id, &state.embedder.embed(answer.trim()));
            }
            Err(e) => tracing::error!("persisting the reply: {e}"),
        }
    });

    out_rx
}
