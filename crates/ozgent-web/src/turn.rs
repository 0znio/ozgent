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
/// How much of the conversation is carried verbatim into a prompt: a
/// quarter of the model's window, between 4k and 32k tokens. It was a fixed
/// 4k, so on a 64k model a few long answers pushed the start of a
/// conversation out of view while three quarters of the window sat unused.
/// What is carried is cached from turn to turn, so a longer window costs
/// room, not time.
pub fn budget_for(context_length: u32) -> Budget {
    Budget {
        total: (context_length as usize / 4).clamp(4096, 32_768),
        reserve_for_reply: 1024,
        recent_messages: 40,
        max_retrieved: 6,
    }
}

/// Characters of the last turn's tool results carried into the next.
const CARRIED_CHARS: usize = 12_000;

/// One question, from whichever surface asked it.
pub struct Turn {
    /// A persona for this conversation only, in place of the model's saved
    /// one: what `ozgent chat --system` means.
    pub system: Option<String>,
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
    /// Tools the person switched off for the main model. Unlike
    /// `native_tools`, an agent called by name keeps its own.
    pub tools_off: Vec<String>,
    pub images: Vec<ImageSource>,
    /// Whether there is a person on the other end who can answer a permission
    /// question.
    pub can_ask: bool,
    /// Who is asking, for the tools that need to know.
    ///
    /// Only the scheduler does, and it needs to: a job created from a chat
    /// answers back into that chat and inherits its tool allowlist, neither of
    /// which can be worked out from the message. `None` is the browser and the
    /// terminal — trusted, and with no chat to default to.
    pub caller: Option<ozgent_schedule::Caller>,
}

/// The standing lines put in front of the conversation.
///
/// A scheduled job's prompt is a standing instruction written for a timer —
/// "a market brief at 10am on weekdays". Handed over with no framing, a model
/// reads the timing as a request to *arrange* that, reaches for the schedule
/// tool, and spends the answer explaining why it could not. Saying plainly
/// that the timer has already fired is what turns the prompt back into the
/// question it is.
/// What the model is told about the date, for the chat and for agents alike.
/// The time itself arrives stamped on each user message, in the same zone.
///
/// The zone is this machine's. It is the one the user set up, and the only
/// one ozgent knows without asking; a phone on the other side of the world
/// talking to it over a channel still gets the owner's clock.
pub(crate) fn date_line() -> String {
    let zone = ozgent_core::Zone::local();
    let now = ozgent_core::DateTime::now().to_unix();
    format!(
        "{} Each user message begins with the time it was sent.",
        zone.local_at(now).prompt_line_in(&zone_label(&zone, now))
    )
}

/// "Asia/Kolkata (UTC+05:30)", or plain "UTC".
fn zone_label(zone: &ozgent_core::Zone, at: i64) -> String {
    let offset = zone.label_at(at);
    if zone.name().eq_ignore_ascii_case(&offset) {
        offset
    } else {
        format!("{} ({offset})", zone.name())
    }
}

pub(crate) fn system_prompt(date_aware: bool, caller: Option<&ozgent_schedule::Caller>) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    if date_aware {
        lines.push(date_line());
    }
    if let Some(name) = caller.and_then(|c| c.origin.strip_prefix("job:")) {
        lines.push(format!(
            "You are the scheduled job \"{name}\", running now because its time came \
             round. What follows is the job's own standing instruction, not a request \
             to set anything up: the schedule already exists and is what woke you. \
             Answer it for today, as the answer itself. Nobody is at a keyboard, so \
             there is no one to ask and no reply coming."
        ));
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// Put the time each user message was sent in front of it.
///
/// This is how the model knows the time now that the system prompt carries
/// only the date. It comes from the stored timestamp rather than the clock, so
/// every earlier message renders exactly as it did on its own turn and the
/// cache holding it stays valid.
fn stamp_user_messages(messages: &mut [ozgent_memory::StoredMessage]) {
    let zone = ozgent_core::Zone::local();
    let today = zone.local_at(ozgent_core::DateTime::now().to_unix());
    for m in messages.iter_mut().filter(|m| m.role == "user") {
        let sent = zone.local_at(m.created_at);
        m.content = format!("{} {}", sent.stamp(&today), m.content);
    }
}

/// Start a turn. The receiver yields every event until `Done` or `Error`.
///
/// Persistence does not depend on the caller draining the receiver to the end:
/// a relay task owns the reply, keeps accumulating after the caller has gone,
/// and writes what it has. That is what makes a closed browser tab — or a
/// phone that went into a tunnel — lose the delivery but not the answer.
pub fn start(state: &State, turn: Turn) -> anyhow::Result<UnboundedReceiver<Event>> {
    let messages = {
    // The question's vector, computed before the store is locked and only when
    // there is something older than the recent window to search. With a real
    // embedding model this is a forward pass — cheap on the GPU, not on the
    // CPU — and a short conversation has nothing to recall anyway.
    let budget = {
        let config = state.config.lock().unwrap_or_else(|e| e.into_inner());
        let key = ozgent_core::resolve(&state.paths, &turn.model)
            .map(|f| f.model.to_string())
            .unwrap_or_else(|_| turn.model.clone());
        budget_for(config.options_for(&key).resolve().context_length)
    };
    let query_vector = {
        let wants = {
            let store = state.store.lock().unwrap_or_else(|e| e.into_inner());
            store.message_count(turn.conversation).unwrap_or(0) as usize > budget.recent_messages
                || !store
                    .embeddings_in_conversation(OwnerKind::Fact, turn.conversation)
                    .unwrap_or_default()
                    .is_empty()
        };
        if wants { state.embedder.embed_query(&turn.message) } else { Vec::new() }
    };

        let store = state.store.lock().unwrap_or_else(|e| e.into_inner());

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

        // Stored in the background, so the reply does not wait for it.
        crate::memory::embed_later(state, id, turn.message.clone());

        let mut assembled = ContextBuilder::new(&store, &state.embedder)
            .with_budget(budget)
            .with_query_vector(query_vector)
            .build(turn.conversation, &turn.message)?;
        // What the last turn's tools returned rides along with the reply they
        // informed, so "summarise that" or "what did Reuters say about it"
        // is answered from what was already read instead of fetched again.
        let asked_at = store.get_message(id).ok().flatten().map(|m| m.seq).unwrap_or(i64::MAX);
        if let Ok(rows) = store.previous_turn_tools(turn.conversation, asked_at) {
            if let Some(material) = carried_material(&rows) {
                if let Some(reply) = assembled.recent.iter_mut().rev().find(|m| m.role == "assistant") {
                    reply.content = format!("{}\n\n{material}", reply.content.trim_end());
                }
            }
        }

        let config = state.config.lock().unwrap_or_else(|e| e.into_inner());
        // The model's persona and response style, then ozgent's own lines.
        // Per model, and stable from turn to turn, so the cached prompt
        // prefix survives; changing either costs one re-read.
        let persona = {
            let key = ozgent_core::resolve(&state.paths, &turn.model)
                .map(|f| f.model.to_string())
                .unwrap_or_else(|_| turn.model.clone());
            let options = config.options_for(&key);
            let persona = turn.system.as_deref().or(options.system_prompt.as_deref());
            ozgent_core::styles::compose(persona, options.style.as_deref(), &config.styles)
        };
        let system = match (persona, system_prompt(config.ui.date_awareness, turn.caller.as_ref())) {
            (Some(p), Some(ours)) => Some(format!("{p}\n\n{ours}")),
            (p, ours) => p.or(ours),
        };
        if config.ui.date_awareness {
            stamp_user_messages(&mut assembled.recent);
        }
        assembled.to_messages(system.as_deref())
    };

    // Read per turn, so an agent saved in Settings a moment ago can be
    // called in the very next message. A handful of small files.
    let catalog = ozgent_core::AgentCatalog::load(&state.paths);
    let agents: Vec<ozgent_core::Agent> = catalog.mentioned(&turn.message).into_iter().cloned().collect();
    // The model may hand a request to an agent itself — but not when the
    // person already named one, and not when handing off is switched off.
    let handoff = if agents.is_empty() && state.config.lock().unwrap_or_else(|e| e.into_inner()).tools.handoff {
        catalog.all().to_vec()
    } else {
        Vec::new()
    };

    // Set before the turn is submitted, and read on the inference thread when
    // the model actually calls the tool. Safe because turns are serialised:
    // one inference thread runs one turn at a time, so the caller in force is
    // always this turn's.
    if let Some(tools) = crate::worker::current_tools(&state.tools) {
        if let Some(scheduler) = &tools.scheduler {
            let mut caller = turn.caller.clone().unwrap_or_else(|| {
                ozgent_schedule::Caller::local("web")
            });
            caller.conversation_id = Some(turn.conversation);
            // A job may never be given tools this conversation did not have.
            if caller.allowed_tools.is_none() {
                caller.allowed_tools = turn.native_tools.clone();
            }
            scheduler.set_caller(caller);
        }
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    state
        .worker
        .submit(Request {
            can_ask: turn.can_ask,
            grant: None,
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
            agents,
            tools_off: turn.tools_off,
            handoff,
            // Every conversation of the person at this machine; only its own
            // for a chat on a messaging channel.
            memory: Some(crate::memory_tool::Reach {
                conversation: turn.conversation,
                everywhere: turn.caller.as_ref().is_none_or(|c| !c.origin.starts_with("chat:")),
            }),
            out: tx,
        })
        // The inference thread is gone, which is ozgent's problem, not the
        // caller's.
        .map_err(|e| anyhow::anyhow!(e))?;

    Ok(relay(state.clone(), turn.conversation, rx))
}

/// Where the reply has got to, in the units the browser slices text by.
///
/// UTF-16, because that is what a JavaScript string index counts. A byte or
/// character offset would put a card in the middle of a word the first time a
/// reply contained an emoji.
fn offset(answer: &str) -> usize {
    answer.encode_utf16().count()
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
        // What each tool returned, whole: stored beside the reply so the next
        // turn, and `memory`, can find it (see `store_tool_results`).
        let mut called: Vec<(String, String, serde_json::Value)> = Vec::new();
        let mut returned: Vec<(String, String, serde_json::Value, serde_json::Value)> = Vec::new();
        // The agent running now, so its calls and reasoning are filed under it.
        let mut agent: Option<usize> = None;
        // When the reply's own reasoning began and last grew, for "thought for
        // 12s" — measured here because the page that watched it may be gone.
        let mut reasoned: Option<(std::time::Instant, std::time::Instant)> = None;
        let mut stats: Option<serde_json::Value> = None;
        // Held back until the reply is stored, so a page that reloads the
        // conversation the moment it sees the end finds the reply there.
        let mut last: Option<Event> = None;

        while let Some(event) = rx.recv().await {
            // Timed across the reply's and any agent's reasoning alike: the
            // page shows both in the one pane this duration labels.
            if let Event::Thinking { text } = &event {
                if !text.trim().is_empty() {
                    let now = std::time::Instant::now();
                    reasoned = Some((reasoned.map_or(now, |(first, _)| first), now));
                }
            }
            match &event {
                Event::Answer { text } => answer.push_str(text),
                Event::Done { generated, tokens_per_second, reused, prompt, prompt_ms, .. } => {
                    stats = Some(serde_json::json!({
                        "generated": generated,
                        "tokens_per_second": (tokens_per_second * 10.0).round() / 10.0,
                        "prompt": prompt,
                        "prompt_ms": prompt_ms,
                        "reused": reused,
                        "thinking_ms": reasoned.map(|(a, b)| (b - a).as_millis() as u64),
                    }));
                }
                Event::Thinking { text } => match agent {
                    // An agent's reasoning belongs in its own block, not in the
                    // message's: a reload has to put it back where it was.
                    Some(i) => {
                        let slot = &mut activity[i]["thinking"];
                        let mut so_far = slot.as_str().unwrap_or_default().to_string();
                        so_far.push_str(text);
                        *slot = so_far.into();
                    }
                    None => thinking.push_str(text),
                },
                Event::ToolCall { id, name, arguments } => {
                    called.push((id.clone(), name.clone(), arguments.clone()));
                    let mut call = serde_json::json!({
                        "kind": "tool",
                        "name": name,
                        "arguments": arguments,
                        // Where in the reply the call happened, so a reload
                        // puts the card between the right paragraphs instead
                        // of stacking every card above the whole answer.
                        "at": offset(&answer),
                    });
                    if let Some(i) = agent {
                        call["agent"] = activity[i]["name"].clone();
                    }
                    activity.push(call);
                }
                Event::ToolResult { id, name, ok, summary, ms, detail } => {
                    if *ok {
                        let arguments = called.iter().rev().find(|(i, ..)| i == id).map(|(.., a)| a.clone()).unwrap_or_default();
                        returned.push((id.clone(), name.clone(), arguments, detail.clone()));
                    }
                    // Attach to the call this answers, so a reload replays the
                    // pair rather than two loose halves.
                    let slot = activity.iter_mut().rev().find(|c| {
                        c["kind"] == "tool" && c["name"] == name.as_str() && c.get("ok").is_none()
                    });
                    match slot {
                        Some(call) => {
                            call["ok"] = (*ok).into();
                            call["ms"] = (*ms).into();
                            call["summary"] = summary.clone().into();
                            call["detail"] = crate::api::bounded(detail);
                        }
                        // A call that was never announced — one refused, or
                        // one the agent was not offered — still happened, and
                        // the transcript should say so.
                        None => {
                            let mut call = serde_json::json!({
                                "kind": "tool", "name": name, "arguments": {},
                                "at": offset(&answer), "ok": ok, "ms": ms,
                                "summary": summary, "detail": crate::api::bounded(detail),
                            });
                            if let Some(i) = agent {
                                call["agent"] = activity[i]["name"].clone();
                            }
                            activity.push(call);
                        }
                    }
                }
                Event::AgentStart { name, description, tools, missing } => {
                    // Reports are separated in the stored text so the model,
                    // reading this turn back later, sees two answers rather
                    // than one run together.
                    if !answer.trim().is_empty() {
                        answer.push_str("\n\n");
                    }
                    activity.push(serde_json::json!({
                        "kind": "agent",
                        "name": name,
                        "description": description,
                        "tools": tools,
                        "missing": missing,
                        "at": offset(&answer),
                    }));
                    agent = Some(activity.len() - 1);
                }
                Event::AgentEnd { ok, ms, calls, rounds, .. } => {
                    if let Some(i) = agent.take() {
                        let block = &mut activity[i];
                        block["end"] = offset(&answer).into();
                        block["ok"] = (*ok).into();
                        block["ms"] = (*ms).into();
                        block["calls"] = (*calls).into();
                        block["rounds"] = (*rounds).into();
                    }
                }
                _ => {}
            }
            let finished = matches!(event, Event::Done { .. } | Event::Error { .. });
            if finished {
                last = Some(event);
                break;
            }
            // A send failure means the caller went away. Stop relaying, but
            // fall through to persist whatever arrived first.
            if out_tx.send(event).is_err() {
                break;
            }
        }
        // Dropping this end is what tells the inference thread to stop, so a
        // caller that disappeared mid-generation still frees the GPU.
        drop(rx);

        // Always said last, whatever became of the reply.
        // A stream that ended with neither `done` nor `error` means the model's
        // thread went away mid-reply. Said, so the page does not simply stop.
        let finish = |last: Option<Event>| {
            let event = last.unwrap_or_else(|| Event::Error {
                message: "The model stopped before finishing this reply. Send it again to retry.".into(),
            });
            let _ = out_tx.send(event);
        };
        if answer.trim().is_empty() {
            finish(last);
            return;
        }
        let store = state.store.lock().unwrap_or_else(|e| e.into_inner());
        let trace = (!thinking.trim().is_empty()).then(|| thinking.trim().to_string());
        let calls =
            (!activity.is_empty()).then(|| serde_json::to_string(&activity).unwrap_or_default());
        store_tool_results(&state, &store, conversation, &returned);
        // Only the end is trimmed: the offsets recorded above count from the
        // start of the reply, and trimming it there would shift every card.
        match store.append_message_full(
            conversation,
            "assistant",
            answer.trim_end(),
            trace.as_deref(),
            calls.as_deref(),
            None,
            0,
        ) {
            Ok(id) => {
                if let Some(stats) = &stats {
                    let _ = store.set_message_stats(id, &stats.to_string());
                }
                crate::memory::embed_later(&state, id, answer.trim().to_string());
            }
            Err(e) => tracing::error!("persisting the reply: {e}"),
        }
        drop(store);
        finish(last);
    });

    out_rx
}

#[cfg(test)]
mod carried {
    use super::*;

    fn row(content: &str, call: &str) -> ozgent_memory::StoredMessage {
        ozgent_memory::StoredMessage {
            id: 1, conversation_id: 1, seq: 1, role: "tool".into(), content: content.into(), thinking: None,
            tool_calls: Some(call.into()), tool_call_id: None, media: None, stats: None, tokens: 0, created_at: 0,
        }
    }

    #[test]
    fn the_last_turns_results_are_carried_within_a_share_each() {
        assert!(carried_material(&[]).is_none(), "a turn without tools carries nothing");
        let long = "x ".repeat(20_000);
        let rows = [
            row("ph50ridt", r#"{"name":"ghostcloak_session_create","arguments":{}}"#),
            row(&"Reuters: tariffs paused. ".repeat(10), r#"{"name":"ghostcloak_page_snapshot","arguments":{"url":"https://reuters.com"}}"#),
            row(&long, r#"{"name":"fetch_url","arguments":{"url":"https://bloomberg.com"}}"#),
        ];
        let block = carried_material(&rows).unwrap();
        assert!(block.contains("ghostcloak_page_snapshot https://reuters.com: Reuters: tariffs paused"), "{block}");
        assert!(!block.contains("ph50ridt"), "an id is not material: {block}");
        assert!(block.contains("fetch_url https://bloomberg.com"), "{block}");
        assert!(block.chars().count() < CARRIED_CHARS + 600, "bounded: {}", block.chars().count());
    }

    #[test]
    fn the_history_grows_with_the_window_within_limits() {
        assert_eq!(budget_for(65_536).total, 16_384);
        assert_eq!(budget_for(8_192).total, 4096);
        assert_eq!(budget_for(262_144).total, 32_768);
    }
}

#[cfg(test)]
mod tests {
    use super::system_prompt;
    use ozgent_schedule::Caller;

    fn job(name: &str) -> Caller {
        Caller { origin: format!("job:{name}"), ..Default::default() }
    }

    #[test]
    fn a_scheduled_run_is_told_that_its_timer_has_already_fired() {
        // Without this the model reads "at 10am on weekdays" as something to
        // arrange, reaches for the schedule tool, and the brief never gets
        // written.
        let text = system_prompt(false, Some(&job("daily-india-market-analysis"))).unwrap();
        assert!(text.contains("daily-india-market-analysis"), "{text}");
        assert!(text.contains("not a request"), "{text}");
    }

    #[test]
    fn a_person_at_a_keyboard_is_not_told_any_of_that() {
        assert_eq!(system_prompt(false, None), None);
        let chat = Caller { origin: "chat:telegram:42".into(), ..Default::default() };
        assert_eq!(system_prompt(false, Some(&chat)), None);
    }

    #[test]
    fn the_date_line_and_the_job_line_are_both_given_when_both_apply() {
        let text = system_prompt(true, Some(&job("brief"))).unwrap();
        assert_eq!(text.lines().count(), 2, "{text}");
    }

    #[test]
    fn date_awareness_alone_is_unchanged_by_any_of_this() {
        let text = system_prompt(true, None).unwrap();
        assert!(text.starts_with("Today is "), "{text}");
        assert!(text.contains("time zone"), "the zone must be named: {text}");
        assert_eq!(text.lines().count(), 1, "{text}");
    }
}


/// Characters of one tool result kept in the conversation.
const TOOL_RESULT_CHARS: usize = 24_000;

/// Store what the turn's tools returned, as `tool` rows just before the
/// reply: never replayed as turns, but carried into the next turn (see
/// `carried_material`) and searchable with `memory`, by word at once and by
/// meaning once embedded, in the background.
fn store_tool_results(
    state: &State,
    store: &ozgent_memory::Store,
    conversation: i64,
    returned: &[(String, String, serde_json::Value, serde_json::Value)],
) {
    for (id, name, arguments, detail) in returned {
        // Looking things up in memory is not something to remember.
        if name == crate::memory_tool::NAME {
            continue;
        }
        let text = match detail {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        if text.trim().is_empty() || text == "null" {
            continue;
        }
        let text: String = text.chars().take(TOOL_RESULT_CHARS).collect();
        let call = serde_json::json!({ "name": name, "arguments": arguments }).to_string();
        match store.append_message_full(conversation, "tool", &text, None, Some(&call), Some(id), 0) {
            // Embedded now only where that takes milliseconds. On the CPU it
            // took seconds, and the next message's tool lookup queued behind
            // it; there the row is found by its words at once and embedded by
            // the next catch-up pass (`memory::backfill`).
            Ok(row) if embedder_on_gpu() => crate::memory::embed_later(state, row, text),
            Ok(_) => {}
            Err(e) => tracing::debug!("storing a tool result: {e}"),
        }
    }
}


/// The last turn's tool results, as a block after the reply they informed:
/// each tool, what it was given, and as much of what it returned as fits in
/// an even share of [`CARRIED_CHARS`]. `None` when the turn used no tools.
fn carried_material(rows: &[ozgent_memory::StoredMessage]) -> Option<String> {
    // A session id, a page id, "ok": what a browser returns for opening a
    // page is not material. What was read is.
    let rows: Vec<&ozgent_memory::StoredMessage> = rows.iter().filter(|r| r.content.trim().chars().count() >= TRIVIAL_CHARS).collect();
    if rows.is_empty() {
        return None;
    }
    // Shared in proportion to length, so one long page is not cut to the
    // size of a short one, and every result keeps at least a little.
    let total: usize = rows.iter().map(|r| r.content.chars().count()).sum();
    let mut out = String::from(
        "[What the tools returned for that reply. Answer follow-up questions about it from here; \
         use tools again only for something newer or not covered. Older results: the memory tool.]",
    );
    for row in rows {
        let call: serde_json::Value = row.tool_calls.as_deref().and_then(|c| serde_json::from_str(c).ok()).unwrap_or_default();
        let name = call.get("name").and_then(|n| n.as_str()).unwrap_or("tool");
        let lead = call
            .get("arguments")
            .and_then(|a| a.as_object())
            .and_then(|a| ["url", "query", "path", "command", "symbol"].iter().find_map(|k| a.get(*k).and_then(|v| v.as_str())))
            .unwrap_or("");
        let flat: String = row.content.split_whitespace().collect::<Vec<_>>().join(" ");
        let len = flat.chars().count();
        let share = (CARRIED_CHARS * len / total.max(1)).max(400);
        let body: String = flat.chars().take(share).collect();
        let more = if len > share { " …" } else { "" };
        out.push_str(&format!("\n- {name} {lead}: {body}{more}"));
    }
    Some(out)
}

/// A tool result shorter than this is an acknowledgement, not something read.
const TRIVIAL_CHARS: usize = 80;

fn embedder_on_gpu() -> bool {
    crate::worker::embed_status().device.as_deref() == Some("gpu")
}
