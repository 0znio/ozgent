//! Running workflows against this server's engine and tools.
//!
//! [`ozgent_flow`] owns the graph and knows nothing about how a step is
//! carried out; this is the other half — the part that has a model loaded and a
//! Python worker running.
//!
//! ## Why a workflow may not simply use any tool
//!
//! A step's arguments are a template, and a webhook fills that template with
//! data from whoever called it. So `run_command` with `{{ trigger.cmd }}` is a
//! remote shell, written entirely in the canvas, with no prompt anywhere.
//!
//! A workflow therefore runs under the same rule as the OpenAI-compatible API:
//! there is nobody to ask, and a program cannot consent on a person's behalf,
//! so anything the policy would *ask* about is refused. Reads run, because the
//! policy already says reads run. An operator who wants a workflow to write
//! files or run commands says so in `[permissions]`, deliberately, once — and
//! the refusal message says exactly that, because a workflow that stops with
//! "declined" and no explanation is a mystery rather than a boundary.

use ozgent_core::permission::Verdict;
use ozgent_flow::{Steps, engine};
use serde_json::{Value, json};

use crate::state::State;
use crate::worker::{Event, Request};

/// How many runs of one flow are kept.
pub const KEEP_RUNS: i64 = 50;

/// Why a tool that asks cannot be used by a workflow, and what to change.
///
/// One function rather than a message at each place that reports it: the
/// canvas greys the tool out and the run stops with a reason, and if those two
/// drifted apart the reason shown before the run and the reason shown after it
/// would be different explanations of the same rule.
pub fn needs_approval(tool: &str) -> String {
    format!(
        "`{tool}` asks before it runs, and a workflow runs with nobody watching. \
         To let workflows use it, set `{tool} = \"allow\"` under [permissions.tools] \
         in config.toml — which allows it everywhere, including from a webhook."
    )
}

pub struct Runner {
    pub state: State,
}

impl Runner {
    pub fn new(state: State) -> Self {
        Self { state }
    }
}

impl Steps for Runner {
    async fn tool(&self, name: &str, arguments: Value) -> Result<Value, String> {
        let tools = crate::worker::current_tools(&self.state.tools)
            .ok_or_else(|| "tools are not running on this server".to_string())?;

        let spec = tools
            .host
            .get(name)
            .ok_or_else(|| format!("there is no tool called `{name}`"))?;
        let effect = spec.effect;

        let verdict = {
            let config = self.state.config.lock().unwrap_or_else(|e| e.into_inner());
            if config.tools.disabled.contains(&name.to_string()) {
                return Err(format!("`{name}` is switched off in settings"));
            }
            let grants = self.state.permissions.grants.lock().unwrap_or_else(|e| e.into_inner());
            config.permissions.verdict(name, effect, &grants)
        };

        let approved = match verdict {
            Verdict::Allow { by_user } => by_user,
            // Said in full, because the alternative is a workflow that stops
            // for a reason the operator cannot see from the canvas.
            Verdict::Ask => return Err(needs_approval(name)),
            Verdict::Deny => return Err(format!("`{name}` is refused by [permissions]")),
        };

        tools
            .host
            .call_approved(name, arguments, approved)
            .await
            .map_err(|e| e.for_model())
    }

    async fn agent(&self, prompt: &str, model: Option<&str>) -> Result<Value, String> {
        let chosen = match model {
            Some(m) => m.to_string(),
            None => {
                let config = self.state.config.lock().unwrap_or_else(|e| e.into_inner());
                config
                    .default_model
                    .clone()
                    .ok_or_else(|| "no model is set; pick one on this step or set a default".to_string())?
            }
        };

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        self.state
            .worker
            .submit(Request {
                model: chosen,
                messages: vec![ozgent_core::Message::user(prompt)],
                thinking: None,
                max_tokens: None,
                // A step that called tools of its own would do work the graph
                // cannot see, and the graph is the point: tool calls are steps.
                tools_enabled: false,
                native_tools: None,
                client_tools: Vec::new(),
                response_grammar: None,
                overrides: None,
                images: Vec::new(),
                // Nobody is watching a workflow, so nothing may ask.
                can_ask: false,
                out: tx,
            })
            .map_err(|e| e.to_string())?;

        let mut answer = String::new();
        while let Some(event) = rx.recv().await {
            match event {
                Event::Answer { text } => answer.push_str(&text),
                Event::Error { message } => return Err(message),
                Event::Done { .. } => break,
                _ => {}
            }
        }

        let text = answer.trim().to_string();
        if text.is_empty() {
            return Err("the model returned nothing".into());
        }
        Ok(json!({ "text": text }))
    }
}

/// Record a finished run, trimming the oldest.
pub fn remember(state: &State, flow_id: i64, run: &engine::Run) -> anyhow::Result<i64> {
    let record = serde_json::to_string(run)?;
    let status = match run.status {
        engine::Status::Ok => "ok",
        engine::Status::Failed => "failed",
    };
    let store = state.store.lock().unwrap();
    Ok(store.add_run(flow_id, &run.trigger, status, &record, run.ms as i64, KEEP_RUNS)?)
}

#[cfg(test)]
mod tests {
    use super::needs_approval;

    #[test]
    fn the_refusal_names_the_setting_that_would_allow_it() {
        // A workflow that stops with "declined" and nothing else is a mystery
        // rather than a boundary: nothing on the canvas says what to change.
        let message = needs_approval("run_command");
        assert!(message.contains("run_command = \"allow\""), "{message}");
        assert!(message.contains("[permissions.tools]"), "{message}");
        assert!(message.contains("config.toml"), "{message}");
    }

    #[test]
    fn the_refusal_says_what_the_change_costs() {
        // Allowing a tool for workflows allows it for the webhook that starts
        // one, which is the part someone would not think of.
        assert!(needs_approval("write_file").contains("including from a webhook"));
    }
}

// ------------------------------------------------------------------- routes

use axum::extract::{Path, State as AxumState};
use axum::response::sse::{Event as SseEvent, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream::Stream;
use ozgent_flow::{Flow, Progress};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

use crate::api::{ApiError, ApiResult};

pub fn routes() -> Router<State> {
    Router::new()
        .route("/api/flows", get(list).post(create))
        .route("/api/flows/palette", get(palette))
        .route(
            "/api/flows/{id}",
            get(one).put(save).delete(remove),
        )
        .route("/api/flows/{id}/run", post(run))
        .route("/api/flows/{id}/runs", get(runs))
        // A webhook is addressed by the flow's public uuid and the trigger
        // step inside it, so one flow can have several ways in and a row id is
        // never exposed.
        .route("/hooks/{uuid}/{node}", post(hook).get(hook_get))
}

#[derive(Serialize)]
struct FlowSummary {
    id: i64,
    uuid: String,
    name: String,
    enabled: bool,
    updated_at: i64,
    /// Whether it could run as it stands. The list greys out the ones that
    /// cannot, so a broken flow is visible before it is opened.
    valid: bool,
    steps: usize,
}

fn summarise(row: &ozgent_memory::StoredFlow) -> FlowSummary {
    let parsed: Option<Flow> = serde_json::from_str(&row.definition).ok();
    FlowSummary {
        id: row.id,
        uuid: row.uuid.clone(),
        name: row.name.clone(),
        enabled: row.enabled,
        updated_at: row.updated_at,
        valid: parsed.as_ref().is_some_and(|f| f.is_valid()),
        steps: parsed.as_ref().map(|f| f.nodes.len()).unwrap_or(0),
    }
}

async fn list(AxumState(state): AxumState<State>) -> ApiResult<Json<Vec<FlowSummary>>> {
    let store = state.store.lock().unwrap();
    Ok(Json(store.list_flows()?.iter().map(summarise).collect()))
}

#[derive(Deserialize)]
struct NewFlow {
    #[serde(default)]
    name: String,
}

async fn create(
    AxumState(state): AxumState<State>,
    Json(body): Json<NewFlow>,
) -> ApiResult<Json<FlowSummary>> {
    let name = if body.name.trim().is_empty() { "New workflow" } else { body.name.trim() };
    // A new flow already has its trigger. An empty canvas gives no clue what
    // to do first, and every flow needs one anyway.
    let starter = Flow {
        name: name.to_string(),
        description: String::new(),
        nodes: vec![ozgent_flow::Node {
            id: "start".into(),
            kind: ozgent_flow::Kind::Manual,
            name: "When I press run".into(),
            x: 80.0,
            y: 160.0,
            params: Default::default(),
        }],
        edges: Vec::new(),
    };
    let store = state.store.lock().unwrap();
    let row = store.create_flow(name, &serde_json::to_string(&starter)?)?;
    Ok(Json(summarise(&row)))
}

#[derive(Serialize)]
struct FlowDetail {
    id: i64,
    uuid: String,
    name: String,
    enabled: bool,
    /// The graph itself, as the canvas edits it.
    definition: serde_json::Value,
    /// Everything wrong with it, in the order a person would fix them.
    problems: Vec<String>,
}

async fn one(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
) -> ApiResult<Json<FlowDetail>> {
    let store = state.store.lock().unwrap();
    let row = store
        .get_flow(id)?
        .ok_or_else(|| ApiError::bad_request("no workflow with that id"))?;

    let parsed: Flow = serde_json::from_str(&row.definition).unwrap_or_default();
    Ok(Json(FlowDetail {
        id: row.id,
        uuid: row.uuid,
        name: row.name,
        enabled: row.enabled,
        definition: serde_json::to_value(&parsed)?,
        problems: parsed.problems().iter().map(|p| p.to_string()).collect(),
    }))
}

#[derive(Deserialize)]
struct SaveFlow {
    definition: Flow,
    #[serde(default)]
    enabled: bool,
}

async fn save(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
    Json(body): Json<SaveFlow>,
) -> ApiResult<Json<FlowDetail>> {
    let problems: Vec<String> = body.definition.problems().iter().map(|p| p.to_string()).collect();
    // Saved even when it does not yet run: half-drawn is the normal state of
    // something being drawn, and refusing to save it would lose the work. What
    // `enabled` gates is whether it *fires*.
    let enabled = body.enabled && problems.is_empty();

    let store = state.store.lock().unwrap();
    let row = store
        .get_flow(id)?
        .ok_or_else(|| ApiError::bad_request("no workflow with that id"))?;
    let name = if body.definition.name.trim().is_empty() {
        row.name.clone()
    } else {
        body.definition.name.trim().to_string()
    };
    store.save_flow(id, &name, &serde_json::to_string(&body.definition)?, enabled)?;

    Ok(Json(FlowDetail {
        id,
        uuid: row.uuid,
        name,
        enabled,
        definition: serde_json::to_value(&body.definition)?,
        problems,
    }))
}

async fn remove(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
) -> ApiResult<axum::http::StatusCode> {
    state.store.lock().unwrap().delete_flow(id)?;
    Ok(axum::http::StatusCode::NO_CONTENT)
}

#[derive(Serialize)]
struct RunSummary {
    id: i64,
    trigger: String,
    status: String,
    ms: i64,
    created_at: i64,
    record: serde_json::Value,
}

async fn runs(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
) -> ApiResult<Json<Vec<RunSummary>>> {
    let store = state.store.lock().unwrap();
    Ok(Json(
        store
            .runs_for(id, 25)?
            .into_iter()
            .map(|r| RunSummary {
                id: r.id,
                trigger: r.trigger,
                status: r.status,
                ms: r.ms,
                created_at: r.created_at,
                record: serde_json::from_str(&r.record).unwrap_or(serde_json::Value::Null),
            })
            .collect(),
    ))
}

/// What the canvas offers in its "add a step" menu.
///
/// The tools come from the running Python worker rather than from a list kept
/// here, which is the whole reason a workflow needs no connector system of its
/// own: a tool dropped into `~/ozgent/tools` is a node the next time the page
/// is opened, with a form built from the schema it already declares.
#[derive(Serialize)]
struct Palette {
    tools: Vec<PaletteTool>,
    models: Vec<String>,
}

#[derive(Serialize)]
struct PaletteTool {
    name: String,
    description: String,
    effect: String,
    /// Whether a workflow may actually use it, given `[permissions]`.
    usable: bool,
    /// Why not, when it cannot.
    #[serde(skip_serializing_if = "Option::is_none")]
    blocked: Option<String>,
    input_schema: serde_json::Value,
}

async fn palette(AxumState(state): AxumState<State>) -> ApiResult<Json<Palette>> {
    let config = state.config.lock().unwrap().clone();
    let mut tools = Vec::new();

    if let Some(running) = crate::worker::current_tools(&state.tools) {
        let grants = state.permissions.grants.lock().unwrap().clone();
        for spec in running.host.tools() {
            if config.tools.disabled.contains(&spec.name) {
                continue;
            }
            let verdict = config.permissions.verdict(&spec.name, spec.effect, &grants);
            let (usable, blocked) = match verdict {
                Verdict::Allow { .. } => (true, None),
                Verdict::Ask => (false, Some(needs_approval(&spec.name))),
                Verdict::Deny => (false, Some("refused in Permissions".to_string())),
            };
            tools.push(PaletteTool {
                name: spec.name.clone(),
                description: ozgent_tools::first_line(&spec.description).to_string(),
                effect: spec.effect.to_string(),
                usable,
                blocked,
                input_schema: serde_json::to_value(&spec.input_schema).unwrap_or_default(),
            });
        }
    }
    tools.sort_by(|a, b| a.name.cmp(&b.name));

    let models = ozgent_core::installed(&state.paths)
        .into_iter()
        .map(|m| m.model.to_string())
        .collect();

    Ok(Json(Palette { tools, models }))
}

// ---------------------------------------------------------------- running

#[derive(Deserialize, Default)]
struct RunRequest {
    /// Which trigger to start from. The only trigger, usually.
    #[serde(default)]
    trigger: Option<String>,
    /// Data the trigger hands to the first step.
    #[serde(default)]
    payload: Option<serde_json::Value>,
}

async fn run(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
    body: Option<Json<RunRequest>>,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    let Json(body) = body.unwrap_or_default();
    let row = {
        let store = state.store.lock().unwrap();
        store.get_flow(id)?.ok_or_else(|| ApiError::bad_request("no workflow with that id"))?
    };
    let flow: Flow = serde_json::from_str(&row.definition)
        .map_err(|e| ApiError::bad_request(format!("this workflow is unreadable: {e}")))?;

    let trigger = match body.trigger {
        Some(t) => t,
        None => flow
            .triggers()
            .next()
            .map(|n| n.id.clone())
            .ok_or_else(|| ApiError::bad_request("this workflow has nothing to start it"))?,
    };
    let payload = body.payload.unwrap_or_else(|| serde_json::json!({}));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<serde_json::Value>();
    tokio::spawn(async move {
        let sender = tx.clone();
        let report = move |progress: Progress| {
            let message = match progress {
                Progress::Started { id } => serde_json::json!({ "type": "started", "id": id }),
                Progress::Finished(record) => {
                    serde_json::json!({ "type": "finished", "step": record })
                }
            };
            let _ = sender.send(message);
        };

        let runner = Runner::new(state.clone());
        let outcome = ozgent_flow::execute(&flow, &trigger, payload, &runner, Some(&report)).await;

        let final_message = match outcome {
            Ok(run) => {
                // Persisted before it is announced, so a page that reloads on
                // seeing "done" finds the run in the history.
                match remember(&state, id, &run) {
                    Ok(run_id) => {
                        serde_json::json!({ "type": "done", "run_id": run_id, "run": run })
                    }
                    Err(e) => {
                        tracing::error!("recording a workflow run: {e}");
                        serde_json::json!({ "type": "done", "run": run })
                    }
                }
            }
            Err(refused) => serde_json::json!({ "type": "refused", "message": refused.to_string() }),
        };
        let _ = tx.send(final_message);
    });

    let stream = futures_util::stream::unfold(rx, |mut rx| async move {
        let message = rx.recv().await?;
        let json = serde_json::to_string(&message).unwrap_or_default();
        Some((Ok(SseEvent::default().data(json)), rx))
    });
    Ok(Sse::new(stream))
}

// --------------------------------------------------------------- webhooks

async fn hook_get(
    state: AxumState<State>,
    path: Path<(String, String)>,
) -> ApiResult<Json<serde_json::Value>> {
    hook(state, path, Json(serde_json::json!({}))).await
}

/// Start a flow because something called its URL.
///
/// The flow must be enabled. Saving a flow does not open its webhook — the
/// switch does, and it is off until someone turns it on, because a URL that
/// exists as soon as a step is dropped on a canvas is a URL nobody knows is
/// live.
async fn hook(
    AxumState(state): AxumState<State>,
    Path((uuid, node)): Path<(String, String)>,
    Json(payload): Json<serde_json::Value>,
) -> ApiResult<Json<serde_json::Value>> {
    let row = {
        let store = state.store.lock().unwrap();
        store.flow_by_uuid(&uuid)?
    };
    // The same answer for "no such flow" and "not switched on": telling the
    // difference would let someone probe for workflows by their identifier.
    let row = row
        .filter(|r| r.enabled)
        .ok_or_else(|| ApiError::bad_request("no workflow is listening there"))?;

    let flow: Flow = serde_json::from_str(&row.definition)
        .map_err(|e| ApiError::bad_request(format!("this workflow is unreadable: {e}")))?;

    let is_webhook = flow
        .node(&node)
        .is_some_and(|n| n.kind == ozgent_flow::Kind::Webhook);
    if !is_webhook {
        return Err(ApiError::bad_request("no workflow is listening there"));
    }

    let runner = Runner::new(state.clone());
    let run = ozgent_flow::execute(&flow, &node, payload, &runner, None)
        .await
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let run_id = remember(&state, row.id, &run).ok();

    Ok(Json(serde_json::json!({
        "status": match run.status {
            ozgent_flow::Status::Ok => "ok",
            ozgent_flow::Status::Failed => "failed",
        },
        "run_id": run_id,
        "ms": run.ms,
        "error": run.failed().and_then(|s| s.error.clone()),
    })))
}

// -------------------------------------------------------------- scheduling

/// Fire scheduled triggers.
///
/// One task for the whole server, waking every [`TICK`] seconds. A timer per
/// flow would be tidier to describe and much worse to live with: flows are
/// edited while running, and a timer is then either stale or has to be found
/// and cancelled.
///
/// Schedules are held in memory only. A restart re-arms every schedule from
/// the moment it starts, so a machine that was off overnight does not wake up
/// and fire a day of missed runs at once — which is the behaviour worth having,
/// and it is stated in the documentation rather than left to be discovered.
pub const TICK: std::time::Duration = std::time::Duration::from_secs(15);

pub fn watch_schedules(state: State) {
    tokio::spawn(async move {
        // (flow, step) → what it is set to, and when it next fires.
        let mut armed: std::collections::HashMap<(i64, String), (ozgent_flow::schedule::Schedule, i64)> =
            std::collections::HashMap::new();
        // Flows with a run in flight, so a slow flow on a short interval does
        // not stack up runs behind itself.
        let busy: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<i64>>> =
            Default::default();

        loop {
            tokio::time::sleep(TICK).await;
            let now = ozgent_core::DateTime::now();
            let unix = unix_now();

            let flows = {
                let store = state.store.lock().unwrap();
                store.enabled_flows().unwrap_or_default()
            };
            let _ = now;

            // Anything no longer enabled stops being armed, so a flow switched
            // off does not fire once more from a stale entry.
            let live: std::collections::HashSet<i64> = flows.iter().map(|f| f.id).collect();
            armed.retain(|(flow, _), _| live.contains(flow));

            for row in flows {
                let Ok(flow) = serde_json::from_str::<Flow>(&row.definition) else { continue };
                // Collected first: the loop hands a copy of the flow to
                // each spawned run, and iterating the flow's own nodes would
                // borrow what is being handed over.
                let schedules: Vec<ozgent_flow::Node> = flow
                    .nodes
                    .iter()
                    .filter(|n| n.kind == ozgent_flow::Kind::Schedule)
                    .cloned()
                    .collect();
                for node in &schedules {
                    let schedule = match ozgent_flow::schedule::read(node) {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::warn!("{}: step `{}`: {e}", row.name, node.label());
                            continue;
                        }
                    };
                    let key = (row.id, node.id.clone());
                    let entry = armed.entry(key).or_insert_with(|| {
                        (schedule, ozgent_flow::schedule::next_after(schedule, unix))
                    });
                    // Edited while running: re-arm from now rather than
                    // honouring a time the old setting produced.
                    if entry.0 != schedule {
                        *entry = (schedule, ozgent_flow::schedule::next_after(schedule, unix));
                        continue;
                    }
                    if unix < entry.1 {
                        continue;
                    }
                    entry.1 = ozgent_flow::schedule::next_after(schedule, unix);

                    if !busy.lock().unwrap().insert(row.id) {
                        tracing::warn!(
                            "{}: the previous run has not finished; skipping this one",
                            row.name
                        );
                        continue;
                    }

                    let state = state.clone();
                    let busy = busy.clone();
                    let flow = flow.clone();
                    let node_id = node.id.clone();
                    let name = row.name.clone();
                    let flow_id = row.id;
                    tokio::spawn(async move {
                        let runner = Runner::new(state.clone());
                        let payload = serde_json::json!({ "fired_at": unix_now() });
                        match ozgent_flow::execute(&flow, &node_id, payload, &runner, None).await {
                            Ok(run) => {
                                if let Some(step) = run.failed() {
                                    tracing::warn!(
                                        "{name}: step `{}` failed: {}",
                                        step.label,
                                        step.error.as_deref().unwrap_or("no reason given")
                                    );
                                }
                                if let Err(e) = remember(&state, flow_id, &run) {
                                    tracing::error!("recording a workflow run: {e}");
                                }
                            }
                            Err(refused) => tracing::warn!("{name}: {refused}"),
                        }
                        busy.lock().unwrap().remove(&flow_id);
                    });
                }
            }
        }
    });
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
