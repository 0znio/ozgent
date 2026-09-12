//! `/scheduler`: the page that owns scheduled jobs.
//!
//! A job can be made from a chat, and for the common case that is the nicest
//! way — you are already talking about the thing. But a chat is a bad place to
//! *review* fifteen jobs, see which one has been quietly failing since Tuesday,
//! or read what last Thursday's brief actually said. That is what this is for,
//! and why it does more than the tool does: the tool is for the sentence you
//! are in the middle of, the page is for everything else.
//!
//! # Why this is not behind the admin password
//!
//! `/admin` guards the gateway and model downloads because those do things the
//! rest of the interface cannot: decide who in the world may reach this
//! machine, and write gigabytes to disk. Scheduling does nothing the chat page
//! does not already do — anyone who can open `/` can run every tool
//! interactively, right now, without waiting for 9:20. Putting a password in
//! front of the scheduler would suggest a boundary that is not there.
//!
//! The boundary that *is* there is the one the channels draw: a job created
//! from a chat can only answer back into that chat, and only with the tools
//! that chat had. See [`ozgent_schedule::tools`].

use axum::extract::{Path, State as AxumState};
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use ozgent_core::schedule::Recur;
use ozgent_memory::jobs::Job;
use ozgent_schedule::{Change, Deliver, Draft, Problem, in_words, unix_now};
use serde::{Deserialize, Serialize};

use crate::api::ApiError;
use crate::state::State;

pub fn router(state: State) -> Router {
    Router::new()
        .route("/scheduler", get(page))
        .route("/scheduler.js", get(script))
        .route("/api/scheduler", get(list).post(create))
        .route("/api/scheduler/preview", post(preview))
        .route(
            "/api/scheduler/{name}",
            get(show).put(change).delete(remove),
        )
        .route("/api/scheduler/{name}/run", post(run_now))
        .with_state(state)
}

async fn page() -> Html<&'static str> {
    Html(include_str!("../assets/scheduler.html"))
}

async fn script() -> impl IntoResponse {
    (
        [("content-type", "text/javascript; charset=utf-8")],
        include_str!("../assets/scheduler.js"),
    )
}

// --------------------------------------------------------------- shapes

/// A job as the page shows it.
#[derive(Serialize)]
struct JobView {
    name: String,
    uuid: String,
    enabled: bool,
    prompt: String,
    agent: Option<String>,
    model: Option<String>,
    /// The rule in its canonical form, which is what the form edits.
    when: String,
    /// The same rule in words, which is what the list shows.
    when_words: String,
    zone: String,
    only_if: Option<String>,
    tools: Option<Vec<String>>,
    deliver: String,
    deliver_to: Option<String>,
    next_run_at: Option<i64>,
    /// "in 4 hours", so the page does not have to do date arithmetic.
    next_in: Option<String>,
    last_run_at: Option<i64>,
    runs: i64,
    failures: i64,
    created_by: String,
    /// How the last run went, so a job that has been failing since Tuesday is
    /// visible in the list rather than only on its own page.
    last_status: Option<String>,
    /// The conversation its runs are written to, for a link into the chat.
    conversation: Option<String>,
}

/// One run, as the history shows it.
#[derive(Serialize)]
struct RunView {
    started_at: i64,
    finished_at: Option<i64>,
    status: String,
    output: Option<String>,
    error: Option<String>,
    delivered: bool,
}

/// What the page needs to draw itself.
#[derive(Serialize)]
struct Overview {
    jobs: Vec<JobView>,
    /// Whether this process is the one that runs jobs.
    hosted: bool,
    /// Who does, when it is not this process — so a page that shows jobs
    /// never running has a reason to give.
    elsewhere: Option<String>,
    /// Whether a gateway is running here at all. Without one, a job set to
    /// deliver to Telegram will fail, and the form should say so first.
    gateway: bool,
    /// Which channels are connected and could actually receive a delivery.
    channels: Vec<String>,
    /// Who each channel admits, so the form can say where "everyone allowed"
    /// actually goes rather than leaving it abstract.
    allowed: std::collections::BTreeMap<String, Vec<String>>,
    /// The machine's zone, named, so the form can say what "local" means.
    zone: String,
    /// Agents a job can be handed to, for the picker.
    agents: Vec<String>,
    /// Every tool the model can call, for the per-job allowlist.
    tools: Vec<String>,
}

/// What the form sends. Every field optional so the same shape serves create
/// and update, and an update only carries what changed.
#[derive(Deserialize, Default)]
struct Body {
    name: Option<String>,
    prompt: Option<String>,
    when: Option<String>,
    zone: Option<String>,
    enabled: Option<bool>,
    agent: Option<String>,
    model: Option<String>,
    only_if: Option<String>,
    tools: Option<Vec<String>>,
    deliver: Option<String>,
    deliver_to: Option<String>,
}

fn view(store: &ozgent_memory::Store, job: &Job) -> JobView {
    let now = unix_now();
    let zone = ozgent_schedule::zone_of(job);
    let last_status = store
        .job_runs(job.id, 1)
        .ok()
        .and_then(|r| r.into_iter().next())
        .map(|r| r.status.as_str().to_string());
    let conversation = job
        .conversation_id
        .and_then(|id| store.get_conversation(id).ok().flatten())
        .map(|c| c.uuid);

    JobView {
        name: job.name.clone(),
        uuid: job.uuid.clone(),
        enabled: job.enabled,
        prompt: job.prompt.clone(),
        agent: job.agent.clone(),
        model: job.model.clone(),
        when: job.recur.clone(),
        when_words: ozgent_schedule::recur_of(job)
            .map(|r| r.describe(&zone))
            .unwrap_or_else(|e| format!("unreadable: {e}")),
        zone: job.zone.clone(),
        only_if: job.only_if.clone(),
        tools: job.tools.as_deref().and_then(|t| serde_json::from_str(t).ok()),
        deliver: job.deliver.clone(),
        deliver_to: job.deliver_to.clone(),
        next_run_at: job.next_run_at,
        next_in: job.next_run_at.map(|at| in_words(at, now)),
        last_run_at: job.last_run_at,
        runs: job.runs,
        failures: job.failures,
        created_by: job.created_by.clone(),
        last_status,
        conversation,
    }
}

/// A [`Problem`] as an HTTP status. A bad request is the user's to fix; a
/// store failure is ozgent's.
fn problem(e: Problem) -> ApiError {
    match e {
        Problem::NoSuchJob(_) => ApiError::not_found(e.to_string()),
        Problem::Store(_) => ApiError::internal(e.to_string()),
        _ => ApiError::bad_request(e.to_string()),
    }
}

// --------------------------------------------------------------- routes

async fn list(AxumState(state): AxumState<State>) -> Result<Json<Overview>, ApiError> {
    let gateway = state.gateway.get();
    let channels = gateway
        .map(|g| {
            let v = g.view();
            let mut live = Vec::new();
            if v.telegram.phase == crate::admin::Phase::Connected {
                live.push("telegram".to_string());
            }
            if v.whatsapp.phase == crate::admin::Phase::Connected {
                live.push("whatsapp".to_string());
            }
            live
        })
        .unwrap_or_default();

    let allowed = {
        let config = state.config.lock().unwrap();
        [
            ("telegram", ozgent_core::ChannelKind::Telegram),
            ("whatsapp", ozgent_core::ChannelKind::WhatsApp),
        ]
        .into_iter()
        .map(|(name, kind)| (name.to_string(), config.channels.access(kind).allow.to_vec()))
        .collect()
    };

    let catalog = ozgent_core::AgentCatalog::load(&state.paths);
    let agents = catalog.all().iter().map(|a| a.name.clone()).collect();
    let tools = crate::worker::current_tools(&state.tools)
        .map(|t| t.host.tools().iter().map(|s| s.name.clone()).collect())
        .unwrap_or_default();

    let store = state.store.lock().unwrap();
    let jobs = store
        .list_jobs()
        .map_err(|e| ApiError::internal(e.to_string()))?
        .iter()
        .map(|j| view(&store, j))
        .collect();

    Ok(Json(Overview {
        jobs,
        hosted: crate::scheduler::hosted(),
        elsewhere: crate::scheduler::held_elsewhere(&state),
        gateway: gateway.is_some(),
        channels,
        allowed,
        zone: ozgent_core::Zone::local().name().to_string(),
        agents,
        tools,
    }))
}

async fn show(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state.store.lock().unwrap();
    let job = store
        .job_by_name(&name)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .ok_or_else(|| problem(Problem::NoSuchJob(name.clone())))?;
    let runs: Vec<RunView> = store
        .job_runs(job.id, 30)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .into_iter()
        .map(|r| RunView {
            started_at: r.started_at,
            finished_at: r.finished_at,
            status: r.status.as_str().to_string(),
            output: r.output,
            error: r.error,
            delivered: r.delivered,
        })
        .collect();

    Ok(Json(serde_json::json!({ "job": view(&store, &job), "runs": runs })))
}

async fn create(
    AxumState(state): AxumState<State>,
    Json(body): Json<Body>,
) -> Result<Json<JobView>, ApiError> {
    let deliver = Deliver::parse(
        body.deliver.as_deref().unwrap_or("none"),
        body.deliver_to.as_deref(),
    )
    .map_err(ApiError::bad_request)?;

    let draft = Draft {
        name: body.name.unwrap_or_default(),
        prompt: body.prompt.unwrap_or_default(),
        when: body.when.unwrap_or_default(),
        zone: body.zone,
        agent: body.agent,
        model: body.model,
        only_if: body.only_if,
        tools: body.tools,
        deliver: Some(deliver),
        conversation_id: None,
        created_by: "web".into(),
    };
    let store = state.store.lock().unwrap();
    let job = ozgent_schedule::create(&store, &draft).map_err(problem)?;
    Ok(Json(view(&store, &job)))
}

async fn change(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
    Json(body): Json<Body>,
) -> Result<Json<JobView>, ApiError> {
    // The page may send a channel with no chat while the user is still
    // filling the form; that is a bad request, not a job pointed at nowhere.
    let deliver = match &body.deliver {
        Some(channel) => Some(
            Deliver::parse(channel, body.deliver_to.as_deref()).map_err(ApiError::bad_request)?,
        ),
        None => None,
    };
    let change = Change {
        name: body.name,
        prompt: body.prompt,
        when: body.when,
        zone: body.zone,
        enabled: body.enabled,
        // An absent field is left alone; an empty one is cleared. That is what
        // lets the form take an agent off a job.
        agent: body.agent.map(|a| Some(a).filter(|a| !a.trim().is_empty())),
        model: body.model.map(|m| Some(m).filter(|m| !m.trim().is_empty())),
        only_if: body.only_if.map(|c| Some(c).filter(|c| !c.trim().is_empty())),
        tools: body.tools.map(Some),
        deliver,
    };
    let store = state.store.lock().unwrap();
    let job = ozgent_schedule::update(&store, &name, &change).map_err(problem)?;
    Ok(Json(view(&store, &job)))
}

async fn remove(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state.store.lock().unwrap();
    let job = store
        .job_by_name(&name)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .ok_or_else(|| problem(Problem::NoSuchJob(name.clone())))?;
    store.delete_job(job.id).map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(serde_json::json!({ "deleted": job.name })))
}

async fn run_now(
    AxumState(state): AxumState<State>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = state.store.lock().unwrap();
    let job = store
        .job_by_name(&name)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .ok_or_else(|| problem(Problem::NoSuchJob(name.clone())))?;
    // Made due rather than run here: the scheduler runs jobs one at a time,
    // and starting a second one from an HTTP handler would be the one path
    // that ignores that.
    store
        .set_next_run(job.id, Some(unix_now()))
        .map_err(|e| ApiError::internal(e.to_string()))?;
    // Written, then said out loud. The loop is asleep on a figure worked out
    // before that write, and without this the button did nothing visible
    // until the next look — up to a minute of a page that looks broken.
    ozgent_schedule::wake();
    Ok(Json(serde_json::json!({
        "queued": job.name,
        "hosted": crate::scheduler::hosted(),
    })))
}

#[derive(Deserialize)]
struct PreviewBody {
    when: String,
    zone: Option<String>,
}

/// Read a rule back and say when it would actually fire.
///
/// The form calls this as you type. Showing the next three fires is the
/// difference between trusting `0 20 * * 1-5` and guessing at it — and it
/// catches the classic mistake of swapping the minute and hour fields before
/// the job has silently not run for a week.
async fn preview(Json(body): Json<PreviewBody>) -> Result<Json<serde_json::Value>, ApiError> {
    let (when, inline) =
        ozgent_core::schedule::split_zone(&body.when).map_err(ApiError::bad_request)?;
    let recur = Recur::parse(&when).map_err(ApiError::bad_request)?;
    let zone = match body.zone.as_deref().filter(|z| !z.is_empty()).or(inline.as_deref()) {
        None | Some("local") => ozgent_core::Zone::local(),
        Some(name) => ozgent_core::Zone::named(name).map_err(ApiError::bad_request)?,
    };
    let now = unix_now();

    let mut fires = Vec::new();
    let mut at = now;
    for _ in 0..3 {
        match recur.next_after(at, now, &zone) {
            Some(next) => {
                fires.push(serde_json::json!({ "at": next, "in": in_words(next, now) }));
                at = next;
            }
            None => break,
        }
    }
    if fires.is_empty() {
        return Err(ApiError::bad_request("that describes a time that never comes around"));
    }

    Ok(Json(serde_json::json!({
        "when": recur.to_string(),
        "words": recur.describe(&zone),
        "zone": zone.name(),
        "fires": fires,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ozgent_memory::Store;
    use ozgent_schedule::Draft;

    fn store() -> Store {
        Store::open_in_memory().unwrap()
    }

    fn draft(name: &str, when: &str) -> Draft {
        Draft {
            name: name.into(),
            prompt: "a brief".into(),
            when: when.into(),
            created_by: "web".into(),
            ..Default::default()
        }
    }

    #[test]
    fn a_job_is_shown_with_its_rule_in_words_as_well_as_stored_form() {
        // The list is unreadable without the words, and the form cannot edit
        // anything but the stored form.
        let s = store();
        let job = ozgent_schedule::create(&s, &draft("brief", "every weekday at 9:20")).unwrap();
        let v = view(&s, &job);
        assert_eq!(v.when, "cron 20 9 * * 1,2,3,4,5");
        assert!(v.when_words.contains("every weekday"), "{}", v.when_words);
        assert!(v.next_in.is_some());
    }

    #[test]
    fn a_job_whose_rule_went_bad_still_renders_rather_than_breaking_the_page() {
        let s = store();
        let job = ozgent_schedule::create(&s, &draft("brief", "every day at 9:20")).unwrap();
        s.update_job(
            job.id,
            &ozgent_memory::JobEdit { recur: Some("gibberish".into()), ..Default::default() },
        )
        .unwrap();
        let job = s.get_job(job.id).unwrap().unwrap();
        let v = view(&s, &job);
        assert!(v.when_words.contains("unreadable"), "{}", v.when_words);
    }

    #[test]
    fn the_last_run_is_carried_into_the_list_so_a_failing_job_is_visible() {
        let s = store();
        let job = ozgent_schedule::create(&s, &draft("brief", "every day at 9:20")).unwrap();
        assert_eq!(view(&s, &job).last_status, None, "nothing has run yet");

        let run = s.start_run(job.id, 1_000).unwrap();
        s.finish_run(run, ozgent_memory::JobStatus::Error, None, Some("no model"), false).unwrap();
        assert_eq!(view(&s, &job).last_status.as_deref(), Some("error"));
    }

    #[tokio::test]
    async fn previewing_a_rule_says_when_it_would_actually_fire() {
        // The point of the preview: seeing three real times catches a swapped
        // minute and hour before the job silently misses a week.
        let body = PreviewBody { when: "every weekday at 9:20".into(), zone: Some("UTC".into()) };
        let out = preview(Json(body)).await.unwrap().0;
        assert_eq!(out["when"], serde_json::json!("cron 20 9 * * 1,2,3,4,5"));
        assert_eq!(out["fires"].as_array().unwrap().len(), 3);
        assert!(out["words"].as_str().unwrap().contains("weekday"));

        // And the fires are in order and in the future.
        let fires = out["fires"].as_array().unwrap();
        let times: Vec<i64> = fires.iter().map(|f| f["at"].as_i64().unwrap()).collect();
        assert!(times.windows(2).all(|w| w[0] < w[1]), "{times:?}");
        assert!(times[0] > unix_now());
    }

    #[tokio::test]
    async fn previewing_a_one_off_shows_the_single_time_it_runs() {
        let at = unix_now() + 7 * 86_400;
        let body = PreviewBody { when: format!("once {at}"), zone: Some("UTC".into()) };
        let out = preview(Json(body)).await.unwrap().0;
        assert_eq!(out["fires"].as_array().unwrap().len(), 1, "it does not repeat");
    }

    #[tokio::test]
    async fn previewing_nonsense_is_a_bad_request_rather_than_a_crash() {
        for when in ["every blursday at 9:00", "", "0 0 31 2 *"] {
            let body = PreviewBody { when: when.into(), zone: None };
            assert!(preview(Json(body)).await.is_err(), "{when:?} was accepted");
        }
    }

    #[tokio::test]
    async fn previewing_in_an_unknown_zone_is_refused_rather_than_silently_utc() {
        let body =
            PreviewBody { when: "every day at 9:20".into(), zone: Some("Mars/Olympus".into()) };
        assert!(preview(Json(body)).await.is_err());
    }

    #[test]
    fn a_problem_maps_to_the_status_that_says_whose_fault_it_is() {
        assert_eq!(problem(Problem::NoSuchJob("x".into())).status(), 404);
        assert_eq!(problem(Problem::Invalid("x".into())).status(), 400);
        assert_eq!(problem(Problem::NameTaken("x".into())).status(), 400);
        assert_eq!(problem(Problem::Store("x".into())).status(), 500);
    }
}
