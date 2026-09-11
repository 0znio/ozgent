//! Finding, downloading and removing models from the browser.
//!
//! The same machinery as `ozgent pull` — the repository listing, the
//! quantisation choice, the parallel resumable downloader — behind endpoints
//! the web interface calls.
//!
//! A download is a job that belongs to the server, not to the page that
//! started it. It keeps going when the tab is closed, any page can watch it,
//! and a reload finds it where it left off. Progress is polled rather than
//! streamed for that reason: a stream would tie the job's visibility to the
//! one connection that asked for it.
//!
//! Cancelling stops the transfer and keeps what arrived. The downloader
//! resumes from its ledger, so pulling the same thing again picks up where it
//! stopped rather than starting over.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::{Path, Query, State as AxumState};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::state::State;

/// Every download this server has started, by id.
#[derive(Default)]
pub struct Pulls {
    jobs: Mutex<BTreeMap<u64, Job>>,
    next: Mutex<u64>,
}

pub type SharedPulls = Arc<Pulls>;

/// Where a download has got to.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Reading the repository and choosing files.
    Resolving,
    Downloading,
    Done,
    Failed,
    Cancelled,
}

/// One download, as the page shows it.
#[derive(Debug, Clone, Serialize)]
pub struct Job {
    pub id: u64,
    pub repo: String,
    pub quant: Option<String>,
    /// The `name:tag` it installs as, once resolved.
    pub model: Option<String>,
    pub alias: Option<String>,
    pub status: Status,
    /// The file being fetched, and which of how many.
    pub file: Option<String>,
    pub file_index: usize,
    pub file_count: usize,
    /// Bytes across every file, including those already on disk.
    pub done: u64,
    pub total: u64,
    /// Measured over the last few seconds, not since the start, so a stall
    /// shows as one.
    pub bytes_per_second: f64,
    pub error: Option<String>,
    pub vision: bool,
    #[serde(skip)]
    task: Option<tokio::task::AbortHandle>,
    #[serde(skip)]
    finished_files: u64,
    #[serde(skip)]
    rate: Rate,
}

/// A rate over a sliding window of samples.
#[derive(Debug, Clone, Default)]
struct Rate {
    samples: std::collections::VecDeque<(Instant, u64)>,
}

impl Rate {
    /// Record `done` bytes now and return bytes per second over the window.
    fn record(&mut self, done: u64) -> f64 {
        let now = Instant::now();
        self.samples.push_back((now, done));
        while self.samples.len() > 2
            && now.duration_since(self.samples[0].0).as_secs_f64() > 5.0
        {
            self.samples.pop_front();
        }
        let (then, before) = self.samples[0];
        let secs = now.duration_since(then).as_secs_f64();
        if secs < 0.2 { 0.0 } else { done.saturating_sub(before) as f64 / secs }
    }
}

impl Pulls {
    fn update(&self, id: u64, f: impl FnOnce(&mut Job)) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(&id) {
            f(job);
        }
    }

    pub fn list(&self) -> Vec<Job> {
        self.jobs.lock().unwrap().values().cloned().collect()
    }

    /// Whether this model is being downloaded right now.
    fn active_for(&self, repo: &str, quant: Option<&str>) -> Option<u64> {
        self.jobs
            .lock()
            .unwrap()
            .values()
            .find(|j| {
                j.repo.eq_ignore_ascii_case(repo)
                    && matches!(j.status, Status::Resolving | Status::Downloading)
                    && (quant.is_none() || j.quant.as_deref().map(str::to_ascii_uppercase)
                        == quant.map(str::to_ascii_uppercase))
            })
            .map(|j| j.id)
    }
}

/// Failures, as `{"error": "..."}` with a status that says whose fault.
pub struct HubFailure(StatusCode, String);

impl IntoResponse for HubFailure {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

/// A hub error, classified: a missing repo or a gated one is the caller's to
/// fix; a network failure is not.
fn from_hub(e: ozgent_hub::HubError) -> HubFailure {
    use ozgent_hub::HubError::*;
    let status = match &e {
        NotFound { .. } | Unauthorized { .. } | Select(_) => StatusCode::BAD_REQUEST,
        RateLimited => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::BAD_GATEWAY,
    };
    HubFailure(status, e.to_string())
}

// ------------------------------------------------------------------ search

#[derive(Deserialize)]
pub struct SearchQuery {
    q: String,
    #[serde(default)]
    limit: Option<usize>,
}

/// `GET /api/hub/search?q=`: GGUF repositories on Hugging Face.
pub async fn search(Query(query): Query<SearchQuery>) -> Result<Json<serde_json::Value>, HubFailure> {
    let q = query.q.trim();
    if q.len() < 2 {
        return Ok(Json(serde_json::json!({ "results": [] })));
    }
    let client = ozgent_hub::Client::new().map_err(from_hub)?;
    let found = client.search(q, query.limit.unwrap_or(20)).await.map_err(from_hub)?;
    let results: Vec<serde_json::Value> = found
        .iter()
        .map(|f| {
            serde_json::json!({
                "id": f.id,
                "downloads": f.downloads,
                "likes": f.likes,
                "vision": f.vision(),
                "created_at": f.created_at,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({ "results": results })))
}

// ------------------------------------------------------------------ a repo

#[derive(Deserialize)]
pub struct RepoQuery {
    repo: String,
}

/// The GPU's memory, for judging what fits.
///
/// Total rather than free: the model this server has loaded is unloaded when
/// another is chosen, so the memory it occupies now is available to the next
/// one. Judging against free memory would call almost everything too big
/// while any model was loaded.
fn vram() -> Option<(u64, String)> {
    ozgent_llama::backend::devices()
        .into_iter()
        .filter(|d| d.is_gpu())
        .max_by_key(|d| d.memory_total)
        .map(|d| (d.memory_total as u64, d.description))
}

/// `GET /api/hub/repo?repo=`: what a repository offers, and what would fit.
pub async fn repo(Query(query): Query<RepoQuery>) -> Result<Json<serde_json::Value>, HubFailure> {
    let request = ozgent_hub::PullRequest::parse(&query.repo);
    if !request.repo_id.contains('/') {
        return Err(HubFailure(
            StatusCode::BAD_REQUEST,
            format!("{:?} is not a repository; they look like owner/name", query.repo),
        ));
    }
    let client = ozgent_hub::Client::new().map_err(from_hub)?;
    let info = client.repo(&request.repo_id, &request.revision).await.map_err(from_hub)?;

    let rows = ozgent_hub::quantisations(&info.files);
    let projector = info.files.iter().find(|f| f.is_mmproj()).map(|f| f.size);
    let gpu = vram();
    let memory = gpu.as_ref().map(|(bytes, _)| *bytes);
    let suggested = ozgent_hub::recommend(&rows, projector.unwrap_or(0), memory);
    let quants: Vec<serde_json::Value> = rows
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let fits = memory.map(|m| q.bytes + projector.unwrap_or(0) < m * 85 / 100);
            serde_json::json!({
                "quant": q.quant,
                "bytes": q.bytes,
                "shards": q.shards,
                "fits": fits,
                "recommended": Some(i) == suggested,
                "name": ozgent_hub::derive_ref(&info.id, &q.quant),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "id": info.id,
        "gated": info.gated,
        "vision": projector.is_some(),
        "projector_bytes": projector,
        "quants": quants,
        "gpu": gpu.map(|(bytes, name)| serde_json::json!({ "name": name, "memory": bytes })),
        // Asked here rather than guessed later: a repo whose files are not
        // GGUF cannot be pulled at all, and the page should say so up front.
        "runnable": !rows.is_empty(),
    })))
}

// ------------------------------------------------------------------- pulls

#[derive(Deserialize)]
pub struct PullBody {
    repo: String,
    #[serde(default)]
    quant: Option<String>,
    /// A short name to call it by, as `ozgent pull --name` sets.
    #[serde(default)]
    alias: Option<String>,
}

/// `POST /api/hub/pull`: start a download and return its job at once.
pub async fn pull(
    AxumState(state): AxumState<State>,
    Json(body): Json<PullBody>,
) -> Result<Json<Job>, HubFailure> {
    let mut request = ozgent_hub::PullRequest::parse(&body.repo);
    if let Some(q) = body.quant.as_deref().map(str::trim).filter(|q| !q.is_empty()) {
        request.quant = Some(q.to_string());
    }
    let alias = body.alias.as_deref().map(str::trim).filter(|a| !a.is_empty()).map(str::to_string);
    // Checked before a byte is fetched: finding the clash after a
    // multi-gigabyte download is a needlessly expensive way to learn it.
    if let Some(a) = &alias {
        ozgent_core::validate_alias(&state.paths, a, None)
            .map_err(|e| HubFailure(StatusCode::BAD_REQUEST, e.to_string()))?;
    }
    // Two jobs writing the same files would corrupt both.
    if let Some(id) = state.pulls.active_for(&request.repo_id, request.quant.as_deref()) {
        return Err(HubFailure(
            StatusCode::CONFLICT,
            format!("{} is already downloading (job {id})", request.repo_id),
        ));
    }

    let id = {
        let mut next = state.pulls.next.lock().unwrap();
        *next += 1;
        *next
    };
    let job = Job {
        id,
        repo: request.repo_id.clone(),
        quant: request.quant.clone(),
        model: None,
        alias: alias.clone(),
        status: Status::Resolving,
        file: None,
        file_index: 0,
        file_count: 0,
        done: 0,
        total: 0,
        bytes_per_second: 0.0,
        error: None,
        vision: false,
        task: None,
        finished_files: 0,
        rate: Rate::default(),
    };
    state.pulls.jobs.lock().unwrap().insert(id, job);

    let pulls = Arc::clone(&state.pulls);
    let paths = state.paths.clone();
    let task = tokio::spawn(async move {
        let outcome = run(&pulls, &paths, id, &request, alias.as_deref()).await;
        if outcome.is_ok() {
            // An earlier attempt at the same model that was cancelled or
            // failed is settled by this one; leaving it offering "Resume"
            // would invite downloading what is already installed.
            let mut jobs = pulls.jobs.lock().unwrap();
            let finished = jobs.get(&id).map(|j| (j.repo.to_ascii_lowercase(), j.quant.clone()));
            if let Some((repo, quant)) = finished {
                jobs.retain(|other, j| {
                    *other == id
                        || !(j.repo.to_ascii_lowercase() == repo
                            && j.quant.as_deref().map(str::to_ascii_uppercase)
                                == quant.as_deref().map(str::to_ascii_uppercase)
                            && matches!(j.status, Status::Cancelled | Status::Failed))
                });
            }
        }
        pulls.update(id, |job| match outcome {
            Ok(()) => {
                job.status = Status::Done;
                job.file = None;
                job.bytes_per_second = 0.0;
            }
            Err(message) => {
                job.status = Status::Failed;
                job.error = Some(message);
                job.bytes_per_second = 0.0;
            }
        });
    });
    state.pulls.update(id, |job| job.task = Some(task.abort_handle()));
    let job = state.pulls.jobs.lock().unwrap().get(&id).cloned().expect("just inserted");
    Ok(Json(job))
}

async fn run(
    pulls: &Pulls,
    paths: &ozgent_core::Paths,
    id: u64,
    request: &ozgent_hub::PullRequest,
    alias: Option<&str>,
) -> Result<(), String> {
    use ozgent_hub::Event;
    let client = ozgent_hub::Client::new().map_err(|e| e.to_string())?;
    let on_event = |event: Event<'_>| match event {
        Event::Resolved { selection, model, .. } => pulls.update(id, |job| {
            job.status = Status::Downloading;
            job.model = Some(model.to_string());
            job.quant = Some(selection.quant.clone());
            job.total = selection.total_bytes;
            job.file_count = selection.weights.len() + usize::from(selection.mmproj.is_some());
            job.vision = selection.mmproj.is_some();
        }),
        Event::FileStart { name, index, .. } => pulls.update(id, |job| {
            job.file = Some(name.to_string());
            job.file_index = index;
        }),
        Event::FileProgress { done, .. } => pulls.update(id, |job| {
            job.done = job.finished_files + done;
            job.bytes_per_second = job.rate.record(job.done);
        }),
        Event::FileDone { name, .. } => pulls.update(id, |job| {
            // The size of the finished file, taken from what was counted, so
            // the running total never jumps backwards at a file boundary.
            job.finished_files = job.done.max(job.finished_files);
            let _ = name;
        }),
    };
    let installed = ozgent_hub::pull(&client, paths, request, &on_event)
        .await
        .map_err(|e| e.to_string())?;
    if let Some(a) = alias {
        ozgent_core::set_alias(paths, &installed.model, Some(a)).map_err(|e| e.to_string())?;
    }
    pulls.update(id, |job| {
        job.done = job.total;
        job.model = Some(installed.model.to_string());
    });
    Ok(())
}

/// `GET /api/hub/pulls`: every download, newest first.
pub async fn pulls(AxumState(state): AxumState<State>) -> Json<Vec<Job>> {
    let mut jobs = state.pulls.list();
    jobs.reverse();
    Json(jobs)
}

/// `DELETE /api/hub/pulls/{id}`: cancel a running download, or clear a
/// finished one from the list.
///
/// Cancelling keeps the partial files: pulling the same model again resumes
/// from them.
pub async fn cancel(
    AxumState(state): AxumState<State>,
    Path(id): Path<u64>,
) -> Result<StatusCode, HubFailure> {
    let mut jobs = state.pulls.jobs.lock().unwrap();
    let Some(job) = jobs.get_mut(&id) else {
        return Err(HubFailure(StatusCode::NOT_FOUND, format!("no download {id}")));
    };
    match job.status {
        Status::Resolving | Status::Downloading => {
            if let Some(task) = job.task.take() {
                task.abort();
            }
            job.status = Status::Cancelled;
            job.bytes_per_second = 0.0;
        }
        _ => {
            jobs.remove(&id);
        }
    }
    Ok(StatusCode::NO_CONTENT)
}

// ----------------------------------------------------------------- removal

/// `DELETE /api/models/{model}`: delete an installed model from disk.
///
/// The model is unloaded first if it is the one in memory, so the inference
/// thread is never left mapping files that are gone. A model being downloaded
/// cannot be deleted: that would race the writer.
pub async fn remove(
    AxumState(state): AxumState<State>,
    Path(model): Path<String>,
) -> Result<Json<serde_json::Value>, HubFailure> {
    let found = ozgent_core::resolve(&state.paths, &model)
        .map_err(|e| HubFailure(StatusCode::NOT_FOUND, e.to_string()))?;
    let reference = found.model.to_string();
    let busy = state.pulls.list().into_iter().any(|j| {
        j.model.as_deref() == Some(reference.as_str())
            && matches!(j.status, Status::Resolving | Status::Downloading)
    });
    if busy {
        return Err(HubFailure(
            StatusCode::CONFLICT,
            format!("{reference} is still downloading; cancel it first"),
        ));
    }

    state.worker.unload();
    std::fs::remove_dir_all(&found.dir).map_err(|e| {
        HubFailure(StatusCode::INTERNAL_SERVER_ERROR, format!("removing {}: {e}", found.dir.display()))
    })?;
    // No empty `name/` directory left behind, so `models/` shows what is
    // actually installed.
    if let Some(parent) = found.dir.parent() {
        if parent != state.paths.models_dir()
            && std::fs::read_dir(parent).map(|mut d| d.next().is_none()).unwrap_or(false)
        {
            let _ = std::fs::remove_dir(parent);
        }
    }

    // A default that no longer exists would make every new chat fail to
    // start. Cleared, and said, rather than left to be discovered.
    let mut cleared_default = false;
    {
        let mut config = state.config.lock().unwrap();
        let was = config.default_model.as_deref().is_some_and(|d| {
            d == reference || Some(d) == found.manifest.alias.as_deref()
        });
        if was {
            config.default_model = None;
            if let Err(e) = config.save(&state.paths) {
                tracing::warn!("clearing the default model: {e}");
            }
            cleared_default = true;
        }
    }
    Ok(Json(serde_json::json!({ "removed": reference, "cleared_default": cleared_default })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rate_is_measured_over_the_recent_window() {
        let mut r = Rate::default();
        assert_eq!(r.record(0), 0.0, "one sample is not a rate");
        std::thread::sleep(std::time::Duration::from_millis(250));
        let rate = r.record(1_000_000);
        assert!(rate > 1_000_000.0 && rate < 5_000_000.0, "{rate}");
    }

    #[test]
    fn a_second_download_of_the_same_model_is_seen_as_a_duplicate() {
        let pulls = Pulls::default();
        let job = Job {
            id: 1,
            repo: "unsloth/Qwen3.5-4B-GGUF".into(),
            quant: Some("Q4_K_M".into()),
            model: None,
            alias: None,
            status: Status::Downloading,
            file: None,
            file_index: 0,
            file_count: 0,
            done: 0,
            total: 0,
            bytes_per_second: 0.0,
            error: None,
            vision: false,
            task: None,
            finished_files: 0,
            rate: Rate::default(),
        };
        pulls.jobs.lock().unwrap().insert(1, job.clone());
        assert_eq!(pulls.active_for("unsloth/qwen3.5-4b-gguf", Some("q4_k_m")), Some(1));
        assert_eq!(pulls.active_for("unsloth/Qwen3.5-4B-GGUF", Some("Q8_0")), None);
        let mut done = job;
        done.status = Status::Done;
        pulls.jobs.lock().unwrap().insert(1, done);
        assert_eq!(pulls.active_for("unsloth/Qwen3.5-4B-GGUF", Some("Q4_K_M")), None);
    }
}
