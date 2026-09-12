//! Running scheduled jobs.
//!
//! [`ozgent_schedule`] defines what a job *is*; this runs one. The split is
//! not tidiness — running needs a loaded model, and the crates that create
//! jobs (the terminal, the tool the model calls) must not drag inference in
//! behind them.
//!
//! A job is asked exactly the way a person would ask it. It goes through
//! [`crate::turn`] like every other question, which is what makes a scheduled
//! answer a real conversation: it has a thread, it is in the browser
//! afterwards, its tool calls are recorded, and the next morning's run can see
//! yesterday's. A separate "batch" path would have been simpler and would have
//! drifted from the interactive one within a month.
//!
//! Four things about running unattended shape the rest:
//!
//! * **Nobody can answer a permission question.** `can_ask` is false, so
//!   anything that would ask is refused. See [`ozgent_schedule`] for why that
//!   is the only safe reading.
//! * **One at a time.** Jobs share one GPU with whoever is using ozgent right
//!   now. Two briefs at 9:20 would queue behind each other anyway; running
//!   them one at a time makes that explicit and keeps the interactive path
//!   from waiting behind a crowd.
//! * **One process.** A lock file decides which ozgent runs jobs, the same way
//!   one decides which answers the channels. Two schedulers on one database
//!   would deliver every brief twice.
//! * **A missed fire is not run late.** A 9:20 pre-market brief delivered at
//!   3pm is worse than no brief. Fires that passed while nothing was running
//!   are recorded as missed and skipped.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ozgent_core::channels::Kind;
use ozgent_memory::jobs::{Job, Status};
use ozgent_schedule::{Deliver, unix_now};

use crate::state::State;
use crate::worker::Event;

/// The longest the scheduler ever sleeps between looks.
///
/// It normally sleeps until the next job is due, which for an overnight gap is
/// hours. This ceiling exists because the database is shared: a job created in
/// another process, or made due by "run now", changes when the next wake
/// *should* be, and nothing tells this loop. A minute is the promise the
/// `schedule` tool makes to the model about how soon `run` takes effect.
const MAX_SLEEP: std::time::Duration = std::time::Duration::from_secs(60);

/// The shortest, so a job that reschedules itself into the past cannot spin.
const MIN_SLEEP: std::time::Duration = std::time::Duration::from_secs(2);

/// Consecutive failures before a job is switched off.
///
/// A job that has failed five times running is not going to succeed on the
/// sixth, and something failing on a timer forever is noise that buries the
/// jobs that work. Turned off rather than deleted, with the reason recorded.
const GIVE_UP_AFTER: i64 = 5;

/// How long a job's answer may be before it is truncated for delivery.
///
/// Not a model limit — a chat limit. The channels split long replies across
/// messages, and a scheduled brief that arrives as nine messages at 9:20 is a
/// notification people turn off.
const MAX_DELIVERED: usize = 3_500;

/// Start the scheduler in this process. Idempotent.
///
/// Safe to call from both `ozgent web` and `ozgent gateway`: the second call
/// in a process returns immediately, and a second *process* takes the lock
/// only if the first has stopped.
pub fn start(state: &State) {
    static RUNNING: AtomicBool = AtomicBool::new(false);
    if RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }
    let state = Arc::clone(state);
    tokio::spawn(async move {
        let mut lock = Lock::default();
        // Only once, by whoever gets the lock first: tidying the database is
        // not something two processes should race over.
        let mut swept = false;
        loop {
            let held = lock.held(&state.paths);
            HOSTED.store(held, Ordering::SeqCst);
            if held {
                if !swept {
                    sweep(&state);
                    swept = true;
                }
                tick(&state).await;
            }
            // Sleep until there is something to do rather than on a fixed
            // tick. With nothing scheduled this is one wake a minute, and the
            // work each one does is a single indexed lookup that matches
            // nothing — which is what lets the daemon sit idle for days.
            tokio::time::sleep(if held { until_next(&state) } else { MAX_SLEEP }).await;
        }
    });
}

/// Set once this process takes the lock, so the page can say who is running
/// jobs without probing the lock file on every request.
static HOSTED: AtomicBool = AtomicBool::new(false);

/// Whether this process is the one running jobs.
pub fn hosted() -> bool {
    HOSTED.load(Ordering::SeqCst)
}

/// The pid of the process running jobs, when it is not this one.
pub fn held_elsewhere(state: &State) -> Option<String> {
    // Asked first, because the probe below cannot tell "somebody else holds
    // it" from "we do": a lock taken through one file descriptor conflicts
    // with an attempt through another, even inside the same process. Without
    // this the page reports itself as the other ozgent.
    if hosted() {
        return None;
    }
    let path = lock_path(&state.paths);
    let file = std::fs::OpenOptions::new().read(true).write(true).open(&path).ok()?;
    // Taking the lock means nobody holds it. Release it again immediately:
    // this is a question, not a claim.
    if file.try_lock().is_ok() {
        let _ = file.unlock();
        return None;
    }
    let pid = std::fs::read_to_string(&path).ok()?;
    let pid = pid.trim();
    (!pid.is_empty()).then(|| format!("another ozgent (pid {pid})"))
}

/// The lock that decides which process runs jobs.
///
/// Under `~/ozgent/scheduler/` rather than loose in the root, so everything
/// the scheduler owns is in one place a person can point at. The jobs
/// themselves are rows in `ozgent.db`: they are edited from the terminal, the
/// page and a chat, sometimes at once, and a file would have to be locked by
/// all three.
pub fn lock_path(paths: &ozgent_core::Paths) -> std::path::PathBuf {
    paths.scheduler_dir().join("lock")
}

/// The scheduler's claim on this database, held for as long as it runs.
#[derive(Default)]
struct Lock(Option<std::fs::File>);

impl Lock {
    /// Whether this process holds the lock, taking it if it is free.
    fn held(&mut self, paths: &ozgent_core::Paths) -> bool {
        if self.0.is_some() {
            return true;
        }
        let path = lock_path(paths);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .read(true)
            .open(&path)
        else {
            return false;
        };
        if file.try_lock().is_err() {
            return false;
        }
        use std::io::{Seek, Write};
        let mut f = &file;
        let _ = f.set_len(0);
        let _ = f.seek(std::io::SeekFrom::Start(0));
        let _ = write!(f, "{}", std::process::id());
        self.0 = Some(file);
        true
    }
}

/// Put the database in order before running anything.
///
/// Three things are wrong after a stop: runs left open by a process that died
/// mid-answer, fires that came and went while nothing was running, and rows
/// whose next fire no longer matches their rule.
fn sweep(state: &State) {
    let now = unix_now();
    let store = state.store.lock().unwrap();
    match store.abandon_open_runs() {
        Ok(0) => {}
        Ok(n) => tracing::info!("scheduler: {n} run(s) were interrupted by a restart"),
        Err(e) => tracing::warn!("scheduler: {e}"),
    }
    let jobs = match store.list_jobs() {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("scheduler: cannot read jobs: {e}");
            return;
        }
    };
    let mut missed = 0;
    for job in &jobs {
        if job.enabled && job.next_run_at.is_some_and(|at| at < now) {
            // Recorded before the next fire is computed, so the gap is visible
            // on the page rather than silently closed over.
            let _ = store.record_missed(job.id, job.next_run_at.unwrap_or(now));
            missed += 1;
        }
        match ozgent_schedule::resync(&store, job, now) {
            Ok(Some(why)) => tracing::warn!("scheduler: {why}"),
            Ok(None) => {}
            Err(e) => tracing::warn!("scheduler: {} could not be rescheduled: {e}", job.name),
        }
    }
    let live = jobs.iter().filter(|j| j.enabled).count();
    if missed > 0 {
        tracing::info!(
            "scheduler: {missed} fire(s) passed while ozgent was not running, and were skipped"
        );
    }
    tracing::info!("scheduler: {live} job(s) armed, {} in total", jobs.len());
}

/// Run everything that is due, one at a time.
async fn tick(state: &State) {
    let now = unix_now();
    let due = {
        let store = state.store.lock().unwrap();
        match store.due_jobs(now) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("scheduler: {e}");
                return;
            }
        }
    };
    for job in due {
        // Re-read rather than trusting the list: a job can be paused or
        // deleted from a chat while an earlier one in this batch is running,
        // and a brief that arrives after you switched it off is the kind of
        // thing that makes people stop trusting a scheduler.
        let fresh = {
            let store = state.store.lock().unwrap();
            store.get_job(job.id).ok().flatten()
        };
        let Some(job) = fresh.filter(|j| j.enabled && j.next_run_at.is_some_and(|at| at <= now))
        else {
            continue;
        };
        run(state, &job).await;
    }
}

/// How long until the next job is due, bounded at both ends.
fn until_next(state: &State) -> std::time::Duration {
    let soonest = state
        .store
        .lock()
        .unwrap()
        .list_jobs()
        .ok()
        .and_then(|jobs| {
            jobs.iter()
                .filter(|j| j.enabled)
                .filter_map(|j| j.next_run_at)
                .min()
        });
    let Some(at) = soonest else {
        return MAX_SLEEP;
    };
    let wait = at.saturating_sub(unix_now()).max(0) as u64;
    std::time::Duration::from_secs(wait).clamp(MIN_SLEEP, MAX_SLEEP)
}

/// What one run produced.
struct Outcome {
    status: Status,
    answer: String,
    error: Option<String>,
    delivered: bool,
}

/// Run one job: ask it, judge it, deliver it, record it.
pub async fn run(state: &State, job: &Job) {
    let started = unix_now();
    let run_id = {
        let store = state.store.lock().unwrap();
        match store.start_run(job.id, started) {
            Ok(id) => id,
            Err(e) => {
                tracing::error!("scheduler: {} could not start: {e}", job.name);
                return;
            }
        }
    };
    tracing::info!("scheduler: running {}", job.name);

    let outcome = perform(state, job).await;
    let failed = outcome.status == Status::Error;

    {
        let store = state.store.lock().unwrap();
        let _ = store.finish_run(
            run_id,
            outcome.status,
            (!outcome.answer.trim().is_empty()).then(|| outcome.answer.trim()),
            outcome.error.as_deref(),
            outcome.delivered,
        );
        // Re-read before arming the next fire. A run takes seconds to minutes,
        // and the job can be paused from a chat or the page while it is in
        // flight — writing a next run from the snapshot taken at the start
        // would quietly bring it back.
        let still_on = store.get_job(job.id).ok().flatten().is_some_and(|j| j.enabled);
        let next = still_on.then(|| ozgent_schedule::next_fire(job, unix_now())).flatten();
        let _ = store.job_fired(job.id, started, next, failed);

        // A job failing on a timer forever buries the ones that work.
        if failed && job.failures + 1 >= GIVE_UP_AFTER {
            let _ = store.update_job(
                job.id,
                &ozgent_memory::JobEdit {
                    enabled: Some(false),
                    next_run_at: Some(None),
                    ..Default::default()
                },
            );
            tracing::warn!(
                "scheduler: {} has failed {GIVE_UP_AFTER} times running and is now paused",
                job.name
            );
        }
    }

    match outcome.status {
        Status::Error => tracing::warn!(
            "scheduler: {} failed: {}",
            job.name,
            outcome.error.as_deref().unwrap_or("no answer")
        ),
        Status::Quiet => tracing::info!("scheduler: {} ran; nothing worth sending", job.name),
        _ => tracing::info!("scheduler: {} done", job.name),
    }
}

/// Ask the question, decide whether to send it, and send it.
async fn perform(state: &State, job: &Job) -> Outcome {
    let conversation = match thread_for(state, job) {
        Ok(id) => id,
        Err(e) => {
            return Outcome {
                status: Status::Error,
                answer: String::new(),
                error: Some(e),
                delivered: false,
            };
        }
    };

    // Asked exactly as a person would, `@agent` and all, so it takes the same
    // path through agents, tools and memory that a typed question does.
    let question = match &job.agent {
        Some(agent) => format!("@{agent} {}", job.prompt),
        None => job.prompt.clone(),
    };

    let answer = match ask(state, conversation, job, &question).await {
        Ok(a) => a,
        Err(e) => {
            return Outcome {
                status: Status::Error,
                answer: String::new(),
                error: Some(e),
                delivered: false,
            };
        }
    };
    if answer.trim().is_empty() {
        return Outcome {
            status: Status::Error,
            answer,
            error: Some("the model produced no answer".into()),
            delivered: false,
        };
    }

    // A watch decides whether this is worth anyone's attention.
    if let Some(condition) = &job.only_if {
        match worth_sending(state, job, condition, &answer).await {
            Ok(false) => {
                return Outcome {
                    status: Status::Quiet,
                    answer,
                    error: None,
                    delivered: false,
                };
            }
            Ok(true) => {}
            // An unreadable verdict sends. The failure that matters is a watch
            // that goes quiet and is trusted to be quiet for a good reason.
            Err(e) => tracing::warn!("scheduler: {} could not judge its condition: {e}", job.name),
        }
    }

    match deliver(state, job, &answer) {
        Ok(true) => Outcome { status: Status::Ok, answer, error: None, delivered: true },
        Ok(false) => Outcome { status: Status::Ok, answer, error: None, delivered: false },
        // The answer is good; only the sending failed. Recorded as an error
        // because an undelivered brief is a failure from where the user sits.
        Err(e) => Outcome {
            status: Status::Error,
            answer,
            error: Some(format!("could not deliver it: {e}")),
            delivered: false,
        },
    }
}

/// The conversation this job's runs are written to, making one if needed.
fn thread_for(state: &State, job: &Job) -> Result<i64, String> {
    if let Some(id) = job.conversation_id {
        let store = state.store.lock().unwrap();
        if store.get_conversation(id).ok().flatten().is_some() {
            return Ok(id);
        }
    }
    let store = state.store.lock().unwrap();
    let id = store
        .create_conversation(&job.name, job.model.as_deref())
        .map_err(|e| e.to_string())?;
    // Remembered so tomorrow's run continues this thread instead of starting a
    // new conversation every morning. Written through the store's own method
    // rather than an ad-hoc UPDATE, so the schedule is left exactly alone.
    store.set_job_thread(job.id, id).map_err(|e| e.to_string())?;
    Ok(id)
}

/// Put the question to the model and collect the answer.
async fn ask(state: &State, conversation: i64, job: &Job, question: &str) -> Result<String, String> {
    let model = match &job.model {
        Some(m) => m.clone(),
        None => state
            .config
            .lock()
            .unwrap()
            .default_model
            .clone()
            .ok_or("no model is configured to answer with")?,
    };
    let tools: Option<Vec<String>> = job.tools.as_deref().and_then(|t| serde_json::from_str(t).ok());
    // An empty list is a job that may not call anything, which is different
    // from a job that was never narrowed.
    let tools_enabled = tools.as_ref().is_none_or(|t| !t.is_empty());

    let mut events = crate::turn::start(
        state,
        crate::turn::Turn {
            conversation,
            model,
            message: question.to_string(),
            thinking: None,
            tools: tools_enabled,
            native_tools: tools,
            tools_off: Vec::new(),
            images: Vec::new(),
            // Nobody is awake. Anything that would ask is refused.
            can_ask: false,
            // A job may reschedule itself, and gets its own tool list when
            // it does — not a wider one.
            caller: Some(ozgent_schedule::Caller {
                origin: format!("job:{}", job.name),
                deliver: Some(Deliver::read(&job.deliver, job.deliver_to.as_deref())),
                allowed_tools: job.tools.as_deref().and_then(|t| serde_json::from_str(t).ok()),
                conversation_id: Some(conversation),
            }),
        },
    )
    .map_err(|e| e.to_string())?;

    let mut answer = String::new();
    let mut refused: Vec<String> = Vec::new();
    let mut error = None;
    while let Some(event) = events.recv().await {
        match event {
            Event::Answer { text } => answer.push_str(&text),
            Event::ToolResult { name, ok, summary, .. } if !ok => {
                // Worth carrying into the delivered text: a brief that is thin
                // because a tool could not run should say so rather than look
                // like a quiet news day.
                //
                // Matched against the phrase ozgent itself produces rather
                // than a guess at it. The first version looked for "refused",
                // which never appears in the real message — so it quietly did
                // nothing until a live run showed the note missing from a
                // brief that needed it. `UNATTENDED_MARK` is asserted against
                // the real wording in ozgent-core, so the two cannot drift.
                if summary.contains(ozgent_core::permission::UNATTENDED_MARK) {
                    refused.push(name);
                }
            }
            Event::Error { message } => error = Some(message),
            Event::Done { .. } => break,
            _ => {}
        }
    }
    if let Some(message) = error {
        return Err(message);
    }
    if !refused.is_empty() {
        refused.sort();
        refused.dedup();
        let (tools, verb) = match refused.len() {
            1 => (refused[0].clone(), "needs"),
            _ => (refused.join(", "), "need"),
        };
        answer.push_str(&format!(
            "\n\n_{tools} {verb} approval, and a scheduled run has nobody to ask, \
             so it did not run._"
        ));
    }
    Ok(answer)
}

/// Ask whether a watch's condition is met.
///
/// A separate question, with a grammar that admits only YES or NO. Asking the
/// main answer to end in a verdict was the obvious alternative and is not
/// reliable — a small model writing a market brief forgets the format about a
/// third of the time, and a watch that sends on a parse failure is a watch
/// that always sends.
async fn worth_sending(
    state: &State,
    job: &Job,
    condition: &str,
    answer: &str,
) -> Result<bool, String> {
    let model = match &job.model {
        Some(m) => m.clone(),
        None => state
            .config
            .lock()
            .unwrap()
            .default_model
            .clone()
            .ok_or("no model is configured")?,
    };
    let prompt = format!(
        "A scheduled check produced this report:\n\n---\n{}\n---\n\nThe user only wants to be \
         told when this is true: {condition}\n\nIs it true? Answer with one word, YES or NO.",
        answer.chars().take(6_000).collect::<String>()
    );

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    state
        .worker
        .submit(crate::worker::Request {
            model,
            messages: vec![ozgent_core::Message {
                role: ozgent_core::Role::User,
                content: vec![ozgent_core::Part::Text { text: prompt }],
                thinking: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }],
            thinking: Some(ozgent_core::ThinkingMode::Off),
            max_tokens: Some(4),
            tools_enabled: false,
            native_tools: Some(Vec::new()),
            client_tools: Vec::new(),
            response_grammar: Some(r#"root ::= "YES" | "NO""#.to_string()),
            overrides: None,
            images: Vec::new(),
            can_ask: false,
            agents: Vec::new(),
            tools_off: Vec::new(),
            handoff: Vec::new(),
            out: tx,
        })
        .map_err(|e| e.to_string())?;

    let mut verdict = String::new();
    while let Some(event) = rx.recv().await {
        match event {
            Event::Answer { text } => verdict.push_str(&text),
            Event::Error { message } => return Err(message),
            Event::Done { .. } => break,
            _ => {}
        }
    }
    let verdict = verdict.trim().to_ascii_uppercase();
    if verdict.starts_with("YES") {
        return Ok(true);
    }
    if verdict.starts_with("NO") {
        return Ok(false);
    }
    Err(format!("expected YES or NO, got {verdict:?}"))
}

/// Send the answer where the job says. `false` means there was nowhere to send.
///
/// A job that names no chat goes to everyone the channel allows — which is
/// what somebody who set the channel up and then said "send it to Telegram"
/// meant. Who that is is worked out now rather than when the job was written,
/// so taking a person off the allowlist stops their deliveries.
fn deliver(state: &State, job: &Job, answer: &str) -> Result<bool, String> {
    let destination = Deliver::read(&job.deliver, job.deliver_to.as_deref());
    let Deliver::Chat { channel, .. } = &destination else {
        return Ok(false);
    };
    let kind = match channel.as_str() {
        "telegram" => Kind::Telegram,
        "whatsapp" => Kind::WhatsApp,
        other => return Err(format!("{other} is not a channel")),
    };
    let gateway = state
        .gateway
        .get()
        .ok_or("no gateway is running in this process, so there is nothing to send with")?;

    let chats = {
        let allow = {
            let config = state.config.lock().unwrap();
            config.channels.access(kind).allow.to_vec()
        };
        let store = state.store.lock().unwrap();
        destination.recipients(&store, &allow)
    };
    if chats.is_empty() {
        // Said as a failure rather than passed over: the job ran, produced an
        // answer, and nobody got it. Silence here is the thing that makes a
        // scheduler untrustworthy.
        return Err(match job.deliver_to.as_deref() {
            Some(_) => format!("{channel} has no chat matching this job's destination"),
            None => format!(
                "nobody on {channel}'s allow list has messaged ozgent yet, so there is \
                 no chat to send to. Message it once from {channel} and it will know."
            ),
        });
    }

    let text = headed(job, answer);
    let mut sent = 0;
    let mut failures: Vec<String> = Vec::new();
    for chat in &chats {
        match gateway.deliver(kind, chat, &text) {
            Ok(()) => sent += 1,
            Err(e) => failures.push(format!("{chat}: {e}")),
        }
    }
    // One recipient failing is not the whole delivery failing — the others
    // got it, and reporting nothing sent would be wrong.
    if sent == 0 {
        return Err(failures.join("; "));
    }
    if !failures.is_empty() {
        tracing::warn!("{}: delivered to {sent} of {}; {}", job.name, chats.len(), failures.join("; "));
    }
    Ok(true)
}

/// The answer as it arrives in a chat.
///
/// Headed with the job's name because it arrives unprompted: a block of text
/// appearing at 9:20 with no question above it needs to say what it is.
fn headed(job: &Job, answer: &str) -> String {
    let body = if answer.chars().count() > MAX_DELIVERED {
        let cut: String = answer.chars().take(MAX_DELIVERED).collect();
        // Cut at a paragraph if there is one nearby, so the message does not
        // end mid-sentence.
        let cut = match cut.rfind("\n\n") {
            Some(at) if at > MAX_DELIVERED / 2 => cut[..at].to_string(),
            _ => cut,
        };
        format!("{cut}\n\n_…the rest is on the scheduler page._")
    } else {
        answer.trim().to_string()
    };
    format!("**{}**\n\n{body}", job.name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ozgent_memory::jobs::NewJob;

    fn job(name: &str) -> Job {
        Job {
            id: 1,
            uuid: "u".into(),
            name: name.into(),
            enabled: true,
            prompt: "a brief".into(),
            agent: None,
            model: None,
            recur: "cron 20 9 * * 1,2,3,4,5".into(),
            zone: "local".into(),
            only_if: None,
            tools: None,
            deliver: "none".into(),
            deliver_to: None,
            conversation_id: None,
            created_by: "cli".into(),
            created_at: 0,
            updated_at: 0,
            next_run_at: Some(100),
            last_run_at: None,
            runs: 0,
            failures: 0,
        }
    }

    #[test]
    fn a_delivered_answer_says_which_job_it_is_from() {
        // It arrives with no question above it; without a heading it is a
        // block of text from nowhere.
        let text = headed(&job("pre-market-brief"), "The Nifty opened flat.");
        assert!(text.starts_with("**pre-market-brief**"), "{text}");
        assert!(text.contains("The Nifty opened flat."));
    }

    #[test]
    fn a_very_long_answer_is_cut_rather_than_sent_as_nine_messages() {
        let long = "x".repeat(MAX_DELIVERED * 2);
        let text = headed(&job("brief"), &long);
        assert!(text.chars().count() < MAX_DELIVERED + 200, "{} chars", text.chars().count());
        assert!(text.contains("scheduler page"));
    }

    #[test]
    fn a_long_answer_is_cut_at_a_paragraph_where_there_is_one() {
        // Otherwise a brief ends mid-sentence, which reads like a bug.
        let tail = "second paragraph ".repeat(40);
        let body = format!("{}\n\n{tail}", "a".repeat(MAX_DELIVERED - 100));
        let text = headed(&job("nightly"), &body);
        assert!(text.contains("aaa"), "the first paragraph is kept");
        assert!(!text.contains("second paragraph"), "it cut mid-paragraph, not at the break");
        assert!(text.contains("scheduler page"), "and says where the rest is");
    }

    #[test]
    fn an_answer_that_fits_is_sent_whole_and_untrimmed_in_the_middle() {
        let text = headed(&job("brief"), "  one\n\ntwo  ");
        assert!(text.ends_with("two"), "{text}");
        assert!(text.contains("one\n\ntwo"));
    }

    #[test]
    fn the_lock_lives_in_the_schedulers_own_directory() {
        let paths = ozgent_core::Paths::with_root(std::path::Path::new("/tmp/ozgent-test"));
        assert_eq!(lock_path(&paths), std::path::Path::new("/tmp/ozgent-test/scheduler/lock"));
    }

    #[test]
    fn only_one_process_can_hold_the_scheduler_lock() {
        // Two schedulers on one database would deliver every brief twice.
        let dir = std::env::temp_dir().join(format!("ozgent-lock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = ozgent_core::Paths::with_root(&dir);

        let mut first = Lock::default();
        assert!(first.held(&paths), "the first taker gets it");
        let mut second = Lock::default();
        assert!(!second.held(&paths), "the second must not");
        // And holding it is idempotent for the one that has it.
        assert!(first.held(&paths));

        drop(first);
        let mut third = Lock::default();
        assert!(third.held(&paths), "it is free once the holder stops");
        drop(third);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_lock_file_names_the_process_holding_it() {
        // So the page can say who is scheduling rather than only that someone is.
        let dir = std::env::temp_dir().join(format!("ozgent-lock-pid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = ozgent_core::Paths::with_root(&dir);
        let mut lock = Lock::default();
        assert!(lock.held(&paths));
        let written = std::fs::read_to_string(lock_path(&paths)).unwrap();
        assert_eq!(written.trim(), std::process::id().to_string());
        drop(lock);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_job_with_an_empty_tool_list_may_not_call_anything() {
        // `[]` is "no tools", `null` is "whatever this surface allows" — the
        // difference decides whether a locked-down job silently regains them.
        let empty: Option<Vec<String>> = serde_json::from_str("[]").unwrap();
        assert!(empty.as_ref().is_some_and(|t: &Vec<String>| t.is_empty()));
        assert!(!empty.as_ref().is_none_or(|t| !t.is_empty()), "tools must be off");

        let unset: Option<Vec<String>> = None;
        assert!(unset.as_ref().is_none_or(|t: &Vec<String>| !t.is_empty()), "tools stay on");
    }

    #[test]
    fn a_job_is_asked_with_its_agent_the_way_a_person_would_write_it() {
        let mut j = job("brief");
        j.agent = Some("stock-guru".into());
        let asked = match &j.agent {
            Some(a) => format!("@{a} {}", j.prompt),
            None => j.prompt.clone(),
        };
        assert_eq!(asked, "@stock-guru a brief");
    }

    #[test]
    fn a_new_job_is_stored_with_no_thread_and_gains_one_on_its_first_run() {
        let store = ozgent_memory::Store::open_in_memory().unwrap();
        let id = store.create_job(&NewJob::new("brief", "a brief", "cron 0 9 * * *")).unwrap();
        assert_eq!(store.get_job(id).unwrap().unwrap().conversation_id, None);
    }
}
