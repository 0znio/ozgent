//! Scheduled jobs, and what happened the last time each one ran.
//!
//! A job is a question ozgent asks itself on a timer — "a pre-market brief" at
//! 9:20 on weekdays — together with where the answer goes. It is stored rather
//! than held in a process because the thing that creates it and the thing that
//! runs it are usually not the same program: a job written from the terminal
//! has to be picked up by whichever `ozgent web` is running, and survive both
//! being restarted.
//!
//! Two decisions here are worth stating, because they are what stops a
//! scheduler becoming a nuisance:
//!
//! * **Every run is recorded, including the boring ones.** A job that quietly
//!   stopped working is the failure mode that matters — a brief that has not
//!   arrived for a week looks exactly like a week with no news. [`JobRun`]
//!   keeps the outcome of each fire so the page can say "last ran at 9:20,
//!   nothing to report" rather than nothing at all.
//! * **History is bounded.** A job every fifteen minutes writes 35,000 rows a
//!   year. Only the most recent [`RUN_HISTORY`] runs of each job are kept.

use crate::store::{Store, StoreError};
use rusqlite::{OptionalExtension, params};

/// How many runs of each job to keep.
///
/// Enough to see a pattern — a fortnight of a daily job, half a day of a
/// fifteen-minute one — without letting a frequent job own the database.
pub const RUN_HISTORY: i64 = 50;

/// The longest a job's prompt may be.
///
/// A prompt is written by whoever may use the scheduler, which on an open
/// channel is not necessarily the owner of the machine. This is generous for
/// anything anyone would schedule and small enough that a thousand jobs is
/// still a small file.
pub const MAX_PROMPT: usize = 8_000;

/// A scheduled job.
#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    pub id: i64,
    /// Stable public identifier, safe to put in a URL.
    pub uuid: String,
    /// The short name a person and a model both use to refer to it:
    /// `pre-market-brief`. Unique, and the thing a chat message names.
    pub name: String,
    pub enabled: bool,
    /// What to ask.
    pub prompt: String,
    /// The agent to ask, without the `@`. `None` asks the default model.
    pub agent: Option<String>,
    /// The model to use, overriding the default for this job only.
    pub model: Option<String>,
    /// The recurrence, in [`ozgent_core::Recur`]'s canonical form.
    pub recur: String,
    /// The zone its wall-clock times are read in: an IANA name, or `local`.
    pub zone: String,
    /// A condition that decides whether the answer is worth sending.
    ///
    /// `None` sends every time. Set, the run produces an answer and a verdict,
    /// and only a met condition is delivered — which is the difference between
    /// a brief you read and a notification you learn to ignore.
    pub only_if: Option<String>,
    /// A JSON array naming the tools this job may use. `None` inherits
    /// whatever the surface running it allows.
    pub tools: Option<String>,
    /// Where the answer goes: `telegram`, `whatsapp`, or `none`.
    pub deliver: String,
    /// The chat it goes to, as that channel spells a chat id.
    pub deliver_to: Option<String>,
    /// The conversation its runs are appended to, so a brief has history and
    /// can be read back in the browser. Created on first run if unset.
    pub conversation_id: Option<i64>,
    /// Where this job came from: `cli`, `web`, or `chat:<channel>:<id>`.
    /// Kept so a job that appears from nowhere can be traced to whoever asked.
    pub created_by: String,
    pub created_at: i64,
    pub updated_at: i64,
    /// When it next fires. `None` means it has no future — a one-off that has
    /// run, or a rule that cannot match again.
    pub next_run_at: Option<i64>,
    pub last_run_at: Option<i64>,
    pub runs: i64,
    /// Consecutive failures. Reset by a run that works; used to back a job off
    /// rather than let it fail on a timer forever.
    pub failures: i64,
}

/// A job as it is created. Everything the caller does not decide has a default.
#[derive(Debug, Clone, PartialEq)]
pub struct NewJob {
    pub name: String,
    pub prompt: String,
    pub recur: String,
    pub zone: String,
    pub agent: Option<String>,
    pub model: Option<String>,
    pub only_if: Option<String>,
    pub tools: Option<String>,
    pub deliver: String,
    pub deliver_to: Option<String>,
    pub conversation_id: Option<i64>,
    pub created_by: String,
    pub next_run_at: Option<i64>,
}

impl NewJob {
    pub fn new(name: impl Into<String>, prompt: impl Into<String>, recur: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            prompt: prompt.into(),
            recur: recur.into(),
            zone: "local".into(),
            agent: None,
            model: None,
            only_if: None,
            tools: None,
            deliver: "none".into(),
            deliver_to: None,
            conversation_id: None,
            created_by: "cli".into(),
            next_run_at: None,
        }
    }
}

/// What a job may be changed to. `None` leaves a field as it was.
///
/// Every field is optional rather than taking a whole [`Job`] because the two
/// callers that edit jobs edit different parts: a chat message changes the
/// time, the page changes anything. Sending a whole row back would let a stale
/// page silently revert a change made from a chat thirty seconds earlier.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobEdit {
    pub name: Option<String>,
    pub prompt: Option<String>,
    pub recur: Option<String>,
    pub zone: Option<String>,
    pub enabled: Option<bool>,
    /// `Some(None)` clears the field; `None` leaves it alone.
    pub agent: Option<Option<String>>,
    pub model: Option<Option<String>>,
    pub only_if: Option<Option<String>>,
    pub tools: Option<Option<String>>,
    pub deliver: Option<String>,
    pub deliver_to: Option<Option<String>>,
    pub next_run_at: Option<Option<i64>>,
}

impl JobEdit {
    /// Whether this would change anything at all.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// How one fire of a job turned out.
#[derive(Debug, Clone, PartialEq)]
pub struct JobRun {
    pub id: i64,
    pub job_id: i64,
    pub started_at: i64,
    pub finished_at: Option<i64>,
    pub status: Status,
    /// What the model answered. Kept even when nothing was delivered, so a
    /// quiet watch can still be inspected.
    pub output: Option<String>,
    pub error: Option<String>,
    pub delivered: bool,
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Started and not yet finished. A row left in this state is a run that
    /// was interrupted — the process stopped mid-answer — and is reported as
    /// such rather than silently counted as a success.
    Running,
    /// Ran, and the answer was delivered.
    Ok,
    /// Ran, and the condition said there was nothing worth sending.
    Quiet,
    /// Did not produce an answer.
    Error,
    /// Was due while nothing was running, and the moment had passed.
    Missed,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Ok => "ok",
            Self::Quiet => "quiet",
            Self::Error => "error",
            Self::Missed => "missed",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "ok" => Self::Ok,
            "quiet" => Self::Quiet,
            "error" => Self::Error,
            "missed" => Self::Missed,
            _ => Self::Running,
        }
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Check a job name, and say why not.
///
/// Names are typed into chats and onto a command line, so they are kept to
/// what survives both: lowercase words joined by hyphens. Rejecting a bad one
/// here means the model gets told to try again rather than creating
/// `Pre Market Brief!!` that nobody can then refer to.
pub fn check_name(name: &str) -> Result<String, String> {
    let name = name.trim().to_lowercase().replace([' ', '_'], "-");
    let name = name.trim_matches('-').to_string();
    if name.is_empty() {
        return Err("a job needs a name".into());
    }
    if name.len() > 48 {
        return Err(format!("{name:?} is too long for a name; keep it under 48 characters"));
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(format!(
            "{name:?} has characters a name cannot hold; use letters, numbers and hyphens"
        ));
    }
    if name.contains("--") {
        return Err(format!("{name:?} has a double hyphen in it"));
    }
    Ok(name)
}

impl Store {
    // ------------------------------------------------------------- jobs

    /// Create a job. Fails if the name is taken.
    pub fn create_job(&self, job: &NewJob) -> Result<i64, StoreError> {
        let now = now();
        let uuid = new_uuid();
        self.raw().execute(
            "INSERT INTO jobs (uuid, name, enabled, prompt, agent, model, recur, zone,
                               only_if, tools, deliver, deliver_to, conversation_id,
                               created_by, created_at, updated_at, next_run_at)
             VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?14, ?15)",
            params![
                uuid,
                job.name,
                job.prompt,
                job.agent,
                job.model,
                job.recur,
                job.zone,
                job.only_if,
                job.tools,
                job.deliver,
                job.deliver_to,
                job.conversation_id,
                job.created_by,
                now,
                job.next_run_at,
            ],
        )?;
        Ok(self.raw().last_insert_rowid())
    }

    pub fn get_job(&self, id: i64) -> Result<Option<Job>, StoreError> {
        self.raw()
            .query_row(&format!("{SELECT} WHERE id = ?1"), params![id], read_job)
            .optional()
            .map_err(Into::into)
    }

    /// Find a job by the name a person or a model used.
    ///
    /// Matches the uuid too, so a page and a chat can both say "job X" without
    /// the caller having to know which kind of identifier it is holding.
    pub fn job_by_name(&self, name: &str) -> Result<Option<Job>, StoreError> {
        let name = name.trim().trim_start_matches('@');
        self.raw()
            .query_row(
                &format!("{SELECT} WHERE name = ?1 COLLATE NOCASE OR uuid = ?1"),
                params![name],
                read_job,
            )
            .optional()
            .map_err(Into::into)
    }

    /// Every job, soonest first, with the disabled ones last.
    pub fn list_jobs(&self) -> Result<Vec<Job>, StoreError> {
        let mut stmt = self.raw().prepare(&format!(
            "{SELECT} ORDER BY enabled DESC,
                                next_run_at IS NULL, next_run_at ASC, name ASC"
        ))?;
        let rows = stmt.query_map([], read_job)?;
        rows.collect::<Result<_, _>>().map_err(Into::into)
    }

    /// Jobs that are enabled and due at or before `now`, soonest first.
    pub fn due_jobs(&self, now: i64) -> Result<Vec<Job>, StoreError> {
        let mut stmt = self.raw().prepare(&format!(
            "{SELECT} WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= ?1
             ORDER BY next_run_at ASC"
        ))?;
        let rows = stmt.query_map(params![now], read_job)?;
        rows.collect::<Result<_, _>>().map_err(Into::into)
    }

    /// Apply an edit. Returns the job as it now is, or `None` if it is gone.
    ///
    /// Built one clause at a time so a field nobody asked about is not written
    /// — two surfaces edit these rows concurrently, and a blanket UPDATE would
    /// let whichever saved last undo the other.
    pub fn update_job(&self, id: i64, edit: &JobEdit) -> Result<Option<Job>, StoreError> {
        let mut sets: Vec<&str> = Vec::new();
        let mut values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        macro_rules! set {
            ($field:ident, $column:literal) => {
                if let Some(v) = &edit.$field {
                    sets.push(concat!($column, " = ?"));
                    values.push(Box::new(v.clone()));
                }
            };
        }
        set!(name, "name");
        set!(prompt, "prompt");
        set!(recur, "recur");
        set!(zone, "zone");
        set!(enabled, "enabled");
        set!(agent, "agent");
        set!(model, "model");
        set!(only_if, "only_if");
        set!(tools, "tools");
        set!(deliver, "deliver");
        set!(deliver_to, "deliver_to");
        set!(next_run_at, "next_run_at");

        if !sets.is_empty() {
            sets.push("updated_at = ?");
            values.push(Box::new(now()));
            // Positional placeholders are numbered by the order they appear,
            // so building the list and the values together keeps them aligned.
            let clause = sets
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}{}", s, i + 1))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!("UPDATE jobs SET {clause} WHERE id = ?{}", values.len() + 1);
            values.push(Box::new(id));
            let refs: Vec<&dyn rusqlite::ToSql> = values.iter().map(|v| v.as_ref()).collect();
            self.raw().execute(&sql, refs.as_slice())?;
        }
        self.get_job(id)
    }

    pub fn delete_job(&self, id: i64) -> Result<bool, StoreError> {
        Ok(self.raw().execute("DELETE FROM jobs WHERE id = ?1", params![id])? > 0)
    }

    /// Record that a job fired: when it last ran, when it runs next, and
    /// whether its failure streak continues.
    pub fn job_fired(
        &self,
        id: i64,
        ran_at: i64,
        next: Option<i64>,
        failed: bool,
    ) -> Result<(), StoreError> {
        self.raw().execute(
            "UPDATE jobs
                SET last_run_at = ?2,
                    next_run_at = ?3,
                    runs        = runs + 1,
                    failures    = CASE WHEN ?4 THEN failures + 1 ELSE 0 END,
                    updated_at  = ?5
              WHERE id = ?1",
            params![id, ran_at, next, failed, now()],
        )?;
        Ok(())
    }

    /// Point a job at the conversation its runs are written to.
    ///
    /// Separate from [`JobEdit`] on purpose: nothing a person edits should be
    /// able to move a job between threads, so this is not a field any form or
    /// tool can reach. Only the runner calls it, once, on a job's first run.
    pub fn set_job_thread(&self, id: i64, conversation_id: i64) -> Result<(), StoreError> {
        self.raw().execute(
            "UPDATE jobs SET conversation_id = ?2 WHERE id = ?1",
            params![id, conversation_id],
        )?;
        Ok(())
    }

    /// Set only the next fire, without counting a run.
    ///
    /// Used when the recurrence is re-evaluated — the rule changed, the
    /// process started, a fire was missed — none of which is a run.
    pub fn set_next_run(&self, id: i64, next: Option<i64>) -> Result<(), StoreError> {
        self.raw().execute(
            "UPDATE jobs SET next_run_at = ?2 WHERE id = ?1",
            params![id, next],
        )?;
        Ok(())
    }

    // -------------------------------------------------------------- runs

    /// Open a run. The id comes back so the outcome can be filled in later.
    pub fn start_run(&self, job_id: i64, at: i64) -> Result<i64, StoreError> {
        self.raw().execute(
            "INSERT INTO job_runs (job_id, started_at, status) VALUES (?1, ?2, 'running')",
            params![job_id, at],
        )?;
        let id = self.raw().last_insert_rowid();
        self.prune_runs(job_id)?;
        Ok(id)
    }

    /// Close a run with how it went.
    pub fn finish_run(
        &self,
        run_id: i64,
        status: Status,
        output: Option<&str>,
        error: Option<&str>,
        delivered: bool,
    ) -> Result<(), StoreError> {
        self.raw().execute(
            "UPDATE job_runs
                SET finished_at = ?2, status = ?3, output = ?4, error = ?5, delivered = ?6
              WHERE id = ?1",
            params![run_id, now(), status.as_str(), output, error, delivered],
        )?;
        Ok(())
    }

    /// Record a run that never started: due while nothing was running.
    pub fn record_missed(&self, job_id: i64, due_at: i64) -> Result<(), StoreError> {
        self.raw().execute(
            "INSERT INTO job_runs (job_id, started_at, finished_at, status)
             VALUES (?1, ?2, ?2, 'missed')",
            params![job_id, due_at],
        )?;
        self.prune_runs(job_id)
    }

    /// A job's runs, most recent first.
    pub fn job_runs(&self, job_id: i64, limit: i64) -> Result<Vec<JobRun>, StoreError> {
        let mut stmt = self.raw().prepare(
            "SELECT id, job_id, started_at, finished_at, status, output, error, delivered
               FROM job_runs WHERE job_id = ?1 ORDER BY started_at DESC, id DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![job_id, limit], |r| {
            Ok(JobRun {
                id: r.get(0)?,
                job_id: r.get(1)?,
                started_at: r.get(2)?,
                finished_at: r.get(3)?,
                status: Status::parse(&r.get::<_, String>(4)?),
                output: r.get(5)?,
                error: r.get(6)?,
                delivered: r.get::<_, i64>(7)? != 0,
            })
        })?;
        rows.collect::<Result<_, _>>().map_err(Into::into)
    }

    /// Close out runs left open by a process that stopped mid-answer.
    ///
    /// Called at startup. Without it a killed server leaves rows that claim to
    /// be running forever, and the page shows a job as busy that is not.
    pub fn abandon_open_runs(&self) -> Result<usize, StoreError> {
        Ok(self.raw().execute(
            "UPDATE job_runs
                SET status = 'error', finished_at = ?1,
                    error = 'ozgent stopped while this was running'
              WHERE status = 'running'",
            params![now()],
        )?)
    }

    fn prune_runs(&self, job_id: i64) -> Result<(), StoreError> {
        self.raw().execute(
            "DELETE FROM job_runs
              WHERE job_id = ?1 AND id NOT IN (
                    SELECT id FROM job_runs WHERE job_id = ?1
                     ORDER BY started_at DESC, id DESC LIMIT ?2)",
            params![job_id, RUN_HISTORY],
        )?;
        Ok(())
    }
}

const SELECT: &str = "SELECT id, uuid, name, enabled, prompt, agent, model, recur, zone,
                             only_if, tools, deliver, deliver_to, conversation_id,
                             created_by, created_at, updated_at, next_run_at,
                             last_run_at, runs, failures
                        FROM jobs";

fn read_job(r: &rusqlite::Row) -> rusqlite::Result<Job> {
    Ok(Job {
        id: r.get(0)?,
        uuid: r.get(1)?,
        name: r.get(2)?,
        enabled: r.get::<_, i64>(3)? != 0,
        prompt: r.get(4)?,
        agent: r.get(5)?,
        model: r.get(6)?,
        recur: r.get(7)?,
        zone: r.get(8)?,
        only_if: r.get(9)?,
        tools: r.get(10)?,
        deliver: r.get(11)?,
        deliver_to: r.get(12)?,
        conversation_id: r.get(13)?,
        created_by: r.get(14)?,
        created_at: r.get(15)?,
        updated_at: r.get(16)?,
        next_run_at: r.get(17)?,
        last_run_at: r.get(18)?,
        runs: r.get(19)?,
        failures: r.get(20)?,
    })
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn new_uuid() -> String {
    crate::store::new_uuid()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Store {
        Store::open_in_memory().unwrap()
    }

    fn job(name: &str) -> NewJob {
        let mut j = NewJob::new(name, "a pre-market brief", "cron 20 9 * * 1,2,3,4,5");
        j.next_run_at = Some(1_000);
        j
    }

    #[test]
    fn a_job_survives_a_round_trip_with_every_field_intact() {
        let s = store();
        let mut new = job("pre-market-brief");
        new.agent = Some("stock-guru".into());
        new.model = Some("coder".into());
        new.zone = "Asia/Kolkata".into();
        new.only_if = Some("only if something moved more than 2%".into());
        new.tools = Some(r#"["web_search"]"#.into());
        new.deliver = "telegram".into();
        new.deliver_to = Some("4242".into());
        new.created_by = "chat:telegram:4242".into();

        let id = s.create_job(&new).unwrap();
        let got = s.get_job(id).unwrap().unwrap();
        assert_eq!(got.name, "pre-market-brief");
        assert_eq!(got.agent.as_deref(), Some("stock-guru"));
        assert_eq!(got.model.as_deref(), Some("coder"));
        assert_eq!(got.zone, "Asia/Kolkata");
        assert_eq!(got.only_if.as_deref(), Some("only if something moved more than 2%"));
        assert_eq!(got.tools.as_deref(), Some(r#"["web_search"]"#));
        assert_eq!(got.deliver, "telegram");
        assert_eq!(got.deliver_to.as_deref(), Some("4242"));
        assert_eq!(got.created_by, "chat:telegram:4242");
        assert_eq!(got.next_run_at, Some(1_000));
        assert!(got.enabled, "a new job is on");
        assert_eq!((got.runs, got.failures), (0, 0));
        assert!(!got.uuid.is_empty());
    }

    #[test]
    fn two_jobs_cannot_share_a_name() {
        // The name is what a chat message refers to; two of them and "change
        // the brief to 8am" has no answer.
        let s = store();
        s.create_job(&job("brief")).unwrap();
        assert!(s.create_job(&job("brief")).is_err());
    }

    #[test]
    fn a_job_is_found_by_name_or_uuid_and_case_does_not_matter() {
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        let uuid = s.get_job(id).unwrap().unwrap().uuid;
        assert_eq!(s.job_by_name("brief").unwrap().unwrap().id, id);
        assert_eq!(s.job_by_name("BRIEF").unwrap().unwrap().id, id);
        assert_eq!(s.job_by_name("@brief").unwrap().unwrap().id, id, "an @ is tolerated");
        assert_eq!(s.job_by_name(&uuid).unwrap().unwrap().id, id);
        assert!(s.job_by_name("nothing").unwrap().is_none());
    }

    #[test]
    fn an_edit_changes_only_what_it_names() {
        // Two surfaces edit these rows. A blanket update would let a stale
        // page revert a change made from a chat a moment earlier.
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        let before = s.get_job(id).unwrap().unwrap();

        let edit = JobEdit { recur: Some("cron 0 8 * * *".into()), ..Default::default() };
        let after = s.update_job(id, &edit).unwrap().unwrap();
        assert_eq!(after.recur, "cron 0 8 * * *");
        assert_eq!(after.prompt, before.prompt, "untouched");
        assert_eq!(after.deliver, before.deliver, "untouched");
        assert!(after.updated_at >= before.updated_at);
    }

    #[test]
    fn an_optional_field_can_be_cleared_as_well_as_set() {
        // "stop using the agent" has to be expressible, and is not the same
        // request as "leave the agent alone".
        let s = store();
        let mut new = job("brief");
        new.agent = Some("stock-guru".into());
        let id = s.create_job(&new).unwrap();

        let cleared = s
            .update_job(id, &JobEdit { agent: Some(None), ..Default::default() })
            .unwrap()
            .unwrap();
        assert_eq!(cleared.agent, None);

        let set = s
            .update_job(id, &JobEdit { agent: Some(Some("other".into())), ..Default::default() })
            .unwrap()
            .unwrap();
        assert_eq!(set.agent.as_deref(), Some("other"));
    }

    #[test]
    fn an_empty_edit_leaves_the_job_exactly_as_it_was() {
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        let before = s.get_job(id).unwrap().unwrap();
        assert!(JobEdit::default().is_empty());
        let after = s.update_job(id, &JobEdit::default()).unwrap().unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn editing_a_job_that_is_gone_reports_it_rather_than_failing() {
        let s = store();
        assert!(s.update_job(999, &JobEdit::default()).unwrap().is_none());
        assert!(!s.delete_job(999).unwrap());
    }

    #[test]
    fn only_enabled_jobs_that_are_actually_due_come_back() {
        let s = store();
        let mut soon = job("soon");
        soon.next_run_at = Some(100);
        let soon = s.create_job(&soon).unwrap();

        let mut later = job("later");
        later.next_run_at = Some(9_000);
        s.create_job(&later).unwrap();

        let mut off = job("off");
        off.next_run_at = Some(100);
        let off = s.create_job(&off).unwrap();
        s.update_job(off, &JobEdit { enabled: Some(false), ..Default::default() }).unwrap();

        let mut never = job("never");
        never.next_run_at = None;
        s.create_job(&never).unwrap();

        let due = s.due_jobs(1_000).unwrap();
        assert_eq!(due.len(), 1, "{:?}", due.iter().map(|j| &j.name).collect::<Vec<_>>());
        assert_eq!(due[0].id, soon);
    }

    #[test]
    fn due_jobs_come_back_soonest_first() {
        let s = store();
        for (name, at) in [("third", 300), ("first", 100), ("second", 200)] {
            let mut j = job(name);
            j.next_run_at = Some(at);
            s.create_job(&j).unwrap();
        }
        let names: Vec<String> = s.due_jobs(1_000).unwrap().into_iter().map(|j| j.name).collect();
        assert_eq!(names, ["first", "second", "third"]);
    }

    #[test]
    fn firing_advances_the_schedule_and_counts_the_run() {
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        s.job_fired(id, 1_000, Some(2_000), false).unwrap();
        let j = s.get_job(id).unwrap().unwrap();
        assert_eq!((j.last_run_at, j.next_run_at, j.runs, j.failures), (Some(1_000), Some(2_000), 1, 0));
    }

    #[test]
    fn a_failure_streak_builds_up_and_is_cleared_by_one_good_run() {
        // The streak is what backs a broken job off; if it never reset, one
        // bad morning would slow a job down forever.
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        for _ in 0..3 {
            s.job_fired(id, 1_000, Some(2_000), true).unwrap();
        }
        assert_eq!(s.get_job(id).unwrap().unwrap().failures, 3);
        s.job_fired(id, 3_000, Some(4_000), false).unwrap();
        let j = s.get_job(id).unwrap().unwrap();
        assert_eq!(j.failures, 0);
        assert_eq!(j.runs, 4, "a failed run is still a run");
    }

    #[test]
    fn a_one_off_with_no_next_fire_stops_being_due() {
        let s = store();
        let id = s.create_job(&job("once")).unwrap();
        s.job_fired(id, 1_000, None, false).unwrap();
        assert!(s.due_jobs(9_999_999).unwrap().is_empty());
        assert_eq!(s.get_job(id).unwrap().unwrap().next_run_at, None);
    }

    #[test]
    fn a_run_records_what_happened() {
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        let run = s.start_run(id, 1_000).unwrap();
        let open = &s.job_runs(id, 10).unwrap()[0];
        assert_eq!(open.status, Status::Running);
        assert!(open.finished_at.is_none());

        s.finish_run(run, Status::Ok, Some("the answer"), None, true).unwrap();
        let done = &s.job_runs(id, 10).unwrap()[0];
        assert_eq!(done.status, Status::Ok);
        assert_eq!(done.output.as_deref(), Some("the answer"));
        assert!(done.delivered);
        assert!(done.finished_at.is_some());
    }

    #[test]
    fn a_quiet_watch_keeps_its_answer_even_though_nothing_was_sent() {
        // Otherwise there is no way to tell a watch that decided not to speak
        // from one that never ran.
        let s = store();
        let id = s.create_job(&job("watch")).unwrap();
        let run = s.start_run(id, 1_000).unwrap();
        s.finish_run(run, Status::Quiet, Some("nothing moved"), None, false).unwrap();
        let got = &s.job_runs(id, 10).unwrap()[0];
        assert_eq!(got.status, Status::Quiet);
        assert_eq!(got.output.as_deref(), Some("nothing moved"));
        assert!(!got.delivered);
    }

    #[test]
    fn run_history_is_bounded() {
        // A job every fifteen minutes writes 35,000 rows a year.
        let s = store();
        let id = s.create_job(&job("chatty")).unwrap();
        for i in 0..RUN_HISTORY + 20 {
            let run = s.start_run(id, 1_000 + i).unwrap();
            s.finish_run(run, Status::Ok, None, None, true).unwrap();
        }
        let kept = s.job_runs(id, 1_000).unwrap();
        assert_eq!(kept.len() as i64, RUN_HISTORY);
        assert_eq!(kept[0].started_at, 1_000 + RUN_HISTORY + 19, "the newest is kept");
    }

    #[test]
    fn runs_come_back_newest_first() {
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        for at in [1_000, 2_000, 3_000] {
            s.start_run(id, at).unwrap();
        }
        let ats: Vec<i64> = s.job_runs(id, 10).unwrap().into_iter().map(|r| r.started_at).collect();
        assert_eq!(ats, [3_000, 2_000, 1_000]);
    }

    #[test]
    fn deleting_a_job_takes_its_history_with_it() {
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        s.start_run(id, 1_000).unwrap();
        assert!(s.delete_job(id).unwrap());
        assert!(s.job_runs(id, 10).unwrap().is_empty());
        assert!(s.get_job(id).unwrap().is_none());
    }

    #[test]
    fn a_run_left_open_by_a_crash_is_closed_at_startup() {
        // Otherwise the page shows a job as busy that stopped hours ago.
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        s.start_run(id, 1_000).unwrap();
        assert_eq!(s.abandon_open_runs().unwrap(), 1);
        let got = &s.job_runs(id, 10).unwrap()[0];
        assert_eq!(got.status, Status::Error);
        assert!(got.error.as_deref().unwrap().contains("stopped"));
        // And it is idempotent: a second start finds nothing left open.
        assert_eq!(s.abandon_open_runs().unwrap(), 0);
    }

    #[test]
    fn a_missed_fire_is_recorded_as_missed_rather_than_forgotten() {
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        s.record_missed(id, 1_000).unwrap();
        let got = &s.job_runs(id, 10).unwrap()[0];
        assert_eq!(got.status, Status::Missed);
        assert!(!got.delivered);
    }

    #[test]
    fn jobs_are_listed_with_the_soonest_first_and_the_disabled_last() {
        let s = store();
        for (name, at) in [("later", Some(9_000)), ("soon", Some(100)), ("never", None)] {
            let mut j = job(name);
            j.next_run_at = at;
            s.create_job(&j).unwrap();
        }
        let mut off = job("off");
        off.next_run_at = Some(1);
        let off = s.create_job(&off).unwrap();
        s.update_job(off, &JobEdit { enabled: Some(false), ..Default::default() }).unwrap();

        let names: Vec<String> = s.list_jobs().unwrap().into_iter().map(|j| j.name).collect();
        assert_eq!(names, ["soon", "later", "never", "off"]);
    }

    #[test]
    fn setting_the_next_fire_does_not_count_as_a_run() {
        let s = store();
        let id = s.create_job(&job("brief")).unwrap();
        s.set_next_run(id, Some(5_000)).unwrap();
        let j = s.get_job(id).unwrap().unwrap();
        assert_eq!(j.next_run_at, Some(5_000));
        assert_eq!(j.runs, 0);
        assert_eq!(j.last_run_at, None);
    }

    // -------------------------------------------------------------- names

    #[test]
    fn a_name_is_tidied_into_something_typeable() {
        assert_eq!(check_name("Pre Market Brief").unwrap(), "pre-market-brief");
        assert_eq!(check_name("  brief  ").unwrap(), "brief");
        assert_eq!(check_name("my_job").unwrap(), "my-job");
        assert_eq!(check_name("-brief-").unwrap(), "brief");
    }

    #[test]
    fn a_name_that_could_not_be_referred_to_is_refused() {
        for bad in ["", "   ", "---", "brief!", "a/b", "brief\nname", &"x".repeat(49)] {
            assert!(check_name(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn a_job_pointing_at_a_deleted_conversation_does_not_break_the_row() {
        // A person deletes a conversation in the browser; the job that was
        // writing into it has to keep running, not fail its foreign key.
        let s = store();
        let c = s.create_conversation("brief", None).unwrap();
        let mut new = job("brief");
        new.conversation_id = Some(c);
        let id = s.create_job(&new).unwrap();

        s.delete_conversation(c).unwrap();
        let got = s.get_job(id).unwrap().expect("the job survives");
        assert_eq!(got.conversation_id, None, "and starts a fresh thread next run");
    }
}
