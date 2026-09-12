//! Scheduled jobs: creating them, changing them, and describing them.
//!
//! Four surfaces can schedule something — the terminal, the browser, a chat
//! app, and the model itself mid-conversation — and they must all mean the
//! same thing by it. So the rules live here once: what a valid job is, what
//! happens to a rule that cannot be parsed, when the next fire is, and how a
//! job reads back in words. Each surface is then only a way of collecting the
//! fields.
//!
//! What is deliberately *not* here is running a job, which needs a model. That
//! lives with whichever process has one loaded. This crate is the definition;
//! the runner is the execution.
//!
//! # The rule that shapes everything else
//!
//! A scheduled job runs with nobody watching. That has one consequence worth
//! stating at the top, because every other decision follows from it: **a job
//! can never approve a tool call.** There is no one to ask at 9:20 in the
//! morning, and a scheduler that answered "yes" on the user's behalf would be
//! a way to turn "read me the news" into `run_command` while they sleep. Tools
//! that run outright still run; anything that would ask is refused, and the
//! refusal is part of what gets delivered so it is visible rather than silent.

pub mod tools;

use ozgent_core::schedule::Recur;
use ozgent_core::zone::Zone;
use ozgent_memory::jobs::{Job, JobEdit, NewJob, check_name};
use ozgent_memory::{Store, StoreError};

pub use tools::{Caller, ScheduleTools, TOOL_NAME};

/// Where a job's answer is sent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Deliver {
    /// Nowhere. The run is recorded and readable on the scheduler page, and
    /// that is all — which is the right default for a job created from the
    /// browser, where there may be no chat to send to.
    Nowhere,
    /// To a chat on a channel.
    ///
    /// `to` names one chat. `None` means *everyone the channel allows*, which
    /// is what someone who set a channel up and then said "send it to
    /// Telegram" meant — they already named who may talk to it, and naming
    /// them again in a chat id they have never seen is not a thing anyone can
    /// do from memory. Who that is is resolved when the message is sent, not
    /// when the job is written, so removing somebody from the allowlist stops
    /// their deliveries.
    Chat { channel: String, to: Option<String> },
}

impl Deliver {
    pub fn channel(&self) -> &str {
        match self {
            Self::Nowhere => "none",
            Self::Chat { channel, .. } => channel,
        }
    }

    pub fn to(&self) -> Option<&str> {
        match self {
            Self::Nowhere => None,
            Self::Chat { to, .. } => to.as_deref(),
        }
    }

    /// Read a channel and chat back out of a stored job.
    pub fn read(channel: &str, to: Option<&str>) -> Self {
        match channel {
            "telegram" | "whatsapp" => Self::Chat {
                channel: channel.to_string(),
                to: to.map(str::trim).filter(|t| !t.is_empty()).map(str::to_string),
            },
            _ => Self::Nowhere,
        }
    }

    /// Check a requested destination before it is saved.
    ///
    /// A channel with no chat named is not an error: it means everyone that
    /// channel allows. Requiring an id here was the bug — a person who set
    /// Telegram up has already said who may use it, and a Telegram chat id is
    /// not something anyone knows by heart.
    pub fn parse(channel: &str, to: Option<&str>) -> Result<Self, String> {
        let channel = channel.trim().to_lowercase();
        match channel.as_str() {
            "" | "none" | "nowhere" => Ok(Self::Nowhere),
            "telegram" | "whatsapp" => Ok(Self::Chat {
                channel,
                to: to.map(str::trim).filter(|t| !t.is_empty()).map(str::to_string),
            }),
            other => Err(format!(
                "{other:?} is not somewhere to send to — use telegram, whatsapp, or none"
            )),
        }
    }

    /// Who a message actually goes to.
    ///
    /// One chat when the job names one. Otherwise every chat this channel has
    /// spoken to that its allowlist *still* admits — checked now rather than
    /// when the job was written, so taking somebody off the list stops their
    /// deliveries without touching the job.
    pub fn recipients(&self, store: &Store, allow: &[String]) -> Vec<String> {
        let Self::Chat { channel, to } = self else { return Vec::new() };
        if let Some(one) = to {
            return vec![one.clone()];
        }
        store
            .channel_chats(channel)
            .unwrap_or_default()
            .into_iter()
            .filter(|chat| {
                // A chat bound before ozgent kept identities has none to check.
                // Its chat id is the one identity always available, and for
                // Telegram that *is* the numeric user id the list can name.
                let mut ids: Vec<&str> = chat.identities.iter().map(String::as_str).collect();
                ids.push(&chat.chat_id);
                ozgent_core::channels::admits(allow, &ids)
            })
            .map(|chat| chat.chat_id)
            .collect()
    }
}

/// Everything needed to create a job, before it is checked.
#[derive(Debug, Clone, Default)]
pub struct Draft {
    pub name: String,
    pub prompt: String,
    /// The recurrence as it was typed, in any form [`Recur::parse`] accepts.
    pub when: String,
    pub zone: Option<String>,
    pub agent: Option<String>,
    pub model: Option<String>,
    pub only_if: Option<String>,
    pub tools: Option<Vec<String>>,
    pub deliver: Option<Deliver>,
    pub conversation_id: Option<i64>,
    pub created_by: String,
}

/// What went wrong, in words meant for whoever typed it — a person or a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Problem {
    /// The request itself does not make sense. Worth retrying differently.
    Invalid(String),
    /// No job by that name.
    NoSuchJob(String),
    /// The name is taken.
    NameTaken(String),
    /// The database said no.
    Store(String),
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(m) => f.write_str(m),
            Self::NoSuchJob(name) => {
                write!(f, "there is no scheduled job called {name:?}")
            }
            Self::NameTaken(name) => write!(
                f,
                "there is already a job called {name:?} — change that one, or pick another name"
            ),
            Self::Store(m) => write!(f, "the scheduler's database refused that: {m}"),
        }
    }
}

impl std::error::Error for Problem {}

impl From<StoreError> for Problem {
    fn from(e: StoreError) -> Self {
        // A unique-index violation is the name clash, and is worth saying
        // properly rather than as a SQL error: it is the one failure here a
        // person can actually do something about.
        let text = e.to_string();
        if text.contains("UNIQUE") && text.contains("jobs") {
            return Self::NameTaken(String::new());
        }
        Self::Store(text)
    }
}

/// The zone a job's times are read in.
///
/// `local` is stored rather than resolved at creation so a machine that moves
/// — or whose zone is corrected — keeps meaning "9:20 where I am", which is
/// what the person wrote.
pub fn zone_of(job: &Job) -> Zone {
    match job.zone.as_str() {
        "local" | "" => Zone::local(),
        name => Zone::named(name).unwrap_or_else(|e| {
            tracing::warn!("job {}: {e}; falling back to local time", job.name);
            Zone::local()
        }),
    }
}

/// The recurrence of a stored job.
///
/// A row whose rule will not parse is a job that can never fire, so it is
/// reported rather than silently skipped — the caller disables it and says so.
pub fn recur_of(job: &Job) -> Result<Recur, String> {
    Recur::parse(&job.recur)
}

/// When a job should next fire, given that it last fired at `after`.
pub fn next_fire(job: &Job, after: i64) -> Option<i64> {
    let recur = recur_of(job).ok()?;
    recur.next_after(after, job.created_at, &zone_of(job))
}

/// Validate a draft and store it.
pub fn create(store: &Store, draft: &Draft) -> Result<Job, Problem> {
    let name = check_name(&draft.name).map_err(Problem::Invalid)?;
    if store.job_by_name(&name)?.is_some() {
        return Err(Problem::NameTaken(name));
    }
    let prompt = draft.prompt.trim();
    if prompt.is_empty() {
        return Err(Problem::Invalid("a job needs something to ask".into()));
    }
    if prompt.len() > ozgent_memory::jobs::MAX_PROMPT {
        return Err(Problem::Invalid(format!(
            "that prompt is {} characters; the limit is {}",
            prompt.len(),
            ozgent_memory::jobs::MAX_PROMPT
        )));
    }
    // "every day at 4pm UTC" names its own zone; "every day at 4pm" means the
    // clock in the room. An explicitly given zone still wins over either.
    let (when, inline) =
        ozgent_core::schedule::split_zone(&draft.when).map_err(Problem::Invalid)?;
    let recur = Recur::parse(&when).map_err(Problem::Invalid)?;
    let zone = check_zone(draft.zone.as_deref().or(inline.as_deref()))?;
    let deliver = draft.deliver.clone().unwrap_or(Deliver::Nowhere);

    let mut new = NewJob::new(&name, prompt, recur.to_string());
    new.zone = zone.clone();
    new.agent = clean(draft.agent.as_deref()).map(|a| a.trim_start_matches('@').to_string());
    new.model = clean(draft.model.as_deref());
    new.only_if = clean(draft.only_if.as_deref());
    new.tools = draft
        .tools
        .as_ref()
        .map(|t| serde_json::to_string(t).unwrap_or_else(|_| "[]".into()));
    new.deliver = deliver.channel().to_string();
    new.deliver_to = deliver.to().map(str::to_string);
    new.conversation_id = draft.conversation_id;
    new.created_by = if draft.created_by.is_empty() {
        "cli".into()
    } else {
        draft.created_by.clone()
    };

    // Computed against the created_at the row is about to get, so an interval
    // job's phase starts now rather than at the epoch.
    let now = unix_now();
    let resolved = named_zone(&zone);
    new.next_run_at = recur.next_after(now, now, &resolved);
    if new.next_run_at.is_none() && recur.repeats() {
        return Err(Problem::Invalid(format!(
            "{:?} describes a time that never comes around",
            draft.when
        )));
    }
    if new.next_run_at.is_none() {
        return Err(Problem::Invalid(
            "that time has already passed — give a time in the future".into(),
        ));
    }
    debug_assert!(
        new.next_run_at.is_some_and(|at| at > now),
        "a job must never be created already overdue"
    );

    let id = store.create_job(&new).map_err(|e| match Problem::from(e) {
        Problem::NameTaken(_) => Problem::NameTaken(name.clone()),
        other => other,
    })?;
    store.get_job(id)?.ok_or_else(|| Problem::Store("the job vanished as it was written".into()))
}

/// Apply a change to an existing job, validating whatever it touches.
///
/// Returns the job as it now is. Any change to the rule, the zone or the
/// enabled flag recomputes the next fire, because all three change it.
pub fn update(store: &Store, name: &str, edit: &Change) -> Result<Job, Problem> {
    let job = store
        .job_by_name(name)?
        .ok_or_else(|| Problem::NoSuchJob(name.to_string()))?;

    let mut apply = JobEdit::default();
    if let Some(new_name) = clean(edit.name.as_deref()) {
        let new_name = check_name(&new_name).map_err(Problem::Invalid)?;
        if !new_name.eq_ignore_ascii_case(&job.name) {
            if store.job_by_name(&new_name)?.is_some() {
                return Err(Problem::NameTaken(new_name));
            }
            apply.name = Some(new_name);
        }
    }
    if let Some(prompt) = clean(edit.prompt.as_deref()) {
        if prompt.len() > ozgent_memory::jobs::MAX_PROMPT {
            return Err(Problem::Invalid("that prompt is too long".into()));
        }
        apply.prompt = Some(prompt);
    }
    let mut recur = recur_of(&job).ok();
    let mut inline_zone = None;
    if let Some(when) = clean(edit.when.as_deref()) {
        let (when, inline) =
            ozgent_core::schedule::split_zone(&when).map_err(Problem::Invalid)?;
        let parsed = Recur::parse(&when).map_err(Problem::Invalid)?;
        apply.recur = Some(parsed.to_string());
        recur = Some(parsed);
        inline_zone = inline;
    }
    let mut zone = job.zone.clone();
    // A zone named in the new time applies to it — retiming a job to
    // "4pm UTC" must not leave it reading 4pm in the old zone.
    if edit.zone.is_some() || inline_zone.is_some() {
        zone = check_zone(edit.zone.as_deref().or(inline_zone.as_deref()))?;
        apply.zone = Some(zone.clone());
    }
    if let Some(enabled) = edit.enabled {
        apply.enabled = Some(enabled);
    }
    if let Some(agent) = &edit.agent {
        apply.agent = Some(
            clean(agent.as_deref()).map(|a| a.trim_start_matches('@').to_string()),
        );
    }
    if let Some(model) = &edit.model {
        apply.model = Some(clean(model.as_deref()));
    }
    if let Some(only_if) = &edit.only_if {
        apply.only_if = Some(clean(only_if.as_deref()));
    }
    if let Some(tools) = &edit.tools {
        apply.tools = Some(
            tools
                .as_ref()
                .map(|t| serde_json::to_string(t).unwrap_or_else(|_| "[]".into())),
        );
    }
    if let Some(deliver) = &edit.deliver {
        apply.deliver = Some(deliver.channel().to_string());
        apply.deliver_to = Some(deliver.to().map(str::to_string));
    }

    // Anything that moves the next fire has to move it now, not at the next
    // tick: a job rescheduled from 9:20 to 8:00 that still says 9:20 until
    // tomorrow has not really been rescheduled.
    let enabled_now = edit.enabled.unwrap_or(job.enabled);
    if apply.recur.is_some() || apply.zone.is_some() || edit.enabled.is_some() {
        let next = match (&recur, enabled_now) {
            (Some(r), true) => {
                let now = unix_now();
                r.next_after(now, job.created_at, &named_zone(&zone))
            }
            _ => None,
        };
        if enabled_now && next.is_none() && recur.is_some() {
            return Err(Problem::Invalid(
                "that leaves the job with no next run — check the time".into(),
            ));
        }
        apply.next_run_at = Some(next);
    }

    if apply.is_empty() {
        return Ok(job);
    }
    store
        .update_job(job.id, &apply)?
        .ok_or_else(|| Problem::NoSuchJob(name.to_string()))
}

/// A requested change. `None` leaves a field alone; `Some(None)` clears it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Change {
    pub name: Option<String>,
    pub prompt: Option<String>,
    pub when: Option<String>,
    pub zone: Option<String>,
    pub enabled: Option<bool>,
    pub agent: Option<Option<String>>,
    pub model: Option<Option<String>>,
    pub only_if: Option<Option<String>>,
    pub tools: Option<Option<Vec<String>>>,
    pub deliver: Option<Deliver>,
}

/// Bring a job's next fire in line with its rule.
///
/// Called when a scheduler starts. A job whose rule no longer parses is turned
/// off rather than left to fail every tick, and the reason comes back so it
/// can be logged once.
pub fn resync(store: &Store, job: &Job, now: i64) -> Result<Option<String>, Problem> {
    if !job.enabled {
        return Ok(None);
    }
    let recur = match recur_of(job) {
        Ok(r) => r,
        Err(why) => {
            store.update_job(
                job.id,
                &JobEdit { enabled: Some(false), next_run_at: Some(None), ..Default::default() },
            )?;
            return Ok(Some(format!("job {} is off: its schedule is unreadable ({why})", job.name)));
        }
    };
    // A one-off whose moment has passed has nothing left to do.
    let next = recur.next_after(now, job.created_at, &zone_of(job));
    if next != job.next_run_at {
        store.set_next_run(job.id, next)?;
    }
    Ok(None)
}

/// A job in one line, as every surface says it.
pub fn summarise(job: &Job) -> String {
    let zone = zone_of(job);
    let when = recur_of(job)
        .map(|r| r.describe(&zone))
        .unwrap_or_else(|_| format!("an unreadable schedule ({})", job.recur));
    let mut line = format!("{} — {when}", job.name);
    if let Some(agent) = &job.agent {
        line.push_str(&format!(", asked of @{agent}"));
    }
    match Deliver::read(&job.deliver, job.deliver_to.as_deref()) {
        Deliver::Chat { channel, to: Some(_) } => {
            line.push_str(&format!(", sent to one {channel} chat"))
        }
        Deliver::Chat { channel, to: None } => {
            line.push_str(&format!(", sent to everyone {channel} allows"))
        }
        Deliver::Nowhere => line.push_str(", kept on the scheduler page"),
    }
    if job.only_if.is_some() {
        line.push_str(", only when its condition is met");
    }
    if !job.enabled {
        line.push_str(" (paused)");
    }
    line
}

/// "in 4 hours", "in 12 minutes", "now" — how a next run is shown.
pub fn in_words(at: i64, now: i64) -> String {
    let d = at - now;
    if d <= 0 {
        return "due now".into();
    }
    // Each boundary sits exactly on its unit, so 3600 seconds reads as an
    // hour rather than as sixty minutes.
    let (n, unit) = match d {
        d if d < 60 => (d, "second"),
        d if d < 3600 => ((d + 30) / 60, "minute"),
        d if d < 86_400 => ((d + 1800) / 3600, "hour"),
        d => ((d + 43_200) / 86_400, "day"),
    };
    if n == 1 { format!("in 1 {unit}") } else { format!("in {n} {unit}s") }
}

fn clean(s: Option<&str>) -> Option<String> {
    s.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

fn check_zone(zone: Option<&str>) -> Result<String, Problem> {
    match clean(zone) {
        None => Ok("local".into()),
        Some(name) if name == "local" => Ok("local".into()),
        Some(name) => {
            Zone::named(&name).map_err(|e| Problem::Invalid(e))?;
            Ok(name)
        }
    }
}

fn named_zone(name: &str) -> Zone {
    match name {
        "local" | "" => Zone::local(),
        other => Zone::named(other).unwrap_or_else(|_| Zone::local()),
    }
}

pub fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> Store {
        Store::open_in_memory().unwrap()
    }

    fn draft(name: &str, when: &str) -> Draft {
        Draft {
            name: name.into(),
            prompt: "a pre-market brief".into(),
            when: when.into(),
            created_by: "cli".into(),
            ..Default::default()
        }
    }

    #[test]
    fn the_request_from_the_feature_works_end_to_end() {
        // "@stock-guru every weekday at 9:20, send me a pre-market brief on
        // Telegram" — the whole thing, as one job.
        let s = store();
        let mut d = draft("pre-market-brief", "every weekday at 9:20");
        d.agent = Some("@stock-guru".into());
        d.deliver = Some(Deliver::Chat { channel: "telegram".into(), to: Some("4242".into()) });

        let job = create(&s, &d).unwrap();
        assert_eq!(job.name, "pre-market-brief");
        assert_eq!(job.agent.as_deref(), Some("stock-guru"), "the @ is not stored");
        assert_eq!(job.recur, "cron 20 9 * * 1,2,3,4,5");
        assert_eq!(job.deliver, "telegram");
        assert_eq!(job.deliver_to.as_deref(), Some("4242"));
        assert!(job.next_run_at.unwrap() > unix_now(), "it has a future");

        let line = summarise(&job);
        assert!(line.contains("every weekday"), "{line}");
        assert!(line.contains("@stock-guru"), "{line}");
        assert!(line.contains("telegram"), "{line}");
    }

    #[test]
    fn a_name_is_tidied_rather_than_rejected_where_it_can_be() {
        let s = store();
        let job = create(&s, &draft("Pre Market Brief", "every day at 9:20")).unwrap();
        assert_eq!(job.name, "pre-market-brief");
    }

    #[test]
    fn the_same_name_twice_is_refused_with_something_actionable() {
        let s = store();
        create(&s, &draft("brief", "every day at 9:20")).unwrap();
        let err = create(&s, &draft("brief", "every day at 8:00")).unwrap_err();
        assert!(matches!(err, Problem::NameTaken(_)));
        assert!(err.to_string().contains("already a job"), "{err}");
    }

    #[test]
    fn a_schedule_that_cannot_be_read_is_refused_at_creation() {
        // Rather than stored and discovered to be dead at 9:20 one morning.
        let s = store();
        for when in ["every blursday at 9:00", "", "at 25:00", "0 0 31 2 *"] {
            let err = create(&s, &draft("brief", when)).unwrap_err();
            assert!(matches!(err, Problem::Invalid(_)), "{when:?} gave {err:?}");
        }
        assert!(s.list_jobs().unwrap().is_empty(), "nothing was stored");
    }

    #[test]
    fn a_job_with_nothing_to_ask_is_refused() {
        let s = store();
        let mut d = draft("brief", "every day at 9:20");
        d.prompt = "   ".into();
        assert!(matches!(create(&s, &d), Err(Problem::Invalid(_))));
    }

    #[test]
    fn an_enormous_prompt_is_refused_rather_than_stored() {
        let s = store();
        let mut d = draft("brief", "every day at 9:20");
        d.prompt = "x".repeat(ozgent_memory::jobs::MAX_PROMPT + 1);
        assert!(matches!(create(&s, &d), Err(Problem::Invalid(_))));
    }

    #[test]
    fn a_one_off_in_the_past_is_refused() {
        let s = store();
        let err = create(&s, &draft("brief", "once 1000")).unwrap_err();
        assert!(err.to_string().contains("already passed"), "{err}");
    }

    #[test]
    fn a_zone_is_checked_before_it_is_stored() {
        let s = store();
        let mut d = draft("brief", "every day at 9:20");
        d.zone = Some("Mars/Olympus".into());
        assert!(matches!(create(&s, &d), Err(Problem::Invalid(_))));

        d.zone = Some("UTC".into());
        assert_eq!(create(&s, &d).unwrap().zone, "UTC");
    }

    #[test]
    fn local_is_the_default_zone_and_is_stored_as_local() {
        // Not resolved to a name: the person meant "where I am", and a laptop
        // that crosses a border should keep meaning that.
        let s = store();
        let job = create(&s, &draft("brief", "every day at 9:20")).unwrap();
        assert_eq!(job.zone, "local");
    }

    // --------------------------------------------------------- changing

    #[test]
    fn changing_the_time_moves_the_next_run_immediately() {
        // A job rescheduled from 9:20 to 8:00 that still says 9:20 until
        // tomorrow has not really been rescheduled.
        let s = store();
        let job = create(&s, &draft("brief", "every day at 9:20")).unwrap();
        let before = job.next_run_at.unwrap();

        let changed = update(
            &s,
            "brief",
            &Change { when: Some("every day at 8:00".into()), ..Default::default() },
        )
        .unwrap();
        assert_eq!(changed.recur, "cron 0 8 * * *");
        assert_ne!(changed.next_run_at.unwrap(), before);
    }

    #[test]
    fn pausing_a_job_takes_away_its_next_run_and_resuming_gives_it_back() {
        let s = store();
        create(&s, &draft("brief", "every day at 9:20")).unwrap();

        let off = update(&s, "brief", &Change { enabled: Some(false), ..Default::default() }).unwrap();
        assert!(!off.enabled);
        assert_eq!(off.next_run_at, None, "a paused job is not due");

        let on = update(&s, "brief", &Change { enabled: Some(true), ..Default::default() }).unwrap();
        assert!(on.enabled);
        assert!(on.next_run_at.unwrap() > unix_now());
    }

    #[test]
    fn changing_one_field_leaves_the_others_alone() {
        let s = store();
        let mut d = draft("brief", "every day at 9:20");
        d.agent = Some("stock-guru".into());
        d.deliver = Some(Deliver::Chat { channel: "telegram".into(), to: Some("4242".into()) });
        let before = create(&s, &d).unwrap();

        let after = update(
            &s,
            "brief",
            &Change { prompt: Some("a different question".into()), ..Default::default() },
        )
        .unwrap();
        assert_eq!(after.prompt, "a different question");
        assert_eq!(after.agent, before.agent);
        assert_eq!(after.deliver_to, before.deliver_to);
        assert_eq!(after.recur, before.recur);
    }

    #[test]
    fn an_agent_can_be_taken_off_a_job_as_well_as_put_on_it() {
        let s = store();
        let mut d = draft("brief", "every day at 9:20");
        d.agent = Some("stock-guru".into());
        create(&s, &d).unwrap();

        let cleared = update(&s, "brief", &Change { agent: Some(None), ..Default::default() }).unwrap();
        assert_eq!(cleared.agent, None);
    }

    #[test]
    fn renaming_onto_a_taken_name_is_refused_but_renaming_to_itself_is_fine() {
        let s = store();
        create(&s, &draft("brief", "every day at 9:20")).unwrap();
        create(&s, &draft("other", "every day at 9:20")).unwrap();

        let err = update(&s, "brief", &Change { name: Some("other".into()), ..Default::default() })
            .unwrap_err();
        assert!(matches!(err, Problem::NameTaken(_)));

        // Re-saving a form without changing the name must not trip over it.
        let same = update(&s, "brief", &Change { name: Some("Brief".into()), ..Default::default() })
            .unwrap();
        assert_eq!(same.name, "brief");
    }

    #[test]
    fn changing_a_job_that_does_not_exist_says_so() {
        let s = store();
        let err = update(&s, "nothing", &Change::default()).unwrap_err();
        assert!(matches!(err, Problem::NoSuchJob(_)));
        assert!(err.to_string().contains("nothing"), "{err}");
    }

    #[test]
    fn a_change_to_an_unreadable_time_is_refused_and_the_job_is_untouched() {
        let s = store();
        let before = create(&s, &draft("brief", "every day at 9:20")).unwrap();
        assert!(update(&s, "brief", &Change { when: Some("never".into()), ..Default::default() }).is_err());
        assert_eq!(s.job_by_name("brief").unwrap().unwrap(), before);
    }

    // ------------------------------------------------------- delivery

    #[test]
    fn a_destination_is_checked_before_it_is_saved() {
        assert_eq!(Deliver::parse("none", None).unwrap(), Deliver::Nowhere);
        assert_eq!(Deliver::parse("", None).unwrap(), Deliver::Nowhere);
        assert_eq!(
            Deliver::parse("telegram", Some("4242")).unwrap(),
            Deliver::Chat { channel: "telegram".into(), to: Some("4242".into()) }
        );
        // A channel with no chat named means everyone that channel allows —
        // see the `delivery` tests. Demanding an id here was the bug: nobody
        // knows their own Telegram chat id.
        assert_eq!(
            Deliver::parse("telegram", None).unwrap(),
            Deliver::Chat { channel: "telegram".into(), to: None }
        );
        assert!(Deliver::parse("email", Some("a@b.c")).is_err());
    }

    #[test]
    fn a_destination_reads_back_as_it_was_written() {
        for d in [
            Deliver::Nowhere,
            Deliver::Chat { channel: "telegram".into(), to: Some("4242".into()) },
            Deliver::Chat { channel: "whatsapp".into(), to: Some("918638680186".into()) },
        ] {
            assert_eq!(Deliver::read(d.channel(), d.to()), d);
        }
    }

    // --------------------------------------------------------- resync

    #[test]
    fn starting_up_brings_a_stale_next_run_in_line() {
        // The process was off overnight; the row still points at yesterday.
        let s = store();
        let job = create(&s, &draft("brief", "every day at 9:20")).unwrap();
        s.set_next_run(job.id, Some(1_000)).unwrap();

        let job = s.get_job(job.id).unwrap().unwrap();
        assert_eq!(resync(&s, &job, unix_now()).unwrap(), None);
        let fixed = s.get_job(job.id).unwrap().unwrap();
        assert!(fixed.next_run_at.unwrap() > unix_now());
    }

    #[test]
    fn a_job_whose_rule_became_unreadable_is_turned_off_and_reported() {
        // Rather than failing on a timer forever with nobody told why.
        let s = store();
        let job = create(&s, &draft("brief", "every day at 9:20")).unwrap();
        s.update_job(
            job.id,
            &JobEdit { recur: Some("gibberish".into()), ..Default::default() },
        )
        .unwrap();

        let job = s.get_job(job.id).unwrap().unwrap();
        let said = resync(&s, &job, unix_now()).unwrap().expect("it must report");
        assert!(said.contains("brief"), "{said}");
        let after = s.get_job(job.id).unwrap().unwrap();
        assert!(!after.enabled);
        assert_eq!(after.next_run_at, None);
    }

    #[test]
    fn resync_leaves_a_paused_job_paused() {
        let s = store();
        let job = create(&s, &draft("brief", "every day at 9:20")).unwrap();
        update(&s, "brief", &Change { enabled: Some(false), ..Default::default() }).unwrap();
        let job = s.get_job(job.id).unwrap().unwrap();

        assert_eq!(resync(&s, &job, unix_now()).unwrap(), None);
        assert_eq!(s.get_job(job.id).unwrap().unwrap().next_run_at, None);
    }

    // ------------------------------------------------------- describing

    #[test]
    fn a_time_until_reads_the_way_a_person_says_it() {
        let now = 1_000_000;
        assert_eq!(in_words(now, now), "due now");
        assert_eq!(in_words(now - 5, now), "due now");
        assert_eq!(in_words(now + 30, now), "in 30 seconds");
        assert_eq!(in_words(now + 60, now), "in 1 minute");
        assert_eq!(in_words(now + 1800, now), "in 30 minutes");
        assert_eq!(in_words(now + 3600, now), "in 1 hour");
        assert_eq!(in_words(now + 4 * 3600, now), "in 4 hours");
        assert_eq!(in_words(now + 3 * 86_400, now), "in 3 days");
    }

    #[test]
    fn a_paused_job_says_so_in_its_summary() {
        let s = store();
        create(&s, &draft("brief", "every day at 9:20")).unwrap();
        let off = update(&s, "brief", &Change { enabled: Some(false), ..Default::default() }).unwrap();
        assert!(summarise(&off).contains("paused"), "{}", summarise(&off));
    }

    #[test]
    fn a_watch_says_that_it_only_speaks_when_its_condition_is_met() {
        let s = store();
        let mut d = draft("watch", "every hour");
        d.only_if = Some("only if the price moved more than 2%".into());
        let job = create(&s, &d).unwrap();
        assert!(summarise(&job).contains("condition"), "{}", summarise(&job));
    }
}

/// The whole life of a job, through the same calls every surface makes.
///
/// Unit tests above cover each step; these check the sequences that only go
/// wrong when steps are combined — which is where a scheduler actually fails.
#[cfg(test)]
mod lifecycle {
    use super::*;
    use ozgent_memory::JobStatus;

    fn store() -> Store {
        Store::open_in_memory().unwrap()
    }

    fn brief() -> Draft {
        Draft {
            name: "pre-market-brief".into(),
            prompt: "a pre-market brief".into(),
            when: "every weekday at 9:20".into(),
            agent: Some("stock-guru".into()),
            deliver: Some(Deliver::Chat { channel: "telegram".into(), to: Some("4242".into()) }),
            created_by: "chat:telegram:4242".into(),
            ..Default::default()
        }
    }

    #[test]
    fn a_job_created_changed_run_and_deleted_leaves_nothing_behind() {
        let s = store();
        let job = create(&s, &brief()).unwrap();

        // It runs twice, one of them quietly.
        for (status, delivered) in [(JobStatus::Ok, true), (JobStatus::Quiet, false)] {
            let run = s.start_run(job.id, unix_now()).unwrap();
            s.finish_run(run, status, Some("the brief"), None, delivered).unwrap();
            s.job_fired(job.id, unix_now(), next_fire(&job, unix_now()), false).unwrap();
        }
        let after = s.get_job(job.id).unwrap().unwrap();
        assert_eq!(after.runs, 2);
        assert_eq!(s.job_runs(job.id, 10).unwrap().len(), 2);

        // Retimed, then paused, then deleted.
        let changed =
            update(&s, "pre-market-brief", &Change { when: Some("every day at 8:00".into()), ..Default::default() })
                .unwrap();
        assert_eq!(changed.recur, "cron 0 8 * * *");
        assert_eq!(changed.runs, 2, "history is not reset by a change");

        update(&s, "pre-market-brief", &Change { enabled: Some(false), ..Default::default() }).unwrap();
        assert!(s.due_jobs(i64::MAX).unwrap().is_empty());

        assert!(s.delete_job(changed.id).unwrap());
        assert!(s.job_by_name("pre-market-brief").unwrap().is_none());
        assert!(s.job_runs(changed.id, 10).unwrap().is_empty(), "history went with it");
    }

    #[test]
    fn a_job_never_becomes_due_twice_for_the_same_fire() {
        // The bug that would deliver every brief twice: firing has to move the
        // schedule strictly forward, every time.
        let s = store();
        let job = create(&s, &brief()).unwrap();
        let mut seen = Vec::new();
        let mut job = job;
        for _ in 0..6 {
            let at = job.next_run_at.expect("a repeating job always has one");
            assert!(!seen.contains(&at), "fired at {at} twice");
            seen.push(at);
            s.job_fired(job.id, at, next_fire(&job, at), false).unwrap();
            job = s.get_job(job.id).unwrap().unwrap();
        }
        assert!(seen.windows(2).all(|w| w[0] < w[1]), "{seen:?}");
    }

    #[test]
    fn pausing_and_resuming_never_leaves_a_job_due_in_the_past() {
        // A resumed job whose next run is still yesterday fires immediately
        // and then keeps firing, which reads as a runaway.
        let s = store();
        create(&s, &brief()).unwrap();
        update(&s, "pre-market-brief", &Change { enabled: Some(false), ..Default::default() }).unwrap();
        let on = update(&s, "pre-market-brief", &Change { enabled: Some(true), ..Default::default() })
            .unwrap();
        assert!(on.next_run_at.unwrap() > unix_now());
    }

    #[test]
    fn a_restart_catches_up_without_running_anything_late() {
        // The process was off overnight. The stale fire is the caller's to
        // record as missed; resync's job is to arm the next real one.
        let s = store();
        let job = create(&s, &brief()).unwrap();
        s.set_next_run(job.id, Some(unix_now() - 6 * 3600)).unwrap();

        let stale = s.get_job(job.id).unwrap().unwrap();
        assert!(stale.next_run_at.unwrap() < unix_now(), "it is overdue");
        assert_eq!(resync(&s, &stale, unix_now()).unwrap(), None);

        let armed = s.get_job(job.id).unwrap().unwrap();
        assert!(armed.next_run_at.unwrap() > unix_now(), "never left in the past");
        assert_eq!(armed.runs, 0, "and nothing was run to catch up");
    }

    #[test]
    fn a_chat_cannot_widen_what_a_job_it_creates_may_do() {
        // The security property, end to end: the tools a job is stored with
        // are the ones the conversation had.
        let s = store();
        let mut draft = brief();
        draft.tools = Some(vec!["web_search".into()]);
        let job = create(&s, &draft).unwrap();
        assert_eq!(job.tools.as_deref(), Some(r#"["web_search"]"#));

        // And nothing in the edit path can add to it, because `Change` only
        // carries what a caller passed and the chat path never passes tools.
        let widened = update(
            &s,
            "pre-market-brief",
            &Change { prompt: Some("something else".into()), ..Default::default() },
        )
        .unwrap();
        assert_eq!(widened.tools.as_deref(), Some(r#"["web_search"]"#));
    }

    #[test]
    fn two_jobs_on_the_same_schedule_are_both_due_and_neither_blocks_the_other() {
        let s = store();
        create(&s, &brief()).unwrap();
        let mut other = brief();
        other.name = "evening-wrap".into();
        create(&s, &other).unwrap();

        let due = s.due_jobs(i64::MAX).unwrap();
        assert_eq!(due.len(), 2);
        // One failing does not touch the other's streak.
        s.job_fired(due[0].id, unix_now(), None, true).unwrap();
        assert_eq!(s.get_job(due[1].id).unwrap().unwrap().failures, 0);
    }
}

/// Who a scheduled answer actually reaches.
///
/// The bug these exist for: a job that said "send it to Telegram" without a
/// chat id was refused at creation, and a person who had already set Telegram
/// up and listed who may use it had no way to name a chat — a Telegram chat id
/// is not something anyone knows by heart.
#[cfg(test)]
mod delivery {
    use super::*;
    use ozgent_memory::Store;

    fn store() -> Store {
        Store::open_in_memory().unwrap()
    }

    /// A chat that has spoken to the bot, with the identities the allowlist
    /// is written in.
    fn seen(s: &Store, channel: &str, chat_id: &str, name: &str, identities: &[&str]) {
        let c = s.create_conversation(name, None).unwrap();
        s.bind_channel_chat(channel, chat_id, c, name, identities).unwrap();
    }

    #[test]
    fn a_channel_with_no_chat_named_is_accepted_rather_than_refused() {
        // This was the bug outright: `Deliver::parse` demanded an id.
        let d = Deliver::parse("telegram", None).expect("must be allowed");
        assert_eq!(d, Deliver::Chat { channel: "telegram".into(), to: None });
        assert_eq!(Deliver::parse("telegram", Some("  ")).unwrap(), d, "blank is the same as none");
        assert_eq!(Deliver::parse("whatsapp", None).unwrap().channel(), "whatsapp");
    }

    #[test]
    fn everyone_the_allowlist_names_gets_it() {
        // The case from the report: Telegram set up with `@Bzinga123`, a job
        // that just says "Telegram", and it has to reach that person.
        let s = store();
        seen(&s, "telegram", "5752856642", "Marco", &["5752856642", "@Bzinga123"]);
        let to = Deliver::parse("telegram", None).unwrap();
        assert_eq!(to.recipients(&s, &["@Bzinga123".into()]), ["5752856642"]);
    }

    #[test]
    fn a_handle_matches_however_it_is_written() {
        let s = store();
        seen(&s, "telegram", "42", "Marco", &["42", "@Bzinga123"]);
        let to = Deliver::parse("telegram", None).unwrap();
        for spelling in ["@Bzinga123", "Bzinga123", "@bzinga123", "  @BZINGA123  "] {
            assert_eq!(to.recipients(&s, &[spelling.into()]), ["42"], "for {spelling:?}");
        }
    }

    #[test]
    fn a_whatsapp_number_matches_the_chat_it_belongs_to() {
        let s = store();
        seen(&s, "whatsapp", "918638680186@s.whatsapp.net", "me", &["918638680186"]);
        let to = Deliver::parse("whatsapp", None).unwrap();
        assert_eq!(
            to.recipients(&s, &["918638680186".into()]),
            ["918638680186@s.whatsapp.net"]
        );
    }

    #[test]
    fn several_allowed_people_all_get_it() {
        let s = store();
        seen(&s, "telegram", "1", "Ada", &["1", "@ada"]);
        seen(&s, "telegram", "2", "Grace", &["2", "@grace"]);
        let to = Deliver::parse("telegram", None).unwrap();
        let mut got = to.recipients(&s, &["@ada".into(), "@grace".into()]);
        got.sort();
        assert_eq!(got, ["1", "2"]);
    }

    #[test]
    fn somebody_taken_off_the_list_stops_receiving() {
        // Checked when the message is sent, not when the job was written —
        // otherwise revoking access leaves scheduled messages still arriving.
        let s = store();
        seen(&s, "telegram", "1", "Ada", &["1", "@ada"]);
        seen(&s, "telegram", "2", "Grace", &["2", "@grace"]);
        let to = Deliver::parse("telegram", None).unwrap();
        assert_eq!(to.recipients(&s, &["@ada".into()]), ["1"], "Grace is no longer allowed");
        assert!(to.recipients(&s, &[]).is_empty(), "an empty list admits nobody");
    }

    #[test]
    fn a_job_that_names_one_chat_still_goes_only_there() {
        let s = store();
        seen(&s, "telegram", "1", "Ada", &["1", "@ada"]);
        seen(&s, "telegram", "2", "Grace", &["2", "@grace"]);
        let to = Deliver::parse("telegram", Some("2")).unwrap();
        assert_eq!(to.recipients(&s, &["@ada".into(), "@grace".into()]), ["2"]);
    }

    #[test]
    fn a_chat_bound_before_identities_were_kept_still_matches_by_its_id() {
        // Rows written before the column existed have no identities. Their
        // chat id is the one identity always available, and on Telegram that
        // is the numeric user id an allowlist can name.
        let s = store();
        seen(&s, "telegram", "5752856642", "Marco", &[]);
        let to = Deliver::parse("telegram", None).unwrap();
        assert_eq!(to.recipients(&s, &["5752856642".into()]), ["5752856642"]);
    }

    #[test]
    fn a_channel_nobody_has_messaged_yet_has_nobody_to_send_to() {
        // Not an error here — the runner turns it into a sentence that says
        // to message the bot once.
        let s = store();
        let to = Deliver::parse("telegram", None).unwrap();
        assert!(to.recipients(&s, &["@someone".into()]).is_empty());
    }

    #[test]
    fn an_open_channel_reaches_everyone_it_has_spoken_to() {
        let s = store();
        seen(&s, "telegram", "1", "Ada", &["1", "@ada"]);
        seen(&s, "telegram", "2", "Grace", &["2", "@grace"]);
        let to = Deliver::parse("telegram", None).unwrap();
        assert_eq!(to.recipients(&s, &["*".into()]).len(), 2);
    }

    #[test]
    fn chats_on_another_channel_are_never_included() {
        let s = store();
        seen(&s, "telegram", "1", "Ada", &["1", "@ada"]);
        seen(&s, "whatsapp", "918638680186@s.whatsapp.net", "me", &["918638680186"]);
        let to = Deliver::parse("telegram", None).unwrap();
        assert_eq!(to.recipients(&s, &["*".into()]), ["1"]);
    }

    #[test]
    fn sending_nowhere_has_no_recipients_at_all() {
        let s = store();
        seen(&s, "telegram", "1", "Ada", &["1", "@ada"]);
        assert!(Deliver::Nowhere.recipients(&s, &["*".into()]).is_empty());
    }
}
