//! The `schedule` tool: scheduling from inside a conversation.
//!
//! Everything the scheduler can do is also reachable from `ozgent scheduler`
//! and from `/scheduler`, and those are the complete interfaces. This is the
//! one that matters day to day, because the moment you want a job is while you
//! are already talking about the thing — "do that every weekday at 9:20" — and
//! having to leave for a form loses the thought.
//!
//! It is offered as a [`ToolSource`] rather than a Python tool for one reason:
//! it writes to ozgent's own database, and the Python worker is deliberately
//! sandboxed away from it. Being a source also means it appears in the
//! terminal, the browser, the API and the chat channels without any of them
//! knowing it exists — they all go through the same [`Toolbox`].
//!
//! [`Toolbox`]: ozgent_tools::Toolbox
//!
//! # What a chat may not do
//!
//! The page can do more than a chat, on purpose. From a conversation you can
//! create a job, retime it, pause it, resume it, delete it and list them. You
//! cannot hand a job a *wider* set of tools than the conversation itself has,
//! and you cannot point one at a chat other than the one you are in. Both
//! would turn "schedule me a reminder" into a way to gain reach, and neither
//! is something a person actually asks for mid-sentence. They are on the page,
//! where there is a password in front of them.

use std::sync::Mutex;

use ozgent_core::ToolSpec;
use ozgent_core::permission::Effect;
use ozgent_memory::Store;
use ozgent_tools::host::ToolCallError;
use ozgent_tools::protocol::RpcError;
use ozgent_tools::source::{Boxed, ToolSource};
use serde_json::{Value, json};

use crate::{Change, Deliver, Draft, Problem, in_words, summarise, unix_now};

/// The name the model calls.
pub const TOOL_NAME: &str = "schedule";

/// Who is asking, and what they are allowed to schedule.
///
/// Set by whichever surface is running the turn, before the turn starts. The
/// defaults are the safe ones: no default destination, and no tools — so a
/// surface that forgets to say produces jobs that can only think, not act.
#[derive(Debug, Clone, Default)]
pub struct Caller {
    /// For the audit trail on the job: `cli`, `web`, `chat:telegram:4242`.
    pub origin: String,
    /// Where a job created here sends its answers unless told otherwise, and
    /// the *only* place it may be pointed at from a chat.
    pub deliver: Option<Deliver>,
    /// The tools this conversation may use. A job created here gets these and
    /// no more. `None` means the surface did not narrow anything.
    pub allowed_tools: Option<Vec<String>>,
    /// The conversation this is being asked from, so a job made here can be
    /// threaded into it rather than starting a new one.
    pub conversation_id: Option<i64>,
}

impl Caller {
    /// The terminal or the browser: trusted, no default destination.
    pub fn local(origin: &str) -> Self {
        Self { origin: origin.to_string(), ..Default::default() }
    }

    /// A chat. Its own chat is the default and only destination.
    pub fn chat(channel: &str, chat_id: &str) -> Self {
        Self {
            origin: format!("chat:{channel}:{chat_id}"),
            deliver: Some(Deliver::Chat { channel: channel.into(), to: Some(chat_id.into()) }),
            ..Default::default()
        }
    }

    /// Whether this caller may send a job's answers to `deliver`.
    ///
    /// A local caller may name anywhere; a chat may only name itself. The
    /// difference is that someone messaging a bot is not necessarily the owner
    /// of the machine, and "schedule a job that messages *this other number*"
    /// is reach they were never given.
    fn may_deliver_to(&self, deliver: &Deliver) -> bool {
        match (&self.deliver, deliver) {
            (_, Deliver::Nowhere) => true,
            // No default destination means a local surface: anywhere is fine.
            (None, _) => true,
            (Some(mine), theirs) => mine == theirs,
        }
    }
}

/// The scheduler, as the model sees it.
///
/// Holds its own connection to the database rather than sharing one. SQLite in
/// WAL mode is built for exactly this, ozgent already relies on it across
/// processes, and the alternative — threading a shared handle through every
/// surface — would have meant changing all of them to hand one over.
pub struct ScheduleTools {
    store: Mutex<Store>,
    caller: Mutex<Caller>,
    specs: Vec<ToolSpec>,
}

impl ScheduleTools {
    /// Open the scheduler against ozgent's database.
    pub fn open(root: &std::path::Path) -> Result<Self, ozgent_memory::StoreError> {
        Ok(Self::with_store(Store::open(root.join("ozgent.db"))?))
    }

    pub fn with_store(store: Store) -> Self {
        Self {
            store: Mutex::new(store),
            caller: Mutex::new(Caller::default()),
            specs: vec![spec()],
        }
    }

    /// Say who the next turn is for. Called before each turn.
    pub fn set_caller(&self, caller: Caller) {
        *self.caller.lock().unwrap_or_else(|e| e.into_inner()) = caller;
    }

    fn caller(&self) -> Caller {
        self.caller.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Run one call, with everything already unpacked.
    fn dispatch(&self, args: &Value) -> Result<Value, String> {
        let store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        let caller = self.caller();
        let action = args
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("create")
            .trim()
            .to_lowercase();
        let job_name = text(args, "job").or_else(|| text(args, "name"));

        match action.as_str() {
            "create" => {
                let deliver = self.wanted_delivery(args, &caller)?;
                let draft = Draft {
                    name: text(args, "name").ok_or("a job needs a name")?,
                    prompt: text(args, "prompt").ok_or("a job needs something to ask")?,
                    when: text(args, "when").ok_or(
                        "a job needs a time — \"every weekday at 9:20\", \"every 2 hours\", \
                         or a cron rule",
                    )?,
                    zone: text(args, "timezone"),
                    agent: text(args, "agent"),
                    model: None,
                    only_if: text(args, "only_if"),
                    tools: caller.allowed_tools.clone(),
                    deliver: Some(deliver),
                    conversation_id: caller.conversation_id,
                    created_by: caller.origin.clone(),
                };
                let job = crate::create(&store, &draft).map_err(describe)?;
                Ok(json!({
                    "scheduled": true,
                    "job": job.name,
                    "summary": summarise(&job),
                    "next_run": job.next_run_at.map(|at| in_words(at, unix_now())),
                    "note": "Tell the user it is scheduled and when it next runs.",
                }))
            }

            "update" | "change" | "reschedule" => {
                let name = job_name.ok_or("say which job to change")?;
                let mut change = Change {
                    prompt: text(args, "prompt"),
                    when: text(args, "when"),
                    zone: text(args, "timezone"),
                    ..Default::default()
                };
                // `agent` and `only_if` are clearable, so an empty string is a
                // request to clear rather than an absent field.
                if let Some(v) = args.get("agent").and_then(Value::as_str) {
                    change.agent = Some(Some(v.to_string()).filter(|s| !s.trim().is_empty()));
                }
                if let Some(v) = args.get("only_if").and_then(Value::as_str) {
                    change.only_if = Some(Some(v.to_string()).filter(|s| !s.trim().is_empty()));
                }
                if args.get("deliver").is_some() || args.get("deliver_to").is_some() {
                    change.deliver = Some(self.wanted_delivery(args, &caller)?);
                }
                if change == Change::default() {
                    return Err("say what to change about it".into());
                }
                let job = crate::update(&store, &name, &change).map_err(describe)?;
                Ok(json!({
                    "changed": true,
                    "job": job.name,
                    "summary": summarise(&job),
                    "next_run": job.next_run_at.map(|at| in_words(at, unix_now())),
                }))
            }

            "pause" | "disable" | "stop" | "resume" | "enable" | "start" => {
                let name = job_name.ok_or("say which job")?;
                let on = matches!(action.as_str(), "resume" | "enable" | "start");
                let job = crate::update(&store, &name, &Change { enabled: Some(on), ..Default::default() })
                    .map_err(describe)?;
                Ok(json!({
                    "job": job.name,
                    "paused": !on,
                    "summary": summarise(&job),
                    "next_run": job.next_run_at.map(|at| in_words(at, unix_now())),
                }))
            }

            "delete" | "remove" | "cancel" => {
                let name = job_name.ok_or("say which job to delete")?;
                let job = store
                    .job_by_name(&name)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| describe(Problem::NoSuchJob(name.clone())))?;
                store.delete_job(job.id).map_err(|e| e.to_string())?;
                Ok(json!({ "deleted": true, "job": job.name }))
            }

            "run" | "run_now" => {
                let name = job_name.ok_or("say which job to run")?;
                let job = store
                    .job_by_name(&name)
                    .map_err(|e| e.to_string())?
                    .ok_or_else(|| describe(Problem::NoSuchJob(name.clone())))?;
                store.set_next_run(job.id, Some(unix_now())).map_err(|e| e.to_string())?;
                crate::wake();
                Ok(json!({
                    "job": job.name,
                    "queued": true,
                    // Said plainly because it is not instant, and a model that
                    // thinks it is will report an answer that has not happened.
                    "note": "It starts in a moment, and takes as long as the question \
                             takes. Its answer is delivered the way the job says, not \
                             as part of this conversation.",
                }))
            }

            "list" | "show" => {
                if let Some(name) = job_name.filter(|_| action == "show") {
                    let job = store
                        .job_by_name(&name)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| describe(Problem::NoSuchJob(name.clone())))?;
                    let runs = store.job_runs(job.id, 5).map_err(|e| e.to_string())?;
                    return Ok(json!({
                        "job": job.name,
                        "summary": summarise(&job),
                        "asks": job.prompt,
                        "paused": !job.enabled,
                        "next_run": job.next_run_at.map(|at| in_words(at, unix_now())),
                        "recent_runs": runs.iter().map(|r| json!({
                            "status": r.status.as_str(),
                            "delivered": r.delivered,
                            "error": r.error,
                        })).collect::<Vec<_>>(),
                    }));
                }
                let jobs = store.list_jobs().map_err(|e| e.to_string())?;
                let now = unix_now();
                Ok(json!({
                    "count": jobs.len(),
                    "jobs": jobs.iter().map(|j| json!({
                        "name": j.name,
                        "summary": summarise(j),
                        "paused": !j.enabled,
                        "next_run": j.next_run_at.map(|at| in_words(at, now)),
                    })).collect::<Vec<_>>(),
                }))
            }

            other => Err(format!(
                "{other:?} is not something the scheduler does. Use create, update, pause, \
                 resume, delete, run, list or show."
            )),
        }
    }

    /// The destination a call is asking for, checked against what it may have.
    fn wanted_delivery(&self, args: &Value, caller: &Caller) -> Result<Deliver, String> {
        let asked = args.get("deliver").and_then(Value::as_str).map(str::trim);
        let wanted = match asked {
            // Nothing said: a chat sends to itself, a local surface sends
            // nowhere. Both are what the person would have picked.
            None => return Ok(caller.deliver.clone().unwrap_or(Deliver::Nowhere)),
            Some(channel) => {
                let to = text(args, "deliver_to")
                    .or_else(|| caller.deliver.as_ref().and_then(|d| d.to().map(str::to_string)));
                Deliver::parse(channel, to.as_deref())?
            }
        };
        if !caller.may_deliver_to(&wanted) {
            return Err(
                "a job created from a chat can only send its answers back to that same chat. \
                 Use the scheduler page to send somewhere else."
                    .into(),
            );
        }
        Ok(wanted)
    }
}

/// A [`Problem`] as the model should read it.
fn describe(p: Problem) -> String {
    p.to_string()
}

fn text(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

impl ToolSource for ScheduleTools {
    fn origin(&self) -> &str {
        "scheduler"
    }

    fn tools(&self) -> &[ToolSpec] {
        &self.specs
    }

    fn call<'a>(
        &'a self,
        name: &'a str,
        arguments: Value,
        _approved: bool,
    ) -> Boxed<'a, Result<Value, ToolCallError>> {
        Box::pin(async move {
            if name != TOOL_NAME {
                return Err(not_found(name));
            }
            // Every failure comes back as a tool error rather than an Err
            // result so the model reads the reason and can correct itself —
            // "that name is taken" is a retry, not a dead end.
            self.dispatch(&arguments).map_err(|message| ToolCallError::Failed {
                name: TOOL_NAME.to_string(),
                error: RpcError { code: -32000, message, data: None },
            })
        })
    }
}

fn not_found(name: &str) -> ToolCallError {
    ToolCallError::Failed {
        name: name.to_string(),
        error: RpcError {
            code: -32601,
            message: format!("there is no tool called {name}"),
            data: None,
        },
    }
}

/// What the model is told the tool does.
///
/// Written to be read by a small model: the actions are listed, the time
/// formats are shown rather than described, and the two things that surprise
/// people — that a job cannot approve tools, and that `run` is not instant —
/// are stated here rather than discovered.
fn spec() -> ToolSpec {
    ToolSpec {
        name: TOOL_NAME.into(),
        description: "Schedule ozgent to ask itself something later, on a repeating timer, and \
send the answer to a chat. Use it when the user asks for something regular \
(\"every weekday at 9:20\", \"every morning\", \"every 2 hours\") or for one \
time in the future, and to list, retime, pause, resume or delete jobs they \
already have.\n\
\n\
Times can be written plainly — \"every weekday at 9:20\", \"every day at 6pm\", \
\"every monday at 18:00\", \"every 30 minutes\", \"in 2 hours\" — or as a cron \
rule like \"20 9 * * 1-5\".\n\
\n\
Times are the user's own local time unless they say otherwise. Do not convert \
to UTC and do not set `timezone` when they simply said \"4pm\" — that is 4pm \
where they are. Only set `timezone` if they name a zone, and then write it in \
full (\"Asia/Kolkata\", \"Europe/London\", \"UTC\").\n\
\n\
A scheduled job runs with nobody watching, so it can never approve a tool call: \
tools that run without asking still run, anything that would ask is refused. \
Say so if the user schedules something that would need approval.\n\
\n\
`run` queues a job to run within about a minute; it does not produce an answer \
here."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "update", "pause", "resume", "delete", "run", "list", "show"],
                    "description": "What to do. Defaults to create.",
                },
                "name": {
                    "type": "string",
                    "description": "A short name for a new job, in-words-like-this, \
                                    e.g. pre-market-brief. Required to create.",
                },
                "job": {
                    "type": "string",
                    "description": "Which existing job to act on, by name.",
                },
                "prompt": {
                    "type": "string",
                    "description": "What to ask each time the job runs. Write it as a complete \
                                    question — the job has no other context.",
                },
                "when": {
                    "type": "string",
                    "description": "When it runs: \"every weekday at 9:20\", \"every day at 6pm\", \
                                    \"every 30 minutes\", \"in 2 hours\", or cron \"20 9 * * 1-5\".",
                },
                "agent": {
                    "type": "string",
                    "description": "An agent to ask instead of the default model, without the @. \
                                    Send an empty string to stop using one.",
                },
                "only_if": {
                    "type": "string",
                    "description": "Only send the answer when this is true, e.g. \"only if the \
                                    price moved more than 2%\". Leave out to send every time.",
                },
                "deliver": {
                    "type": "string",
                    "enum": ["telegram", "whatsapp", "none"],
                    "description": "Where the answer goes. Defaults to this chat when asked from \
                                    one, and nowhere otherwise.",
                },
                "deliver_to": {
                    "type": "string",
                    "description": "One chat to send to. Almost never needed: left out, the \
                                    answer goes to this chat when asked from one, and to \
                                    everyone the channel allows otherwise.",
                },
                "timezone": {
                    "type": "string",
                    "description": "Only when the user names a zone. An IANA name like \
                                    Asia/Kolkata or Europe/London, or UTC. Leave it out for \
                                    a plain time like \"4pm\", which means 4pm where they are.",
                },
            },
            "required": [],
            "additionalProperties": false,
        }),
        output_schema: None,
        // Creating a job is creating something durable that will run later,
        // by itself. That is worth being asked about once.
        effect: Effect::Write,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> ScheduleTools {
        ScheduleTools::with_store(Store::open_in_memory().unwrap())
    }

    /// Call the tool the way the toolbox does, and unwrap the happy path.
    async fn call(t: &ScheduleTools, args: Value) -> Result<Value, String> {
        match t.call(TOOL_NAME, args, false).await {
            Ok(v) => Ok(v),
            Err(ToolCallError::Failed { error, .. }) => Err(error.message),
            Err(e) => Err(e.to_string()),
        }
    }

    fn brief() -> Value {
        json!({
            "action": "create",
            "name": "pre-market-brief",
            "prompt": "a pre-market brief on the Nifty",
            "when": "every weekday at 9:20",
            "agent": "stock-guru",
        })
    }

    #[tokio::test]
    async fn a_model_can_schedule_something_from_a_chat() {
        let t = tools();
        t.set_caller(Caller::chat("telegram", "4242"));
        let out = call(&t, brief()).await.unwrap();
        assert_eq!(out["scheduled"], json!(true));
        assert_eq!(out["job"], json!("pre-market-brief"));
        assert!(out["summary"].as_str().unwrap().contains("every weekday"));
        assert!(out["next_run"].as_str().is_some(), "it must say when");
    }

    #[tokio::test]
    async fn a_job_made_in_a_chat_answers_into_that_chat_without_being_told() {
        // The whole point of scheduling from a chat: the destination is
        // obvious and asking for it would be a form.
        let t = tools();
        t.set_caller(Caller::chat("telegram", "4242"));
        call(&t, brief()).await.unwrap();

        let store = t.store.lock().unwrap();
        let job = store.job_by_name("pre-market-brief").unwrap().unwrap();
        assert_eq!(job.deliver, "telegram");
        assert_eq!(job.deliver_to.as_deref(), Some("4242"));
        assert_eq!(job.created_by, "chat:telegram:4242");
    }

    #[tokio::test]
    async fn a_job_made_from_a_chat_cannot_be_pointed_at_someone_elses_chat() {
        // Otherwise "schedule me a reminder" is a way to make ozgent message
        // a number the asker was never allowed to reach.
        let t = tools();
        t.set_caller(Caller::chat("telegram", "4242"));
        let mut args = brief();
        args["deliver"] = json!("telegram");
        args["deliver_to"] = json!("9999");

        let err = call(&t, args).await.unwrap_err();
        assert!(err.contains("same chat"), "{err}");
        assert!(t.store.lock().unwrap().list_jobs().unwrap().is_empty(), "nothing stored");
    }

    #[tokio::test]
    async fn a_job_made_from_a_chat_may_still_choose_to_send_nowhere() {
        let t = tools();
        t.set_caller(Caller::chat("telegram", "4242"));
        let mut args = brief();
        args["deliver"] = json!("none");
        call(&t, args).await.unwrap();
        let store = t.store.lock().unwrap();
        assert_eq!(store.job_by_name("pre-market-brief").unwrap().unwrap().deliver, "none");
    }

    #[tokio::test]
    async fn the_terminal_may_send_a_job_anywhere() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        let mut args = brief();
        args["deliver"] = json!("whatsapp");
        args["deliver_to"] = json!("918638680186");
        call(&t, args).await.unwrap();

        let store = t.store.lock().unwrap();
        let job = store.job_by_name("pre-market-brief").unwrap().unwrap();
        assert_eq!(job.deliver, "whatsapp");
        assert_eq!(job.deliver_to.as_deref(), Some("918638680186"));
    }

    #[tokio::test]
    async fn a_job_gets_the_tools_the_conversation_had_and_no_more() {
        // A channel narrows what the model may call; a job created from it
        // must not quietly widen that.
        let t = tools();
        let mut caller = Caller::chat("telegram", "4242");
        caller.allowed_tools = Some(vec!["web_search".into(), "fetch_url".into()]);
        t.set_caller(caller);
        call(&t, brief()).await.unwrap();

        let store = t.store.lock().unwrap();
        let job = store.job_by_name("pre-market-brief").unwrap().unwrap();
        assert_eq!(job.tools.as_deref(), Some(r#"["web_search","fetch_url"]"#));
    }

    #[tokio::test]
    async fn a_bad_time_comes_back_as_something_the_model_can_act_on() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        let mut args = brief();
        args["when"] = json!("every blursday at 9am");
        let err = call(&t, args).await.unwrap_err();
        assert!(err.contains("day") || err.contains("cron"), "unhelpful: {err}");
    }

    #[tokio::test]
    async fn a_missing_field_says_which_one() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        for (missing, expect) in [("name", "name"), ("prompt", "ask"), ("when", "time")] {
            let mut args = brief();
            args.as_object_mut().unwrap().remove(missing);
            let err = call(&t, args).await.unwrap_err();
            assert!(err.contains(expect), "removing {missing} gave {err:?}");
        }
    }

    #[tokio::test]
    async fn the_name_clash_is_explained_rather_than_reported_as_a_database_error() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, brief()).await.unwrap();
        let err = call(&t, brief()).await.unwrap_err();
        assert!(err.contains("already a job"), "{err}");
        assert!(!err.contains("UNIQUE"), "a SQL error reached the model: {err}");
    }

    #[tokio::test]
    async fn a_job_can_be_retimed_from_a_chat() {
        let t = tools();
        t.set_caller(Caller::chat("telegram", "4242"));
        call(&t, brief()).await.unwrap();

        let out = call(
            &t,
            json!({ "action": "update", "job": "pre-market-brief", "when": "every weekday at 8:00" }),
        )
        .await
        .unwrap();
        assert_eq!(out["changed"], json!(true));
        assert!(out["summary"].as_str().unwrap().contains("08:00"), "{}", out["summary"]);
    }

    #[tokio::test]
    async fn a_job_can_be_paused_and_resumed_from_a_chat() {
        let t = tools();
        t.set_caller(Caller::chat("telegram", "4242"));
        call(&t, brief()).await.unwrap();

        let off = call(&t, json!({ "action": "pause", "job": "pre-market-brief" })).await.unwrap();
        assert_eq!(off["paused"], json!(true));
        assert_eq!(off["next_run"], Value::Null, "a paused job has no next run");

        let on = call(&t, json!({ "action": "resume", "job": "pre-market-brief" })).await.unwrap();
        assert_eq!(on["paused"], json!(false));
        assert!(on["next_run"].as_str().is_some());
    }

    #[tokio::test]
    async fn a_job_can_be_deleted_from_a_chat() {
        let t = tools();
        t.set_caller(Caller::chat("telegram", "4242"));
        call(&t, brief()).await.unwrap();
        let out = call(&t, json!({ "action": "delete", "job": "pre-market-brief" })).await.unwrap();
        assert_eq!(out["deleted"], json!(true));
        assert!(t.store.lock().unwrap().list_jobs().unwrap().is_empty());
    }

    #[tokio::test]
    async fn listing_says_what_is_scheduled_and_when() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, brief()).await.unwrap();
        let out = call(&t, json!({ "action": "list" })).await.unwrap();
        assert_eq!(out["count"], json!(1));
        let first = &out["jobs"][0];
        assert_eq!(first["name"], json!("pre-market-brief"));
        assert!(first["next_run"].as_str().is_some());
    }

    #[tokio::test]
    async fn listing_nothing_is_a_count_of_zero_rather_than_an_error() {
        let t = tools();
        let out = call(&t, json!({ "action": "list" })).await.unwrap();
        assert_eq!(out["count"], json!(0));
        assert_eq!(out["jobs"], json!([]));
    }

    #[tokio::test]
    async fn showing_a_job_includes_how_its_recent_runs_went() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, brief()).await.unwrap();
        {
            let store = t.store.lock().unwrap();
            let job = store.job_by_name("pre-market-brief").unwrap().unwrap();
            let run = store.start_run(job.id, 1_000).unwrap();
            store
                .finish_run(run, ozgent_memory::JobStatus::Ok, Some("the brief"), None, true)
                .unwrap();
        }
        let out = call(&t, json!({ "action": "show", "job": "pre-market-brief" })).await.unwrap();
        assert_eq!(out["recent_runs"][0]["status"], json!("ok"));
        assert_eq!(out["recent_runs"][0]["delivered"], json!(true));
        assert_eq!(out["asks"], json!("a pre-market brief on the Nifty"));
    }

    #[tokio::test]
    async fn running_a_job_now_says_it_is_queued_rather_than_answered() {
        // A model told "done" would report a brief that has not been written.
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, brief()).await.unwrap();
        let out = call(&t, json!({ "action": "run", "job": "pre-market-brief" })).await.unwrap();
        assert_eq!(out["queued"], json!(true));
        assert!(out["note"].as_str().unwrap().contains("not as part of this conversation"));

        let store = t.store.lock().unwrap();
        let job = store.job_by_name("pre-market-brief").unwrap().unwrap();
        assert!(job.next_run_at.unwrap() <= unix_now(), "it is due now");
    }

    #[tokio::test]
    async fn acting_on_a_job_that_does_not_exist_says_so_for_every_action() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        for action in ["update", "pause", "resume", "delete", "run", "show"] {
            let err = call(&t, json!({ "action": action, "job": "ghost", "when": "every hour" }))
                .await
                .unwrap_err();
            assert!(err.contains("ghost"), "{action}: {err}");
        }
    }

    #[tokio::test]
    async fn an_update_that_changes_nothing_asks_what_to_change() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, brief()).await.unwrap();
        let err = call(&t, json!({ "action": "update", "job": "pre-market-brief" }))
            .await
            .unwrap_err();
        assert!(err.contains("what to change"), "{err}");
    }

    #[tokio::test]
    async fn an_unknown_action_lists_the_ones_that_exist() {
        let t = tools();
        let err = call(&t, json!({ "action": "teleport", "job": "x" })).await.unwrap_err();
        assert!(err.contains("create"), "{err}");
    }

    #[tokio::test]
    async fn creating_is_the_default_action_because_it_is_what_is_usually_meant() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        let mut args = brief();
        args.as_object_mut().unwrap().remove("action");
        assert_eq!(call(&t, args).await.unwrap()["scheduled"], json!(true));
    }

    #[tokio::test]
    async fn the_agent_can_be_taken_off_a_job_with_an_empty_string() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, brief()).await.unwrap();
        call(&t, json!({ "action": "update", "job": "pre-market-brief", "agent": "" }))
            .await
            .unwrap();
        let store = t.store.lock().unwrap();
        assert_eq!(store.job_by_name("pre-market-brief").unwrap().unwrap().agent, None);
    }

    #[tokio::test]
    async fn calling_something_that_is_not_this_tool_is_reported_normally() {
        let t = tools();
        let err = call(&t, json!({})).await.unwrap_err();
        assert!(!err.is_empty());
        match t.call("something_else", json!({}), false).await {
            Err(ToolCallError::Failed { error, .. }) => {
                assert!(error.message.contains("no tool called"), "{}", error.message);
            }
            other => panic!("expected a not-found error, got {other:?}"),
        }
    }

    #[test]
    fn the_tool_is_offered_under_one_stable_name() {
        // The name goes into the prompt and into permission rules; changing it
        // silently would invalidate both.
        let t = tools();
        assert_eq!(t.tools().len(), 1);
        assert_eq!(t.tools()[0].name, "schedule");
        assert_eq!(t.origin(), "scheduler");
    }

    #[test]
    fn scheduling_is_a_write_so_it_is_asked_about_once() {
        assert_eq!(spec().effect, Effect::Write);
    }

    #[test]
    fn the_description_warns_about_the_two_things_that_surprise_people() {
        let d = spec().description;
        assert!(d.contains("approve"), "must say a job cannot approve tools");
        assert!(d.contains("within about a minute") || d.contains("does not produce"),
                "must say run is not instant");
    }
}

#[cfg(test)]
mod zones {
    use super::*;
    use serde_json::json;

    fn tools() -> ScheduleTools {
        ScheduleTools::with_store(Store::open_in_memory().unwrap())
    }

    async fn call(t: &ScheduleTools, args: Value) -> Result<Value, String> {
        match t.call(TOOL_NAME, args, false).await {
            Ok(v) => Ok(v),
            Err(ToolCallError::Failed { error, .. }) => Err(error.message),
            Err(e) => Err(e.to_string()),
        }
    }

    fn job(name: &str, when: &str) -> Value {
        json!({ "action": "create", "name": name, "prompt": "a brief", "when": when })
    }

    #[tokio::test]
    async fn a_plain_time_is_stored_as_local_not_as_a_fixed_zone() {
        // The user is in India and said "4pm". That has to keep meaning the
        // clock in the room, including if this machine is ever moved.
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, job("afternoon", "every day at 4pm")).await.unwrap();
        let store = t.store.lock().unwrap();
        let stored = store.job_by_name("afternoon").unwrap().unwrap();
        assert_eq!(stored.zone, "local");
        assert_eq!(stored.recur, "cron 0 16 * * *");
    }

    #[tokio::test]
    async fn a_zone_named_in_the_phrase_is_taken_from_it() {
        // "every day at 4pm UTC" used to be a parse error.
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, job("utc-job", "every day at 4pm UTC")).await.unwrap();
        let store = t.store.lock().unwrap();
        let stored = store.job_by_name("utc-job").unwrap().unwrap();
        assert_eq!(stored.zone, "UTC");
        assert_eq!(stored.recur, "cron 0 16 * * *");
    }

    #[tokio::test]
    async fn an_explicit_timezone_field_still_wins() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        let mut args = job("explicit", "every day at 4pm");
        args["timezone"] = json!("UTC");
        call(&t, args).await.unwrap();
        let store = t.store.lock().unwrap();
        assert_eq!(store.job_by_name("explicit").unwrap().unwrap().zone, "UTC");
    }

    #[tokio::test]
    async fn retiming_into_a_named_zone_moves_the_zone_with_it() {
        // Otherwise "make it 4pm UTC" leaves the job reading 4pm locally.
        let t = tools();
        t.set_caller(Caller::local("cli"));
        call(&t, job("shifty", "every day at 9am")).await.unwrap();
        call(&t, json!({ "action": "update", "job": "shifty", "when": "every day at 4pm UTC" }))
            .await
            .unwrap();
        let store = t.store.lock().unwrap();
        let stored = store.job_by_name("shifty").unwrap().unwrap();
        assert_eq!(stored.zone, "UTC");
        assert_eq!(stored.recur, "cron 0 16 * * *");
    }

    #[tokio::test]
    async fn an_ambiguous_abbreviation_comes_back_as_advice_rather_than_a_guess() {
        let t = tools();
        t.set_caller(Caller::local("cli"));
        let err = call(&t, job("ambiguous", "every day at 4pm IST")).await.unwrap_err();
        assert!(err.contains("Asia/Kolkata"), "{err}");
        assert!(t.store.lock().unwrap().list_jobs().unwrap().is_empty());
    }

    #[test]
    fn the_description_tells_the_model_not_to_convert_to_utc() {
        // The failure this prevents: a model helpfully "converting" 4pm IST to
        // 10:30 UTC and storing that as a local time.
        let d = spec().description;
        assert!(d.contains("Do not convert"), "{d}");
        assert!(d.contains("local time"), "{d}");
    }
}
