//! `ozgent scheduler`: jobs from the terminal.
//!
//! The same jobs the `/scheduler` page shows and the `schedule` tool creates —
//! one database, three ways in. This one exists because the terminal is where
//! you are when something has gone wrong, and because a command is scriptable
//! in a way a page is not.
//!
//! Every command works whether or not a daemon is running. A job written here
//! is a row; the next scheduler to look picks it up. The one thing that cannot
//! be faked is *running* a job, which needs a model — so `run` marks it due and
//! says who will pick it up, rather than pretending to have run it.

use anyhow::{Context, Result};
use ozgent_core::{Config, Paths};
use ozgent_memory::jobs::{Job, Status};
use ozgent_memory::Store;
use ozgent_schedule::{Change, Deliver, Draft, in_words, summarise, unix_now};

use crate::cli::SchedulerCommand;
use crate::setup::{ask, choose, confirm};

pub fn run(paths: &Paths, config: &Config, command: Option<SchedulerCommand>) -> Result<()> {
    let store = Store::open(paths.root().join("ozgent.db"))
        .context("opening ozgent's database")?;

    match command.unwrap_or(SchedulerCommand::List) {
        SchedulerCommand::List => list(paths, &store),
        SchedulerCommand::Add { name, prompt, when, agent, only_if, deliver, to, timezone } => {
            add(paths, config, &store, Added { name, prompt, when, agent, only_if, deliver, to, timezone })
        }
        SchedulerCommand::Show { job } => show(&store, &job),
        SchedulerCommand::Set { job, name, prompt, when, agent, only_if, deliver, to, timezone } => {
            set(&store, &job, Added { name, prompt, when, agent, only_if, deliver, to, timezone })
        }
        SchedulerCommand::Pause { job } => enable(&store, &job, false),
        SchedulerCommand::Resume { job } => enable(&store, &job, true),
        SchedulerCommand::Run { job } => run_now(paths, &store, &job),
        SchedulerCommand::Rm { job, force } => remove(&store, &job, force),
        SchedulerCommand::When { when, timezone } => when_does_it_run(&when.join(" "), timezone),
    }
}

/// The flags `add` and `set` share.
struct Added {
    name: Option<String>,
    prompt: Option<String>,
    when: Option<String>,
    agent: Option<String>,
    only_if: Option<String>,
    deliver: Option<String>,
    to: Option<String>,
    timezone: Option<String>,
}

// ------------------------------------------------------------------ list

fn list(paths: &Paths, store: &Store) -> Result<()> {
    let jobs = store.list_jobs()?;
    if jobs.is_empty() {
        println!("nothing is scheduled");
        println!();
        println!("  ozgent scheduler add");
        println!();
        println!("Or ask for one in a chat: \"every weekday at 9:20, send me a pre-market brief\".");
        return Ok(());
    }

    let now = unix_now();
    for job in &jobs {
        let mark = match (job.enabled, job.failures > 0) {
            (false, _) => "paused ",
            (true, true) => "failing",
            (true, false) => "on     ",
        };
        println!("{mark} {}", summarise(job));
        match job.next_run_at {
            Some(at) if job.enabled => println!("         next {}", in_words(at, now)),
            _ => {}
        }
    }

    println!();
    if running(paths) {
        println!("A scheduler is running, so these will fire.");
    } else {
        // The failure nobody diagnoses: perfectly good jobs, and nothing
        // awake to run them.
        println!("Nothing is running them. Start the daemon:");
        println!();
        println!("    ozgent daemon install");
    }
    Ok(())
}

/// Whether some process is running jobs.
fn running(paths: &Paths) -> bool {
    let path = paths.scheduler_dir().join("lock");
    let Ok(file) = std::fs::OpenOptions::new().read(true).write(true).open(&path) else {
        return false;
    };
    match file.try_lock() {
        Ok(()) => {
            let _ = file.unlock();
            false
        }
        Err(_) => true,
    }
}

// ------------------------------------------------------------------- add

fn add(paths: &Paths, config: &Config, store: &Store, given: Added) -> Result<()> {
    // Given everything, this is a scriptable one-liner; given nothing, it is
    // a wizard. Half and half asks only for what is missing, which is what
    // someone who forgot one flag actually wants.
    let interactive = given.name.is_none() || given.prompt.is_none() || given.when.is_none();
    if interactive {
        println!("A scheduled job");
        println!();
    }

    let name = match given.name {
        Some(n) => n,
        None => ask("  a short name, like pre-market-brief\n  > ")?,
    };
    let prompt = match given.prompt {
        Some(p) => p,
        None => {
            println!();
            println!("  What should it ask, each time it runs? Write it as a whole question —");
            println!("  the job has no other context.");
            ask("  > ")?
        }
    };
    let when = match given.when {
        Some(w) => w,
        None => {
            println!();
            println!("  When? \"every weekday at 9:20\", \"every day at 6pm\", \"every 2 hours\",");
            println!("  or a cron rule like \"20 9 * * 1-5\".");
            ask("  > ")?
        }
    };

    let agent = match given.agent {
        Some(a) => Some(a),
        None if interactive => pick_agent(paths)?,
        None => None,
    };
    let deliver = match &given.deliver {
        Some(channel) => Deliver::parse(channel, given.to.as_deref()).map_err(anyhow::Error::msg)?,
        None if interactive => pick_delivery(config, store)?,
        None => Deliver::Nowhere,
    };

    let draft = Draft {
        name,
        prompt,
        when,
        zone: given.timezone,
        agent,
        model: None,
        only_if: given.only_if,
        tools: None,
        deliver: Some(deliver),
        conversation_id: None,
        created_by: "cli".into(),
    };
    let job = ozgent_schedule::create(store, &draft).map_err(anyhow::Error::msg)?;

    println!();
    println!("  ✓ {}", summarise(&job));
    if let Some(at) = job.next_run_at {
        println!("    first run {}", in_words(at, unix_now()));
    }
    if !running(paths) {
        println!();
        println!("  Nothing is running jobs yet:  ozgent daemon install");
    }
    Ok(())
}

/// Offer the agents this machine has, since a job is usually one agent's work.
fn pick_agent(paths: &Paths) -> Result<Option<String>> {
    let catalog = ozgent_core::AgentCatalog::load(paths);
    let agents = catalog.all();
    if agents.is_empty() {
        return Ok(None);
    }
    println!();
    let mut options = vec!["the default model".to_string()];
    options.extend(agents.iter().map(|a| format!("@{} — {}", a.name, a.definition.description)));
    let refs: Vec<&str> = options.iter().map(String::as_str).collect();
    let picked = choose("  Who should answer it?", &refs, 0)?;
    Ok((picked > 0).then(|| agents[picked - 1].name.clone()))
}

/// Offer somewhere to send the answer, from chats ozgent has actually seen.
///
/// Typing a Telegram chat id from memory is not something anyone can do, and
/// a job pointed at the wrong one fails silently every morning. The chats it
/// has spoken to are the only ones worth offering.
fn pick_delivery(config: &Config, store: &Store) -> Result<Deliver> {
    let mut chats: Vec<(String, String, String)> = Vec::new();
    for (channel, on) in [
        ("telegram", config.channels.telegram.enabled),
        ("whatsapp", config.channels.whatsapp.enabled),
    ] {
        if !on {
            continue;
        }
        for chat in store.channel_chats(channel).unwrap_or_default() {
            let who = if chat.display.is_empty() { chat.chat_id.clone() } else { chat.display.clone() };
            chats.push((channel.to_string(), chat.chat_id, who));
        }
    }
    // Every channel that is on gets an "everyone allowed" option, whether or
    // not anybody has messaged it yet — that is what most people want, and
    // making it depend on having a chat on record would hide the obvious
    // choice on a fresh install.
    let channels: Vec<&str> = [
        ("telegram", config.channels.telegram.enabled),
        ("whatsapp", config.channels.whatsapp.enabled),
    ]
    .into_iter()
    .filter(|(_, on)| *on)
    .map(|(name, _)| name)
    .collect();
    if channels.is_empty() {
        return Ok(Deliver::Nowhere);
    }

    println!();
    let mut options = vec!["keep it on the scheduler page".to_string()];
    let mut picks: Vec<Deliver> = vec![Deliver::Nowhere];
    for channel in &channels {
        let who = config.channels.access(match *channel {
            "telegram" => ozgent_core::ChannelKind::Telegram,
            _ => ozgent_core::ChannelKind::WhatsApp,
        });
        let audience = if who.allow.is_empty() {
            "nobody is allowed there yet".to_string()
        } else {
            who.allow.join(", ")
        };
        options.push(format!("{channel} — everyone allowed ({audience})"));
        picks.push(Deliver::Chat { channel: channel.to_string(), to: None });
    }
    // And any single chat it has actually spoken to, for a job meant for one
    // person when several are allowed.
    for (channel, to, who) in &chats {
        options.push(format!("{channel} — only {who}"));
        picks.push(Deliver::Chat { channel: channel.clone(), to: Some(to.clone()) });
    }

    let refs: Vec<&str> = options.iter().map(String::as_str).collect();
    let picked = choose("  Where should the answer go?", &refs, 0)?;
    Ok(picks.get(picked).cloned().unwrap_or(Deliver::Nowhere))
}

// ------------------------------------------------------------------ show

fn show(store: &Store, name: &str) -> Result<()> {
    let job = find(store, name)?;
    let now = unix_now();

    println!("{}", summarise(&job));
    println!();
    println!("  asks        {}", job.prompt);
    println!("  when        {}", job.recur);
    if job.zone != "local" {
        println!("  timezone    {}", job.zone);
    }
    if let Some(condition) = &job.only_if {
        println!("  only if     {condition}");
    }
    match job.next_run_at {
        Some(at) if job.enabled => println!("  next run    {}", in_words(at, now)),
        _ => println!("  next run    never — it is paused"),
    }
    println!("  runs        {}", job.runs);
    if job.failures > 0 {
        println!("  failing     {} in a row", job.failures);
    }
    println!("  created by  {}", job.created_by);

    let runs = store.job_runs(job.id, 10)?;
    if runs.is_empty() {
        println!();
        println!("  it has not run yet");
        return Ok(());
    }
    println!();
    println!("  recent runs");
    for run in runs {
        let when = ozgent_core::DateTime::from_unix(run.started_at);
        let mark = match run.status {
            Status::Ok => "✓",
            Status::Quiet => "·",
            Status::Error => "✗",
            Status::Missed => "–",
            Status::Running => "…",
        };
        let detail = match (&run.status, &run.error) {
            (Status::Error, Some(e)) => e.clone(),
            (Status::Quiet, _) => "nothing worth sending".into(),
            (Status::Missed, _) => "ozgent was not running".into(),
            (Status::Running, _) => "running now".into(),
            _ if run.delivered => "sent".into(),
            _ => "answered".into(),
        };
        println!(
            "    {mark} {} {:02}:{:02}  {detail}",
            when.iso_date(),
            when.hour,
            when.minute
        );
    }
    Ok(())
}

// ------------------------------------------------------------- changing

fn set(store: &Store, name: &str, given: Added) -> Result<()> {
    let deliver = match &given.deliver {
        Some(channel) => {
            Some(Deliver::parse(channel, given.to.as_deref()).map_err(anyhow::Error::msg)?)
        }
        None => None,
    };
    let change = Change {
        name: given.name,
        prompt: given.prompt,
        when: given.when,
        zone: given.timezone,
        enabled: None,
        // A flag that was given but empty clears the field; one that was not
        // given leaves it alone. `--agent ""` takes the agent off.
        agent: given.agent.map(|a| Some(a).filter(|a| !a.trim().is_empty())),
        model: None,
        only_if: given.only_if.map(|c| Some(c).filter(|c| !c.trim().is_empty())),
        tools: None,
        deliver,
    };
    if change == Change::default() {
        anyhow::bail!(
            "say what to change: --when, --prompt, --agent, --only-if, --deliver, --name"
        );
    }
    let job = ozgent_schedule::update(store, name, &change).map_err(anyhow::Error::msg)?;
    println!("✓ {}", summarise(&job));
    if let Some(at) = job.next_run_at {
        println!("  next run {}", in_words(at, unix_now()));
    }
    Ok(())
}

fn enable(store: &Store, name: &str, on: bool) -> Result<()> {
    let job = ozgent_schedule::update(
        store,
        name,
        &Change { enabled: Some(on), ..Default::default() },
    )
    .map_err(anyhow::Error::msg)?;
    println!("✓ {}", summarise(&job));
    Ok(())
}

fn run_now(paths: &Paths, store: &Store, name: &str) -> Result<()> {
    let job = find(store, name)?;
    store.set_next_run(job.id, Some(unix_now()))?;
    if running(paths) {
        println!("✓ {} will run within a minute", job.name);
        println!("  ozgent scheduler show {}   to see how it went", job.name);
    } else {
        // Marked due either way, so it runs the moment something starts. Said
        // plainly, because "queued" with nothing to run it is a lie.
        println!("✓ {} is due, but nothing is running jobs", job.name);
        println!();
        println!("    ozgent daemon install");
    }
    Ok(())
}

fn remove(store: &Store, name: &str, force: bool) -> Result<()> {
    let job = find(store, name)?;
    if !force {
        println!("{}", summarise(&job));
        if !confirm(&format!("delete {} and its history?", job.name), false)? {
            println!("left alone");
            return Ok(());
        }
    }
    store.delete_job(job.id)?;
    println!("✓ {} deleted", job.name);
    Ok(())
}

fn find(store: &Store, name: &str) -> Result<Job> {
    store.job_by_name(name)?.ok_or_else(|| {
        anyhow::anyhow!("there is no scheduled job called {name:?}. `ozgent scheduler` lists them.")
    })
}

// ------------------------------------------------------------------ when

/// Read a time back and print when it would actually fire.
///
/// Worth its own command: `20 9 * * 1-5` and `9 20 * * 1-5` are both valid and
/// only one of them is nine twenty in the morning. Seeing three real times is
/// how that gets caught before a week of nothing happening.
fn when_does_it_run(when: &str, timezone: Option<String>) -> Result<()> {
    let (when, inline) =
        ozgent_core::schedule::split_zone(when).map_err(anyhow::Error::msg)?;
    let recur = ozgent_core::Recur::parse(&when).map_err(anyhow::Error::msg)?;
    let zone = match timezone.as_deref().filter(|z| !z.is_empty()).or(inline.as_deref()) {
        None | Some("local") => ozgent_core::Zone::local(),
        Some(name) => ozgent_core::Zone::named(name).map_err(anyhow::Error::msg)?,
    };
    let now = unix_now();

    // A job in another zone is shown in that zone *and* in this one. Reading
    // "16:00 UTC" and working out what that is on your own clock is exactly
    // the arithmetic this command exists to save.
    let here = ozgent_core::Zone::local();
    let elsewhere = zone.offset_at(now) != here.offset_at(now);

    println!("{}", recur.describe(&zone));
    println!("  stored as   {recur}");
    println!("  timezone    {}", zone.name());
    if elsewhere {
        println!("  your clock  {} ({})", here.name(), here.label_at(now));
    }
    println!();

    let mut at = now;
    let mut any = false;
    for _ in 0..5 {
        let Some(next) = recur.next_after(at, now, &zone) else { break };
        let t = zone.local_at(next);
        let mine = if elsewhere {
            let m = here.local_at(next);
            format!("  =  {:02}:{:02} your time", m.hour, m.minute)
        } else {
            String::new()
        };
        println!(
            "  {} {}  {:02}:{:02}{mine}   {}",
            t.iso_date(),
            &t.weekday_name()[..3],
            t.hour,
            t.minute,
            in_words(next, now)
        );
        at = next;
        any = true;
    }
    if !any {
        anyhow::bail!("that describes a time that never comes around");
    }
    Ok(())
}
