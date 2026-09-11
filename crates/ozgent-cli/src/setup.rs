//! `ozgent gateway <channel>` and `ozgent admin`: setting things up by
//! answering questions, instead of editing `config.toml` by hand.
//!
//! Every change is saved to `config.toml` straight away. A running
//! `ozgent web` or `ozgent gateway` follows the file and applies it within a
//! couple of seconds, so none of this needs a restart.

use std::io::{IsTerminal, Write};
use std::time::Duration;

use anyhow::{Context, Result};
use ozgent_core::channels::{self, Kind, is_open_to_everyone};
use ozgent_core::{Config, Paths, secret};

use crate::cli::{AdminCommand, ChannelAction, GatewayCommand};

// ------------------------------------------------------------------ prompts

/// Read one line. `None` at the end of input, so a script piping answers in
/// runs out cleanly instead of looping.
fn read_line() -> Option<String> {
    let mut s = String::new();
    match std::io::stdin().read_line(&mut s) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(s.trim_end_matches(['\n', '\r']).to_string()),
    }
}

fn ask(prompt: &str) -> Result<String> {
    print!("{prompt}");
    std::io::stdout().flush()?;
    read_line().context("no more input")
}

/// A yes/no question. Enter takes the default.
fn confirm(prompt: &str, default: bool) -> Result<bool> {
    let hint = if default { "[Y/n]" } else { "[y/N]" };
    loop {
        let a = ask(&format!("{prompt} {hint} "))?;
        match a.trim().to_ascii_lowercase().as_str() {
            "" => return Ok(default),
            "y" | "yes" => return Ok(true),
            "n" | "no" => return Ok(false),
            _ => println!("  y or n"),
        }
    }
}

/// Pick one of several numbered options. Returns the index.
fn choose(prompt: &str, options: &[&str], default: usize) -> Result<usize> {
    println!("{prompt}");
    for (i, o) in options.iter().enumerate() {
        println!("  {}  {o}", i + 1);
    }
    loop {
        let a = ask(&format!("> [{}] ", default + 1))?;
        let a = a.trim();
        if a.is_empty() {
            return Ok(default);
        }
        match a.parse::<usize>() {
            Ok(n) if (1..=options.len()).contains(&n) => return Ok(n - 1),
            _ => println!("  a number from 1 to {}", options.len()),
        }
    }
}

/// Read something secret without echoing it. Falls back to a plain line when
/// stdin is not a terminal, so it can be piped in.
fn ask_secret(prompt: &str) -> Result<String> {
    use ozgent_render::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers, read};
    use ozgent_render::crossterm::terminal::{disable_raw_mode, enable_raw_mode};

    print!("{prompt}");
    std::io::stdout().flush()?;
    if !std::io::stdin().is_terminal() {
        let line = read_line().context("no more input");
        println!();
        return line;
    }
    enable_raw_mode()?;
    let mut out = String::new();
    let result = loop {
        let Ok(event) = read() else { break Err(anyhow::anyhow!("could not read the terminal")) };
        let Event::Key(key) = event else { continue };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        match key.code {
            KeyCode::Enter => break Ok(()),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                break Err(anyhow::anyhow!("cancelled"));
            }
            KeyCode::Backspace => {
                if out.pop().is_some() {
                    print!("\x08 \x08");
                }
            }
            KeyCode::Char(c) => {
                out.push(c);
                print!("•");
            }
            _ => {}
        }
        let _ = std::io::stdout().flush();
    };
    disable_raw_mode()?;
    println!();
    result.map(|()| out)
}

/// A spinner on one line while `work` runs, for steps that take a while and
/// have no way of saying how far along they are.
async fn spinning<T>(label: &str, work: impl std::future::Future<Output = T>) -> T {
    let tty = std::io::stdout().is_terminal();
    let label = label.to_string();
    let spin = tokio::spawn(async move {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let started = std::time::Instant::now();
        let mut i = 0;
        loop {
            if tty {
                print!("\r  {} {label} {}s ", FRAMES[i % FRAMES.len()], started.elapsed().as_secs());
                let _ = std::io::stdout().flush();
            }
            i += 1;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });
    let out = work.await;
    spin.abort();
    if tty {
        print!("\r\x1b[2K");
        let _ = std::io::stdout().flush();
    }
    out
}

fn save(paths: &Paths, config: &Config) -> Result<()> {
    config.save(paths).context("saving config.toml")
}

/// A line saying whether the change is already live.
fn applied_note(paths: &Paths) {
    match ozgent_channels::gateway::held_elsewhere(paths) {
        Some(who) => println!("  Saved. {who} is running and picks this up within a few seconds."),
        None => println!("  Saved."),
    }
}

// ------------------------------------------------------------------ gateway

pub async fn gateway(paths: &Paths, config: Config, command: GatewayCommand) -> Result<()> {
    match command {
        GatewayCommand::Status => {
            status(paths, &config).await;
            Ok(())
        }
        GatewayCommand::Telegram { action } => channel(paths, config, Kind::Telegram, action).await,
        GatewayCommand::Whatsapp { action } => channel(paths, config, Kind::WhatsApp, action).await,
    }
}

async fn channel(paths: &Paths, mut config: Config, kind: Kind, action: Option<ChannelAction>) -> Result<()> {
    let Some(action) = action else {
        return if is_set_up(paths, &config, kind) {
            menu(paths, config, kind).await
        } else {
            match kind {
                Kind::Telegram => telegram_setup(paths, config).await,
                Kind::WhatsApp => whatsapp_setup(paths, config).await,
            }
        };
    };
    match action {
        ChannelAction::Setup => match kind {
            Kind::Telegram => telegram_setup(paths, config).await,
            Kind::WhatsApp => whatsapp_setup(paths, config).await,
        },
        ChannelAction::Allowed => {
            print_allowed(&config, kind);
            Ok(())
        }
        ChannelAction::Allow { who } => {
            let entries = entries(&who);
            if entries.is_empty() {
                anyhow::bail!("say who: {}", example(kind));
            }
            // All checked before any is added, so a typo in the third leaves
            // the first two unsaved rather than half the list applied.
            for w in &entries {
                channels::normalise_identity(kind, w).map_err(|e| anyhow::anyhow!(e))?;
            }
            for w in &entries {
                allow(&mut config, kind, w)?;
            }
            save(paths, &config)?;
            applied_note(paths);
            Ok(())
        }
        ChannelAction::Deny { who } => {
            for w in entries(&who) {
                let id = channels::normalise_identity(kind, &w).unwrap_or_else(|_| w.trim().to_string());
                if config.channels.revoke(kind, &id) {
                    println!("  {id} can no longer message it");
                } else {
                    println!("  {id} was not on the list");
                }
            }
            save(paths, &config)?;
            applied_note(paths);
            Ok(())
        }
        ChannelAction::Tools { tools } => {
            match tools {
                Some(spec) => {
                    let known = tool_names(paths, &config).await;
                    *config.channels.tools_mut(kind) = parse_tools(&spec, &known)?;
                }
                None => choose_tools(paths, &mut config, kind).await?,
            }
            save(paths, &config)?;
            println!("  tools: {}", describe_tools(config.channels.access(kind).tools));
            applied_note(paths);
            Ok(())
        }
        ChannelAction::Token { token } => {
            if kind != Kind::Telegram {
                anyhow::bail!("only Telegram has a bot token. WhatsApp is linked: ozgent gateway whatsapp link");
            }
            let token = match token {
                Some(t) => t,
                None => ask_secret("  new bot token (hidden): ")?,
            };
            let bot = check_token(&token).await?;
            config.channels.telegram.token = token.trim().to_string();
            config.channels.set_enabled(Kind::Telegram, true);
            save(paths, &config)?;
            println!("  ✓ now using {bot}");
            applied_note(paths);
            Ok(())
        }
        ChannelAction::Link => {
            if kind != Kind::WhatsApp {
                anyhow::bail!("Telegram is not linked; it uses a bot token: ozgent gateway telegram token");
            }
            if ozgent_channels::gateway::whatsapp_linked(paths) {
                println!("  This machine is linked already. Linking another account signs this one out first.");
                if !confirm("  Sign out and link a different account?", false)? {
                    return Ok(());
                }
                sign_out_whatsapp(paths, &config).await?;
            }
            let who = link_whatsapp(paths, &config).await?;
            config.channels.set_enabled(Kind::WhatsApp, true);
            save(paths, &config)?;
            println!("  ✓ linked {who}");
            applied_note(paths);
            Ok(())
        }
        ChannelAction::Signout => sign_out(paths, config, kind).await,
        ChannelAction::On | ChannelAction::Off => {
            let on = matches!(action, ChannelAction::On);
            if on && !is_set_up(paths, &config, kind) {
                anyhow::bail!("{kind} is not set up yet. Run: ozgent gateway {kind}");
            }
            config.channels.set_enabled(kind, on);
            save(paths, &config)?;
            println!("  {kind} is {}", if on { "on" } else { "off" });
            applied_note(paths);
            Ok(())
        }
    }
}

/// Entries from the command line: one per argument, or several in one
/// separated by commas. A phone number with spaces is one quoted argument.
fn entries(args: &[String]) -> Vec<String> {
    args.iter()
        .flat_map(|a| a.split(','))
        .map(|w| w.trim().to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

fn example(kind: Kind) -> &'static str {
    match kind {
        Kind::Telegram => "a user id like 4242, or a @username",
        Kind::WhatsApp => "a phone number with its country code, like +91 98765 43210",
    }
}

fn is_set_up(paths: &Paths, config: &Config, kind: Kind) -> bool {
    match kind {
        Kind::Telegram => {
            !config.channels.telegram.token.trim().is_empty()
                || std::env::var("OZGENT_TELEGRAM_TOKEN").is_ok_and(|t| !t.trim().is_empty())
        }
        Kind::WhatsApp => ozgent_channels::gateway::whatsapp_linked(paths),
    }
}

/// Add one entry to a channel's list, checked and normalised.
fn allow(config: &mut Config, kind: Kind, input: &str) -> Result<()> {
    if input.trim().is_empty() {
        return Ok(());
    }
    let id = channels::normalise_identity(kind, input).map_err(|e| anyhow::anyhow!(e))?;
    if id == "*" {
        println!("  WARNING: \"*\" lets anyone who finds it use this machine's tools.");
        if !confirm("  Really let everyone in?", false)? {
            return Ok(());
        }
    }
    if config.channels.admit(kind, &id) {
        println!("  ✓ {id} can message it");
    } else {
        println!("  {id} was already allowed");
    }
    Ok(())
}

fn print_allowed(config: &Config, kind: Kind) {
    let list = config.channels.access(kind).allow;
    if kind == Kind::WhatsApp && config.channels.whatsapp.self_chat {
        println!("  you, in your own \"Message yourself\" chat");
    }
    if list.is_empty() {
        if !(kind == Kind::WhatsApp && config.channels.whatsapp.self_chat) {
            println!("  nobody yet — {}", example(kind));
        }
        return;
    }
    for id in list {
        if id == "*" {
            println!("  * — EVERYONE");
        } else {
            println!("  {id}");
        }
    }
}

async fn check_token(token: &str) -> Result<String> {
    let token = token.trim();
    if !ozgent_web::admin::looks_like_token(token) {
        anyhow::bail!("that does not look like a bot token. It is two parts with a colon, like 123456789:AA…");
    }
    let api = ozgent_channels::telegram::Telegram::new(token.to_string());
    spinning("checking with Telegram…", api.identify())
        .await
        .map_err(|e| anyhow::anyhow!("Telegram refused the token: {e}"))
}

// ------------------------------------------------------------------ tools

/// Every tool this machine has, with what it does. Starts the tool worker for
/// a moment, since that is the only thing that knows.
async fn tool_list(paths: &Paths, config: &Config) -> Vec<(String, String)> {
    let started = spinning("looking up the tools…", crate::start_tools(paths, config)).await;
    let Ok(host) = started else { return Vec::new() };
    let mut out: Vec<(String, String)> = host
        .tools()
        .iter()
        .filter(|t| !config.tools.disabled.contains(&t.name))
        .map(|t| (t.name.clone(), config.permissions.rule_for(&t.name, t.effect).to_string()))
        .collect();
    out.sort();
    host.shutdown().await;
    out
}

async fn tool_names(paths: &Paths, config: &Config) -> Vec<String> {
    tool_list(paths, config).await.into_iter().map(|(n, _)| n).collect()
}

/// `all`, `none`, or names separated by commas.
fn parse_tools(spec: &str, known: &[String]) -> Result<Option<Vec<String>>> {
    match spec.trim().to_ascii_lowercase().as_str() {
        "all" | "*" => return Ok(None),
        "none" | "" => return Ok(Some(Vec::new())),
        _ => {}
    }
    let mut out = Vec::new();
    for name in spec.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        if !known.is_empty() && !known.iter().any(|k| k == name) {
            anyhow::bail!("no tool called {name:?}. This machine has: {}", known.join(", "));
        }
        if !out.iter().any(|o| o == name) {
            out.push(name.to_string());
        }
    }
    Ok(Some(out))
}

fn describe_tools(tools: Option<&[String]>) -> String {
    match tools {
        None => "all of them, each following your permission rules".into(),
        Some([]) => "none — it only chats".into(),
        Some(list) => list.join(", "),
    }
}

async fn choose_tools(paths: &Paths, config: &mut Config, kind: Kind) -> Result<()> {
    let pick = choose(
        &format!("Which tools may it use from {kind}?"),
        &[
            "all of them (each still follows your permission rules)",
            "only the ones I pick",
            "none — just chat",
        ],
        if config.channels.access(kind).tools.is_some() { 1 } else { 0 },
    )?;
    let choice = match pick {
        0 => None,
        2 => Some(Vec::new()),
        _ => {
            let tools = tool_list(paths, config).await;
            if tools.is_empty() {
                println!("  No tools are available right now (is Python installed?). Leaving it at none.");
                Some(Vec::new())
            } else {
                let current = config.channels.access(kind).tools.map(<[String]>::to_vec).unwrap_or_default();
                for (i, (name, rule)) in tools.iter().enumerate() {
                    let mark = if current.contains(name) { "●" } else { " " };
                    println!("  {mark} {:>2}  {name:<16} {rule}", i + 1);
                }
                loop {
                    let a = ask("  numbers, separated by commas: ")?;
                    let picked: Result<Vec<String>, _> = a
                        .split([',', ' '])
                        .filter(|s| !s.trim().is_empty())
                        .map(|s| {
                            s.trim()
                                .parse::<usize>()
                                .ok()
                                .and_then(|n| tools.get(n.wrapping_sub(1)))
                                .map(|(name, _)| name.clone())
                                .ok_or(s.trim().to_string())
                        })
                        .collect();
                    match picked {
                        Ok(list) => break Some(list),
                        Err(bad) => println!("  {bad:?} is not one of the numbers"),
                    }
                }
            }
        }
    };
    *config.channels.tools_mut(kind) = choice;
    Ok(())
}

// ------------------------------------------------------------------ telegram

async fn telegram_setup(paths: &Paths, mut config: Config) -> Result<()> {
    println!("Telegram");
    println!();
    println!("  1. In Telegram, open @BotFather and send /newbot. Answer its two questions.");
    println!("  2. It replies with a token like 123456789:AAH… — paste it below.");
    println!();
    let (token, bot) = loop {
        let token = ask_secret("  bot token (hidden): ")?;
        if token.trim().is_empty() {
            anyhow::bail!("no token given; nothing was changed");
        }
        match check_token(&token).await {
            Ok(bot) => break (token.trim().to_string(), bot),
            Err(e) => println!("  ✗ {e}"),
        }
    };
    println!("  ✓ this is {bot}");
    config.channels.telegram.token = token.clone();
    println!();

    let who = choose(
        &format!("Who should be able to message {bot}?"),
        &["only me", "me and other people", "only other people"],
        0,
    )?;
    if who != 2 {
        match find_owner(paths, &token, &bot).await? {
            Some(id) => {
                config.channels.admit(Kind::Telegram, &id);
            }
            None => println!("  Skipped. Add yourself later: ozgent gateway telegram allow <id>"),
        }
    }
    if who != 0 {
        println!();
        println!("  Their Telegram user ids or @usernames, separated by commas.");
        println!("  (Or, once it is running, they can send /pair with the code ozgent shows you.)");
        loop {
            let a = ask("  > ")?;
            let mut ok = true;
            for w in a.split(',') {
                if let Err(e) = allow(&mut config, Kind::Telegram, w) {
                    println!("  ✗ {e}");
                    ok = false;
                }
            }
            if ok {
                break;
            }
        }
    }

    println!();
    choose_tools(paths, &mut config, Kind::Telegram).await?;
    if config.channels.telegram.tools.as_ref().is_none_or(|t| !t.is_empty()) {
        println!();
        let approve = confirm(
            "When a tool asks first (writing files, running commands), may it be approved from Telegram?",
            true,
        )?;
        config.channels.set_approve(Kind::Telegram, approve);
    }

    config.channels.set_enabled(Kind::Telegram, true);
    save(paths, &config)?;
    finish(paths, "Telegram").await
}

/// Learn the owner's user id by having them send a code to the bot.
async fn find_owner(paths: &Paths, token: &str, bot: &str) -> Result<Option<String>> {
    let code = format!("OZ{}", &secret::random_token(3).to_ascii_uppercase());
    println!();
    println!("  Now, from your own Telegram, send this to {bot}:");
    println!();
    println!("      {code}");
    println!();
    println!("  (or type your numeric user id here and press Enter; @userinfobot tells you it)");

    let api = ozgent_channels::telegram::Telegram::new(token.to_string());
    let mut typed = tokio::task::spawn_blocking(read_line);
    let waiting = api.wait_for_code(&code, Duration::from_secs(300));
    tokio::pin!(waiting);

    let found = tokio::select! {
        line = &mut typed => {
            let line = line.ok().flatten().unwrap_or_default();
            if line.trim().is_empty() {
                return Ok(None);
            }
            return channels::normalise_identity(Kind::Telegram, &line)
                .map(Some)
                .map_err(|e| anyhow::anyhow!(e));
        }
        found = &mut waiting => found,
    };
    let _ = paths;
    match found {
        Ok(Some(who)) => {
            let shown = match &who.username {
                Some(u) => format!("{} ({u}, id {})", who.name, who.id),
                None => format!("{} (id {})", who.name, who.id),
            };
            println!("  ✓ found you: {shown}");
            let _ = api.say(&who.chat, "✓ ozgent: you're set up. I'll answer here once the gateway is running.").await;
            print!("  press Enter to continue ");
            let _ = std::io::stdout().flush();
            let _ = typed.await;
            Ok(Some(who.id))
        }
        Ok(None) => {
            println!("  Nothing arrived in five minutes.");
            print!("  press Enter to continue ");
            let _ = std::io::stdout().flush();
            let _ = typed.await;
            Ok(None)
        }
        Err(e) => {
            println!("  ✗ {e}");
            println!("  Type your numeric user id instead (or press Enter to skip):");
            let line = typed.await.ok().flatten().unwrap_or_default();
            if line.trim().is_empty() {
                return Ok(None);
            }
            channels::normalise_identity(Kind::Telegram, &line).map(Some).map_err(|e| anyhow::anyhow!(e))
        }
    }
}

// ------------------------------------------------------------------ whatsapp

fn bridge(paths: &Paths, config: &Config) -> Result<std::path::PathBuf> {
    ozgent_channels::whatsapp::locate(config.channels.whatsapp.bridge.as_deref(), paths)
}

/// Install the bridge if it is not, then show the QR code until it is scanned.
async fn link_whatsapp(paths: &Paths, config: &Config) -> Result<String> {
    if let Some(who) = ozgent_channels::gateway::held_elsewhere(paths) {
        if ozgent_channels::gateway::whatsapp_linked(paths) {
            anyhow::bail!(
                "{who} is answering WhatsApp with this machine's session. Stop it first, or link from its /admin page."
            );
        }
    }
    let wa = &config.channels.whatsapp;
    let bridge = bridge(paths, config)?;
    if !ozgent_channels::whatsapp::is_installed(&bridge) {
        spinning(
            "installing the WhatsApp bridge (once; about a minute)…",
            ozgent_channels::whatsapp::install_quietly(&wa.node, &bridge),
        )
        .await?;
        println!("  ✓ bridge installed");
    }
    let state = paths.channel_dir("whatsapp").join("auth");
    ozgent_channels::whatsapp::login(&wa.node, &bridge, &state).await
}

async fn sign_out_whatsapp(paths: &Paths, config: &Config) -> Result<()> {
    if let Some(who) = ozgent_channels::gateway::held_elsewhere(paths) {
        anyhow::bail!("{who} is using the WhatsApp session. Sign out from its /admin page, or stop it first.");
    }
    let state = paths.channel_dir("whatsapp").join("auth");
    let result = match bridge(paths, config) {
        Ok(b) => {
            spinning(
                "signing out on WhatsApp…",
                ozgent_channels::whatsapp::unlink(&config.channels.whatsapp.node, &b, &state),
            )
            .await
        }
        Err(_) => ozgent_channels::whatsapp::logout(&state),
    };
    match result {
        Ok(()) => println!("  ✓ signed out; the device is gone from your phone's Linked devices"),
        Err(e) => println!("  {e}"),
    }
    Ok(())
}

async fn whatsapp_setup(paths: &Paths, mut config: Config) -> Result<()> {
    println!("WhatsApp");
    println!();
    println!("  This links your own WhatsApp account to this machine as a second device,");
    println!("  the way WhatsApp Web does. Automating a personal account is against");
    println!("  WhatsApp's terms, and accounts have been banned for it — use a number you");
    println!("  can afford to lose.");
    println!();
    if !confirm("  Link WhatsApp?", true)? {
        return Ok(());
    }
    if ozgent_channels::gateway::whatsapp_linked(paths) {
        sign_out_whatsapp(paths, &config).await?;
    }
    let who = link_whatsapp(paths, &config).await?;
    let own = channels::normalise_identity(Kind::WhatsApp, &who).ok();
    println!("  ✓ linked {who}");
    println!();

    println!("  Which phone numbers may message it? With the country code, separated by commas.");
    println!("  Your own number ({who}) means your \"Message yourself\" chat.");
    loop {
        let a = ask("  > ")?;
        if a.trim().is_empty() {
            println!("  Nobody yet. Add numbers later: ozgent gateway whatsapp allow <number>");
            break;
        }
        let mut ok = true;
        for w in a.split(',').filter(|w| !w.trim().is_empty()) {
            match channels::normalise_identity(Kind::WhatsApp, w) {
                Ok(id) if Some(&id) == own.as_ref() => {
                    // Messages to yourself arrive as "from you"; the allowlist
                    // cannot express that, the self-chat setting does.
                    config.channels.whatsapp.self_chat = true;
                    println!("  ✓ your own \"Message yourself\" chat will be answered");
                }
                Ok(id) => {
                    if let Err(e) = allow(&mut config, Kind::WhatsApp, &id) {
                        println!("  ✗ {e}");
                        ok = false;
                    }
                }
                Err(e) => {
                    println!("  ✗ {e}");
                    ok = false;
                }
            }
        }
        if ok {
            break;
        }
    }
    if !config.channels.whatsapp.allow.is_empty() {
        println!("  (ozgent replies to them as you, from your number.)");
    }

    println!();
    choose_tools(paths, &mut config, Kind::WhatsApp).await?;
    if config.channels.whatsapp.tools.as_ref().is_none_or(|t| !t.is_empty()) {
        println!();
        let approve = confirm(
            "When a tool asks first (writing files, running commands), may it be approved from WhatsApp?",
            true,
        )?;
        config.channels.set_approve(Kind::WhatsApp, approve);
    }

    config.channels.set_enabled(Kind::WhatsApp, true);
    save(paths, &config)?;
    finish(paths, "WhatsApp").await
}

/// The end of a setup: say it is saved, and how it starts answering.
async fn finish(paths: &Paths, name: &str) -> Result<()> {
    println!();
    match ozgent_channels::gateway::held_elsewhere(paths) {
        Some(who) => {
            println!("✓ {name} is set up. {who} is running and starts answering within a few seconds.");
        }
        None => {
            println!("✓ {name} is set up.");
            println!();
            println!("  It answers while `ozgent gateway` or `ozgent web` is running.");
            println!("  Change anything later with `ozgent gateway {}`, or on /admin.", name.to_ascii_lowercase());
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ menus

async fn sign_out(paths: &Paths, mut config: Config, kind: Kind) -> Result<()> {
    match kind {
        Kind::Telegram => {
            config.channels.telegram.token.clear();
            config.channels.telegram.enabled = false;
            save(paths, &config)?;
            println!("  Token forgotten and Telegram switched off.");
            println!("  It still works on Telegram's side; revoke it with @BotFather (/revoke) to kill it.");
            if std::env::var("OZGENT_TELEGRAM_TOKEN").is_ok() {
                println!("  $OZGENT_TELEGRAM_TOKEN is also set; unset it too.");
            }
        }
        Kind::WhatsApp => {
            sign_out_whatsapp(paths, &config).await?;
            config.channels.whatsapp.enabled = false;
            save(paths, &config)?;
        }
    }
    applied_note(paths);
    Ok(())
}

async fn menu(paths: &Paths, mut config: Config, kind: Kind) -> Result<()> {
    let account = match kind {
        Kind::Telegram => {
            let token = std::env::var("OZGENT_TELEGRAM_TOKEN")
                .ok()
                .filter(|t| !t.trim().is_empty())
                .unwrap_or_else(|| config.channels.telegram.token.clone());
            let api = ozgent_channels::telegram::Telegram::new(token);
            tokio::time::timeout(Duration::from_secs(8), api.identify())
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or_else(|| "the bot".into())
        }
        Kind::WhatsApp => "your account".into(),
    };
    loop {
        let on = config.channels.enabled && config.channels.enabled(kind);
        let n = config.channels.access(kind).allow.len()
            + usize::from(kind == Kind::WhatsApp && config.channels.whatsapp.self_chat);
        println!();
        println!("{kind} · {account} · {} · {n} allowed", if on { "on" } else { "off" });
        let tools = describe_tools(config.channels.access(kind).tools);
        let approve = if config.channels.access(kind).approve { "yes" } else { "no" };
        let mut options: Vec<String> = vec![
            "who may message it".into(),
            "allow someone".into(),
            "remove someone".into(),
            format!("tools                 ({tools})"),
            format!("approve tool calls    ({approve})"),
        ];
        match kind {
            Kind::Telegram => options.push("change the bot token".into()),
            Kind::WhatsApp => {
                options.push(format!(
                    "your own chat         ({})",
                    if config.channels.whatsapp.self_chat { "answered" } else { "not answered" }
                ));
                options.push(format!(
                    "group chats           ({})",
                    if config.channels.whatsapp.groups { "answered" } else { "ignored" }
                ));
                options.push("link a different account".into());
            }
        }
        options.push(if on { "turn it off".into() } else { "turn it on".into() });
        options.push("sign out".into());
        options.push("done".into());
        let labels: Vec<&str> = options.iter().map(String::as_str).collect();
        let pick = choose("", &labels, labels.len() - 1)?;
        let label = labels[pick];

        match label {
            "done" => return Ok(()),
            "who may message it" => print_allowed(&config, kind),
            "allow someone" => {
                println!("  {}; several separated by commas", example(kind));
                let a = ask("  > ")?;
                for w in a.split(',') {
                    if let Err(e) = allow(&mut config, kind, w) {
                        println!("  ✗ {e}");
                    }
                }
                save(paths, &config)?;
                applied_note(paths);
            }
            "remove someone" => {
                let list = config.channels.access(kind).allow.to_vec();
                if list.is_empty() {
                    println!("  nobody is on the list");
                    continue;
                }
                let mut opts: Vec<&str> = list.iter().map(String::as_str).collect();
                opts.push("never mind");
                let i = choose("  Remove whom?", &opts, opts.len() - 1)?;
                if i < list.len() {
                    config.channels.revoke(kind, &list[i]);
                    save(paths, &config)?;
                    println!("  {} can no longer message it", list[i]);
                    applied_note(paths);
                }
            }
            l if l.starts_with("tools") => {
                choose_tools(paths, &mut config, kind).await?;
                save(paths, &config)?;
                applied_note(paths);
            }
            l if l.starts_with("approve") => {
                let now = confirm("  May a tool that asks first be approved from the chat?", config.channels.access(kind).approve)?;
                config.channels.set_approve(kind, now);
                save(paths, &config)?;
                applied_note(paths);
            }
            "change the bot token" => {
                let token = ask_secret("  new bot token (hidden): ")?;
                match check_token(&token).await {
                    Ok(bot) => {
                        config.channels.telegram.token = token.trim().to_string();
                        save(paths, &config)?;
                        println!("  ✓ now using {bot}");
                        applied_note(paths);
                    }
                    Err(e) => println!("  ✗ {e}"),
                }
            }
            l if l.starts_with("your own chat") => {
                config.channels.whatsapp.self_chat = !config.channels.whatsapp.self_chat;
                save(paths, &config)?;
                applied_note(paths);
            }
            l if l.starts_with("group chats") => {
                if !config.channels.whatsapp.groups {
                    println!("  In a group, anyone allowed can steer ozgent where everyone can see it.");
                    if !confirm("  Answer in group chats?", false)? {
                        continue;
                    }
                }
                config.channels.whatsapp.groups = !config.channels.whatsapp.groups;
                save(paths, &config)?;
                applied_note(paths);
            }
            "link a different account" => {
                if !confirm("  This signs the current account out first. Go on?", false)? {
                    continue;
                }
                sign_out_whatsapp(paths, &config).await?;
                match link_whatsapp(paths, &config).await {
                    Ok(who) => {
                        println!("  ✓ linked {who}");
                        config.channels.set_enabled(Kind::WhatsApp, true);
                        save(paths, &config)?;
                        applied_note(paths);
                    }
                    Err(e) => println!("  ✗ {e}"),
                }
            }
            "turn it off" | "turn it on" => {
                config.channels.set_enabled(kind, label == "turn it on");
                save(paths, &config)?;
                applied_note(paths);
            }
            "sign out" => {
                if confirm(&format!("  Sign {kind} out?"), false)? {
                    return sign_out(paths, config, kind).await;
                }
            }
            _ => {}
        }
    }
}

// ------------------------------------------------------------------ status

/// The chats bound to a channel. Opened read-only and closed again: the
/// gateway may be running against the same file, and WAL lets a reader in.
fn bound_chats(paths: &Paths, channel: &str) -> Result<Vec<ozgent_memory::ChannelChat>> {
    let store = ozgent_memory::Store::open(paths.root().join("ozgent.db"))?;
    Ok(store.channel_chats(channel)?)
}

async fn status(paths: &Paths, config: &Config) {
    match ozgent_channels::gateway::held_elsewhere(paths) {
        Some(who) => println!("gateway   running — {who}"),
        None => println!("gateway   not running (start it with `ozgent gateway` or `ozgent web`)"),
    }
    if !config.channels.enabled {
        println!("          channels are switched off; setting one up switches them on");
    }
    println!();

    for kind in [Kind::Telegram, Kind::WhatsApp] {
        let access = config.channels.access(kind);
        println!("  {kind}");
        let set_up = is_set_up(paths, config, kind);
        println!(
            "    state     {}",
            match (set_up, config.channels.enabled(kind)) {
                (false, _) => "not set up — run `ozgent gateway ".to_string() + kind.as_str() + "`",
                (true, true) => "on".to_string(),
                (true, false) => "off".to_string(),
            }
        );
        match kind {
            Kind::Telegram => {
                // Never the token itself: it is a password, and this output
                // gets pasted into issues.
                let from_env = std::env::var("OZGENT_TELEGRAM_TOKEN").is_ok_and(|t| !t.trim().is_empty());
                let has = !config.channels.telegram.token.trim().is_empty() || from_env;
                println!(
                    "    token     {}",
                    if from_env { "from $OZGENT_TELEGRAM_TOKEN" } else if has { "set" } else { "missing" }
                );
            }
            Kind::WhatsApp => {
                println!("    linked    {}", if set_up { "yes" } else { "no" });
                println!(
                    "    own chat  {}",
                    if config.channels.whatsapp.self_chat { "answered" } else { "not answered" }
                );
                if config.channels.whatsapp.groups {
                    println!("    groups    answered");
                }
            }
        }
        if is_open_to_everyone(access.allow) {
            println!("    allowed   EVERYONE — anyone who finds it can use this machine's tools");
        } else if access.allow.is_empty() {
            println!("    allowed   nobody{}", if kind == Kind::WhatsApp && config.channels.whatsapp.self_chat { " else" } else { "" });
        } else {
            println!("    allowed   {}", access.allow.join(", "));
        }
        println!("    tools     {}", describe_tools(access.tools));
        println!("    approve   {}", if access.approve { "yes, from the chat" } else { "no — anything that asks is refused" });

        if let Ok(chats) = bound_chats(paths, kind.as_str()) {
            if !chats.is_empty() {
                println!("    chats     {}", chats.len());
                for chat in chats.iter().take(5) {
                    let who = if chat.display.is_empty() { &chat.chat_id } else { &chat.display };
                    println!("              {who}");
                }
            }
        }
        println!();
    }

    let model = config
        .channels
        .model
        .clone()
        .or_else(|| config.default_model.clone())
        .unwrap_or_else(|| "none set".into());
    println!("  answering with  {model}");
}

// ------------------------------------------------------------------ admin

pub fn admin(paths: &Paths, mut config: Config, command: AdminCommand) -> Result<()> {
    match command {
        AdminCommand::Status => {
            match config.web.admin_hash() {
                None => {
                    println!("admin     no password — /admin is closed");
                    println!("          open it with: ozgent admin setup");
                }
                Some(h) if !secret::is_hash(h) => {
                    println!("admin     [web] admin_password_hash is not a password hash, so /admin is closed");
                    println!("          fix it with: ozgent admin reset");
                }
                Some(_) => {
                    println!("admin     password set (stored as an Argon2id hash)");
                    println!("          http://localhost:7333/admin while `ozgent web` runs");
                    println!("          forgot it? ozgent admin reset");
                }
            }
            Ok(())
        }
        AdminCommand::Setup | AdminCommand::Reset => {
            let resetting = matches!(command, AdminCommand::Reset);
            let exists = config.web.admin_hash().is_some_and(secret::is_hash);
            if exists && !resetting {
                println!("A password is set already.");
                if !confirm("Replace it?", false)? {
                    return Ok(());
                }
            }
            println!("The /admin page controls the Telegram and WhatsApp gateway and model downloads.");
            println!("At least {} characters. Only a hash is stored.", secret::MIN_PASSWORD);
            let password = loop {
                let first = ask_secret("new password: ")?;
                if let Err(e) = secret::check_strength(&first) {
                    println!("  ✗ {e}");
                    continue;
                }
                let again = ask_secret("again: ")?;
                if first != again {
                    println!("  ✗ those were different; once more");
                    continue;
                }
                break first;
            };
            let hash = secret::hash_password(&password).map_err(|e| anyhow::anyhow!(e))?;
            config.web.admin_password_hash = Some(hash);
            save(paths, &config)?;
            println!();
            println!("✓ Saved.");
            if exists || resetting {
                println!("  Every browser signed in to /admin is signed out, and any lockout is lifted.");
            }
            println!("  Open http://localhost:7333/admin while `ozgent web` is running.");
            Ok(())
        }
        AdminCommand::Disable => {
            if config.web.admin_password_hash.take().is_none() {
                println!("No password was set; /admin is already closed.");
                return Ok(());
            }
            save(paths, &config)?;
            println!("✓ Password removed. /admin is closed until `ozgent admin setup`.");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_are_parsed_from_the_short_forms() {
        let known = vec!["web_search".to_string(), "fetch_url".to_string()];
        assert_eq!(parse_tools("all", &known).unwrap(), None);
        assert_eq!(parse_tools("none", &known).unwrap(), Some(vec![]));
        assert_eq!(
            parse_tools("web_search, fetch_url, web_search", &known).unwrap(),
            Some(vec!["web_search".to_string(), "fetch_url".to_string()])
        );
        assert!(parse_tools("rm_rf", &known).is_err());
    }

    #[test]
    fn entries_are_split_by_argument_and_by_comma() {
        let args = vec!["@ada_l".to_string(), "4242, 99".to_string(), " ".to_string()];
        assert_eq!(entries(&args), vec!["@ada_l", "4242", "99"]);
        assert_eq!(entries(&["+91 98765 43210".to_string()]), vec!["+91 98765 43210"]);
    }

    #[test]
    fn a_tool_list_reads_as_a_sentence() {
        assert!(describe_tools(None).starts_with("all"));
        assert!(describe_tools(Some(&[])).starts_with("none"));
        assert_eq!(describe_tools(Some(&["a".to_string(), "b".to_string()])), "a, b");
    }
}
