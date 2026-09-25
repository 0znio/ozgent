//! `ozgent mcp …`: see the servers, and add, install, switch and remove them.
//!
//! Changes are written to `config.toml`; a running daemon notices the file
//! changed and reconnects within seconds, so nothing here talks to it. The
//! same rules as the admin page apply: a server added here runs in ozgent's
//! sandbox unless `--no-sandbox` says otherwise, and an install from the
//! registry shows the exact command and asks before it is saved.

use std::collections::BTreeMap;
use std::io::IsTerminal;

use anyhow::{Context, Result, bail};
use ozgent_core::mcp::{self, Server};
use ozgent_core::{Config, Paths};
use ozgent_mcp::registry;

use crate::cli::{McpCommand, McpPlacement};
use crate::setup::{ask, ask_secret, confirm};

pub(crate) async fn run(paths: &Paths, config: &Config, command: Option<McpCommand>) -> Result<()> {
    match command.unwrap_or(McpCommand::Status) {
        McpCommand::Status => status(paths, config).await,
        McpCommand::Search { query } => search(&query.join(" ")).await,
        McpCommand::Install { registry_name, version, name, option, set, placement, yes } => {
            install(paths, &registry_name, version.as_deref(), name, option, &set, &placement, yes).await
        }
        McpCommand::Add { name, npm, pypi, url, version, env, header, placement, command } => {
            add(paths, &name, npm, pypi, url, version, &env, &header, &placement, command)
        }
        McpCommand::Import { file, name, placement, yes } => import(paths, file, name, &placement, yes),
        McpCommand::Remove { name } => remove(paths, &name),
        McpCommand::Enable { name } => switch_one(paths, &name, true),
        McpCommand::Disable { name } => switch_one(paths, &name, false),
        McpCommand::On => switch_all(paths, true),
        McpCommand::Off => switch_all(paths, false),
    }
}

const PICKED_UP: &str = "A running ozgent picks this up within a few seconds.";

/// Change `config.toml`, reading it fresh so nothing else's change is lost.
fn change(paths: &Paths, f: impl FnOnce(&mut Config) -> Result<()>) -> Result<()> {
    let mut config = Config::load(paths)?;
    f(&mut config)?;
    config.save(paths)?;
    Ok(())
}

fn pairs(what: &str, raw: &[String]) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for item in raw {
        let (k, v) = item.split_once('=').with_context(|| format!("{what} {item:?} needs a NAME=VALUE form"))?;
        let k = k.trim();
        if k.is_empty() || !k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            bail!("{k:?} is not a valid {what} name");
        }
        out.insert(k.to_string(), v.to_string());
    }
    Ok(out)
}

fn place(server: &mut Server, p: &McpPlacement) {
    let stdio = server.url.is_none();
    // A container is Docker's to isolate; the sandbox would only stand
    // between the Docker client and its daemon.
    let container = server.command.as_deref() == Some("docker");
    server.sandbox = stdio && !container && !p.no_sandbox;
    server.network = !p.no_network;
    server.folders = p.folders.iter().map(|f| std::path::absolute(f).unwrap_or_else(|_| f.clone())).collect();
    if server.sandbox {
        for f in registry::folders_in_args(&server.args) {
            if !server.folders.contains(&f) {
                server.folders.push(f);
            }
        }
    }
    server.trust_hints = p.trust_hints;
}

fn save_new(paths: &Paths, name: &str, server: Server) -> Result<()> {
    mcp::valid_name(name).map_err(anyhow::Error::msg)?;
    server.transport().map_err(|e| anyhow::anyhow!("{e}"))?;
    change(paths, |c| {
        if c.mcp.servers.contains_key(name) {
            bail!("there is already a server called {name}; `ozgent mcp remove {name}` first");
        }
        c.mcp.servers.insert(name.to_string(), server);
        c.mcp.enabled = true;
        Ok(())
    })?;
    println!("added {name}. {PICKED_UP}");
    println!("`ozgent mcp` connects to it and lists its tools.");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add(
    paths: &Paths,
    name: &str,
    npm: Option<String>,
    pypi: Option<String>,
    url: Option<String>,
    version: Option<String>,
    env: &[String],
    header: &[String],
    placement: &McpPlacement,
    command: Vec<String>,
) -> Result<()> {
    let mut server = if let Some(package) = npm {
        registry::plan_package("npm", &package, version.as_deref(), &command).map_err(anyhow::Error::msg)?
    } else if let Some(package) = pypi {
        registry::plan_package("pypi", &package, version.as_deref(), &command).map_err(anyhow::Error::msg)?
    } else if let Some(url) = url {
        if !command.is_empty() {
            bail!("a --url server runs nothing here, so it takes no command");
        }
        Server { url: Some(url), headers: pairs("header", header)?, ..Default::default() }
    } else {
        let (program, args) = command.split_first().context("give the command after --, or --npm, --pypi or --url")?;
        Server { command: Some(program.clone()), args: args.to_vec(), source: Some("added by hand".into()), ..Default::default() }
    };
    if !header.is_empty() && server.url.is_none() {
        bail!("--header is for a --url server");
    }
    server.env = pairs("variable", env)?;
    place(&mut server, placement);
    println!("will run: {}{}", registry::preview(&server), if server.sandbox { "  (in the sandbox)" } else { "" });
    if server.sandbox && !server.folders.is_empty() {
        let list: Vec<String> = server.folders.iter().map(|f| f.display().to_string()).collect();
        println!("may use:  {}", list.join(", "));
    }
    save_new(paths, name, server)
}

async fn search(query: &str) -> Result<()> {
    let (found, _) = registry::search(query, None).await.map_err(anyhow::Error::msg)?;
    if found.is_empty() {
        println!("nothing in the registry matches {query:?}");
        return Ok(());
    }
    for l in &found {
        let kinds: Vec<String> = l
            .options
            .iter()
            .map(|o| if o.supported && registry::runner_present(o) { o.kind.clone() } else { format!("({})", o.kind) })
            .collect();
        println!("{}  {}  [{}]", l.name, l.version, kinds.join(" "));
        let about = l.title.clone().filter(|t| !t.is_empty()).map(|t| format!("{t} — {}", l.description)).unwrap_or(l.description.clone());
        if !about.is_empty() {
            println!("  {}", about.chars().take(160).collect::<String>());
        }
    }
    println!();
    println!("Kinds in brackets cannot run here. Install with: ozgent mcp install <name>");
    println!("Anyone can publish to the registry: check a server's source before installing it.");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn install(
    paths: &Paths,
    registry_name: &str,
    version: Option<&str>,
    name: Option<String>,
    option: Option<usize>,
    set: &[String],
    placement: &McpPlacement,
    yes: bool,
) -> Result<()> {
    let listing = registry::fetch(registry_name, version.unwrap_or("latest")).await.map_err(anyhow::Error::msg)?;
    println!("{}  {}", listing.name, listing.version);
    if !listing.description.is_empty() {
        println!("  {}", listing.description);
    }
    match listing.repository.as_deref().or(listing.website.as_deref()) {
        Some(src) => println!("  source: {src}"),
        None => println!("  it lists no source repository"),
    }

    let usable: Vec<usize> = listing
        .options
        .iter()
        .enumerate()
        .filter(|(_, o)| o.supported && registry::runner_present(o))
        .map(|(i, _)| i)
        .collect();
    let choice = match option {
        Some(i) if usable.contains(&i) => i,
        Some(i) => bail!("option {i} cannot be used here"),
        None => *usable.first().with_context(|| {
            let why: Vec<String> = listing
                .options
                .iter()
                .map(|o| format!("{}: {}", o.kind, o.why_not.clone().unwrap_or_else(|| format!("needs {}", o.runner.clone().unwrap_or_default()))))
                .collect();
            format!("none of its ways to run can be used here ({})", why.join("; "))
        })?,
    };
    let picked = &listing.options[choice];

    let mut values = BTreeMap::new();
    for item in set {
        let (k, v) = item.split_once('=').with_context(|| format!("--set {item:?} needs KEY=VALUE"))?;
        values.insert(k.trim().to_string(), v.to_string());
    }
    let interactive = std::io::stdin().is_terminal();
    for input in &picked.inputs {
        if values.contains_key(&input.key) {
            continue;
        }
        if !input.required && (input.default.is_some() || !interactive) {
            continue;
        }
        if !interactive {
            bail!("{} is required: --set {}=…", input.name, input.key);
        }
        let about = if input.description.is_empty() { String::new() } else { format!(" ({})", input.description) };
        let prompt = format!("{}{about}{}: ", input.name, if input.required { "" } else { " [optional]" });
        let value = if input.secret { ask_secret(&prompt)? } else { ask(&prompt)? };
        if !value.trim().is_empty() {
            values.insert(input.key.clone(), value);
        }
    }

    let mut server = registry::plan(&listing, choice, &values).map_err(anyhow::Error::msg)?;
    place(&mut server, placement);
    let name = name.unwrap_or_else(|| registry::suggested_name(&listing.name));
    println!();
    println!("name:      {name}  (its tools: {name}_…)");
    println!("will run:  {}", registry::preview(&server));
    if server.sandbox && !server.folders.is_empty() {
        let list: Vec<String> = server.folders.iter().map(|f| f.display().to_string()).collect();
        println!("may use:   {}", list.join(", "));
    }
    if server.url.is_none() {
        println!(
            "sandbox:   {}",
            if server.sandbox { if server.network { "yes, with network" } else { "yes, no network" } } else { "no" }
        );
    }
    if !yes && !confirm("Add it?", true)? {
        println!("not added");
        return Ok(());
    }
    save_new(paths, &name, server)
}

fn import(
    paths: &Paths,
    file: Option<std::path::PathBuf>,
    name: Option<String>,
    placement: &McpPlacement,
    yes: bool,
) -> Result<()> {
    let text = match &file {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
        None => {
            if std::io::stdin().is_terminal() {
                println!("Paste the MCP settings (JSON), then press Ctrl-D on an empty line:");
            }
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            buf
        }
    };
    let parsed = ozgent_mcp::import::parse(&text, name.as_deref()).map_err(anyhow::Error::msg)?;
    let existing: Vec<String> = Config::load(paths)?.mcp.servers.keys().cloned().collect();
    let mut ready = Vec::new();
    for entry in parsed {
        match entry.server {
            Ok(mut server) => {
                if existing.contains(&entry.name) {
                    println!("✗ {}: there is already a server called that; `ozgent mcp remove {}` first", entry.name, entry.name);
                    continue;
                }
                place(&mut server, placement);
                println!("{}:  {}{}", entry.name, registry::preview(&server), if server.sandbox { "  (in the sandbox)" } else { "" });
                if server.sandbox && !server.folders.is_empty() {
                    let list: Vec<String> = server.folders.iter().map(|f| f.display().to_string()).collect();
                    println!("    may use {}", list.join(", "));
                }
                for note in &entry.notes {
                    println!("    note: {note}");
                }
                ready.push((entry.name, server));
            }
            Err(e) => println!("✗ {}: {e}", entry.name),
        }
    }
    if ready.is_empty() {
        bail!("nothing to add");
    }
    // The paste used stdin, so asking needs the terminal itself.
    if !yes {
        if file.is_none() && !std::io::stdin().is_terminal() {
            bail!("add --yes to save without being asked, when the JSON comes from a pipe");
        }
        if !confirm(&format!("Add {}?", if ready.len() == 1 { "it".to_string() } else { format!("these {}", ready.len()) }), true)? {
            println!("not added");
            return Ok(());
        }
    }
    let count = ready.len();
    change(paths, |c| {
        for (name, server) in ready {
            c.mcp.servers.insert(name, server);
        }
        c.mcp.enabled = true;
        Ok(())
    })?;
    println!("added {count}. {PICKED_UP}");
    Ok(())
}

fn remove(paths: &Paths, name: &str) -> Result<()> {
    let config = Config::load(paths)?;
    let launcher = ozgent_mcp::Launcher::from_config(&config, paths);
    change(paths, |c| {
        if c.mcp.servers.remove(name).is_none() {
            bail!("there is no server called {name}");
        }
        // Its tools' rules go with it; left behind, a later server given the
        // same name would inherit an `allow` nobody gave it.
        let prefix = format!("{}_", mcp::tool_name(name, "x").strip_suffix("_x").unwrap_or(name));
        let others: Vec<String> = c
            .mcp
            .servers
            .keys()
            .map(|n| format!("{}_", mcp::tool_name(n, "x").strip_suffix("_x").unwrap_or(n)))
            .filter(|p| p.len() > prefix.len() && p.starts_with(&prefix))
            .collect();
        c.permissions.tools.retain(|tool, _| !tool.starts_with(&prefix) || others.iter().any(|o| tool.starts_with(o)));
        Ok(())
    })?;
    if let Some(home) = launcher.map(|l| l.home(name)).filter(|h| h.is_dir()) {
        let _ = std::fs::remove_dir_all(home);
    }
    println!("removed {name}. {PICKED_UP}");
    Ok(())
}

fn switch_one(paths: &Paths, name: &str, on: bool) -> Result<()> {
    change(paths, |c| {
        let server = c.mcp.servers.get_mut(name).with_context(|| format!("there is no server called {name}"))?;
        server.enabled = on;
        Ok(())
    })?;
    println!("{name} is {}. {PICKED_UP}", if on { "on" } else { "off" });
    Ok(())
}

fn switch_all(paths: &Paths, on: bool) -> Result<()> {
    change(paths, |c| {
        c.mcp.enabled = on;
        Ok(())
    })?;
    println!("MCP servers are {}. {PICKED_UP}", if on { "on" } else { "off" });
    Ok(())
}

/// Connect to every server, as a chat would, and show what it offers.
///
/// Connects for real rather than reading the file back: "it is in
/// config.toml" and "the model can call it" are different claims, and the
/// gap between them is exactly what this exists to show.
async fn status(paths: &Paths, config: &Config) -> Result<()> {
    if config.mcp.servers.is_empty() {
        println!("No MCP servers are configured.");
        println!();
        println!("  ozgent mcp search <words>                  find one in the MCP Registry");
        println!("  ozgent mcp install <registry name>         install it");
        println!("  ozgent mcp add files --npm @modelcontextprotocol/server-filesystem -- ~/notes");
        println!();
        println!("Or on the admin page, under MCP.");
        return Ok(());
    }
    if !config.mcp.enabled {
        println!("MCP servers are switched off: `ozgent mcp on` to use them.");
        println!();
    }
    println!("connecting…");
    println!();
    let launcher = ozgent_mcp::Launcher::from_config(config, paths);
    let (sources, statuses) = ozgent_mcp::connect_all_with(&config.mcp, launcher.as_ref()).await;

    for st in &statuses {
        let settings = &config.mcp.servers[&st.name];
        let state = match st.state {
            ozgent_mcp::State::Connected => format!("connected{}", st.server.as_deref().map(|s| format!(" · {s}")).unwrap_or_default()),
            ozgent_mcp::State::Failed => "failed".into(),
            ozgent_mcp::State::Disabled => "switched off".into(),
            ozgent_mcp::State::Off => "off".into(),
        };
        println!("{}  {state}", st.name);
        println!("  runs          {}", registry::preview(settings));
        if settings.url.is_none() {
            println!("  sandbox       {}", if settings.sandbox { "yes" } else { "no" });
        }
        println!(
            "  hints         {}",
            if settings.trust_hints { "trusted — read-only tools run unasked" } else { "not trusted — every tool asks" }
        );
        if let Some(e) = &st.error {
            println!("  error         {e}");
            for line in st.log.iter().rev().take(8).collect::<Vec<_>>().into_iter().rev() {
                println!("    | {line}");
            }
        }
        if !st.tools.is_empty() {
            println!("  tools         {} offered of {}", st.tools.iter().filter(|t| t.offered).count(), st.tools.len());
            for tool in &st.tools {
                let rule = config.permissions.rule_for(&tool.name, tool.effect);
                let offered = if tool.offered { "" } else { "  (not offered)" };
                println!("    {:<34} {:<8} {rule}{offered}", tool.name, tool.effect.to_string());
                if !tool.description.is_empty() {
                    println!("      {}", tool.description);
                }
            }
        }
        println!();
    }

    for source in &sources {
        source.shutdown().await;
    }
    Ok(())
}
