//! The `ozgent` binary.

mod chat;
mod input;
mod cli;
mod logging;
mod permission;
mod status;
mod tui;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{AgentCommand, ChannelCommand, Cli, Command, ConfigCommand, OptionFlags, ToolsCommand};
use ozgent_core::{Config, ModelRef, Paths};
use ozgent_tools::{HostConfig, ToolHost, Toolbox};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Paths first, because the log file lives under them. Anything that fails
    // before this point still reports itself through the error return.
    let paths = match &cli.root {
        Some(dir) => Paths::with_root(dir),
        None => Paths::discover().context("locating the ozgent directory")?,
    };
    logging::init(cli.verbose, Some(&paths.logs_dir()));
    let config = Config::load(&paths).context("loading config.toml")?;

    match cli.command {
        None => chat(&paths, &config, cli.model, &cli.options).await,
        Some(Command::Chat { model, options }) => chat(&paths, &config, model, &options).await,
        Some(Command::Run { model, prompt, options }) => {
            run_once(&paths, &config, &model, &prompt.join(" "), &options).await
        }
        Some(Command::Web { port, host, no_open, options }) => {
            // `--no-open` had no effect and nothing ever opened a browser,
            // though `--help` said one would.
            if !no_open {
                open_browser(port);
            }
            // The flags were parsed and thrown away: `ozgent web --ctx 32k`
            // printed nothing and changed nothing.
            ozgent_web::serve(paths, config, &host, port, options.to_options()?).await
        }
        Some(Command::Gateway { web, port, host, options }) => {
            gateway(paths, config, web, &host, port, options.to_options()?).await
        }
        Some(Command::Channel { command }) => channel(&paths, config, command).await,
        Some(Command::Agent { command }) => agent(&paths, command),
        Some(Command::Mcp) => mcp(&config).await,
        Some(Command::Serve { port, host, api_key, options }) => {
            // The environment is the right place for a secret; a flag lands in
            // shell history and in `ps`.
            let api_key = api_key.or_else(|| std::env::var("OZGENT_API_KEY").ok());
            let overrides = options.to_options()?;
            ozgent_web::serve_api(paths, config, &host, port, api_key, overrides).await
        }
        Some(Command::Pull { repo, quant, as_ref, name, revision, max_size, list }) => {
            pull(&paths, &repo, quant, as_ref, name, revision, max_size, list).await
        }
        Some(Command::Import { weights, as_ref, name, mmproj, copy }) => {
            import(&paths, weights, as_ref, name, mmproj, copy)
        }
        Some(Command::List) => list_models(&paths),
        Some(Command::Show { model }) => show_model(&paths, &config, &model),
        Some(Command::Alias { model, alias, clear }) => set_alias(&paths, &model, alias, clear),
        Some(Command::DefaultModel { model, clear }) => {
            default_model(&paths, model, clear)
        }
        Some(Command::Rm { model, force }) => remove_model(&paths, &model, force),
        Some(Command::Tools { command }) => tools(&paths, &config, command).await,
        Some(Command::Config { command }) => config_cmd(&paths, &config, command),
        Some(Command::Doctor) => doctor(&paths, &config).await,
        Some(Command::Logs { lines, follow, path }) => show_logs(&paths, lines, follow, path),
    }
}

/// Open the web interface once the server is listening.
///
/// Spawned rather than awaited: the browser is launched a moment later, from
/// another thread, because `serve` does not return until the server stops and
/// opening first would race the port being bound. `localhost` regardless of
/// the bind address — a browser on this machine reaches it either way, and
/// `0.0.0.0` is not a destination.
fn open_browser(port: u16) {
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(600));
        let url = format!("http://localhost:{port}");
        // Whichever of these exists; a machine with no browser is not an
        // error, the address was printed either way.
        for opener in ["xdg-open", "open", "wslview"] {
            if std::process::Command::new(opener)
                .arg(&url)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .is_ok()
            {
                return;
            }
        }
    });
}

/// Print the tail of the shared log, optionally following it.
///
/// Deliberately not a log viewer. It answers "what happened" without making
/// the user remember where the file lives, and `--path` hands it to whatever
/// they would rather use.
fn show_logs(paths: &Paths, lines: usize, follow: bool, path_only: bool) -> Result<()> {
    use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};

    let file_path = paths.logs_dir().join("ozgent.log");
    if path_only {
        println!("{}", file_path.display());
        return Ok(());
    }
    if !file_path.exists() {
        println!("No log yet at {}.", file_path.display());
        println!("It is written as soon as ozgent runs anything.");
        return Ok(());
    }

    let file = std::fs::File::open(&file_path)
        .with_context(|| format!("opening {}", file_path.display()))?;
    let mut reader = BufReader::new(file);

    // Keep only the last `lines`: a rolled log is up to 8 MiB and printing all
    // of it is never what was wanted.
    let mut tail: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut line = String::new();
    while reader.read_line(&mut line)? > 0 {
        if tail.len() == lines {
            tail.pop_front();
        }
        tail.push_back(std::mem::take(&mut line));
    }
    let mut out = std::io::stdout().lock();
    for l in &tail {
        out.write_all(l.as_bytes())?;
    }
    out.flush()?;

    if !follow {
        return Ok(());
    }
    // Resume from where the tail ended rather than re-reading the file.
    let mut at = reader.stream_position()?;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(400));
        let Ok(file) = std::fs::File::open(&file_path) else { continue };
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        // Rotation replaces the file under us and the new one is shorter.
        if len < at {
            at = 0;
        }
        if len == at {
            continue;
        }
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(at))?;
        let mut chunk = String::new();
        while reader.read_line(&mut chunk)? > 0 {
            out.write_all(chunk.as_bytes())?;
            chunk.clear();
        }
        out.flush()?;
        at = len;
    }
}


// --------------------------------------------------------------- models

fn list_models(paths: &Paths) -> Result<()> {
    let models = ozgent_core::installed(paths);
    if models.is_empty() {
        println!("No models installed.");
        println!("Try: ozgent pull <huggingface-repo> --name <short-name>");
        return Ok(());
    }

    let alias_w = ozgent_core::registry::alias_width(&models).max(5);
    println!("{:<alias_w$}  {:<30} {:<10} {:>10}", "ALIAS", "MODEL", "QUANT", "SIZE");
    for m in models {
        println!(
            "{:<alias_w$}  {:<30} {:<10} {:>10}",
            m.manifest.alias.as_deref().unwrap_or("-"),
            m.model.to_string(),
            m.manifest.quantization.as_deref().unwrap_or("-"),
            m.manifest.size_bytes.map(human_size).unwrap_or_else(|| "-".into()),
        );
    }
    Ok(())
}

/// Walk `models/<name>/<tag>` and read each manifest.
///
/// Names may contain a namespace, so the walk is depth-limited rather than
/// assuming exactly two levels.
fn installed_models(paths: &Paths) -> Result<Vec<(ModelRef, ozgent_core::Manifest)>> {
    let root = paths.models_dir();
    let mut out = Vec::new();
    let mut stack = vec![(root.clone(), 0usize)];

    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if path.join(ozgent_core::manifest::MANIFEST_FILE).is_file() {
                match ozgent_core::Manifest::load(&path) {
                    Ok(m) => out.push((m.model_ref(), m)),
                    Err(e) => tracing::warn!("skipping {}: {e}", path.display()),
                }
            } else if depth < 3 {
                stack.push((path, depth + 1));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn show_model(paths: &Paths, config: &Config, model: &str) -> Result<()> {
    let found = ozgent_core::resolve(paths, model)?;
    let (r, dir, manifest) = (found.model, found.dir, found.manifest);

    let resolved = config
        .options_for(&r.to_string())
        .merge(&manifest.defaults)
        .resolve();

    println!("{r}");
    if let Some(alias) = &manifest.alias {
        println!("  alias         {alias}");
    }
    println!("  directory     {}", dir.display());
    println!("  weights       {}", manifest.weights[0].display());
    if let Some(p) = &manifest.mmproj {
        println!("  projector     {}", p.display());
    }
    println!("  capabilities  {:?}", manifest.capabilities);
    println!();
    println!("resolved settings:");
    println!("  gpu layers    {}", resolved.gpu_layers);
    println!("  cpu moe       {}", resolved.cpu_moe);
    println!("  context       {}", resolved.context_length);
    println!("  kv cache      k={:?} v={:?}", resolved.cache_type_k, resolved.cache_type_v);
    println!("  flash attn    {}", resolved.flash_attention);
    println!("  temperature   {}", resolved.temperature);
    println!("  thinking      {:?}", resolved.thinking);
    println!("  effort        {}", resolved.reasoning_effort);
    println!("  tools         {}", resolved.tools);
    Ok(())
}

fn remove_model(paths: &Paths, model: &str, force: bool) -> Result<()> {
    let found = ozgent_core::resolve(paths, model)?;
    let (r, dir) = (found.model, found.dir);

    if !force {
        // Deleting weights is slow to undo, so require an explicit yes.
        eprint!("Delete {} and everything in it? [y/N] ", dir.display());
        use std::io::Write;
        std::io::stderr().flush().ok();
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    std::fs::remove_dir_all(&dir).with_context(|| format!("removing {}", dir.display()))?;
    // Leave no empty parent behind, so `models/` reflects what is installed.
    if let Some(parent) = dir.parent() {
        if parent != paths.models_dir()
            && std::fs::read_dir(parent).map(|mut d| d.next().is_none()).unwrap_or(false)
        {
            std::fs::remove_dir(parent).ok();
        }
    }
    println!("Removed {r}");
    Ok(())
}

// --------------------------------------------------------------- import

#[allow(clippy::too_many_arguments)]
fn import(
    paths: &Paths,
    weights: PathBuf,
    as_ref: Option<String>,
    name: Option<String>,
    mmproj: Option<PathBuf>,
    copy: bool,
) -> Result<()> {
    // Validate the alias before downloading or linking anything, so a clash is
    // reported before the work rather than after it.
    if let Some(alias) = &name {
        ozgent_core::validate_alias(paths, alias, None)?;
    }
    let reference = as_ref.unwrap_or_else(|| ozgent_hub::suggest_reference(&weights));
    let out = ozgent_hub::import(
        paths,
        &ozgent_hub::ImportRequest { reference, weights, mmproj, copy },
    )?;
    if let Some(alias) = &name {
        ozgent_core::set_alias(paths, &out.model, Some(alias))?;
    }

    println!("Imported {} at {}", out.model, out.dir.display());
    if let Some(alias) = &name {
        println!("  alias: {alias}");
    }
    println!(
        "  {} {}",
        if out.copied { "copied" } else { "hard-linked" },
        out.manifest.weights[0].display()
    );
    if out.manifest.supports_vision() {
        println!("  vision enabled (projector installed)");
    }
    println!("  ozgent run {} \"hello\"", name.as_deref().unwrap_or(&out.model.to_string()));
    Ok(())
}

// ---------------------------------------------------------------- alias

fn set_alias(paths: &Paths, model: &str, alias: Option<String>, clear: bool) -> Result<()> {
    let found = ozgent_core::resolve(paths, model)?;
    match (alias, clear) {
        (Some(a), _) => {
            ozgent_core::set_alias(paths, &found.model, Some(&a))?;
            println!("{} is now also known as {a}", found.model);
        }
        (None, true) => {
            ozgent_core::set_alias(paths, &found.model, None)?;
            println!("cleared the alias for {}", found.model);
        }
        (None, false) => match found.manifest.alias {
            Some(a) => println!("{} is aliased to {a}", found.model),
            None => println!("{} has no alias", found.model),
        },
    }
    Ok(())
}

/// Show, set, or clear the model used when none is named.
///
/// The name is stored exactly as the user typed it, alias and all, rather
/// than expanded to `name:tag`: an alias is the name they chose, it survives
/// re-pulling the model at a different quantisation, and `ozgent list` shows
/// it. Resolving first is still worth doing — it turns a typo into an error
/// here instead of at the start of the next chat.
fn default_model(paths: &Paths, model: Option<String>, clear: bool) -> Result<()> {
    let mut config = ozgent_core::Config::load(paths)?;
    match (model, clear) {
        (Some(name), _) => {
            let found = ozgent_core::resolve(paths, &name)?;
            config.default_model = Some(name.clone());
            config.save(paths)?;
            println!("{} ({}) is now the default", name, found.model);
        }
        (None, true) => {
            config.default_model = None;
            config.save(paths)?;
            println!("cleared; name a model, or set one with `ozgent default <model>`");
        }
        (None, false) => match &config.default_model {
            Some(name) => println!("{name}"),
            None => println!("no default set. Try: ozgent default <model>"),
        },
    }
    Ok(())
}

// ----------------------------------------------------------------- pull

#[allow(clippy::too_many_arguments)]
/// Available quantisations in a repo, largest last, with shard sizes summed.
///
/// A quantisation can be split across several shard files, so summing matters:
/// reporting one shard's size would understate a large model badly.
fn quant_rows(info: &ozgent_hub::RepoInfo) -> Vec<(String, u64, usize)> {
    let mut by_quant: std::collections::BTreeMap<String, (u64, usize)> = Default::default();
    for f in info.files.iter().filter(|f| f.is_gguf() && !f.is_mmproj()) {
        if let Some(q) = ozgent_hub::quant_of(&f.path) {
            let e = by_quant.entry(q).or_insert((0, 0));
            e.0 += f.size;
            e.1 += 1;
        }
    }
    let mut rows: Vec<(String, u64, usize)> =
        by_quant.into_iter().map(|(q, (b, n))| (q, b, n)).collect();
    rows.sort_by_key(|(_, size, _)| *size);
    rows
}

/// Ask which quantisation to download.
///
/// Only called when the terminal is interactive and no `--quant` was given;
/// scripts keep the automatic choice so piped invocations do not block on a
/// prompt that nothing will answer.
fn choose_quant(
    rows: &[(String, u64, usize)],
    mmproj: Option<u64>,
    free_vram: Option<u64>,
) -> Result<Option<String>> {
    use std::io::{IsTerminal, Write};
    if rows.is_empty() || !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return Ok(None);
    }

    // Q4_K_M is the default when it fits: quality per byte flattens out above
    // it, so the largest that merely *fits* would trade a lot of VRAM — and
    // the context length it would otherwise buy — for very little. Falling
    // back to the largest that fits keeps small-VRAM machines working.
    let projector = mmproj.unwrap_or(0);
    let fits = |bytes: u64| free_vram.is_some_and(|v| bytes + projector < v * 85 / 100);
    let default = rows
        .iter()
        .position(|(q, b, _)| q.eq_ignore_ascii_case("Q4_K_M") && fits(*b))
        .or_else(|| rows.iter().rposition(|(_, b, _)| fits(*b)))
        .unwrap_or(0);

    eprintln!("available quantisations:");
    for (i, (q, bytes, shards)) in rows.iter().enumerate() {
        let shards = if *shards > 1 { format!(" · {shards} shards") } else { String::new() };
        let note = match free_vram {
            Some(_) if fits(*bytes) => "  fits in VRAM",
            Some(_) => "  larger than VRAM, will offload to CPU",
            None => "",
        };
        let mark = if i == default { ">" } else { " " };
        eprintln!("{mark} {:>2}. {:<10} {:>9}{shards}{note}", i + 1, q, ozgent_hub::human(*bytes));
    }
    if let Some(size) = mmproj {
        eprintln!("   vision projector adds {}", ozgent_hub::human(size));
    }

    eprint!("choose [1-{}] (enter for {}): ", rows.len(), rows[default].0);
    std::io::stderr().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let line = line.trim();
    if line.is_empty() {
        return Ok(Some(rows[default].0.clone()));
    }
    // Accept a number from the list or the quantisation spelled out.
    if let Ok(n) = line.parse::<usize>() {
        let row = rows
            .get(n.wrapping_sub(1))
            .with_context(|| format!("no option {n}; there are {}", rows.len()))?;
        return Ok(Some(row.0.clone()));
    }
    let found = rows
        .iter()
        .find(|(q, _, _)| q.eq_ignore_ascii_case(line))
        .with_context(|| format!("{line:?} is not one of the available quantisations"))?;
    Ok(Some(found.0.clone()))
}

async fn pull(
    paths: &Paths,
    repo: &str,
    quant: Option<String>,
    as_ref: Option<String>,
    name: Option<String>,
    revision: String,
    max_size_gb: Option<f64>,
    list_only: bool,
) -> Result<()> {
    use ozgent_hub::{Client, Event, PullRequest, human};

    // Check the alias first: a clash after a multi-gigabyte download is a
    // needlessly expensive way to find out.
    if let Some(alias) = &name {
        ozgent_core::validate_alias(paths, alias, None)?;
    }

    let client = Client::new()?;
    let mut request = PullRequest::parse(repo);
    request.revision = revision;
    if quant.is_some() {
        request.quant = quant;
    }
    request.as_ref = as_ref;
    request.budget_bytes = max_size_gb.map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as u64);

    if list_only {
        let info = client.repo(&request.repo_id, &request.revision).await?;
        println!("{} ({} files)", info.id, info.files.len());
        if info.gated { println!("  gated: accept the licence on Hugging Face and set HF_TOKEN"); }
        println!();
        println!("{:<12} {:>10}  {}", "QUANT", "SIZE", "SHARDS");
        for (q, size, shards) in quant_rows(&info) {
            println!("{q:<12} {:>10}  {shards}", human(size));
        }
        if let Some(mm) = info.files.iter().find(|f| f.is_mmproj()) {
            println!("\nvision projector: {} ({})", mm.path, human(mm.size));
        }
        return Ok(());
    }

    // With no quantisation named, offer the choice rather than picking one
    // silently — the trade-off between size and quality is the user's to make.
    if request.quant.is_none() {
        let info = client.repo(&request.repo_id, &request.revision).await?;
        let rows = quant_rows(&info);
        let mmproj = info.files.iter().find(|f| f.is_mmproj()).map(|f| f.size);
        let free_vram = ozgent_llama::backend::best_gpu().map(|d| d.memory_free as u64);
        request.quant = choose_quant(&rows, mmproj, free_vram)?;
    }

    // Progress is drawn on stderr, in place, so piped stdout stays clean.
    let bar = std::sync::Mutex::new(None::<ozgent_hub::Bar>);
    let on_event = move |event: Event<'_>| {
        use std::io::Write;
        let mut err = std::io::stderr();
        match event {
            Event::Resolved { repo, selection, model } => {
                eprintln!("{} -> {model}", repo.id);
                eprintln!(
                    "  {} · {} file(s) · {}",
                    selection.quant,
                    selection.weights.len() + usize::from(selection.mmproj.is_some()),
                    human(selection.total_bytes)
                );
            }
            Event::FileStart { name, index, total, size } => {
                eprintln!("[{index}/{total}] {name}");
                *bar.lock().unwrap() =
                    Some(ozgent_hub::Bar::new(name, size, terminal_columns(), use_colour()));
            }
            Event::FileProgress { done, total, .. } => {
                let mut guard = bar.lock().unwrap();
                if let Some(b) = guard.as_mut() {
                    // Re-read every draw so a resize mid-download is followed.
                    // A line wider than the terminal wraps, and the redraw
                    // then scrolls a screenful of bars instead of replacing
                    // one.
                    b.set_columns(terminal_columns());
                    // The bar decides when a redraw is due; on a fast link a
                    // write per chunk would cost more than the transfer.
                    if let Some(line) = b.update(done) {
                        let _ = write!(err, "\r\x1b[2K{line}");
                        let _ = err.flush();
                    }
                } else {
                    let _ = write!(err, "\r  {} / {}", human(done), human(total));
                    let _ = err.flush();
                }
            }
            Event::FileDone { name, skipped } => {
                let mut guard = bar.lock().unwrap();
                let _ = write!(err, "\r\x1b[2K");
                match (skipped, guard.as_ref()) {
                    (true, _) => eprintln!("  {name}: already present"),
                    (false, Some(b)) => eprintln!("{}", b.finish(b_total(b))),
                    (false, None) => eprintln!("  {name}: done"),
                }
                *guard = None;
            }
        }
    };

    let installed = ozgent_hub::pull(&client, paths, &request, &on_event).await?;
    if let Some(alias) = &name {
        ozgent_core::set_alias(paths, &installed.model, Some(alias))?;
    }

    println!("Installed {} at {}", installed.model, installed.dir.display());
    if let Some(alias) = &name {
        println!("  alias: {alias}");
    }
    if installed.manifest.supports_vision() {
        println!("  vision enabled (projector installed)");
    }
    println!(
        "  ozgent run {} \"hello\"",
        name.as_deref().unwrap_or(&installed.model.to_string())
    );
    Ok(())
}

// ---------------------------------------------------------------- tools

async fn tools(paths: &Paths, config: &Config, command: ToolsCommand) -> Result<()> {
    let host = start_tools(paths, config).await?;

    let result = match command {
        ToolsCommand::List { schema } => {
            println!(
                "{} tools · python {} · worker {}",
                host.tools().len(),
                host.python().map(|p| p.python_version()).unwrap_or("—"),
                host.python().map(|p| p.worker_version()).unwrap_or("—")
            );
            for t in host.tools() {
                println!("\n{}", t.name);
                // One tool per file is the layout; saying which file makes
                // that visible, and tells anyone adding their own where the
                // built-ins live to copy from.
                if let Some(source) = host.source_of(&t.name) {
                    println!("  {}", shorten_home(&source));
                }
                if !t.description.is_empty() {
                    for line in t.description.lines() {
                        println!("  {line}");
                    }
                }
                if schema {
                    println!("  {}", serde_json::to_string_pretty(&t.input_schema)?);
                }
            }
            for e in host.load_errors() {
                eprintln!("warning: {e}");
            }
            Ok(())
        }
        ToolsCommand::Call { name, arguments } => {
            let args: serde_json::Value = serde_json::from_str(&arguments)
                .with_context(|| format!("arguments must be a JSON object, got {arguments:?}"))?;
            match host.call(&name, args).await {
                Ok(v) => {
                    println!("{}", serde_json::to_string_pretty(&v)?);
                    Ok(())
                }
                Err(e) => Err(anyhow::anyhow!("{e}")),
            }
        }
    };

    host.shutdown().await;
    result
}

/// Everything the model can call: ozgent's own Python tools, plus whatever
/// the configured MCP servers offer.
///
/// A server that will not start is reported and skipped rather than fatal —
/// one broken entry in `config.toml` must not take away the tools that work,
/// and must certainly not stop `ozgent chat` opening.
pub(crate) async fn start_tools(paths: &Paths, config: &Config) -> Result<Toolbox> {
    let host_config = HostConfig::from_config(&config.tools, paths)?;
    let python = ToolHost::start(host_config)
        .await
        .context("starting the Python tool worker")?;

    let (sources, problems) = ozgent_mcp::connect_all(&config.mcp).await;
    for problem in &problems {
        eprintln!("warning: mcp: {problem}");
    }
    Ok(Toolbox::new(Some(python), sources))
}

// --------------------------------------------------------------- config

fn config_cmd(paths: &Paths, config: &Config, command: ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Path => {
            println!("{}", paths.config_file().display());
            Ok(())
        }
        ConfigCommand::Show { model } => {
            match model {
                Some(m) => {
                    let resolved = config.options_for(&m).resolve();
                    println!("effective settings for {m}:");
                    println!("  gpu layers   {}", resolved.gpu_layers);
                    println!("  cpu moe      {}", resolved.cpu_moe);
                    println!("  context      {}", resolved.context_length);
                    println!("  temperature  {}", resolved.temperature);
                    println!("  thinking     {:?}", resolved.thinking);
                }
                None => print!("{}", toml::to_string_pretty(config)?),
            }
            Ok(())
        }
        ConfigCommand::Init { force } => {
            let path = paths.config_file();
            anyhow::ensure!(
                force || !path.exists(),
                "{} already exists; pass --force to overwrite",
                path.display()
            );
            paths.ensure()?;
            Config::default().save(paths)?;
            println!("Wrote {}", path.display());
            Ok(())
        }
    }
}

// --------------------------------------------------------------- doctor

async fn doctor(paths: &Paths, config: &Config) -> Result<()> {
    println!("ozgent {}", env!("CARGO_PKG_VERSION"));
    println!();

    println!("directories");
    for (label, dir) in [
        ("root", paths.root().to_path_buf()),
        ("models", paths.models_dir()),
        ("tools", paths.tools_dir()),
        ("config", paths.config_file()),
    ] {
        let mark = if dir.exists() { "ok" } else { "missing" };
        println!("  {label:<8} {:<50} {mark}", dir.display());
    }

    println!();
    println!("models");
    let models = installed_models(paths)?;
    if models.is_empty() {
        println!("  none installed");
    } else {
        for (r, _) in &models {
            println!("  {r}");
        }
    }

    println!();
    println!("tools");
    match start_tools(paths, config).await {
        Ok(host) => {
            if let Some(python) = host.python() {
                println!("  python        {}", python.python_version());
                println!("  worker        {}", python.worker_version());
            }
            println!("  tools loaded  {}", host.tools().len());
            // Only the servers that actually connected. One that did not
            // has already said why, a line above, and listing it here as
            // "0 tools" reads as a server that works and offers nothing.
            for (name, _) in config.mcp.active() {
                let from = format!("mcp:{name}");
                let count = host
                    .tools()
                    .iter()
                    .filter(|t| host.source_of(&t.name).as_deref() == Some(from.as_str()))
                    .count();
                if count > 0 {
                    println!("  {name:<13} {count} tools");
                }
            }
            for e in host.load_errors() {
                println!("  warning       {e}");
            }
            host.shutdown().await;
        }
        Err(e) => println!("  unavailable   {e}"),
    }

    println!();
    println!("backends");
    // This section used to say the engine had not landed yet, which stopped
    // being true long ago. A diagnostic that reports a working subsystem as
    // absent is worse than none, because it is the first thing anyone runs.
    let devices = ozgent_llama::backend::devices();
    if devices.is_empty() {
        println!("  none found; ozgent will run on the CPU");
    }
    for d in &devices {
        println!(
            "  [{}] {:<8} {:<38} {:.1}/{:.1} GiB free  gpu={}",
            d.index,
            d.backend,
            d.description,
            d.free_gib(),
            d.total_gib(),
            d.is_gpu()
        );
    }
    println!("  gpu offload   {}", ozgent_llama::backend::supports_gpu_offload());

    // What the runtime concluded about the model it would actually load, which
    // is the part that decides speed and whether speculation is even possible.
    if let Some((first, _)) = models.first() {
        println!();
        println!("model check ({first})");
        match ozgent_core::resolve(paths, &first.to_string()) {
            Ok(found) => {
                let resolved = config.options_for(&first.to_string()).resolve();
                let weights = found.manifest.primary_weights(&found.dir);
                match ozgent_llama::engine::Engine::load(&weights, &resolved) {
                    Ok(engine) => {
                        println!("  layers        {} ({} on gpu)", engine.n_layer(), engine.gpu_layers_used());
                        println!("  trained ctx   {}", engine.n_ctx_train());
                        println!("  reasoning     {}", engine.is_reasoning_model());
                        // Hybrid and recurrent models cannot roll back a
                        // rejected draft by trimming the cache, which is the
                        // single fact that decides how speculation behaves.
                        println!("  rollback safe {}", engine.rollback_safe());
                        // What the expert-offload planner sees. Zero expert
                        // bytes means a dense model and `cpu_moe = auto`
                        // correctly does nothing.
                        if let Some(l) = ozgent_llama::layout::read(&weights) {
                            println!(
                                "  per layer     {:.1} MiB ({:.1} MiB routed experts)",
                                l.bytes_per_layer as f64 / 1048576.0,
                                l.expert_bytes_per_layer as f64 / 1048576.0
                            );
                            println!("  mixture       {}", l.is_moe());
                        }
                    }
                    Err(e) => println!("  cannot load   {e}"),
                }
            }
            Err(e) => println!("  unresolved    {e}"),
        }
    }
    Ok(())
}

// ----------------------------------------------------------------- chat

async fn chat(
    paths: &Paths,
    config: &Config,
    model: Option<String>,
    options: &OptionFlags,
) -> Result<()> {
    chat::run(paths, config, model, options).await
}

/// A dim one-line note to stderr, so it never lands in piped output.
fn theme_hint(text: &str) -> String {
    format!("\x1b[2m{text}\x1b[0m")
}

/// Context window given to a draft model.
///
/// It is re-synced to the confirmed transcript every round and never needs the
/// target's full window; a large KV here would only take VRAM the target needs.
const DRAFT_CONTEXT: u32 = 4096;

async fn run_once(
    paths: &Paths,
    config: &Config,
    model: &str,
    prompt: &str,
    options: &OptionFlags,
) -> Result<()> {
    use ozgent_core::{Message, ThinkingMode};
    use ozgent_llama::engine::{Engine, StopReason};
    use ozgent_llama::thinking::{Chunk, ThinkingFilter};
    use ozgent_render::{MarkdownRenderer, StreamRenderer, Theme};
    use std::io::Write;

    // Accepts an alias or a full `name:tag`.
    let found = ozgent_core::resolve(paths, model)?;
    let (r, dir, manifest) = (found.model, found.dir, found.manifest);

    // The full precedence chain: defaults < config.toml < manifest < flags.
    let resolved = config
        .options_for(&r.to_string())
        .merge(&manifest.defaults)
        .merge(&options.to_options()?)
        .resolve();

    let prompt_text = if prompt.trim().is_empty() {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        prompt.to_string()
    };
    anyhow::ensure!(!prompt_text.trim().is_empty(), "no prompt given");

    let weights = manifest.primary_weights(&dir);
    let loading = std::time::Instant::now();
    let engine = Engine::load(&weights, &resolved).context("loading the model")?;
    tracing::info!(
        "loaded {} layers ({} on gpu) in {:?}",
        engine.n_layer(),
        engine.gpu_layers_used(),
        loading.elapsed()
    );

    let mut messages = Vec::new();
    // Tell the model what day it is, or "latest" and "tomorrow" resolve
    // against its training data rather than reality.
    let system = match (config.ui.date_awareness, &resolved.system_prompt) {
        (true, Some(base)) => Some(format!(
            "{}\n\n{base}",
            ozgent_core::DateTime::now().prompt_line()
        )),
        (true, None) => Some(ozgent_core::DateTime::now().prompt_line()),
        (false, base) => base.clone(),
    };
    if let Some(system) = system {
        messages.push(Message::system(system));
    }
    // Images the user referenced by path or URL are detected automatically.
    let extracted = ozgent_llama::vision::extract(&prompt_text);
    let mut projector = None;
    let mut images = Vec::new();
    if extracted.has_images() {
        if let Some(mmproj) = manifest.projector_path(&dir) {
            let loaded = engine.projector(&mmproj, &resolved).context("loading the projector")?;
            images = ozgent_llama::mtmd::load_media(&extracted.images)
                .context("reading the image(s)")?;
            projector = Some(loaded);
        } else {
            eprintln!("note: {r} has no vision projector; the image(s) will be ignored");
        }
    }

    // The marker stands where the image belongs in the conversation.
    let user_text = match &projector {
        Some(p) => ozgent_llama::mtmd::with_markers(p.marker(), &extracted.text, images.len()),
        None => extracted.text.clone(),
    };
    messages.push(Message::user(user_text));

    let rendered_prompt = engine.render_prompt_with(&messages, resolved.thinking, resolved.reasoning_effort)?;
    if !engine.has_chat_template() {
        eprintln!("note: this GGUF carries no chat template; using a generic format");
    }

    // A draft model, when one is configured. Loaded here so it lives exactly as
    // long as the session that verifies its proposals.
    //
    // Its own context is small and its KV cheap: it only ever holds the same
    // transcript as the target, and rolls itself back after every proposal.
    let draft_ref = match &resolved.speculative {
        ozgent_core::accel::Speculative::Draft { model, gpu_layers } => {
            Some((model.clone(), *gpu_layers))
        }
        _ => None,
    };
    let draft_loaded = match &draft_ref {
        Some((name, gpu_layers)) => {
            let found = ozgent_core::resolve(paths, name)
                .with_context(|| format!("draft model {name:?} is not installed"))?;
            let weights = found.manifest.primary_weights(&found.dir);
            let mut opts = resolved.clone();
            // The drafter never needs the target's context window: it is
            // re-synced to the confirmed transcript every round, and a large
            // KV here would take VRAM the target needs.
            opts.context_length = resolved.context_length.min(DRAFT_CONTEXT);
            opts.speculative = ozgent_core::accel::Speculative::Off;
            if let Some(n) = gpu_layers {
                opts.gpu_layers = ozgent_core::GpuLayers::Count(*n);
            }
            let engine = Engine::load(&weights, &opts).with_context(|| {
                format!("loading draft model {name:?}")
            })?;
            Some((engine, opts))
        }
        None => None,
    };
    let mut draft_session = match &draft_loaded {
        Some((engine, opts)) => Some(engine.session(opts)?),
        None => None,
    };

    let mut session = engine.session(&resolved)?;

    // A drafter proposes token *ids*. If the two models do not share a
    // vocabulary those ids mean different words to each, and verification
    // silently compares unrelated things — so the mismatch is refused rather
    // than discovered as nonsense output.
    if let Some(draft) = draft_session.as_ref() {
        anyhow::ensure!(
            draft.n_vocab() == session.n_vocab(),
            "the draft model's vocabulary ({}) does not match {}'s ({}); \
             speculation needs both models to share a tokenizer",
            draft.n_vocab(),
            r,
            session.n_vocab(),
        );
        eprintln!(
            "{}",
            theme_hint(&format!(
                "drafting with {}",
                draft_ref.as_ref().map(|(m, _)| m.as_str()).unwrap_or("?")
            ))
        );
    }

    // Reasoning is separated from the answer, then the answer is rendered as
    // streaming markdown.
    let plain = options.plain || !config.ui.markdown;
    let theme = if plain { Theme::plain() } else { Theme::default() };
    let width = terminal_width();
    let mut markdown = StreamRenderer::new(MarkdownRenderer::new(theme.clone(), width));
    let mut filter = ThinkingFilter::new(resolved.thinking);
    if let Some(close) = Engine::stream_starts_inside(&rendered_prompt) {
        filter = filter.starting_inside(close);
    }

    let mut out = std::io::stdout();
    let mut showed_thinking = false;
    let mut produced_answer = false;

    let media = projector.as_ref().map(|p| (p, &images[..], &extracted.images[..]));
    let (stats, reason) = session.generate_drafted(&rendered_prompt, media, resolved.max_tokens, draft_session.as_mut(), |piece| {
        for chunk in filter.push(piece) {
            match chunk {
                Chunk::Thinking(text) => {
                    if resolved.thinking != ThinkingMode::Off && config.ui.show_thinking {
                        if !showed_thinking {
                            showed_thinking = true;
                            eprint!("{}", theme.style(theme.thinking, "thinking: "));
                        }
                        eprint!("{}", theme.style(theme.thinking, &text));
                        let _ = std::io::stderr().flush();
                    }
                }
                Chunk::Answer(text) => {
                    produced_answer = true;
                    if showed_thinking {
                        showed_thinking = false;
                        eprintln!();
                    }
                    let _ = markdown.push(&text, &mut out);
                }
            }
        }
        true
    })?;

    for chunk in filter.finish() {
        if let Chunk::Answer(text) = chunk {
            let _ = markdown.push(&text, &mut out);
        }
    }
    markdown.finish(&mut out)?;
    println!();

    if stats.generated_tokens > 0 && !produced_answer {
        eprintln!(
            "note: the model produced no answer{}. Raise --max-tokens, or use --think on to see what it did.",
            if filter.saw_thinking() { " — it spent the whole budget reasoning" } else { "" }
        );
    }

    if options.stats || config.ui.show_stats {
        eprintln!(
            "\n{} prompt tokens ({:.1}/s) · {} generated ({:.1}/s) · stopped: {reason:?}",
            stats.prompt_tokens,
            stats.prompt_tokens_per_second(),
            stats.generated_tokens,
            stats.tokens_per_second(),
        );
        eprintln!(
            "host {} ms of {} ms ({:.1}%) · drafted {} accepted {}",
            stats.callback_ms,
            stats.generation_ms,
            stats.callback_ms as f64 * 100.0 / stats.generation_ms.max(1) as f64,
            stats.drafted_tokens,
            stats.accepted_drafts,
        );
        eprintln!(
            "drafts proposed {} accepted {} ({:.0}%) · bookkeeping {} us",
            stats.proposed_drafts,
            stats.accepted_drafts,
            stats.accepted_drafts as f64 * 100.0 / stats.proposed_drafts.max(1) as f64,
            stats.spec_ms,
        );
    }
    if reason == StopReason::ContextFull {
        eprintln!("note: the context filled up; raise --ctx for a longer answer");
    }
    Ok(())
}

/// Terminal width, falling back to a readable default when not a tty.
fn terminal_width() -> usize {
    ozgent_render::terminal_width()
}

/// Report a command that exists but has no engine behind it yet.
///
/// Better an explicit, accurate message than a confusing failure deeper down.
fn not_yet(what: &str) -> Result<()> {
    anyhow::bail!("{what} needs the inference engine, which is still being built")
}

/// The byte total a finished bar should report.
fn b_total(bar: &ozgent_hub::Bar) -> u64 {
    bar.total()
}

/// Replace the home directory with `~`, so a path fits on one line.
///
/// Purely cosmetic, and deliberately conservative: anything not under home is
/// printed as it is rather than shortened by guesswork.
fn shorten_home(path: &str) -> String {
    let Some(home) = std::env::var_os("HOME") else { return path.to_string() };
    let home = home.to_string_lossy().into_owned();
    match path.strip_prefix(home.as_str()) {
        Some(rest) if rest.starts_with('/') => format!("~{rest}"),
        _ => path.to_string(),
    }
}

fn human_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_path_outside_home_is_left_alone() {
        assert_eq!(super::shorten_home("/usr/lib/ozgent/x.py"), "/usr/lib/ozgent/x.py");
    }

    #[test]
    fn home_itself_is_not_mistaken_for_a_prefix() {
        // `/home/alice-backup` starts with `/home/alice` but is not inside it.
        let Some(home) = std::env::var_os("HOME") else { return };
        let sibling = format!("{}-backup/x.py", home.to_string_lossy());
        assert_eq!(super::shorten_home(&sibling), sibling);
    }

    use super::*;

    #[test]
    fn sizes_are_human_readable() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(1024), "1.0 KB");
        assert_eq!(human_size(7_400_000_000), "6.9 GB");
    }

    #[test]
    fn listing_an_empty_models_dir_is_not_an_error() {
        let dir = std::env::temp_dir().join(format!("ozgent-cli-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("models")).unwrap();
        let paths = Paths::with_root(&dir);
        assert!(installed_models(&paths).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nested_model_directories_are_discovered() {
        let dir = std::env::temp_dir().join(format!("ozgent-cli-nested-{}", std::process::id()));
        let model_dir = dir.join("models").join("gemma4").join("12b");
        std::fs::create_dir_all(&model_dir).unwrap();

        let r = ModelRef::parse("gemma4:12b").unwrap();
        ozgent_core::Manifest::new(&r, "model.gguf").save(&model_dir).unwrap();

        let found = installed_models(&Paths::with_root(&dir)).unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0.to_string(), "gemma4:12b");

        std::fs::remove_dir_all(&dir).ok();
    }
}

// ------------------------------------------------------------------ channels

/// Run the messaging gateway, and optionally the web interface beside it.
///
/// One [`App`](ozgent_web::state::App) serves both when `--web` is given, which
/// is the point of doing it in one process: the inference thread, the loaded
/// model, the tool host and the permission grants are all per-`App`, so two of
/// them would mean two copies of the model in VRAM and an answer given in the
/// browser having no effect on a question asked from a chat.
async fn gateway(
    paths: Paths,
    config: Config,
    web: bool,
    host: &str,
    port: u16,
    options: ozgent_core::Options,
) -> Result<()> {
    if !config.channels.enabled {
        anyhow::bail!(
            "channels are switched off. Add this to {}:\n\n  [channels]\n  enabled = true\n\n\
             Then turn one on — see `ozgent channel --help` and docs/channels.md.",
            paths.config_file().display()
        );
    }
    let state = ozgent_web::state::App::new(paths.clone(), config, options).await?;

    if web {
        let serving = ozgent_web::serve_with(state.clone(), host, port);
        let answering = ozgent_channels::gateway::run(state, paths);
        // Either one stopping ends the process: a gateway with no engine and a
        // server with no channels are both half a program.
        tokio::select! {
            r = serving => r,
            r = answering => r,
        }
    } else {
        ozgent_channels::gateway::run(state, paths).await
    }
}

async fn channel(paths: &Paths, mut config: Config, command: ChannelCommand) -> Result<()> {
    use ozgent_core::channels::Kind;

    let parse_kind = |name: &str| -> Result<Kind> {
        Kind::parse(name)
            .ok_or_else(|| anyhow::anyhow!("no channel called `{name}`. Try telegram or whatsapp."))
    };

    match command {
        ChannelCommand::Status => {
            channel_status(paths, &config);
            Ok(())
        }

        ChannelCommand::Install { channel } => match parse_kind(&channel)? {
            Kind::Telegram => {
                println!("Telegram needs no install. Get a token from @BotFather and put it in");
                println!("{}:", paths.config_file().display());
                println!();
                println!("  [channels]");
                println!("  enabled = true");
                println!();
                println!("  [channels.telegram]");
                println!("  enabled = true");
                println!("  token   = \"…\"");
                Ok(())
            }
            Kind::WhatsApp => {
                let bridge = ozgent_channels::whatsapp::locate(
                    config.channels.whatsapp.bridge.as_deref(),
                    paths,
                )?;
                println!("Installing the WhatsApp bridge in {}", bridge.display());
                ozgent_channels::whatsapp::install(&config.channels.whatsapp.node, &bridge).await?;
                println!("Done. Next: ozgent channel login whatsapp");
                Ok(())
            }
        },

        ChannelCommand::Login { channel } => match parse_kind(&channel)? {
            Kind::Telegram => {
                anyhow::bail!(
                    "Telegram is not linked by scanning; it uses a bot token. \
                     Run `ozgent channel install telegram` to see how."
                )
            }
            Kind::WhatsApp => {
                let wa = &config.channels.whatsapp;
                let bridge = ozgent_channels::whatsapp::locate(wa.bridge.as_deref(), paths)?;
                if !ozgent_channels::whatsapp::is_installed(&bridge) {
                    anyhow::bail!(
                        "the bridge is not installed yet. Run `ozgent channel install whatsapp`."
                    );
                }
                let state = paths.channel_dir("whatsapp").join("auth");
                let who = ozgent_channels::whatsapp::login(&wa.node, &bridge, &state).await?;
                println!();
                println!("Linked {who}.");
                println!();
                println!("The credentials are in {}.", state.display());
                println!("They can read and send your messages — treat that like a password.");
                println!();
                println!("Next: turn the channel on in {}:", paths.config_file().display());
                println!();
                println!("  [channels]");
                println!("  enabled = true");
                println!();
                println!("  [channels.whatsapp]");
                println!("  enabled = true");
                println!();
                println!("To talk to it in your own chat with yourself, add:");
                println!();
                println!("  self_chat = true");
                println!();
                println!("Leave that off if you use that chat as a notepad.");
                Ok(())
            }
        },

        ChannelCommand::Logout { channel } => {
            let kind = parse_kind(&channel)?;
            let state = paths.channel_dir(kind.as_str());
            match kind {
                Kind::WhatsApp => {
                    ozgent_channels::whatsapp::logout(&state.join("auth"))?;
                    println!("Unlinked. Remove the device from WhatsApp ▸ Linked devices too,");
                    println!("so the session on their side is gone as well.");
                }
                Kind::Telegram => {
                    config.channels.telegram.token.clear();
                    config.save(paths)?;
                    println!("Token cleared. It is still valid on Telegram's side —");
                    println!("revoke it with @BotFather if it should stop working.");
                }
            }
            Ok(())
        }

        ChannelCommand::Allow { channel, identity } => {
            let kind = parse_kind(&channel)?;
            if config.channels.admit(kind, &identity) {
                config.save(paths)?;
                println!("{identity} can now talk to {kind}.");
            } else {
                println!("{identity} was already allowed on {kind}.");
            }
            Ok(())
        }

        ChannelCommand::Deny { channel, identity } => {
            let kind = parse_kind(&channel)?;
            let allow = match kind {
                Kind::Telegram => &mut config.channels.telegram.allow,
                Kind::WhatsApp => &mut config.channels.whatsapp.allow,
            };
            let before = allow.len();
            allow.retain(|a| !a.eq_ignore_ascii_case(identity.trim()));
            if allow.len() == before {
                println!("{identity} was not on the {kind} list.");
                return Ok(());
            }
            config.save(paths)?;
            println!("{identity} can no longer talk to {kind}.");
            Ok(())
        }
    }
}

/// The chats bound to a channel, for `ozgent channel status`.
///
/// Opened read-only and closed again: the gateway may be running against the
/// same file, and WAL lets a reader in while it writes.
fn bound_chats(paths: &Paths, channel: &str) -> Result<Vec<ozgent_memory::ChannelChat>> {
    let store = ozgent_memory::Store::open(paths.root().join("ozgent.db"))?;
    Ok(store.channel_chats(channel)?)
}

fn channel_status(paths: &Paths, config: &Config) {
    use ozgent_core::channels::{Kind, is_open_to_everyone};

    println!("channels{}", if config.channels.enabled { "" } else { "  (switched off)" });
    println!();

    for kind in [Kind::Telegram, Kind::WhatsApp] {
        let access = config.channels.access(kind);
        let on = match kind {
            Kind::Telegram => config.channels.telegram.enabled,
            Kind::WhatsApp => config.channels.whatsapp.enabled,
        };
        println!("  {kind}");
        println!("    enabled   {}", if on { "yes" } else { "no" });

        match kind {
            Kind::Telegram => {
                // Never the token itself: it is a password, and this output
                // gets pasted into issues.
                let has = !config.channels.telegram.token.trim().is_empty()
                    || std::env::var("OZGENT_TELEGRAM_TOKEN").is_ok();
                println!("    token     {}", if has { "set" } else { "missing" });
            }
            Kind::WhatsApp => {
                let linked = paths.channel_dir("whatsapp").join("auth").join("creds.json").is_file();
                println!("    linked    {}", if linked { "yes" } else { "no" });
                println!(
                    "    self-chat {}",
                    if config.channels.whatsapp.self_chat {
                        "on — your own notes are answered"
                    } else {
                        "off"
                    }
                );
                let installed = ozgent_channels::whatsapp::locate(
                    config.channels.whatsapp.bridge.as_deref(),
                    paths,
                )
                .map(|b| ozgent_channels::whatsapp::is_installed(&b))
                .unwrap_or(false);
                println!("    bridge    {}", if installed { "installed" } else { "not installed" });
            }
        }

        if is_open_to_everyone(access.allow) {
            println!("    allowed   EVERYONE — anyone who finds it can use this machine's tools");
        } else if access.allow.is_empty() {
            println!("    allowed   nobody");
        } else {
            println!("    allowed   {}", access.allow.join(", "));
        }
        match access.tools {
            Some(list) if list.is_empty() => println!("    tools     none"),
            Some(list) => println!("    tools     {}", list.join(", ")),
            None => println!("    tools     all of them"),
        }

        match bound_chats(paths, kind.as_str()) {
            Ok(chats) if !chats.is_empty() => {
                println!("    chats     {}", chats.len());
                for chat in chats.iter().take(5) {
                    let who = if chat.display.is_empty() { &chat.chat_id } else { &chat.display };
                    println!("              {who}");
                }
            }
            _ => {}
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

// ---------------------------------------------------------------------- mcp

/// Show every configured MCP server and what it offers.
///
/// Connects for real rather than reading the file back: "it is in config.toml"
/// and "the model can call it" are different claims, and the gap between them
/// is exactly what this command exists to show.
/// The file a new agent starts from, with every field present and explained.
const AGENT_TEMPLATE: &str = r#"# An ozgent agent. Call it by writing @NAME in a message.
# Everything here is read again on the next message, so edit and try.

# One line, shown in the @ panel.
description = "What this agent is for"

# The only tools it may call. `ozgent tools` lists what this machine has.
tools = ["web_search", "fetch_url"]

# Tool rounds before it must answer (1-32).
max_rounds = 8

# auto, on or off. Leave out to follow the chat.
# thinking = "auto"

# Leave out to use the model's own.
# temperature = 0.4

instructions = """
Say what the agent does, how it should go about it, and what its report
should look like. Be specific: this is the whole of its job description.
"""

# Per-tool rules on top of your global ones: allow, ask or deny.
# A global deny always wins.
[permissions]
web_search = "allow"
fetch_url = "allow"
"#;

/// `ozgent agent …`
fn agent(paths: &Paths, command: AgentCommand) -> Result<()> {
    use ozgent_core::agents::{self, Origin};
    let catalog = ozgent_core::AgentCatalog::load(paths);
    let origin = |o: Origin| match o {
        Origin::Builtin => "built in",
        Origin::User => "yours",
        Origin::Override => "edited built-in",
    };
    match command {
        AgentCommand::List => {
            for a in catalog.all() {
                println!("@{:<22} {}  ({})", a.name, a.definition.description, origin(a.origin));
                println!("  {:<22} tools: {}", "", a.definition.tools.join(", "));
            }
            for e in &catalog.errors {
                eprintln!("warning: {e}");
            }
            println!("\nfiles: {}", paths.agents_dir().display());
        }
        AgentCommand::Show { name } => {
            let name = name.trim_start_matches('@');
            let a = catalog.get(name).with_context(|| format!("no agent named {name}"))?;
            println!("@{}  ({})", a.name, origin(a.origin));
            print!("{}", toml::to_string_pretty(&a.definition)?);
        }
        AgentCommand::New { name, from } => {
            let name = name.trim_start_matches('@').to_string();
            agents::validate_name(&name)?;
            let path = ozgent_core::Agent::path(paths, &name);
            if path.exists() || catalog.get(&name).is_some() {
                anyhow::bail!("@{name} already exists; `ozgent agent edit {name}` changes it");
            }
            std::fs::create_dir_all(paths.agents_dir())?;
            match from {
                Some(source) => {
                    let source = source.trim_start_matches('@');
                    let base = catalog.get(source).with_context(|| format!("no agent named {source}"))?;
                    agents::save(paths, &name, &base.definition)?;
                }
                None => std::fs::write(&path, AGENT_TEMPLATE.replace("NAME", &name))?,
            }
            println!("created {}", path.display());
            open_in_editor(&path, &name)?;
        }
        AgentCommand::Edit { name } => {
            let name = name.trim_start_matches('@').to_string();
            let a = catalog.get(&name).with_context(|| format!("no agent named {name}"))?;
            let path = ozgent_core::Agent::path(paths, &name);
            if a.origin == Origin::Builtin {
                // The copy is what gets edited; deleting it restores the original.
                agents::save(paths, &name, &a.definition)?;
                println!("copied the built-in @{name} to {}", path.display());
            }
            open_in_editor(&path, &name)?;
        }
        AgentCommand::Rm { name } => {
            let name = name.trim_start_matches('@');
            match agents::remove(paths, name)? {
                Origin::Override => println!("removed your version; the built-in @{name} is back"),
                _ => println!("deleted @{name}"),
            }
        }
    }
    Ok(())
}

/// Open an agent file in `$VISUAL` or `$EDITOR`, then check what was saved.
///
/// Checked on the way out because a typo in the file would otherwise surface
/// as the agent silently missing from the @ panel next time.
fn open_in_editor(path: &std::path::Path, name: &str) -> Result<()> {
    let Some(editor) = std::env::var_os("VISUAL").or_else(|| std::env::var_os("EDITOR")) else {
        println!("set $EDITOR to open it here, or edit {} yourself", path.display());
        return Ok(());
    };
    let status = std::process::Command::new(&editor).arg(path).status()
        .with_context(|| format!("running {}", editor.to_string_lossy()))?;
    if !status.success() {
        anyhow::bail!("the editor exited with {status}");
    }
    let text = std::fs::read_to_string(path)?;
    match ozgent_core::Agent::parse(name, &text, ozgent_core::agents::Origin::User) {
        Ok(_) => println!("@{name} is ready. Write @{name} in a message to call it."),
        Err(e) => println!("warning: @{name} will not load until this is fixed: {e}"),
    }
    Ok(())
}

async fn mcp(config: &Config) -> Result<()> {
    if !config.mcp.enabled {
        println!("MCP is switched off. Add to {}:", "config.toml");
        println!();
        println!("  [mcp]");
        println!("  enabled = true");
        println!();
        println!("Then a server, e.g.:");
        println!();
        println!("  [mcp.servers.files]");
        println!("  command = \"npx\"");
        println!("  args    = [\"-y\", \"@modelcontextprotocol/server-filesystem\", \"/tmp\"]");
        return Ok(());
    }
    if config.mcp.servers.is_empty() {
        println!("MCP is on, but no servers are configured.");
        return Ok(());
    }

    println!("connecting…");
    println!();
    let (sources, problems) = ozgent_mcp::connect_all(&config.mcp).await;

    for source in &sources {
        // `mcp:<name>` — the name is what its tools are prefixed with.
        let name = source.origin().strip_prefix("mcp:").unwrap_or(source.origin());
        let trusted = config
            .mcp
            .servers
            .get(name)
            .is_some_and(|s| s.trust_hints);
        println!("{name}");
        println!(
            "  hints         {}",
            if trusted { "trusted — read-only tools run unasked" } else { "not trusted — every tool asks" }
        );
        println!("  tools         {}", source.tools().len());
        for tool in source.tools() {
            let rule = config.permissions.rule_for(&tool.name, tool.effect);
            println!("    {:<34} {:<8} {rule}", tool.name, tool.effect.to_string());
            if !tool.description.is_empty() {
                println!("      {}", ozgent_tools::first_line(&tool.description));
            }
        }
        println!();
    }

    if !problems.is_empty() {
        println!("could not be used");
        for problem in &problems {
            println!("  {problem}");
        }
        println!();
    }

    for source in &sources {
        source.shutdown().await;
    }

    if sources.is_empty() && problems.is_empty() {
        println!("Nothing is enabled.");
    }
    Ok(())
}

/// The terminal's width, or a conservative default when there is no terminal.
///
/// Eighty is the safe fallback: too wide and the line wraps into the scrolling
/// mess this exists to prevent, too narrow only wastes space.
fn terminal_columns() -> usize {
    ozgent_render::crossterm::terminal::size().map(|(w, _)| w as usize).unwrap_or(80)
}

/// Whether to colour the progress line.
///
/// Off when stderr is not a terminal — a log file full of escape sequences is
/// worse than a plain one — and off when NO_COLOR is set, which is the
/// convention every other tool honours.
fn use_colour() -> bool {
    use std::io::IsTerminal;
    std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal()
}
