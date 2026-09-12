//! `ozgent daemon`: one long-running ozgent that everything else talks to.
//!
//! Until now every way into ozgent brought its own everything. `ozgent chat`
//! loads a model, `ozgent web` loads a model, `ozgent gateway` loads a model.
//! Run two and there are two copies in VRAM; run none and a job scheduled for
//! 9:20 does not happen, because nothing was awake to run it.
//!
//! The daemon is the one process that is always there. It owns the model, the
//! database, the tool worker, the messaging channels and the scheduler, and it
//! serves the web interface and the OpenAI- and Anthropic-compatible API over
//! the same loaded model. A browser is a client of it. A phone messaging
//! Telegram is a client of it. A job at 9:20 is the daemon talking to itself.
//!
//! # Being cheap while doing nothing
//!
//! A daemon's real workload is the twenty-three hours a day it does nothing,
//! and the only honest measure is what it costs then. Three things are done
//! about it, in descending order of how much they matter:
//!
//! * **The model is not resident.** Nothing is loaded until a question
//!   arrives, and it is dropped again after `[web] idle_unload_minutes`.
//!   This is the whole ballgame: a 4B model at Q4 is around 3 GB, and holding
//!   it overnight for one morning brief is 3 GB of nothing.
//! * **Freed memory is given back.** `free()` returns memory to glibc, not to
//!   the kernel, so a process that loads and unloads a model looks like it
//!   never let go. The daemon asks for the arenas to be trimmed after every
//!   unload — see [`ozgent_web::worker::release_memory`].
//! * **It sleeps until there is something to do.** The scheduler wakes when
//!   the next job is due rather than on a fixed tick, with a one-minute
//!   ceiling so changes made in another process are noticed.
//!
//! What is left is a tokio runtime, an HTTP listener, a SQLite handle, and a
//! polling connection per configured channel.
//!
//! # Init systems
//!
//! systemd is not the only one, and a program that assumes it is simply does
//! not install on Void, Alpine, Artix or a Mac. So the service file is
//! *generated* per init system — see [`Init`] — and installed automatically
//! wherever that can be done as the user who owns `~/ozgent`. Where it cannot
//! (the init systems with no per-user services), the file is still written,
//! correctly, and the two commands to install it are printed rather than a
//! `sudo` being run on someone's behalf.

use anyhow::{Context, Result};
use ozgent_core::{Config, Paths};
use std::path::{Path, PathBuf};

/// The name the service is known by, whichever init runs it.
pub const SERVICE: &str = "ozgent";

/// Run the daemon in the foreground until stopped.
///
/// Foreground on purpose: an init system wants a process it can supervise, not
/// one that forks away and leaves it holding nothing. Under a service manager
/// stdout is the log, which is why this prints a compact block rather than the
/// banner `ozgent web` shows a person who just typed a command.
pub async fn run(
    paths: Paths,
    config: Config,
    host: &str,
    port: u16,
    options: ozgent_core::Options,
) -> Result<()> {
    let idle = config.web.idle_unload_minutes;
    let state = ozgent_web::state::App::new(paths.clone(), config, options).await?;

    // The channels, in this process, over the same model. `Mode::Web` because
    // there is no terminal to print a QR code to — linking WhatsApp on a
    // daemon is done from /admin.
    ozgent_channels::gateway::start(state.clone(), paths, ozgent_channels::gateway::Mode::Web);

    tracing::info!("ozgent daemon on {host}:{port}");
    match idle {
        0 => tracing::info!("the model is kept loaded once a question arrives"),
        n => tracing::info!("the model is dropped after {n} idle minutes"),
    }

    // Everything else — the web interface, the API and the scheduler — comes
    // up inside here, over the same App.
    ozgent_web::serve_with(state, host, port).await
}

// ------------------------------------------------------------------- init

/// A service manager this machine might be running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Init {
    Systemd,
    /// macOS.
    Launchd,
    /// Alpine, Gentoo, Artix, Devuan.
    OpenRc,
    /// Void, Artix.
    Runit,
    /// Obarun, Artix. Its source format varies enough that ozgent writes the
    /// pieces and leaves the compilation to the administrator.
    S6,
    /// Chimera, Artix. One of the few besides systemd with user services.
    Dinit,
    Unknown,
}

impl Init {
    pub fn name(self) -> &'static str {
        match self {
            Self::Systemd => "systemd",
            Self::Launchd => "launchd",
            Self::OpenRc => "OpenRC",
            Self::Runit => "runit",
            Self::S6 => "s6",
            Self::Dinit => "dinit",
            Self::Unknown => "no service manager I recognise",
        }
    }

    /// Whether a service can be installed without root.
    ///
    /// The ones that can are the ones that should be: ozgent's data is one
    /// user's, and a root-owned daemon reading `~/ozgent` is a worse
    /// arrangement than a user service, not a more serious one.
    pub fn per_user(self) -> bool {
        matches!(self, Self::Systemd | Self::Launchd | Self::Dinit)
    }
}

/// Work out what is supervising this machine.
///
/// Checked in order of how definitive the evidence is. `/run/systemd/system`
/// existing means systemd is *running*, which `systemctl` being installed does
/// not — several distributions ship the binary as a dependency of something
/// else. The same logic applies down the list, which is why each case looks
/// for a runtime directory rather than a command on `$PATH`.
pub fn detect() -> Init {
    if cfg!(target_os = "macos") {
        return Init::Launchd;
    }
    if Path::new("/run/systemd/system").is_dir() {
        return Init::Systemd;
    }
    if Path::new("/run/openrc").is_dir() || Path::new("/run/openrc/softlevel").exists() {
        return Init::OpenRc;
    }
    // Void mounts /run/runit; some others only have the service directory.
    if Path::new("/run/runit").is_dir()
        || Path::new("/etc/runit/runsvdir/current").exists()
        || Path::new("/var/service").is_dir()
    {
        return Init::Runit;
    }
    if Path::new("/run/s6-rc").is_dir() || Path::new("/run/service").is_dir() {
        return Init::S6;
    }
    if Path::new("/run/dinitctl").exists() {
        return Init::Dinit;
    }
    // Nothing is running, but something is installed: a container, or a
    // machine mid-setup. Better than claiming there is no way to do this.
    for (command, init) in [
        ("systemctl", Init::Systemd),
        ("rc-update", Init::OpenRc),
        ("sv", Init::Runit),
        ("s6-rc", Init::S6),
        ("dinitctl", Init::Dinit),
    ] {
        if which(command).is_some() {
            return init;
        }
    }
    Init::Unknown
}

/// A service file: where it goes, and what is in it.
pub struct Service {
    pub path: PathBuf,
    pub contents: String,
    /// Whether it has to be executable, as runit and OpenRC require.
    pub executable: bool,
}

/// Build the service file for an init system.
pub fn service_file(
    init: Init,
    exe: &Path,
    paths: &Paths,
    host: &str,
    port: u16,
    user: &str,
) -> Option<Service> {
    let exe = exe.display();
    let root = paths.root().display();
    let home = dirs::home_dir()?;

    Some(match init {
        Init::Systemd => Service {
            path: home.join(".config/systemd/user").join(format!("{SERVICE}.service")),
            executable: false,
            contents: format!(
                "[Unit]\n\
                 Description=ozgent — a local AI assistant\n\
                 Documentation=https://github.com/0znio/ozgent\n\
                 # Channels and model downloads need a route out, not just an address.\n\
                 After=network-online.target\n\
                 Wants=network-online.target\n\
                 \n\
                 [Service]\n\
                 Type=simple\n\
                 ExecStart={exe} daemon --host {host} --port {port}\n\
                 # Stated rather than inherited, so the unit does not depend on\n\
                 # the environment of whoever happened to install it.\n\
                 Environment=OZGENT_HOME={root}\n\
                 Environment=OZGENT_LOG=info\n\
                 Restart=on-failure\n\
                 RestartSec=5\n\
                 # A model that will not fit fails instantly, every time. Without\n\
                 # a burst limit that is a restart loop that never stops.\n\
                 StartLimitIntervalSec=300\n\
                 StartLimitBurst=5\n\
                 # Long enough for a large model to finish loading before a stop\n\
                 # is escalated to a kill.\n\
                 TimeoutStopSec=60\n\
                 \n\
                 [Install]\n\
                 WantedBy=default.target\n"
            ),
        },

        Init::Launchd => Service {
            path: home.join("Library/LaunchAgents/com.ozgent.daemon.plist"),
            executable: false,
            contents: format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                 <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
                 \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
                 <plist version=\"1.0\">\n\
                 <dict>\n\
                 \x20 <key>Label</key><string>com.ozgent.daemon</string>\n\
                 \x20 <key>ProgramArguments</key>\n\
                 \x20 <array>\n\
                 \x20   <string>{exe}</string>\n\
                 \x20   <string>daemon</string>\n\
                 \x20   <string>--host</string><string>{host}</string>\n\
                 \x20   <string>--port</string><string>{port}</string>\n\
                 \x20 </array>\n\
                 \x20 <key>EnvironmentVariables</key>\n\
                 \x20 <dict>\n\
                 \x20   <key>OZGENT_HOME</key><string>{root}</string>\n\
                 \x20   <key>OZGENT_LOG</key><string>info</string>\n\
                 \x20 </dict>\n\
                 \x20 <key>RunAtLoad</key><true/>\n\
                 \x20 <key>KeepAlive</key>\n\
                 \x20 <dict><key>SuccessfulExit</key><false/></dict>\n\
                 \x20 <key>StandardOutPath</key><string>{root}/logs/daemon.log</string>\n\
                 \x20 <key>StandardErrorPath</key><string>{root}/logs/daemon.log</string>\n\
                 </dict>\n\
                 </plist>\n"
            ),
        },

        Init::OpenRc => Service {
            path: PathBuf::from("/etc/init.d").join(SERVICE),
            executable: true,
            contents: format!(
                "#!/sbin/openrc-run\n\
                 # ozgent — a local AI assistant\n\
                 \n\
                 description=\"ozgent daemon\"\n\
                 command=\"{exe}\"\n\
                 command_args=\"daemon --host {host} --port {port}\"\n\
                 command_background=true\n\
                 # Runs as the user whose ~/ozgent this is, not as root: the data\n\
                 # is one person's, and root would only widen what a tool can reach.\n\
                 command_user=\"{user}\"\n\
                 pidfile=\"/run/{SERVICE}.pid\"\n\
                 output_log=\"{root}/logs/daemon.log\"\n\
                 error_log=\"{root}/logs/daemon.log\"\n\
                 export OZGENT_HOME=\"{root}\"\n\
                 export OZGENT_LOG=\"info\"\n\
                 \n\
                 depend() {{\n\
                 \x20   need net\n\
                 \x20   after firewall\n\
                 }}\n\
                 \n\
                 start_pre() {{\n\
                 \x20   checkpath --directory --owner {user} --mode 0755 \"{root}/logs\"\n\
                 }}\n"
            ),
        },

        Init::Runit => Service {
            path: PathBuf::from("/etc/sv").join(SERVICE).join("run"),
            executable: true,
            contents: format!(
                "#!/bin/sh\n\
                 # ozgent — a local AI assistant\n\
                 exec 2>&1\n\
                 export OZGENT_HOME=\"{root}\"\n\
                 export OZGENT_LOG=\"info\"\n\
                 # chpst drops to the user who owns ~/ozgent; runit itself is root.\n\
                 exec chpst -u {user} {exe} daemon --host {host} --port {port}\n"
            ),
        },

        Init::S6 => Service {
            path: PathBuf::from("/etc/s6/sv").join(SERVICE).join("run"),
            executable: true,
            contents: format!(
                "#!/bin/execlineb -P\n\
                 # ozgent — a local AI assistant\n\
                 fdmove -c 2 1\n\
                 export OZGENT_HOME {root}\n\
                 export OZGENT_LOG info\n\
                 s6-setuidgid {user}\n\
                 {exe} daemon --host {host} --port {port}\n"
            ),
        },

        Init::Dinit => Service {
            path: home.join(".config/dinit.d").join(SERVICE),
            executable: false,
            contents: format!(
                "# ozgent — a local AI assistant\n\
                 type = process\n\
                 command = {exe} daemon --host {host} --port {port}\n\
                 env-file = \n\
                 restart = true\n\
                 smooth-recovery = true\n\
                 logfile = {root}/logs/daemon.log\n\
                 depends-on = network\n"
            ),
        },

        Init::Unknown => return None,
    })
}

/// Write the service file, and start it where that can be done as this user.
pub fn install(paths: &Paths, host: &str, port: u16) -> Result<()> {
    let init = detect();
    let exe = std::env::current_exe().context("cannot find where ozgent is installed")?;
    // Resolved, because a service pointing at a symlink in a temporary
    // directory stops working the moment that directory goes.
    let exe = exe.canonicalize().unwrap_or(exe);
    let user = whoami();

    let Some(service) = service_file(init, &exe, paths, host, port, &user) else {
        println!("This machine has {}.", init.name());
        println!();
        println!("ozgent can still run in the background — anything that keeps a command");
        println!("running will do:");
        println!();
        println!("    {} daemon --host {host} --port {port}", exe.display());
        println!();
        println!("Set OZGENT_HOME={} so it reads the right directory.", paths.root().display());
        return Ok(());
    };

    println!("Service manager: {}", init.name());

    if !init.per_user() {
        // These have no per-user services, so installing means writing under
        // /etc as root. The file is generated here — it is the part that has
        // to be right — and the two commands are printed rather than a sudo
        // being run on somebody's behalf.
        let staged = paths.root().join("service").join(
            service.path.file_name().unwrap_or_else(|| std::ffi::OsStr::new(SERVICE)),
        );
        if let Some(dir) = staged.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&staged, &service.contents)
            .with_context(|| format!("writing {}", staged.display()))?;
        println!("wrote {}", staged.display());
        println!();
        println!("{} services live under /etc, so these two need root:", init.name());
        println!();
        for line in root_install(init, &staged, &service.path) {
            println!("    {line}");
        }
        return Ok(());
    }

    if let Some(dir) = service.path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(&service.path, &service.contents)
        .with_context(|| format!("writing {}", service.path.display()))?;
    if service.executable {
        make_executable(&service.path)?;
    }
    println!("wrote {}", service.path.display());

    match enable(init, &service.path) {
        Ok(()) => println!("ozgent is running, and starts again when you log in."),
        Err(e) => {
            println!();
            println!("The service file is written, but starting it failed: {e}");
            println!("Start it yourself with:");
            for line in start_commands(init, &service.path) {
                println!("    {line}");
            }
            return Ok(());
        }
    }

    println!();
    if init == Init::Systemd && !lingering() {
        // A user unit stops when the last session for that user ends, which on
        // a server is the moment you close SSH — exactly when a daemon is most
        // expected to keep going. Lingering is the fix, and it is not obvious.
        println!("It stops when you log out. To keep it running across reboots");
        println!("without logging in:");
        println!();
        println!("    sudo loginctl enable-linger {user}");
        println!();
    }
    println!("    http://localhost:{port}             the web interface");
    println!("    http://localhost:{port}/scheduler   scheduled jobs");
    println!("    ozgent daemon status                is it running");
    Ok(())
}

/// The commands to install a root-owned service, for an init with no user ones.
fn root_install(init: Init, from: &Path, to: &Path) -> Vec<String> {
    let (from, to) = (from.display(), to.display());
    match init {
        Init::OpenRc => vec![
            format!("sudo install -Dm755 {from} {to}"),
            format!("sudo rc-update add {SERVICE} default && sudo rc-service {SERVICE} start"),
        ],
        Init::Runit => vec![
            format!("sudo install -Dm755 {from} {to}"),
            // Void uses /var/service; other runit distributions use
            // /etc/service. Whichever exists is the right one.
            format!(
                "sudo ln -s /etc/sv/{SERVICE} \
                 \"$([ -d /var/service ] && echo /var/service || echo /etc/service)/\""
            ),
        ],
        Init::S6 => vec![
            format!("sudo install -Dm755 {from} {to}"),
            format!("sudo s6-rc-bundle-update add default {SERVICE}  # or your distribution's equivalent"),
        ],
        _ => vec![format!("sudo install -Dm755 {from} {to}")],
    }
}

/// Turn the service on and start it now.
fn enable(init: Init, path: &Path) -> Result<()> {
    match init {
        Init::Systemd => {
            systemctl(&["daemon-reload"])?;
            systemctl(&["enable", "--now", &format!("{SERVICE}.service")])
        }
        Init::Launchd => {
            let uid = users_id();
            // `bootstrap` is the modern spelling; `load` is what older macOS
            // has. Trying the new one first means no deprecation warning on a
            // current system and no failure on an old one.
            let target = format!("gui/{uid}");
            if launchctl(&["bootstrap", &target, &path.display().to_string()]).is_err() {
                launchctl(&["load", "-w", &path.display().to_string()])?;
            }
            Ok(())
        }
        Init::Dinit => {
            exec("dinitctl", &["enable", SERVICE])?;
            // Already running is not a failure worth reporting.
            let _ = exec("dinitctl", &["start", SERVICE]);
            Ok(())
        }
        _ => anyhow::bail!("{} services are installed by hand", init.name()),
    }
}

fn start_commands(init: Init, path: &Path) -> Vec<String> {
    match init {
        Init::Systemd => vec![format!("systemctl --user enable --now {SERVICE}")],
        Init::Launchd => {
            vec![format!("launchctl bootstrap gui/$(id -u) {}", path.display())]
        }
        Init::Dinit => vec![format!("dinitctl enable {SERVICE}")],
        _ => vec![format!("see {}", path.display())],
    }
}

/// Stop it, turn it off, and remove the service file.
pub fn uninstall(paths: &Paths) -> Result<()> {
    let init = detect();
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("ozgent"));
    let Some(service) = service_file(init, &exe, paths, "127.0.0.1", 7333, &whoami()) else {
        println!("there is no service to remove");
        return Ok(());
    };
    if !service.path.exists() {
        println!("there is no ozgent service installed");
        if !init.per_user() {
            println!("If you installed one by hand, remove {}", service.path.display());
        }
        return Ok(());
    }
    if !init.per_user() {
        println!("{} services live under /etc, so removing one needs root:", init.name());
        println!();
        println!("    sudo rc-service {SERVICE} stop 2>/dev/null || true");
        println!("    sudo rm {}", service.path.display());
        return Ok(());
    }

    // Failures here are reported and not fatal: a service already stopped, or
    // never enabled, must not stop the file being removed.
    match init {
        Init::Systemd => {
            if let Err(e) = systemctl(&["disable", "--now", &format!("{SERVICE}.service")]) {
                eprintln!("warning: {e}");
            }
        }
        Init::Launchd => {
            let target = format!("gui/{}/com.ozgent.daemon", users_id());
            if launchctl(&["bootout", &target]).is_err() {
                let _ = launchctl(&["unload", "-w", &service.path.display().to_string()]);
            }
        }
        Init::Dinit => {
            let _ = exec("dinitctl", &["stop", SERVICE]);
            let _ = exec("dinitctl", &["disable", SERVICE]);
        }
        _ => {}
    }
    std::fs::remove_file(&service.path)
        .with_context(|| format!("removing {}", service.path.display()))?;
    if init == Init::Systemd {
        let _ = systemctl(&["daemon-reload"]);
    }
    println!("removed {}", service.path.display());
    println!("Your models, conversations and settings are untouched.");
    Ok(())
}

/// Say whether it is installed and running, and what it is doing.
pub fn status(paths: &Paths, config: &Config) -> Result<()> {
    let init = detect();
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("ozgent"));
    let service = service_file(init, &exe, paths, "127.0.0.1", 7333, &whoami());

    println!("init          {}", init.name());
    match &service {
        Some(s) if s.path.exists() => println!("service       {}", s.path.display()),
        _ => {
            println!("service       not installed");
            println!();
            println!("Install it with:  ozgent daemon install");
            println!("Or run it here:   ozgent daemon");
            return Ok(());
        }
    }

    if let Some(state) = running_state(init) {
        println!("state         {state}");
    }
    if init == Init::Systemd {
        println!(
            "lingering     {}",
            if lingering() { "on" } else { "off — it stops when you log out" }
        );
    }
    match config.web.idle_unload_minutes {
        0 => println!("idle          the model is kept loaded"),
        n => println!("idle          the model is dropped after {n} minutes"),
    }

    // Whether it is answering the channels and running jobs is not something
    // an init system knows; the lock files are what actually decide.
    println!(
        "scheduler     {}",
        held(&paths.root().join("scheduler.lock"))
            .then_some("running jobs")
            .unwrap_or("not running")
    );
    println!(
        "channels      {}",
        held(&paths.root().join("channels/gateway.lock"))
            .then_some("answered here")
            .unwrap_or("not answered")
    );
    println!();
    println!("logs          {}", log_hint(init, paths));
    Ok(())
}

fn running_state(init: Init) -> Option<String> {
    match init {
        Init::Systemd => {
            let active = output("systemctl", &["--user", "is-active", SERVICE])?;
            let enabled = output("systemctl", &["--user", "is-enabled", SERVICE])
                .unwrap_or_else(|| "unknown".into());
            Some(format!("{active} ({enabled} at login)"))
        }
        Init::Launchd => {
            let out = output("launchctl", &["list"])?;
            Some(if out.contains("com.ozgent.daemon") { "running".into() } else { "stopped".into() })
        }
        Init::Dinit => output("dinitctl", &["status", SERVICE]).map(|s| {
            s.lines().find(|l| l.contains("State")).unwrap_or(&s).trim().to_string()
        }),
        _ => None,
    }
}

fn log_hint(init: Init, paths: &Paths) -> String {
    match init {
        Init::Systemd => format!("journalctl --user -u {SERVICE} -f"),
        _ => format!("tail -f {}/logs/daemon.log", paths.root().display()),
    }
}

/// Whether some process holds this lock file.
fn held(path: &Path) -> bool {
    let Ok(file) = std::fs::OpenOptions::new().read(true).write(true).open(path) else {
        return false;
    };
    // Taking it means nobody had it. Give it straight back.
    match file.try_lock() {
        Ok(()) => {
            let _ = file.unlock();
            false
        }
        Err(_) => true,
    }
}

// --------------------------------------------------------------- helpers

fn whoami() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "root".into())
}

fn users_id() -> String {
    output("id", &["-u"]).unwrap_or_else(|| "1000".into())
}

/// Whether this user's services keep running after they log out.
fn lingering() -> bool {
    let user = whoami();
    output("loginctl", &["show-user", &user, "--property=Linger"])
        .map(|o| o.contains("Linger=yes"))
        .unwrap_or(false)
}

fn systemctl(args: &[&str]) -> Result<()> {
    let mut all = vec!["--user"];
    all.extend_from_slice(args);
    exec("systemctl", &all)
}

fn launchctl(args: &[&str]) -> Result<()> {
    exec("launchctl", args)
}

fn exec(program: &str, args: &[&str]) -> Result<()> {
    let status = std::process::Command::new(program)
        .args(args)
        .status()
        .with_context(|| format!("running {program}"))?;
    if !status.success() {
        anyhow::bail!("{program} {} failed", args.join(" "));
    }
    Ok(())
}

/// A query whose answer is its output. These exit non-zero for perfectly
/// ordinary answers — `is-active` on a stopped unit — so the status is ignored
/// and the text is what matters.
fn output(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms)?;
    }
    Ok(())
}

fn which(program: &str) -> Option<PathBuf> {
    std::env::var_os("PATH")?
        .to_string_lossy()
        .split(':')
        .map(|dir| Path::new(dir).join(program))
        .find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths() -> Paths {
        Paths::with_root("/home/someone/ozgent")
    }

    fn built(init: Init) -> Service {
        service_file(
            init,
            Path::new("/usr/local/bin/ozgent"),
            &paths(),
            "127.0.0.1",
            7333,
            "someone",
        )
        .expect("every init but Unknown produces a file")
    }

    const REAL: [Init; 6] =
        [Init::Systemd, Init::Launchd, Init::OpenRc, Init::Runit, Init::S6, Init::Dinit];

    #[test]
    fn every_init_system_gets_a_service_that_starts_the_daemon() {
        // The thing that must be true of all of them, however differently they
        // spell it: the right binary, the right subcommand, the right address.
        for init in REAL {
            let s = built(init);
            assert!(s.contents.contains("/usr/local/bin/ozgent"), "{}: {}", init.name(), s.contents);
            assert!(s.contents.contains("daemon"), "{}", init.name());
            assert!(s.contents.contains("7333"), "{}", init.name());
            assert!(s.contents.contains("127.0.0.1"), "{}", init.name());
        }
    }

    #[test]
    fn every_service_names_the_ozgent_directory_rather_than_inheriting_it() {
        // Otherwise the daemon reads a different directory from the terminal
        // that installed it, and the jobs you can see are not the ones running.
        for init in REAL {
            let s = built(init);
            assert!(
                s.contents.contains("/home/someone/ozgent"),
                "{} does not set OZGENT_HOME: {}",
                init.name(),
                s.contents
            );
        }
    }

    #[test]
    fn a_root_owned_service_drops_to_the_user_who_owns_the_data() {
        // OpenRC, runit and s6 run as root. A daemon reading one person's
        // conversations as root is a worse arrangement, not a more serious one.
        for init in [Init::OpenRc, Init::Runit, Init::S6] {
            let s = built(init);
            assert!(s.contents.contains("someone"), "{} stays root: {}", init.name(), s.contents);
        }
    }

    #[test]
    fn a_root_owned_service_file_is_executable_and_a_user_one_need_not_be() {
        for init in [Init::OpenRc, Init::Runit, Init::S6] {
            assert!(built(init).executable, "{} must be executable", init.name());
        }
        for init in [Init::Systemd, Init::Launchd, Init::Dinit] {
            assert!(!built(init).executable, "{} is not a script", init.name());
        }
    }

    #[test]
    fn a_user_service_goes_under_the_home_directory_and_a_system_one_under_etc() {
        let home = dirs::home_dir().unwrap();
        for init in [Init::Systemd, Init::Launchd, Init::Dinit] {
            assert!(
                built(init).path.starts_with(&home),
                "{} must not need root: {}",
                init.name(),
                built(init).path.display()
            );
            assert!(init.per_user(), "{}", init.name());
        }
        for init in [Init::OpenRc, Init::Runit, Init::S6] {
            assert!(built(init).path.starts_with("/etc"), "{}", init.name());
            assert!(!init.per_user(), "{}", init.name());
        }
    }

    #[test]
    fn an_unrecognised_init_produces_no_service_rather_than_a_wrong_one() {
        assert!(
            service_file(Init::Unknown, Path::new("/x"), &paths(), "127.0.0.1", 7333, "u")
                .is_none()
        );
    }

    #[test]
    fn the_systemd_unit_restarts_on_failure_but_gives_up_on_a_loop() {
        let unit = built(Init::Systemd).contents;
        assert!(unit.contains("Restart=on-failure"), "{unit}");
        assert!(unit.contains("StartLimitBurst="), "{unit}");
    }

    #[test]
    fn the_systemd_unit_waits_for_the_network_and_starts_at_login() {
        let unit = built(Init::Systemd).contents;
        assert!(unit.contains("After=network-online.target"), "{unit}");
        assert!(unit.contains("WantedBy=default.target"), "{unit}");
    }

    #[test]
    fn no_service_caps_memory_or_deprioritises_the_process() {
        // A model load is a single enormous allocation: a MemoryMax that looks
        // generous on one machine kills the load on another. And the daemon
        // serves the user's own interactive chat — nice-ing it makes their
        // typing slower to help nobody.
        for init in REAL {
            let c = built(init).contents;
            for forbidden in ["MemoryMax", "MemoryHigh", "Nice=", "CPUWeight="] {
                assert!(!c.contains(forbidden), "{} sets {forbidden}", init.name());
            }
        }
    }

    #[test]
    fn the_systemd_unit_is_valid_ini_with_the_three_sections_it_needs() {
        let unit = built(Init::Systemd).contents;
        for section in ["[Unit]", "[Service]", "[Install]"] {
            assert!(unit.contains(section), "missing {section}");
        }
        for line in unit.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }
            assert!(line.contains('='), "not a setting: {line:?}");
        }
    }

    #[test]
    fn the_launchd_plist_is_well_formed_xml_with_balanced_tags() {
        let plist = built(Init::Launchd).contents;
        assert!(plist.starts_with("<?xml"), "{plist}");
        assert!(plist.trim_end().ends_with("</plist>"), "{plist}");
        for tag in ["dict", "array", "plist"] {
            let open = plist.matches(&format!("<{tag}>")).count()
                + plist.matches(&format!("<{tag} ")).count();
            let close = plist.matches(&format!("</{tag}>")).count();
            assert_eq!(open, close, "<{tag}> is unbalanced in:\n{plist}");
        }
        // Each argument is its own <string>, or launchd passes one long word.
        assert!(plist.contains("<string>daemon</string>"), "{plist}");
        assert!(plist.contains("<string>--host</string>"), "{plist}");
    }

    #[test]
    fn the_shell_based_services_start_with_a_shebang() {
        // A run script without one is not executed, it is read as shell by
        // whatever happens to be running — which sometimes works and sometimes
        // does something else entirely.
        assert!(built(Init::OpenRc).contents.starts_with("#!/sbin/openrc-run"));
        assert!(built(Init::Runit).contents.starts_with("#!/bin/sh"));
        assert!(built(Init::S6).contents.starts_with("#!/bin/execlineb"));
    }

    #[test]
    fn the_runit_service_replaces_the_shell_rather_than_forking() {
        // runit supervises the process it started. A run script that forks and
        // exits looks to runit like a service that keeps crashing.
        let script = built(Init::Runit).contents;
        assert!(script.contains("exec chpst"), "{script}");
        // A trailing `&` is what backgrounding looks like; `exec 2>&1` is a
        // redirect and is exactly what runit wants.
        for line in script.lines() {
            assert!(!line.trim_end().ends_with('&'), "backgrounded: {line:?}");
        }
    }

    #[test]
    fn the_openrc_service_declares_what_it_needs() {
        let script = built(Init::OpenRc).contents;
        assert!(script.contains("need net"), "{script}");
        assert!(script.contains("command_background=true"), "{script}");
        assert!(script.contains("pidfile="), "openrc needs one when backgrounding");
    }

    #[test]
    fn the_root_install_commands_name_both_the_source_and_the_destination() {
        for init in [Init::OpenRc, Init::Runit, Init::S6] {
            let lines =
                root_install(init, Path::new("/home/someone/ozgent/service/ozgent"), &built(init).path);
            assert!(!lines.is_empty(), "{}", init.name());
            let all = lines.join("\n");
            assert!(all.contains("/home/someone/ozgent/service/"), "{}: {all}", init.name());
            assert!(all.contains("sudo"), "{}: {all}", init.name());
        }
    }

    #[test]
    fn detection_returns_something_usable_on_this_machine() {
        // Whatever this machine is, detection must not panic and must name
        // something. On the developer's machine and in CI this is systemd.
        let init = detect();
        assert!(!init.name().is_empty());
    }

    #[test]
    fn a_lock_nobody_holds_reads_as_free_and_a_held_one_as_held() {
        let dir = std::env::temp_dir().join(format!("ozgent-daemon-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&path)
            .unwrap();
        assert!(!held(&path), "nobody holds it yet");
        file.try_lock().unwrap();
        assert!(held(&path));
        drop(file);
        assert!(!held(&path));
        // A file that does not exist is not a held lock either.
        assert!(!held(&dir.join("missing.lock")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
