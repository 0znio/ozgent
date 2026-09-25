//! A stdio server inside ozgent's sandbox, trying to get out.
//!
//! The fixture server gains a `probe` tool (with `FIXTURE_PROBE=1`) that
//! reads, writes, connects and looks at its environment on request. Run
//! sandboxed, it must reach its own folder and nothing else: not ozgent's
//! home, not a folder beside the one it was given, not the network, not a
//! secret in ozgent's own environment.

use std::path::{Path, PathBuf};

use ozgent_core::mcp;
use ozgent_mcp::{Launcher, Server};
use ozgent_tools::ToolSource;
use serde_json::{Value, json};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Landlock is what does the confining; without it the launcher refuses to
/// run anything, which is a different test.
fn landlock() -> bool {
    let probe = format!(
        "import sys; sys.path.insert(0, {:?}); from ozgent_tools import sandbox; sys.exit(0 if sandbox.landlock_abi() > 0 else 1)",
        repo().join("python").display().to_string()
    );
    std::process::Command::new("python3").args(["-c", &probe]).status().is_ok_and(|s| s.success())
}

/// A directory removed when the test ends.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir()
            .join(format!("ozgent-mcp-sandbox-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Scene {
    _dir: TempDir,
    ozgent_home: PathBuf,
    folder: PathBuf,
    beside: PathBuf,
    launcher: Launcher,
}

fn scene() -> Scene {
    let dir = TempDir::new();
    let root = dir.path().canonicalize().unwrap();
    let ozgent_home = root.join("ozgent");
    std::fs::create_dir_all(ozgent_home.join("configs")).unwrap();
    std::fs::write(ozgent_home.join("configs/config.toml"), "token = 'secret'\n").unwrap();
    let folder = root.join("notes");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join("allowed.txt"), "you may read this").unwrap();
    // Run from inside the folder it may use: the repository is not one.
    std::fs::copy(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/server.py"),
        folder.join("server.py"),
    )
    .unwrap();
    let beside = root.join("private");
    std::fs::create_dir_all(&beside).unwrap();
    std::fs::write(beside.join("secret.txt"), "not for servers").unwrap();
    let launcher = Launcher {
        python: "python3".into(),
        runtime_path: repo().join("python"),
        homes: ozgent_home.join("mcp"),
        protected: vec![ozgent_home.clone()],
    };
    Scene { _dir: dir, ozgent_home, folder, beside, launcher }
}

fn settings(folder: &Path, network: bool) -> mcp::Server {
    mcp::Server {
        command: Some("python3".into()),
        args: vec![folder.join("server.py").display().to_string()],
        env: [("FIXTURE_PROBE".to_string(), "1".to_string())].into(),
        sandbox: true,
        network,
        folders: vec![folder.to_path_buf()],
        timeout_seconds: 20,
        ..Default::default()
    }
}

async fn probe(server: &Server, args: Value) -> Value {
    server.call("fix_probe", args, false).await.expect("the probe answers")
}

#[tokio::test]
async fn a_sandboxed_server_reaches_its_folder_and_nothing_else() {
    if !landlock() {
        eprintln!("skipping: no Landlock here");
        return;
    }
    let s = scene();
    // A secret in ozgent's own environment must not reach the server.
    // SAFETY: test-only; nothing else reads this variable.
    unsafe { std::env::set_var("OZGENT_TEST_API_KEY", "must-not-leak") };
    let (server, _) = Server::connect_with("fix", &settings(&s.folder, false), Some(&s.launcher), Default::default())
        .await
        .expect("the fixture starts in the sandbox");

    let out = probe(
        &server,
        json!({
            "read": [
                s.folder.join("allowed.txt"),
                s.ozgent_home.join("configs/config.toml"),
                s.beside.join("secret.txt"),
            ],
            // With its own /proc writable, the system's settings must still
            // be out of reach: they need privileges it does not have.
            "write": [s.folder.join("made.txt"), s.beside.join("escaped.txt"),
                      PathBuf::from("/proc/sys/kernel/hostname"), PathBuf::from("/proc/sysrq-trigger")],
            "connect": [["1.1.1.1", 443]],
            "udp": [["1.1.1.1", 53]],
            "env": ["OZGENT_TEST_API_KEY", "FIXTURE_PROBE"],
        }),
    )
    .await;
    let get = |k: String| out[&k].as_str().unwrap_or("").to_string();

    assert!(get(format!("read {}", s.folder.join("allowed.txt").display())).starts_with("ok"), "{out}");
    assert!(get(format!("read {}", s.ozgent_home.join("configs/config.toml").display())).starts_with("refused"), "{out}");
    assert!(get(format!("read {}", s.beside.join("secret.txt").display())).starts_with("refused"), "{out}");
    assert!(get(format!("write {}", s.folder.join("made.txt").display())).starts_with("ok"), "{out}");
    assert!(get(format!("write {}", s.beside.join("escaped.txt").display())).starts_with("refused"), "{out}");
    assert!(!s.beside.join("escaped.txt").exists());
    assert!(get("write /proc/sys/kernel/hostname".into()).starts_with("refused"), "{out}");
    assert!(get("write /proc/sysrq-trigger".into()).starts_with("refused"), "{out}");
    assert!(get("connect 1.1.1.1:443".into()).starts_with("refused"), "{out}");
    assert!(get("udp 1.1.1.1:53".into()).starts_with("refused"), "{out}");
    assert_eq!(get("env OZGENT_TEST_API_KEY".into()), "<unset>", "{out}");
    assert_eq!(get("env FIXTURE_PROBE".into()), "1", "its configured environment arrives: {out}");
    // It lives in a home of its own, under ozgent's.
    assert_eq!(get("home".into()), s.launcher.home("fix").display().to_string(), "{out}");
    server.shutdown().await;
}

/// Whether the sandbox gets namespaces here: without them nothing can be
/// hidden behind a private mount (stock Ubuntu 24.04, most CI).
fn namespaces() -> bool {
    let probe = format!(
        "import sys; sys.path.insert(0, {:?}); from ozgent_tools import sandbox; sys.exit(0 if sandbox.private_proc_available() else 1)",
        repo().join("python").display().to_string()
    );
    std::process::Command::new("python3").args(["-c", &probe]).status().is_ok_and(|s| s.success())
}

#[tokio::test]
async fn the_system_is_readable_and_the_session_is_not() {
    if !landlock() {
        eprintln!("skipping: no Landlock here");
        return;
    }
    let s = scene();
    let (server, _) = Server::connect_with("fix", &settings(&s.folder, true), Some(&s.launcher), Default::default())
        .await
        .expect("the fixture starts in the sandbox");
    let home = std::env::var("HOME").unwrap_or_default();
    let dbus = "/run/dbus/system_bus_socket";
    let out = probe(
        &server,
        json!({
            // What programs look for and an allowlist kept missing: DNS
            // settings (on Ubuntu a link into /run), hardware under /sys.
            "read": ["/etc/resolv.conf", "/sys/devices/system/cpu/online", format!("{home}/.bashrc")],
            "list": ["/usr/lib", format!("{home}")],
            "unix": [dbus],
        }),
    )
    .await;
    let get = |k: String| out[&k].as_str().unwrap_or("").to_string();
    if std::path::Path::new("/etc/resolv.conf").exists() {
        assert!(get("read /etc/resolv.conf".into()).starts_with("ok"), "{out}");
    }
    assert!(get("read /sys/devices/system/cpu/online".into()).starts_with("ok"), "{out}");
    assert!(get("list /usr/lib".into()).starts_with("ok"), "{out}");
    // People's homes stay shut.
    assert!(get(format!("list {home}")).starts_with("refused"), "{out}");
    assert!(!get(format!("read {home}/.bashrc")).starts_with("ok"), "{out}");
    // The system bus socket: Landlock does not stop a connect, the private
    // mount over /run/dbus does — where there is a mount namespace.
    if namespaces() && std::path::Path::new(dbus).exists() {
        assert!(get(format!("unix {dbus}")).starts_with("refused"), "{out}");
    }
    server.shutdown().await;
}

#[tokio::test]
async fn a_server_allowed_the_network_can_open_a_socket() {
    if !landlock() {
        eprintln!("skipping: no Landlock here");
        return;
    }
    let s = scene();
    let (server, _) = Server::connect_with("fix", &settings(&s.folder, true), Some(&s.launcher), Default::default())
        .await
        .expect("the fixture starts in the sandbox");
    // A UDP send needs no reachable host to succeed locally; it shows the
    // socket was allowed at all.
    let out = probe(&server, json!({ "udp": [["127.0.0.1", 9]] })).await;
    assert!(out["udp 127.0.0.1:9"].as_str().unwrap_or("").starts_with("ok"), "{out}");
    server.shutdown().await;
}

#[tokio::test]
async fn a_server_asking_for_the_sandbox_without_one_is_refused() {
    let s = scene();
    let result = Server::connect_with("fix", &settings(&s.folder, false), None, Default::default()).await;
    let Err(e) = result else { panic!("it should not start unsandboxed") };
    assert!(e.to_string().contains("sandbox"), "{e}");
}
