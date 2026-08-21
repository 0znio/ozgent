//! The Rust host driving a real Python worker over a real pipe.
//!
//! This is the integration that the whole tool design rests on, so it is
//! tested against the actual interpreter rather than a mock.

use ozgent_tools::{HostConfig, ToolHost};
use serde_json::json;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn runtime_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python")
}

/// Writes a scratch tool directory that is removed when the guard drops.
struct ToolDir(PathBuf);

impl ToolDir {
    fn new(label: &str, files: &[(&str, &str)]) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "ozgent-tools-{}-{}-{:?}",
            label,
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, body) in files {
            std::fs::write(dir.join(name), body).unwrap();
        }
        Self(dir)
    }
}

impl Drop for ToolDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const DEMO: &str = r#"
import asyncio
from typing import Annotated, Literal
from ozgent_tools import tool, ToolError, get_config

@tool
def add(a: int, b: int = 1, mode: Literal["sum", "diff"] = "sum") -> int:
    """Add two numbers."""
    print("a stray print must not corrupt the protocol")
    return a + b if mode == "sum" else a - b

@tool
async def sleeper(seconds: float) -> str:
    """Sleep, to exercise timeouts and cancellation."""
    await asyncio.sleep(seconds)
    return "finished"

@tool
def boom() -> str:
    """Raise an unexpected exception."""
    raise RuntimeError("kaboom")

@tool
def refuse() -> str:
    """Raise a ToolError the model should see."""
    raise ToolError("cannot do that", retryable=True)

@tool
def echo_config() -> dict:
    """Return this tool's configured settings."""
    return get_config("echo_config")
"#;

async fn host_with(label: &str, timeout: Duration, config: serde_json::Value) -> (ToolHost, ToolDir) {
    let dir = ToolDir::new(label, &[("demo.py", DEMO)]);
    let host = ToolHost::start(HostConfig {
        python: "python3".into(),
        runtime_path: runtime_path(),
        tool_paths: vec![dir.0.clone()],
        disabled: Vec::new(),
        tool_config: config,
        timeout,
        ..Default::default()
    })
    .await
    .expect("worker should start");
    (host, dir)
}

#[tokio::test]
async fn discovers_builtin_and_user_tools() {
    let (host, _d) = host_with("discover", Duration::from_secs(30), json!({})).await;

    let names: Vec<&str> = host.tools().iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"web_search"), "built-ins must load: {names:?}");
    assert!(names.contains(&"add"), "user tools must load: {names:?}");
    assert!(host.load_errors().is_empty(), "{:?}", host.load_errors());
    assert!(!host.python_version().is_empty());

    host.shutdown().await;
}

#[tokio::test]
async fn schema_derived_from_python_signature_reaches_rust() {
    let (host, _d) = host_with("schema", Duration::from_secs(30), json!({})).await;

    let add = host.get("add").expect("add should be registered");
    assert_eq!(add.description, "Add two numbers.");
    assert_eq!(add.input_schema["properties"]["a"]["type"], "integer");
    assert_eq!(add.input_schema["properties"]["b"]["default"], 1);
    assert_eq!(add.input_schema["properties"]["mode"]["enum"][0], "sum");
    assert_eq!(add.input_schema["required"][0], "a");

    let search = host.get("web_search").expect("web_search should be registered");
    assert!(search.output_schema.is_some(), "web_search declares an output schema");

    host.shutdown().await;
}

#[tokio::test]
async fn the_model_cannot_choose_the_search_provider() {
    // Which engine runs a search is the user's configuration. It was once a
    // tool parameter, and models overrode a configured Brave key with
    // duckduckgo — searching somewhere the user never asked for. The guard is
    // structural: no property to set, and `additionalProperties: false` so a
    // model that invents one is rejected rather than quietly obeyed.
    let (host, _d) = host_with("provider", Duration::from_secs(30), json!({})).await;

    let search = host.get("web_search").expect("web_search should be registered");
    let properties = &search.input_schema["properties"];
    assert!(
        properties.get("provider").is_none(),
        "provider must not be exposed to the model: {properties}"
    );
    assert_eq!(
        search.input_schema["additionalProperties"], false,
        "an invented provider argument must be refused, not ignored"
    );

    host.shutdown().await;
}

#[tokio::test]
async fn calls_a_tool_and_gets_a_typed_result() {
    let (host, _d) = host_with("call", Duration::from_secs(30), json!({})).await;

    let out = host.call("add", json!({ "a": 2, "b": 40 })).await.unwrap();
    assert_eq!(out, json!(42));

    let defaulted = host.call("add", json!({ "a": 10 })).await.unwrap();
    assert_eq!(defaulted, json!(11), "python-side default must apply");

    host.shutdown().await;
}

#[tokio::test]
async fn bad_arguments_are_rejected_as_the_models_fault() {
    let (host, _d) = host_with("badargs", Duration::from_secs(30), json!({})).await;

    let err = host.call("add", json!({ "a": 1, "mode": "sideways" })).await.unwrap_err();
    let message = err.for_model();
    assert!(message.contains("sideways"), "{message}");
    assert!(!message.contains("Do not retry"), "the model can fix this: {message}");

    host.shutdown().await;
}

#[tokio::test]
async fn a_tool_crash_is_reported_and_the_worker_survives() {
    let (host, _d) = host_with("crash", Duration::from_secs(30), json!({})).await;

    let err = host.call("boom", json!({})).await.unwrap_err();
    let message = err.for_model();
    assert!(message.contains("RuntimeError"), "{message}");
    assert!(message.contains("Do not retry"), "a tool bug is not the model's to retry: {message}");

    // The crucial property: inference continues after a tool blows up.
    let after = host.call("add", json!({ "a": 1, "b": 1 })).await.unwrap();
    assert_eq!(after, json!(2), "worker must survive a crashing tool");

    host.shutdown().await;
}

#[tokio::test]
async fn tool_errors_carry_the_retryable_hint() {
    let (host, _d) = host_with("refuse", Duration::from_secs(30), json!({})).await;

    let err = host.call("refuse", json!({})).await.unwrap_err();
    let message = err.for_model();
    assert!(message.contains("cannot do that"), "{message}");
    assert!(message.contains("retried"), "retryable hint should reach the model: {message}");

    host.shutdown().await;
}

#[tokio::test]
async fn a_hanging_tool_times_out_and_is_cancelled() {
    let (host, _d) = host_with("timeout", Duration::from_millis(600), json!({})).await;

    let started = Instant::now();
    let err = host.call("sleeper", json!({ "seconds": 30 })).await.unwrap_err();
    let elapsed = started.elapsed();

    assert!(err.for_model().contains("timed out"), "{}", err.for_model());
    assert!(elapsed < Duration::from_secs(3), "timeout was not enforced: {elapsed:?}");

    // The worker is still usable, and the cancelled tool is no longer running.
    let after = host.call("add", json!({ "a": 5 })).await.unwrap();
    assert_eq!(after, json!(6));

    host.shutdown().await;
}

#[tokio::test]
async fn concurrent_calls_overlap() {
    let (host, _d) = host_with("concurrent", Duration::from_secs(30), json!({})).await;
    let host = std::sync::Arc::new(host);

    let started = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..4 {
        let h = std::sync::Arc::clone(&host);
        handles.push(tokio::spawn(async move {
            h.call("sleeper", json!({ "seconds": 0.5 })).await
        }));
    }
    for handle in handles {
        assert_eq!(handle.await.unwrap().unwrap(), json!("finished"));
    }
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_millis(1500),
        "four 0.5s calls serialised instead of overlapping: {elapsed:?}"
    );

    host.shutdown().await;
}

#[tokio::test]
async fn config_from_toml_reaches_the_python_tool() {
    let config = json!({ "echo_config": { "provider": "brave", "max_results": 7 } });
    let (host, _d) = host_with("config", Duration::from_secs(30), config).await;

    let out = host.call("echo_config", json!({})).await.unwrap();
    assert_eq!(out["provider"], "brave");
    assert_eq!(out["max_results"], 7);

    host.shutdown().await;
}

#[tokio::test]
async fn unknown_tool_is_rejected_with_the_available_list() {
    let (host, _d) = host_with("unknown", Duration::from_secs(30), json!({})).await;

    let err = host.call("teleport", json!({})).await.unwrap_err();
    assert!(err.for_model().contains("web_search"), "{}", err.for_model());

    host.shutdown().await;
}

#[tokio::test]
async fn a_broken_tool_file_is_reported_without_blocking_the_rest() {
    let dir = ToolDir::new(
        "broken",
        &[("good.py", DEMO), ("bad.py", "this is not python !!!\n")],
    );
    let host = ToolHost::start(HostConfig {
        python: "python3".into(),
        runtime_path: runtime_path(),
        tool_paths: vec![dir.0.clone()],
        timeout: Duration::from_secs(30),
        ..Default::default()
    })
    .await
    .unwrap();

    assert!(host.get("add").is_some(), "a broken sibling must not block a good tool");
    assert!(
        host.load_errors().iter().any(|e| e.contains("bad.py")),
        "the failure must be surfaced: {:?}",
        host.load_errors()
    );

    host.shutdown().await;
}

#[tokio::test]
async fn disabled_tools_are_not_offered_to_the_model() {
    let dir = ToolDir::new("disabled", &[("demo.py", DEMO)]);
    let host = ToolHost::start(HostConfig {
        python: "python3".into(),
        runtime_path: runtime_path(),
        tool_paths: vec![dir.0.clone()],
        disabled: vec!["boom".into(), "web_search".into()],
        timeout: Duration::from_secs(30),
        ..Default::default()
    })
    .await
    .unwrap();

    assert!(host.get("boom").is_none());
    assert!(host.get("web_search").is_none());
    assert!(host.get("add").is_some());

    host.shutdown().await;
}

// ------------------------------------------------------------- file tools

/// A file big enough to force selection, where the answer to the obvious
/// question lives in a function the question does not name.
///
/// `render_invoice` is what someone would search for; the rounding rule they
/// actually want is in `apply_tax`, which shares none of the query's words.
/// Padding keeps the file over the read budget so the selection path runs.
fn big_source() -> String {
    let mut s = String::from(
        "import os\n\n\
         def unrelated_alpha(x):\n    return x + 1\n\n\
         def apply_tax(amount):\n    # Rounds half up, to the cent.\n    return round(amount * 1.2, 2)\n\n\
         def render_invoice(items):\n    total = sum(i.price for i in items)\n    return apply_tax(total)\n\n",
    );
    for i in 0..200 {
        s.push_str(&format!("def filler_{i}(n):\n    return n * {i}\n\n"));
    }
    s
}

#[tokio::test]
async fn a_small_file_is_returned_whole() {
    let dir = ToolDir::new("read-small", &[("small.py", "def hello():\n    return 1\n")]);
    let (host, _d) = host_with(
        "read-small-host",
        Duration::from_secs(30),
        json!({ "read_file": { "root": dir.0.to_str().unwrap() } }),
    )
    .await;

    let out = host
        .call("read_file", json!({ "path": "small.py" }))
        .await
        .expect("read should succeed");
    assert_eq!(out["mode"], "whole");
    assert_eq!(out["omitted"], 0);
    assert!(out["content"].as_str().unwrap().contains("def hello"));
    host.shutdown().await;
}

#[tokio::test]
async fn a_large_file_returns_the_relevant_part_and_what_it_depends_on() {
    // The point of the tool: searching for the function you know about must
    // also bring back the one it calls, or the answer is not in the excerpt.
    let dir = ToolDir::new("read-big", &[("big.py", big_source().as_str())]);
    let (host, _d) = host_with(
        "read-big-host",
        Duration::from_secs(30),
        json!({ "read_file": { "root": dir.0.to_str().unwrap(), "max_lines": 60 } }),
    )
    .await;

    let out = host
        .call("read_file", json!({ "path": "big.py", "query": "render_invoice" }))
        .await
        .expect("read should succeed");

    assert_eq!(out["mode"], "relevant");
    let content = out["content"].as_str().unwrap();
    assert!(content.contains("def render_invoice"), "the match itself: {content}");
    assert!(
        content.contains("def apply_tax"),
        "the dependency must be pulled in even though the query never named it: {content}"
    );
    assert!(out["omitted"].as_i64().unwrap() > 0, "a large file must omit something");
    assert!(
        !content.contains("def filler_150"),
        "irrelevant definitions must stay out"
    );
    host.shutdown().await;
}

#[tokio::test]
async fn a_large_file_without_a_query_offers_an_outline() {
    let dir = ToolDir::new("read-outline", &[("big.py", big_source().as_str())]);
    let (host, _d) = host_with(
        "read-outline-host",
        Duration::from_secs(30),
        json!({ "read_file": { "root": dir.0.to_str().unwrap(), "max_lines": 40 } }),
    )
    .await;

    let out = host.call("read_file", json!({ "path": "big.py" })).await.unwrap();
    assert_eq!(out["mode"], "head");
    let names: Vec<&str> = out["outline"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e["name"].as_str())
        .collect();
    assert!(names.contains(&"apply_tax"), "outline should name every definition: {names:?}");
    host.shutdown().await;
}

#[tokio::test]
async fn reads_outside_the_configured_root_are_refused() {
    // Traversal is checked after resolution, so `../` cannot escape.
    let dir = ToolDir::new("read-escape", &[("ok.py", "x = 1\n")]);
    let (host, _d) = host_with(
        "read-escape-host",
        Duration::from_secs(30),
        json!({ "read_file": { "root": dir.0.to_str().unwrap() } }),
    )
    .await;

    let err = host
        .call("read_file", json!({ "path": "../../../../etc/passwd" }))
        .await
        .expect_err("must refuse to leave the root");
    assert!(err.for_model().contains("outside"), "{}", err.for_model());
    host.shutdown().await;
}

#[tokio::test]
async fn writing_is_refused_unless_it_is_switched_on() {
    let dir = ToolDir::new("write-off", &[]);
    let (host, _d) = host_with(
        "write-off-host",
        Duration::from_secs(30),
        json!({ "write_file": { "root": dir.0.to_str().unwrap() } }),
    )
    .await;

    let err = host
        .call("write_file", json!({ "path": "new.txt", "content": "hi" }))
        .await
        .expect_err("writing must be opt-in");
    assert!(err.for_model().contains("disabled"), "{}", err.for_model());
    assert!(!dir.0.join("new.txt").exists(), "nothing may be written");
    host.shutdown().await;
}

#[tokio::test]
async fn writing_creates_a_file_and_refuses_to_clobber_one() {
    let dir = ToolDir::new("write-on", &[]);
    let (host, _d) = host_with(
        "write-on-host",
        Duration::from_secs(30),
        json!({ "write_file": { "root": dir.0.to_str().unwrap(), "enabled": true } }),
    )
    .await;

    let out = host
        .call("write_file", json!({ "path": "notes.md", "content": "one\ntwo\n" }))
        .await
        .expect("create should succeed");
    assert_eq!(out["lines_written"], 2);
    assert_eq!(std::fs::read_to_string(dir.0.join("notes.md")).unwrap(), "one\ntwo\n");

    // A second create must not silently replace the first.
    let err = host
        .call("write_file", json!({ "path": "notes.md", "content": "clobber" }))
        .await
        .expect_err("create must not overwrite");
    assert!(err.for_model().contains("already exists"), "{}", err.for_model());
    assert_eq!(std::fs::read_to_string(dir.0.join("notes.md")).unwrap(), "one\ntwo\n");

    // Overwrite is available when asked for explicitly.
    host.call(
        "write_file",
        json!({ "path": "notes.md", "content": "replaced\n", "mode": "overwrite" }),
    )
    .await
    .expect("explicit overwrite should succeed");
    assert_eq!(std::fs::read_to_string(dir.0.join("notes.md")).unwrap(), "replaced\n");
    host.shutdown().await;
}
