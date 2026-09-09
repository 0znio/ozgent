//! The client against a real server, over a real pipe.
//!
//! `tests/fixtures/server.py` speaks the actual protocol, so these cover the
//! framing, the ordering and the timeout as well as the parsing — the parts
//! that unit tests over JSON literals cannot reach.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ozgent_core::mcp;
use ozgent_core::permission::Effect;
use ozgent_mcp::Server;
use ozgent_tools::ToolSource;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/server.py")
}

fn settings() -> mcp::Server {
    mcp::Server {
        command: Some("python3".into()),
        args: vec![fixture().to_string_lossy().into_owned()],
        timeout_seconds: 5,
        ..Default::default()
    }
}

/// Skips rather than fails when python3 is missing; the fixture is a test
/// dependency, not something ozgent needs at runtime.
async fn connect(settings: mcp::Server) -> Option<Server> {
    if std::process::Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipping: python3 is not installed");
        return None;
    }
    Some(Server::connect("fix", &settings).await.expect("connecting to the fixture"))
}

#[tokio::test]
async fn the_handshake_reports_what_the_server_says_it_is() {
    let Some(server) = connect(settings()).await else { return };
    assert_eq!(server.info().name, "fixture");
    assert_eq!(server.info().version, "0.1.0");
    assert!(server.info().has_tools);
}

#[tokio::test]
async fn chatter_on_stdout_does_not_break_the_connection() {
    // The fixture prints a line before saying anything protocol-shaped, which
    // servers really do. Treating that as a fatal parse error would make the
    // whole server unusable over one stray print.
    assert!(connect(settings()).await.is_some());
}

#[tokio::test]
async fn every_page_of_the_tool_list_is_read() {
    let Some(server) = connect(settings()).await else { return };
    let names: Vec<&str> = server.tools().iter().map(|t| t.name.as_str()).collect();
    // Two from the first page, three usable from the second.
    assert_eq!(names, ["fix_echo", "fix_wipe", "fix_structured", "fix_explode", "fix_sleep"]);
}

#[tokio::test]
async fn a_tool_with_no_name_is_not_offered() {
    let Some(server) = connect(settings()).await else { return };
    assert!(server.tools().len() == 5, "{:?}", server.tools().len());
}

#[tokio::test]
async fn tools_are_named_after_the_server_they_came_from() {
    let Some(server) = connect(settings()).await else { return };
    assert!(server.tools().iter().all(|t| t.name.starts_with("fix_")));
    assert_eq!(server.origin(), "mcp:fix");
}

#[tokio::test]
async fn a_servers_own_claim_about_a_tool_is_ignored_by_default() {
    // The fixture marks `echo` read-only. Believed, that would make it run
    // without anyone being asked — on the say-so of the program that wants to
    // be run.
    let Some(server) = connect(settings()).await else { return };
    for tool in server.tools() {
        assert_eq!(tool.effect, Effect::Unknown, "{}", tool.name);
    }
}

#[tokio::test]
async fn a_trusted_servers_claim_is_used() {
    let trusted = mcp::Server { trust_hints: true, ..settings() };
    let Some(server) = connect(trusted).await else { return };
    let effect = |name: &str| server.tools().iter().find(|t| t.name == name).unwrap().effect;
    assert_eq!(effect("fix_echo"), Effect::Read);
    assert_eq!(effect("fix_wipe"), Effect::Write);
    // Said nothing either way, so still asked about.
    assert_eq!(effect("fix_structured"), Effect::Unknown);
}

#[tokio::test]
async fn a_call_reaches_the_tool_under_its_own_name() {
    // The model says `fix_echo`; the server only knows `echo`.
    let Some(server) = connect(settings()).await else { return };
    let out = server
        .call("fix_echo", serde_json::json!({ "text": "hello" }), false)
        .await
        .expect("the call");
    assert_eq!(out, serde_json::json!({ "text": "hello" }));
}

#[tokio::test]
async fn structured_output_arrives_as_data() {
    let Some(server) = connect(settings()).await else { return };
    let out = server.call("fix_structured", serde_json::json!({}), false).await.unwrap();
    assert_eq!(out["count"], 2);
    assert_eq!(out["items"][1], "b");
}

#[tokio::test]
async fn a_tool_that_reports_failure_reaches_the_model_as_a_result() {
    let Some(server) = connect(settings()).await else { return };
    let err = server.call("fix_explode", serde_json::json!({}), false).await.unwrap_err();
    let text = err.for_model();
    assert!(text.contains("it went wrong"), "{text}");
    // Phrased as something the model may act on, not as ozgent breaking.
    assert!(!text.contains("Do not retry"), "{text}");
}

#[tokio::test]
async fn calling_something_the_server_does_not_have_is_refused_locally() {
    let Some(server) = connect(settings()).await else { return };
    let err = server.call("fix_nonsense", serde_json::json!({}), false).await.unwrap_err();
    assert!(err.for_model().contains("does not offer"), "{}", err.for_model());
}

#[tokio::test]
async fn a_tool_that_never_answers_is_cancelled_rather_than_hanging() {
    // The inference thread is waiting on this; a call with no timeout would
    // take the whole session with it.
    let Some(server) = connect(settings()).await else { return };
    let started = std::time::Instant::now();
    let err = server.call("fix_sleep", serde_json::json!({}), false).await.unwrap_err();
    assert!(started.elapsed() < std::time::Duration::from_secs(20), "waited too long");
    assert!(err.for_model().contains("timed out"), "{}", err.for_model());
}

#[tokio::test]
async fn only_the_named_tools_are_offered_when_a_list_is_given() {
    let narrowed = mcp::Server { tools: Some(vec!["echo".into()]), ..settings() };
    let Some(server) = connect(narrowed).await else { return };
    let names: Vec<&str> = server.tools().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["fix_echo"]);
}

#[tokio::test]
async fn a_server_and_the_python_tools_merge_into_one_list() {
    // What everything above this layer actually sees.
    let Some(server) = connect(settings()).await else { return };
    let box_ = ozgent_tools::Toolbox::new(None, vec![std::sync::Arc::new(server)]);
    assert!(box_.get("fix_echo").is_some());
    assert_eq!(box_.source_of("fix_echo").as_deref(), Some("mcp:fix"));

    let out = box_.call_approved("fix_echo", serde_json::json!({ "text": "via the box" }), false)
        .await
        .unwrap();
    assert_eq!(out, serde_json::json!({ "text": "via the box" }));
}

#[tokio::test]
async fn environment_reaches_the_server() {
    // Most servers take their credentials this way, so a dropped `env` looks
    // like the server rejecting the key.
    let mut env = BTreeMap::new();
    env.insert("OZGENT_FIXTURE_CHECK".into(), "present".into());
    let with_env = mcp::Server { env, ..settings() };
    assert!(connect(with_env).await.is_some());
}
