//! The `ozgent web` server.
//!
//! A local chat application over the same engine, model registry and memory
//! store the terminal client uses, so a conversation started in one shows up
//! in the other.

pub mod api;
pub mod media;
pub mod openai;
pub mod state;
pub mod worker;

use ozgent_core::{Config, Paths};

/// Run the server until the process is stopped.
pub async fn serve(paths: Paths, config: Config, host: &str, port: u16) -> anyhow::Result<()> {
    let state = state::App::new(paths, config).await?;
    let app = api::router(state).layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        anyhow::anyhow!("cannot bind {addr}: {e}. Is another ozgent web already running?")
    })?;

    println!("ozgent web on http://{addr}");
    println!("press ctrl-c to stop");
    axum::serve(listener, app).await?;
    Ok(())
}

/// Run the OpenAI-compatible API server.
///
/// Kept separate from [`serve`] so the two can be exposed independently: the
/// chat UI is for a person on this machine, the API is for whatever else wants
/// the model, and they usually deserve different bind addresses.
pub async fn serve_api(
    paths: Paths,
    config: Config,
    host: &str,
    port: u16,
    api_key: Option<String>,
) -> anyhow::Result<()> {
    let state = state::App::new(paths, config).await?;
    let app = openai::router(state, openai::ApiKey(api_key.clone()))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        anyhow::anyhow!("cannot bind {addr}: {e}. Is another ozgent already using that port?")
    })?;

    println!("ozgent API on http://{addr}/v1");
    if api_key.is_some() {
        println!("a bearer token is required");
    } else if host != "127.0.0.1" && host != "localhost" {
        println!("warning: bound beyond loopback with no --api-key; anyone who can reach this port can use your models");
    }
    println!("press ctrl-c to stop");
    axum::serve(listener, app).await?;
    Ok(())
}
