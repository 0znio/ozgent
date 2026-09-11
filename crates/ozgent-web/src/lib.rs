//! The `ozgent web` server.
//!
//! A local chat application over the same engine, model registry and memory
//! store the terminal client uses, so a conversation started in one shows up
//! in the other.

pub mod agents;
pub mod anthropic;
pub mod api;
pub mod media;
pub mod openai;
pub mod permission;
pub mod state;
pub mod turn;
pub mod worker;

use ozgent_core::{Config, Paths};

/// Run the server until the process is stopped.
pub async fn serve(
    paths: Paths,
    config: Config,
    host: &str,
    port: u16,
    cli: ozgent_core::Options,
) -> anyhow::Result<()> {
    let state = state::App::new(paths, config, cli).await?;
    serve_with(state, host, port).await
}

/// Serve an application that has already been built.
///
/// Split out so one process can run the web interface and the messaging
/// gateway over the *same* [`state::App`]. That sharing is not a convenience:
/// the inference thread, the model it has loaded, the tool host and the
/// permission grants are all per-`App`, so two `App`s in one process would
/// mean two copies of the model in VRAM and a permission answered in the
/// browser having no effect on a question asked over Telegram.
pub async fn serve_with(state: state::State, host: &str, port: u16) -> anyhow::Result<()> {
    // The OpenAI- and Anthropic-compatible API on the same port, over the
    // same loaded model. Run separately, `ozgent serve` would load a second
    // copy of the model into VRAM just so another program could talk to it.
    let app = api::router(state.clone())
        .merge(openai::router(state, openai::ApiKey(None)))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        anyhow::anyhow!("cannot bind {addr}: {e}. Is another ozgent web already running?")
    })?;

    // `0.0.0.0` is a bind address, not somewhere to point a browser. Both
    // addresses are printed and labelled, because the one to hand someone
    // else is never the one this machine uses.
    println!("ozgent web");
    println!("  this machine:  http://localhost:{port}");
    println!("  API:           http://localhost:{port}/v1  (OpenAI and Anthropic compatible)");
    if host == "0.0.0.0" || host == "::" {
        match lan_address() {
            Some(ip) => {
                println!("  same network:  http://{ip}:{port}");
                println!("                 (phones and other computers on the same wifi)");
            }
            None => println!("  same network:  no network address found; is this machine offline?"),
        }
        println!();
        println!("  There is no password. Anyone who can reach that address can read");
        println!("  your conversations and change settings. `--host 127.0.0.1` keeps it");
        println!("  to this machine.");
    }
    println!();
    println!("press ctrl-c to stop");
    axum::serve(listener, app).await?;
    Ok(())
}

/// This machine's address on the local network.
///
/// Found by asking the routing table which source address it would use to
/// reach the internet, via a UDP socket that is never sent on — no traffic
/// leaves, and no DNS is involved. Beats enumerating interfaces, which needs
/// a dependency and still has to guess which one matters.
fn lan_address() -> Option<std::net::IpAddr> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("192.0.2.1:9").ok()?;
    let ip = socket.local_addr().ok()?.ip();
    (!ip.is_loopback() && !ip.is_unspecified()).then_some(ip)
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
    cli: ozgent_core::Options,
) -> anyhow::Result<()> {
    let state = state::App::new(paths, config, cli).await?;
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
