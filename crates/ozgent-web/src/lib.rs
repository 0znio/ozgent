//! The `ozgent web` server.
//!
//! A local chat application over the same engine, model registry and memory
//! store the terminal client uses, so a conversation started in one shows up
//! in the other.

pub mod access;
pub mod admin;
mod admin_access;
pub mod agents;
pub mod anthropic;
pub mod api;
pub mod history;
pub mod hub;
pub mod media;
pub mod memory;
pub mod openai;
pub mod permission;
pub mod pool;
pub mod scheduler;
pub mod scheduler_api;
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
    state::watch_config(&state);
    // Jobs run in whichever process gets the lock. Starting it here means
    // `ozgent web` is usually that process, which is what people leave running.
    scheduler::start(&state);
    // Old messages get vectors from the current embedding model, a batch at
    // a time. After a minute, so a server that was started to answer one
    // question answers it first.
    {
        let state = state.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            memory::backfill(&state);
        });
    }
    let app = api::router(state.clone())
        .merge(admin::router(state.clone()))
        .merge(scheduler_api::router(state.clone()))
        .merge(history::router(state.clone()))
        .merge(openai::router(state.clone(), openai::ApiKey(None)))
        .layer(axum::extract::DefaultBodyLimit::max(body_limit(&state)))
        .layer(axum::middleware::from_fn_with_state(state.clone(), access::guard))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        anyhow::anyhow!("cannot bind {addr}: {e}. Is another ozgent web already running?")
    })?;
    // `tap_io` does nothing but lets axum hand handlers the peer address.
    let listener = axum::serve::ListenerExt::tap_io(access::GuardedListener::new(listener, state.clone()), |_| {});
    // Only now, with the port held: a server that failed to bind must not
    // replace the token of the one that is running.
    let token = access::issue_token(&state.paths.local_token_file())
        .map_err(|e| anyhow::anyhow!("cannot write {}: {e}", state.paths.local_token_file().display()))?;
    let _ = state.local_token.set(token);

    // Under a service manager stdout is the journal, and a banner addressed to
    // somebody at a keyboard — "press ctrl-c to stop" — is noise in it that is
    // also untrue. Deciding on the terminal rather than on which command was
    // typed gets this right for every supervisor, and for a pipe.
    if !std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        tracing::info!("listening on {host}:{port}");
        tracing::info!("web, admin, scheduler and the /v1 API are all on this port");
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await?;
        return Ok(());
    }

    // `0.0.0.0` is a bind address, not somewhere to point a browser. Both
    // addresses are printed and labelled, because the one to hand someone
    // else is never the one this machine uses.
    println!("ozgent web");
    println!("  this machine:  http://localhost:{port}");
    println!("  API:           http://localhost:{port}/v1  (OpenAI and Anthropic compatible)");
    println!("  admin:         http://localhost:{port}/admin  (gateway and model downloads)");
    println!("  scheduler:     http://localhost:{port}/scheduler");
    if host == "0.0.0.0" || host == "::" {
        match lan_address() {
            Some(ip) => {
                println!("  same network:  http://{ip}:{port}");
                println!("                 (phones and other computers on the same wifi)");
            }
            None => println!("  same network:  no network address found; is this machine offline?"),
        }
        println!();
        println!("  Other machines sign in with the admin password (`ozgent admin setup`),");
        println!("  and programs need an API key from the admin page. Who may connect at");
        println!("  all is set there too, under Access.");
    }
    println!();
    println!("press ctrl-c to stop");
    // With the peer's address, so failed admin sign-ins are counted per
    // address rather than for everyone at once.
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .await?;
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
    if let Some(k) = &api_key {
        state.shield.set_legacy_key(k.clone());
    }
    // The same checks as the web server: addresses, rate, host, origin, keys.
    // Keys created on the admin page work here too.
    let app = openai::router(state.clone(), openai::ApiKey(None))
        .layer(axum::extract::DefaultBodyLimit::max(body_limit(&state)))
        .layer(axum::middleware::from_fn_with_state(state.clone(), access::guard))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
        anyhow::anyhow!("cannot bind {addr}: {e}. Is another ozgent already using that port?")
    })?;
    // `tap_io` does nothing but lets axum hand handlers the peer address.
    let listener = axum::serve::ListenerExt::tap_io(access::GuardedListener::new(listener, state.clone()), |_| {});

    println!("ozgent API on http://{addr}/v1");
    if api_key.is_some() {
        println!("a bearer token is required");
    } else if host != "127.0.0.1" && host != "localhost" {
        println!("callers on other machines need an API key from the admin page (or --api-key)");
    }
    println!("press ctrl-c to stop");
    axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await?;
    Ok(())
}

/// The largest request body accepted, from `[web.access] max_body_mb`. Read
/// at start: the limit is a property of the router.
fn body_limit(state: &state::State) -> usize {
    let mb = state.config.lock().unwrap_or_else(|e| e.into_inner()).web.access.max_body_mb.max(1);
    mb as usize * 1024 * 1024
}
