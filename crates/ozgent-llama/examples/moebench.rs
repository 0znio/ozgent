//! Prefill and decode rates for one model under the placement knobs being
//! compared, reported separately because they respond to different things:
//! prefill is compute-bound and can move expert work to the GPU; decode is
//! bandwidth-bound and cannot.
//!
//! Knobs by environment: OZ_CPU_MOE (auto|all|off|N), OZ_CTX, OZ_UBATCH,
//! OZ_BATCH, OZ_THREADS, OZ_NO_MMAP, OZ_NO_REPACK, OZ_NO_HOST, OZ_NO_OP_OFFLOAD,
//! OZ_PROMPT_REPEAT (prompt length), OZ_TOKENS (decode length).

fn env<T: std::str::FromStr>(k: &str) -> Option<T> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: moebench <model.gguf>");
    tracing_subscriber::fmt().with_writer(std::io::stderr).with_target(false).init();
    let mut o = ozgent_core::Options::default();
    o.context_length = Some(env("OZ_CTX").unwrap_or(8192));
    o.temperature = Some(0.0);
    o.top_k = Some(1);
    o.speculative = Some(ozgent_core::accel::Speculative::Off);
    o.ubatch = env("OZ_UBATCH");
    o.batch_size = env("OZ_BATCH");
    o.threads = env("OZ_THREADS");
    if std::env::var("OZ_NO_MMAP").is_ok() {
        o.use_mmap = Some(false);
    }
    if let Ok(v) = std::env::var("OZ_CPU_MOE") {
        o.cpu_moe = Some(v.parse().map_err(|e| format!("OZ_CPU_MOE: {e}"))?);
    }
    if let Ok(v) = std::env::var("OZ_NGL") {
        o.gpu_layers = Some(v.parse().map_err(|e| format!("OZ_NGL: {e}"))?);
    }
    let opts = o.resolve();

    let t = std::time::Instant::now();
    let engine = ozgent_llama::engine::Engine::load(std::path::Path::new(&path), &opts)?;
    let load = t.elapsed().as_secs_f64();
    println!("load {load:.1}s, {} of {} layers on gpu", engine.gpu_layers_used(), engine.n_layer());

    let repeat: usize = env("OZ_PROMPT_REPEAT").unwrap_or(60);
    let tokens: u32 = env("OZ_TOKENS").unwrap_or(96);
    let prompt = format!(
        "{}\nIn one detailed paragraph, explain how a lighthouse lens concentrates light.",
        "Background notes on coastal navigation, maritime history and optics. ".repeat(repeat)
    );
    let free = || {
        ozgent_llama::backend::best_gpu().map(|d| d.memory_free as u64 / (1 << 20)).unwrap_or(0)
    };
    println!("free after load: {} MiB", free());
    let rounds: usize = env("OZ_ROUNDS").unwrap_or(2);
    for round in 0..rounds {
        let mut session = engine.session(&opts)?;
        println!("free after context: {} MiB", free());
        let (stats, _) = session.generate(&prompt, tokens, |_| true)?;
        println!("free after generating: {} MiB", free());
        println!(
            "round {round}: prefill {:6.1} tok/s over {} tokens   decode {:5.1} tok/s over {}",
            stats.prompt_tokens_per_second(),
            stats.prompt_tokens,
            stats.tokens_per_second(),
            stats.generated_tokens
        );
    }
    Ok(())
}
