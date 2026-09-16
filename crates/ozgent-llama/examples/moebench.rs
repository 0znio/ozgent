//! Prefill and decode rates for one model under the placement knobs being
//! compared, reported separately because they respond to different things:
//! prefill is compute-bound and can move expert work to the GPU; decode is
//! bandwidth-bound and cannot.
//!
//! Knobs by environment: OZ_CPU_MOE (auto|all|off|N), OZ_CTX, OZ_UBATCH,
//! OZ_BATCH, OZ_THREADS, OZ_NO_MMAP, OZ_NGL, OZ_ROUNDS, OZ_PROMPT_REPEAT (prompt
//! length), OZ_TOKENS (decode length), OZ_VERIFY (verification cost by draft
//! length instead of a generation run).

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
    o.speculative = Some(match std::env::var("OZ_SPEC").as_deref() {
        Ok("mtp") => ozgent_core::accel::Speculative::Mtp,
        Ok("ngram") => ozgent_core::accel::Speculative::Ngram,
        Ok("auto") => ozgent_core::accel::Speculative::Auto,
        _ => ozgent_core::accel::Speculative::Off,
    });
    // Drafting is off for the rate runs, but the draft ceiling still decides
    // how deep a recurrent rollback ring the context asks for — which is what
    // `OZ_VERIFY` is measuring. Zero asks for none, which is the old
    // behaviour and so the other half of that A/B.
    if let Some(k) = env::<u32>("OZ_DRAFT") {
        o.speculative_tuning =
            Some(ozgent_core::accel::SpeculativeTuning { draft_tokens: k, ..Default::default() });
    }
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

    if std::env::var("OZ_VERIFY").is_ok() {
        let ks = [1usize, 2, 3, 4, 6, 8];
        for (k, ms) in engine.probe_verify_cost(&opts, &ks, 9)? {
            println!("verify k={k}: {ms:6.1} ms  ({:.1} ms per token)", ms / k as f64);
        }
        return Ok(());
    }
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
        let mut text = String::new();
        let (stats, _) = session.generate(&prompt, tokens, |t| {
            text.push_str(t);
            true
        })?;
        if std::env::var("OZ_ECHO").is_ok() {
            println!("----8<----\n{text}\n---->8----");
        }
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
