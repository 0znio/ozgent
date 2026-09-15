//! Do N callers sharing one context finish faster than N callers in a queue?
//!
//! `batched` measured the forward pass in isolation: one pass carrying eight
//! sequences cost a third of eight passes carrying one. This measures the
//! thing that claim has to survive — a real generation loop, with prefill,
//! sampling, detokenisation and a stop check per token, driven by several
//! threads through [`ozgent_llama::hub::Hub`].
//!
//! It also checks the part that matters more than the speed: that sharing a
//! pass does not change what each caller gets. Every thread generates greedily
//! from a different prompt, and the same prompts are then generated alone. The
//! text is printed both ways.

use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use ozgent_llama::hub::Hub;
use std::time::Instant;

const PROMPTS: [&str; 8] = [
    "The capital of France is",
    "Water boils at",
    "The largest planet in the solar system is",
    "In computing, a compiler is",
    "The first person to walk on the moon was",
    "A prime number is",
    "The speed of light is roughly",
    "Photosynthesis is the process by which",
];

/// Generate `n` tokens greedily on `seq`, decoding through the hub.
fn run<'a>(
    hub: &std::sync::Arc<Hub<'a>>,
    model: &llama_cpp_2::model::LlamaModel,
    seq: i32,
    prompt: &str,
    n: usize,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut slot = hub.slot(seq);
    slot.clear();
    // A real turn's prompt is a system prompt plus tool schemas plus history,
    // which is chunked prefill rather than a handful of tokens.
    let pad: usize = std::env::var("HUB_PAD").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let prompt = if pad > 0 {
        format!("{}{prompt}", "The following is background material. ".repeat(pad))
    } else {
        prompt.to_string()
    };
    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let mut sampler = LlamaSampler::greedy();
    let mut pos = 0i32;
    let mut out = String::new();

    // Prefill in chunks the hub can carry, so a long prompt does not have to
    // wait for a batch wide enough to hold all of it at once.
    let chunk = hub.n_batch().min(512);
    let mut logits = None;
    for part in tokens.chunks(chunk) {
        let last = pos as usize + part.len() == tokens.len();
        let got = slot.run(
            part.to_vec(),
            pos,
            if last { ozgent_llama::hub::Logits::Last } else { ozgent_llama::hub::Logits::None },
        )?;
        pos += part.len() as i32;
        if last {
            logits = got.into_last();
        }
    }

    for _ in 0..n {
        let row = match logits.take() {
            Some(r) => r,
            None => break,
        };
        let mut candidates = llama_cpp_2::token::data_array::LlamaTokenDataArray::from_iter(
            row.iter().enumerate().map(|(i, &l)| {
                llama_cpp_2::token::data::LlamaTokenData::new(LlamaToken(i as i32), l, 0.0)
            }),
            false,
        );
        candidates.apply_sampler(&sampler);
        let token = candidates.selected_token().expect("a sampler selects a token");
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        out.push_str(&model.token_to_str(token, llama_cpp_2::model::Special::Tokenize)?);
        let got = slot.run(vec![token], pos, ozgent_llama::hub::Logits::Last)?;
        pos += 1;
        logits = got.into_last();
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: hubbed <model.gguf> [callers] [tokens]");
    let callers: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let tokens: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(48);

    let mut base = ozgent_core::Options::default();
    base.context_length =
        Some(std::env::var("HUB_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(1024));
    // The cache type changes how attention is computed, so it has to be
    // controllable here: the first measurement showed a width-two pass
    // costing exactly two narrow ones, which the isolated probe did not.
    if std::env::var("HUB_F16").is_ok() {
        base.cache_type_k = Some(ozgent_core::accel::CacheType::F16);
        base.cache_type_v = Some(ozgent_core::accel::CacheType::F16);
    }
    if std::env::var("HUB_NOFLASH").is_ok() {
        base.flash_attention = Some(false);
    }
    if let Ok(b) = std::env::var("HUB_BATCH") {
        base.batch_size = b.parse().ok();
    }
    let opts = base.resolve();
    println!("kv {:?}/{:?} batch {} ubatch {:?} flash {}",
        opts.cache_type_k, opts.cache_type_v, opts.batch_size, opts.ubatch, opts.flash_attention);
    let engine = ozgent_llama::engine::Engine::load(std::path::Path::new(&path), &opts)?;
    // Zero means leave the window adaptive, which is the real behaviour.
    let hold: u64 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    // `UpTo` is the shared-pool path the daemon takes; `Exact` divides the
    // window between slots instead.
    let want = if std::env::var("HUB_UNIFIED").is_ok() {
        ozgent_llama::engine::Slots::UpTo(callers as u32)
    } else {
        ozgent_llama::engine::Slots::Exact(callers as u32)
    };
    let (hub, window) = engine.hub(&opts, want)?;
    let hub = if hold > 0 {
        std::sync::Arc::new(
            std::sync::Arc::try_unwrap(hub)
                .unwrap_or_else(|_| unreachable!())
                .with_window(std::time::Duration::from_micros(hold)),
        )
    } else {
        hub
    };
    let model = engine.model();
    println!("{} of {} layers on the gpu", engine.gpu_layers_used(), engine.n_layer());
    println!(
        "{callers} slots, {window} tokens of window each, {tokens} tokens apiece, {hold}us hold\n"
    );

    // Together.
    let started = Instant::now();
    let shared: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..callers)
            .map(|i| {
                let hub = &hub;
                s.spawn(move || run(hub, model, i as i32, PROMPTS[i % PROMPTS.len()], tokens))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap().unwrap()).collect()
    });
    let together = started.elapsed().as_secs_f64();
    let (passes, merged) = hub.traffic();
    let (spent, _) = hub.spent();
    let wide_ms = spent * 1000.0 / passes.max(1) as f64;
    hub.reset_traffic();

    // Apart, through the same hub, one caller at a time — which is what the
    // daemon does today.
    hub.with_context(|c| c.clear_kv_cache());
    let started = Instant::now();
    let alone: Vec<String> = (0..callers)
        .map(|i| run(&hub, model, i as i32, PROMPTS[i % PROMPTS.len()], tokens).unwrap())
        .collect();
    let apart = started.elapsed().as_secs_f64();
    let (spent_alone, passes_alone) = hub.spent();
    let narrow_ms = spent_alone * 1000.0 / passes_alone.max(1) as f64;

    let produced: usize = shared.iter().map(|s| s.len()).sum();
    println!("together {together:6.2}s   apart {apart:6.2}s   {:.2}x", apart / together.max(0.001));
    println!("average batch width {:.2} over {passes} passes", merged as f64 / passes.max(1) as f64);
    println!(
        "pass  {wide_ms:5.1} ms wide vs {narrow_ms:5.1} ms narrow   \
         ({:.0}% of wall batched, {:.0}% alone)",
        spent * 100.0 / together.max(0.001),
        spent_alone * 100.0 / apart.max(0.001)
    );
    println!("{produced} bytes produced\n");

    let mut same = 0;
    for (i, (a, b)) in shared.iter().zip(&alone).enumerate() {
        let mark = if a == b {
            same += 1;
            "same"
        } else {
            "DIFFERS"
        };
        println!("[{i}] {mark}  {:?}", a.trim());
        if a != b {
            println!("      alone: {:?}", b.trim());
        }
    }
    println!("\n{same}/{callers} identical to generating alone");
    Ok(())
}
