//! What does the small sliding-window cache buy, and does it stay correct?
//!
//! Models with sliding-window layers (Spark-X2.5: 27 of 36 on a 512-token
//! window) used to get a full-size cache for every layer, because llama.cpp's
//! C API defaults `swa_full` to true. This compares that against the small
//! cache inside one process — alternating, since separate runs on a laptop
//! drift by more than the effect — on the paths the daemon actually takes:
//!
//! - decode speed with the context already `depth` tokens deep, one caller;
//! - four callers at once through the shared pool;
//! - a second turn that rewinds the cache far past the window (an edited
//!   reply), compared with the same prompt prefilled cold. A window with a
//!   hole in it shows up here as a different answer.
//!
//! cargo run --release -p ozgent-llama -p ozgent-cli --features ozgent-cli/cuda \
//!   --example swabench -- <model.gguf> [depths] [rounds]
//!
//! `depths` is comma separated (default 0,4096,16384,28672); `SWA_CTX` sets the
//! pooled window (default 32768).
use ozgent_llama::engine::{Engine, Session, Slots};
use std::time::Instant;

const SYSTEM: &str = "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜>";

/// A turn in Spark's own template, thinking off.
fn user(text: &str) -> String {
    format!("<｜start▁of▁sentence｜><|User|>{text}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>")
}

/// Filler of about `n` tokens that a model can keep continuing.
fn filler(engine: &Engine, n: usize) -> String {
    let mut out = String::new();
    let mut i = 1u64;
    let mut step = 64;
    loop {
        for _ in 0..step {
            out.push_str(&format!("Entry {i}: {i} squared is {}, and {i} cubed is {}.\n", i * i, i * i * i));
            i += 1;
        }
        let have = engine
            .model()
            .str_to_token(&out, llama_cpp_2::model::AddBos::Never)
            .map(|t| t.len())
            .unwrap_or(0);
        if have >= n {
            return out;
        }
        // Lines lengthen as the numbers grow, so step by what they cost now,
        // and short of the target.
        let per_line = (have as f64 / (i - 1) as f64).max(1.0) * 1.5;
        step = (((n - have) as f64 / per_line) as usize).max(1);
    }
}

fn opts(ctx: u32, speculate: bool) -> ozgent_core::Resolved {
    let mut o = ozgent_core::Options {
        temperature: Some(0.0),
        context_length: Some(ctx),
        ..Default::default()
    };
    if !speculate {
        o.speculative = Some(ozgent_core::Speculative::Off);
    }
    o.resolve()
}

fn generate(s: &mut Session<'_>, prompt: &str, n: u32) -> (String, f64, usize) {
    let mut text = String::new();
    let (stats, _) = s
        .generate(prompt, n, |t| {
            text.push_str(t);
            true
        })
        .expect("generating");
    (text, stats.tokens_per_second(), stats.reused_tokens)
}

fn free_mib() -> u64 {
    ozgent_llama::backend::best_gpu().map(|d| d.memory_free as u64 / (1 << 20)).unwrap_or(0)
}

fn common_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[derive(Default)]
struct Tally {
    speed: Vec<(usize, Vec<f64>)>,
    together: Vec<f64>,
    vram: Vec<u64>,
    window: Vec<u32>,
    turn2: Vec<(String, String, usize)>,
}

fn run_mode(path: &str, full: bool, depths: &[usize], ctx: u32, t: &mut Tally) {
    // SAFETY: single-threaded here; nothing else reads the environment now.
    unsafe { std::env::set_var("OZGENT_SWA_FULL", if full { "1" } else { "0" }) };
    let o = opts(ctx, false);
    let before = free_mib();
    let engine = Engine::load_for(std::path::Path::new(path), &o, 5, |_| {}).expect("loading");
    let loaded = free_mib();
    let mut sessions = engine.sessions(&o, Slots::UpTo(4)).expect("opening");
    let opened = free_mib();
    let window = sessions[0].n_ctx();
    println!(
        "  {}: {} of {} layers on the GPU; weights {} MiB, context {} MiB, window {window}",
        if full { "full-size " } else { "windowed  " },
        engine.gpu_layers_used(),
        engine.n_layer(),
        before.saturating_sub(loaded),
        loaded.saturating_sub(opened),
    );
    t.vram.push(loaded.saturating_sub(opened));
    t.window.push(window);

    // One caller, context already `depth` deep.
    for (i, &depth) in depths.iter().enumerate() {
        if depth as u32 + 256 > window {
            continue;
        }
        let s = &mut sessions[0];
        s.reset();
        let prompt = format!("{SYSTEM}{}", user(&format!("{}Continue the list.", filler(&engine, depth))));
        // Twice: the first turn after a load reads weights off disk.
        let _ = generate(s, &prompt, 8);
        s.reset();
        let (_, tps, _) = generate(s, &prompt, 128);
        if t.speed.len() <= i {
            t.speed.push((depth, Vec::new()));
        }
        t.speed[i].1.push(tps);
        println!("    depth {depth:>6}: {tps:6.1} tok/s");
    }

    // Four callers at once, each a quarter of the pool deep.
    let each = (window as usize / 4).saturating_sub(512).min(6144);
    let prompts: Vec<String> = (0..4)
        .map(|i| format!("{SYSTEM}{}", user(&format!("{}Continue the list. ({i})", filler(&engine, each)))))
        .collect();
    for s in sessions.iter_mut() {
        s.reset();
    }
    let started = Instant::now();
    let produced: usize = std::thread::scope(|scope| {
        let handles: Vec<_> = sessions
            .iter_mut()
            .zip(&prompts)
            .map(|(s, p)| scope.spawn(move || generate(s, p, 128).0.len()))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    let _ = produced;
    let secs = started.elapsed().as_secs_f64();
    let rate = 4.0 * 128.0 / secs;
    t.together.push(rate);
    println!("    4 callers x {each} deep: {rate:6.1} tok/s together (incl. prefill, {secs:.1}s)");
    drop(sessions);

    if std::env::var("SWA_SKIP_TURN").is_ok() {
        return;
    }
    // A second turn that goes back past the window. Speculation on, as in the
    // daemon, so draft rollback is exercised too.
    let o = opts(ctx, true);
    let mut sessions = engine.sessions(&o, Slots::UpTo(4)).expect("opening");
    let question = user(&format!(
        "{}Write a long story about a lighthouse keeper, at least 900 words.",
        filler(&engine, 3000)
    ));
    let first = format!("{SYSTEM}{question}");
    let (reply, _, _) = generate(&mut sessions[0], &first, 800);
    // The reply edited after 40 characters, then a new question: the shared
    // prefix ends ~760 tokens before the end of the cache.
    let cut: String = reply.chars().take(40).collect();
    let second = format!(
        "{first}{cut}<｜end▁of▁sentence｜>{}",
        user("Now summarise the story so far in three sentences.")
    );
    let (walked, _, reused) = generate(&mut sessions[0], &second, 96);
    sessions[1].reset();
    let (cold, _, _) = generate(&mut sessions[1], &second, 96);
    let same = common_chars(&walked, &cold);
    println!(
        "    rewound turn: reused {reused} tokens; {} ({same} of {} chars agree with a cold prefill)",
        if walked == cold { "IDENTICAL" } else { "differs" },
        cold.chars().count()
    );
    t.turn2.push((walked, cold, reused));
}

/// Rewind a sequence past its window after another has churned the cache,
/// then compare the next token's logits with a cold prefill of the same
/// prefix. Text comparisons cannot tell a hole from batch-shape noise; this
/// can: noise moves logits by hundredths, a missing window by whole units.
fn rewind_check(path: &str, full: bool, ctx: u32) {
    use llama_cpp_2::context::session::LlamaStateSeqFlags;
    use ozgent_llama::hub::Logits;
    // SAFETY: single-threaded here.
    unsafe { std::env::set_var("OZGENT_SWA_FULL", if full { "1" } else { "0" }) };
    let o = opts(ctx, false);
    let engine = Engine::load_for(std::path::Path::new(path), &o, 5, |_| {}).expect("loading");
    let (hub, _) = engine.hub(&o, Slots::UpTo(4)).expect("opening");
    let tok = |text: &str, bos: bool| {
        let add = if bos { llama_cpp_2::model::AddBos::Always } else { llama_cpp_2::model::AddBos::Never };
        engine.model().str_to_token(text, add).expect("tokenizing")
    };
    let all = tok(&filler(&engine, 4400), true);
    let (head, rest) = all.split_at(3000);
    let tail = &rest[..900];
    let churn = tok(&filler(&engine, 6000), true);
    let probe = rest[900];
    let prefill = |slot: &mut ozgent_llama::hub::Slot<'_>, tokens: &[llama_cpp_2::token::LlamaToken], from: i32| {
        let mut pos = from;
        let mut last = None;
        for chunk in tokens.chunks(512) {
            last = slot.run(chunk.to_vec(), pos, Logits::Last).expect("decoding").into_last();
            pos += chunk.len() as i32;
        }
        last
    };
    let l = head.len() as i32;

    // Cold reference.
    let mut c = hub.slot(2);
    c.clear();
    prefill(&mut c, head, 0);
    let reference = c.run(vec![probe], l, Logits::Last).expect("decoding").into_last().unwrap();
    c.clear();

    let mut results = Vec::new();
    for restore in [false, true] {
        let mut a = hub.slot(0);
        let mut b = hub.slot(1);
        a.clear();
        b.clear();
        prefill(&mut a, head, 0);
        let window = a.with_context(|ctx| ctx.state_seq_get(0, LlamaStateSeqFlags::PARTIAL_ONLY)).expect("saving");
        prefill(&mut a, tail, l);
        // Another conversation writes enough to send the cache's ring round.
        prefill(&mut b, &churn, 0);
        let _ = a.trim(l);
        if restore {
            a.with_context(|ctx| ctx.state_seq_set(&window, 0)).expect("restoring");
        }
        let row = a.run(vec![probe], l, Logits::Last).expect("decoding").into_last().unwrap();
        let diff = row.iter().zip(&reference).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let argmax = |r: &[f32]| r.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        results.push((restore, diff, argmax(&row) == argmax(&reference)));
        a.clear();
        b.clear();
    }
    for (restore, diff, same) in results {
        println!(
            "  {} rewind by {}: max |logit - cold| = {diff:7.3}, top token {}",
            if full { "full-size" } else { "windowed " },
            if restore { "trim + saved window" } else { "trim alone         " },
            if same { "same" } else { "DIFFERENT" }
        );
    }

    // Raw pass time, one token at a time, no sampling or session around it.
    let mut a = hub.slot(0);
    for depth in [0usize, 16384] {
        a.clear();
        let ctx_tokens = tok(&filler(&engine, depth.max(16)), true);
        let ctx_tokens = &ctx_tokens[..depth.max(16).min(ctx_tokens.len())];
        prefill(&mut a, ctx_tokens, 0);
        let mut pos = ctx_tokens.len() as i32;
        let mut times = Vec::new();
        for i in 0..160 {
            let t = Instant::now();
            let _ = a.run(vec![ctx_tokens[i % ctx_tokens.len()]], pos, Logits::Last).expect("decoding");
            times.push(t.elapsed().as_secs_f64() * 1000.0);
            pos += 1;
        }
        println!(
            "  {} raw pass at depth {depth:>5}: {:.2} ms",
            if full { "full-size" } else { "windowed " },
            median(&times[20..])
        );
    }
}

fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: swabench <model.gguf> [depths] [rounds]");
    let depths: Vec<usize> = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "0,4096,16384,28672".into())
        .split(',')
        .filter_map(|d| d.trim().parse().ok())
        .collect();
    let rounds: usize = std::env::args().nth(3).and_then(|r| r.parse().ok()).unwrap_or(2);
    let ctx: u32 = std::env::var("SWA_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(32768);

    if std::env::var("SWA_CHECK").is_ok() {
        for r in 0..rounds {
            rewind_check(&path, r % 2 == 0, ctx);
            rewind_check(&path, r % 2 != 0, ctx);
        }
        return;
    }
    let mut full = Tally::default();
    let mut windowed = Tally::default();
    for r in 0..rounds {
        println!("round {}", r + 1);
        // Alternate which goes first, so warming and thermals fall on both.
        if r % 2 == 0 {
            run_mode(&path, true, &depths, ctx, &mut full);
            run_mode(&path, false, &depths, ctx, &mut windowed);
        } else {
            run_mode(&path, false, &depths, ctx, &mut windowed);
            run_mode(&path, true, &depths, ctx, &mut full);
        }
    }

    println!("\nmedians over {rounds} rounds          full-size   windowed");
    println!("context VRAM (MiB)            {:>10} {:>10}", full.vram.iter().max().unwrap_or(&0), windowed.vram.iter().max().unwrap_or(&0));
    println!("window opened (tokens)        {:>10} {:>10}", full.window.iter().min().unwrap_or(&0), windowed.window.iter().min().unwrap_or(&0));
    for ((depth, a), (_, b)) in full.speed.iter().zip(&windowed.speed) {
        let (a, b) = (median(a), median(b));
        println!("decode at depth {depth:>6} (tok/s) {a:>10.1} {b:>10.1}   {:+.1}%", (b / a - 1.0) * 100.0);
    }
    let (a, b) = (median(&full.together), median(&windowed.together));
    println!("4 callers together (tok/s)    {a:>10.1} {b:>10.1}   {:+.1}%", (b / a - 1.0) * 100.0);
    // The same mode's rewound turn against its own cold prefill, and the two
    // modes' cold answers against each other.
    let agree = |t: &Tally| t.turn2.iter().filter(|(w, c, _)| w == c).count();
    println!("rewound == cold               {:>7}/{} {:>7}/{}", agree(&full), full.turn2.len(), agree(&windowed), windowed.turn2.len());
    if let (Some(f), Some(w)) = (full.turn2.first(), windowed.turn2.first()) {
        println!("cold answers across modes: {} of {} chars agree", common_chars(&f.1, &w.1), f.1.chars().count());
        println!("\nfull-size cold : {:?}", f.1.trim());
        println!("windowed cold  : {:?}", w.1.trim());
        println!("windowed walked: {:?}", w.0.trim());
    }
}
