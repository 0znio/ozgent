//! Does drafting from the NextN head change the answer?
//!
//! Speculation is only ever a latency trick: the target re-samples every token
//! it keeps, so the output must be the output it would have produced anyway.
//! The failure mode when that is wrong is not an error — it is a plausible
//! reply that is quietly not what the model meant to say. So this generates
//! the same prompt twice, greedily, with the head drafting and without, and
//! compares.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: nextn <model.gguf> [prompt]");
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Write a short paragraph about the sea.".into());

    // Greedy, or the two runs differ through sampling alone and the comparison
    // means nothing.
    let mut base = ozgent_core::Options::default();
    base.temperature = Some(0.0);
    base.top_k = Some(1);
    let opts = base.clone().resolve();

    let engine = ozgent_llama::engine::Engine::load(std::path::Path::new(&path), &opts)?;
    {
        let mut session = engine.session(&opts)?;
        println!("nextn heads declared: {}", session.nextn_heads());
        let (width, magnitude) = session.probe_nextn("The capital of France is")?;
        println!("hidden state:         {width} wide, magnitude {magnitude:.1}");
        if width == 0 {
            println!("\nno NextN head here; nothing to compare.");
            return Ok(());
        }
    }

    use std::sync::atomic::Ordering::Relaxed;
    let run = |spec, label: &str| -> Result<(String, f64), Box<dyn std::error::Error>> {
        ozgent_llama::hub::PASS_MICROS.store(0, Relaxed);
        ozgent_llama::hub::PASS_COUNT.store(0, Relaxed);
        ozgent_llama::mtp::STEP_MICROS.store(0, Relaxed);
        ozgent_llama::mtp::STEPS.store(0, Relaxed);
        ozgent_llama::mtp::PROPOSE_MICROS.store(0, Relaxed);
        ozgent_llama::engine::PICK_MICROS.store(0, Relaxed);
        let wall = std::time::Instant::now();
        let mut o = base.clone();
        o.speculative = Some(spec);
        let resolved = o.resolve();
        let mut session = engine.session(&resolved)?;
        let mut text = String::new();
        let (stats, _) = session.generate(&prompt, 120, |t| {
            text.push_str(t);
            true
        })?;
        let wall = wall.elapsed().as_secs_f64() * 1000.0;
        let pm = ozgent_llama::hub::PASS_MICROS.load(Relaxed) as f64 / 1000.0;
        let pc = ozgent_llama::hub::PASS_COUNT.load(Relaxed);
        let dm = ozgent_llama::mtp::PROPOSE_MICROS.load(Relaxed) as f64 / 1000.0;
        let ds = ozgent_llama::mtp::STEPS.load(Relaxed);
        println!(
            "{label:12} {:5.1} tok/s   proposed {:3}  accepted {:3}",
            stats.tokens_per_second(),
            stats.proposed_drafts,
            stats.accepted_drafts
        );
        println!(
            "             {wall:6.0} ms wall = {pm:6.0} ms in {pc} target passes              + {dm:5.0} ms in {ds} draft steps + {:.0} ms elsewhere",
            wall - pm - dm
        );
        println!(
            "             of which choosing tokens: {:.0} ms",
            ozgent_llama::engine::PICK_MICROS.load(Relaxed) as f64 / 1000.0
        );
        Ok((text, stats.tokens_per_second()))
    };

    println!();
    let (off, rate_off) = run(ozgent_core::accel::Speculative::Off, "no drafting")?;
    let (mtp, rate_mtp) = run(ozgent_core::accel::Speculative::Mtp, "nextn head")?;
    let (ngram, _) = run(ozgent_core::accel::Speculative::Ngram, "n-grams")?;

    let (pm, pc, pt) = (
        ozgent_llama::hub::PASS_MICROS.swap(0, Relaxed),
        ozgent_llama::hub::PASS_COUNT.swap(0, Relaxed),
        ozgent_llama::hub::PASS_TOKENS.swap(0, Relaxed),
    );
    if pc > 0 {
        println!(
            "\ntarget passes: {pc}, {:.2} ms each, {:.1} tokens each, {:.0} ms total",
            pm as f64 / 1000.0 / pc as f64,
            pt as f64 / pc as f64,
            pm as f64 / 1000.0
        );
    }
    let steps = ozgent_llama::mtp::STEPS.swap(0, std::sync::atomic::Ordering::Relaxed);
    let micros = ozgent_llama::mtp::STEP_MICROS.swap(0, std::sync::atomic::Ordering::Relaxed);
    if steps > 0 {
        println!(
            "\ndraft step: {:.2} ms each over {steps} steps",
            micros as f64 / 1000.0 / steps as f64
        );
    }

    println!();
    let same = |a: &str, b: &str| if a == b { "identical" } else { "DIFFERENT" };
    println!("nextn vs none:  {}", same(&mtp, &off));
    println!("ngram vs none:  {}", same(&ngram, &off));
    let diverge = |a: &str, b: &str| -> String {
        match a.chars().zip(b.chars()).position(|(x, y)| x != y) {
            None if a.len() == b.len() => "identical".into(),
            None => format!("one is a prefix of the other; {} vs {} chars", a.len(), b.len()),
            Some(i) => {
                let head: String = a.chars().take(i).collect();
                let tail_a: String = a.chars().skip(i).take(28).collect();
                let tail_b: String = b.chars().skip(i).take(28).collect();
                format!(
                    "diverges at char {i}, after {:?}\n      one says {:?}\n      other says {:?}",
                    head.chars().rev().take(24).collect::<String>().chars().rev().collect::<String>(),
                    tail_a, tail_b
                )
            }
        }
    };
    println!("  nextn: {}", diverge(&mtp, &off));
    println!("  ngram: {}", diverge(&ngram, &off));
    if mtp == off {
        println!("\nVERDICT: same answer, {:.0}% the speed.", 100.0 * rate_mtp / rate_off.max(0.001));
    }
    Ok(())
}
