//! Drafting with the model's own head, through the engine, against not.
//!
//! Usage: mtp <model.gguf> [rounds] [prompt-file...]
//!
//! Loads once and alternates sessions with speculation off and on, so a hot
//! or throttled GPU slows both alike. Greedy, so the two answers can be
//! compared; see `crate::mtp` for why they may part at a near-tie.

use ozgent_core::accel::Speculative;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_writer(std::io::stderr).with_target(false).init();
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: mtp <model.gguf> [rounds] [prompt-file...]");
    let rounds: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(2);
    let files: Vec<String> = args.collect();
    let prompts: Vec<(String, String)> = if files.is_empty() {
        vec![(
            "fridge".into(),
            "Explain in detail how a refrigerator works, covering the refrigerant cycle, \
             the compressor, the condenser, the expansion valve and the evaporator."
                .into(),
        )]
    } else {
        files
            .iter()
            .map(|f| (f.rsplit('/').next().unwrap_or(f).to_string(), std::fs::read_to_string(f).unwrap()))
            .collect()
    };
    let tokens: u32 = std::env::var("TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let slots: u32 = std::env::var("SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);

    let mut base = ozgent_core::Options::default();
    base.temperature = Some(0.0);
    base.top_k = Some(1);
    base.thinking = Some(ozgent_core::ThinkingMode::Off);
    if let Ok(ctx) = std::env::var("CTX") {
        base.context_length = ctx.parse().ok();
    }
    // ONLY=off loads without the head at all — the real cost of turning
    // drafting off, where the head's layer no longer takes VRAM — and
    // ONLY=on runs just the drafted side.
    let only = std::env::var("ONLY").ok();
    let mut load = base.clone();
    if only.as_deref() == Some("off") {
        load.speculative = Some(Speculative::Off);
    }
    let load = load.resolve();
    let engine = ozgent_llama::engine::Engine::load_for(std::path::Path::new(&path), &load, slots + 1, |_| {})?;

    let run = |spec: Speculative, prompt: &str| -> Result<(String, f64, usize, usize, f64), Box<dyn std::error::Error>> {
        let mut o = base.clone();
        o.speculative = Some(spec);
        let resolved = o.resolve();
        let mut session = if slots > 1 {
            engine.sessions(&resolved, ozgent_llama::engine::Slots::UpTo(slots))?.remove(0)
        } else {
            engine.session(&resolved)?
        };
        let messages = vec![ozgent_core::Message::user(prompt)];
        let rendered = engine.render_prompt_with(
            &messages,
            ozgent_core::ThinkingMode::Off,
            Default::default(),
        )?;
        let mut text = String::new();
        let (stats, _) = session.generate(&rendered, tokens, |t| {
            text.push_str(t);
            true
        })?;
        Ok((
            text,
            stats.tokens_per_second(),
            stats.proposed_drafts,
            stats.accepted_drafts,
            stats.prompt_ms as f64,
        ))
    };

    if std::env::var("MULTI").is_ok() {
        // Three turns of one conversation on one session: the second and
        // third reuse the cache (checkpoints on a hybrid model), which is
        // where a drafter whose cache disagreed with the target's would show.
        let follow = ["Now make it shorter, in five bullet points.", "Which of those points matters most, and why?"];
        let conversation = |spec: Speculative| -> Vec<(String, f64)> {
            let mut o = base.clone();
            o.speculative = Some(spec);
            let resolved = o.resolve();
            let mut session = engine.session(&resolved).unwrap();
            let mut messages = vec![ozgent_core::Message::user(prompts[0].1.clone())];
            let mut out = Vec::new();
            for turn in 0..=follow.len() {
                let rendered = engine
                    .render_prompt_with(&messages, ozgent_core::ThinkingMode::Off, Default::default())
                    .unwrap();
                let mut text = String::new();
                let (stats, _) = session
                    .generate(&rendered, tokens, |t| {
                        text.push_str(t);
                        true
                    })
                    .unwrap();
                out.push((text.clone(), stats.tokens_per_second()));
                messages.push(ozgent_core::Message::assistant(text));
                if turn < follow.len() {
                    messages.push(ozgent_core::Message::user(follow[turn]));
                }
            }
            out
        };
        for _ in 0..rounds {
            let off = conversation(Speculative::Off);
            let on = conversation(Speculative::Auto);
            for (i, (a, b)) in off.iter().zip(&on).enumerate() {
                let parts = a.0.chars().zip(b.0.chars()).position(|(x, y)| x != y);
                println!(
                    "turn {}: off {:5.1} tok/s   head {:5.1} tok/s   {:+.0}%   text {}",
                    i + 1,
                    a.1,
                    b.1,
                    100.0 * (b.1 / a.1 - 1.0),
                    match parts {
                        None if a.0 == b.0 => "identical".to_string(),
                        None => "one ends early".into(),
                        Some(c) => format!("parts at char {c} of {}", a.0.len()),
                    }
                );
            }
        }
        return Ok(());
    }

    if let Some(n) = std::env::var("CONCURRENT").ok().and_then(|v| v.parse::<usize>().ok()) {
        // n conversations at once through one hub, each on its own prompt,
        // drafting off and then on. Wall time for all of them, and each
        // text against the same conversation run alone with drafting off.
        let render = |p: &str| {
            engine
                .render_prompt_with(&[ozgent_core::Message::user(p)], ozgent_core::ThinkingMode::Off, Default::default())
                .unwrap()
        };
        let texts: Vec<String> = (0..n).map(|i| render(&prompts[i % prompts.len()].1)).collect();
        let together = |spec: Speculative| -> (f64, usize, Vec<String>) {
            let mut o = base.clone();
            o.speculative = Some(spec);
            let resolved = o.resolve();
            let sessions = engine.sessions(&resolved, ozgent_llama::engine::Slots::UpTo(n as u32)).unwrap();
            let started = std::time::Instant::now();
            let results: Vec<(usize, String)> = std::thread::scope(|scope| {
                let handles: Vec<_> = sessions
                    .into_iter()
                    .zip(&texts)
                    .map(|(mut session, prompt)| {
                        scope.spawn(move || {
                            let mut text = String::new();
                            let (stats, _) = session
                                .generate(prompt, tokens, |t| {
                                    text.push_str(t);
                                    true
                                })
                                .unwrap();
                            (stats.generated_tokens, text)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let secs = started.elapsed().as_secs_f64();
            let total: usize = results.iter().map(|r| r.0).sum();
            (total as f64 / secs, total, results.into_iter().map(|r| r.1).collect())
        };
        for _ in 0..rounds {
            let (off, _, off_texts) = together(Speculative::Off);
            let (on, _, on_texts) = together(Speculative::Auto);
            let (_, _, off_again) = together(Speculative::Off);
            let same = off_texts.iter().zip(&on_texts).filter(|(a, b)| a == b).count();
            let control = off_texts.iter().zip(&off_again).filter(|(a, b)| a == b).count();
            println!("  control: drafting off twice, {control}/{n} texts identical");
            if std::env::var("SHOW").is_ok() {
                for (a, b) in off_texts.iter().zip(&on_texts) {
                    let i = a.chars().zip(b.chars()).position(|(x, y)| x != y).unwrap_or(a.len());
                    let tail = |t: &str| t.chars().skip(i.saturating_sub(30)).take(90).collect::<String>();
                    println!("  parts at {i}:\n    off: {:?}\n    on:  {:?}", tail(a), tail(b));
                }
            }
            println!(
                "{n} at once: off {off:5.1} tok/s total   head {on:5.1} tok/s total   {:+.0}%   {same}/{n} texts identical",
                100.0 * (on / off - 1.0)
            );
        }
        return Ok(());
    }

    for (name, prompt) in &prompts {
        let (mut off_rates, mut on_rates) = (Vec::new(), Vec::new());
        let (mut off_text, mut on_text) = (String::new(), String::new());
        let (mut prop, mut acc) = (0, 0);
        let (mut off_pp, mut on_pp) = (0.0, 0.0);
        for _ in 0..rounds {
            if only.as_deref() != Some("on") {
                let (t, r, _, _, pp) = run(Speculative::Off, prompt)?;
                off_rates.push(r);
                off_text = t;
                off_pp += pp;
            }
            if only.as_deref() == Some("off") {
                continue;
            }
            let (t, r, p, a, pp) = run(Speculative::Auto, prompt)?;
            on_rates.push(r);
            on_text = t;
            prop += p;
            acc += a;
            on_pp += pp;
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        if off_rates.is_empty() {
            off_rates.push(0.0);
        }
        if on_rates.is_empty() {
            on_rates.push(0.0);
        }
        let (off, on) = (median(&mut off_rates), median(&mut on_rates));
        let same = off_text
            .chars()
            .zip(on_text.chars())
            .position(|(a, b)| a != b)
            .map_or_else(
                || if off_text.len() == on_text.len() { "identical".to_string() } else { "one ends early".into() },
                |i| format!("parts at char {i} of {}", off_text.len()),
            );
        println!(
            "{name:10} off {off:5.1} tok/s   head {on:5.1} tok/s   {:+.0}%   accepted {acc}/{prop}   prefill {:.0} vs {:.0} ms   text {same}",
            100.0 * (on / off - 1.0),
            off_pp / rounds as f64,
            on_pp / rounds as f64,
        );
    }
    Ok(())
}
