//! Can this model be rewound mid-generation?
//!
//! Speculative decoding is refused on hybrid models because rejecting a draft
//! discards KV entries the recurrent layers have already absorbed. Snapshotting
//! that state and restoring it is the way round — if it actually works. The
//! failure mode is silently wrong output, so this checks rather than assumes:
//! generate, rewind, generate again, and compare.
//!
//! cargo run --release -p ozgent-llama -p ozgent-cli --features ozgent-cli/cuda --example rewind -- <model.gguf>
fn main() {
    let path = std::env::args().nth(1).expect("usage: rewind <model.gguf>");
    // Greedy, or the two runs differ through sampling alone and the
    // comparison says nothing about the rewind.
    let opts = ozgent_core::Options {
        temperature: Some(0.0),
        ..Default::default()
    }
    .resolve();
    let engine = ozgent_llama::engine::Engine::load(std::path::Path::new(&path), &opts)
        .expect("loading the model");
    let mut session = engine.session(&opts).expect("creating a session");

    let prompt = "Explain in one paragraph why the sky is blue.";
    for (partial, on_device, label) in [
        (false, false, "full state, host   "),
        (true, false, "recurrent only, host"),
        (false, true, "full state, device "),
        (true, true, "recurrent only, dev "),
    ] {
        match session.probe_rewind(prompt, 24, partial, on_device) {
            Ok((a, b, size)) => {
                let same = a == b;
                println!(
                    "{label}  snapshot {:>8.2} MB  rewind {}",
                    size as f64 / 1048576.0,
                    if same { "EXACT" } else { "DIVERGED" }
                );
                if !same {
                    println!("      before: {:?}", &a.chars().take(60).collect::<String>());
                    println!("      after : {:?}", &b.chars().take(60).collect::<String>());
                }
            }
            Err(e) => println!("{label}  unavailable: {e}"),
        }
    }

    // Size is only half the question: a snapshot on every draft step has to fit
    // inside the per-token budget, which at ~56 tok/s is about 17.8 ms.
    println!();
    // Partial states hold only what a trim cannot reach — the recurrent or
    // sliding-window part — so on a hybrid model they should be a fraction of
    // the size, and unlike a device-held state they are plain host bytes with
    // no layout to go stale. If one is cheap enough, a shared context can
    // speculate after all.
    for (on_device, partial, label) in [
        (false, false, "host full   "),
        (true, false, "device full "),
        (false, true, "host partial"),
    ] {
        match session.probe_snapshot_cost(prompt, 20, on_device, partial) {
            Ok((snap, restore, bytes)) => println!(
                "{label}  snapshot {snap:6.2} ms   restore {restore:6.2} ms   host bytes {:>9}",
                bytes
            ),
            Err(e) => println!("{label}  unavailable: {e}"),
        }
    }
}

