//! How much state a sequence carries, split by what a KV trim can and cannot undo.
//!
//! Draft rejection in speculative decoding discards the KV entries a rejected
//! token produced. That works for attention layers and silently corrupts
//! recurrent ones, which have already absorbed the token. The way out is to
//! snapshot the recurrent part and restore it instead of trimming — but only if
//! the snapshot is small enough to take on every draft step.
//!
//! Run: cargo run --release -p ozgent-llama --features cuda --example statesize -- <model.gguf>
fn main() {
    let path = std::env::args().nth(1).expect("usage: statesize <model.gguf>");
    let opts = ozgent_core::Options::default().resolve();

    let engine = ozgent_llama::engine::Engine::load(std::path::Path::new(&path), &opts)
        .expect("loading the model");
    let mut session = engine.session(&opts).expect("creating a session");

    for tokens in [0u32, 64, 256] {
        if tokens > 0 {
            let prompt = "word ".repeat(tokens as usize);
            let _ = session.generate(&prompt, 1, |_| true);
        }
        let full = session.state_bytes(false, false);
        let partial = session.state_bytes(true, false);
        let on_dev = session.state_bytes(true, true);
        println!(
            "after ~{tokens:>4} tokens   full {:>9.2} MB   recurrent-only {:>7.2} MB   on-device {:>7.2} MB",
            full as f64 / 1048576.0,
            partial as f64 / 1048576.0,
            on_dev as f64 / 1048576.0,
        );
    }
}
