//! Does decoding several sequences together cost less than decoding them apart?
//!
//! The daemon answers one request at a time. Four callers at once take four
//! times as long as one, measured, with the last waiting the whole queue out.
//! Batching would let one forward pass advance every sequence at once — but
//! only if the GPU is not already saturated at a batch of one. If it is, the
//! restructure buys nothing and should not be attempted.
//!
//! So this measures the thing the decision rests on, before anything is built:
//! N sequences one token each, in one batch versus N.

use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: batched <model.gguf>");
    let widest = 8u32;

    let mut base = ozgent_core::Options::default();
    base.context_length = Some(4096);
    let opts = base.resolve();
    let engine = ozgent_llama::engine::Engine::load(std::path::Path::new(&path), &opts)?;

    println!("{} of {} layers on the gpu", engine.gpu_layers_used(), engine.n_layer());
    println!("sequences   one batch    separate batches   ratio");
    for n in [1usize, 2, 4, 8] {
        let (together, apart) = engine.probe_batched_decode(n, widest, 40)?;
        println!(
            "{n:>9}   {together:7.2} ms   {apart:14.2} ms   {:.2}x",
            apart / together.max(0.001)
        );
    }
    println!();
    println!("A ratio near 1.0 means batching buys nothing: the card is already");
    println!("busy with one sequence. Near N means N callers could be served for");
    println!("the price of one.");
    let _ = LlamaBatch::new(1, 1);
    let _ = AddBos::Always;
    Ok(())
}
