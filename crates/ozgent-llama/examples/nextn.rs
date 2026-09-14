//! Does this model's NextN head actually produce anything?
//!
//! The feasibility question for MTP drafting, answered before the drafter is
//! written: a head that returns nothing, or returns a row of zeros, would make
//! the whole thing a great deal of work producing no drafts.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: nextn <model.gguf> [prompt]");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "The capital of France is".into());

    let opts = ozgent_core::Options::default().resolve();
    let engine = ozgent_llama::engine::Engine::load(std::path::Path::new(&path), &opts)?;
    let mut session = engine.session(&opts)?;

    println!("nextn heads declared: {}", session.nextn_heads());
    let (width, magnitude) = session.probe_nextn(&prompt)?;
    println!("embedding width:      {width}");
    println!("sum of |values|:      {magnitude:.3}");
    println!();
    if width == 0 {
        println!("VERDICT: no NextN row came back — MTP drafting is not available here.");
    } else if magnitude == 0.0 {
        println!("VERDICT: a row came back but it is all zeros — the head did not run.");
    } else {
        println!("VERDICT: the head ran and produced a real hidden state. Drafting is viable.");
    }
    Ok(())
}
