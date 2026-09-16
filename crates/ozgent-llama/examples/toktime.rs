//! How long does it take to turn a prompt into tokens, and is the first time
//! different from the rest?
//!
//! A turn's reported prompt time was 1.6 s on a prompt whose prefill took 45
//! ms. The only thing between the two clocks is tokenising, which should not be
//! able to account for it — so either it can, or the measurement is lying.

use llama_cpp_2::model::AddBos;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: toktime <model.gguf>");
    let repeat: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(120);
    let mut o = ozgent_core::Options::default();
    o.context_length = Some(8192);
    let opts = o.resolve();
    let engine = ozgent_llama::engine::Engine::load(std::path::Path::new(&path), &opts)?;
    let model = engine.model();

    // Roughly the shape of a real turn: a system prompt plus tool schemas is a
    // long run of JSON-ish text, not prose.
    let prompt = format!(
        "{}\n",
        r#"{"name":"web_search","description":"Search the web for a query and return results with titles, urls and snippets.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"The search terms."},"count":{"type":"integer"}},"required":["query"]}}"#
            .repeat(repeat)
    );
    println!("prompt is {} bytes", prompt.len());

    for round in 0..6 {
        let t = std::time::Instant::now();
        let tokens = model.str_to_token(&prompt, AddBos::Always)?;
        println!("round {round}: {} tokens in {:?}", tokens.len(), t.elapsed());
    }
    Ok(())
}
