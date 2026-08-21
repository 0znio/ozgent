//! Prove the multimodal path end to end against a real model and image.
//!
//! Usage: vision <model.gguf> <mmproj.gguf> <image> [prompt]

use ozgent_core::Options;
use ozgent_llama::engine::Engine;
use ozgent_llama::mtmd::Media;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: vision <model.gguf> <mmproj.gguf> <image> [prompt]");
        std::process::exit(2);
    }
    let question = args.get(4).cloned().unwrap_or_else(|| "Describe this image.".into());

    let opts = Options { context_length: Some(4096), ..Default::default() }.resolve();
    let engine = Engine::load(std::path::Path::new(&args[1]), &opts)?;
    println!("model loaded: {} layers", engine.n_layer());

    let projector = engine.projector(std::path::Path::new(&args[2]), &opts)?;
    println!("projector loaded; vision supported: {}", projector.supports_vision());
    println!("media marker: {:?}", projector.marker());

    let bytes = std::fs::read(&args[3])?;
    println!("image: {} bytes", bytes.len());

    let mut session = engine.session(&opts)?;
    let prompt = format!(
        "<|im_start|>user\n{}{question}<|im_end|>\n<|im_start|>assistant\n",
        projector.marker()
    );

    let sources = [ozgent_core::ImageSource::Path { path: args[3].clone().into() }];
    let images = [Media { bytes }];
    let (stats, reason) = session.generate_with_media(
        &prompt,
        Some((&projector, &images[..], &sources[..])),
        120,
        |piece| {
            print!("{piece}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
            true
        },
    )?;
    println!(
        "\n\n{} prompt positions, {} generated at {:.1} tok/s, stopped: {reason:?}",
        stats.prompt_tokens,
        stats.generated_tokens,
        stats.tokens_per_second()
    );
    Ok(())
}
