//! Real embeddings, from a model rather than from a hash.
//!
//! The memory layer fuses keyword search with vector similarity, which only
//! buys anything if the two disagree — a keyword index finds "reduce latency"
//! for the query "reduce latency", and the vector side is what should find it
//! for "make it faster". A lexical stand-in on the vector side makes the fusion
//! a keyword search blended with itself.
//!
//! An embedding model is a different mode of the same engine: the context is
//! created with embeddings enabled and a pooling strategy, and after decoding a
//! sequence llama.cpp hands back one vector for the whole thing instead of
//! logits for the last token.

use std::path::Path;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

use crate::engine::EngineError;

/// Longest text embedded in one go.
///
/// Anything past this is truncated rather than chunked: a memory note that
/// long is already past the point where one vector describes it well, and
/// silently averaging several chunks would be worse than using the opening.
const MAX_TOKENS: usize = 512;

/// Texts embedded per decode.
///
/// Each needs its own sequence lane and its own slice of context, so the group
/// size is bounded by what the KV allocation is worth rather than by anything
/// about the model.
const GROUP: usize = 8;

/// A loaded embedding model.
pub struct Embedder {
    model: LlamaModel,
    dimensions: usize,
    n_ctx: u32,
}

impl Embedder {
    /// Load `path` as an embedding model.
    pub fn load(path: &Path, gpu_layers: u32) -> Result<Self, EngineError> {
        let backend = crate::engine::backend_handle()?;
        if !path.exists() {
            return Err(EngineError::Missing { path: path.display().to_string() });
        }
        let params = Box::pin(LlamaModelParams::default().with_n_gpu_layers(gpu_layers));
        let model = LlamaModel::load_from_file(backend, path, &params).map_err(|e| {
            EngineError::Load { path: path.display().to_string(), reason: e.to_string() }
        })?;
        let dimensions = model.n_embd() as usize;
        // Embedding models are small and their windows short; asking for more
        // than the model was trained on wastes memory and changes nothing.
        let n_ctx = model.n_ctx_train().min(MAX_TOKENS as u32).max(64);
        Ok(Self { model, dimensions, n_ctx })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Embed one text into a unit-length vector.
    ///
    /// Normalised because every consumer compares with cosine similarity, and
    /// normalising once here means the comparison is a dot product.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EngineError> {
        Ok(self.embed_batch(std::slice::from_ref(&text.to_string()))?.pop().unwrap_or_default())
    }

    /// Embed several texts, reusing one context.
    ///
    /// Each text gets its own sequence in the batch, so a set of memory notes
    /// costs one decode rather than one per note.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EngineError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let backend = crate::engine::backend_handle()?;
        // One sequence per text, so a group is embedded in a single decode.
        // The context has to be told how many: it defaults to one, and a batch
        // referring to sequence 1 against a one-sequence context is rejected
        // as invalid — which llama-cpp-2 reports as "n_tokens == 0", a message
        // that sends you looking at the batch size instead.
        let lanes = texts.len().min(GROUP) as u32;
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(self.n_ctx * lanes))
            .with_n_batch(self.n_ctx * lanes)
            .with_n_seq_max(lanes)
            .with_embeddings(true)
            // `Unspecified` lets the model's own metadata choose. Forcing a
            // strategy the model was not trained for produces vectors that
            // decode fine and compare badly — Qwen3-Embedding pools the last
            // token, while most BERT-style encoders pool the mean.
            .with_pooling_type(LlamaPoolingType::Unspecified);

        let mut out = Vec::with_capacity(texts.len());
        // Chunked so a long list cannot demand one enormous context.
        for group in texts.chunks(GROUP) {
            let mut context = self
                .model
                .new_context(backend, params.clone())
                .map_err(|e| EngineError::Context(e.to_string()))?;
            let mut batch = LlamaBatch::new((self.n_ctx as usize) * group.len(), group.len() as i32);

            let mut used = Vec::with_capacity(group.len());
            for (seq, text) in group.iter().enumerate() {
                // `Always` on a model that declares no BOS token yields an
                // empty sequence, and an empty batch reaches llama.cpp as
                // "n_tokens == 0" — an error that says nothing about the cause.
                let mut tokens = self
                    .model
                    .str_to_token(text, AddBos::Never)
                    .map_err(|e| EngineError::Tokenize(e.to_string()))?;
                tokens.truncate(self.n_ctx as usize);
                if tokens.is_empty() {
                    used.push(false);
                    continue;
                }
                for (pos, token) in tokens.iter().enumerate() {
                    // Every token is marked as an output. Pooling averages over
                    // the sequence's token outputs, so marking none leaves
                    // llama.cpp with nothing to pool and it rejects the batch.
                    batch
                        .add(*token, pos as i32, &[seq as i32], true)
                        .map_err(|e| EngineError::Batch(e.to_string()))?;
                }
                used.push(true);
            }

            if used.iter().all(|ok| !ok) {
                // Nothing tokenised; decoding would fail with an error that
                // names the batch rather than the reason.
                out.extend(group.iter().map(|_| vec![0.0; self.dimensions]));
                continue;
            }
            context
                .decode(&mut batch)
                .map_err(|e| EngineError::Decode(e.to_string()))?;

            for (seq, ok) in used.iter().enumerate() {
                if !ok {
                    out.push(vec![0.0; self.dimensions]);
                    continue;
                }
                let raw = context
                    .embeddings_seq_ith(seq as i32)
                    .map_err(|e| EngineError::Decode(e.to_string()))?;
                out.push(normalise(raw));
            }
        }
        Ok(out)
    }
}

/// Scale to unit length, leaving an all-zero vector alone.
fn normalise(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalising_gives_unit_length() {
        let v = normalise(&[3.0, 4.0]);
        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-6, "got {len}");
    }

    #[test]
    fn an_all_zero_vector_survives_normalising() {
        // Dividing by zero would poison every later comparison with NaN.
        let v = normalise(&[0.0, 0.0, 0.0]);
        assert!(v.iter().all(|x| *x == 0.0), "{v:?}");
    }

    #[test]
    fn normalising_preserves_direction() {
        let a = normalise(&[1.0, 2.0, 2.0]);
        let b = normalise(&[10.0, 20.0, 20.0]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "{a:?} vs {b:?}");
        }
    }
}
