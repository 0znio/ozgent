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

/// The context kept ready between calls. Texts up to this long are embedded
/// in it; a longer one gets a context of its own size for that call, up to the
/// model's trained window, which is then released.
///
/// Why not keep the whole window ready: llama.cpp allocates the cache for all
/// of it up front. Qwen3-Embedding's 32,768 tokens are about 2 GB at 8 bits —
/// more than the model itself — held permanently for a length that rarely
/// arrives. The limit used to be 512, applied silently, so a long reply was
/// represented by its opening alone; it is now the model's own.
pub const WORKING_TOKENS: u32 = 8192;

/// Micro-batch for a last-token-pooling model, whatever the context.
const UBATCH: u32 = 2048;

/// Texts embedded per decode, at most. They share one pool of cells, so a
/// group is packed by tokens, not by count: many short texts at once, or one
/// long one alone.
const GROUP: usize = 8;

/// An input longer than the embedder takes, in the strict mode the API uses.
#[derive(Debug)]
pub struct TooLong {
    pub index: usize,
    pub tokens: usize,
    pub limit: usize,
}

/// Whether a text is something to find, or something to be found.
///
/// Retrieval-trained models are asymmetric: a query is embedded with an
/// instruction saying what it is looking for, a stored document without one.
/// Using the same form for both still works, but ranks worse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Query,
    Document,
}

/// A loaded embedding model.
///
/// Field order matters: `context` borrows `model` and must be dropped first,
/// which Rust does in declaration order.
pub struct Embedder {
    /// One context, reused. Creating it — cache, compute buffers, the first
    /// graph — was most of the cost of every call: about 25 ms of a 28 ms
    /// embedding of a short message on the GPU. Behind a mutex because the
    /// embedder is shared; one context is one decode at a time anyway.
    context: std::sync::Mutex<Option<(u32, llama_cpp_2::context::LlamaContext<'static>)>>,
    /// Boxed so its address is stable for the context's borrow.
    model: Box<LlamaModel>,
    dimensions: usize,
    /// Longest text embedded whole: the model's trained window, or less if
    /// the operator capped it.
    n_ctx: u32,
    /// Whether the model is on the GPU, where a bigger context has to fit
    /// beside whatever else is there.
    gpu: bool,
    /// Bytes of 8-bit cache per token, for sizing a context before making it.
    kv_per_token: u64,
    /// Whether the model pools its last token, so a long text may be fed in
    /// several micro-batches: the cache carries the earlier ones forward and
    /// only the final token's output is used. A mean- or CLS-pooled encoder
    /// must see a sequence in one micro-batch.
    last_pooling: bool,
    /// The special tokens the model's tokenizer says to add. Qwen3-Embedding
    /// pools the *last* token and was trained with `<|endoftext|>` there;
    /// tokenising without it pooled whatever word the text happened to end on.
    add_bos: bool,
    add_eos: bool,
    /// What the model's own documentation puts in front of each role.
    prefixes: (Option<String>, Option<String>),
    name: String,
}

/// The prefixes a model family was trained with, as (query, document).
///
/// Only families whose documentation specifies them; anything else gets none,
/// which is what it was trained for.
fn prefixes_for(name: &str) -> (Option<String>, Option<String>) {
    let n = name.to_ascii_lowercase();
    let some = |s: &str| Some(s.to_string());
    if n.contains("qwen3") && n.contains("embed") {
        // Qwen3-Embedding: an instruction on the query side only.
        (some("Instruct: Given a message from a conversation, retrieve earlier messages and notes that are relevant to it\nQuery:"), None)
    } else if n.contains("nomic") {
        (some("search_query: "), some("search_document: "))
    } else if n.contains("e5") && !n.contains("mistral") {
        (some("query: "), some("passage: "))
    } else if n.contains("bge") || n.contains("mxbai") {
        (some("Represent this sentence for searching relevant passages: "), None)
    } else {
        (None, None)
    }
}

impl Embedder {
    /// Load `path` as an embedding model.
    pub fn load(path: &Path, gpu_layers: u32) -> Result<Self, EngineError> {
        Self::load_with(path, gpu_layers, 0)
    }

    /// As [`Embedder::load`], taking texts of up to `max_tokens` whole; `0`
    /// means the whole window the model was trained on.
    pub fn load_with(path: &Path, gpu_layers: u32, max_tokens: u32) -> Result<Self, EngineError> {
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
        let trained = model.n_ctx_train().max(64);
        let n_ctx = if max_tokens == 0 { trained } else { trained.min(max_tokens.max(64)) };
        let arch = model.meta_val_str("general.architecture").unwrap_or_default();
        let head = (model.n_embd() as u64) / (model.n_head().max(1) as u64);
        let width = |key: &str| {
            model.meta_val_str(&format!("{arch}.attention.{key}")).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(head)
        };
        let per_layer = model.n_head_kv() as u64 * (width("key_length") + width("value_length"));
        // q8_0 stores 34 bytes per 32 values.
        let kv_per_token = (model.n_layer() as u64 * per_layer * 34).div_ceil(32);
        // llama.cpp's LLAMA_POOLING_TYPE_LAST.
        let last_pooling = model
            .meta_val_str(&format!("{arch}.pooling_type"))
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            == Some(3);
        let flag = |key: &str| model.meta_val_str(key).is_ok_and(|v| v.trim().eq_ignore_ascii_case("true"));
        // For measurement: `OZGENT_EMBED_PLAIN=1` embeds as ozgent did before
        // — no special tokens, no prefixes — so the two can be compared.
        let plain = std::env::var("OZGENT_EMBED_PLAIN").is_ok_and(|v| v == "1");
        let name = model
            .meta_val_str("general.name")
            .or_else(|_| model.meta_val_str("general.basename"))
            .unwrap_or_default();
        let prefixes = if plain { (None, None) } else { prefixes_for(&name) };
        Ok(Self {
            context: std::sync::Mutex::new(None),
            add_bos: !plain && flag("tokenizer.ggml.add_bos_token"),
            add_eos: !plain && flag("tokenizer.ggml.add_eos_token"),
            prefixes,
            name,
            model: Box::new(model),
            dimensions,
            n_ctx,
            gpu: gpu_layers > 0,
            kv_per_token,
            last_pooling,
        })
    }

    /// The model's own name, from its metadata.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Embed texts in `role`: queries get the model's query prefix, documents
    /// its document prefix.
    pub fn embed_as(&self, role: Role, texts: &[String]) -> Result<Vec<Vec<f32>>, EngineError> {
        let prefix = match role {
            Role::Query => self.prefixes.0.as_deref(),
            Role::Document => self.prefixes.1.as_deref(),
        };
        match prefix {
            None => self.embed_batch(texts),
            Some(p) => self.embed_batch(&texts.iter().map(|t| format!("{p}{t}")).collect::<Vec<_>>()),
        }
    }

    /// Tokens for one text, with the special tokens the model expects and
    /// room kept for them inside the window.
    fn tokens(&self, text: &str) -> Result<Vec<llama_cpp_2::token::LlamaToken>, EngineError> {
        let mut tokens = self
            .model
            .str_to_token(text, AddBos::Never)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        let reserved = self.add_bos as usize + self.add_eos as usize;
        tokens.truncate((self.reachable() as usize).saturating_sub(reserved));
        if tokens.is_empty() {
            return Ok(tokens);
        }
        if self.add_bos {
            tokens.insert(0, self.model.token_bos());
        }
        if self.add_eos {
            tokens.push(self.model.token_eos());
        }
        Ok(tokens)
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The longest text embedded whole, in tokens, counting the special
    /// tokens the model adds.
    pub fn max_tokens(&self) -> usize {
        self.n_ctx as usize
    }

    /// The longest text that can be embedded whole *now*: the limit, or on
    /// the GPU what fits in the memory free at this moment — beside a chat
    /// model and the reserve its next decode needs, which a bigger cache must
    /// never take, since llama.cpp answers that by aborting the process.
    fn reachable(&self) -> u32 {
        let base = WORKING_TOKENS.min(self.n_ctx);
        if !self.gpu || self.n_ctx <= base {
            return self.n_ctx;
        }
        let Some(free) = crate::backend::best_gpu().filter(|d| d.is_gpu()).map(|d| d.memory_free as u64) else {
            return self.n_ctx;
        };
        // The working context is already allocated; only growth beyond it
        // needs new room, plus scratch for the wider pass.
        let spare = free.saturating_sub(crate::backend::decode_reserve() + (256 << 20));
        let extra = spare / self.kv_per_token.max(1);
        (base as u64 + extra).min(self.n_ctx as u64) as u32
    }

    /// One pass over a group, every token an output: what a mean- or
    /// CLS-pooled encoder needs, since it pools over the whole sequence.
    fn decode_whole(
        &self,
        context: &mut llama_cpp_2::context::LlamaContext<'static>,
        group: &[Vec<llama_cpp_2::token::LlamaToken>],
        cells: usize,
    ) -> Result<(), EngineError> {
        let mut batch = LlamaBatch::new(cells.max(1), group.len() as i32);
        let mut any = false;
        for (lane, tokens) in group.iter().enumerate() {
            for (pos, token) in tokens.iter().enumerate() {
                batch
                    .add(*token, pos as i32, &[lane as i32], true)
                    .map_err(|e| EngineError::Batch(e.to_string()))?;
                any = true;
            }
        }
        if any {
            context.decode(&mut batch).map_err(|e| EngineError::Decode(e.to_string()))?;
        }
        Ok(())
    }

    /// A group for a last-token-pooling model, in two steps.
    ///
    /// llama.cpp treats every token of an embedding decode as an output and
    /// reserves a row of vocabulary logits for each — 600 KB a token for
    /// Qwen3's 151k vocabulary, so 5 GB of host memory for an 8k text and a
    /// kill by the OOM killer at 20k — and runs the vocabulary projection on
    /// all of them, a quarter of the model's work, for rows nobody reads.
    ///
    /// Only the last token's state is pooled, and in a causal model that
    /// state is the same however the earlier tokens reached the cache. So
    /// they are decoded with embeddings off and no outputs, in bounded
    /// slices; then the final tokens alone, with embeddings on. Verified
    /// equal to a single pass (cosine 0.999999).
    fn decode_last_pooled(
        &self,
        context: &mut llama_cpp_2::context::LlamaContext<'static>,
        group: &[Vec<llama_cpp_2::token::LlamaToken>],
    ) -> Result<(), EngineError> {
        use ozgent_mtmd_sys::llama_cpp_sys_2::llama_set_embeddings;
        let slice = UBATCH as usize;
        // Every token but each text's last, interleaved into slices.
        let prefix: Vec<(i32, i32, llama_cpp_2::token::LlamaToken)> = group
            .iter()
            .enumerate()
            .flat_map(|(lane, t)| {
                t.iter().take(t.len().saturating_sub(1)).enumerate().map(move |(pos, tok)| (lane as i32, pos as i32, *tok))
            })
            .collect();
        // SAFETY: the context is live and owned by this embedder.
        unsafe { llama_set_embeddings(context.as_ptr(), false) };
        let mut result = Ok(());
        for chunk in prefix.chunks(slice) {
            let mut batch = LlamaBatch::new(chunk.len(), group.len() as i32);
            for (lane, pos, tok) in chunk {
                if let Err(e) = batch.add(*tok, *pos, &[*lane], false) {
                    result = Err(EngineError::Batch(e.to_string()));
                    break;
                }
            }
            if result.is_ok() {
                result = context.decode(&mut batch).map_err(|e| EngineError::Decode(e.to_string()));
            }
            if result.is_err() {
                break;
            }
        }
        // SAFETY: as above. Turned back on even after a failure: the next
        // call must find the context as it expects.
        unsafe { llama_set_embeddings(context.as_ptr(), true) };
        result?;
        let mut last = LlamaBatch::new(group.len().max(1), group.len() as i32);
        let mut any = false;
        for (lane, tokens) in group.iter().enumerate() {
            if let Some(tok) = tokens.last() {
                last.add(*tok, tokens.len() as i32 - 1, &[lane as i32], true)
                    .map_err(|e| EngineError::Batch(e.to_string()))?;
                any = true;
            }
        }
        if any {
            context.decode(&mut last).map_err(|e| EngineError::Decode(e.to_string()))?;
        }
        Ok(())
    }

    /// Make a context of `size` cells.
    fn make_context(&self, size: u32) -> Result<llama_cpp_2::context::LlamaContext<'static>, EngineError> {
        let backend = crate::engine::backend_handle()?;
        // One pool shared by up to eight sequences, so short texts pack
        // together and a long one can have all of it. The context has to be
        // told how many sequences: it defaults to one, and a batch referring
        // to sequence 1 against a one-sequence context is rejected as invalid
        // — which llama-cpp-2 reports as "n_tokens == 0", a message that sends
        // you looking at the batch size instead.
        // The micro-batch bounds the attention scratch, which grows with its
        // square: a 20k-token micro-batch asked for 12.7 GB. A last-token
        // pooling model can take a long text in slices; any other must see a
        // sequence whole, and those are short-window encoders.
        let ubatch = if self.last_pooling { size.min(UBATCH) } else { size };
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(size))
            .with_n_batch(ubatch)
            .with_n_ubatch(ubatch)
            .with_n_seq_max(GROUP as u32)
            .with_kv_unified(true)
            .with_embeddings(true)
            // An 8-bit cache: half the memory of f16, and a vector cannot
            // tell the difference.
            .with_flash_attention_policy(ozgent_mtmd_sys::llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO)
            .with_type_k(llama_cpp_2::context::params::KvCacheType::Q8_0)
            .with_type_v(llama_cpp_2::context::params::KvCacheType::Q8_0)
            // `Unspecified` lets the model's own metadata choose. Forcing a
            // strategy the model was not trained for produces vectors that
            // decode fine and compare badly — Qwen3-Embedding pools the last
            // token, most BERT-style encoders the mean.
            .with_pooling_type(LlamaPoolingType::Unspecified);
        // SAFETY: the model is boxed, so its address outlives any move of
        // `self`, and `context` is declared before `model`, so it is dropped
        // first. The 'static is never observable outside.
        let model: &'static LlamaModel = unsafe { &*(self.model.as_ref() as *const LlamaModel) };
        crate::llamalog::clear();
        model.new_context(backend, params).map_err(|e| {
            // llama.cpp says why only in its log; "null reference" says nothing.
            EngineError::Context(crate::llamalog::reason().unwrap_or_else(|| e.to_string()))
        })
    }

    /// Refuse any text longer than [`Embedder::max_tokens`] instead of
    /// truncating it — what an API caller is owed.
    pub fn check_lengths(&self, role: Role, texts: &[String]) -> Result<(), TooLong> {
        let prefix = match role {
            Role::Query => self.prefixes.0.as_deref().unwrap_or(""),
            Role::Document => self.prefixes.1.as_deref().unwrap_or(""),
        };
        let reserved = self.add_bos as usize + self.add_eos as usize;
        for (index, text) in texts.iter().enumerate() {
            let tokens = self
                .model
                .str_to_token(&format!("{prefix}{text}"), AddBos::Never)
                .map(|t| t.len())
                .unwrap_or(0)
                + reserved;
            let limit = self.reachable() as usize;
            if tokens > limit {
                return Err(TooLong { index, tokens, limit });
            }
        }
        Ok(())
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
        // Tokenised up front, so groups can be packed by size and the
        // context sized for the longest.
        let mut tokenised = Vec::with_capacity(texts.len());
        for text in texts {
            tokenised.push(self.tokens(text)?);
        }
        let longest = tokenised.iter().map(Vec::len).max().unwrap_or(0) as u32;
        let base = WORKING_TOKENS.min(self.n_ctx);
        // The working context for everyday lengths; for a longer text, one of
        // its size, rounded up so a run of long texts reuses it.
        let size = if longest <= base { base } else { longest.div_ceil(4096).saturating_mul(4096).min(self.n_ctx).max(longest) };
        let mut slot = self.context.lock().unwrap_or_else(|e| e.into_inner());
        if slot.as_ref().is_none_or(|(have, _)| *have != size) {
            // Released before the new one is made, so the two never coexist.
            *slot = None;
            *slot = Some((size, self.make_context(size)?));
        }
        let context = &mut slot.as_mut().expect("just created").1;
        let mut out = vec![Vec::new(); texts.len()];
        let mut next = 0;
        while next < tokenised.len() {
            // As many texts as fit in the pool, up to eight.
            let mut end = next;
            let mut used_cells = 0;
            while end < tokenised.len() && end - next < GROUP {
                let n = tokenised[end].len();
                if end > next && used_cells + n > size as usize {
                    break;
                }
                used_cells += n;
                end += 1;
            }
            let group = &tokenised[next..end];
            // Whatever the last call left in the cache is not this call's.
            context.clear_kv_cache();
            if self.last_pooling {
                self.decode_last_pooled(context, group)?;
            } else {
                self.decode_whole(context, group, used_cells)?;
            }
            for (lane, tokens) in group.iter().enumerate() {
                out[next + lane] = if tokens.is_empty() {
                    // Nothing tokenised; a zero vector compares as nothing.
                    vec![0.0; self.dimensions]
                } else {
                    let raw = context
                        .embeddings_seq_ith(lane as i32)
                        .map_err(|e| EngineError::Decode(e.to_string()))?;
                    normalise(raw)
                };
            }
            next = end;
        }
        // A grown context goes back once it has served: the memory it holds
        // is for a length that rarely comes twice.
        if size > base {
            *slot = None;
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
