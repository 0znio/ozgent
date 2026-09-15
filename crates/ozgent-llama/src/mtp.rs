//! Drafting from the model's own multi-token-prediction head.
//!
//! Qwen3.5 appends a trained NextN block past the end of its main stack. Given
//! the hidden state of the token just decoded, that block predicts the token
//! after it. So the model carries its own drafter: no second set of weights,
//! no VRAM for them, and — unlike n-grams — a draft drawn from the model's
//! distribution rather than from repetition in the context. Measured on a 4B,
//! ordinary prose produces exactly zero n-gram drafts, because there is no
//! repetition to find. This is the case that fills.
//!
//! **One drafter per conversation.** A drafter belongs to the sequence it
//! drafts for: it writes into that sequence's cells and takes them back out
//! again, and llama.cpp scopes both to the sequence id it is given. Nothing
//! here is shared between conversations except the weights and the memory
//! they all already share, so several may draft at once.
//!
//! **The draft context shares the target's memory.** Created with `ctx_other`
//! pointing at the target, which is what lets llama.cpp skip the "catch-up
//! decode" that otherwise replays the target's batch into a second cache. It
//! costs compute buffers, not weights and not a second KV cache.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;

use crate::engine::EngineError;

/// Why this does not pay on Qwen3.5, whatever the threshold.
///
/// Speculation rests on one assumption: that verifying `k` drafted tokens in a
/// single pass costs about what decoding one token costs. Where that holds,
/// every accepted draft is very nearly free. On this model it does not hold,
/// and the measurement is unambiguous — drafting cuts the number of target
/// passes almost in half and the time spent in them does not move:
///
/// ```text
/// threshold   passes   ms in passes   ms per pass
///   none        122        2571          21.1
///   0.95         76        2600          34.2
///   0.70         69        2460          35.7
///   0.50         72        2617          36.3
/// ```
///
/// A pass carrying two tokens costs 34 ms where a pass carrying one costs 21.
/// The draft steps themselves are cheap and behave exactly as designed — 2.9
/// ms each against a 21 ms pass, and the acceptance rate is what the threshold
/// was swept for — but there is nothing for them to win. Every accepted token
/// has to be paid for in the verification pass at close to full price.
///
/// The reason is the architecture. Most of this model's layers are recurrent:
/// they walk a sequence one position at a time rather than attending over it
/// at once, so `k` tokens of one sequence is `k` steps of work. Batching
/// across *different* conversations is a different matter and does pay — those
/// are one step each, taken together, which is the 2.2x the hub measures.
/// Within one sequence there is no such saving to find.
///
/// So this is not a tuning problem and no threshold fixes it. It stays behind
/// `--spec mtp` rather than joining `auto`, and on a model whose layers are
/// ordinary attention it would be worth revisiting — the drafting machinery is
/// correct, the model simply gives it nothing to earn.
///
/// How sure the head must be to justify another forward pass.
///
/// Every proposal costs a decode of the NextN block, so a draft unlikely to be
/// kept is not free the way an n-gram guess is. The first token is always
/// proposed — the head has just been handed a fresh hidden state and its
/// opinion there is the whole point — and each one after it has to earn its
/// pass.
///
/// Swept on a 4B rather than chosen. Against 48 tok/s undrafted, on prose:
///
/// ```text
/// ungated   41.1 tok/s   9 of 48 accepted
/// 0.50      44.2         71 of 114
/// 0.70      48.6         69 of 97
/// 0.85      47.3         65 of 91
/// 0.95      52.6         65 of 82
/// ```
///
/// The pattern is not "accept more", it is "propose less". Every threshold
/// above lands roughly the same number of tokens; what changes is how many
/// passes were spent failing to. Ungated drafting was *slower* than no
/// drafting at all.
const MIN_CONFIDENCE: f32 = 0.95;

/// Microseconds spent inside draft decodes, and how many there were. The
/// whole economics of drafting from the head rests on a step being cheap, so
/// it is counted rather than assumed.
pub static STEP_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Microseconds spent inside `propose` as a whole, which is the decodes plus
/// everything drafting does around them.
pub static PROPOSE_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// The threshold in force, overridable while it is being tuned.
fn min_confidence() -> f32 {
    std::env::var("OZGENT_MTP_MIN_CONFIDENCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MIN_CONFIDENCE)
}

/// A `llama_batch` carrying both a token and an embedding row per position.
///
/// `llama_batch_init` allocates one or the other; the NextN graph needs both,
/// so the token array is added afterwards. llama.cpp's own driver does exactly
/// this, with the same comment about it.
struct EmbdBatch {
    raw: sys::llama_batch,
    capacity: usize,
    n_embd: usize,
}

impl EmbdBatch {
    fn new(capacity: usize, n_embd: usize) -> Self {
        // SAFETY: capacity and n_embd are positive; the handle is freed in Drop.
        let mut raw = unsafe {
            sys::llama_batch_init(capacity as i32, n_embd as i32, 1)
        };
        // SAFETY: `llama_batch_init` left `token` null because an embedding
        // width was given. The NextN graph reads both, so it is allocated here
        // and released in Drop alongside the rest.
        raw.token = unsafe {
            libc_malloc(std::mem::size_of::<sys::llama_token>() * capacity) as *mut sys::llama_token
        };
        Self { raw, capacity, n_embd }
    }

    /// Put one position in the batch: its token, its hidden state, its
    /// position, and a request for logits.
    fn set(&mut self, token: LlamaToken, hidden: &[f32], pos: i32, seq: i32) {
        assert!(hidden.len() == self.n_embd, "hidden state is the wrong width");
        assert!(self.capacity >= 1);
        // SAFETY: every array was allocated with `capacity` entries and index
        // zero is within it; `hidden` is checked to be exactly one row wide.
        unsafe {
            *self.raw.token.add(0) = token.0;
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), self.raw.embd, self.n_embd);
            *self.raw.pos.add(0) = pos;
            *self.raw.n_seq_id.add(0) = 1;
            *(*self.raw.seq_id.add(0)).add(0) = seq;
            *self.raw.logits.add(0) = 1;
        }
        self.raw.n_tokens = 1;
    }
}

impl Drop for EmbdBatch {
    fn drop(&mut self) {
        // SAFETY: the token array was allocated here; everything else belongs
        // to llama.cpp and is freed by its own destructor.
        unsafe {
            if !self.raw.token.is_null() {
                libc_free(self.raw.token as *mut std::ffi::c_void);
                self.raw.token = std::ptr::null_mut();
            }
            sys::llama_batch_free(self.raw);
        }
    }
}

unsafe extern "C" {
    #[link_name = "malloc"]
    fn libc_malloc(size: usize) -> *mut std::ffi::c_void;
    #[link_name = "free"]
    fn libc_free(p: *mut std::ffi::c_void);
}

/// The model's NextN head, wired up to propose tokens.
pub struct MtpDrafter<'a> {
    min_confidence: f32,
    context: LlamaContext<'a>,
    batch: EmbdBatch,
    /// The conversation this drafter belongs to. Every cell it writes and
    /// every cell it takes back out is scoped to this sequence.
    seq: i32,
    n_embd: usize,
    n_vocab: i32,
}

impl<'a> MtpDrafter<'a> {
    /// Build a drafter over `model`, sharing `target`'s memory.
    ///
    /// Returns `None` when the model carries no NextN head, which is most
    /// models — that is not a failure, there is simply nothing to draft with.
    pub fn new(
        model: &'a LlamaModel,
        backend: &'static LlamaBackend,
        target: &LlamaContext<'_>,
        n_ctx: u32,
        seq: i32,
        n_seq_max: u32,
        unified: bool,
    ) -> Result<Option<Self>, EngineError> {
        // SAFETY: the model outlives this call.
        let heads = unsafe { sys::llama_model_n_layer_nextn(model.as_ptr()) };
        if heads <= 0 {
            return Ok(None);
        }
        let n_embd = model.n_embd() as usize;
        let params = LlamaContextParams::default()
            .with_context_type(LlamaContextType::Mtp)
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            // One token at a time, but never narrower than the number of
            // sequences: llama.cpp sizes its output allowance from `n_batch`
            // and then requires it to cover `n_seq_max`, so a batch of one on
            // a context that knows about five conversations trips an assert
            // before anything is decoded.
            .with_n_batch(env_u32("OZGENT_MTP_NBATCH").unwrap_or(n_seq_max.max(1)))
            // The draft context shares the target's memory, so it has to agree
            // with the target about how many sequences that memory holds and
            // how they are laid out. Left at the default it would refuse the
            // sequence id of every conversation but the first.
            .with_n_seq_max(env_u32("OZGENT_MTP_NSEQ").unwrap_or(n_seq_max))
            .with_kv_unified(env_u32("OZGENT_MTP_UNIFIED").map_or(unified, |v| v != 0))
            .with_embeddings(false);
        let context = model
            .new_context_with_ctx_other(backend, params, target)
            .map_err(|e| EngineError::Context(format!("mtp draft context: {e}")))?;
        // The head this drafter runs. Zero is the first, and models declaring
        // one head have only that.
        unsafe { crate::nextn::set_head(context.as_ptr(), 0) };
        // Masked on the draft side: only the position asking for logits needs
        // a hidden state back. The *target* is the one that must be unmasked.
        unsafe { crate::nextn::set_enabled(context.as_ptr(), true, true) };
        Ok(Some(Self {
            min_confidence: min_confidence(),
            batch: EmbdBatch::new(1, n_embd),
            context,
            seq,
            n_embd,
            n_vocab: model.n_vocab(),
        }))
    }

    /// Propose up to `want` tokens following `token`, whose hidden state in
    /// the target is `hidden`.
    ///
    /// Greedy, deliberately. A drafter exists to guess what the target would
    /// have said; sampling its own randomness into the guess only lowers the
    /// acceptance rate, and the target re-samples every token it keeps, so the
    /// output is the target's either way.
    pub fn propose(
        &mut self,
        token: LlamaToken,
        hidden: &[f32],
        pos: i32,
        want: usize,
    ) -> Result<Vec<LlamaToken>, EngineError> {
        let whole = std::time::Instant::now();
        let mut drafted = Vec::with_capacity(want);
        let mut token = token;
        let mut hidden = hidden.to_vec();

        for step in 0..want {
            self.batch.set(token, &hidden, pos + step as i32, self.seq);
            // SAFETY: the batch and context are both live.
            let started = std::time::Instant::now();
            let rc = unsafe { sys::llama_decode(self.context.as_ptr(), self.batch.raw) };
            // Read here so the timing covers the wait as well as the launch:
            // `llama_decode` queues and returns, and reading a row is what
            // synchronises.
            let logits = unsafe { sys::llama_get_logits_ith(self.context.as_ptr(), 0) };
            STEP_MICROS.fetch_add(
                started.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = logits;
            if rc != 0 {
                // Not an error worth failing the turn over: a draft that could
                // not be produced is a turn that decodes normally.
                tracing::debug!("mtp draft decode returned {rc} at step {step}");
                break;
            }
            let Some((next, confidence)) = self.argmax() else { break };
            // Stop as soon as the head stops being sure.
            //
            // This is what separates a NextN drafter from an n-gram one. An
            // n-gram proposal is a hash lookup and costs nothing, so proposing
            // five to land one is free. Every NextN proposal is a forward pass
            // through the block, so the same ratio is five times the cost for
            // one token of benefit — measured, that made drafting *slower*
            // than not drafting: 41.1 tok/s against 48.1, with 9 of 48
            // proposals accepted. Drafting only while the head is confident
            // spends the passes where they are likely to be kept.
            if confidence < self.min_confidence && !drafted.is_empty() {
                break;
            }
            drafted.push(next);

            // Chain: the head's own hidden state feeds the next step.
            match unsafe { crate::nextn::embedding(self.context.as_ptr(), 0, self.n_embd) } {
                Some(h) => hidden = h,
                None => break,
            }
            token = next;
        }
        // Take the draft back out of the cache.
        //
        // The draft context shares the target's memory — that is what makes it
        // cheap — which means every speculative decode writes into the cache
        // the target is about to use. Leaving those cells behind corrupts the
        // target's idea of what is resident: the next generation came back
        // "Decode Error -1: n_tokens == 0", because the position bookkeeping
        // on either side no longer agreed. The draft region is everything from
        // where this started, so that is what goes.
        //
        // SAFETY: the memory handle belongs to the live draft context.
        unsafe {
            let mem = sys::llama_get_memory(self.context.as_ptr());
            if !mem.is_null() {
                sys::llama_memory_seq_rm(mem, self.seq, pos, -1);
            }
        }
        PROPOSE_MICROS.fetch_add(
            whole.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(drafted)
    }

    /// The most likely token from the last decode, and how sure the head is.
    ///
    /// Confidence is the softmax probability of the chosen token, which is
    /// what decides whether another forward pass is worth spending.
    fn argmax(&self) -> Option<(LlamaToken, f32)> {
        // SAFETY: a decode requesting logits at index 0 just succeeded, so the
        // row exists and is `n_vocab` wide.
        let logits = unsafe { sys::llama_get_logits_ith(self.context.as_ptr(), 0) };
        if logits.is_null() || self.n_vocab <= 0 {
            return None;
        }
        let row = unsafe { std::slice::from_raw_parts(logits, self.n_vocab as usize) };
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        // Softmax over the row, shifted by the maximum so the exponentials
        // cannot overflow. Only the chosen token's share is wanted.
        let top = row[best];
        let total: f32 = row.iter().map(|v| (v - top).exp()).sum();
        let confidence = if total > 0.0 { 1.0 / total } else { 0.0 };
        Some((LlamaToken(best as i32), confidence))
    }
}
