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
//! **Only one sequence.** llama.cpp's own driver tracks a draft state per
//! sequence, with per-sequence batch bounds, pending hidden states and KV
//! region resets. ozgent decodes one sequence at a time, and carrying that
//! machinery for a case that does not arise would be several hundred lines of
//! bookkeeping whose failure mode is silently wrong output. If batching ever
//! lands, this grows with it.
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
    fn set(&mut self, token: LlamaToken, hidden: &[f32], pos: i32) {
        assert!(hidden.len() == self.n_embd, "hidden state is the wrong width");
        assert!(self.capacity >= 1);
        // SAFETY: every array was allocated with `capacity` entries and index
        // zero is within it; `hidden` is checked to be exactly one row wide.
        unsafe {
            *self.raw.token.add(0) = token.0;
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), self.raw.embd, self.n_embd);
            *self.raw.pos.add(0) = pos;
            *self.raw.n_seq_id.add(0) = 1;
            *(*self.raw.seq_id.add(0)).add(0) = 0;
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
    context: LlamaContext<'a>,
    batch: EmbdBatch,
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
            .with_n_batch(1)
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
            batch: EmbdBatch::new(1, n_embd),
            context,
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
        let mut drafted = Vec::with_capacity(want);
        let mut token = token;
        let mut hidden = hidden.to_vec();

        for step in 0..want {
            self.batch.set(token, &hidden, pos + step as i32);
            // SAFETY: the batch and context are both live.
            let rc = unsafe { sys::llama_decode(self.context.as_ptr(), self.batch.raw) };
            if rc != 0 {
                // Not an error worth failing the turn over: a draft that could
                // not be produced is a turn that decodes normally.
                tracing::debug!("mtp draft decode returned {rc} at step {step}");
                break;
            }
            let Some(next) = self.argmax() else { break };
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
                sys::llama_memory_seq_rm(mem, 0, pos, -1);
            }
        }
        Ok(drafted)
    }

    /// The most likely token from the last decode.
    fn argmax(&self) -> Option<LlamaToken> {
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
        Some(LlamaToken(best as i32))
    }
}
