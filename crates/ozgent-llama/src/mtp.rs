//! Drafting from the model's own multi-token-prediction head.
//!
//! Qwen3.5 and its relatives append a trained NextN block past the end of the
//! main stack. Fed the token just chosen and the target's hidden state at the
//! position before it, that block predicts the token after. So the model
//! carries its own drafter: no second set of weights to choose, and a draft
//! drawn from the model's own distribution rather than from repetition in the
//! context — the case n-gram drafting cannot serve at all.
//!
//! **What it is worth.** Qwen3.5-4B-MTP on an RTX 5050 Laptop, greedy, one
//! drafted token per round, through the engine against plain decoding in the
//! same process, with the text identical either way:
//!
//! ```text
//!   prompt   plain       drafted     accepted
//!   code     57.0 tok/s  80.8 tok/s    88%
//!   math     56.8        80.8          90%
//!   Hindi    57.0        75.7          76%
//!   prose    56.8        72.2          69%
//! ```
//!
//! Four conversations at once through one context: 133 tok/s against 188
//! together. Qwen3.6-35B-A3B-MTP with its experts in system memory: code 27.9
//! against 38.1, prose 30.2 against 35.6.
//!
//! One token, not more. Every draft step reads the whole output head — 521
//! MB of a 248k vocabulary — and costs about 4 ms against an 18.7 ms target
//! pass, while a second drafted token lands only half the time. Two cost more
//! than they returned at every confidence threshold swept (0.6 to 0.9).
//!
//! **Why the first attempt lost, 43 against 48 tok/s.** Three things, each
//! measured when this was rebuilt:
//!
//! * The draft context was believed to share the target's memory. For this
//!   architecture llama.cpp gives it a KV cache of its own for the NextN
//!   layer, and nothing ever filled it: the head drafted without being able
//!   to attend to a single earlier position. It now absorbs every position
//!   the target decodes ([`Drafter::absorb`]) — worth 6 points of acceptance
//!   and 5% of speed, for 2.5% of prefill time.
//! * Rejected drafts were undone by snapshotting the whole sequence state and
//!   restoring it, which undoes the confirmed token too and costs a pass to
//!   redo. llama.cpp's `n_rs_seq` ring keeps the recurrent state after each
//!   of the last few tokens of a batch, so a rejection is a trim, exact to
//!   the bit on a prefilled prompt.
//! * The head was sampled with a softmax over the whole vocabulary to gate
//!   on confidence, which a one-token draft does not need.
//!
//! **Output.** The target samples every token it keeps, so a draft can change
//! only how fast the answer arrives. Not quite bit-for-bit: a verification
//! batch of two tokens reduces floats in a different order than a batch of
//! one, and where the model's top two choices are within a twentieth of a
//! logit of each other (the median gap is 2.7) the other one can win. The same
//! is true of every change of batch shape llama.cpp makes, prefill included.
//! Measured with an oracle drafter: perfect drafts reproduced 400 tokens of
//! plain decoding exactly; drafts that were always wrong, forcing a rollback
//! every round, first differed at such a near-tie after 264 tokens.
//!
//! **One drafter per context.** The hub owns it, beside the context it drafts
//! for, and every conversation drafts into its own sequence of it. Its cost is
//! one layer's cache and a small scratch, not a copy per conversation.

use std::collections::{HashMap, VecDeque};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;

use crate::engine::EngineError;

/// Tokens drafted per round. See the module documentation for why one.
pub const DRAFT_TOKENS: usize = 1;

/// Hidden states kept per sequence, most recent positions. A verification
/// batch is at most `1 + DRAFT_TOKENS` rows and only the row before the next
/// batch is ever looked up, so a handful covers any rollback.
const RECENT: usize = 8;

/// Rows of logits a draft decode asks for at most. The scratch is sized for
/// the worst case — every row of a micro-batch producing a full vocabulary of
/// logits — which at the default 512 is half a gigabyte for nothing.
const DRAFT_OUTPUTS: u32 = 8;

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
        let mut raw = unsafe { sys::llama_batch_init(capacity as i32, n_embd as i32, 1) };
        // SAFETY: `llama_batch_init` left `token` null because an embedding
        // width was given. The NextN graph reads both, so it is allocated here
        // and released in Drop alongside the rest.
        raw.token = unsafe {
            libc_malloc(std::mem::size_of::<sys::llama_token>() * capacity) as *mut sys::llama_token
        };
        Self { raw, capacity, n_embd }
    }

    fn clear(&mut self) {
        self.raw.n_tokens = 0;
    }

    fn len(&self) -> usize {
        self.raw.n_tokens as usize
    }

    /// Append one position. `hidden` of `None` is a row of zeros: the
    /// target's state there was never seen, which costs the head a little
    /// context and nothing else.
    fn push(&mut self, token: LlamaToken, hidden: Option<&[f32]>, pos: i32, seq: i32, logits: bool) {
        let i = self.len();
        assert!(i < self.capacity, "draft batch overflow");
        // SAFETY: every array was allocated with `capacity` entries and `i`
        // is below it; a hidden row is exactly `n_embd` wide.
        unsafe {
            *self.raw.token.add(i) = token.0;
            let row = self.raw.embd.add(i * self.n_embd);
            match hidden {
                Some(h) if h.len() == self.n_embd => {
                    std::ptr::copy_nonoverlapping(h.as_ptr(), row, self.n_embd)
                }
                _ => std::ptr::write_bytes(row, 0, self.n_embd),
            }
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = 1;
            *(*self.raw.seq_id.add(i)).add(0) = seq;
            *self.raw.logits.add(i) = logits as i8;
        }
        self.raw.n_tokens += 1;
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

/// What the target decoded for one sequence in one pass, for the drafter to
/// absorb: `tokens` at `pos..`, whose hidden states are rows `first..` of the
/// pass that produced them.
pub struct Absorbed<'t> {
    pub seq: i32,
    pub pos: i32,
    pub tokens: &'t [LlamaToken],
    pub first: i32,
}

/// The model's NextN head, with a cache of its own, drafting for every
/// conversation on one context.
pub struct Drafter<'a> {
    context: LlamaContext<'a>,
    batch: EmbdBatch,
    n_embd: usize,
    n_vocab: usize,
    /// The target's hidden states at the latest positions of each sequence.
    /// A position's row is what the head is fed beside the token *after* it.
    recent: HashMap<i32, VecDeque<(i32, Vec<f32>)>>,
}

// Like the hub's own context: a pointer into llama.cpp, used by one thread at
// a time because the hub keeps it behind a mutex.
unsafe impl Send for Drafter<'_> {}

impl<'a> Drafter<'a> {
    /// Open a draft context over `model`, which must have been loaded with its
    /// MTP layers. `None` when the model has no head, which is most models.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &'a LlamaModel,
        backend: &'static LlamaBackend,
        n_ctx: u32,
        n_batch: u32,
        n_seq_max: u32,
        unified: bool,
        flash: bool,
    ) -> Result<Option<Self>, EngineError> {
        // SAFETY: the model outlives this call.
        if unsafe { sys::llama_model_n_layer_nextn(model.as_ptr()) } <= 0 {
            return Ok(None);
        }
        let n_embd = model.n_embd() as usize;
        let params = LlamaContextParams::default()
            .with_context_type(LlamaContextType::Mtp)
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            // Wide enough to absorb a whole prefill chunk in one decode.
            .with_n_batch(n_batch)
            .with_n_outputs_max(DRAFT_OUTPUTS)
            // Sequence ids are the target's, so the two must agree on how many
            // there are and how their cells are laid out.
            .with_n_seq_max(n_seq_max)
            .with_kv_unified(unified)
            .with_flash_attention_policy(if flash {
                sys::LLAMA_FLASH_ATTN_TYPE_AUTO
            } else {
                sys::LLAMA_FLASH_ATTN_TYPE_DISABLED
            })
            .with_type_k(llama_cpp_2::context::params::KvCacheType::Q8_0)
            .with_type_v(if flash {
                llama_cpp_2::context::params::KvCacheType::Q8_0
            } else {
                llama_cpp_2::context::params::KvCacheType::F16
            })
            .with_embeddings(false);
        let context = model
            .new_context(backend, params)
            .map_err(|e| EngineError::Context(format!("mtp draft context: {e}")))?;
        // SAFETY: the context was just created. Masked on the draft side: only
        // the row asking for logits needs a hidden state back.
        unsafe {
            crate::nextn::set_head(context.as_ptr(), 0);
            crate::nextn::set_enabled(context.as_ptr(), true, true);
        }
        Ok(Some(Self {
            batch: EmbdBatch::new(n_batch.max(2) as usize, n_embd),
            context,
            n_embd,
            n_vocab: model.n_vocab() as usize,
            recent: HashMap::new(),
        }))
    }

    /// Take in what the target just decoded, so the head can attend to it.
    ///
    /// `target` is the context that ran the pass, with NextN embeddings on
    /// and unmasked, so row `i` of it is the hidden state at batch index `i`.
    /// Each position is fed with the target's state at the position before
    /// it — shifted by one, which is what the head was trained on.
    ///
    /// # Safety
    /// `target` must be the live context whose last decode produced these
    /// rows, and nothing may have decoded on it since.
    pub unsafe fn absorb(&mut self, target: *mut sys::llama_context, runs: &[Absorbed<'_>]) {
        let mem = unsafe { sys::llama_get_memory(self.context.as_ptr()) };
        let row = |i: i32| unsafe { crate::nextn::embedding(target, i, self.n_embd) };
        self.batch.clear();
        for run in runs {
            if run.tokens.is_empty() {
                continue;
            }
            // Whatever the head held from here on described positions this
            // sequence has since rewritten: a rejected draft, a new turn.
            unsafe { sys::llama_memory_seq_rm(mem, run.seq, run.pos, -1) };
            let recent = self.recent.entry(run.seq).or_default();
            recent.retain(|(p, _)| *p < run.pos);
            let before = recent.iter().find(|(p, _)| *p == run.pos - 1).map(|(_, h)| h.clone());
            let mut previous = before;
            for (j, token) in run.tokens.iter().enumerate() {
                if self.batch.len() == self.batch.capacity {
                    break;
                }
                self.batch.push(*token, previous.as_deref(), run.pos + j as i32, run.seq, false);
                previous = row(run.first + j as i32);
            }
            // Only the tail is ever looked up again.
            let n = run.tokens.len();
            let mut kept = 0usize;
            for j in n.saturating_sub(RECENT)..n {
                if let Some(h) = row(run.first + j as i32) {
                    recent.push_back((run.pos + j as i32, h));
                    kept += 1;
                }
            }
            tracing::trace!(
                "mtp absorbed seq {} pos {}..{} ({} rows, {kept} hidden states kept)",
                run.seq,
                run.pos,
                run.pos + n as i32,
                n
            );
            while recent.len() > RECENT {
                recent.pop_front();
            }
        }
        if self.batch.len() == 0 {
            return;
        }
        // SAFETY: the batch and context are live.
        let rc = unsafe { sys::llama_decode(self.context.as_ptr(), self.batch.raw) };
        if rc != 0 {
            // A head that fell behind drafts worse, and that is all.
            tracing::debug!("mtp catch-up decode returned {rc}");
        }
    }

    /// Guess the token after each request's `token`, which its sequence is
    /// about to decode at `pos`, in one decode for all of them: every draft
    /// step reads the whole output head, so four conversations drafting
    /// together pay for that read once. `None` for a sequence whose state
    /// before `pos` was never seen, or when the head could not run.
    pub fn propose(&mut self, requests: &[(i32, LlamaToken, i32)]) -> Vec<Option<LlamaToken>> {
        let mut out = vec![None; requests.len()];
        let mut rows = Vec::with_capacity(requests.len());
        self.batch.clear();
        for (i, &(seq, token, pos)) in requests.iter().enumerate() {
                let Some(hidden) = self
                .recent
                .get(&seq)
                .and_then(|r| r.iter().find(|(p, _)| *p == pos - 1))
                .map(|(_, h)| h.clone())
            else {
                tracing::trace!("mtp has no hidden state for seq {seq} at {}", pos - 1);
                continue;
            };
            if self.batch.len() == self.batch.capacity || rows.len() as u32 >= DRAFT_OUTPUTS {
                break;
            }
            self.batch.push(token, Some(&hidden), pos, seq, true);
            rows.push(i);
        }
        if rows.is_empty() {
            return out;
        }
        // SAFETY: the batch and context are live.
        let rc = unsafe { sys::llama_decode(self.context.as_ptr(), self.batch.raw) };
        if rc != 0 {
            tracing::debug!("mtp draft decode returned {rc}");
            return out;
        }
        for (row, &i) in rows.iter().enumerate() {
            // SAFETY: the decode asked for logits on every row it carried.
            let logits = unsafe { sys::llama_get_logits_ith(self.context.as_ptr(), row as i32) };
            if logits.is_null() {
                continue;
            }
            let row = unsafe { std::slice::from_raw_parts(logits, self.n_vocab) };
            let mut best = 0usize;
            for (j, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = j;
                }
            }
            out[i] = Some(LlamaToken(best as i32));
        }
        out
    }

    /// Forget `seq` from `from` on. The cells go now; the hidden states the
    /// next absorb would have dropped anyway.
    pub fn forget(&mut self, seq: i32, from: i32) {
        // SAFETY: the context is live.
        unsafe {
            let mem = sys::llama_get_memory(self.context.as_ptr());
            sys::llama_memory_seq_rm(mem, seq, from.max(0), -1);
        }
        if let Some(r) = self.recent.get_mut(&seq) {
            r.retain(|(p, _)| *p < from);
        }
    }

    /// Give `to` a copy of what `from` holds, as the target's shared prefix is
    /// lent to a conversation.
    pub fn copy(&mut self, from: i32, to: i32) {
        self.forget(to, 0);
        // SAFETY: the context is live.
        unsafe {
            let mem = sys::llama_get_memory(self.context.as_ptr());
            sys::llama_memory_seq_cp(mem, from, to, -1, -1);
        }
        if let Some(r) = self.recent.get(&from).cloned() {
            self.recent.insert(to, r);
        }
    }
}
