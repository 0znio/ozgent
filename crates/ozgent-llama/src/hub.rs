//! One context, several conversations advancing through it together.
//!
//! A forward pass reads every weight in the model regardless of how many
//! tokens ride along. Decoding one token for four callers in four passes
//! therefore reads the weights four times to do work one pass could have
//! done, which is why a daemon that answers requests strictly in turn leaves
//! most of the card idle: it is bandwidth-bound on a batch of one.
//!
//! [`Hub`] is the thing that lets them share. It owns the context, and every
//! caller that wants a token hands its request to the hub instead of decoding
//! for itself. Whichever caller arrives to find nobody decoding becomes the
//! *driver*: it collects every request already waiting, merges them into a
//! single batch, runs one pass, hands each caller back its own logits, and
//! goes round again with whatever arrived in the meantime. Everyone else
//! sleeps until their row appears.
//!
//! Two properties matter and are worth stating plainly, because they are what
//! make this safe to put under a generation loop that already works:
//!
//! * **No token changes.** Every caller still gets the logits row produced for
//!   its own sequence, and still samples it with its own sampler. Riding along
//!   in a wider batch changes the order floats are reduced in, so a given
//!   token is not bit-identical to what a lone decode would have produced —
//!   the same caveat that already applies to any batched prefill — but nothing
//!   about the sampling decision is shared between callers.
//! * **No dedicated thread.** The driver is just whichever caller got there
//!   first. There is nothing to start, nothing to shut down, and a hub with
//!   one caller behaves exactly like calling `decode` directly, minus a
//!   mutex.
//!
//! Positions and sequence ids belong to the caller. The hub does not track
//! what is in the cache; it moves batches through the context and gives the
//! logits back.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::token::LlamaToken;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// A caller's request for one forward pass over some tokens.
pub struct Work {
    /// Which sequence in the shared cache these tokens belong to.
    pub seq: i32,
    /// The tokens, in order.
    pub tokens: Vec<LlamaToken>,
    /// Position of `tokens[0]` in that sequence.
    pub pos: i32,
    /// Which rows of logits the caller needs back.
    pub logits: Logits,
}

/// Which logits a request wants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Logits {
    /// None at all — a prefill chunk that is not the last one.
    None,
    /// The final token's, which is what generation needs.
    Last,
    /// Every token's, which is what verifying a draft needs.
    All,
}

/// What came back.
#[derive(Default)]
pub struct Outcome {
    /// One row per token whose logits were asked for, in token order.
    pub rows: Vec<Vec<f32>>,
}

impl Outcome {
    /// The last row, which is what a caller asking for [`Logits::Last`] wants.
    pub fn last(&self) -> Option<&[f32]> {
        self.rows.last().map(|r| r.as_slice())
    }

    pub fn into_last(mut self) -> Option<Vec<f32>> {
        self.rows.pop()
    }
}

#[derive(Debug)]
pub enum HubError {
    Decode(String),
    Batch(String),
    /// More tokens in one request than the context's batch can ever hold.
    TooWide { tokens: usize, n_batch: usize },
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode(e) => write!(f, "decode failed: {e}"),
            Self::Batch(e) => write!(f, "batch failed: {e}"),
            Self::TooWide { tokens, n_batch } => {
                write!(f, "{tokens} tokens do not fit a batch of {n_batch}")
            }
        }
    }
}

impl std::error::Error for HubError {}

/// Microseconds spent inside a pass, process-wide, for diagnostics.
pub static PASS_MICROS: AtomicU64 = AtomicU64::new(0);

struct Pending {
    id: u64,
    work: Work,
}

#[derive(Default)]
struct Queue {
    waiting: Vec<Pending>,
    ready: HashMap<u64, Result<Outcome, String>>,
    /// Whether somebody is currently running passes on everyone's behalf.
    driving: bool,
    next_id: u64,
    /// Passes run, and how many requests rode in them. The whole point of the
    /// exercise, so it is counted rather than inferred.
    passes: u64,
    merged: u64,
    /// Total seconds inside `decode`.
    spent: f64,
    /// Exponential moving average of what a pass costs, in seconds. The
    /// gather window is derived from it; see [`Hub::gather`].
    pass_secs: f64,
    /// Slots that are generating and so expected to ask for another token the
    /// moment they get one. The driver waits for these before running a pass;
    /// see [`Hub::gather`].
    running: usize,
}

/// The prefix every conversation on this hub begins with, held once.
///
/// A system prompt and a set of tool schemas are the same thousand-odd tokens
/// at the front of every turn, and each conversation was prefilling them from
/// cold. They only have to exist in the cache once: a sequence of their own
/// holds them, and a new conversation takes a copy by `seq_cp`, which for the
/// attention cells is a change of ownership rather than any recomputation.
///
/// Why a sequence of its own, rather than copying a prefix out of whichever
/// conversation happens to have one. llama.cpp's recurrent `seq_cp` ignores
/// the position range and copies the state as it stands *now* — there is only
/// one, at the end of the sequence. Copying the first four hundred tokens of a
/// conversation that has since run to four thousand would hand over a
/// recurrent state belonging to position four thousand, and the model would
/// continue from a place it had never been. Taking a whole sequence that holds
/// nothing but the shared prefix is the version of this that is true for every
/// architecture.
#[derive(Default)]
struct Commons {
    /// Exactly what the commons sequence holds, or empty.
    tokens: Vec<LlamaToken>,
    /// The previous prompt seen, so a shared prefix can be noticed without
    /// anybody declaring one.
    previous: Vec<LlamaToken>,
    /// Set while a slot is filling the commons, so the others do not all
    /// decide to do it at once.
    filling: bool,
}

/// A shared context several callers decode through.
pub struct Hub<'a> {
    /// Held only while a pass is actually running, by the driver alone.
    context: Mutex<LlamaContext<'a>>,
    queue: Mutex<Queue>,
    woke: Condvar,
    n_batch: usize,
    n_vocab: usize,
    slots: u32,
    commons: Mutex<Commons>,
    /// Set only when a measurement wants the window held still.
    fixed_window: Option<Duration>,
}

// A `LlamaContext` is a pointer into llama.cpp, which is happy to be used from
// any thread so long as only one is inside it at a time. The mutex is what
// guarantees that, and the hub never hands the context out.
unsafe impl Send for Hub<'_> {}
unsafe impl Sync for Hub<'_> {}

impl<'a> Hub<'a> {
    pub fn new(context: LlamaContext<'a>, n_vocab: usize, slots: u32) -> Self {
        let n_batch = (context.n_batch() as usize).max(1);
        Self {
            context: Mutex::new(context),
            queue: Mutex::new(Queue::default()),
            woke: Condvar::new(),
            n_batch,
            n_vocab,
            slots: slots.max(1),
            commons: Mutex::new(Commons::default()),
            fixed_window: None,
        }
    }

    /// Pin the gather window instead of deriving it. Only for measuring what
    /// the adaptive one is worth; see [`Hub::window`].
    pub fn with_window(mut self, window: Duration) -> Self {
        self.fixed_window = Some(window);
        self
    }

    /// Claim `seq` as a slot of this hub, for as long as the handle lives.
    ///
    /// Taken by `Arc` rather than by reference so a slot can be owned by the
    /// conversation using it — a [`crate::engine::Session`] holds its slot,
    /// and a session that borrowed the hub could not.
    pub fn slot(self: &Arc<Self>, seq: i32) -> Slot<'a> {
        Slot { hub: Arc::clone(self), seq, running: false }
    }

    pub fn n_batch(&self) -> usize {
        self.n_batch
    }

    /// How many conversations this hub was built to carry.
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// The sequence that holds the shared prefix, which is the one past the
    /// conversations.
    pub fn commons_seq(&self) -> i32 {
        self.slots as i32
    }

    /// What the shared prefix currently holds.
    pub fn commons(&self) -> Vec<LlamaToken> {
        self.commons.lock().unwrap().tokens.clone()
    }

    /// Note a prompt, and say what to do about the shared prefix.
    ///
    /// Returns the tokens a caller should put into the commons sequence, when
    /// this prompt reveals that the current one is missing something every
    /// conversation shares. Nobody declares the shared prefix: it is whatever
    /// two consecutive prompts turn out to begin with.
    pub fn consider(&self, prompt: &[LlamaToken]) -> Option<Vec<LlamaToken>> {
        let mut c = self.commons.lock().unwrap();
        let shared = shared_head(&c.previous, prompt);
        c.previous = prompt.to_vec();
        if c.filling || shared < MIN_COMMONS {
            return None;
        }
        // Only grown, and only by enough to be worth a rebuild: a commons that
        // chased every prompt would be refilled constantly and save nothing.
        if shared <= c.tokens.len() + c.tokens.len() / 4 && !c.tokens.is_empty() {
            return None;
        }
        // It must stay a prefix of what it already was, or conversations
        // holding a copy of the old one are describing a cache that no longer
        // matches.
        c.filling = true;
        Some(prompt[..shared].to_vec())
    }

    /// Publish what the commons sequence now holds, or give up on filling it.
    pub fn filled(&self, tokens: Vec<LlamaToken>) {
        let mut c = self.commons.lock().unwrap();
        c.tokens = tokens;
        c.filling = false;
    }

    /// Put `tokens` into the commons sequence and publish them.
    ///
    /// Costs one prefill of the same tokens the calling turn was about to
    /// prefill anyway — it then borrows them straight back — so the first
    /// conversation to notice a shared prefix pays nothing for it and every
    /// later one starts from it for free.
    pub fn fill_commons(&self, tokens: &[LlamaToken]) -> Result<(), HubError> {
        let seq = self.commons_seq();
        self.with_context(|c| {
            let _ = c.clear_kv_cache_seq(Some(seq as u32), None, None);
        });
        let mut pos = 0i32;
        for chunk in tokens.chunks(self.n_batch) {
            let last = pos as usize + chunk.len() == tokens.len();
            // The final row is asked for so the pass always has an output;
            // nothing reads it.
            let want = if last { Logits::Last } else { Logits::None };
            if let Err(e) = self.run(Work { seq, tokens: chunk.to_vec(), pos, logits: want }) {
                self.filled(Vec::new());
                return Err(e);
            }
            pos += chunk.len() as i32;
        }
        self.filled(tokens.to_vec());
        Ok(())
    }

    /// Hand a copy of the shared prefix to `seq`, which must hold nothing.
    pub fn lend(&self, seq: i32) -> Result<usize, HubError> {
        let c = self.commons.lock().unwrap();
        if c.tokens.is_empty() {
            return Ok(0);
        }
        let n = c.tokens.len();
        let from = self.commons_seq();
        self.with_context(|ctx| {
            let _ = ctx.clear_kv_cache_seq(Some(seq as u32), None, None);
            ctx.kv_cache_seq_cp(from, seq, None, None)
                .map_err(|e| HubError::Decode(e.to_string()))
        })?;
        Ok(n)
    }

    /// True when this hub carries one conversation, so its context is not
    /// shared with anybody.
    ///
    /// The paths that reach past the hub for a raw context pointer — drafting
    /// from a NextN head, which opens a second context over the same memory —
    /// are only sound then.
    pub fn solo(&self) -> bool {
        self.slots <= 1
    }

    /// The raw context, for the few things llama.cpp only exposes that way.
    ///
    /// Sound only while nothing else is using the context: check [`solo`]
    /// first, or hold the lock through [`with_context`].
    ///
    /// [`solo`]: Hub::solo
    /// [`with_context`]: Hub::with_context
    pub fn raw(&self) -> *mut ozgent_mtmd_sys::llama_cpp_sys_2::llama_context {
        self.context.lock().unwrap().as_ptr()
    }

    /// Passes run and requests carried, since the hub was made.
    ///
    /// `merged / passes` is the average batch width: 1.0 means nothing ever
    /// shared a pass and the hub bought nothing.
    pub fn traffic(&self) -> (u64, u64) {
        let q = self.queue.lock().unwrap();
        (q.passes, q.merged)
    }

    /// Seconds spent inside `decode`, and passes run, since the last reset.
    /// Separated from wall time so a slow pass can be told from a slow slot.
    pub fn spent(&self) -> (f64, u64) {
        let q = self.queue.lock().unwrap();
        (q.spent, q.passes)
    }

    pub fn reset_traffic(&self) {
        let mut q = self.queue.lock().unwrap();
        q.passes = 0;
        q.merged = 0;
        q.spent = 0.0;
    }

    /// Reach the context directly, for the operations that are not decodes —
    /// clearing a sequence, reading state, evaluating images.
    ///
    /// Blocks while a pass is in flight, which is the point: these all mutate
    /// cache the pass is reading.
    pub fn with_context<T>(&self, f: impl FnOnce(&mut LlamaContext<'a>) -> T) -> T {
        let mut ctx = self.context.lock().unwrap();
        f(&mut ctx)
    }

    /// Run `work` through the model, sharing a pass with whoever else is
    /// waiting, and return its logits.
    pub fn run(&self, work: Work) -> Result<Outcome, HubError> {
        if work.tokens.len() > self.n_batch {
            return Err(HubError::TooWide { tokens: work.tokens.len(), n_batch: self.n_batch });
        }

        let id = {
            let mut q = self.queue.lock().unwrap();
            let id = q.next_id;
            q.next_id += 1;
            q.waiting.push(Pending { id, work });
            self.woke.notify_all();
            id
        };

        loop {
            // Either my answer is here, or somebody is producing it, or
            // nobody is and it falls to me.
            {
                let mut q = self.queue.lock().unwrap();
                loop {
                    if let Some(done) = q.ready.remove(&id) {
                        return done.map_err(HubError::Decode);
                    }
                    if !q.driving {
                        q.driving = true;
                        break;
                    }
                    q = self.woke.wait(q).unwrap();
                }
            }

            // I am the driver until my own request is answered. Anything that
            // arrives mid-pass is picked up by the next round, which is what
            // makes the batching continuous rather than a fixed window.
            let outcome = self.drive(id);

            let mut q = self.queue.lock().unwrap();
            q.driving = false;
            self.woke.notify_all();
            match outcome {
                Some(done) => return done.map_err(HubError::Decode),
                // A pass failed for somebody else, or nothing was runnable;
                // the error is already filed against whoever owned it. Go
                // round and wait to be driven.
                None => continue,
            }
        }
    }

    /// Run passes until `mine` has an answer. Returns it, or `None` if the
    /// queue emptied without producing one.
    fn drive(&self, mine: u64) -> Option<Result<Outcome, String>> {
        loop {
            let batch = {
                let mut q = self.gather();
                let taken = take_batch(&mut q.waiting, self.n_batch);
                if taken.is_empty() {
                    return q.ready.remove(&mine);
                }
                q.passes += 1;
                q.merged += taken.len() as u64;
                taken
            };
            // Sorted by sequence, which is not cosmetic.
            //
            // llama.cpp splits a batch into micro-batches before running it,
            // and for a cache with one stream per sequence the rule it uses is
            // "accept only increasing sequence ids". A batch in arrival order
            // therefore breaks into one micro-batch per slot — and a
            // micro-batch is a forward pass, so the weights are read once per
            // slot and batching buys exactly nothing. That is not a subtle
            // effect: it cost a measured 2.0x at two slots, the whole win.
            // Sorting turns the same requests into one micro-batch.
            let mut batch = batch;
            batch.sort_by_key(|p| p.work.seq);

            let started = Instant::now();
            let results = self.pass(&batch);
            let took = started.elapsed().as_secs_f64();

            let mut q = self.queue.lock().unwrap();
            // Smoothed, because a pass that happened to wait on the queue lock
            // says nothing about what the model costs.
            q.pass_secs = if q.pass_secs <= 0.0 { took } else { q.pass_secs * 0.8 + took * 0.2 };
            q.spent += took;
            for (p, result) in batch.into_iter().zip(results) {
                q.ready.insert(p.id, result);
            }
            self.woke.notify_all();
            if let Some(done) = q.ready.remove(&mine) {
                return Some(done);
            }
        }
    }

    /// Hold the pass open until every running slot has asked for its token.
    ///
    /// Without this the hub batches almost nothing, and the reason is worth
    /// writing down because the first measurement was a flat 1.05x. Slots do
    /// not arrive together: a pass ends, every slot wakes, and each then
    /// samples and detokenises before asking for its next token. The driver,
    /// already awake and holding the queue, finds it empty and runs a pass of
    /// one — so four slots decoding in lockstep still paid for four passes.
    ///
    /// The fix is to wait for the field, but only for slots that are actually
    /// coming: a slot that has stopped to run a tool parks itself and is not
    /// counted. The wait is bounded so a slot that parks without saying so
    /// costs one short delay rather than a hang.
    ///
    /// How long to wait is not a constant, and a first attempt that made it
    /// one measured the point: a millisecond and a half gathered two slots of
    /// four, while four milliseconds gathered all four. The right bound
    /// depends on the model — on a small one the slots' own sampling is slower
    /// than the pass, on a large one far faster — so it is derived from what
    /// a pass has been costing rather than picked.
    fn gather(&self) -> std::sync::MutexGuard<'_, Queue> {
        let mut q = self.queue.lock().unwrap();
        if q.waiting.is_empty() {
            return q;
        }
        let deadline = Instant::now() + self.window(&q);
        while field_incomplete(q.waiting.len(), q.running) {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            let (guard, _) = self.woke.wait_timeout(q, deadline - now).unwrap();
            q = guard;
        }
        q
    }

    /// How long to hold a pass open for slots that have not arrived.
    ///
    /// Waiting `t` to carry `w` requests instead of one is profitable whenever
    /// `t < (w - 1) * pass`, since the wide pass costs about what the narrow
    /// one did. Even at `w = 2` that allows a whole pass of waiting, so a
    /// fraction of one is a conservative bound that cannot lose much: at worst
    /// the slots were not coming and the hub paid `WINDOW_SHARE` of a pass,
    /// once, before running exactly the batch it would have run anyway.
    ///
    /// Before any pass has been timed there is nothing to scale against, so
    /// the floor stands in.
    fn window(&self, q: &Queue) -> Duration {
        if let Some(fixed) = self.fixed_window {
            return fixed;
        }
        if q.pass_secs <= 0.0 {
            return MIN_WINDOW;
        }
        Duration::from_secs_f64(q.pass_secs * WINDOW_SHARE).clamp(MIN_WINDOW, MAX_WINDOW)
    }

    /// One forward pass carrying every request in `batch`.
    fn pass(&self, batch: &[Pending]) -> Vec<Result<Outcome, String>> {
        let total: usize = batch.iter().map(|p| p.work.tokens.len()).sum();
        let mut llama = LlamaBatch::new(total.max(1), 1);
        // Where each request's final token landed, so its logits can be found
        // again once the pass is done.
        let mut rows: Vec<Vec<i32>> = Vec::with_capacity(batch.len());
        let mut filled = 0i32;

        for p in batch {
            let last = p.work.tokens.len().saturating_sub(1);
            let mut mine = Vec::new();
            for (i, token) in p.work.tokens.iter().enumerate() {
                let wants = match p.work.logits {
                    Logits::None => false,
                    Logits::Last => i == last,
                    Logits::All => true,
                };
                if let Err(e) = llama.add(*token, p.work.pos + i as i32, &[p.work.seq], wants) {
                    return batch.iter().map(|_| Err(format!("batch failed: {e}"))).collect();
                }
                if wants {
                    mine.push(filled + i as i32);
                }
            }
            rows.push(mine);
            filled += p.work.tokens.len() as i32;
        }

        let mut ctx = self.context.lock().unwrap();
        let started = Instant::now();
        if let Err(e) = ctx.decode(&mut llama) {
            return batch.iter().map(|_| Err(e.to_string())).collect();
        }

        // Copied out rather than borrowed: the caller samples on its own
        // thread, long after this pass has been overwritten by the next one.
        //
        // The copy itself is cheap — under a millisecond for four rows of a
        // 150k vocabulary. What is not cheap is the first read, because
        // `llama_get_logits_ith` synchronises with the scheduler: `decode`
        // queues work and returns, so the GPU is waited on here rather than
        // there. Anything timing `decode` alone is timing submission.
        let out = rows
            .iter()
            .map(|mine| {
                let rows = mine
                    .iter()
                    .map(|&row| {
                        let slice = ctx.get_logits_ith(row);
                        slice[..self.n_vocab.min(slice.len())].to_vec()
                    })
                    .collect();
                Ok(Outcome { rows })
            })
            .collect();
        PASS_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        out
    }
}

/// The shortest prefix worth holding in a sequence of its own.
///
/// Below this the copy costs more bookkeeping than the prefill it saves.
const MIN_COMMONS: usize = 128;

/// How far two token sequences agree from the start.
fn shared_head(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Whether the driver should keep holding a pass open for slots that have not
/// asked yet.
///
/// The whole correctness of the gather window rests on `running` counting only
/// slots that are actually coming back. A slot that finished its turn and did
/// not park is still counted, so the driver waits the full window for it on
/// every pass — measured at 33.4 tok/s against 45.3 for a lone caller on a
/// four-slot context, because three slots that had finished long before were
/// still being waited for.
fn field_incomplete(waiting: usize, running: usize) -> bool {
    waiting < running
}

/// Take as many waiting requests as one batch can carry, oldest first.
///
/// Oldest first because a request that keeps losing to newer arrivals never
/// finishes. The first request is taken whatever its size — it has already
/// been checked against `n_batch`, so refusing it here would hang it forever.
fn take_batch(waiting: &mut Vec<Pending>, n_batch: usize) -> Vec<Pending> {
    let mut taken = Vec::new();
    let mut width = 0usize;
    let mut i = 0;
    while i < waiting.len() {
        let len = waiting[i].work.tokens.len();
        if taken.is_empty() || width + len <= n_batch {
            width += len;
            taken.push(waiting.remove(i));
        } else {
            i += 1;
        }
    }
    taken
}

/// What fraction of a pass the driver will spend waiting for the field.
///
/// See [`Hub::window`] for why anything below 1.0 is conservative.
const WINDOW_SHARE: f64 = 0.5;

/// Floor and ceiling on that wait. The floor covers the first pass, before
/// there is any timing to scale against; the ceiling stops a very slow model
/// from making a parked slot's mistake expensive.
const MIN_WINDOW: Duration = Duration::from_micros(250);
const MAX_WINDOW: Duration = Duration::from_millis(20);

/// One conversation's claim on a hub.
///
/// Counted as running while it is generating, so the driver knows to wait for
/// it, and parked while it is doing anything else — running a tool, waiting on
/// a user — so it does not hold everyone else up.
pub struct Slot<'a> {
    hub: Arc<Hub<'a>>,
    seq: i32,
    running: bool,
}

impl<'a> Slot<'a> {
    pub fn seq(&self) -> i32 {
        self.seq
    }

    pub fn hub(&self) -> &Hub<'a> {
        &self.hub
    }

    /// Say that this slot is generating and will keep asking for tokens.
    pub fn resume(&mut self) {
        if !self.running {
            self.running = true;
            self.hub.queue.lock().unwrap().running += 1;
        }
    }

    /// Say that this slot has stopped asking, so nobody waits for it.
    pub fn park(&mut self) {
        if self.running {
            self.running = false;
            let mut q = self.hub.queue.lock().unwrap();
            q.running = q.running.saturating_sub(1);
            // A driver may be holding a pass open for this slot right now.
            self.hub.woke.notify_all();
        }
    }

    /// Decode `tokens` at `pos` on this slot's sequence.
    pub fn run(
        &mut self,
        tokens: Vec<LlamaToken>,
        pos: i32,
        logits: Logits,
    ) -> Result<Outcome, HubError> {
        self.resume();
        self.hub.run(Work { seq: self.seq, tokens, pos, logits })
    }

    /// Reach the context for the operations that are not decodes.
    pub fn with_context<T>(&self, f: impl FnOnce(&mut LlamaContext<'a>) -> T) -> T {
        self.hub.with_context(f)
    }

    /// Whether this slot may take device-resident snapshots of its sequence.
    ///
    /// llama.cpp caches one buffer per *context* for these, and reuses it
    /// whenever the next state happens to be the same total size. Two
    /// sequences taking snapshots in turn therefore hand each other a buffer
    /// laid out for the other's cells, and llama.cpp answers that by aborting
    /// the process — which is what four concurrent drafting turns did, inside
    /// ggml's allocator, on the first round.
    ///
    /// Granting the right to exactly one slot is not enough, which was worth
    /// finding out: with only sequence 0 snapshotting and the other three
    /// merely decoding, four concurrent turns still aborted. Other sequences
    /// writing to the same cache is what invalidates the views, not only
    /// other sequences snapshotting.
    ///
    /// So the right belongs to a slot with the context to itself. A model
    /// whose cache *can* trim a rejected draft never needs this and
    /// speculates freely however many slots there are.
    pub fn may_snapshot(&self) -> bool {
        self.hub.solo()
    }

    /// Drop positions `from..` from this slot's sequence. False means this
    /// cache cannot drop a partial range, which sliding-window and recurrent
    /// caches cannot.
    pub fn trim(&self, from: i32) -> Result<bool, String> {
        self.with_context(|c| {
            c.clear_kv_cache_seq(Some(self.seq as u32), Some(from as u32), None)
                .map_err(|e| e.to_string())
        })
    }

    /// Forget everything cached for this slot, leaving the others alone.
    pub fn clear(&self) {
        self.hub.with_context(|c| {
            let _ = c.clear_kv_cache_seq(Some(self.seq as u32), None, None);
        });
    }
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pending(id: u64, tokens: usize) -> Pending {
        Pending {
            id,
            work: Work {
                seq: 0,
                tokens: vec![LlamaToken(1); tokens],
                pos: 0,
                logits: Logits::Last,
            },
        }
    }

    #[test]
    fn a_shared_head_stops_at_the_first_difference() {
        let a: Vec<LlamaToken> = [1, 2, 3, 4].iter().map(|i| LlamaToken(*i)).collect();
        let b: Vec<LlamaToken> = [1, 2, 9, 4].iter().map(|i| LlamaToken(*i)).collect();
        assert_eq!(shared_head(&a, &b), 2);
        assert_eq!(shared_head(&a, &a), 4);
        assert_eq!(shared_head(&a, &[]), 0);
        assert_eq!(shared_head(&a[..2], &a), 2, "a shorter sequence bounds it");
    }

    #[test]
    fn the_driver_waits_only_while_somebody_is_still_coming() {
        // Two slots generating, one has asked: wait for the other.
        assert!(field_incomplete(1, 2));
        // Both have asked: nothing left to wait for.
        assert!(!field_incomplete(2, 2));
        // The bug this guards: a slot that finished its turn without parking
        // is still counted as running, so the driver waits a window per pass
        // for a caller that will never arrive.
        assert!(!field_incomplete(1, 1), "a parked slot must not be waited for");
        // More waiting than running is possible when a slot queues again
        // before another has parked; it must not wait.
        assert!(!field_incomplete(3, 2));
    }

    #[test]
    fn a_full_batch_of_single_tokens_is_taken_whole() {
        let mut waiting: Vec<_> = (0..8).map(|i| pending(i, 1)).collect();
        let taken = take_batch(&mut waiting, 32);
        assert_eq!(taken.len(), 8);
        assert!(waiting.is_empty());
    }

    #[test]
    fn what_does_not_fit_waits_for_the_next_pass() {
        let mut waiting = vec![pending(0, 24), pending(1, 24), pending(2, 4)];
        let taken = take_batch(&mut waiting, 32);
        // The first is taken, the second does not fit beside it, the third
        // does — so a wide prefill does not block a narrow decode behind it.
        assert_eq!(taken.iter().map(|p| p.id).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting[0].id, 1);
    }

    #[test]
    fn an_oversized_first_request_is_still_taken() {
        // It was already checked against `n_batch` on the way in; leaving it
        // would hang the caller rather than fail it.
        let mut waiting = vec![pending(0, 64)];
        let taken = take_batch(&mut waiting, 32);
        assert_eq!(taken.len(), 1);
    }

    #[test]
    fn nothing_waiting_is_an_empty_pass() {
        let mut waiting: Vec<Pending> = Vec::new();
        assert!(take_batch(&mut waiting, 32).is_empty());
    }
}
