//! Several models resident at once, bounded by the memory that exists.
//!
//! One model used to be loaded at a time, and a request naming a different one
//! swapped it. That is correct and, once the terminal became a client too,
//! wrong in practice: a browser on one model and a terminal on another made
//! every alternation a full unload and reload. Measured before this existed —
//! five requests alternating between two models produced four loads.
//!
//! So models are held in a pool, one thread each, and a request goes to the
//! thread that already has its model.
//!
//! # Why a thread per model
//!
//! Not a taste in concurrency. The engine and the session that borrows it live
//! in one stack frame *deliberately* — see [`crate::worker`] — so that the
//! borrow checker is satisfied without a self-referential struct. Holding
//! several engines in one collection would break exactly that. Giving each its
//! own thread keeps every engine in its own frame, and the problem never
//! arises. Two models can then also generate at the same time, which a single
//! thread could not do at all.
//!
//! # There is no maximum
//!
//! Not a count, anyway. A count is always wrong for somebody: three is
//! generous on an 8 GB laptop and absurd on a 64 GB card where ten models fit
//! comfortably. The limit is the memory actually free at the moment of
//! loading, read from the driver — which also accounts for every other process
//! on the card, something bookkeeping of our own could never do.
//!
//! What happens when a model does not fit is the interesting part, and it is
//! not "refuse":
//!
//! 1. Offload what fits. A model half on the GPU is slower than one wholly on
//!    it and enormously faster than nothing, and it is what the person asked
//!    for. [`ozgent_llama::backend::Plan`] decides how much.
//! 2. If that would be a *poor* fit, first evict models nobody is using and
//!    try again — an idle model holding VRAM is worth less than the one being
//!    asked for right now.
//! 3. Never evict a model that is mid-generation. Somebody is reading it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use crate::worker::Job;

/// Below this share of a model on the GPU, making room is worth a try.
///
/// Not a tuned constant so much as a statement about when eviction pays.
/// Losing a third of the layers to the CPU is a real slowdown; losing a
/// handful is not worth throwing away a model somebody may ask for next.
const POOR_FIT: f32 = 0.67;

/// How long to wait for the driver to actually release evicted memory.
///
/// Freeing is asynchronous: the thread drops its engine and returns, and the
/// VRAM comes back some time afterwards. Loading into memory that has not been
/// released yet is how a careful plan still ends in an out-of-memory error.
const RELEASE_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// One loaded model.
struct Entry {
    /// Distinguishes this thread from a later one under the same name.
    ///
    /// A model evicted and immediately asked for again has a *new* thread
    /// keyed the same; without an id the old one finishing would deregister
    /// the replacement, and the pool would forget a model that is loaded.
    id: u64,
    tx: Sender<Job>,
    /// When it last started or finished work, for choosing what to evict.
    last_used: Arc<AtomicI64>,
    /// Generating right now. Never evicted while true.
    busy: Arc<AtomicBool>,
}

/// Every model currently loaded, and the lock that serialises loading one.
pub struct Pool {
    models: Mutex<HashMap<String, Entry>>,
    next_id: AtomicU64,
    /// Held only while deciding whether a model needs starting.
    ///
    /// Microseconds, and separate from `loading`, which is held for the
    /// seconds a load takes. Two callers that both find a model missing would
    /// otherwise each start a thread for it.
    spawning: Mutex<()>,
    /// Held across planning *and* loading.
    ///
    /// Two loads that overlap would each read the same free memory, each
    /// conclude they fit, and together not fit. Generation is not serialised —
    /// only the decision and the allocation are.
    loading: Mutex<()>,
}

impl Default for Pool {
    fn default() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            spawning: Mutex::new(()),
            loading: Mutex::new(()),
        }
    }
}

/// A model's own handle on the pool, held by its thread.
#[derive(Clone)]
pub struct Membership {
    pool: Arc<Pool>,
    key: String,
    id: u64,
    last_used: Arc<AtomicI64>,
    busy: Arc<AtomicBool>,
}

impl Membership {
    /// Mark this model as working, so it is not evicted under someone's feet.
    pub fn working(&self, yes: bool) {
        self.busy.store(yes, Ordering::SeqCst);
        self.last_used.store(now(), Ordering::SeqCst);
    }

    /// Take the loading lock, and make room if this model would be squeezed.
    ///
    /// `plan` is re-run after any eviction, because the whole point of
    /// evicting is that the answer changes. The guard must be held until the
    /// weights are resident — that is what stops two loads reading the same
    /// free memory and both concluding they fit.
    pub fn admit<P: Replan>(&self, plan: P) -> (std::sync::MutexGuard<'_, ()>, P::Out) {
        // It may be embedding this very turn's message as the chat model is
        // admitted — a busy model is never evicted — so its current job is
        // waited out first, briefly, and before taking the loading lock,
        // which a job still loading the embedder needs itself.
        if self.key != crate::worker::EMBED_KEY {
            let waited = std::time::Instant::now();
            while self.pool.embedder_busy() && waited.elapsed() < EMBEDDER_WAIT {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        let guard = self.pool.loading.lock().unwrap_or_else(|e| e.into_inner());
        // A chat model is placed on a card without the embedding model on it.
        // Loaded first — a memory check at start-up embeds a word — it took
        // 2.9 GB the chat model then planned around: its context came up
        // short, and a CUDA graph it needed mid-reply found no memory and
        // llama.cpp aborted the process. The embedder comes back in a second,
        // beside the chat model, and only where there is room above it and
        // its next decode (see `load_embedder`).
        if self.key != crate::worker::EMBED_KEY && self.pool.evict_idle(&self.key, Evict::Embedder) > 0 {
            wait_for_release();
        }
        let first = plan.plan();
        if P::is_full(&first) {
            return (guard, first);
        }
        // Short of a full fit: what is cheap to bring back goes first — the
        // embedding model, which reloads in a second, and chat models nobody
        // has used for a while. A 4B loaded 27 of 32 layers on the GPU beside
        // an idle embedding model and decoded slower for it. Only a poor fit
        // is worth unloading a model someone may still be about to use.
        let freed = if P::is_poor(&first) {
            self.pool.evict_idle(&self.key, Evict::AnyIdle)
        } else {
            self.pool.evict_idle(&self.key, Evict::Cheap)
        };
        if freed == 0 {
            return (guard, first);
        }
        wait_for_release();
        let second = plan.plan();
        tracing::info!(
            "unloaded {freed} idle model(s) to make room: {} -> {}",
            P::describe(&first),
            P::describe(&second)
        );
        (guard, second)
    }

    /// Take the loading lock without making room: for a load that must fit
    /// around what is resident rather than push it out — the embedding model,
    /// which is never worth evicting a chat model for.
    pub fn loading(&self) -> std::sync::MutexGuard<'_, ()> {
        self.pool.loading.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Deregister this model. Called when its thread stops.
    pub fn leave(&self) {
        self.pool.remove(&self.key, self.id);
    }
}

/// Something that can be planned twice: once as things stand, and again after
/// memory has been freed.
///
/// A trait rather than a closure so the pool can stay ignorant of what a plan
/// is — it lives in `ozgent-llama`, which this crate must not reach into to
/// make a scheduling decision.
pub trait Replan {
    type Out;
    fn plan(&self) -> Self::Out;
    fn is_full(out: &Self::Out) -> bool;
    fn is_poor(out: &Self::Out) -> bool;
    fn describe(out: &Self::Out) -> String;
}

/// Which idle models to unload to make room.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Evict {
    /// Every idle one: the new model would otherwise barely be on the GPU.
    AnyIdle,
    /// Only those cheap to bring back: the embedding model, and chat models
    /// idle for at least [`LONG_IDLE_SECS`].
    Cheap,
    /// Only the embedding model.
    Embedder,
}

/// The longest a chat model's admission waits for the embedding model to
/// finish a job, so as to place itself on a card without it.
const EMBEDDER_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// A chat model idle this long is not in the middle of being used.
pub const LONG_IDLE_SECS: i64 = 5 * 60;

/// Planning a model's placement on the GPU.
pub struct PlanFor<F>(pub F);

impl<F> Replan for PlanFor<F>
where
    F: Fn() -> ozgent_llama::backend::Plan,
{
    type Out = ozgent_llama::backend::Plan;

    fn plan(&self) -> Self::Out {
        (self.0)()
    }

    fn is_full(out: &Self::Out) -> bool {
        out.is_full()
    }

    fn is_poor(out: &Self::Out) -> bool {
        !out.is_full() && out.share() < POOR_FIT
    }

    fn describe(out: &Self::Out) -> String {
        if out.is_full() {
            "all layers on the GPU".to_string()
        } else {
            format!("{} of {} layers", out.layers, out.total_layers)
        }
    }
}

impl Pool {
    /// Whether the embedding model is resident and in the middle of a job.
    fn embedder_busy(&self) -> bool {
        let models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        models.get(crate::worker::EMBED_KEY).is_some_and(|e| e.busy.load(Ordering::SeqCst))
    }

    /// Guard the check-then-start sequence. See [`Pool::spawning`].
    pub fn starting(&self) -> std::sync::MutexGuard<'_, ()> {
        self.spawning.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The thread serving this model, if it is loaded.
    pub fn get(&self, key: &str) -> Option<Sender<Job>> {
        let models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        models.get(key).map(|e| {
            e.last_used.store(now(), Ordering::SeqCst);
            e.tx.clone()
        })
    }

    /// Register a model that is about to start loading.
    pub fn insert(self: &Arc<Self>, key: &str, tx: Sender<Job>) -> Membership {
        let last_used = Arc::new(AtomicI64::new(now()));
        // Busy from the moment it is registered: it is loading, and a model
        // being loaded must not be chosen as the one to evict. Otherwise two
        // loads racing can each pick the other, and both lose.
        let busy = Arc::new(AtomicBool::new(true));
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let mut models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        models.insert(
            key.to_string(),
            Entry { id, tx, last_used: last_used.clone(), busy: busy.clone() },
        );
        Membership { pool: Arc::clone(self), key: key.to_string(), id, last_used, busy }
    }

    /// Forget a model whose thread has stopped, if it is still the one
    /// registered under that name.
    fn remove(&self, key: &str, id: u64) {
        let mut models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        if models.get(key).is_some_and(|e| e.id == id) {
            models.remove(key);
        }
    }

    /// How many models are loaded.
    pub fn loaded(&self) -> usize {
        self.models.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// The models that are loaded, most recently used first.
    pub fn names(&self) -> Vec<String> {
        let models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<(i64, String)> = models
            .iter()
            .map(|(k, e)| (e.last_used.load(Ordering::SeqCst), k.clone()))
            .collect();
        rows.sort_by(|a, b| b.0.cmp(&a.0));
        rows.into_iter().map(|(_, k)| k).collect()
    }

    /// Unload everything.
    pub fn unload_all(&self) {
        let models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        for entry in models.values() {
            let _ = entry.tx.send(Job::Unload);
        }
    }

    /// Ask idle models to unload, oldest first. Returns how many were asked.
    ///
    /// Every idle one rather than just enough to fit, because "enough" cannot
    /// be known until the plan is re-run and the plan cannot be re-run per
    /// candidate without reloading the file each time. Models are cheap to
    /// bring back and the alternative — evicting one, re-planning, evicting
    /// another — pays the release wait once per model instead of once.
    fn evict_idle(&self, keep: &str, which: Evict) -> usize {
        let models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        let now = now();
        let mut idle: Vec<(i64, &String, &Entry)> = models
            .iter()
            .filter(|(name, e)| name.as_str() != keep && !e.busy.load(Ordering::SeqCst))
            .filter(|(name, e)| match which {
                Evict::AnyIdle => true,
                Evict::Embedder => name.as_str() == crate::worker::EMBED_KEY,
                Evict::Cheap => {
                    name.as_str() == crate::worker::EMBED_KEY
                        || now - e.last_used.load(Ordering::SeqCst) >= LONG_IDLE_SECS
                }
            })
            .map(|(name, e)| (e.last_used.load(Ordering::SeqCst), name, e))
            .collect();
        idle.sort_by_key(|(at, _, _)| *at);

        let mut asked = 0;
        for (_, name, entry) in idle {
            tracing::info!("unloading {name} to make room");
            if entry.tx.send(Job::Unload).is_ok() {
                asked += 1;
            }
        }
        asked
    }
}

/// Wait for the driver to hand back memory an evicted model was holding.
///
/// Polls rather than sleeps a fixed time: on a small model the memory is back
/// almost at once, and waiting ten seconds for it would make every eviction
/// feel like a stall.
fn wait_for_release() {
    let Some(before) = free_vram() else {
        std::thread::sleep(std::time::Duration::from_millis(200));
        return;
    };
    let deadline = std::time::Instant::now() + RELEASE_WAIT;
    while std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(100));
        match free_vram() {
            // Any rise means the unload has begun landing. A further wait for
            // it to finish is what the 8% headroom in the plan is for.
            Some(now) if now > before => {
                std::thread::sleep(std::time::Duration::from_millis(250));
                return;
            }
            _ => {}
        }
    }
    tracing::warn!("waited {RELEASE_WAIT:?} for evicted memory to come back; loading anyway");
}

fn free_vram() -> Option<u64> {
    ozgent_llama::backend::best_gpu().map(|d| d.memory_free as u64)
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ozgent_llama::backend::Plan;

    fn plan(layers: u32, total: u32) -> Plan {
        Plan { layers, experts: 0, expert_tensors: 0, total_layers: total, free_bytes: 0 }
    }

    #[test]
    fn a_full_offload_is_never_worth_evicting_for() {
        assert!(!PlanFor::<fn() -> Plan>::is_poor(&plan(32, 32)));
        // A file whose layer count could not be read counts as full, not as
        // a reason to throw another model away.
        assert!(!PlanFor::<fn() -> Plan>::is_poor(&plan(u32::MAX, 0)));
    }

    #[test]
    fn losing_most_of_a_model_to_the_cpu_is_worth_evicting_for() {
        assert!(PlanFor::<fn() -> Plan>::is_poor(&plan(0, 32)));
        assert!(PlanFor::<fn() -> Plan>::is_poor(&plan(10, 32)));
    }

    #[test]
    fn losing_a_handful_of_layers_is_not_worth_throwing_a_model_away_for() {
        // Evicting something somebody may ask for next, to gain three layers,
        // is a bad trade.
        assert!(!PlanFor::<fn() -> Plan>::is_poor(&plan(29, 32)));
        assert!(!PlanFor::<fn() -> Plan>::is_poor(&plan(24, 32)));
    }

    #[test]
    fn a_plan_describes_itself_for_the_log() {
        assert_eq!(PlanFor::<fn() -> Plan>::describe(&plan(32, 32)), "all layers on the GPU");
        assert_eq!(PlanFor::<fn() -> Plan>::describe(&plan(21, 32)), "21 of 32 layers");
    }

    // ------------------------------------------------------------ the map

    fn pool() -> Arc<Pool> {
        Arc::new(Pool::default())
    }

    #[test]
    fn a_model_is_found_by_name_once_it_is_registered() {
        let pool = pool();
        let (tx, _rx) = std::sync::mpsc::channel();
        assert!(pool.get("a").is_none());
        pool.insert("a", tx);
        assert!(pool.get("a").is_some());
        assert_eq!(pool.loaded(), 1);
    }

    #[test]
    fn several_models_are_held_at_once_with_no_ceiling_in_the_way() {
        // The limit is memory, not a count. Ten is not special; nothing here
        // should care how many there are.
        let pool = pool();
        let mut keep = Vec::new();
        for i in 0..10 {
            let (tx, rx) = std::sync::mpsc::channel();
            keep.push(rx);
            pool.insert(&format!("m{i}"), tx);
        }
        assert_eq!(pool.loaded(), 10);
    }

    #[test]
    fn asking_for_a_model_marks_it_as_recently_used() {
        let pool = pool();
        let (a, _ra) = std::sync::mpsc::channel();
        let (b, _rb) = std::sync::mpsc::channel();
        pool.insert("old", a);
        pool.insert("new", b);
        // Both were inserted in the same second, so order them explicitly.
        {
            let models = pool.models.lock().unwrap_or_else(|e| e.into_inner());
            models["old"].last_used.store(1, Ordering::SeqCst);
            models["new"].last_used.store(2, Ordering::SeqCst);
        }
        assert_eq!(pool.names(), ["new", "old"]);
        pool.get("old");
        assert_eq!(pool.names()[0], "old", "using it moves it to the front");
    }

    #[test]
    fn a_near_fit_unloads_only_what_is_cheap_to_bring_back() {
        let pool = pool();
        let (e, re) = std::sync::mpsc::channel();
        let (old, rold) = std::sync::mpsc::channel();
        let (fresh, rfresh) = std::sync::mpsc::channel();
        let (me, _mine) = std::sync::mpsc::channel();
        for m in [pool.insert(crate::worker::EMBED_KEY, e), pool.insert("old", old), pool.insert("fresh", fresh)] {
            m.working(false);
        }
        pool.insert("me", me);
        {
            let models = pool.models.lock().unwrap_or_else(|e| e.into_inner());
            models["old"].last_used.store(now() - LONG_IDLE_SECS - 1, Ordering::SeqCst);
        }
        assert_eq!(pool.evict_idle("me", Evict::Cheap), 2);
        assert!(matches!(re.try_recv(), Ok(Job::Unload)), "the embedding model reloads in a second");
        assert!(matches!(rold.try_recv(), Ok(Job::Unload)), "nobody has used it for a while");
        assert!(rfresh.try_recv().is_err(), "a model used a moment ago stays");
    }

    #[test]
    fn eviction_takes_the_least_recently_used_and_spares_the_one_loading() {
        let pool = pool();
        let (a, ra) = std::sync::mpsc::channel();
        let (b, rb) = std::sync::mpsc::channel();
        let (me, mine) = std::sync::mpsc::channel();
        let ma = pool.insert("a", a);
        let mb = pool.insert("b", b);
        pool.insert("me", me);
        ma.working(false);
        mb.working(false);

        assert_eq!(pool.evict_idle("me", Evict::AnyIdle), 2);
        assert!(matches!(ra.try_recv(), Ok(Job::Unload)));
        assert!(matches!(rb.try_recv(), Ok(Job::Unload)));
        assert!(mine.try_recv().is_err(), "the model being loaded is spared");
    }

    #[test]
    fn a_model_in_the_middle_of_answering_is_never_evicted() {
        // Somebody is reading it. Taking it away mid-generation would end
        // their turn with nothing.
        let pool = pool();
        let (busy, busy_rx) = std::sync::mpsc::channel();
        let (idle, idle_rx) = std::sync::mpsc::channel();
        let working = pool.insert("busy", busy);
        let resting = pool.insert("idle", idle);
        working.working(true);
        resting.working(false);

        assert_eq!(pool.evict_idle("none", Evict::AnyIdle), 1);
        assert!(busy_rx.try_recv().is_err(), "the busy one stays");
        assert!(matches!(idle_rx.try_recv(), Ok(Job::Unload)));
    }

    #[test]
    fn a_model_just_registered_counts_as_busy_because_it_is_loading() {
        // Otherwise two loads racing can each pick the other as the one to
        // evict, and both lose.
        let pool = pool();
        let (tx, rx) = std::sync::mpsc::channel();
        pool.insert("loading", tx);
        assert_eq!(pool.evict_idle("other", Evict::AnyIdle), 0);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn unloading_everything_reaches_every_model() {
        let pool = pool();
        let mut receivers = Vec::new();
        for i in 0..3 {
            let (tx, rx) = std::sync::mpsc::channel();
            receivers.push(rx);
            pool.insert(&format!("m{i}"), tx);
        }
        pool.unload_all();
        for rx in &receivers {
            assert!(matches!(rx.try_recv(), Ok(Job::Unload)));
        }
    }

    #[test]
    fn a_thread_that_stopped_removes_only_its_own_entry() {
        // A model evicted and immediately asked for again has a new thread
        // under the same name. The old one finishing must not deregister the
        // replacement, or the pool forgets a model that is loaded and loads a
        // third copy next time it is asked for.
        let pool = pool();
        let (old, _old_rx) = std::sync::mpsc::channel();
        let first = pool.insert("a", old);
        let (new, _new_rx) = std::sync::mpsc::channel();
        pool.insert("a", new);

        first.leave();
        assert_eq!(pool.loaded(), 1, "the replacement survives");
    }

    #[test]
    fn a_thread_removes_itself_when_it_is_still_the_one_registered() {
        let pool = pool();
        let (tx, _rx) = std::sync::mpsc::channel();
        let me = pool.insert("a", tx);
        me.leave();
        assert_eq!(pool.loaded(), 0);
    }

    #[test]
    fn an_empty_pool_answers_everything_without_panicking() {
        let pool = pool();
        assert_eq!(pool.loaded(), 0);
        assert!(pool.names().is_empty());
        assert!(pool.get("nothing").is_none());
        assert_eq!(pool.evict_idle("x", Evict::AnyIdle), 0);
        pool.unload_all();
    }
}
