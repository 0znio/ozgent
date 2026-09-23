//! The memory layer's embeddings, from the real embedding model.
//!
//! Memory recall fuses keyword search with vector similarity, and the vector
//! half only earns its place with a model that knows "make it faster" and
//! "reduce latency" mean the same thing. The daemon — which the web page, the
//! terminal and every channel all go through — was still building the lexical
//! stand-in, so recall was a keyword search fused with itself, beside an
//! installed embedding model nothing used.
//!
//! Three things keep a real model from costing the conversation anything:
//!
//! * **Stored texts are embedded in the background**, after the message is
//!   saved and outside the database lock, so no reply waits for them.
//! * **The question is embedded only when there is something to search**:
//!   a conversation that still fits in the recent window has nothing older to
//!   recall, and pays nothing.
//! * **A model change re-embeds in the background.** Vectors from another
//!   model are not comparable, whatever their width; they are cleared and
//!   rebuilt a batch at a time.

use ozgent_llama::embed::Role;
use ozgent_memory::{Embedder, HashingEmbedder, OwnerKind};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::state::State;
use crate::worker::Worker;

/// The daemon's embedder: the embedding model, or the lexical fallback when
/// embeddings are off or none is installed.
pub struct MemoryEmbedder {
    worker: Worker,
    fallback: HashingEmbedder,
    warned: AtomicBool,
}

impl MemoryEmbedder {
    pub fn new(worker: Worker) -> Self {
        Self { worker, fallback: HashingEmbedder::default(), warned: AtomicBool::new(false) }
    }

    /// Whether a real model is behind this.
    pub fn real(&self) -> bool {
        crate::worker::embed_status().model.is_some() || self.worker.embedding_model().is_some()
    }

    fn run(&self, role: Role, text: &str) -> Vec<f32> {
        if !self.real() {
            return self.fallback.embed(text);
        }
        match block(|| self.worker.embed_as(role, vec![text.to_string()])) {
            Ok(mut v) => v.pop().unwrap_or_default(),
            Err(e) => {
                if !self.warned.swap(true, Ordering::Relaxed) {
                    tracing::warn!("embedding model unavailable, recalling by keywords this turn: {e}");
                }
                self.fallback.embed(text)
            }
        }
    }
}

impl Embedder for MemoryEmbedder {
    fn dimensions(&self) -> usize {
        match crate::worker::embed_status().dimensions {
            0 => self.fallback.dimensions(),
            n => n,
        }
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        self.run(Role::Document, text)
    }

    fn embed_query(&self, text: &str) -> Vec<f32> {
        self.run(Role::Query, text)
    }
}

/// Run blocking work from wherever we are: off the async runtime's worker
/// thread when on one, directly otherwise.
fn block<T>(f: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => tokio::task::block_in_place(f),
        _ => f(),
    }
}

/// Embed a stored message after the fact, without holding up anybody.
pub fn embed_later(state: &State, id: i64, text: String) {
    if text.trim().is_empty() {
        return;
    }
    let state = state.clone();
    tokio::task::spawn_blocking(move || {
        let vector = state.embedder.embed(&text);
        if vector.iter().all(|x| *x == 0.0) {
            return;
        }
        let store = state.store.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = store.put_embedding(OwnerKind::Message, id, &vector) {
            tracing::debug!("storing an embedding: {e}");
        }
    });
}

static BACKFILLING: AtomicBool = AtomicBool::new(false);

/// Bring every stored message's vector up to the current model.
///
/// In the background, a small batch at a time with a pause between, so a
/// history of thousands of messages costs a chat nothing noticeable. Started
/// at start-up and whenever the embedding model changes; a second call while
/// one runs does nothing.
pub fn backfill(state: &State) {
    if BACKFILLING.swap(true, Ordering::SeqCst) {
        return;
    }
    let state = state.clone();
    tokio::spawn(async move {
        let result = run_backfill(&state).await;
        BACKFILLING.store(false, Ordering::SeqCst);
        match result {
            Ok(0) => {}
            Ok(n) => tracing::info!("embedded {n} earlier messages for memory recall"),
            Err(e) => tracing::info!("memory backfill stopped: {e}"),
        }
    });
}

/// Whether a backfill is running now.
pub fn backfilling() -> bool {
    BACKFILLING.load(Ordering::SeqCst)
}

async fn run_backfill(state: &State) -> Result<usize, String> {
    let Some(model) = state.worker.embedding_model() else { return Ok(0) };
    // Learn the width by embedding once; this also loads the model.
    let worker = state.worker.clone();
    let probe = tokio::task::spawn_blocking(move || worker.embed_as(Role::Document, vec!["ozgent".into()]))
        .await
        .map_err(|e| e.to_string())??;
    let dim = probe.first().map(Vec::len).unwrap_or(0);
    if dim == 0 {
        return Ok(0);
    }

    // Vectors of the same width can still be another model's. The model in
    // use is recorded beside the database; a different one means none of the
    // old vectors mean anything now.
    let marker = state.paths.cache_dir().join("embedding-model");
    let previous = std::fs::read_to_string(&marker).ok().map(|s| s.trim().to_string());
    if previous.as_deref() != Some(model.as_str()) {
        if previous.is_some() {
            let cleared = state.store.lock().unwrap_or_else(|e| e.into_inner()).clear_embeddings().unwrap_or(0);
            tracing::info!("embedding model changed to {model}; re-embedding {cleared} stored vectors");
        }
        let _ = std::fs::create_dir_all(state.paths.cache_dir());
        let _ = std::fs::write(&marker, &model);
    }

    let mut done = 0usize;
    loop {
        let batch = state
            .store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .messages_needing_embeddings(dim, 16)
            .map_err(|e| e.to_string())?;
        if batch.is_empty() {
            return Ok(done);
        }
        let texts: Vec<String> = batch.iter().map(|(_, t)| t.clone()).collect();
        let worker = state.worker.clone();
        let vectors = tokio::task::spawn_blocking(move || worker.embed_as(Role::Document, texts))
            .await
            .map_err(|e| e.to_string())??;
        let stored = {
            let store = state.store.lock().unwrap_or_else(|e| e.into_inner());
            let mut stored = 0;
            for ((id, _), v) in batch.iter().zip(&vectors) {
                if v.len() == dim && store.put_embedding(OwnerKind::Message, *id, v).is_ok() {
                    stored += 1;
                }
            }
            stored
        };
        // A batch that stored nothing would be fetched again forever.
        if stored == 0 {
            return Err(format!("the embedding model returned no usable vectors ({model})"));
        }
        done += batch.len();
        // Yield to anything a person is waiting on.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }
}
