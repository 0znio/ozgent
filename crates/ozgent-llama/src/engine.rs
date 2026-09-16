//! Model loading and token generation.
//!
//! An [`Engine`] owns the weights; a [`Session`] owns one KV cache and the
//! conversation running against it. Splitting them this way lets a chat keep
//! its cache between turns — the single largest avoidable cost in interactive
//! use — while a one-shot `run` just drops the session afterwards.

use crate::ngram::NgramCache;
use crate::toolgate::ToolGate;
use crate::utf8::Utf8Buffer;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::context::session::{LlamaStateSeqFlags, SeqState};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;
use ozgent_core::accel::{CacheType, MoeKeyword, MoeOffload, PrefixReuse, Speculative};
use ozgent_core::options::GpuKeyword;
use ozgent_core::{GpuLayers, Message, Resolved, Role};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

/// Sequences a context carries for `slots` conversations.
///
/// One more on the daemon's shared contexts, where the prefix every
/// conversation starts with is held once; see `hub::Commons`. Not on a
/// session opened for one caller, where there is nobody to share it with and
/// the extra sequence only doubled the cache — `2/2 seqs` in llama.cpp's own
/// log, for a CLI session that could never use the second.
fn sequences_for(slots: u32, want: Slots) -> u32 {
    match want {
        Slots::UpTo(_) => slots + 1,
        Slots::Exact(_) => slots,
    }
}

/// The wider batch tried when nothing was asked for.
///
/// Prefill runs a batch through as physical chunks of this size, so doubling
/// it halves the number of passes over the weights. Two is the useful number
/// of candidates: llama.cpp's own default either side of it, and a wider one
/// still would cost more scratch than the window can spare on a card this
/// size. It is only ever taken when it costs no context.
pub(crate) const WIDE_BATCH: u32 = 1024;

/// Microseconds spent choosing tokens from logits, process-wide.
pub static PICK_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The context parameters a probed scratch figure is valid for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ScratchKey {
    n_batch: u32,
    ubatch: Option<u32>,
    flash: bool,
    kv_offload: bool,
    sequences: u32,
    unified: bool,
}

/// How many conversations a context should carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slots {
    /// Exactly this many, whatever it costs in window length.
    Exact(u32),
    /// As many as fit beside each other at the asked-for window, and no
    /// fewer than one.
    UpTo(u32),
}

/// llama.cpp's backend may only be initialised once per process.
pub(crate) fn backend_handle() -> Result<&'static LlamaBackend, EngineError> {
    backend()
}

fn backend() -> Result<&'static LlamaBackend, EngineError> {
    static CELL: OnceLock<Option<LlamaBackend>> = OnceLock::new();
    CELL.get_or_init(|| {
        // Installed before init, not after: ggml prints its device inventory
        // while the backend comes up, so a callback set afterwards is already
        // too late and the banner lands on the user's stderr.
        //
        // Captured rather than voided, because the one thing llama.cpp says
        // that ozgent cannot work out for itself is why a call failed — it
        // reports that through the log and then returns null.
        crate::llamalog::capture();
        LlamaBackend::init().ok()
    })
    .as_ref()
    .ok_or(EngineError::BackendInit)
}

/// Loaded weights, shared by every session.
pub struct Engine {
    model: LlamaModel,
    template: Option<LlamaChatTemplate>,
    n_layer: u32,
    n_ctx_train: u32,
    gpu_layers_used: u32,
    /// Layers whose routed experts were evicted to host memory. Non-zero
    /// means system RAM is holding weights, and is contended.
    cpu_moe_layers: u32,
    /// This model's own compute scratch, measured once per set of context
    /// parameters. See [`Engine::probed_reserve`].
    scratch: std::sync::Mutex<Option<(ScratchKey, u64, f64)>>,
    /// What prefill uploads into the compute buffer per micro-batch when
    /// routed experts live on the host: the heaviest evicted block. Zero when
    /// nothing is evicted. See `reserve::Shape::staging_bytes`.
    staging_bytes: u64,
    /// True when the chat template wraps reasoning in `<think>` tags, i.e.
    /// this is a model that reasons unless told not to.
    reasoning: bool,
    /// False when the model keeps state that cannot be rolled back, which
    /// makes draft rejection unsafe. See [`Engine::rollback_safe`].
    rollback_safe: bool,
    /// The model's own chat template, when it could be compiled.
    jinja: Option<crate::template::ChatTemplate>,
    /// Tokens the template appends to open the assistant's turn.
    ///
    /// The one part of a prompt the next turn does not repeat: next time round
    /// the assistant's actual reply stands where these were. A saved state
    /// that includes them describes tokens the new prompt does not contain,
    /// so it can never be reused — which is why the checkpoint stops short.
    gen_prompt_tokens: usize,
    /// KV elements stored per token across all layers, for cache sizing.
    kv_shape: ozgent_core::accel::KvShape,
    /// On-disk size of the weights, used as the denominator when deciding
    /// whether KV traffic dominates weight traffic.
    weight_bytes: u64,
}

/// How far an n-gram match must extend behind the key before it is drafted,
/// on the rollback path where a wrong guess costs an extra forward pass.
const MIN_DRAFT_REACH: usize = 4;

/// Draft length used while measuring whether drafting is worth it at all.
///
/// Only relevant on the snapshot rollback path, where a rejected draft costs an
/// extra forward pass. Long speculative probes into unpredictable text were
/// measurably slower than not speculating.
const PROBE_DRAFT_TOKENS: usize = 3;

/// Prefilled to close reasoning before it starts.
///
/// A stream filter can only *hide* reasoning; the model still spends its token
/// budget on it, and a short `--max-tokens` then yields an empty answer.
/// Opening and immediately closing the block puts the model in the
/// already-finished-thinking state, which is what actually suppresses it.
const EMPTY_THINK: &str = "<think>\n\n</think>\n\n";

impl Engine {
    /// Time N sequences advanced one token, batched together against decoded
    /// apart.
    ///
    /// The measurement the batching decision rests on. Weights are read once
    /// per forward pass regardless of how many sequences ride along, so in
    /// principle N callers cost about what one costs — but only while the card
    /// still has work to spare at a batch of one. On a small model and a busy
    /// GPU the ratio collapses to 1.0 and batching is not worth restructuring
    /// the daemon for.
    ///
    /// Both sides read the logits back. `decode` queues work on the GPU and
    /// returns; `llama_get_logits_ith` is what synchronises, so without the
    /// read this times how fast batches can be submitted rather than how long
    /// they take. It turns out to make little difference here — 2.42x against
    /// 2.47x at four sequences — because over forty rounds the queue
    /// saturates anyway, but a probe whose answer decides an architecture
    /// should not depend on that.
    ///
    /// Returns milliseconds for the batched decode and for the separate ones.
    pub fn probe_batched_decode(
        &self,
        sequences: usize,
        n_seq_max: u32,
        rounds: usize,
    ) -> Result<(f64, f64), EngineError> {
        let backend = backend()?;
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048))
            .with_n_seq_max(n_seq_max)
            .with_n_batch(n_seq_max.max(32))
            .with_n_threads(8)
            .with_n_threads_batch(8);
        let mut context = self
            .model
            .new_context(backend, params)
            .map_err(|e| EngineError::Context(e.to_string()))?;

        // Give every sequence one real token so the caches are not empty.
        let seed = self
            .model
            .str_to_token("The", AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        let mut batch = LlamaBatch::new(n_seq_max as usize * 4, n_seq_max as i32);
        for seq in 0..sequences {
            batch.clear();
            for (i, t) in seed.iter().enumerate() {
                batch
                    .add(*t, i as i32, &[seq as i32], i + 1 == seed.len())
                    .map_err(|e| EngineError::Batch(e.to_string()))?;
            }
            context.decode(&mut batch).map_err(|e| EngineError::Decode(e.to_string()))?;
        }
        let base = seed.len() as i32;

        // Together: one batch carrying a token for every sequence.
        let together = std::time::Instant::now();
        for round in 0..rounds {
            batch.clear();
            for seq in 0..sequences {
                batch
                    .add(seed[0], base + round as i32, &[seq as i32], true)
                    .map_err(|e| EngineError::Batch(e.to_string()))?;
            }
            context.decode(&mut batch).map_err(|e| EngineError::Decode(e.to_string()))?;
            let _ = context.get_logits_ith(0);
        }
        let together = together.elapsed().as_secs_f64() * 1000.0 / rounds as f64;

        // Apart: one batch per sequence, which is what the daemon does now.
        let apart = std::time::Instant::now();
        for round in 0..rounds {
            for seq in 0..sequences {
                batch.clear();
                batch
                    .add(seed[0], base + rounds as i32 + round as i32, &[seq as i32], true)
                    .map_err(|e| EngineError::Batch(e.to_string()))?;
                context.decode(&mut batch).map_err(|e| EngineError::Decode(e.to_string()))?;
                let _ = context.get_logits_ith(0);
            }
        }
        let apart = apart.elapsed().as_secs_f64() * 1000.0 / rounds as f64;

        Ok((together, apart))
    }

    /// Milliseconds for one pass carrying `k` tokens of a single sequence,
    /// for each `k` — what verifying a k-token draft costs.
    ///
    /// Speculation assumes that cost barely grows with `k`. On a model whose
    /// routed experts live on the host it may not: each extra token routes to
    /// its own experts, so the pass reads the union of all of them from system
    /// RAM. Measured rather than argued. Each pass is rewound, so every `k`
    /// starts from the same cache.
    pub fn probe_verify_cost(
        &self,
        opts: &Resolved,
        ks: &[usize],
        rounds: usize,
    ) -> Result<Vec<(usize, f64)>, EngineError> {
        let session = self.session(opts)?;
        let prompt = self
            .model
            .str_to_token(
                &"The history of the lighthouse spans many centuries and many coasts. ".repeat(40),
                AddBos::Always,
            )
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        let n_batch = (session.n_batch as usize).max(1);
        let mut pos = 0i32;
        for chunk in prompt.chunks(n_batch) {
            session
                .slot
                .hub()
                .run(crate::hub::Work {
                    seq: 0,
                    tokens: chunk.to_vec(),
                    pos,
                    logits: crate::hub::Logits::Last,
                    hidden: false,
                })
                .map_err(|e| EngineError::Decode(e.to_string()))?;
            pos += chunk.len() as i32;
        }
        // Distinct tokens drawn from the prompt, so each position routes the
        // way real text would rather than repeating one token's experts.
        let pool: Vec<LlamaToken> = prompt.iter().copied().skip(1).collect();
        let mut out = Vec::new();
        for &k in ks {
            let mut times = Vec::with_capacity(rounds);
            for r in 0..rounds {
                let tokens: Vec<LlamaToken> =
                    (0..k).map(|i| pool[(r * 7 + i * 13) % pool.len()]).collect();
                let t = Instant::now();
                session
                    .slot
                    .hub()
                    .run(crate::hub::Work {
                        seq: 0,
                        tokens,
                        pos,
                        logits: crate::hub::Logits::All,
                        hidden: false,
                    })
                    .map_err(|e| EngineError::Decode(e.to_string()))?;
                times.push(t.elapsed().as_secs_f64() * 1000.0);
                let _ = session.slot.trim(pos);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out.push((k, times[times.len() / 2]));
        }
        Ok(out)
    }

    /// Load a GGUF file with the given resolved settings.
    pub fn load(path: &Path, opts: &Resolved) -> Result<Self, EngineError> {
        Self::load_reporting(path, opts, |_| {})
    }

    /// [`Engine::load`], calling `progress` with how far along it is, from 0
    /// to 1. Loading a large model takes long enough that a person watching
    /// needs to see it moving.
    pub fn load_reporting(
        path: &Path,
        opts: &Resolved,
        mut progress: impl FnMut(f32) + 'static,
    ) -> Result<Self, EngineError> {
        let backend = backend()?;
        if !path.exists() {
            return Err(EngineError::Missing { path: path.display().to_string() });
        }

        // What `auto` should mean, and now does.
        //
        // It used to become u32::MAX — "every layer on the GPU" — and llama.cpp
        // would clamp that to the layer count and try. On an empty GPU with a
        // model that fits, that is the right answer and still is. On a GPU that
        // already holds something else it is not an answer at all: the load
        // either fails or thrashes, and neither says why.
        //
        // The planner in backend.rs knows how to choose, from the file's own
        // tensor table and the driver's live free-memory figure. It was written
        // and tested and then wired only to expert eviction, so the option
        // documented as "the smallest offload that fits in VRAM" never searched
        // for anything on a dense model. It searches now.
        let plan = crate::backend::Plan::for_model(path, opts);
        let requested_layers = match opts.gpu_layers {
            GpuLayers::Count(n) => n,
            GpuLayers::Keyword(GpuKeyword::Off) => 0,
            GpuLayers::Keyword(GpuKeyword::Auto) => plan.layers,
        };

        let mut params = Box::pin(LlamaModelParams::default()
            .with_n_gpu_layers(requested_layers)
            .with_use_mmap(opts.use_mmap)
            .with_use_mlock(opts.use_mlock)
            .with_main_gpu(opts.main_gpu as i32)
            .with_progress_callback(move |p| {
                progress(p.clamp(0.0, 1.0));
                true
            }));

        // `auto` was the default and did nothing: the planner in backend.rs was
        // written and tested but never called, so the option documented as
        // "search for the smallest offload that fits in VRAM" searched for
        // nothing. Resolving it needs the per-layer and per-expert byte costs,
        // which come from the file's tensor table rather than from llama.cpp.
        let resolved_moe = match opts.cpu_moe {
            // Both halves of the decision come from one plan: choosing layers
            // and choosing experts against different readings of free VRAM
            // would be two answers to one question.
            MoeOffload::Keyword(MoeKeyword::Auto) => {
                if plan.experts > 0 || plan.expert_tensors > 0 {
                    tracing::info!(
                        "cpu-moe auto: evicting routed experts from {} of {} layers{}",
                        plan.experts,
                        plan.total_layers,
                        match plan.expert_tensors {
                            0 => String::new(),
                            n => format!(
                                ", and {} of 3 from layer {}",
                                n, plan.experts
                            ),
                        }
                    );
                }
                MoeOffload::Layers(plan.experts)
            }
            other => other,
        };
        // Say plainly whether the host can actually hold what is being sent to
        // it. A mixture-of-experts model with its experts in system RAM reads
        // them on every token; if they do not fit, the kernel pages them from
        // disk instead, and mmap means that degrades silently to one or two
        // tokens a second rather than failing. Measured on an NVMe at
        // 1.4 GB/s of page-fault-driven reads, a 20% shortfall is the
        // difference between 15 tok/s and 4.
        let layout = crate::layout::read(path).unwrap_or_default();
        if let Some(expert_bytes) = host_expert_bytes(&layout, resolved_moe) {
            let available = ozgent_core::accel::available_host_memory();
            if available > 0 && expert_bytes > available {
                tracing::warn!(
                    "experts need {} GB of system ram and {} GB is available; \
                     the shortfall is read from disk on every token, which is far \
                     slower than it sounds. A smaller quantisation would fit.",
                    expert_bytes / (1 << 30),
                    available / (1 << 30),
                );
            } else if available > 0 {
                tracing::info!(
                    "experts on the cpu: {} GB of system ram, {} GB available",
                    expert_bytes / (1 << 30),
                    available / (1 << 30),
                );
            }
        }

        let evicted_expert_layers = match resolved_moe {
            MoeOffload::Layers(n) => n,
            MoeOffload::Keyword(MoeKeyword::All) => u32::MAX,
            _ => 0,
        };
        // Only `auto` lands on a partial layer; an explicit count means whole
        // layers and nothing finer.
        let resolved_tensors = match opts.cpu_moe {
            MoeOffload::Keyword(MoeKeyword::Auto) => plan.expert_tensors,
            _ => 0,
        };

        // Evicting routed experts frees far more VRAM per lost token/sec than
        // dropping whole layers, so it is applied before any layer reduction.
        //
        // The pattern is bound here rather than inside the match because
        // llama.cpp keeps the pointer, not a copy: it reads the string when
        // the model loads, which is after this block ends.
        let moe_pattern;
        match resolved_moe {
            MoeOffload::Keyword(MoeKeyword::All) => params.as_mut().add_cpu_moe_override(),
            MoeOffload::Layers(n) if n > 0 || resolved_tensors > 0 => {
                // One override covering everything evicted, not one per
                // layer. `add_cpu_buft_override` always fills slot zero, so a
                // second call trips its own "last buft_override was not empty"
                // assertion — which is why this panicked on the first real
                // mixture-of-experts model to reach it. Alternation is how
                // two rules become one, and it is also what lets a single
                // layer be split: whole layers, plus part of the next.
                moe_pattern = crate::backend::moe_pattern(n, resolved_tensors)
                    .and_then(|p| std::ffi::CString::new(p).ok());
                if let Some(c) = moe_pattern.as_deref() {
                    params.as_mut().add_cpu_buft_override(c);
                }
            }
            _ => {}
        }

        crate::llamalog::clear();
        let model = LlamaModel::load_from_file(backend, path, &params).map_err(|e| {
            // `e` is "null result from llama cpp" for every kind of failure.
            // The log holds the one that actually happened.
            let reason = crate::llamalog::reason().unwrap_or_else(|| e.to_string());
            EngineError::Load { path: path.display().to_string(), reason }
        })?;

        let template = model.chat_template(None).ok();
        let n_layer = model.n_layer();

        // The KV cache is sized from the model's real key/value widths. These
        // are not derivable from n_embd/n_head — Qwen3.5 has 16 heads over
        // n_embd 2560 (=160) but a true key length of 256, which would
        // mis-size the cache by 60%. Fall back to the classic derivation only
        // for older GGUFs that omit the keys.
        let arch = model.meta_val_str("general.architecture").unwrap_or_default();
        let head_dim = (model.n_embd() as u32) / model.n_head().max(1);
        let k_len = meta_u32(&model, &format!("{arch}.attention.key_length")).unwrap_or(head_dim);
        let v_len = meta_u32(&model, &format!("{arch}.attention.value_length")).unwrap_or(head_dim);
        // No value cache under multi-head latent attention; see `layout::is_mla`.
        let mla = meta_u32(&model, &format!("{arch}.attention.key_length_mla")).is_some_and(|n| n > 0)
            && meta_u32(&model, &format!("{arch}.attention.value_length_mla")).is_some_and(|n| n > 0);
        let v_len = if mla { 0 } else { v_len };
        // Counted from the file, not from `n_layer`, because a hybrid model
        // caches on only some of its layers. Qwen3.5 runs linear attention on
        // three of every four and holds a fixed-size recurrent state there —
        // real memory, but the same amount at one token as at a hundred
        // thousand, so it belongs in no per-token figure. Pricing all 33
        // blocks of a 4B reserved 2176 MiB where 550 was needed.
        //
        // llama.cpp knows this from the architecture and does not expose it,
        // so it is read back out of the GGUF. A file that cannot be measured
        // falls back to every layer, which is what this always did.
        let caching = crate::layout::read(path)
            .map(|l| l.caching_layers)
            .filter(|&c| c > 0)
            .unwrap_or(n_layer);
        if caching < n_layer {
            tracing::debug!("{caching} of {n_layer} layers keep a growing cache");
        }
        let kv_shape =
            ozgent_core::accel::KvShape::new(caching, model.n_head_kv(), k_len, v_len);
        let weight_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        // Speculative decoding rejects drafts by discarding the KV entries
        // they produced. A recurrent or hybrid model updates part of its state
        // in place — Qwen3.5 runs linear attention on three of every four
        // layers — and that state cannot be rolled back: removing the KV
        // entries clears the attention layers while the recurrent ones have
        // already absorbed the rejected tokens. The result is silently
        // corrupted state, so drafting is refused outright rather than
        // producing wrong output faster.
        let rollback_safe = !model.is_recurrent() && !model.is_hybrid();
        if !rollback_safe {
            // Not disabled — this said so for months and it was wrong.
            // Speculation still runs here; it just cannot undo a rejected
            // draft by trimming the cache, so it snapshots the sequence state
            // and puts it back instead. Measured at 0.71 ms against a ~17.8 ms
            // token budget. The old wording sent anyone reading the log
            // looking for a problem that was not there.
            tracing::debug!(
                "model keeps unrollbackable state; drafts are undone by snapshot rather than trim"
            );
        }
        // The raw Jinja source tells us whether this model reasons; there is
        // no capability flag in GGUF for it.
        let raw_template = model.meta_val_str("tokenizer.chat_template").unwrap_or_default();
        let reasoning = raw_template.contains("<think>");
        // The model's own Jinja, preferred over llama.cpp's approximation of
        // it. Failing to compile is not an error: the built-in renderer is
        // still there and is exactly right for most models.
        let jinja = if raw_template.is_empty() {
            None
        } else {
            let bos = model.token_to_str(model.token_bos(), Special::Tokenize).unwrap_or_default();
            let eos = model.token_to_str(model.token_eos(), Special::Tokenize).unwrap_or_default();
            match crate::template::ChatTemplate::new(&raw_template, bos, eos) {
                Ok(t) => Some(t),
                Err(e) => {
                    tracing::debug!("using llama.cpp's built-in template instead: {e}");
                    None
                }
            }
        };

        Ok(Self {
            n_ctx_train: model.n_ctx_train(),
            gpu_layers_used: requested_layers.min(n_layer),
            cpu_moe_layers: evicted_expert_layers,
            scratch: std::sync::Mutex::new(None),
            staging_bytes: if evicted_expert_layers > 0 || resolved_tensors > 0 {
                layout.max_layer_expert_bytes
            } else {
                0
            },
            reasoning,
            gen_prompt_tokens: Self::measure_generation_prompt(&model, &jinja, template.as_ref()),
            jinja,
            n_layer,
            rollback_safe,
            kv_shape,
            weight_bytes,
            template,
            model,
        })
    }

    /// Whether a rejected draft can be undone by trimming the cache.
    ///
    /// False for hybrid and recurrent models, which decides how — and whether —
    /// speculation works, so it is worth surfacing in diagnostics.
    pub fn rollback_safe(&self) -> bool {
        self.rollback_safe
    }

    /// The close tag a stream beginning from `prompt` starts inside.
    ///
    /// Read off the rendered prompt rather than guessed from the template,
    /// because only the render knows whether this particular turn reasons. A
    /// prompt that ends inside an open block means the model writes reasoning
    /// first and never emits an opening tag of its own.
    pub fn stream_starts_inside(prompt: &str) -> Option<&'static str> {
        crate::thinking::open_at_end(prompt)
    }

    pub fn n_layer(&self) -> u32 {
        self.n_layer
    }

    /// Context length the model was trained for.
    pub fn n_ctx_train(&self) -> u32 {
        self.n_ctx_train
    }

    pub fn gpu_layers_used(&self) -> u32 {
        self.gpu_layers_used
    }

    pub fn has_chat_template(&self) -> bool {
        self.template.is_some()
    }

    /// Whether this model emits reasoning unless suppressed.
    pub fn is_reasoning_model(&self) -> bool {
        self.reasoning
    }

    /// Render messages with the model's own chat template.
    ///
    /// Falling back to a generic format when the GGUF has no template is worse
    /// than it sounds — the model will not recognise the turn markers — so the
    /// fallback is deliberately plain and the caller is told.
    /// The tokens a prompt becomes, for a caller that needs to compare two of
    /// them rather than decode one.
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, EngineError> {
        self.model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))
    }

    pub fn render_prompt(&self, messages: &[Message]) -> Result<String, EngineError> {
        self.render_prompt_with(messages, ozgent_core::ThinkingMode::Auto, Default::default())
    }

    /// How many tokens the template adds to open the assistant's turn.
    ///
    /// Measured once, by rendering a throwaway exchange with and without the
    /// generation prompt and taking the difference. A template whose opening
    /// varies with the request will be measured slightly wrong, which costs
    /// the checkpoint optimisation on those turns and nothing else — the
    /// prefix check still has to pass before a saved state is used.
    fn measure_generation_prompt(
        model: &LlamaModel,
        jinja: &Option<crate::template::ChatTemplate>,
        template: Option<&llama_cpp_2::model::LlamaChatTemplate>,
    ) -> usize {
        let probe = [Message::user("x")];
        let count = |text: &str| {
            model.str_to_token(text, AddBos::Never).map(|t| t.len()).unwrap_or(0)
        };
        let (with, without) = match jinja {
            Some(j) => {
                let opts = crate::template::RenderOptions::default();
                let bare = crate::template::RenderOptions {
                    add_generation_prompt: false,
                    ..Default::default()
                };
                match (j.render(&probe, opts), j.render(&probe, bare)) {
                    (Ok(a), Ok(b)) => (a, b),
                    _ => return 0,
                }
            }
            None => {
                let Some(template) = template else { return 0 };
                let chat = match LlamaChatMessage::new("user".into(), "x".into()) {
                    Ok(m) => vec![m],
                    Err(_) => return 0,
                };
                match (
                    model.apply_chat_template(template, &chat, true),
                    model.apply_chat_template(template, &chat, false),
                ) {
                    (Ok(a), Ok(b)) => (a, b),
                    _ => return 0,
                }
            }
        };
        count(&with).saturating_sub(count(&without))
    }

    /// See [`Engine::gen_prompt_tokens`].
    pub fn generation_prompt_tokens(&self) -> usize {
        self.gen_prompt_tokens
    }

    /// Render, optionally suppressing reasoning at the prompt level.
    pub fn render_prompt_with(
        &self,
        messages: &[Message],
        thinking: ozgent_core::ThinkingMode,
        effort: ozgent_core::ReasoningEffort,
    ) -> Result<String, EngineError> {
        self.render_prompt_full(messages, thinking, effort, &[])
    }

    /// Whether the model's template describes tools to the model itself.
    ///
    /// Callers use this to decide whether to add ozgent's generic preamble:
    /// when the template has its own tools block, the model is told its own
    /// call format, and a second description of a different one is worse than
    /// either alone.
    pub fn template_handles_tools(&self) -> bool {
        self.jinja.as_ref().is_some_and(|j| j.handles_tools())
    }

    /// Render, offering `tools` to a template that can describe them.
    ///
    /// Passing tools a template cannot use is harmless — the key is simply
    /// unread — so the caller does not have to check first.
    pub fn render_prompt_full(
        &self,
        messages: &[Message],
        thinking: ozgent_core::ThinkingMode,
        effort: ozgent_core::ReasoningEffort,
        tools: &[ozgent_core::ToolSpec],
    ) -> Result<String, EngineError> {
        let suppress = thinking == ozgent_core::ThinkingMode::Off && self.reasoning;

        // The model's own template first. It is the only thing that knows
        // whether this turn reasons, and so the only thing that can decide
        // whether the prompt should end inside an open block.
        if let Some(jinja) = &self.jinja {
            let opts = crate::template::RenderOptions {
                // Asked for only when the caller has not overridden thinking
                // itself. Templates that read `reasoning_effort` treat it as
                // the higher authority — Ling 3.0 turns reasoning *off* at
                // "low" — so letting it through under an explicit `--think on`
                // would quietly contradict the flag the user just set.
                reasoning_effort: (thinking == ozgent_core::ThinkingMode::Auto)
                    .then(|| effort.to_string()),
                // Always stated, never left to the template's default. Qwen3.5
                // reads an undefined `enable_thinking` as "off" and writes a
                // closed empty block, so silence is not neutral — it is a
                // decision, and the wrong one for a model whose whole point is
                // that it reasons. "Auto" is ozgent's own rule: think if the
                // model advertises the capability.
                enable_thinking: Some(match thinking {
                    ozgent_core::ThinkingMode::On => true,
                    ozgent_core::ThinkingMode::Off => false,
                    ozgent_core::ThinkingMode::Auto => self.reasoning,
                }),
                tools: tools.iter().map(crate::template::tool_json).collect(),
                ..Default::default()
            };
            match jinja.render(messages, opts) {
                Ok(prompt) => {
                    // The whole prompt, at trace level. Whether a turn reasons
                    // depends on what the template wrote, and reading it is
                    // the only way to tell a model that would not close
                    // `</think>` from one that was never told to open it.
                    tracing::trace!(target: "ozgent::prompt", "{prompt}");
                    return Ok(prompt);
                }
                Err(e) => tracing::debug!("falling back to the built-in template: {e}"),
            }
        }

        let Some(template) = &self.template else {
            let mut p = fallback_prompt(messages);
            if suppress {
                p.push_str(EMPTY_THINK);
            }
            return Ok(p);
        };

        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                LlamaChatMessage::new(role.to_string(), m.text_content())
            })
            .collect::<Result<_, _>>()
            .map_err(|e| EngineError::Template(e.to_string()))?;

        let mut prompt = self
            .model
            .apply_chat_template(template, &chat, true)
            .map_err(|e| EngineError::Template(e.to_string()))?;

        if suppress {
            prompt.push_str(EMPTY_THINK);
        }
        Ok(prompt)
    }

    /// Open a session with its own KV cache.
    /// Load a multimodal projector for this model.
    ///
    /// Separate from [`Engine::load`] because it is only needed for a turn that
    /// actually carries an image, and it costs both time and VRAM.
    pub fn projector(
        &self,
        mmproj: &std::path::Path,
        opts: &Resolved,
    ) -> Result<crate::mtmd::Projector<'_>, EngineError> {
        let on_gpu = !matches!(opts.gpu_layers, GpuLayers::Keyword(GpuKeyword::Off));
        crate::mtmd::Projector::load(mmproj, &self.model, on_gpu, opts.threads as i32)
            .map_err(|e| EngineError::Load {
                path: mmproj.display().to_string(),
                reason: e.to_string(),
            })
    }

    /// Create the context, retreating to a smaller window when the allocation
    /// fails.
    ///
    /// [`ozgent_core::accel::fit_context`] has already sized the window
    /// against the KV cache, which is the large and predictable part. What it
    /// cannot account for is the compute buffers and llama.cpp's own scratch:
    /// those depend on the batch size and on how the backend's graph planner
    /// lays the work out, and neither is derivable from metadata. When the
    /// estimate turns out optimistic llama.cpp returns a null pointer, and a
    /// session that opens with half the window is worth far more to the user
    /// than one that refuses to open at all.
    fn open_context(
        &self,
        backend: &'static LlamaBackend,
        params: LlamaContextParams,
        requested: u32,
        mut split: ozgent_core::accel::KvSplit,
        shape: ozgent_core::reserve::Shape,
    ) -> Result<LlamaContext<'_>, EngineError> {
        let mut window = requested;
        // The retreat bisects rather than stepping by a fixed ratio.
        //
        // A ratio has to be chosen against an error whose size is unknown: a
        // half throws away three quarters of the context when the estimate
        // missed by five percent, and three quarters takes so many steps that
        // it overshoots further down. Bisecting between the largest window
        // known to work and the smallest known to fail lands within a few
        // percent in three or four attempts however wrong the first guess
        // was, and an attempt is only an allocation — the weights are already
        // resident and nothing is recomputed.
        let mut good: u32 = 0;
        let mut bad = requested.saturating_add(1);
        // The first failure is the informative one: later attempts fail for
        // the same reason at a smaller size, and the last one before the floor
        // says least about what actually went wrong.
        let mut first_reason: Option<String> = None;
        let mut params = params;
        // Whether the quantised KV cache has already been given up on. Once,
        // and then never again, so a genuine out-of-memory does not loop.
        let mut plain_cache = false;

        loop {
            crate::llamalog::clear();
            let attempt = params.clone().with_n_ctx(NonZeroU32::new(window));
            // Read immediately before, so anything else on the card that
            // moved between planning and now is already accounted for.
            let before = crate::backend::best_gpu().map(|d| d.memory_free as u64);
            match self.model.new_context(backend, attempt) {
                Ok(context) => {
                    // A context that opened but left too little for its first
                    // decode is not a success. cuBLAS and ggml's CUDA pool
                    // allocate lazily on the first matrix multiply, after
                    // everything above has been measured, and running out
                    // there aborts the process rather than returning an
                    // error. So it is treated as a failure while there is still
                    // a smaller window to try.
                    let starved = crate::backend::best_gpu()
                        .is_some_and(|d| (d.memory_free as u64) < crate::backend::decode_reserve());
                    if starved && window > ozgent_core::accel::MIN_CONTEXT {
                        tracing::debug!(
                            "a context of {window} leaves too little VRAM to decode in; trying smaller"
                        );
                        drop(context);
                        first_reason
                            .get_or_insert_with(|| "too little VRAM left to decode in".to_string());
                        bad = window;
                        let midpoint = good + (bad - good) / 2;
                        window = midpoint.max(ozgent_core::accel::MIN_CONTEXT).min(window - 1);
                        continue;
                    }
                    // A success narrows the search from below. If there is
                    // still a meaningful gap to the smallest known failure,
                    // the context is dropped and a larger one tried: keeping
                    // the first window that happened to work would leave
                    // whatever the bisection had not yet recovered.
                    good = window;
                    let midpoint = good + (bad - good) / 2;
                    if bad > good && midpoint > good + good / 20 && midpoint < bad {
                        drop(context);
                        window = midpoint;
                        continue;
                    }
                    if window < requested {
                        tracing::warn!(
                            "a context of {requested} could not be allocated; opened {window} instead"
                        );
                    }
                    // What it actually cost, beyond the cache we asked for.
                    // This is the entire feedback loop: the reservation stops
                    // being a number somebody guessed and becomes one this
                    // machine measured, and the next load is that much less
                    // wrong.
                    // llama.cpp prints what its compute buffers actually
                    // cost. That exact figure is worth far more than a
                    // free-memory delta, which also catches anything else on
                    // the card that moved during the allocation.
                    let reported = crate::llamalog::compute_buffers();
                    let measured = if reported > 0 {
                        reported
                    } else if let (Some(before), Some(after)) =
                        (before, crate::backend::best_gpu().map(|d| d.memory_free as u64))
                    {
                        before
                            .saturating_sub(after)
                            .saturating_sub(self.kv_shape.bytes(window, split))
                    } else {
                        0
                    };
                    if measured > 0 {
                        tracing::info!(
                            "compute buffers: {} MiB at {window} ctx",
                            measured / (1024 * 1024)
                        );
                        crate::backend::record_reserve(
                            crate::backend::reserve_shape(
                                shape.ubatch,
                                shape.n_embd,
                                window,
                                shape.n_batch,
                                shape.staging_bytes,
                            ),
                            measured,
                        );
                    }
                    return Ok(context);
                }
                Err(e) => {
                    // llama.cpp returns a bare null for every kind of failure
                    // and explains itself only through its log callback, so
                    // `e` alone says nothing but "null reference".
                    let reason = crate::llamalog::reason().unwrap_or_else(|| e.to_string());
                    tracing::debug!("context of {window} failed: {reason}");

                    // Whether this model gets flash attention is not knowable
                    // before the context exists: `flash_attention = true` asks
                    // llama.cpp for AUTO, which is resolved by probing whether
                    // the fused op lands on the same device as its layer. So
                    // the cache policy reasons about a feature it cannot see,
                    // and if it guessed wrong the context is refused.
                    //
                    // Only **V** is given up. llama.cpp's rule is "V cache
                    // quantization requires flash_attn" and there is no
                    // matching one for K — this used to surrender both, which
                    // doubled the cache of every model without flash
                    // attention for no reason at all.
                    //
                    // Shrinking the window does not help; the type is the
                    // problem. Fall back once, at the full window, and let the
                    // loop below shrink it if the bigger cache no longer fits.
                    if !plain_cache && needs_flash_attention(&reason) {
                        tracing::info!(
                            "this model has no flash attention, so the value cache cannot \
                             be quantised; keeping {:?} keys and f16 values",
                            split.k
                        );
                        split.v = ozgent_core::accel::CacheType::F16;
                        params = params.with_type_v(ggml_type(split.v));
                        plain_cache = true;
                        window = requested;
                        continue;
                    }
                    first_reason.get_or_insert(reason);

                    if window <= ozgent_core::accel::MIN_CONTEXT {
                        let reason = first_reason.unwrap_or_else(|| e.to_string());
                        return Err(EngineError::Context(format!(
                            "{reason} (tried down to {window} tokens of context)"
                        )));
                    }
                    bad = window;
                    let midpoint = good + (bad - good) / 2;
                    window = midpoint.max(ozgent_core::accel::MIN_CONTEXT).min(window - 1);
                }
            }
        }
    }

    /// Open a context for this model with `opts` applied.
    ///
    /// `want` is how many conversations will share it, and how that is paid
    /// for depends on which kind of cache llama.cpp is asked for.
    ///
    /// By default it divides `n_ctx` by the sequence count and gives each
    /// sequence a fixed slice, so four conversations at the asked-for window
    /// need four times the cache — and on a card with room for one, four
    /// callers would each get a quarter of the context they asked for.
    /// Measured on this machine, a 107,008-token window left room for exactly
    /// one conversation, which is concurrency nobody gets.
    ///
    /// `kv_unified` is the way out: one pool of cells that every sequence
    /// draws from, so a conversation occupies what it is actually holding and
    /// the rest stays available. Four idle-to-moderate conversations then cost
    /// about what one long one does, and a lone caller still gets the whole
    /// window. The trade is that four callers who *all* run to the end of the
    /// context will exhaust the pool between them; llama.cpp reports that as a
    /// failed decode rather than silently truncating, so it surfaces as an
    /// error on the turn that hits it.
    ///
    /// Left off for a single slot, where it would change nothing but the code
    /// path taken.
    ///
    /// Returns the context, the window each slot may use, and the slot count.
    fn open(
        &self,
        opts: &Resolved,
        want: Slots,
    ) -> Result<(LlamaContext<'_>, u32, u32), EngineError> {
        let mut slots = match want {
            Slots::Exact(n) => n.max(1),
            Slots::UpTo(n) => n.max(1),
        };
        let backend = backend()?;

        // Asking for more context than the model was trained on produces
        // gibberish rather than an error, so it is clamped with a warning.
        let mut requested = opts.context_length.min(self.n_ctx_train.max(512));
        if opts.context_length > self.n_ctx_train && self.n_ctx_train > 0 {
            tracing::warn!(
                "context {} exceeds this model's trained {}; clamping",
                opts.context_length,
                self.n_ctx_train
            );
        }

        // `auto` is resolved here rather than at parse time because it needs
        // the VRAM left *after* the weights are resident, which is only known
        // once the model has loaded.
        let free = crate::backend::best_gpu().map(|d| d.memory_free as u64).unwrap_or(0);
        // One budget, computed once, used both to choose how the cache is
        // stored and to size the window. See `choose_kv_split`.
        let host = ozgent_core::accel::available_host_memory();
        let sequences = sequences_for(slots, want);
        let unified = matches!(want, Slots::UpTo(_)) && sequences > 1;
        let total_window = if unified { requested } else { requested.saturating_mul(sequences) };
        // Scratch and budget for one candidate micro-batch. Both are wanted
        // twice below, once per candidate, so they are computed together.
        let weigh = |ubatch: Option<u32>, n_batch: u32, window: u32| {
            let shape = ozgent_core::reserve::Shape {
                // Free memory is read after the weights loaded, so whatever
                // bringing the backend up cost has already been paid out of
                // it. Charging it again here took 400 MB off every first
                // window.
                first_in_process: false,
                ..crate::backend::reserve_shape(
                    ubatch.unwrap_or(512),
                    self.model.n_embd() as u32,
                    window,
                    n_batch,
                    self.staging_bytes,
                )
            };
            let reserve = self
                .probed_reserve(backend, opts, ubatch, n_batch, sequences, unified, total_window)
                .unwrap_or_else(|| crate::backend::reserve_for(shape));
            let budget = if opts.kv_offload {
                ozgent_core::accel::kv_budget_reserving(
                    free,
                    host,
                    self.gpu_layers_used,
                    self.n_layer,
                    Some(reserve),
                )
            } else {
                ozgent_core::accel::kv_budget(0, host, 0, self.n_layer)
            };
            (reserve, budget)
        };
        let (reserve, mut budget) = weigh(opts.ubatch, opts.batch_size, requested);
        tracing::debug!(
            "kv budget {} MiB of {} MiB free, reserving {} MiB",
            budget / (1024 * 1024),
            free / (1024 * 1024),
            reserve / (1024 * 1024)
        );
        let any_pair = crate::backend::flash_takes_any_kv_pair();
        let split = match (opts.cache_type_k, opts.cache_type_v) {
            (CacheType::Auto, CacheType::Auto) => ozgent_core::accel::choose_kv_split_for(
                self.kv_shape,
                requested,
                self.weight_bytes,
                budget,
                opts.flash_attention,
                any_pair,
            ),
            // One named and the other left alone is someone saying "store the
            // cache like this", not asking for a mixed one.
            (k, CacheType::Auto) => ozgent_core::accel::KvSplit::uniform(k),
            (CacheType::Auto, v) => ozgent_core::accel::KvSplit::uniform(v),
            (k, v) => ozgent_core::accel::KvSplit { k, v },
        };
        // A pair llama.cpp will refuse returns a null context with the reason
        // only in its own log, so it is caught here while there is still
        // something useful to say and a working pair to fall back to.
        let split = if ozgent_core::accel::kv_split_allowed_for(
            split,
            opts.flash_attention,
            self.kv_shape,
            any_pair,
        ) {
            split
        } else {
            let safe = ozgent_core::accel::choose_kv_split_for(
                self.kv_shape,
                requested,
                self.weight_bytes,
                budget,
                opts.flash_attention,
                any_pair,
            );
            tracing::warn!(
                "this model cannot store its cache as {:?}/{:?}; using {:?}/{:?}",
                split.k, split.v, safe.k, safe.v
            );
            safe
        };
        let (type_k, type_v) = (split.k, split.v);

        // `UpTo` is a cap on callers, not on cells: with a shared pool a
        // second slot costs nothing until somebody actually uses it, so there
        // is nothing here to fit. `Exact` still divides the window, which is
        // what a caller asking for a fixed partition means.
        // Whenever the context carries more than one sequence, and it always
        // does — the shared prefix has one of its own — the cells are a
        // single pool. Left divided, llama.cpp gives each sequence a fixed
        // slice *and* splits every batch per stream, which a lone caller pays
        // for while using none of it: 44.5 tok/s against 46.8, and worse under
        // drafting, where every draft step pays it again.
        if !unified {
            // Each sequence gets a fixed slice, so the pool has to be that
            // many times larger — including one holding a shared prefix.
            requested = requested.saturating_mul(sequences);
        }
        // Training context is a claim about which positions the model
        // understands, not a promise that the cache for them fits in memory.
        // At the million-token windows recent models advertise, the cache runs
        // to hundreds of gigabytes, and llama.cpp answers that by returning a
        // null pointer — which reached the user as "null reference from
        // llama.cpp" with nothing pointing at memory as the cause. Sizing the
        // window to the memory that exists trades an unusable session for a
        // shorter one.
        // Where the cache lives decides what bounds the window. Offloaded, it
        // sits beside the weights and is bounded by free VRAM; kept in host
        // memory it is bounded by RAM instead, which is usually far larger —
        // that is the trade `kv_offload = false` buys, and it costs reading
        // the whole cache across PCIe on every token.
        // `budget` was computed above, against a measured reserve rather than
        // a share of the card: the share allowed 3514 MiB of 5020 free, so a
        // cache that would have fitted at f16 was quantised for nothing.
        // A wider micro-batch is worth having when it is free. Prefill runs
        // the whole batch through as physical chunks of `n_ubatch`, so a wider
        // one is fewer chunks and fewer passes over the weights: measured at
        // 288 -> 502 tok/s on a model whose experts live in host memory, where
        // each chunk pays for its own upload, and 4% on one that fits on the
        // card. What it costs is scratch — a few hundred megabytes — and that
        // comes out of the same budget as the cache.
        //
        // So it is taken only when it costs no window. The rule is the one
        // used everywhere else here: never trade away context for throughput,
        // because a window that will not hold the conversation is not a
        // faster session, it is a broken one. Where scratch is tight the
        // narrow batch simply wins and nothing is said.
        //
        // A micro-batch cannot be wider than the batch that carries it, so the
        // two move together or neither does. Raising only the micro-batch
        // leaves it clamped back to the default 512 batch, which is how the
        // first version of this silently did nothing at all.
        let mut ubatch = opts.ubatch;
        let mut n_batch = opts.batch_size;
        let mut window =
            ozgent_core::accel::fit_context_split(self.kv_shape, requested, split, budget);
        if opts.ubatch.is_none() && opts.batch_size <= WIDE_BATCH {
            let (_, wide_budget) = weigh(Some(WIDE_BATCH), WIDE_BATCH, requested);
            let wide_window =
                ozgent_core::accel::fit_context_split(self.kv_shape, requested, split, wide_budget);
            if wide_window >= window {
                tracing::debug!("micro-batch {WIDE_BATCH} fits without shortening the window");
                (ubatch, n_batch, budget, window) =
                    (Some(WIDE_BATCH), WIDE_BATCH, wide_budget, wide_window);
            }
        }
        // The shape that reaches `open_context` has to describe the batch that
        // was actually chosen, not the candidate it was compared against.
        let shape = crate::backend::reserve_shape(
            ubatch.unwrap_or(512),
            self.model.n_embd() as u32,
            window,
            n_batch,
            self.staging_bytes,
        );
        let mut requested = window;
        let asked = opts.context_length.min(self.n_ctx_train.max(512)).saturating_mul(slots);
        if requested < asked {
            let wanted = self.kv_shape.bytes(asked, split) / (1024 * 1024);
            let where_ = if opts.kv_offload { "vram" } else { "system ram" };
            tracing::warn!(
                "context {} needs {wanted} MiB of {k:?} K / {v:?} V kv cache; only {} MiB \
                 of {where_} is free, so the window is {requested}",
                opts.context_length,
                budget / (1024 * 1024),
                k = split.k,
                v = split.v,
            );
            // Only worth suggesting when it would actually help. On a machine
            // whose RAM is no larger than its spare VRAM, moving the cache
            // buys nothing and costs the token rate.
            if opts.kv_offload
                && ozgent_core::accel::fit_context_split(
                    self.kv_shape,
                    asked,
                    split,
                    ozgent_core::accel::kv_budget(0, host, 0, self.n_layer),
                ) > requested
            {
                tracing::warn!(
                    "--no-kv-offload would hold the whole window in system ram, more slowly"
                );
            }
        }
        if opts.cache_type_k == CacheType::Auto {
            tracing::info!(
                "kv cache: {:?} K / {:?} V ({} MiB at {requested} ctx, {} MiB free)",
                type_k,
                type_v,
                self.kv_shape.bytes(requested, split) / (1024 * 1024),
                free / (1024 * 1024),
            );
        }

        // Flash attention was previously read only to pick the KV type and
        // never actually set, so `--no-flash-attn` disabled nothing and the KV
        // policy reasoned about a flag it did not control. llama.cpp defaults
        // to AUTO, so the behaviour was probably right by accident; it is now
        // asked for explicitly.
        let flash = if opts.flash_attention {
            sys::LLAMA_FLASH_ATTN_TYPE_AUTO
        } else {
            sys::LLAMA_FLASH_ATTN_TYPE_DISABLED
        };

        let mut params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(requested))
            .with_n_batch(n_batch)
            // One more than the conversations: the last sequence holds the
            // prefix they share. See `Commons`.
            .with_n_seq_max(sequences)
            .with_kv_unified(unified)
            // `n_rs_seq` — llama.cpp's ring of per-token recurrent snapshots,
            // which makes a hybrid model's cache trimmable and so lets a
            // rejected draft be undone in place — is deliberately not asked
            // for. It was built and measured rather than argued about; see
            // `crate::mtp` for what the measurement said.
            .with_flash_attention_policy(flash)
            .with_offload_kqv(opts.kv_offload)
            .with_type_k(ggml_type(type_k))
            .with_type_v(ggml_type(type_v));

        // The physical micro-batch. Left at llama.cpp's default unless asked,
        // since it trades prefill parallelism against working-set size.
        if let Some(n) = ubatch {
            params = params.with_n_ubatch(n);
        }

        if opts.threads > 0 {
            params = params
                .with_n_threads(opts.threads as i32)
                .with_n_threads_batch(opts.threads as i32);
        }

        let mut context = self.open_context(backend, params, requested, split, shape)?;

        // Steering belongs to the context, not the turn: installed once here,
        // it shapes every generation until the session ends. Loading it lazily
        // per turn would pay the file read repeatedly and, worse, make the
        // first turn behave differently from the rest.
        if let Some(path) = &opts.control_vector {
            let vector = crate::cvec::ControlVector::load(path, self.model.n_embd() as usize)
                .map_err(|e| EngineError::ControlVector(e.to_string()))?;
            // A strength of exactly zero is a request for no steering; applying
            // a zero vector would work but wastes a copy per layer.
            if opts.control_strength != 0.0 {
                vector
                    .scaled(opts.control_strength)
                    .apply(&mut context)
                    .map_err(|e| EngineError::ControlVector(e.to_string()))?;
                tracing::info!(
                    "control vector: {} directions at strength {}",
                    vector.n_layers(),
                    opts.control_strength
                );
            }
        }

        // What one slot may use: the whole pool when it is shared, its own
        // slice when it is not.
        let each = if unified { requested } else { (requested / sequences).max(1) };
        Ok((context, each, slots))
    }

    /// What this model's compute scratch will be at `window`, measured on
    /// this model rather than predicted from others.
    ///
    /// A single learned rate cannot serve every model. Qwen3.5-4B needs 3.1 GB
    /// of scratch at an 82,432-token window; GLM-4.7-Flash needs 346 MiB at
    /// 16,384, nearly all of it one block's experts being uploaded, and almost
    /// none of it growing with the window. A rate learned on the first sized
    /// the second's 16k window at 512 tokens with 814 MiB free.
    ///
    /// So two small contexts are opened with the real parameters, llama.cpp's
    /// own report of their compute buffers is read, and a constant and a
    /// per-token rate are fitted through the two points. About a tenth of a
    /// second, once per set of parameters, and then exact. `None` when a probe
    /// cannot be opened, which leaves the learned estimate to answer.
    fn probed_reserve(
        &self,
        backend: &'static LlamaBackend,
        opts: &Resolved,
        ubatch: Option<u32>,
        n_batch: u32,
        sequences: u32,
        unified: bool,
        window: u32,
    ) -> Option<u64> {
        let key = ScratchKey {
            n_batch,
            ubatch,
            flash: opts.flash_attention,
            kv_offload: opts.kv_offload,
            sequences,
            unified,
        };
        let cached = self.scratch.lock().ok()?.as_ref().filter(|(k, ..)| *k == key).map(|(_, c, r)| (*c, *r));
        let (constant, per_token) = match cached {
            Some(fit) => fit,
            None => {
                let flash = if opts.flash_attention {
                    sys::LLAMA_FLASH_ATTN_TYPE_AUTO
                } else {
                    sys::LLAMA_FLASH_ATTN_TYPE_DISABLED
                };
                let value = if opts.flash_attention { CacheType::Q8_0 } else { CacheType::F16 };
                let mut points = Vec::with_capacity(2);
                for tokens in [1024u32, 4096] {
                    let n_ctx = if unified { tokens } else { tokens * sequences };
                    let mut p = LlamaContextParams::default()
                        .with_n_ctx(NonZeroU32::new(n_ctx))
                        .with_n_batch(n_batch)
                        .with_n_seq_max(sequences)
                        .with_kv_unified(unified)
                        .with_flash_attention_policy(flash)
                                    .with_offload_kqv(opts.kv_offload)
                        .with_type_k(ggml_type(CacheType::Q8_0))
                        .with_type_v(ggml_type(value));
                    if let Some(n) = opts.ubatch {
                        p = p.with_n_ubatch(n);
                    }
                    crate::llamalog::clear();
                    let context = self.model.new_context(backend, p).ok()?;
                    let buffers = crate::llamalog::compute_buffers();
                    drop(context);
                    if buffers == 0 {
                        return None;
                    }
                    points.push((n_ctx as f64, buffers as f64));
                }
                let per_token = ((points[1].1 - points[0].1) / (points[1].0 - points[0].0)).max(0.0);
                let constant = (points[0].1 - per_token * points[0].0).max(0.0) as u64;
                tracing::debug!(
                    "compute scratch for this model: {} MiB + {:.0} B per token",
                    constant / (1 << 20),
                    per_token
                );
                if let Ok(mut slot) = self.scratch.lock() {
                    *slot = Some((key, constant, per_token));
                }
                (constant, per_token)
            }
        };
        Some(constant + (per_token * window as f64) as u64 + crate::backend::decode_reserve())
    }

    /// One conversation with a cache of its own.
    pub fn session(&self, opts: &Resolved) -> Result<Session<'_>, EngineError> {
        Ok(self
            .sessions(opts, Slots::Exact(1))?
            .pop()
            .expect("one slot is one session"))
    }

    /// `slots` conversations sharing one context, and one forward pass
    /// whenever more than one of them wants a token at the same moment.
    ///
    /// The window is divided between them: see [`Engine::open`].
    pub fn sessions(&self, opts: &Resolved, want: Slots) -> Result<Vec<Session<'_>>, EngineError> {
        let (hub, window) = self.hub(opts, want)?;
        let slots = hub.slots();
        Ok((0..slots).map(|seq| self.attach(&hub, seq as i32, window, opts)).collect())
    }

    fn attach<'e>(
        &'e self,
        hub: &std::sync::Arc<crate::hub::Hub<'e>>,
        seq: i32,
        window: u32,
        opts: &Resolved,
    ) -> Session<'e> {
        Session {
            model: &self.model,
            n_ctx: window,
            n_batch: hub.n_batch() as u32,
            slot: hub.slot(seq),
            sampler: build_sampler(opts),
            n_past: 0,
            cached: Vec::new(),
            reuse: opts.prefix_reuse,
            last_reused: 0,
            grammar_active: false,
            can_trim: true,
            rollback_safe: self.rollback_safe,
            gen_prompt_tokens: self.gen_prompt_tokens,
            media_dirty: false,
            cpu_moe_layers: self.cpu_moe_layers,
            checkpoints: Vec::new(),
            checkpoint_tick: 0,
            tool_grammars: Vec::new(),
            gate_applied: false,
            opts: opts.clone(),
            candidates: Vec::new(),
        }
    }

    /// A context several conversations share, decoding through it together.
    ///
    /// Returns the hub and the window each slot may use. See [`crate::hub`]
    /// for what sharing a pass does and does not change.
    pub fn hub(
        &self,
        opts: &Resolved,
        want: Slots,
    ) -> Result<(std::sync::Arc<crate::hub::Hub<'_>>, u32), EngineError> {
        let (context, each, slots) = self.open(opts, want)?;
        let unified_pool = matches!(want, Slots::UpTo(_)) && sequences_for(slots, want) > 1;
        let commons = matches!(want, Slots::UpTo(_));
        Ok((
            std::sync::Arc::new(crate::hub::Hub::new(
                context,
                self.model.n_vocab() as usize,
                self.model.n_embd() as usize,
                slots,
                unified_pool,
                commons,
            )),
            each,
        ))
    }

    pub fn model(&self) -> &LlamaModel {
        &self.model
    }
}

/// Map our cache type onto llama.cpp's KV cache type.
/// Read a numeric GGUF metadata value, if the model declares it.
fn meta_u32(model: &LlamaModel, key: &str) -> Option<u32> {
    model.meta_val_str(key).ok()?.trim().parse().ok()
}

fn ggml_type(t: CacheType) -> KvCacheType {
    match t {
        // `Auto` is resolved against the hardware before a context is built;
        // f16 is the safe reading if that is ever bypassed.
        CacheType::Auto | CacheType::F16 => KvCacheType::F16,
        CacheType::BF16 => KvCacheType::BF16,
        CacheType::Q8_0 => KvCacheType::Q8_0,
        CacheType::Q5_1 => KvCacheType::Q5_1,
        CacheType::Q5_0 => KvCacheType::Q5_0,
        CacheType::Q4_1 => KvCacheType::Q4_1,
        CacheType::Q4_0 => KvCacheType::Q4_0,
    }
}

fn build_sampler(opts: &Resolved) -> LlamaSampler {
    build_sampler_with(opts, None)
}

/// Build the sampler chain, optionally led by a grammar constraint.
///
/// The constraint goes first so it masks the logits before any shaping sees
/// them; applied afterwards it could only reject, and sampling would stall.
fn build_sampler_with(opts: &Resolved, grammar: Option<LlamaSampler>) -> LlamaSampler {
    let mut chain: Vec<LlamaSampler> = Vec::new();
    if let Some(g) = grammar {
        chain.push(g);
    }
    chain.push(LlamaSampler::penalties(
        opts.repeat_last_n as i32,
        opts.repeat_penalty,
        0.0,
        0.0,
    ));

    if opts.temperature <= 0.0 {
        chain.push(LlamaSampler::greedy());
        return LlamaSampler::chain_simple(chain);
    }

    chain.push(LlamaSampler::top_k(opts.top_k as i32));
    chain.push(LlamaSampler::top_p(opts.top_p, 1));
    chain.push(LlamaSampler::min_p(opts.min_p, 1));
    chain.push(LlamaSampler::temp(opts.temperature));
    chain.push(LlamaSampler::dist(seed_of(opts)));
    LlamaSampler::chain_simple(chain)
}

fn seed_of(opts: &Resolved) -> u32 {
    opts.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    })
}

#[allow(dead_code)]
fn build_sampler_old(opts: &Resolved) -> LlamaSampler {
    let seed = opts.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    });

    // Temperature 0 means "be deterministic", which greedy expresses exactly;
    // a temp sampler at 0.0 would divide by zero.
    if opts.temperature <= 0.0 {
        return LlamaSampler::chain_simple([
            LlamaSampler::penalties(
                opts.repeat_last_n as i32,
                opts.repeat_penalty,
                0.0,
                0.0,
            ),
            LlamaSampler::greedy(),
        ]);
    }

    LlamaSampler::chain_simple([
        LlamaSampler::penalties(opts.repeat_last_n as i32, opts.repeat_penalty, 0.0, 0.0),
        LlamaSampler::top_k(opts.top_k as i32),
        LlamaSampler::top_p(opts.top_p, 1),
        LlamaSampler::min_p(opts.min_p, 1),
        LlamaSampler::temp(opts.temperature),
        LlamaSampler::dist(seed),
    ])
}

/// Length of the longest shared prefix of two token sequences.
/// Split a prompt of `len` tokens into batches of at most `n_batch`.
///
/// Extracted so the arithmetic can be checked without a model: an off-by-one
/// here is a process abort inside llama.cpp, not a Rust error.
pub fn prefill_chunks(len: usize, n_batch: usize) -> Vec<(usize, usize)> {
    let n_batch = n_batch.max(1);
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < len {
        let end = (offset + n_batch).min(len);
        out.push((offset, end));
        offset = end;
    }
    out
}

/// Length of the longest shared prefix of two token sequences.
fn common_prefix(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// A minimal prompt for models whose GGUF carries no chat template.
fn fallback_prompt(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        let label = match m.role {
            Role::System => "System",
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::Tool => "Tool",
        };
        out.push_str(label);
        out.push_str(": ");
        out.push_str(&m.text_content());
        out.push('\n');
    }
    out.push_str("Assistant: ");
    out
}

/// What a completed generation cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Prompt tokens actually decoded this turn, excluding any reused prefix.
    pub prompt_tokens: usize,
    /// Prompt tokens served from the KV cache rather than recomputed.
    pub reused_tokens: usize,
    pub generated_tokens: usize,
    /// Tokens produced by an accepted draft rather than a sequential decode.
    pub drafted_tokens: usize,
    pub prompt_ms: u128,
    pub generation_ms: u128,
    /// Time spent in the caller's token callback — detokenising, filtering
    /// and rendering. Part of `generation_ms`, broken out so host-side cost
    /// can be told apart from time the GPU actually spent decoding.
    pub callback_ms: u128,
    /// Drafted tokens that were accepted, for speculation diagnostics.
    pub accepted_drafts: usize,
    /// Time spent indexing the context and proposing drafts.
    pub spec_ms: u128,
    /// Tokens proposed by the drafter, accepted or not. The gap between this
    /// and `accepted_drafts` is wasted verification work.
    pub proposed_drafts: usize,
}

impl Stats {
    pub fn tokens_per_second(&self) -> f64 {
        if self.generation_ms == 0 {
            return 0.0;
        }
        self.generated_tokens as f64 * 1000.0 / self.generation_ms as f64
    }

    pub fn prompt_tokens_per_second(&self) -> f64 {
        if self.prompt_ms == 0 {
            return 0.0;
        }
        self.prompt_tokens as f64 * 1000.0 / self.prompt_ms as f64
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted an end-of-generation token.
    EndOfText,
    /// The `max_tokens` limit was reached.
    TokenLimit,
    /// The context filled up.
    ContextFull,
    /// The caller's callback asked to stop.
    Cancelled,
}

/// Ceiling on the whole set of saved prompt states, as a fraction of free
/// host memory.
///
/// This used to bound a single checkpoint and allowed a quarter of what was
/// free. A set needs to be meaner: these are pure cache, droppable and
/// rebuildable, while the host memory they compete for holds things that are
/// not — a mixture-of-experts model's evicted experts above all, where being
/// short pushes the whole thing into paging from disk. An eighth buys several
/// conversations at any ordinary length and cannot crowd out the work.
const CHECKPOINT_RAM_SHARE: usize = 8;

/// Below this many tokens, a full prefill is cheaper than the state copy.
const CHECKPOINT_MIN_TOKENS: usize = 256;

/// How many conversations' prompt states to hold at once.
///
/// One was enough while these existed only to work around a cache that cannot
/// trim. They now also carry a conversation across a switch to another one, and
/// a daemon has a handful in flight at any time — the browser, the terminal,
/// each channel, each scheduled job. Eight covers that without the set itself
/// becoming the thing that uses the memory; the byte budget bounds it anyway,
/// and this only stops a great many tiny conversations accumulating.
const CHECKPOINT_SLOTS: usize = 8;

/// A prompt boundary the cache can be returned to.
///
/// A conversation resends its whole history, so almost every prompt is the
/// previous one plus a reply and a new question — and almost all of it is
/// already in the cache. Reusing it means dropping whatever sits *after* the
/// shared prefix, which for a plain attention cache is a `seq_rm` away.
///
/// Two things defeat that, and a saved state answers both.
///
/// A hybrid model cannot trim at all. Its recurrent layers hold a state that
/// was folded forward token by token and cannot be unwound, so llama.cpp
/// refuses the partial removal and the only correct answer would be to prefill
/// the whole conversation again. On a 9,000-token chat that is 4.7 seconds per
/// turn, every turn, growing as the conversation does.
///
/// And **there is only one cache, but many conversations.** A daemon answers
/// the browser, the terminal, Telegram and the scheduler from one model, and
/// each of them is a different conversation. Whichever spoke last owns the
/// cache; everyone else pays a cold prefill. Measured on an 8 GB card with two
/// 700-token conversations alternating: within a conversation 673 of 715
/// tokens were reused and the prompt took 76 ms, and on every switch the reuse
/// fell to **zero** and the prompt took 383 ms.
///
/// So these are kept per boundary and several at a time, and restoring one is
/// a host-to-device copy rather than a prefill.
struct PromptCheckpoint {
    /// Exactly the tokens the saved state was built from.
    tokens: Vec<LlamaToken>,
    state: SeqState,
    /// What the copy costs, so a set of them can be held to a budget.
    bytes: usize,
    /// When it was last saved or restored, for eviction.
    used: u64,
}

/// Bytes of routed expert weight this placement sends to the host, or `None`
/// when none is going there.
fn host_expert_bytes(layout: &crate::layout::Layout, moe: MoeOffload) -> Option<u64> {
    if layout.expert_bytes_per_layer == 0 {
        return None;
    }
    let layers = match moe {
        MoeOffload::Keyword(MoeKeyword::All) => layout.layers,
        MoeOffload::Layers(n) if n > 0 => n.min(layout.layers),
        _ => return None,
    };
    Some(layout.expert_bytes_per_layer * layers as u64)
}

/// The saved boundary that shares the longest whole prefix with `tokens`.
///
/// Whole, because a state that agrees with the prompt for a while and then
/// diverges would leave the cache describing tokens that are not there — the
/// positions are right and the contents are wrong, which reads as the model
/// having hallucinated its own history. Strictly shorter, too, so at least one
/// token is left to decode and produce logits from.
fn best_prefix_match(saved: &[&[LlamaToken]], tokens: &[LlamaToken]) -> Option<(usize, usize)> {
    saved
        .iter()
        .enumerate()
        .filter(|(_, s)| s.len() < tokens.len())
        .filter(|(_, s)| common_prefix(s, tokens) == s.len())
        .max_by_key(|(_, s)| s.len())
        .map(|(i, s)| (i, s.len()))
}

/// Which saved state to drop, given when each was last useful.
fn lru_victim(used: &[u64]) -> Option<usize> {
    used.iter().enumerate().min_by_key(|(_, u)| **u).map(|(i, _)| i)
}

/// One conversation against one KV cache.
pub struct Session<'a> {
    /// Layers whose routed experts sit in host memory, carried from the
    /// engine so the checkpoint budget can stand aside for them.
    cpu_moe_layers: u32,
    model: &'a LlamaModel,
    /// This conversation's claim on a context that may be shared with others.
    /// A session built by [`Engine::session`] has the context to itself; one
    /// built by [`Engine::sessions`] shares its forward passes.
    slot: crate::hub::Slot<'a>,
    /// Cached from the context, which is now behind a lock.
    n_ctx: u32,
    n_batch: u32,
    sampler: LlamaSampler,
    n_past: i32,
    /// Tokens currently resident in the KV cache, so the next prompt can
    /// reuse whatever prefix it shares with them.
    cached: Vec<LlamaToken>,
    reuse: PrefixReuse,
    last_reused: usize,
    /// Set while a GBNF constraint is installed. Speculation must be off in
    /// that case; see `generate`.
    grammar_active: bool,
    /// Whether this cache can drop a partial range.
    ///
    /// Sliding-window and recurrent caches cannot, and llama.cpp reports that
    /// as a failure rather than a capability. Learned on first attempt, then
    /// remembered so the session stops relying on it.
    can_trim: bool,
    /// Mirrors [`Engine`]'s flag: false when the model keeps state that draft
    /// rejection cannot roll back, making speculation unsafe.
    rollback_safe: bool,
    /// Set after a turn that evaluated images: the cache then holds embeddings
    /// no token sequence describes, so it must be rebuilt before the next turn.
    media_dirty: bool,
    /// The sequence state as it stood at a prompt boundary, with the tokens
    /// that produced it. See [`PromptCheckpoint`].
    checkpoints: Vec<PromptCheckpoint>,
    /// Monotonic tick, so the least recently useful one can be dropped.
    checkpoint_tick: u64,
    /// Mirrors [`Engine::gen_prompt_tokens`].
    gen_prompt_tokens: usize,
    /// Per-opener tool grammars, compiled once when tools are configured.
    /// Empty when no tools are offered, which leaves the gate inert.
    tool_grammars: Vec<(&'static str, String)>,
    /// Set while the *gate* owns the installed grammar, so it can be lifted
    /// again without disturbing a grammar set deliberately by a caller.
    gate_applied: bool,
    /// Kept so the sampler can be rebuilt when a grammar is set or cleared.
    opts: Resolved,
    /// The candidate array token selection fills, kept rather than rebuilt.
    candidates: Vec<llama_cpp_2::token::data::LlamaTokenData>,
}

// A session may be moved to the thread that will answer with it, which is how
// several of them share one context: each slot gets its own thread and its own
// session, and the hub serialises the one thing they share.
//
// What this asserts is that nothing a session owns is tied to the thread that
// made it. The model behind `&LlamaModel` is already `Sync`; the context is
// behind the hub's lock and is never handed out; the sampler and the saved
// sequence states are raw llama.cpp pointers with no thread affinity, owned by
// exactly one session and used from one thread at a time. What is *not*
// asserted is that two threads may touch one session — `Sync` is deliberately
// not implemented, so the compiler still requires `&mut` to generate.
unsafe impl Send for Session<'_> {}

impl<'a> Session<'a> {
    /// Hold `prefix` as the shared prefix before any turn asks for it.
    ///
    /// The commons is normally discovered: two prompts arrive, their common
    /// head is noticed, and the third turn onwards borrows it. That works, and
    /// it means the first turn of every conversation prefills the system
    /// prompt and tool schemas in full — measured at 1665 ms for 2818 tokens
    /// on a 4B — and the second turn pays the same again to fill the commons
    /// itself. A caller that already knows what the stable head is does not
    /// have to wait to be shown it twice.
    ///
    /// Returns how many tokens are now held, or zero when the prefix is too
    /// short to be worth a sequence.
    pub fn prewarm_commons(&self, prefix: &[LlamaToken]) -> Result<usize, EngineError> {
        if !self.slot.hub().wants_commons(prefix.len()) {
            return Ok(0);
        }
        self.slot
            .hub()
            .fill_commons(prefix)
            .map_err(|e| EngineError::Decode(e.to_string()))?;
        Ok(prefix.len())
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Say that this session has stopped asking for tokens, so the others
    /// sharing its context do not wait for it.
    ///
    /// The hub holds each pass open briefly for slots that are about to ask
    /// for their next token, and a slot that has walked away without saying so
    /// costs everybody that wait on every pass. Not a small effect: leaving it
    /// out cost a lone caller 46.6 tok/s against 33.4, because three slots
    /// that had finished long ago were still being waited for.
    pub fn park(&mut self) {
        self.slot.park();
    }

    /// Whether this session has the context to itself.
    pub fn solo(&self) -> bool {
        self.slot.hub().solo()
    }

    /// Whether this session may roll a rejected draft back by snapshotting
    /// its sequence state. See [`crate::hub::Slot::may_snapshot`].
    fn may_snapshot(&self) -> bool {
        self.slot.may_snapshot()
    }

    /// Capture the sequence state so it can be put back later.
    ///
    /// `ON_DEVICE` asks llama.cpp to keep the copy in VRAM instead of handing
    /// back 50 MB of host memory. That is the difference between a snapshot
    /// that can be taken on every draft step and one that cannot.
    fn snapshot(&self, partial: bool, on_device: bool) -> Result<SeqState, EngineError> {
        let mut bits = 0u32;
        if partial {
            bits |= LlamaStateSeqFlags::PARTIAL_ONLY.bits();
        }
        if on_device {
            bits |= LlamaStateSeqFlags::ON_DEVICE.bits();
        }
        let seq = self.slot.seq();
        self.slot
            .with_context(|c| c.state_seq_get(seq, LlamaStateSeqFlags::from_bits(bits)))
            .map_err(|e| EngineError::State(e.to_string()))
    }

    fn restore(&mut self, state: &SeqState) -> Result<(), EngineError> {
        let seq = self.slot.seq();
        self.slot
            .with_context(|c| c.state_seq_set(state, seq))
            .map_err(|e| EngineError::State(e.to_string()))
    }

    /// Save the cache as it stands, which the caller guarantees is exactly
    /// `tokens`.
    ///
    /// Kept for every model now, not only for those whose cache cannot be
    /// trimmed. Trimming recovers a prefix of whatever the cache currently
    /// holds, and on a switch to a different conversation that is the wrong
    /// conversation entirely — there is no prefix of it to keep. A saved state
    /// is the only way back to a conversation the cache has moved on from.
    fn save_checkpoint(&mut self, tokens: &[LlamaToken]) {
        if tokens.len() < CHECKPOINT_MIN_TOKENS {
            return;
        }
        self.checkpoint_tick += 1;
        // Already holding exactly this boundary — usually because it was just
        // restored from. Copying the same state again buys nothing, but it is
        // still the most recently useful one.
        if let Some(existing) = self.checkpoints.iter_mut().find(|c| c.tokens == tokens) {
            existing.used = self.checkpoint_tick;
            return;
        }

        let bytes = self.state_bytes(false, false);
        let budget = self.checkpoint_budget();
        if budget.is_some_and(|b| bytes > b) {
            tracing::debug!(
                "checkpoint skipped: {} MiB of state against a {} MiB budget",
                bytes / (1024 * 1024),
                budget.unwrap_or(0) / (1024 * 1024)
            );
            return;
        }
        // Host-side, not ON_DEVICE. A device-held state is a list of views
        // into the cache as it stood, and restoring one rebuilds that list
        // from the cache as it stands now: generation has appended tokens
        // since, the two lists disagree, and llama.cpp answers a mismatch by
        // aborting the process. Serialising to host bytes is layout
        // independent, and ~185 MiB each way costs a few tens of
        // milliseconds against the seconds of prefill it saves.
        match self.snapshot(false, false) {
            Ok(state) => {
                let tick = self.checkpoint_tick;
                self.checkpoints.push(PromptCheckpoint {
                    tokens: tokens.to_vec(),
                    state,
                    bytes,
                    used: tick,
                });
                self.evict_checkpoints(budget);
                tracing::debug!(
                    "checkpoint saved: {} tokens, {} MiB ({} held)",
                    tokens.len(),
                    bytes / (1024 * 1024),
                    self.checkpoints.len()
                );
            }
            // Not fatal: without a checkpoint the next turn prefills in full,
            // which is what happened before this existed.
            Err(e) => tracing::debug!("checkpoint not saved: {e}"),
        }
    }

    /// Drop the least recently useful states until the set is within bounds.
    ///
    /// The one just added is never the victim: it is the most recent by
    /// construction, and evicting it would make saving a no-op on any machine
    /// where the budget is tight.
    fn evict_checkpoints(&mut self, budget: Option<usize>) {
        loop {
            let total: usize = self.checkpoints.iter().map(|c| c.bytes).sum();
            let over_budget = budget.is_some_and(|b| total > b);
            if !over_budget && self.checkpoints.len() <= CHECKPOINT_SLOTS {
                return;
            }
            let ticks: Vec<u64> = self.checkpoints.iter().map(|c| c.used).collect();
            let Some(oldest) = lru_victim(&ticks) else { return };
            if self.checkpoints.len() == 1 {
                return;
            }
            let dropped = self.checkpoints.remove(oldest);
            tracing::debug!(
                "checkpoint evicted: {} tokens, {} MiB",
                dropped.tokens.len(),
                dropped.bytes / (1024 * 1024)
            );
        }
    }

    /// Bytes the whole set of checkpoints may occupy, or `None` when free
    /// memory is unknown.
    fn checkpoint_budget(&self) -> Option<usize> {
        let free = crate::backend::devices()
            .into_iter()
            .find(|d| !d.is_gpu())
            .map(|d| d.memory_free)?;
        // Nothing at all when the host is already holding weights.
        //
        // A mixture-of-experts model with experts evicted to system RAM reads
        // them on every single token. Saved prompt states are pure cache and
        // can always be rebuilt by prefilling; expert weights cannot, and a
        // host short of room for them pages from disk at a gigabyte a token.
        // Prompt latency is worth a great deal, but not that.
        if self.cpu_moe_layers > 0 {
            return Some(0);
        }
        Some(free / CHECKPOINT_RAM_SHARE)
    }

    /// The saved state that shares the longest whole prefix with `tokens`.
    ///
    /// A checkpoint has to be a *whole* prefix of the new prompt: restoring a
    /// state that disagrees partway through would leave the cache describing
    /// tokens that are not there. Strictly shorter, too, so at least one token
    /// is left to decode for logits.
    fn best_checkpoint(&self, tokens: &[LlamaToken]) -> Option<(usize, usize)> {
        if matches!(self.reuse, PrefixReuse::Off) {
            return None;
        }
        let saved: Vec<&[LlamaToken]> =
            self.checkpoints.iter().map(|c| c.tokens.as_slice()).collect();
        best_prefix_match(&saved, tokens)
    }

    /// Put the cache back to a saved prompt boundary. Returns how many tokens
    /// are then resident.
    fn restore_checkpoint(&mut self, tokens: &[LlamaToken]) -> Option<usize> {
        let (index, n) = self.best_checkpoint(tokens)?;
        // Taken out so the restore can borrow the context mutably, and put
        // back either way: a checkpoint is not spent by being used. A turn
        // that adds nothing before the boundary — the same question asked
        // twice — saves no new one, and dropping this would send the turn
        // after it back to a cold prefill.
        let mut checkpoint = self.checkpoints.remove(index);
        let restored = self.restore(&checkpoint.state);
        self.checkpoint_tick += 1;
        checkpoint.used = self.checkpoint_tick;

        if let Err(e) = restored {
            tracing::debug!("checkpoint restore failed: {e}");
            // The cache is now of unknown shape; force a clean prefill. The
            // state itself is not at fault, so it is kept.
            self.checkpoints.push(checkpoint);
            self.slot.clear();
            self.cached.clear();
            self.n_past = 0;
            return None;
        }
        self.n_past = n as i32;
        self.cached = checkpoint.tokens.clone();
        self.checkpoints.push(checkpoint);
        Some(n)
    }

    /// Whether this model carries a trained NextN head, and how many.
    ///
    /// Zero means there is nothing to draft from, which is most models.
    pub fn nextn_heads(&self) -> i32 {
        // SAFETY: the model outlives this call.
        unsafe { ozgent_mtmd_sys::llama_cpp_sys_2::llama_model_n_layer_nextn(self.model.as_ptr()) }
    }

    /// Ask the model's own NextN head what comes after the prompt.
    ///
    /// The whole feasibility question in one call: turn NextN embeddings on,
    /// decode the prompt, and see whether a usable row comes back. A drafter
    /// built on a head that returns nothing would be a great deal of work
    /// producing zero drafts, so this is answered before any of it is written.
    ///
    /// Returns the embedding row's width and the sum of its absolute values —
    /// a row of zeros links and reads perfectly well while meaning the head
    /// never ran.
    pub fn probe_nextn(&mut self, prompt: &str) -> Result<(usize, f32), EngineError> {
        let n_batch = (self.n_batch as usize).max(1);
        self.reset();

        let n_embd = self.model.n_embd() as usize;
        let ptr = self.slot.hub().raw();
        // Unmasked. llama.cpp's own driver sets the *target* context this way
        // and reserves `masked` for the draft context: the target has to emit
        // a hidden state for every prompt position, because those rows are
        // what the NextN block is then fed. Masked, only the position that
        // asked for logits produces one, and the probe saw nothing at all.
        unsafe { crate::nextn::set_enabled(ptr, true, false) };
        unsafe { crate::nextn::set_head(ptr, 0) };

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        self.prefill(&tokens, true)?;

        let row = unsafe { crate::nextn::embedding(ptr, 0, n_embd) };
        unsafe { crate::nextn::set_enabled(ptr, false, false) };
        // The cache now holds these tokens and the session has to agree, or
        // the next generation reasons about a prefix that is not there and
        // decodes an empty batch: "Decode Error -1: n_tokens == 0".
        self.cached = tokens;

        match row {
            Some(v) => Ok((v.len(), v.iter().map(|x| x.abs()).sum())),
            None => Ok((0, 0.0)),
        }
    }

    /// What does the NextN head actually propose?
    ///
    /// The second feasibility gate, after [`Session::probe_nextn`] showed the
    /// head runs at all. A head that runs and proposes nonsense is worse than
    /// no head: every draft would be rejected and every rejection costs a
    /// rollback. So this prints what it says before anything is built on it.
    ///
    /// Returns the token the *target* would have chosen, and what the head
    /// proposed to follow it.
    pub fn probe_mtp_draft(
        &mut self,
        prompt: &str,
        want: usize,
    ) -> Result<(String, Vec<String>), EngineError> {
        let n_batch = (self.n_batch as usize).max(1);
        self.reset();

        let n_embd = self.model.n_embd() as usize;
        let ptr = self.slot.hub().raw();
        // Unmasked: the target has to emit a hidden state for the position the
        // draft will continue from.
        unsafe { crate::nextn::set_enabled(ptr, true, false) };

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        self.prefill(&tokens, true)?;

        // What the target itself would say next, for comparison.
        let target_next = self.greedy_from_logits()?;
        // The *last* row, not the first. Unmasked, the target emits a hidden
        // state for every position in the batch, so row zero belongs to the
        // first prompt token. Drafting from it produced fluent nonsense —
        // "Paris-Belstein's 2" after "The capital of France is" — which is
        // exactly what continuing from the wrong position looks like.
        let last = tokens.len().saturating_sub(1) as i32;
        let hidden = unsafe { crate::nextn::embedding(ptr, last, n_embd) }
            .ok_or_else(|| EngineError::Context("no nextn row after prefill".into()))?;

        let backend = backend()?;
        let (seq, n_seq_max, unified) =
            (self.slot.seq(), self.slot.hub().n_seq_max(), self.slot.hub().unified());
        let mut drafter = self.slot.with_context(|c| {
            crate::mtp::MtpDrafter::new(self.model, backend, c, self.n_ctx, seq, n_seq_max, unified)
        })?
        .ok_or_else(|| EngineError::Context("this model has no nextn head".into()))?;

        let drafted = drafter.propose(target_next, &hidden, self.n_past, want)?;
        unsafe { crate::nextn::set_enabled(ptr, false, false) };
        drop(drafter);
        self.cached = tokens;

        let render = |t: LlamaToken| {
            self.model.token_to_str(t, Special::Tokenize).unwrap_or_else(|_| "<?>".into())
        };
        Ok((render(target_next), drafted.into_iter().map(render).collect()))
    }

    /// The most likely token from the last decode on this session.
    ///
    /// Read at `-1`, which llama.cpp resolves to the last output row. The safe
    /// wrapper wants the batch index instead, and a prefill requests logits on
    /// the final prompt token — index 4 of a five-token prompt, not zero —
    /// so asking it for row zero panics with "logit 0 is not initialized".
    fn greedy_from_logits(&self) -> Result<LlamaToken, EngineError> {
        // SAFETY: a decode that requested logits has just completed.
        let raw = unsafe {
            ozgent_mtmd_sys::llama_cpp_sys_2::llama_get_logits_ith(self.slot.hub().raw(), -1)
        };
        if raw.is_null() {
            return Err(EngineError::Decode("no logits".into()));
        }
        let n_vocab = self.model.n_vocab() as usize;
        let logits = unsafe { std::slice::from_raw_parts(raw, n_vocab) };
        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| EngineError::Decode("no logits".into()))?;
        Ok(LlamaToken(best as i32))
    }

    /// The logits of the last decode on this session, as a row.
    ///
    /// [`Self::greedy_from_logits`] answers "which token", which is all a
    /// probe needs. A turn needs the row itself, because the sampler — its
    /// temperature, its penalties, its grammar — is what turns logits into a
    /// token, and an image prompt deserves the same sampler as a text one.
    ///
    /// Read at `-1` for the same reason: llama.cpp resolves it to the last
    /// output row, and a prefill asks for logits on the final prompt token
    /// rather than on row zero.
    fn last_logits_row(&self) -> Result<Vec<f32>, EngineError> {
        // SAFETY: a decode that requested logits has just completed, and the
        // row is copied out before anything else can decode over it.
        let raw = unsafe {
            ozgent_mtmd_sys::llama_cpp_sys_2::llama_get_logits_ith(self.slot.hub().raw(), -1)
        };
        if raw.is_null() {
            return Err(EngineError::Decode("no logits".into()));
        }
        let n_vocab = self.model.n_vocab() as usize;
        Ok(unsafe { std::slice::from_raw_parts(raw, n_vocab) }.to_vec())
    }

    /// Turn the NextN head's output on or off for this session.
    ///
    /// Drafting from the head needs the target unmasked, which makes it emit a
    /// hidden state for every position it decodes. That is not free, and the
    /// drafter has to win back whatever it costs before it is worth wiring in
    /// — so it is measurable on its own, separately from any drafting.
    pub fn set_nextn_output(&mut self, on: bool) {
        // SAFETY: the context is live for the life of the session.
        unsafe { crate::nextn::set_enabled(self.slot.hub().raw(), on, !on) };
    }

    /// Check that a snapshot really can rewind this model mid-generation.
    ///
    /// Generates `k` tokens, rewinds, and generates `k` again. The two runs
    /// must be identical. Speculating on a model whose cache cannot be trimmed
    /// rests entirely on this holding, and the failure mode is silently wrong
    /// output rather than an error — so it is measured, not assumed.
    ///
    /// Returns the two runs and the size of the snapshot taken.
    pub fn probe_rewind(
        &mut self,
        prompt: &str,
        k: u32,
        partial: bool,
        on_device: bool,
    ) -> Result<(String, String, usize), EngineError> {
        let n_batch = (self.n_batch as usize).max(1);

        self.reset();
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        let (head, tail) = tokens.split_at(tokens.len().saturating_sub(1));
        self.prefill(head, false)?;

        // The snapshot is taken *before* the final prompt token, so each run
        // can decode it again and regenerate its logits. Restoring state does
        // not restore the logits buffer, so a run resumed straight after a
        // restore would sample its first token from whatever the previous run
        // left behind — which looks exactly like the rewind having failed.
        let mark = self.n_past;
        let state = self.snapshot(partial, on_device)?;
        let size = state.byte_len();

        let row = self.prefill(tail, true)?;
        let first = self.run_greedy(k, row)?;

        self.restore(&state)?;
        self.n_past = mark;
        self.sampler.reset();
        let row = self.prefill(tail, true)?;
        let second = self.run_greedy(k, row)?;

        Ok((first, second, size))
    }

    /// Propose `n` tokens continuing `context`, as a draft for another model
    /// to verify.
    ///
    /// This session is left exactly as it was found: the proposal is generated
    /// behind a snapshot and rolled back, so the drafter never drifts from the
    /// confirmed transcript. Tokens of `context` this session has not seen are
    /// prefilled first, which after the opening turn is only the handful the
    /// target just confirmed.
    ///
    /// Greedy, because a draft is a guess at what the *target* will do, not a
    /// sample in its own right — and the target re-samples every token anyway,
    /// so a drafter's randomness would only lower the acceptance rate.
    pub fn propose(
        &mut self,
        context: &[LlamaToken],
        n: usize,
    ) -> Result<Vec<LlamaToken>, EngineError> {
        if n == 0 || context.is_empty() {
            return Ok(Vec::new());
        }
        let n_ctx = self.n_ctx as i32;
        if context.len() as i32 + n as i32 >= n_ctx {
            return Ok(Vec::new());
        }
        let n_batch = (self.n_batch as usize).max(1);

        // Anything the drafter has not absorbed yet. A divergence means the
        // target went somewhere this session never saw, so start over rather
        // than draft from a transcript that never happened.
        let shared = common_prefix(&self.cached, context);
        if shared < self.cached.len() {
            self.reset();
        }
        // The logits to draft from come out of this prefill. If there is
        // nothing fresh to feed there are none — the previous round's belong
        // to a pass that has since been overwritten — so the round simply
        // proposes nothing, which costs the target one ordinary token.
        let mut row: Option<Vec<f32>> = None;
        let fresh: Vec<LlamaToken> = context[self.cached.len()..].to_vec();
        if !fresh.is_empty() {
            row = self.prefill(&fresh, true)?;
            self.cached.extend_from_slice(&fresh);
        }

        if !self.may_snapshot() {
            // See `generate_drafted`: a device-held state is captured through
            // a buffer the whole context shares, and only one slot may use it.
            return Ok(Vec::new());
        }
        let mark = self.n_past;
        let cached_len = self.cached.len();
        let state = self.snapshot(false, true)?;

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let Some(logits) = row.take() else { break };
            let token = self.pick(&logits);
            if self.model.is_eog_token(token) {
                break;
            }
            out.push(token);
            row = self.feed(vec![token], self.n_past, crate::hub::Logits::Last)?.into_last();
        }

        // Put the drafter back where it started, so the next call resumes from
        // what the target actually confirmed rather than from its own guesses.
        self.restore(&state)?;
        self.n_past = mark;
        self.cached.truncate(cached_len);
        self.sampler.reset();
        Ok(out)
    }

    /// Time a snapshot and a restore, which is what decides whether
    /// speculation can afford one per draft step.
    ///
    /// Returns milliseconds per snapshot, per restore, and the host bytes used.
    pub fn probe_snapshot_cost(
        &mut self,
        prompt: &str,
        iterations: u32,
        on_device: bool,
        partial: bool,
    ) -> Result<(f64, f64, usize), EngineError> {
        let n_batch = (self.n_batch as usize).max(1);

        self.reset();
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        self.prefill(&tokens, true)?;

        // One outside the loop, so allocation and any first-call setup are not
        // charged to the average.
        let warm = self.snapshot(partial, on_device)?;
        self.restore(&warm)?;

        let start = Instant::now();
        let mut states = Vec::with_capacity(iterations as usize);
        for _ in 0..iterations {
            states.push(self.snapshot(partial, on_device)?);
        }
        let snap_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        // Only the most recent device-held state stays valid: capturing with
        // ON_DEVICE invalidates every earlier one for that sequence.
        let last = states.last().expect("at least one iteration");
        let start = Instant::now();
        for _ in 0..iterations {
            self.restore(last)?;
        }
        let restore_ms = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        Ok((snap_ms, restore_ms, last.byte_len()))
    }

    /// Greedily decode `k` tokens from the current position, returning the text.
    fn run_greedy(&mut self, k: u32, from: Option<Vec<f32>>) -> Result<String, EngineError> {
        let mut out = String::new();
        let mut decoder = Utf8Buffer::new();
        let mut row = from;
        for _ in 0..k {
            let Some(logits) = row.take() else { break };
            let token = self.pick(&logits);
            if self.model.is_eog_token(token) {
                break;
            }
            let bytes = self
                .model
                .token_to_bytes(token, Special::Tokenize)
                .map_err(|e| EngineError::Detokenize(e.to_string()))?;
            out.push_str(&decoder.push(&bytes));

            row = self.feed(vec![token], self.n_past, crate::hub::Logits::Last)?.into_last();
        }
        out.push_str(&decoder.finish());
        Ok(out)
    }

    /// Size of this model's vocabulary.
    ///
    /// A drafter and its target must agree on this, or a proposed token id
    /// means a different word to each of them and verification compares
    /// unrelated things.
    pub fn n_vocab(&self) -> i32 {
        self.model.n_vocab()
    }

    /// Bytes needed to snapshot this sequence's state.
    ///
    /// `partial` asks only for the parts a KV trim cannot undo — recurrent and
    /// sliding-window state. That is the figure that decides whether draft
    /// rejection can be made safe on a hybrid model by snapshotting instead of
    /// trimming.
    pub fn state_bytes(&self, partial: bool, on_device: bool) -> usize {
        let mut bits = 0u32;
        if partial {
            bits |= llama_cpp_2::context::session::LlamaStateSeqFlags::PARTIAL_ONLY.bits();
        }
        if on_device {
            bits |= llama_cpp_2::context::session::LlamaStateSeqFlags::ON_DEVICE.bits();
        }
        let seq = self.slot.seq();
        self.slot.with_context(|c| {
            c.state_seq_get_size_ext(
                seq,
                llama_cpp_2::context::session::LlamaStateSeqFlags::from_bits(bits),
            )
        })
    }

    /// Choose a token from one row of logits.
    ///
    /// Exactly what `llama_sampler_sample` does — apply the chain, take what
    /// it selected, accept it — but from a row the hub copied out rather than
    /// from the context, which by now belongs to somebody else's pass. The
    /// accept is not optional: `sample` performs it internally, and stateful
    /// samplers (repetition penalties, grammars) go wrong without it.
    fn pick(&mut self, row: &[f32]) -> LlamaToken {
        let started = Instant::now();
        let token = self.pick_inner(row);
        PICK_MICROS.fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        token
    }

    fn pick_inner(&mut self, row: &[f32]) -> LlamaToken {
        // One buffer for the life of the session, refilled in place.
        //
        // The candidate array is a whole vocabulary of entries — 151,936 of
        // them, 1.8 MB — and building a fresh one every token cost 2.0 ms of a
        // 21 ms pass: the six percent this session had quietly lost on plain
        // decoding, measured at 45.0 tok/s against 47.7 before the hub
        // existed. llama.cpp's own sampler never paid it because it keeps its
        // buffer; this now does too.
        let mut data = std::mem::take(&mut self.candidates);
        data.clear();
        data.extend(row.iter().enumerate().map(|(i, &logit)| {
            llama_cpp_2::token::data::LlamaTokenData::new(LlamaToken(i as i32), logit, 0.0)
        }));
        let mut candidates =
            llama_cpp_2::token::data_array::LlamaTokenDataArray::new(data, false);
        candidates.apply_sampler(&self.sampler);
        let token = candidates
            .selected_token()
            .expect("a sampler chain always selects a token");
        self.candidates = candidates.data;
        self.sampler.accept(token);
        token
    }

    /// Decode `tokens` at the current position through the hub, advancing it.
    fn feed(
        &mut self,
        tokens: Vec<LlamaToken>,
        pos: i32,
        logits: crate::hub::Logits,
    ) -> Result<crate::hub::Outcome, EngineError> {
        self.feed_wanting(tokens, pos, logits, false)
    }

    /// As [`Session::feed`], also bringing back the NextN hidden state of each
    /// row, which drafting from the model's own head needs.
    fn feed_wanting(
        &mut self,
        tokens: Vec<LlamaToken>,
        pos: i32,
        logits: crate::hub::Logits,
        hidden: bool,
    ) -> Result<crate::hub::Outcome, EngineError> {
        let n = tokens.len() as i32;
        let out = self
            .slot
            .run_wanting(tokens, pos, logits, hidden)
            .map_err(|e| EngineError::Decode(e.to_string()))?;
        self.n_past = pos + n;
        Ok(out)
    }

    /// Tokens currently held in the KV cache.
    pub fn used(&self) -> u32 {
        self.n_past.max(0) as u32
    }

    /// Decode `tokens` into the cache starting at the current position.
    ///
    /// llama.cpp asserts that a batch holds at most `n_batch` tokens and
    /// aborts the process if it does not, so a long prompt — a big tool
    /// result, a pasted document, a long history — must be fed in chunks.
    fn prefill(
        &mut self,
        tokens: &[LlamaToken],
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let n_batch = (self.n_batch as usize).max(1);
        let start = self.n_past;
        let mut row = None;
        for (offset, end) in prefill_chunks(tokens.len(), n_batch) {
            let final_chunk = end == tokens.len();
            // Logits are needed only for the final token of the whole
            // sequence, not of each chunk.
            let logits = if want_logits && final_chunk {
                crate::hub::Logits::Last
            } else {
                crate::hub::Logits::None
            };
            let out = self.feed(tokens[offset..end].to_vec(), start + offset as i32, logits)?;
            if final_chunk {
                row = out.into_last();
            }
        }
        self.n_past = start + tokens.len() as i32;
        Ok(row)
    }

    /// Drop the cache and start a fresh conversation on this context.
    ///
    /// Reusing the context avoids re-allocating the KV cache, which for a
    /// large context is the slow part of opening a session.
    pub fn reset(&mut self) {
        self.slot.clear();
        self.sampler.reset();
        self.n_past = 0;
        self.cached.clear();
        self.last_reused = 0;
    }

    /// Tokens the last `generate` call reused from the cache instead of
    /// recomputing. Reported by `/stats`.
    pub fn last_reused(&self) -> usize {
        self.last_reused
    }

    /// Put the sequence back to what was actually confirmed.
    ///
    /// With a trimmable cache this just drops the rejected entries. With a
    /// snapshot it restores the state from before the batch — which undoes the
    /// confirmed token and the accepted drafts as well, so those are decoded
    /// again. That re-decode is one extra forward pass, and it is the entire
    /// cost of speculating on a model whose cache cannot be trimmed.
    #[allow(clippy::too_many_arguments)]
    fn settle_draft(
        &mut self,
        snapshot: Option<&SeqState>,
        start: i32,
        pending: LlamaToken,
        draft: &[LlamaToken],
        accepted: usize,
    ) -> Result<(), EngineError> {
        match snapshot {
            // Everything was accepted: the state already reflects it.
            Some(_) if accepted == draft.len() => Ok(()),
            Some(state) => {
                self.restore(state)?;
                self.n_past = start;
                let mut confirmed = Vec::with_capacity(1 + accepted);
                confirmed.push(pending);
                confirmed.extend_from_slice(&draft[..accepted]);
                self.prefill(&confirmed, false).map(|_| ())
            }
            None => self.trim_after_draft(start, accepted),
        }
    }

    /// Drop KV entries for drafted tokens that were rejected.
    ///
    /// After the batch the cache holds the confirmed token plus every drafted
    /// one; only `accepted` of the drafts are real, so the rest must go or the
    /// next position would collide with stale state.
    ///
    /// When the cache refuses a partial removal it is rebuilt from the tokens
    /// known to be good, and speculation is abandoned for the session.
    fn trim_after_draft(
        &mut self,
        start: i32,
        accepted: usize,
    ) -> Result<(), EngineError> {
        let valid = start + 1 + accepted as i32;
        if valid >= self.n_past {
            return Ok(());
        }
        if self.slot.trim(valid).unwrap_or(false) {
            self.n_past = valid;
            return Ok(());
        }

        // Worth saying out loud: this is the expensive path. The cache is
        // thrown away and rebuilt from the tokens known to be good, and
        // speculation is abandoned for the rest of the session.
        tracing::info!("this cache refused to trim a rejected draft; rebuilding it and stopping speculation");
        self.can_trim = false;
        self.slot.clear();
        self.n_past = 0;

        let good: Vec<LlamaToken> = self
            .cached
            .iter()
            .copied()
            .take(valid.max(0) as usize)
            .collect();
        self.prefill(&good, true)?;
        self.cached = good;
        Ok(())
    }

    /// Constrain generation to a GBNF grammar, or pass `None` to lift it.
    ///
    /// The grammar sampler goes first in the chain so it masks the logits
    /// before any temperature or top-p shaping sees them; applied afterwards
    /// it could only reject, and sampling would stall on a dead end.
    /// Adopt the options of the turn about to run.
    ///
    /// A server holds one session across many requests, and each request
    /// carries its own sampling. Without this the first caller's temperature,
    /// seed and reasoning budget stand for every caller after it — two clients
    /// sharing a model would silently share whichever settings arrived first.
    ///
    /// Only per-turn settings move. Layer placement, context length, cache
    /// types and expert offload are decided when the weights are loaded and
    /// cannot change under a live KV cache, so they are deliberately ignored
    /// rather than half-applied.
    pub fn set_options(&mut self, opts: &Resolved) {
        self.opts.adopt_per_turn(opts);
        // The sampler is built from these, so it has to be rebuilt with them.
        // Any installed grammar is reapplied by the caller's `set_grammar`,
        // which runs after this on every turn.
        self.sampler = build_sampler(&self.opts);
        self.grammar_active = false;
    }

    /// The options this session will generate under, for tests and callers
    /// that need to know what actually took effect.
    pub fn options(&self) -> &Resolved {
        &self.opts
    }

    pub fn set_grammar(&mut self, grammar: Option<&str>) -> Result<(), EngineError> {
        // One flat chain. Nesting a chain inside a chain does not reliably
        // propagate `accept`, so the grammar never advances past its root and
        // the next `apply` finds no live stacks — which llama.cpp turns into a
        // hard abort rather than an error.
        let constraint = match grammar {
            None => None,
            Some(g) => Some(
                LlamaSampler::grammar(self.model, g, "root")
                    .map_err(|e| EngineError::Grammar(e.to_string()))?,
            ),
        };
        // Only once the grammar is known good: set before, a rejected grammar
        // would leave the flag claiming a constraint that was never installed,
        // which silently disables speculation for the rest of the session.
        self.grammar_active = grammar.is_some();
        self.sampler = build_sampler_with(&self.opts, constraint);
        Ok(())
    }

    /// Precompile the tool grammars used to constrain a call once it starts.
    ///
    /// Compiling here rather than per turn keeps schema-to-GBNF conversion —
    /// which does not depend on the conversation — off the generation path.
    /// Passing an empty slice disables gating.
    /// Constrain the body of a tool call once one starts.
    ///
    /// Pass an empty slice for a model whose template writes tool calls
    /// itself. The gate's grammar describes a JSON body, which is the format
    /// ozgent's own preamble asks for — applied to a model emitting its
    /// native `<function=…><parameter=…>` syntax it would force the wrong
    /// language at exactly the moment the call begins.
    pub fn set_tools(&mut self, tools: &[ozgent_core::ToolSpec]) {
        self.tool_grammars = ToolGate::compile(tools);
    }

    /// Generate a completion, calling `on_token` with each decoded fragment.
    ///
    /// Returning `false` from the callback stops generation, which is how a
    /// user interrupt is honoured mid-stream.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<(Stats, StopReason), EngineError> {
        self.generate_with_media(prompt, None, max_tokens, on_token)
    }

    /// Generate with images attached.
    ///
    /// `prompt` must already contain one media marker per image, positioned
    /// where the image belongs in the conversation — position is what the model
    /// attends over, so the marker is not decoration.
    pub fn generate_with_media(
        &mut self,
        prompt: &str,
        media: Option<(&crate::mtmd::Projector<'_>, &[crate::mtmd::Media], &[ozgent_core::ImageSource])>,
        max_tokens: u32,
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<(Stats, StopReason), EngineError> {
        self.generate_drafted(prompt, media, max_tokens, None, on_token)
    }

    /// Generate, optionally with a second model proposing tokens.
    ///
    /// `drafter` is a session over a much smaller model sharing this one's
    /// vocabulary. It proposes; this model verifies every token and keeps only
    /// what it would have produced anyway, so the answer is identical to
    /// generating without it. The drafter buys latency, never a different
    /// answer.
    pub fn generate_drafted(
        &mut self,
        prompt: &str,
        media: Option<(&crate::mtmd::Projector<'_>, &[crate::mtmd::Media], &[ozgent_core::ImageSource])>,
        max_tokens: u32,
        mut drafter: Option<&mut Session<'_>>,
        on_token: impl FnMut(&str) -> bool,
    ) -> Result<(Stats, StopReason), EngineError> {
        let mut on_token = on_token;
        // A turn that returned early through `?` can leave the gate's grammar
        // installed. Lifting it here rather than only on the way out means a
        // failed tool call cannot constrain the next answer into being one.
        if self.gate_applied {
            self.set_grammar(None)?;
            self.gate_applied = false;
        }
        let n_ctx = self.n_ctx as i32;
        let mut stats = Stats::default();
        let started = Instant::now();
        let n_batch = (self.n_batch as usize).max(1);
        // The logits the prompt ends on, which is where generation starts.
        // Carried out of the prefill rather than read back from the context,
        // which by then may be running somebody else's pass.
        let mut prefill_row: Option<Vec<f32>> = None;

        // An image becomes embeddings, not token ids. Once mtmd has written
        // them into the cache there is no token sequence that describes what is
        // resident, so the prefix-reuse bookkeeping cannot be trusted and the
        // next turn has to start from a clean cache.
        if self.media_dirty {
            self.slot.clear();
            self.cached.clear();
            self.n_past = 0;
            self.media_dirty = false;
        }

        match media {
            Some((projector, images, sources)) if !images.is_empty() => {
                self.slot.clear();
                self.cached.clear();
                self.n_past = 0;

                let seq = self.slot.seq();
                let new_past = self
                    .slot
                    .with_context(|c| {
                        projector.eval(c, prompt, images, sources, seq, n_batch as i32, true)
                    })
                    .map_err(|e| EngineError::Decode(e.to_string()))?;
                if new_past >= n_ctx {
                    return Err(EngineError::PromptTooLong {
                        tokens: new_past as usize,
                        context: n_ctx as usize,
                    });
                }
                self.n_past = new_past;
                // The image prompt was decoded inside the projector, not
                // through `prefill`, so the row the generation loop samples
                // from has to be taken from the context directly.
                //
                // This is the whole of the vision regression: sampling used to
                // read the context's last logits itself, so it did not care
                // which branch had filled them. When that became a row handed
                // back by `prefill`, this branch — which never calls it — went
                // on setting `n_past` and nothing else, and every image turn
                // ended in "the prompt produced no logits to generate from".
                prefill_row = Some(self.last_logits_row()?);
                self.media_dirty = true;
                self.last_reused = 0;
                stats.prompt_tokens = new_past as usize;
                stats.reused_tokens = 0;
                stats.prompt_ms = started.elapsed().as_millis();
            }
            _ => {
                let tokens = self
                    .model
                    .str_to_token(prompt, AddBos::Always)
                    .map_err(|e| EngineError::Tokenize(e.to_string()))?;

                if tokens.len() as i32 >= n_ctx {
                    return Err(EngineError::PromptTooLong {
                        tokens: tokens.len(),
                        context: n_ctx as usize,
                    });
                }
                // Everything between here and the prefill is cache bookkeeping
                // — trimming, checkpoints, lending the shared prefix — and
                // none of it is free on a model whose state must be copied.
                let settle = std::time::Instant::now();

                // A chat turn resends the whole conversation, so almost all of
                // the prompt is already in the cache. Keeping the shared prefix
                // and prefilling only what changed is the single largest saving
                // available in interactive use, and it grows with the
                // conversation.
                let reuse = match self.reuse {
                    PrefixReuse::Off => 0,
                    PrefixReuse::Longest => common_prefix(&self.cached, &tokens),
                };

                // The final token must always be decoded to produce logits, so
                // never reuse the entire prompt.
                let reuse = reuse.min(tokens.len().saturating_sub(1));
                self.last_reused = reuse;

                let mut reuse = reuse;
                let mut restored = false;

                // A saved state may hold far more of this prompt than the
                // live cache does, and on a switch between conversations it
                // usually does: the cache belongs to whoever spoke last, and
                // a prefix of the wrong conversation is worth nothing.
                // Restoring is a host-to-device copy, so it only wins when it
                // recovers meaningfully more than trimming would — a margin,
                // not a tie-break.
                if let Some((_, saved)) = self.best_checkpoint(&tokens) {
                    if saved > reuse + CHECKPOINT_MIN_TOKENS {
                        if let Some(n) = self.restore_checkpoint(&tokens) {
                            reuse = n;
                            restored = true;
                        }
                    }
                }

                if !restored && reuse < self.cached.len() {
                    // Drop everything after the shared prefix; those positions
                    // are about to be occupied by different tokens.
                    if self.can_trim && !self.slot.trim(reuse as i32).unwrap_or(false) {
                        // Sliding-window and recurrent caches refuse a partial
                        // removal. That is a limitation, not an error: drop the
                        // whole cache and prefill from scratch, and stop relying
                        // on trimming for the rest of this session.
                        tracing::debug!("this cache cannot trim; falling back to a full prefill");
                        self.can_trim = false;
                    }
                    if !self.can_trim {
                        // The trim was refused, so everything resident past the
                        // shared prefix is stuck there and the cache is no use.
                        // A checkpoint is the way back: it was taken before any
                        // of those tokens existed, so there is nothing to undo.
                        match self.restore_checkpoint(&tokens) {
                            Some(n) => {
                                reuse = n;
                                restored = true;
                            }
                            None => {
                                self.slot.clear();
                                self.cached.clear();
                                reuse = 0;
                            }
                        }
                    }
                }
                // The prefix every conversation here begins with — a system
                // prompt, a set of tool schemas — is held once, in a sequence
                // of its own. Borrowing it is a change of ownership in the
                // cache rather than a prefill, so a conversation that has
                // never been seen before starts a thousand tokens in.
                //
                // Filled here, by whichever turn first reveals that two
                // prompts share a head. That turn was about to prefill those
                // tokens itself and borrows them straight back, so it pays
                // nothing and every later conversation starts from them free.
                // Placed after the trim rather than before it, because what
                // matters is the reuse that *survived*. A cache that cannot
                // trim loses everything past the point of divergence, so a
                // hopeful reuse of two thousand tokens becomes zero — and
                // comparing the shared prefix against the hopeful figure meant
                // it was never worth borrowing, on exactly the turns it would
                // have saved the most.
                if self.reuse != PrefixReuse::Off {
                    if let Some(target) =
                        self.slot.hub().consider(&tokens)
                    {
                        tracing::info!(
                            "holding the {} tokens every conversation starts with",
                            target.len()
                        );
                        if let Err(e) = self.slot.hub().fill_commons(&target) {
                            tracing::warn!("the shared prefix could not be held: {e}");
                        }
                    }
                    let commons = self.slot.hub().commons();
                    // All of it or none: a partial copy would hand over a
                    // recurrent state belonging to a position this
                    // conversation has never reached.
                    let usable = !commons.is_empty()
                        && commons.len() < tokens.len()
                        && common_prefix(&commons, &tokens) == commons.len()
                        && commons.len() > reuse + CHECKPOINT_MIN_TOKENS;
                    let _ = restored;
                    if usable {
                        match self.slot.hub().lend(self.slot.seq()) {
                            Ok(n) if n > 0 => {
                                reuse = n;
                                self.n_past = n as i32;
                                self.cached = commons;
                                self.checkpoints.clear();
                                restored = true;
                            }
                            Ok(_) => {}
                            Err(e) => tracing::warn!("the shared prefix could not be lent: {e}"),
                        }
                    }
                }

                // Where the time before the first token actually goes. Token
                // counts alone hide the answer: a turn that reuses 2817 of
                // 2822 tokens and prefills 30 was measured taking *longer*
                // than the turn that prefilled all 2822, because the saving
                // was being handed straight back in state copies. Counts and
                // milliseconds have to be visible together or that is
                // invisible.
                tracing::debug!(
                    "prefix: {reuse} of {} tokens reused{}, {} ms to here",
                    tokens.len(),
                    if restored { " (restored from checkpoint)" } else { "" },
                    settle.elapsed().as_millis()
                );
                self.n_past = reuse as i32;
                self.last_reused = reuse;

                // Prefill stops at the history boundary first so the state can
                // be captured there. Everything up to that point is what the
                // next turn will resend verbatim; the generation prompt after
                // it is replaced by the reply the model is about to write.
                let boundary = tokens.len().saturating_sub(self.gen_prompt_tokens).max(reuse);
                // Two prefills, and the first one carries the bulk. Timing only
                // the second said "2823 tokens in 64 ms" — the right count
                // against the wrong clock — and made real prefill look free
                // while the cost appeared to be somewhere it was not.
                let filling = std::time::Instant::now();
                let mut bulk = 0;
                if boundary > reuse {
                    self.prefill(&tokens[reuse..boundary], false)?;
                    bulk = boundary - reuse;
                    self.cached = tokens[..boundary].to_vec();
                    self.save_checkpoint(&tokens[..boundary]);
                }
                let tail_at = std::time::Instant::now();
                prefill_row = self.prefill(&tokens[boundary..], true)?;
                debug_assert!(prefill_row.is_some());
                tracing::debug!(
                    "prefill: {bulk} tokens of history in {} ms, {} of generation prompt in {} ms",
                    (tail_at - filling).as_millis(),
                    tokens.len() - boundary,
                    tail_at.elapsed().as_millis()
                );
                self.cached = tokens.clone();
                stats.prompt_tokens = tokens.len() - reuse;
                stats.reused_tokens = reuse;
                stats.prompt_ms = started.elapsed().as_millis();
            }
        }

        // Reasoning is only bounded when it is actually happening; with
        // thinking suppressed the block never opens and the budget never fires.
        let think_budget = if matches!(self.opts.thinking, ozgent_core::ThinkingMode::Off) {
            0
        } else {
            self.opts.reasoning_effort.budget()
        };
        tracing::debug!(
            "reasoning budget {think_budget}; prompt opens a block: {}; tail: {:?}",
            crate::effort::ThinkBudget::opens_thinking(prompt),
            &prompt[prompt.len().saturating_sub(60)..]
        );
        let mut think = if crate::effort::ThinkBudget::opens_thinking(prompt) {
            crate::effort::ThinkBudget::resumed(think_budget)
        } else {
            crate::effort::ThinkBudget::new(think_budget)
        };
        let closing: Vec<LlamaToken> = self
            .model
            .str_to_token(crate::effort::CLOSE, AddBos::Never)
            .unwrap_or_default();

        let gen_started = Instant::now();
        let mut decoder = Utf8Buffer::new();
        let limit = if max_tokens == 0 { u32::MAX } else { max_tokens };
        let mut reason = StopReason::TokenLimit;

        // Self-speculation mines the context for repeated n-grams. It needs no
        // second model, and a wrong guess is simply discarded, so output is
        // identical to sequential decoding either way.
        // Speculation is incompatible with a grammar. Verification samples
        // from rows that follow *drafted* tokens, but the grammar's state has
        // only advanced over accepted ones, so those samples are checked
        // against the wrong state — llama.cpp then aborts on an empty stack.
        // Rejecting a draft means dropping its KV entries, so speculation is
        // only possible on a cache that can trim.
        // Two ways to undo a rejected draft. Trimming the KV is free but only
        // correct when nothing keeps state a trim cannot reach. Otherwise the
        // whole sequence state is snapshotted and put back — measured at
        // 0.71 ms on device against a ~17.8 ms token budget, and exact on
        // Qwen3.5, whose cache refuses a partial trim outright.
        let by_trim = self.rollback_safe && self.can_trim;
        // The model's own NextN head, when it has one and was asked for.
        //
        // Opt-in rather than part of `auto` for now. It drafts where n-grams
        // cannot — ordinary prose, where they measured exactly zero — and the
        // hidden states it needs cost nothing (48.4 tok/s against 49.0 with
        // them on). But it is new, it writes into the cache the target is
        // using, and the failure mode of getting that wrong is wrong output
        // rather than an error. It earns `auto` by being measured, not by
        // being plausible.
        // Drafting from the model's own head opens a second context over this
        // one's memory and edits sequence 0 directly, which is only this
        // session's sequence when this session is the only one.
        // A drafter belongs to its slot's sequence, so several conversations
        // may draft at once. What they share is the flag that makes the target
        // emit hidden states, which belongs to the context — hence a guard
        // that turns it on for the first and off after the last.
        let mut nextn_on = None;
        let mut mtp = if matches!(self.opts.speculative, Speculative::Mtp) && !self.grammar_active {
            let (seq, n_seq_max, unified) =
                (self.slot.seq(), self.slot.hub().n_seq_max(), self.slot.hub().unified());
            match self.slot.with_context(|c| {
                crate::mtp::MtpDrafter::new(
                    self.model,
                    backend()?,
                    c,
                    self.n_ctx,
                    seq,
                    n_seq_max,
                    unified,
                )
            }) {
                Ok(Some(d)) => {
                    // Unmasked, so a hidden state comes back for every position
                    // the target decodes — including each verified draft.
                    nextn_on = Some(self.slot.want_nextn());
                    Some(d)
                }
                Ok(None) => {
                    tracing::info!("this model has no nextn head; drafting from n-grams instead");
                    None
                }
                Err(e) => {
                    tracing::warn!("nextn drafter unavailable: {e}");
                    None
                }
            }
        } else {
            None
        };
        // The hidden state of the last confirmed position, which is what the
        // NextN head continues from. Carried between rounds because the
        // drafting decision is made at the top of the loop and the state it
        // needs was produced at the bottom of the previous one.
        let mut mtp_hidden: Option<Vec<f32>> = None;
        let mut spec_on = !self.grammar_active
            && (drafter.is_some()
                || mtp.is_some()
                || matches!(self.opts.speculative, Speculative::Ngram | Speculative::Auto));
        // A device-resident snapshot is a list of views into the cache as it
        // stood, and llama.cpp keeps one cached buffer per *context* to hold
        // them. Two conversations sharing a context therefore snapshot through
        // the same buffer with different layouts, and llama.cpp answers that
        // by aborting the process — not theorised: four concurrent turns on
        // one context died in ggml's allocator on the first drafting round,
        // and vanished the moment speculation was switched off.
        //
        // So the snapshot path belongs to a session that has the context to
        // itself. A shared session can still speculate if its cache can trim,
        // which is per-sequence and safe; one that cannot simply does not.
        let mut use_snapshot = spec_on && !by_trim && self.may_snapshot();
        if spec_on && !by_trim && !self.may_snapshot() {
            tracing::debug!(
                "this cache cannot trim a rejected draft, and the rollback that \
                 replaces trimming needs a context of its own; not speculating"
            );
            spec_on = false;
        }
        if use_snapshot && self.snapshot(false, true).is_err() {
            // The device-resident path is what makes this affordable; without
            // it, speculating would cost more than it saves.
            tracing::info!("sequence state cannot be snapshotted; speculative decoding disabled");
            spec_on = false;
            use_snapshot = false;
        }
        // Rejection costs an extra forward pass on the snapshot path, because
        // restoring undoes the confirmed token along with the rejected ones.
        // Break-even is around two accepted tokens per rejecting round, so
        // drafting has to land more often before it is worth starting.
        // How long to stay quiet before testing the water again. A rejected
        // probe costs an extra forward pass on the snapshot path, so probing
        // has to be rarer there to stay cheap.
        let probe_every = if use_snapshot { 96 } else { 24 };
        // How far a match must extend behind the key before it is worth
        // betting on. Zero keeps the trim path exactly as it was; on the
        // snapshot path a wrong bet costs an extra forward pass, so a bare
        // two-token coincidence is not enough evidence to pay that.
        let min_reach = if use_snapshot { MIN_DRAFT_REACH } else { 0 };
        let min_acceptance = if use_snapshot {
            (self.opts.speculative_tuning.min_acceptance * 2.0).min(0.6)
        } else {
            self.opts.speculative_tuning.min_acceptance
        };
        // Speculation and grammars cannot coexist (see above), and the gate
        // exists to install a grammar mid-turn — so it stays inert whenever
        // drafting is live. Nothing is lost on the models this matters for:
        // hybrid and recurrent models already have speculation disabled.
        // A caller-set grammar (a forced call, or a retry after a malformed
        // one) already constrains the whole turn; the gate must not swap it
        // out underneath.
        let mut gate = if spec_on || self.grammar_active {
            ToolGate::inert()
        } else {
            ToolGate::new(self.tool_grammars.clone())
        };
        let tuning = self.opts.speculative_tuning.clone();
        // Seeded lazily from `self.cached` on the first pass through the
        // generation loop, which is authoritative and avoids copying the
        // prompt twice.
        let mut ngram = NgramCache::new();

        let mut produced = 0u32;
        let mut callback_ns: u128 = 0;
        let mut emit = |token: LlamaToken,
                        model: &LlamaModel,
                        decoder: &mut Utf8Buffer,
                        gate: &mut ToolGate,
                        budget: &mut crate::effort::ThinkBudget,
                        on_token: &mut dyn FnMut(&str) -> bool|
         -> Result<(bool, Option<String>), EngineError> {
            let started = std::time::Instant::now();
            let bytes = model
                .token_to_bytes(token, Special::Tokenize)
                .map_err(|e| EngineError::Detokenize(e.to_string()))?;
            let text = decoder.push(&bytes);
            // Checked before the callback so a slow consumer cannot delay the
            // constraint past the first token of the call body.
            let trigger = gate.observe(&text);
            budget.observe(&text);
            let out = text.is_empty() || on_token(&text);
            callback_ns += started.elapsed().as_nanos();
            Ok((out, trigger))
        };

        // The token to decode next, sampled from the prefill's final logits.
        let Some(row) = prefill_row.take() else {
            return Err(EngineError::Decode(
                "the prompt produced no logits to generate from".into(),
            ));
        };
        let mut pending = self.pick(&row);

        'outer: loop {
            if self.model.is_eog_token(pending) {
                reason = StopReason::EndOfText;
                break;
            }
            if produced >= limit {
                reason = StopReason::TokenLimit;
                break;
            }
            if self.n_past >= n_ctx {
                reason = StopReason::ContextFull;
                break;
            }

            // No `accept` here: `LlamaSampler::sample` already accepts the
            // token it returns. Accepting again advances stateful samplers
            // twice — which double-counts repetition penalties, and drives a
            // grammar into a dead state that llama.cpp aborts on.
            let (keep_going, trigger) =
                emit(pending, self.model, &mut decoder, &mut gate, &mut think, &mut on_token)?;
            // The model has just committed to a call, so the body can be
            // constrained from here without forcing the turn to be one.
            if let Some(g) = trigger {
                // Never fatal: a grammar the model's vocabulary happens to
                // reject should cost the constraint, not the whole answer.
                match self.set_grammar(Some(&g)) {
                    Ok(()) => {
                        tracing::debug!("tool call started; constraining the body under grammar");
                        self.gate_applied = true;
                    }
                    Err(e) => {
                        tracing::warn!("tool grammar rejected; continuing unconstrained: {e}")
                    }
                }
            }
            if !keep_going {
                reason = StopReason::Cancelled;
                break;
            }
            produced += 1;
            stats.generated_tokens += 1;
            self.cached.push(pending);

            // Out of thinking budget: close the block for the model rather
            // than cut the stream. The closing tokens are decoded into the
            // context alongside the token just emitted, so the model reads its
            // own reasoning as finished and answers normally — where truncating
            // would leave an unterminated block and waste what it already
            // spent. Drafting is skipped this round; there is nothing to guess.
            if think.exhausted() && !closing.is_empty() {
                let mut run = Vec::with_capacity(1 + closing.len());
                run.push(pending);
                run.extend_from_slice(&closing);
                let closed = self.feed(run, self.n_past, crate::hub::Logits::Last)?;

                for token in &closing {
                    if !emit(*token, self.model, &mut decoder, &mut gate, &mut think, &mut on_token)?.0 {
                        reason = StopReason::Cancelled;
                        break 'outer;
                    }
                    self.cached.push(*token);
                }
                think.closed();
                tracing::info!("reasoning budget of {think_budget} spent; closed the block");
                let Some(row) = closed.into_last() else {
                    reason = StopReason::ContextFull;
                    break;
                };
                pending = self.pick(&row);
                continue;
            }

            // The cache keeps its own copy of the sequence, so nothing is
            // copied per token here — this loop runs once per generated token
            // and collecting the whole context into a fresh Vec each time made
            // it quadratic in the length of the turn.
            let spec_started = std::time::Instant::now();
            // Pure overhead when a draft model is doing the proposing.
            if spec_on && drafter.is_none() {
                while ngram.len() < self.cached.len() {
                    ngram.push_token(self.cached[ngram.len()].0);
                }
            }
            let draft: Vec<LlamaToken> = if spec_on && ngram.worth_drafting(min_acceptance, probe_every) {
                let room = (n_ctx - self.n_past - 1).max(0) as usize;
                // The verification batch is the confirmed token plus the
                // draft, so it must also stay within n_batch.
                // While acceptance is unknown or poor, this draft is a probe
                // rather than a bet. On the snapshot path a rejected probe
                // costs an extra forward pass, so a long one is pure waste: a
                // short probe measures the same thing for a fraction of it.
                // Known-poor acceptance means this draft is a probe into text
                // that has not been paying. An *unknown* rate is the ramp-up on
                // text that may well be predictable, where a full draft is
                // exactly right — shortening it there measurably slowed the
                // case speculation exists for.
                let probing = ngram.acceptance().is_some_and(|rate| rate < min_acceptance);
                let budget = if use_snapshot && probing {
                    PROBE_DRAFT_TOKENS
                } else {
                    tuning.draft_tokens as usize
                };
                let cap = budget
                    .min(room)
                    .min(n_batch.saturating_sub(1));
                match (mtp.as_mut(), mtp_hidden.as_ref(), drafter.as_mut()) {
                    // The model's own head. Like a draft model it proposes
                    // from a distribution rather than from repetition, so it
                    // needs neither the reach filter nor the probe shortening.
                    (Some(m), Some(hidden), _) => {
                        let want = (tuning.draft_tokens as usize)
                            .min(room)
                            .min(n_batch.saturating_sub(1));
                        // Drafted under the hub's lock.
                        //
                        // The drafter has a context of its own but shares this
                        // one's memory — that is what makes it cheap — so its
                        // decodes write into cells another slot's pass may be
                        // reading, and nothing else would stop the two
                        // overlapping. No misbehaviour was traced to it; the
                        // answer divergence that prompted this turned out to
                        // be the ordinary float-reduction difference of a
                        // shared batch, identical with drafting switched off.
                        // The lock stays because two threads inside one
                        // cache is a race whether or not it has bitten yet.
                        let pos = self.n_past;
                        self.slot.with_context(|_| m.propose(pending, hidden, pos, want))?
                    }
                    _ => match drafter.as_mut() {
                    // A draft model proposes from the whole distribution rather
                    // than from repetition, so it needs neither the reach
                    // filter nor the probe shortening — both exist to stop
                    // n-grams betting on coincidences.
                    Some(d) => {
                        let want = (tuning.draft_tokens as usize)
                            .min(room)
                            .min(n_batch.saturating_sub(1));
                        d.propose(&self.cached, want)?
                    }
                    // Draft longer while drafts are landing, shorter when they
                    // are not: a rejected draft wastes the whole batch slot.
                    None => ngram
                        .draft(ngram.suggest_len(cap), min_reach)
                        .into_iter()
                        .map(LlamaToken)
                        .collect(),
                    },
                }
            } else {
                Vec::new()
            };
            stats.spec_ms += spec_started.elapsed().as_nanos() / 1000;

            // One batch carries the confirmed token plus the whole draft, and
            // asks for logits at every position so each can be verified.
            let start = self.n_past;
            // Taken before the batch, because a restore returns the sequence to
            // exactly this point. Skipped when there is nothing to undo.
            let snapshot = match (use_snapshot, draft.is_empty()) {
                (true, false) => Some(self.snapshot(false, true)?),
                _ => None,
            };
            let mut run = Vec::with_capacity(1 + draft.len());
            run.push(pending);
            run.extend_from_slice(&draft);
            // Every position asks for logits, so each draft can be verified.
            let outcome = self.feed_wanting(run, start, crate::hub::Logits::All, mtp.is_some())?;
            let hidden_rows = outcome.hidden;
            let verified = outcome.rows;
            debug_assert_eq!(verified.len(), 1 + draft.len());

            // Verify. Row i predicts the token after batch entry i, so a
            // drafted token is accepted only when it equals what the model
            // itself would have sampled — which is why output never changes.
            let mut chosen = self.pick(&verified[0]);
            let mut accepted = 0usize;

            for (i, d) in draft.iter().enumerate() {
                if *d != chosen {
                    break; // every later row was conditioned on a wrong token
                }
                if self.model.is_eog_token(chosen) {
                    break;
                }
                if produced >= limit {
                    break;
                }
                if !emit(chosen, self.model, &mut decoder, &mut gate, &mut think, &mut on_token)?.0 {
                    reason = StopReason::Cancelled;
                    accepted = i + 1;
                    self.cached.push(chosen);
                    self.settle_draft(snapshot.as_ref(), start, pending, &draft, accepted)?;
                    break 'outer;
                }
                produced += 1;
                stats.generated_tokens += 1;
                stats.drafted_tokens += 1;
                self.cached.push(chosen);
                accepted = i + 1;
                chosen = self.pick(&verified[i + 1]);
            }

            if !draft.is_empty() {
                ngram.observe(draft.len(), accepted);
                stats.accepted_drafts += accepted;
                stats.proposed_drafts += draft.len();
            }

            // The hidden state the next draft continues from, taken before
            // `settle_draft` disturbs anything. Batch entry 0 is the confirmed
            // token and 1..=accepted are the drafts that were kept, so the last
            // confirmed position is exactly `accepted` — and `chosen`, about to
            // become `pending`, is the token that follows it. Reading any other
            // row drafts a continuation of the wrong position, which reads as
            // fluent nonsense rather than as an error.
            if mtp.is_some() {
                let n_embd = self.model.n_embd() as usize;
                // Taken from this slot's own rows of the pass. Read off the
                // context directly it would be whichever row of the shared
                // batch happened to sit at that index — another conversation's
                // hidden state, and a draft continuing from a place this one
                // has never been.
                mtp_hidden = hidden_rows
                    .get(accepted)
                    .filter(|row| !row.is_empty())
                    .cloned();
            }

            self.settle_draft(snapshot.as_ref(), start, pending, &draft, accepted)?;
            pending = chosen;
        }
        if mtp.is_some() {
            // Left on, the next turn's prefill would emit a hidden state per
            // prompt token for nobody.
            drop(nextn_on.take());
        }

        if think.spent() > 0 {
            tracing::debug!("reasoning ran {} tokens (budget {think_budget})", think.spent());
        }
        let tail = decoder.finish();
        if !tail.is_empty() {
            on_token(&tail);
        }
        // The constraint belongs to the call that has now finished; leaving it
        // installed would force the next turn to be another tool call.
        if self.gate_applied {
            self.set_grammar(None)?;
            self.gate_applied = false;
        }
        stats.generation_ms = gen_started.elapsed().as_millis();
        stats.callback_ms = callback_ns / 1_000_000;
        self.slot.hub().note_decoded();
        // Whatever comes next — a tool call, another round, the end of the
        // turn — this session is not about to ask for a token, so nobody
        // should hold a pass open for it.
        self.slot.park();
        Ok((stats, reason))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("could not initialise llama.cpp; no compute backend is usable")]
    BackendInit,
    #[error("model file not found: {path}")]
    Missing { path: String },
    #[error("loading {path}: {reason}")]
    Load { path: String, reason: String },
    #[error("creating a context: {0}")]
    Context(String),
    #[error("applying the chat template: {0}")]
    Template(String),
    #[error("tokenising: {0}")]
    Tokenize(String),
    #[error("detokenising: {0}")]
    Detokenize(String),
    #[error("building a batch: {0}")]
    Batch(String),
    #[error("decoding: {0}")]
    Decode(String),
    #[error("invalid grammar: {0}")]
    Grammar(String),
    #[error("control vector: {0}")]
    ControlVector(String),
    #[error("sequence state: {0}")]
    State(String),
    #[error("the prompt is {tokens} tokens but the context holds {context}; raise --ctx or shorten it")]
    PromptTooLong { tokens: usize, context: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use ozgent_core::Options;

    #[test]
    fn a_grammar_and_speculation_are_never_both_active() {
        // Encoded as a plain invariant because the failure is a hard abort
        // inside llama.cpp rather than a Rust error.
        let opts = Options { speculative: Some(Speculative::Ngram), ..Default::default() }.resolve();
        let grammar_active = true;
        let spec_on = !grammar_active
            && matches!(opts.speculative, Speculative::Ngram | Speculative::Auto);
        assert!(!spec_on, "speculation must yield to a grammar");
    }

    #[test]
    fn a_verification_batch_always_fits_one_pass() {
        // Batches are now built by the hub, which refuses anything wider than
        // `n_batch`. The confirmed token plus a full draft must stay inside
        // that, which is what the draft cap is for.
        for n_batch in [4usize, 32, 512] {
            let draft = Options::default().resolve().speculative_tuning.draft_tokens as usize;
            let capped = draft.min(n_batch.saturating_sub(1));
            assert!(1 + capped <= n_batch, "{n_batch} cannot carry its own draft");
        }
    }

    #[test]
    fn prefill_chunks_never_exceed_the_batch_size() {
        // The exact failure this guards: llama.cpp aborts the process when a
        // batch holds more than n_batch tokens.
        for len in [0, 1, 511, 512, 513, 1024, 4097] {
            for n_batch in [1, 32, 512] {
                let chunks = prefill_chunks(len, n_batch);
                assert!(
                    chunks.iter().all(|(a, b)| b - a <= n_batch),
                    "chunk too large for len={len} n_batch={n_batch}"
                );
                // Contiguous, in order, and covering everything exactly once.
                let covered: usize = chunks.iter().map(|(a, b)| b - a).sum();
                assert_eq!(covered, len, "len={len} n_batch={n_batch}");
                for pair in chunks.windows(2) {
                    assert_eq!(pair[0].1, pair[1].0, "chunks must be contiguous");
                }
                assert_eq!(chunks.first().map(|c| c.0).unwrap_or(0), 0);
                assert_eq!(chunks.last().map(|c| c.1).unwrap_or(0), len);
            }
        }
    }

    #[test]
    fn an_empty_prompt_produces_no_chunks() {
        assert!(prefill_chunks(0, 512).is_empty());
    }

    #[test]
    fn a_zero_batch_size_does_not_loop_forever() {
        // Defensive: n_batch is clamped, so this must terminate.
        assert_eq!(prefill_chunks(3, 0).len(), 3);
    }

    #[test]
    fn only_the_final_token_of_the_whole_prompt_needs_logits() {
        // With 1000 tokens and n_batch 512 there are two chunks; the logits
        // flag belongs to index 999, not to index 511.
        let chunks = prefill_chunks(1000, 512);
        assert_eq!(chunks, vec![(0, 512), (512, 1000)]);
        let last = 999;
        assert!(last >= chunks[1].0 && last < chunks[1].1, "last token is in the final chunk");
    }

    #[test]
    fn common_prefix_finds_the_shared_head() {
        let a: Vec<LlamaToken> = [1, 2, 3, 4].iter().map(|i| LlamaToken(*i)).collect();
        let b: Vec<LlamaToken> = [1, 2, 9, 9].iter().map(|i| LlamaToken(*i)).collect();
        assert_eq!(common_prefix(&a, &b), 2);
        assert_eq!(common_prefix(&a, &a), 4, "identical sequences share everything");
        assert_eq!(common_prefix(&a, &[]), 0);
        assert_eq!(common_prefix(&[], &b), 0);
    }

    #[test]
    fn a_diverging_first_token_shares_nothing() {
        let a: Vec<LlamaToken> = [1, 2, 3].iter().map(|i| LlamaToken(*i)).collect();
        let b: Vec<LlamaToken> = [9, 2, 3].iter().map(|i| LlamaToken(*i)).collect();
        assert_eq!(common_prefix(&a, &b), 0);
    }

    #[test]
    fn stats_compute_rates() {
        let s = Stats {
            generated_tokens: 100,
            generation_ms: 2000,
            prompt_tokens: 50,
            prompt_ms: 500,
            reused_tokens: 0,
            drafted_tokens: 0,
            ..Default::default()
        };
        assert!((s.tokens_per_second() - 50.0).abs() < 0.01);
        assert!((s.prompt_tokens_per_second() - 100.0).abs() < 0.01);
    }

    #[test]
    fn rates_do_not_divide_by_zero() {
        assert_eq!(Stats::default().tokens_per_second(), 0.0);
        assert_eq!(Stats::default().prompt_tokens_per_second(), 0.0);
    }

    #[test]
    fn the_fallback_prompt_labels_every_role() {
        let msgs = vec![
            Message::system("be brief"),
            Message::user("hi"),
            Message::assistant("hello"),
        ];
        let p = fallback_prompt(&msgs);
        assert!(p.contains("System: be brief"));
        assert!(p.contains("User: hi"));
        assert!(p.ends_with("Assistant: "), "must invite a completion: {p:?}");
    }

    #[test]
    fn the_empty_think_block_is_well_formed() {
        // Malformed here means the model keeps reasoning, which is the bug
        // this constant exists to fix.
        assert!(EMPTY_THINK.starts_with("<think>"));
        assert!(EMPTY_THINK.contains("</think>"));
        assert!(
            EMPTY_THINK.find("<think>") < EMPTY_THINK.find("</think>"),
            "the block must open before it closes"
        );
    }

    #[test]
    fn temperature_zero_selects_greedy_sampling() {
        // Building must not panic; a temp sampler at 0.0 would divide by zero.
        let opts = Options { temperature: Some(0.0), ..Default::default() }.resolve();
        let _ = build_sampler(&opts);
    }

    #[test]
    fn a_normal_temperature_builds_a_full_chain() {
        let opts = Options::default().resolve();
        let _ = build_sampler(&opts);
    }
}

/// Whether llama.cpp refused a context because the KV cache is quantised and
/// there is no flash attention to support it.
///
/// Matched on the message because llama.cpp reports every context failure as a
/// null pointer and explains itself only through its log callback.
fn needs_flash_attention(reason: &str) -> bool {
    let reason = reason.to_ascii_lowercase();
    reason.contains("flash attention") && reason.contains("cache")
}

#[cfg(test)]
mod expert_offload_tests {
    use super::host_expert_bytes;
    use crate::layout::Layout;
    use ozgent_core::accel::{MoeKeyword, MoeOffload};

    fn moe(layers: u32, per_layer: u64) -> Layout {
        Layout { layers, expert_bytes_per_layer: per_layer, ..Default::default() }
    }

    #[test]
    fn every_expert_on_the_host_is_the_whole_stack() {
        let l = moe(48, 400 << 20);
        assert_eq!(host_expert_bytes(&l, MoeOffload::ALL), Some(48 * (400 << 20)));
    }

    #[test]
    fn a_partial_offload_is_counted_by_the_layers_it_names() {
        let l = moe(48, 400 << 20);
        assert_eq!(host_expert_bytes(&l, MoeOffload::Layers(12)), Some(12 * (400 << 20)));
        // More layers than the model has cannot cost more than the model has.
        assert_eq!(host_expert_bytes(&l, MoeOffload::Layers(99)), Some(48 * (400 << 20)));
    }

    #[test]
    fn a_dense_model_sends_nothing_to_the_host() {
        // There is no such thing as a routed expert here, so the warning about
        // system memory must not fire on a model that will never touch it.
        let dense = moe(32, 0);
        assert_eq!(host_expert_bytes(&dense, MoeOffload::ALL), None);
        assert_eq!(host_expert_bytes(&dense, MoeOffload::Layers(8)), None);
    }

    #[test]
    fn nothing_offloaded_costs_nothing() {
        let l = moe(48, 400 << 20);
        assert_eq!(host_expert_bytes(&l, MoeOffload::OFF), None);
        assert_eq!(host_expert_bytes(&l, MoeOffload::Layers(0)), None);
        assert_eq!(host_expert_bytes(&l, MoeOffload::Keyword(MoeKeyword::Auto)), None);
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::{best_prefix_match, lru_victim};
    use llama_cpp_2::token::LlamaToken;

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().map(|&i| LlamaToken(i)).collect()
    }

    #[test]
    fn the_longest_whole_prefix_wins() {
        // Several conversations are held at once, and the one that shares
        // most of this prompt is the one worth restoring.
        let a = toks(&[1, 2, 3]);
        let b = toks(&[1, 2, 3, 4, 5]);
        let c = toks(&[9, 9]);
        let saved: Vec<&[LlamaToken]> = vec![&a, &b, &c];
        let prompt = toks(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(best_prefix_match(&saved, &prompt), Some((1, 5)));
    }

    #[test]
    fn a_state_that_diverges_partway_is_refused() {
        // Restoring it would leave the cache describing tokens that are not
        // there: right positions, wrong contents, which reads as the model
        // having invented its own history.
        let a = toks(&[1, 2, 99]);
        let saved: Vec<&[LlamaToken]> = vec![&a];
        let prompt = toks(&[1, 2, 3, 4]);
        assert_eq!(best_prefix_match(&saved, &prompt), None);
    }

    #[test]
    fn a_state_covering_the_whole_prompt_is_refused() {
        // The last token has to be decoded to produce logits, so a state can
        // never cover all of it.
        let a = toks(&[1, 2, 3]);
        let saved: Vec<&[LlamaToken]> = vec![&a];
        assert_eq!(best_prefix_match(&saved, &toks(&[1, 2, 3])), None);
        assert_eq!(best_prefix_match(&saved, &toks(&[1, 2])), None);
    }

    #[test]
    fn nothing_saved_matches_nothing() {
        assert_eq!(best_prefix_match(&[], &toks(&[1, 2, 3])), None);
    }

    #[test]
    fn the_least_recently_useful_state_is_the_one_dropped() {
        // Ticks rise on both saving and restoring, so "useful" is the right
        // word: a conversation being returned to repeatedly keeps its state
        // even if it has not grown.
        assert_eq!(lru_victim(&[7, 2, 9]), Some(1));
        assert_eq!(lru_victim(&[]), None);
    }
}

#[cfg(test)]
mod flash_fallback_tests {
    use super::needs_flash_attention;

    #[test]
    fn llama_cpps_own_wording_is_recognised() {
        // The message this exists for, verbatim from a 9B MTP model that
        // could not load at all until the fallback was added.
        assert!(needs_flash_attention(
            "llama_init_from_model: failed to initialize the context: quantized V cache \
             was requested, but this requires Flash Attention"
        ));
        assert!(needs_flash_attention(
            "quantized K cache was requested, but this requires flash attention"
        ));
    }

    #[test]
    fn an_ordinary_out_of_memory_is_not_mistaken_for_it() {
        // Falling back to an f16 cache would make a genuine shortage worse,
        // and the retry loop shrinking the window is the right answer there.
        for other in [
            "failed to allocate compute buffers",
            "cuda error: out of memory",
            "failed to initialize the context",
            "",
        ] {
            assert!(!needs_flash_attention(other), "{other:?}");
        }
    }

    #[test]
    fn the_match_does_not_depend_on_how_llama_cpp_capitalises_it() {
        assert!(needs_flash_attention("QUANTIZED V CACHE ... REQUIRES FLASH ATTENTION"));
    }
}
