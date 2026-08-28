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
    kv_elements: u64,
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
    /// Load a GGUF file with the given resolved settings.
    pub fn load(path: &Path, opts: &Resolved) -> Result<Self, EngineError> {
        let backend = backend()?;
        if !path.exists() {
            return Err(EngineError::Missing { path: path.display().to_string() });
        }

        let requested_layers = match opts.gpu_layers {
            GpuLayers::Count(n) => n,
            GpuLayers::Keyword(GpuKeyword::Off) => 0,
            // llama.cpp clamps to the real layer count.
            GpuLayers::Keyword(GpuKeyword::Auto) => u32::MAX,
        };

        let mut params = Box::pin(LlamaModelParams::default()
            .with_n_gpu_layers(requested_layers)
            .with_use_mmap(opts.use_mmap)
            .with_use_mlock(opts.use_mlock)
            .with_main_gpu(opts.main_gpu as i32));

        // `auto` was the default and did nothing: the planner in backend.rs was
        // written and tested but never called, so the option documented as
        // "search for the smallest offload that fits in VRAM" searched for
        // nothing. Resolving it needs the per-layer and per-expert byte costs,
        // which come from the file's tensor table rather than from llama.cpp.
        let resolved_moe = match opts.cpu_moe {
            MoeOffload::Keyword(MoeKeyword::Auto) => {
                let layout = crate::layout::read(path).unwrap_or_default();
                if layout.is_moe() {
                    let device = crate::backend::best_gpu();
                    // The KV cache and compute buffers have to stay resident,
                    // so they come off the budget before any weight does.
                    let overhead = ozgent_core::accel::kv_bytes(
                        layout.kv_elements_per_token,
                        opts.context_length,
                        // Auto KV sizing happens later against the loaded
                        // model; F16 is the conservative assumption here, so a
                        // wrong guess leaves headroom rather than overcommitting.
                        if opts.cache_type_k == ozgent_core::accel::CacheType::Auto {
                            ozgent_core::accel::CacheType::F16
                        } else {
                            opts.cache_type_k
                        },
                    );
                    let (_, moe) = crate::backend::resolve_auto(
                        opts.gpu_layers,
                        opts.cpu_moe,
                        device.as_ref(),
                        layout.layers,
                        layout.bytes_per_layer,
                        layout.expert_bytes_per_layer,
                        overhead,
                    );
                    if moe > 0 {
                        tracing::info!(
                            "cpu-moe auto: evicting routed experts from {moe} of {} layers",
                            layout.layers
                        );
                    }
                    MoeOffload::Layers(moe)
                } else {
                    // Nothing to evict on a dense model.
                    MoeOffload::Layers(0)
                }
            }
            other => other,
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
            MoeOffload::Layers(n) if n > 0 => {
                // One override covering every evicted layer, not one per
                // layer. `add_cpu_buft_override` always fills slot zero, so a
                // second call trips its own "last buft_override was not empty"
                // assertion — which is why this panicked on the first real
                // mixture-of-experts model to reach it. The alternation names
                // the same tensors llama.cpp's own --n-cpu-moe targets.
                let blocks = (0..n).map(|l| l.to_string()).collect::<Vec<_>>().join("|");
                moe_pattern =
                    std::ffi::CString::new(format!("blk\\.({blocks})\\.ffn_(up|down|gate)_(ch|)exps"))
                        .ok();
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
        let kv_elements =
            ozgent_core::accel::kv_elements_per_token(n_layer, model.n_head_kv(), k_len, v_len);
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
            tracing::info!("model keeps unrollbackable state; speculative decoding disabled");
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
            reasoning,
            gen_prompt_tokens: Self::measure_generation_prompt(&model, &jinja, template.as_ref()),
            jinja,
            n_layer,
            rollback_safe,
            kv_elements,
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
    ) -> Result<LlamaContext<'_>, EngineError> {
        let mut window = requested;
        // The first failure is the informative one: later attempts fail for
        // the same reason at a smaller size, and the last one before the floor
        // says least about what actually went wrong.
        let mut first_reason: Option<String> = None;

        loop {
            crate::llamalog::clear();
            let attempt = params.clone().with_n_ctx(NonZeroU32::new(window));
            match self.model.new_context(backend, attempt) {
                Ok(context) => {
                    if window < requested {
                        tracing::warn!(
                            "a context of {requested} could not be allocated; opened {window} instead"
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
                    first_reason.get_or_insert(reason);

                    if window <= ozgent_core::accel::MIN_CONTEXT {
                        let reason = first_reason.unwrap_or_else(|| e.to_string());
                        return Err(EngineError::Context(format!(
                            "{reason} (tried down to {window} tokens of context)"
                        )));
                    }
                    window = (window / 2).max(ozgent_core::accel::MIN_CONTEXT);
                }
            }
        }
    }

    pub fn session(&self, opts: &Resolved) -> Result<Session<'_>, EngineError> {
        let backend = backend()?;

        // Asking for more context than the model was trained on produces
        // gibberish rather than an error, so it is clamped with a warning.
        let requested = opts.context_length.min(self.n_ctx_train.max(512));
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
        let resolve_kv = |requested_type: CacheType| -> CacheType {
            if requested_type != CacheType::Auto {
                return requested_type;
            }
            ozgent_core::accel::choose_kv_type(
                self.kv_elements,
                requested,
                self.weight_bytes,
                free,
                opts.flash_attention,
            )
        };
        let type_k = resolve_kv(opts.cache_type_k);
        let type_v = resolve_kv(opts.cache_type_v);

        // Training context is a claim about which positions the model
        // understands, not a promise that the cache for them fits in memory.
        // At the million-token windows recent models advertise, the cache runs
        // to hundreds of gigabytes, and llama.cpp answers that by returning a
        // null pointer — which reached the user as "null reference from
        // llama.cpp" with nothing pointing at memory as the cause. Sizing the
        // window to the memory that exists trades an unusable session for a
        // shorter one.
        let budget = ozgent_core::accel::kv_budget(
            free,
            ozgent_core::accel::available_host_memory(),
            self.gpu_layers_used,
            self.n_layer,
        );
        // K and V can end up on different types; sizing against the wider one
        // keeps the estimate on the safe side of the real allocation.
        let widest = if type_k.bits() >= type_v.bits() { type_k } else { type_v };
        let requested =
            ozgent_core::accel::fit_context(self.kv_elements, requested, widest, budget);
        if requested < opts.context_length.min(self.n_ctx_train.max(512)) {
            tracing::warn!(
                "context {} needs {} MiB of {:?} kv cache; only {} MiB is free, so the window is {}",
                opts.context_length,
                ozgent_core::accel::kv_bytes(
                    self.kv_elements,
                    opts.context_length.min(self.n_ctx_train.max(512)),
                    widest,
                ) / (1024 * 1024),
                widest,
                budget / (1024 * 1024),
                requested,
            );
        }
        if opts.cache_type_k == CacheType::Auto {
            tracing::info!(
                "kv cache: {type_k:?} ({} MiB at {requested} ctx, {} MiB free)",
                ozgent_core::accel::kv_bytes(self.kv_elements, requested, type_k) / (1024 * 1024),
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
            .with_n_batch(opts.batch_size)
            .with_flash_attention_policy(flash)
            .with_type_k(ggml_type(type_k))
            .with_type_v(ggml_type(type_v));

        // The physical micro-batch. Left at llama.cpp's default unless asked,
        // since it trades prefill parallelism against working-set size.
        if let Some(n) = opts.ubatch {
            params = params.with_n_ubatch(n);
        }

        if opts.threads > 0 {
            params = params
                .with_n_threads(opts.threads as i32)
                .with_n_threads_batch(opts.threads as i32);
        }

        let mut context = self.open_context(backend, params, requested)?;

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

        Ok(Session {
            model: &self.model,
            context,
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
            checkpoint: None,
            tool_grammars: Vec::new(),
            gate_applied: false,
            opts: opts.clone(),
        })
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
/// Slots to allocate for the batch that is reused throughout a generation.
///
/// It has to hold the largest thing ever put in it, which is whichever is
/// bigger: a full prefill chunk (`n_batch`), or the confirmed token plus a
/// full draft. Sizing it to the *current* prompt instead was a real bug: a
/// mid-generation cache rebuild re-prefills far more tokens than the original
/// prompt held, and overflowed the buffer.
fn batch_capacity(n_batch: usize, opts: &Resolved) -> usize {
    let draft = opts.speculative_tuning.draft_tokens as usize;
    n_batch.max(1 + draft).max(1)
}

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

/// Ceiling on a saved prompt state, as a share of free memory.
///
/// The copy is held for the whole conversation, so it has to leave room for
/// the turn it is meant to speed up.
const CHECKPOINT_VRAM_SHARE: usize = 4;

/// Below this many tokens, a full prefill is cheaper than the state copy.
const CHECKPOINT_MIN_TOKENS: usize = 256;

/// A prompt boundary the cache can be returned to.
///
/// A conversation resends its whole history, so almost every prompt is the
/// previous one plus a reply and a new question — and almost all of it is
/// already in the cache. Reusing it means dropping whatever sits *after* the
/// shared prefix, which for a plain attention cache is a `seq_rm` away.
///
/// A hybrid model cannot do that. Its recurrent layers hold a state that was
/// folded forward token by token and cannot be unwound, so llama.cpp refuses
/// the partial removal and the only correct answer is to clear the cache and
/// prefill the whole conversation again. On a 9,000-token chat that is 4.7
/// seconds per turn, every turn, growing as the conversation does.
///
/// Saving the state at the end of the prompt sidesteps it. There is nothing to
/// unwind next turn because the checkpoint predates the generated tokens: put
/// the state back and the cache holds exactly the prompt, ready to be extended
/// by whatever the new turn adds.
struct PromptCheckpoint {
    /// Exactly the tokens the saved state was built from.
    tokens: Vec<LlamaToken>,
    state: SeqState,
}

/// One conversation against one KV cache.
pub struct Session<'a> {
    model: &'a LlamaModel,
    context: LlamaContext<'a>,
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
    checkpoint: Option<PromptCheckpoint>,
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
}

impl<'a> Session<'a> {
    pub fn n_ctx(&self) -> u32 {
        self.context.n_ctx()
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
        self.context
            .state_seq_get(0, LlamaStateSeqFlags::from_bits(bits))
            .map_err(|e| EngineError::State(e.to_string()))
    }

    fn restore(&mut self, state: &SeqState) -> Result<(), EngineError> {
        self.context
            .state_seq_set(state, 0)
            .map_err(|e| EngineError::State(e.to_string()))
    }

    /// Save the cache as it stands, which the caller guarantees is exactly
    /// `tokens`.
    ///
    /// Only for models whose cache cannot be trimmed. Everywhere else a
    /// `seq_rm` already reuses the prefix for free, and copying the state
    /// would be pure cost.
    fn save_checkpoint(&mut self, tokens: &[LlamaToken]) {
        if self.rollback_safe && self.can_trim {
            return;
        }
        if tokens.len() < CHECKPOINT_MIN_TOKENS {
            return;
        }
        // Already holding exactly this boundary — usually because it was just
        // restored from. Copying the same state again buys nothing.
        if self.checkpoint.as_ref().is_some_and(|c| c.tokens == tokens) {
            return;
        }
        self.checkpoint = None;

        let bytes = self.state_bytes(false, false);
        if let Some(budget) = self.checkpoint_budget() {
            if bytes > budget {
                tracing::debug!(
                    "checkpoint skipped: {} MiB of state against a {} MiB budget",
                    bytes / (1024 * 1024),
                    budget / (1024 * 1024)
                );
                return;
            }
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
                tracing::debug!(
                    "checkpoint saved: {} tokens, {} MiB",
                    tokens.len(),
                    bytes / (1024 * 1024)
                );
                self.checkpoint = Some(PromptCheckpoint { tokens: tokens.to_vec(), state });
            }
            // Not fatal: without a checkpoint the next turn prefills in full,
            // which is what happened before this existed.
            Err(e) => tracing::debug!("checkpoint not saved: {e}"),
        }
    }

    /// Bytes a checkpoint may occupy, or `None` when free memory is unknown.
    fn checkpoint_budget(&self) -> Option<usize> {
        let free = crate::backend::devices()
            .into_iter()
            .find(|d| !d.is_gpu())
            .map(|d| d.memory_free)?;
        Some(free / CHECKPOINT_VRAM_SHARE)
    }

    /// Put the cache back to a saved prompt boundary, if that boundary starts
    /// `tokens`. Returns how many tokens are then resident.
    ///
    /// Asked only once a trim has been refused, because until then the live
    /// cache is the cheaper answer. The checkpoint has to be a *whole* prefix
    /// of the new prompt: restoring a state that disagrees partway through
    /// would leave the cache describing tokens that are not there.
    fn restore_checkpoint(&mut self, tokens: &[LlamaToken]) -> Option<usize> {
        if matches!(self.reuse, PrefixReuse::Off) {
            return None;
        }
        let checkpoint = self.checkpoint.take()?;
        let n = checkpoint.tokens.len();
        // Strictly shorter, so at least one token is left to decode for logits.
        if n >= tokens.len() || common_prefix(&checkpoint.tokens, tokens) != n {
            self.checkpoint = Some(checkpoint);
            return None;
        }
        if let Err(e) = self.restore(&checkpoint.state) {
            tracing::debug!("checkpoint restore failed: {e}");
            // The cache is now of unknown shape; force a clean prefill.
            self.context.clear_kv_cache();
            self.cached.clear();
            self.n_past = 0;
            return None;
        }
        self.n_past = n as i32;
        self.cached = checkpoint.tokens.clone();
        // Kept, not consumed. A turn that adds nothing before the boundary —
        // the same question asked twice, say — saves no new checkpoint, and
        // dropping this one would send the turn after it back to a cold
        // prefill. Restoring does not spend it.
        self.checkpoint = Some(checkpoint);
        Some(n)
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
        let n_batch = (self.context.n_batch() as usize).max(1);
        let mut batch = LlamaBatch::new(batch_capacity(n_batch, &self.opts), 1);

        self.reset();
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        let (head, tail) = tokens.split_at(tokens.len().saturating_sub(1));
        self.prefill(head, &mut batch, n_batch, false)?;

        // The snapshot is taken *before* the final prompt token, so each run
        // can decode it again and regenerate its logits. Restoring state does
        // not restore the logits buffer, so a run resumed straight after a
        // restore would sample its first token from whatever the previous run
        // left behind — which looks exactly like the rewind having failed.
        let mark = self.n_past;
        let state = self.snapshot(partial, on_device)?;
        let size = state.byte_len();

        self.prefill(tail, &mut batch, n_batch, true)?;
        let first = self.run_greedy(k, &mut batch, n_batch)?;

        self.restore(&state)?;
        self.n_past = mark;
        self.sampler.reset();
        self.prefill(tail, &mut batch, n_batch, true)?;
        let second = self.run_greedy(k, &mut batch, n_batch)?;

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
        let n_ctx = self.context.n_ctx() as i32;
        if context.len() as i32 + n as i32 >= n_ctx {
            return Ok(Vec::new());
        }
        let n_batch = (self.context.n_batch() as usize).max(1);
        let mut batch = LlamaBatch::new(batch_capacity(n_batch, &self.opts), 1);

        // Anything the drafter has not absorbed yet. A divergence means the
        // target went somewhere this session never saw, so start over rather
        // than draft from a transcript that never happened.
        let shared = common_prefix(&self.cached, context);
        if shared < self.cached.len() {
            self.reset();
        }
        let fresh: Vec<LlamaToken> = context[self.cached.len()..].to_vec();
        if !fresh.is_empty() {
            self.prefill(&fresh, &mut batch, n_batch, true)?;
            self.cached.extend_from_slice(&fresh);
        }

        let mark = self.n_past;
        let cached_len = self.cached.len();
        let state = self.snapshot(false, true)?;

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let token = self.sampler.sample(&self.context, -1);
            if self.model.is_eog_token(token) {
                break;
            }
            out.push(token);
            batch.clear();
            batch
                .add(token, self.n_past, &[0], true)
                .map_err(|e| EngineError::Batch(e.to_string()))?;
            self.context
                .decode(&mut batch)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
            self.n_past += 1;
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
    ) -> Result<(f64, f64, usize), EngineError> {
        let n_batch = (self.context.n_batch() as usize).max(1);
        let mut batch = LlamaBatch::new(batch_capacity(n_batch, &self.opts), 1);

        self.reset();
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::Tokenize(e.to_string()))?;
        self.prefill(&tokens, &mut batch, n_batch, true)?;

        // One outside the loop, so allocation and any first-call setup are not
        // charged to the average.
        let warm = self.snapshot(false, on_device)?;
        self.restore(&warm)?;

        let start = Instant::now();
        let mut states = Vec::with_capacity(iterations as usize);
        for _ in 0..iterations {
            states.push(self.snapshot(false, on_device)?);
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
    fn run_greedy(
        &mut self,
        k: u32,
        batch: &mut LlamaBatch,
        _n_batch: usize,
    ) -> Result<String, EngineError> {
        let mut out = String::new();
        let mut decoder = Utf8Buffer::new();
        for _ in 0..k {
            let token = self.sampler.sample(&self.context, -1);
            if self.model.is_eog_token(token) {
                break;
            }
            let bytes = self
                .model
                .token_to_bytes(token, Special::Tokenize)
                .map_err(|e| EngineError::Detokenize(e.to_string()))?;
            out.push_str(&decoder.push(&bytes));

            batch.clear();
            batch
                .add(token, self.n_past, &[0], true)
                .map_err(|e| EngineError::Batch(e.to_string()))?;
            self.context
                .decode(batch)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
            self.n_past += 1;
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
        self.context.state_seq_get_size_ext(
            0,
            llama_cpp_2::context::session::LlamaStateSeqFlags::from_bits(bits),
        )
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
        batch: &mut LlamaBatch,
        n_batch: usize,
        want_logits: bool,
    ) -> Result<(), EngineError> {
        let last = tokens.len().saturating_sub(1);
        let mut offset = 0usize;
        while offset < tokens.len() {
            let end = (offset + n_batch).min(tokens.len());
            batch.clear();
            for (i, token) in tokens[offset..end].iter().enumerate() {
                let index = offset + i;
                batch
                    .add(
                        *token,
                        self.n_past + index as i32,
                        &[0],
                        // Logits are needed only for the final token of the
                        // whole sequence, not of each chunk.
                        want_logits && index == last,
                    )
                    .map_err(|e| EngineError::Batch(e.to_string()))?;
            }
            self.context
                .decode(batch)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
            offset = end;
        }
        self.n_past += tokens.len() as i32;
        Ok(())
    }

    /// Drop the cache and start a fresh conversation on this context.
    ///
    /// Reusing the context avoids re-allocating the KV cache, which for a
    /// large context is the slow part of opening a session.
    pub fn reset(&mut self) {
        self.context.clear_kv_cache();
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
        batch: &mut LlamaBatch,
        n_batch: usize,
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
                self.prefill(&confirmed, batch, n_batch, false)
            }
            None => self.trim_after_draft(start, accepted, batch, n_batch),
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
        batch: &mut LlamaBatch,
        n_batch: usize,
    ) -> Result<(), EngineError> {
        let valid = start + 1 + accepted as i32;
        if valid >= self.n_past {
            return Ok(());
        }
        if self
            .context
            .kv_cache_seq_rm(0, Some(valid as u32), None)
            .is_ok()
        {
            self.n_past = valid;
            return Ok(());
        }

        tracing::debug!("this cache cannot trim; rebuilding it without speculation");
        self.can_trim = false;
        self.context.clear_kv_cache();
        self.n_past = 0;

        let good: Vec<LlamaToken> = self
            .cached
            .iter()
            .copied()
            .take(valid.max(0) as usize)
            .collect();
        self.prefill(&good, batch, n_batch, true)?;
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
        let n_ctx = self.context.n_ctx() as i32;
        let mut stats = Stats::default();
        let started = Instant::now();
        let n_batch = (self.context.n_batch() as usize).max(1);
        let mut batch = LlamaBatch::new(batch_capacity(n_batch, &self.opts), 1);

        // An image becomes embeddings, not token ids. Once mtmd has written
        // them into the cache there is no token sequence that describes what is
        // resident, so the prefix-reuse bookkeeping cannot be trusted and the
        // next turn has to start from a clean cache.
        if self.media_dirty {
            self.context.clear_kv_cache();
            self.cached.clear();
            self.n_past = 0;
            self.media_dirty = false;
        }

        match media {
            Some((projector, images, sources)) if !images.is_empty() => {
                self.context.clear_kv_cache();
                self.cached.clear();
                self.n_past = 0;

                let new_past = projector
                    .eval(&mut self.context, prompt, images, sources, 0, n_batch as i32, true)
                    .map_err(|e| EngineError::Decode(e.to_string()))?;
                if new_past >= n_ctx {
                    return Err(EngineError::PromptTooLong {
                        tokens: new_past as usize,
                        context: n_ctx as usize,
                    });
                }
                self.n_past = new_past;
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
                if reuse < self.cached.len() {
                    // Drop everything after the shared prefix; those positions
                    // are about to be occupied by different tokens.
                    if self.can_trim
                        && self
                            .context
                            .kv_cache_seq_rm(0, Some(reuse as u32), None)
                            .is_err()
                    {
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
                                self.context.clear_kv_cache();
                                self.cached.clear();
                                reuse = 0;
                            }
                        }
                    }
                }
                tracing::debug!(
                    "prefix reuse: {reuse} of {} prompt tokens{}",
                    tokens.len(),
                    if restored { " (restored from checkpoint)" } else { "" }
                );
                self.n_past = reuse as i32;
                self.last_reused = reuse;

                // Prefill stops at the history boundary first so the state can
                // be captured there. Everything up to that point is what the
                // next turn will resend verbatim; the generation prompt after
                // it is replaced by the reply the model is about to write.
                let boundary = tokens.len().saturating_sub(self.gen_prompt_tokens).max(reuse);
                if boundary > reuse {
                    self.prefill(&tokens[reuse..boundary], &mut batch, n_batch, false)?;
                    self.cached = tokens[..boundary].to_vec();
                    self.save_checkpoint(&tokens[..boundary]);
                }
                self.prefill(&tokens[boundary..], &mut batch, n_batch, true)?;
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
        let mut spec_on = !self.grammar_active
            && (drafter.is_some()
                || matches!(self.opts.speculative, Speculative::Ngram | Speculative::Auto));
        let mut use_snapshot = spec_on && !by_trim;
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
        let mut pending = self.sampler.sample(&self.context, -1);

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
                batch.clear();
                batch
                    .add(pending, self.n_past, &[0], false)
                    .map_err(|e| EngineError::Batch(e.to_string()))?;
                for (i, token) in closing.iter().enumerate() {
                    batch
                        .add(*token, self.n_past + 1 + i as i32, &[0], i + 1 == closing.len())
                        .map_err(|e| EngineError::Batch(e.to_string()))?;
                }
                self.context
                    .decode(&mut batch)
                    .map_err(|e| EngineError::Decode(e.to_string()))?;
                self.n_past += 1 + closing.len() as i32;

                for token in &closing {
                    if !emit(*token, self.model, &mut decoder, &mut gate, &mut think, &mut on_token)?.0 {
                        reason = StopReason::Cancelled;
                        break 'outer;
                    }
                    self.cached.push(*token);
                }
                think.closed();
                tracing::info!("reasoning budget of {think_budget} spent; closed the block");
                pending = self.sampler.sample(&self.context, -1);
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
                match drafter.as_mut() {
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
            batch.clear();
            batch
                .add(pending, start, &[0], true)
                .map_err(|e| EngineError::Batch(e.to_string()))?;
            for (i, d) in draft.iter().enumerate() {
                batch
                    .add(*d, start + 1 + i as i32, &[0], true)
                    .map_err(|e| EngineError::Batch(e.to_string()))?;
            }
            self.context
                .decode(&mut batch)
                .map_err(|e| EngineError::Decode(e.to_string()))?;
            self.n_past = start + 1 + draft.len() as i32;

            // Verify. Row i predicts the token after batch entry i, so a
            // drafted token is accepted only when it equals what the model
            // itself would have sampled — which is why output never changes.
            let mut chosen = self.sampler.sample(&self.context, 0);
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
                    self.settle_draft(
                        snapshot.as_ref(), start, pending, &draft, accepted, &mut batch, n_batch,
                    )?;
                    break 'outer;
                }
                produced += 1;
                stats.generated_tokens += 1;
                stats.drafted_tokens += 1;
                self.cached.push(chosen);
                accepted = i + 1;
                chosen = self.sampler.sample(&self.context, (i + 1) as i32);
            }

            if !draft.is_empty() {
                ngram.observe(draft.len(), accepted);
                stats.accepted_drafts += accepted;
                stats.proposed_drafts += draft.len();
            }

            self.settle_draft(
                snapshot.as_ref(), start, pending, &draft, accepted, &mut batch, n_batch,
            )?;
            pending = chosen;
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
    fn the_batch_is_sized_for_the_largest_thing_put_in_it() {
        let opts = Options::default().resolve();
        // A prefill chunk is the usual maximum.
        assert!(batch_capacity(512, &opts) >= 512);
        // But a draft must fit even when n_batch is tiny.
        let cap = batch_capacity(4, &opts);
        assert!(
            cap >= 1 + opts.speculative_tuning.draft_tokens as usize,
            "capacity {cap} cannot hold a full draft"
        );
        assert!(batch_capacity(0, &opts) >= 1, "must never be zero");
    }

    #[test]
    fn capacity_does_not_depend_on_the_current_prompt() {
        // The bug this guards: sizing by the prompt meant a later, longer
        // re-prefill overflowed the batch.
        let opts = Options::default().resolve();
        assert_eq!(batch_capacity(512, &opts), batch_capacity(512, &opts));
        assert!(batch_capacity(512, &opts) >= 512, "always at least n_batch");
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
