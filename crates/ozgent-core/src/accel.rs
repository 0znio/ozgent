//! Acceleration settings that postdate Ollama's design.
//!
//! Ollama exposes little beyond `num_gpu`. The knobs here are the ones that
//! actually decide whether a large model runs at a usable speed on a small
//! card, and they are first-class in ozgent rather than hidden behind env vars.

use serde::{Deserialize, Serialize};

/// Quantisation applied to the KV cache.
///
/// The cache grows linearly with context and, past a few thousand tokens,
/// dominates VRAM. Storing it at `q8_0` halves it for almost no measurable
/// quality loss, which buys context length or lets more layers stay resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheType {
    /// Size the cache against the hardware at load time. See [`choose_kv_type`].
    #[default]
    Auto,
    F16,
    BF16,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0,
}

impl CacheType {
    /// The `ggml_type` enum discriminant llama.cpp expects.
    pub fn to_ggml(self) -> i32 {
        match self {
            // `Auto` is resolved by `choose_kv_type` before a context is
            // built; f16 is the safe reading if that is ever skipped.
            Self::Auto | Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::BF16 => 30,
        }
    }

    /// Approximate bits per element, for VRAM estimates.
    pub fn bits(self) -> f32 {
        match self {
            Self::Auto | Self::F16 | Self::BF16 => 16.0,
            Self::Q8_0 => 8.5,
            Self::Q5_1 => 6.0,
            Self::Q5_0 => 5.5,
            Self::Q4_1 => 5.0,
            Self::Q4_0 => 4.5,
        }
    }

    /// Quantised K requires flash attention in llama.cpp; V does not.
    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::Auto | Self::F16 | Self::BF16)
    }
}

impl std::str::FromStr for CacheType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "f16" | "fp16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            "q8_0" | "q8" => Ok(Self::Q8_0),
            "q5_1" => Ok(Self::Q5_1),
            "q5_0" | "q5" => Ok(Self::Q5_0),
            "q4_1" => Ok(Self::Q4_1),
            "q4_0" | "q4" => Ok(Self::Q4_0),
            other => Err(format!("unknown cache type {other:?}")),
        }
    }
}

// ----------------------------------------------------------- KV cache sizing

/// KV elements stored per token, summed over every layer.
///
/// Head dimension is *not* `n_embd / n_head` in general — Qwen3.5 has
/// `n_embd` 2560 over 16 heads (=160) but a real key length of 256, so
/// deriving it would mis-size the cache by 60%. Callers read the true
/// key/value lengths from GGUF metadata.
pub fn kv_elements_per_token(n_layer: u32, n_head_kv: u32, k_len: u32, v_len: u32) -> u64 {
    n_layer as u64 * n_head_kv as u64 * (k_len as u64 + v_len as u64)
}

/// Bytes the KV cache occupies at `n_ctx` tokens.
pub fn kv_bytes(elements_per_token: u64, n_ctx: u32, t: CacheType) -> u64 {
    // bits() is fractional (q8_0 is 8.5 including its scale), so the whole
    // product is computed in floating point; integer maths would truncate the
    // scale away and under-size the cache by 6%.
    (elements_per_token as f64 * n_ctx as f64 * t.bits() as f64 / 8.0) as u64
}

/// Fraction of free VRAM the KV cache may claim when nothing better is known.
///
/// Only a fallback now. The real answer comes from [`crate::reserve`], which
/// subtracts a *measured* number of bytes rather than a share of whatever the
/// card happens to have — see the note there for why a percentage is the
/// wrong shape. This remains for the paths that have no reserve to hand, and
/// for host memory, where there is no backend context to measure.
const KV_VRAM_SHARE: u64 = 70;

/// Above this ratio of KV traffic to weight traffic, quantising the cache is a
/// speed win rather than a memory compromise. See [`choose_kv_type`].
const KV_TRAFFIC_RATIO: f64 = 0.25;

/// Preference order: fastest and most accurate first, smallest last.
const KV_CANDIDATES: [CacheType; 4] =
    [CacheType::F16, CacheType::Q8_0, CacheType::Q5_1, CacheType::Q4_0];

/// Choose a KV cache type for this model on this hardware.
///
/// Two independent pressures decide this, and conflating them is why a fixed
/// default is always wrong for some model:
///
/// 1. **Fit.** The cache must fit in what is left after the weights, or the
///    model does not load at all. This is absolute and checked first.
/// 2. **Traffic.** Attention reads the *entire* cache every token. At short
///    context that is a few percent of the weight traffic, so `f16` wins by
///    avoiding dequantisation. Once the cache approaches the weights in size
///    — past roughly 11k tokens on a typical dense model — it becomes the
///    dominant read, and halving it with `q8_0` is faster despite the dequant.
///
/// `free_bytes` is measured *after* the weights are resident, so it already
/// accounts for however many layers were offloaded.
///
/// A quantised K cache requires flash attention in llama.cpp, so with it off
/// the answer is always `f16` — that path is already the slow one, and
/// quantising V alone saves too little to be worth the mixed-type complexity.
pub fn choose_kv_type(
    elements_per_token: u64,
    n_ctx: u32,
    weight_bytes: u64,
    free_bytes: u64,
    flash_attention: bool,
) -> CacheType {
    if !flash_attention {
        return CacheType::F16;
    }
    let budget = free_bytes / 100 * KV_VRAM_SHARE;

    // Expected steady-state read is half the window; the fit check below still
    // uses the full window, because running out of VRAM is not recoverable.
    let avg_f16 = kv_bytes(elements_per_token, n_ctx / 2, CacheType::F16);
    let traffic_bound =
        weight_bytes > 0 && avg_f16 as f64 > weight_bytes as f64 * KV_TRAFFIC_RATIO;

    for candidate in KV_CANDIDATES {
        if traffic_bound && candidate == CacheType::F16 {
            continue;
        }
        if kv_bytes(elements_per_token, n_ctx, candidate) <= budget {
            return candidate;
        }
    }
    // Nothing fits cleanly. The smallest cache is the only chance of loading.
    CacheType::Q4_0
}

// ------------------------------------------------------- fitting the context

/// Smallest context worth opening. Below this a chat cannot hold a system
/// prompt and one exchange, so failing loudly beats handing back a stub.
pub const MIN_CONTEXT: u32 = 512;

/// Contexts are clamped to a multiple of this, so a shrunk window is a round
/// number a user can recognise rather than 178_431.
const CONTEXT_GRAIN: u32 = 256;

/// The largest context at or below `requested` whose KV cache fits `budget`.
///
/// A model's advertised training context is not a promise that the cache for
/// it fits in memory: at 1M tokens a mid-sized model's cache runs to hundreds
/// of gigabytes. llama.cpp does not refuse that politely — it returns a null
/// pointer, which reaches the user as "null reference from llama.cpp" with no
/// hint that memory was the problem. Sizing the window to the memory that
/// actually exists turns that into a working session with a smaller window.
///
/// `budget` of zero means the memory could not be measured, in which case the
/// request is honoured unchanged: guessing a limit from nothing would shrink
/// windows that would have worked.
pub fn fit_context(elements_per_token: u64, requested: u32, t: CacheType, budget: u64) -> u32 {
    if budget == 0 || elements_per_token == 0 || requested <= MIN_CONTEXT {
        return requested;
    }
    if kv_bytes(elements_per_token, requested, t) <= budget {
        return requested;
    }
    // Bytes per token is fractional for the quantised types, so the division
    // happens in floating point; integer maths would round the scale away and
    // over-estimate what fits.
    let per_token = elements_per_token as f64 * t.bits() as f64 / 8.0;
    let affordable = (budget as f64 / per_token) as u64;
    let affordable = u32::try_from(affordable).unwrap_or(u32::MAX).min(requested);
    let rounded = affordable / CONTEXT_GRAIN * CONTEXT_GRAIN;
    rounded.max(MIN_CONTEXT)
}

// --------------------------------------------------- the two halves of a cache

/// How many elements per token each half of the cache holds, and how wide a
/// single head is in each — which llama.cpp needs to divide by the block size.
///
/// `k` and `v` are the layers whose cache grows with the window. A model with
/// sliding-window layers keeps a second cache for them, and that one does not
/// grow: llama.cpp sizes it once, from the window and the sequence count (see
/// [`swa_cells`]), so it is carried here as a fixed number of cells rather
/// than folded into the per-token figure. Zero for every model without one,
/// which leaves them priced exactly as before.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvShape {
    pub k: u64,
    pub v: u64,
    pub k_len: u32,
    pub v_len: u32,
    /// Elements per cell across the sliding-window layers, each half.
    pub swa_k: u64,
    pub swa_v: u64,
    /// Cells the sliding-window cache holds whatever the window. Clamped to
    /// the window itself, as llama.cpp does; `u32::MAX` prices a full-size
    /// sliding-window cache, which is what `swa_full` allocates.
    pub swa_cells: u32,
}

impl KvShape {
    pub fn new(n_layer: u32, n_head_kv: u32, k_len: u32, v_len: u32) -> Self {
        let per = n_layer as u64 * n_head_kv as u64;
        Self { k: per * k_len as u64, v: per * v_len as u64, k_len, v_len, ..Self::default() }
    }

    /// The same shape with `swa_layers` of its layers on a sliding window of
    /// `cells` cells. Those layers leave the growing figure and become a
    /// fixed one.
    pub fn with_window(self, n_layer: u32, swa_layers: u32, n_head_kv: u32, cells: u32) -> Self {
        let swa_layers = swa_layers.min(n_layer);
        if swa_layers == 0 {
            return self;
        }
        let full = Self::new(n_layer - swa_layers, n_head_kv, self.k_len, self.v_len);
        let swa = Self::new(swa_layers, n_head_kv, self.k_len, self.v_len);
        Self { swa_k: swa.k, swa_v: swa.v, swa_cells: cells, ..full }
    }

    /// Both halves together, for callers that size against a single type.
    pub fn total(self) -> u64 {
        self.k + self.v
    }

    /// Bytes at `n_ctx` tokens, with each half stored its own way.
    pub fn bytes(self, n_ctx: u32, split: KvSplit) -> u64 {
        kv_bytes(self.k, n_ctx, split.k) + kv_bytes(self.v, n_ctx, split.v) + self.window_bytes(n_ctx, split)
    }

    /// The sliding-window cache at `n_ctx` tokens: never more than the window
    /// asked for, never more than its own fixed size.
    pub fn window_bytes(self, n_ctx: u32, split: KvSplit) -> u64 {
        let cells = n_ctx.min(self.swa_cells);
        kv_bytes(self.swa_k, cells, split.k) + kv_bytes(self.swa_v, cells, split.v)
    }
}

/// Cells llama.cpp gives a sliding-window cache that is not full-size.
///
/// Its own rule, from `llama_kv_cache_iswa`: one window per sequence when
/// they share a pool (one per stream otherwise, each stream its own cache),
/// plus a micro-batch of room for the tokens being written, padded to 256 and
/// never more than the window itself. `n_ctx` is the whole context, as the
/// rest of this module counts it.
pub fn swa_cells(n_swa: u32, n_ctx: u32, n_seq: u32, unified: bool, n_ubatch: u32) -> u32 {
    let n_seq = n_seq.max(1);
    let (per_stream_ctx, windows, streams) =
        if unified { (n_ctx, n_seq, 1) } else { (n_ctx / n_seq, 1, n_seq) };
    let want = n_swa.saturating_mul(windows).saturating_add(n_ubatch).min(per_stream_ctx);
    want.div_ceil(256).saturating_mul(256).saturating_mul(streams)
}

/// How each half of the cache is stored. They are not the same question.
///
/// K is what every query is scored against: an error there moves attention to
/// the wrong token, and the mistake compounds through the rest of the sequence.
/// V is only summed *after* the weights have been decided, so an error there is
/// averaged across however many tokens attended — it fades instead of steering.
///
/// So the two degrade at very different rates, and giving them one type throws
/// that away. `q8_0` K with `q4_0` V costs 13 bits per element against `q5_1`
/// everywhere at 12, and is markedly better output for that 8%: the sensitive
/// half stays near-lossless and the forgiving half absorbs the loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSplit {
    pub k: CacheType,
    pub v: CacheType,
}

impl KvSplit {
    pub const fn uniform(t: CacheType) -> Self {
        Self { k: t, v: t }
    }
}

impl Default for KvSplit {
    fn default() -> Self {
        Self::uniform(CacheType::F16)
    }
}

/// Every block-quantised cache type ggml offers has 32 elements to a block, so
/// a head narrower than that — or not a multiple of it — cannot be stored in
/// one. llama.cpp checks this per layer and returns a null context if it fails.
const KV_BLOCK: u32 = 32;

/// Whether llama.cpp will accept this pair, given flash attention and the
/// model's head widths.
///
/// The rules are *asymmetric*, and getting them the wrong way round is
/// expensive: ozgent previously believed a quantised **K** needed flash
/// attention, when in fact it is **V** that does. Every model with no flash
/// attention was therefore given an f16 cache on both halves when its K could
/// have been quantised for free — which on a 9B at 32k is over a gigabyte of
/// VRAM handed back for nothing.
pub fn kv_split_allowed(split: KvSplit, flash: bool, shape: KvShape) -> bool {
    kv_split_allowed_for(split, flash, shape, true)
}

/// Whether a flash-attention kernel exists for this pair on the backend in use.
///
/// `any_pair` is false for CUDA built without `GGML_CUDA_FA_ALL_QUANTS`, which
/// is the default: its kernels then cover only F16/F16, Q4_0/Q4_0 and
/// Q8_0/Q8_0 (fattn.cu). Everything else — any Q5_1, and *any* pair whose two
/// halves differ — has no kernel, so llama.cpp quietly runs without flash
/// attention, refuses the quantised V it no longer supports, and falls back to
/// f16 values and an attention path with far more scratch. Measured on
/// GLM-4.7-Flash: a Q5_1/Q4_0 cache asked for 762 MiB of compute buffer where
/// Q8_0 needed 354, which shrank the window and then ran prefill out of memory.
pub fn flash_has_kernel(split: KvSplit, any_pair: bool) -> bool {
    if any_pair {
        return true;
    }
    let native = |t: CacheType| matches!(t, CacheType::F16 | CacheType::Q4_0 | CacheType::Q8_0);
    split.k == split.v && native(split.k)
}

/// [`kv_split_allowed`] for a backend that may not have a kernel for every
/// pair. See [`flash_has_kernel`].
pub fn kv_split_allowed_for(split: KvSplit, flash: bool, shape: KvShape, any_pair: bool) -> bool {
    if flash && !flash_has_kernel(split, any_pair) {
        return false;
    }
    // "V cache quantization requires flash_attn" — llama-context.cpp. There is
    // no matching rule for K.
    if split.v.is_quantized() && !flash {
        return false;
    }
    // With flash attention on, both halves must divide the block size. With it
    // off the check is skipped entirely, which is why an odd head width can
    // still take a quantised K.
    if flash {
        if split.k.is_quantized() && shape.k_len % KV_BLOCK != 0 {
            return false;
        }
        if split.v.is_quantized() && shape.v_len % KV_BLOCK != 0 {
            return false;
        }
    }
    true
}

/// Pairs to try when flash attention is on, best quality first.
///
/// V is spent before K at every step, and K only moves once V has reached the
/// floor — see [`KvSplit`] for why that is the right order. The list is also
/// monotonically smaller, so the first rung that fits is both the best
/// available and the largest that fits.
/// `f16` is deliberately *not* the first rung.
///
/// It is the most precise, and on a machine with memory to spare that looks
/// like the obvious default — which is how ozgent ended up spending a freed
/// gigabyte of an 8 GB card on it and using more VRAM than before a round of
/// work meant to use less. The precision bought nothing measurable: `q8_0` is
/// near-lossless by construction, and generation measured 48.3 tok/s against
/// f16's 48.5 — inside the noise.
///
/// So the ceiling is `q8_0` and the memory goes back to the card, where it can
/// hold another model, more offloaded layers, or a longer window. `f16` is
/// still reached when a model cannot take a quantised key cache at all; see
/// [`kv_ladder`], which filters this list.
const FLASH_LADDER: [KvSplit; 5] = [
    KvSplit { k: CacheType::Q8_0, v: CacheType::Q8_0 }, // 17.0 bits
    KvSplit { k: CacheType::Q8_0, v: CacheType::Q5_1 }, // 14.5
    KvSplit { k: CacheType::Q8_0, v: CacheType::Q4_0 }, // 13.0
    KvSplit { k: CacheType::Q5_1, v: CacheType::Q4_0 }, // 10.5
    KvSplit { k: CacheType::Q4_0, v: CacheType::Q4_0 }, //  9.0
];

/// Pairs to try without flash attention, where V must stay f16.
///
/// This ladder used to be the single entry `f16`/`f16`, because ozgent had the
/// rule the wrong way round and believed K was the restricted half. Every one
/// of these rungs was unreachable, and a model with no flash attention paid
/// full price for a cache whose keys it could always have quantised.
const PLAIN_LADDER: [KvSplit; 3] = [
    KvSplit { k: CacheType::Q8_0, v: CacheType::F16 }, // 24.5 bits
    KvSplit { k: CacheType::Q5_1, v: CacheType::F16 }, // 22.0
    KvSplit { k: CacheType::Q4_0, v: CacheType::F16 }, // 20.5
];

/// The pairs worth trying on this model. Ordered best-first and smallest-last.
pub fn kv_ladder(flash: bool, shape: KvShape) -> Vec<KvSplit> {
    kv_ladder_for(flash, shape, true)
}

/// [`kv_ladder`] for a backend whose flash attention may not take every pair.
///
/// Filtering the mixed rungs out leaves Q8_0/Q8_0 then Q4_0/Q4_0 on such a
/// backend, which is coarser than the full ladder — but a rung that silently
/// turns flash attention off costs far more than the bits it saves.
pub fn kv_ladder_for(flash: bool, shape: KvShape, any_pair: bool) -> Vec<KvSplit> {
    let ladder: &[KvSplit] = if flash { &FLASH_LADDER } else { &PLAIN_LADDER };
    let usable: Vec<KvSplit> = ladder
        .iter()
        .copied()
        .filter(|s| kv_split_allowed_for(*s, flash, shape, any_pair))
        .collect();
    if usable.is_empty() {
        // Every quantised form was refused for this model — a head width that
        // does not divide the block size, most likely. f16 always works, and
        // is the only reason it is still reachable at all.
        return vec![KvSplit::uniform(CacheType::F16)];
    }
    usable
}

/// Choose how to store each half of the cache on this hardware.
///
/// Same two pressures as before — it has to fit, and past a certain size
/// quantising is a speed win rather than a compromise — but now spent on the
/// half that can afford it. See [`KvSplit`].
/// `budget` is the memory the cache may actually claim — the same figure the
/// window is then fitted against. These used to be computed separately, from
/// two different rules, so a type could be chosen against one budget and the
/// window sized against another; at a 262,144-token window that disagreement
/// picked a cache 600 MiB larger than would fit and llama.cpp refused it.
pub fn choose_kv_split(
    shape: KvShape,
    n_ctx: u32,
    weight_bytes: u64,
    budget: u64,
    flash: bool,
) -> KvSplit {
    choose_kv_split_for(shape, n_ctx, weight_bytes, budget, flash, true)
}

/// [`choose_kv_split`] for a backend whose flash attention may not take every
/// pair. See [`flash_has_kernel`].
pub fn choose_kv_split_for(
    shape: KvShape,
    n_ctx: u32,
    weight_bytes: u64,
    budget: u64,
    flash: bool,
    any_pair: bool,
) -> KvSplit {
    let avg_f16 = kv_bytes(shape.total(), n_ctx / 2, CacheType::F16);
    let traffic_bound =
        weight_bytes > 0 && avg_f16 as f64 > weight_bytes as f64 * KV_TRAFFIC_RATIO;

    let usable = kv_ladder_for(flash, shape, any_pair);

    for split in &usable {
        // Past the traffic threshold an f16 K is the slow choice, not the
        // accurate one; skip it while anything smaller is still available.
        if traffic_bound && split.k == CacheType::F16 && usable.len() > 1 {
            continue;
        }
        if shape.bytes(n_ctx, *split) <= budget {
            return *split;
        }
    }
    // Nothing fits cleanly. The smallest pair this model can take is the only
    // chance of opening a context at all.
    *usable.last().unwrap_or(&KvSplit::uniform(CacheType::F16))
}

/// The largest window that fits, with each half stored its own way.
pub fn fit_context_split(shape: KvShape, requested: u32, split: KvSplit, budget: u64) -> u32 {
    if requested == 0 || shape.total() == 0 {
        return requested;
    }
    let per_token = shape.k as f64 * split.k.bits() as f64 / 8.0
        + shape.v as f64 * split.v.bits() as f64 / 8.0;
    if per_token <= 0.0 {
        return requested;
    }
    // A sliding-window cache is paid for before the first growing token. It
    // is charged at the window asked for, the most it can be, so the fit
    // below stays conservative when the window comes out smaller.
    let budget = budget.saturating_sub(shape.window_bytes(requested, split));
    let affordable = (budget as f64 / per_token) as u64;
    let affordable = u32::try_from(affordable).unwrap_or(u32::MAX).min(requested);
    (affordable / CONTEXT_GRAIN * CONTEXT_GRAIN).max(MIN_CONTEXT)
}

/// Memory the KV cache may claim, given where the model's layers ended up.
///
/// llama.cpp places each layer's cache next to that layer's weights, so a
/// partly offloaded model splits its cache across both pools in the same
/// proportion. Budgeting only VRAM would shrink the window of a CPU-heavy
/// model that had plenty of RAM, and budgeting only RAM would let a
/// fully-offloaded one overrun the card.
///
/// Both pools are discounted by [`KV_VRAM_SHARE`]: compute buffers and
/// llama.cpp's own scratch are not predictable from metadata, and an
/// over-tight fit fails at allocation rather than degrading.
pub fn kv_budget(free_vram: u64, available_ram: u64, gpu_layers: u32, n_layer: u32) -> u64 {
    kv_budget_reserving(free_vram, available_ram, gpu_layers, n_layer, None)
}

/// As [`kv_budget`], but taking a measured reserve off the VRAM side instead
/// of a percentage of it.
///
/// The percentage was costing real memory: on an 8 GB card with 5020 MiB free
/// after the weights, it allowed 3514 and left 1506 idle — enough that a cache
/// which would have fitted at f16 was quantised for no reason. Host memory
/// still uses the share, because nothing there corresponds to a backend
/// context and system RAM has no hard edge to fall off.
pub fn kv_budget_reserving(
    free_vram: u64,
    available_ram: u64,
    gpu_layers: u32,
    n_layer: u32,
    reserve_bytes: Option<u64>,
) -> u64 {
    let vram = match reserve_bytes {
        Some(r) => free_vram.saturating_sub(r),
        None => free_vram / 100 * KV_VRAM_SHARE,
    };
    let share = |bytes: u64| bytes / 100 * KV_VRAM_SHARE;
    let vram_share = |_bytes: u64| vram;
    if n_layer == 0 {
        return vram.max(share(available_ram));
    }
    let on_gpu = gpu_layers.min(n_layer) as u64;
    let on_cpu = (n_layer as u64).saturating_sub(on_gpu);
    let total = n_layer as u64;
    vram_share(free_vram) * on_gpu / total + share(available_ram) * on_cpu / total
}

/// Memory the host can still hand out, in bytes. Zero when it cannot be read.
///
/// Zero is a meaningful answer rather than a failure: every caller treats an
/// unknown budget as "do not clamp", which is the right call when the
/// alternative is shrinking a window on a guess.
pub fn available_host_memory() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // MemAvailable is the kernel's own estimate of what a new allocation
        // can get without swapping, which is exactly the question here.
        // MemFree is not: it excludes reclaimable page cache and would report
        // a few hundred megabytes on a machine with plenty to spare.
        let Ok(text) = std::fs::read_to_string("/proc/meminfo") else { return 0 };
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kib: u64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0);
                return kib * 1024;
            }
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        // There is no MemAvailable equivalent without linking libc, and this
        // is read once per session, so shelling out is affordable. Physical
        // memory is the total, not what is free; the caller's share keeps the
        // over-estimate from being acted on directly.
        let out = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output().ok();
        out.and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|total| total / 2)
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// How many mixture-of-experts layers keep their routed experts in system RAM.
///
/// In an MoE model the routed experts are most of the weight but only a
/// fraction of the work per token: the router picks a handful and skips the
/// rest. Attention is the reverse — small, but touched by every token. So
/// evicting experts to CPU frees far more VRAM per lost token/sec than
/// dropping whole layers with `gpu_layers` does.
///
/// Throughput against this value is V-shaped: take the smallest setting that
/// still fits, because every extra layer past that is pure loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MoeOffload {
    /// Offload the experts of the first N layers.
    Layers(u32),
    #[serde(with = "moe_keyword")]
    Keyword(MoeKeyword),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeKeyword {
    /// Search for the smallest offload that fits the model in VRAM.
    Auto,
    /// Keep every expert on the GPU.
    Off,
    /// Offload every routed expert; attention stays on the GPU.
    All,
}

impl MoeOffload {
    pub const AUTO: Self = Self::Keyword(MoeKeyword::Auto);
    pub const OFF: Self = Self::Keyword(MoeKeyword::Off);
    pub const ALL: Self = Self::Keyword(MoeKeyword::All);
}

impl std::str::FromStr for MoeOffload {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::AUTO),
            "off" | "none" => Ok(Self::OFF),
            "all" | "max" => Ok(Self::ALL),
            other => other
                .parse::<u32>()
                .map(Self::Layers)
                .map_err(|_| format!("expected a layer count, \"auto\", \"off\", or \"all\"; got {other:?}")),
        }
    }
}

impl std::fmt::Display for MoeOffload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Layers(n) => write!(f, "{n}"),
            Self::Keyword(MoeKeyword::Auto) => f.write_str("auto"),
            Self::Keyword(MoeKeyword::Off) => f.write_str("off"),
            Self::Keyword(MoeKeyword::All) => f.write_str("all"),
        }
    }
}

mod moe_keyword {
    use super::MoeKeyword;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &MoeKeyword, s: S) -> Result<S::Ok, S::Error> {
        match v {
            MoeKeyword::Auto => "auto",
            MoeKeyword::Off => "off",
            MoeKeyword::All => "all",
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<MoeKeyword, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(MoeKeyword::Auto),
            "off" | "none" => Ok(MoeKeyword::Off),
            "all" | "max" => Ok(MoeKeyword::All),
            other => Err(serde::de::Error::custom(format!(
                "expected \"auto\", \"off\", or \"all\", got {other:?}"
            ))),
        }
    }
}

/// Speculative decoding strategy.
///
/// A cheap drafter proposes several tokens; the big model verifies them in one
/// batch. Accepted drafts are free, and the output is bit-identical to
/// unaccelerated decoding — this is a pure latency win, not an approximation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Speculative {
    /// No speculation.
    Off,

    /// Use the model's own multi-token-prediction head, when it has one.
    /// Costs no extra VRAM, so it is the default whenever it is available.
    Mtp,

    /// Draft from repeated n-grams in the context. Needs no second model and
    /// no MTP head; excellent on code and editing tasks, where the model
    /// quotes its input verbatim, and near-useless on free prose.
    Ngram,

    /// Draft with a second, much smaller model. The strongest general-purpose
    /// option, at the cost of loading another set of weights.
    Draft {
        /// Draft model, as `name:tag`.
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gpu_layers: Option<u32>,
    },

    /// Try MTP, then n-gram, then nothing, based on what the model supports.
    Auto,
}

impl Default for Speculative {
    fn default() -> Self {
        Self::Auto
    }
}

/// Tuning shared by every speculative strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeculativeTuning {
    /// Tokens to propose per step. Too high wastes verification compute when
    /// the drafter is wrong.
    pub draft_tokens: u32,
    /// Abandon a draft once the drafter's own confidence falls below this.
    pub min_confidence: f32,
    /// Stop speculating if the rolling acceptance rate drops under this, since
    /// below roughly this point verification costs more than it saves.
    pub min_acceptance: f32,
}

impl Default for SpeculativeTuning {
    fn default() -> Self {
        Self { draft_tokens: 16, min_confidence: 0.75, min_acceptance: 0.2 }
    }
}

/// Reuse of the KV cache between turns.
///
/// A chat turn re-sends the whole conversation. Recomputing that prefix every
/// turn is the single largest avoidable cost in an interactive session, and it
/// grows as the conversation does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixReuse {
    /// Keep the longest common prefix and re-prefill only what changed.
    #[default]
    Longest,
    /// Recompute the prompt every turn. Slow; useful when debugging.
    Off,
}

#[cfg(test)]
mod tests {

    // Qwen3.5-4B as measured from its GGUF: 32 layers, 4 KV heads, K and V
    // both 256 wide. Used as the worked example throughout these tests.
    const QWEN: (u32, u32, u32, u32) = (32, 4, 256, 256);
    const GIB: u64 = 1024 * 1024 * 1024;

    fn qwen_elements() -> u64 {
        let (l, h, k, v) = QWEN;
        kv_elements_per_token(l, h, k, v)
    }

    fn qwen_shape() -> KvShape {
        let (l, h, k, v) = QWEN;
        KvShape::new(l, h, k, v)
    }

    #[test]
    fn the_two_halves_sum_to_what_one_number_used_to_say() {
        assert_eq!(qwen_shape().total(), qwen_elements());
    }

    /// Spark-X2.5-4B: 36 layers, 27 on a 512-token window, 4 KV heads of 256.
    fn spark_shape(cells: u32) -> KvShape {
        KvShape::new(36, 4, 256, 256).with_window(36, 27, 4, cells)
    }

    #[test]
    fn the_window_cache_is_sized_the_way_llama_cpp_sizes_it() {
        // Pooled: a window per sequence plus a micro-batch, padded to 256.
        // The daemon's shape — four conversations and the commons, 1024 wide.
        assert_eq!(swa_cells(512, 65_536, 5, true, 1024), 3584);
        // Never more than the context.
        assert_eq!(swa_cells(512, 2048, 5, true, 1024), 2048);
        // One session, the default micro-batch: 1024 cells.
        assert_eq!(swa_cells(512, 32_768, 1, true, 512), 1024);
        // Divided: each stream a window of its own.
        assert_eq!(swa_cells(512, 32_768, 4, false, 512), 4 * 1024);
    }

    #[test]
    fn only_full_attention_layers_grow_with_the_window() {
        let q8 = KvSplit::uniform(CacheType::Q8_0);
        let dense = KvShape::new(36, 4, 256, 256);
        let spark = spark_shape(3584);
        // 9 of 36 layers grow.
        assert_eq!(spark.total() * 4, dense.total());
        // At 64k the windowed cache is under a third of the full-size one,
        // and it is the fixed part that makes up the difference from a quarter.
        let (d, w) = (dense.bytes(65_536, q8), spark.bytes(65_536, q8));
        assert!(w * 3 < d, "{w} vs {d}");
        assert_eq!(w - dense.bytes(65_536, q8) / 4, spark.window_bytes(65_536, q8));
        // A shape with no window prices exactly as before.
        assert_eq!(dense.with_window(36, 0, 4, 3584), dense);
    }

    #[test]
    fn the_window_comes_off_the_budget_before_any_token() {
        let q8 = KvSplit::uniform(CacheType::Q8_0);
        let spark = spark_shape(3584);
        let budget = spark.bytes(40_960, q8);
        let fitted = fit_context_split(spark, 65_536, q8, budget);
        // Charged at the window asked for, so never more than what fits.
        assert!(spark.bytes(fitted, q8) <= budget, "{fitted}");
        assert!(fitted >= 36_864, "{fitted}");
    }

    #[test]
    fn every_ladder_gets_smaller_as_it_goes() {
        // The fit search takes the first rung that fits, so a rung larger than
        // the one above it would be skipped over and never chosen — and the
        // last rung has to be the smallest or the fallback is not a fallback.
        for (name, ladder) in [("flash", &FLASH_LADDER[..]), ("plain", &PLAIN_LADDER[..])] {
            let shape = qwen_shape();
            let sizes: Vec<u64> = ladder.iter().map(|s| shape.bytes(4096, *s)).collect();
            for pair in sizes.windows(2) {
                assert!(pair[0] > pair[1], "{name} ladder is not descending: {sizes:?}");
            }
        }
    }

    #[test]
    fn keys_are_never_stored_worse_than_values() {
        // The asymmetry the split exists for. Both halves degrade down the
        // ladder, but K stays at least as precise as V at every rung, because
        // an error in K steers attention and an error in V only blurs it.
        let mut previous = FLASH_LADDER[0];
        for split in FLASH_LADDER {
            assert!(split.k.bits() >= split.v.bits(), "K is worse than V at {split:?}");
            assert!(split.k.bits() <= previous.k.bits(), "K went back up at {split:?}");
            assert!(split.v.bits() <= previous.v.bits(), "V went back up at {split:?}");
            previous = split;
        }
    }

    #[test]
    fn without_flash_attention_only_the_keys_can_give() {
        // llama.cpp pins V to f16 here, so this ladder spends K — which it
        // could never do while ozgent had the rule the wrong way round.
        let mut previous = PLAIN_LADDER[0];
        for split in PLAIN_LADDER {
            assert_eq!(split.v, CacheType::F16, "V must stay f16: {split:?}");
            assert!(split.k.bits() <= previous.k.bits(), "K went back up at {split:?}");
            previous = split;
        }
        assert!(PLAIN_LADDER.len() > 1, "there must be something to fall back to");
    }

    #[test]
    fn a_roomy_card_is_not_spent_on_precision_nobody_can_measure() {
        // The regression this guards: after the cache was corrected to a
        // quarter of its old size, f16 became affordable and `auto` took it —
        // spending a freed gigabyte of an 8 GB card on a difference that
        // measured 48.5 tok/s against q8_0's 48.3. The memory is worth more
        // than the precision; it holds another model or more layers.
        let shape = qwen_shape();
        let roomy = shape.bytes(32_768, KvSplit::uniform(CacheType::F16)) * 8;
        let chosen = choose_kv_split(shape, 32_768, 2_740_000_000, roomy, true);
        assert_eq!(chosen.k, CacheType::Q8_0, "{chosen:?}");
        assert_eq!(chosen.v, CacheType::Q8_0, "{chosen:?}");
    }

    #[test]
    fn a_model_that_cannot_quantise_at_all_still_gets_a_cache() {
        // f16 is no longer a rung, so it has to be reachable some other way.
        // A head width that does not divide the block size filters every
        // quantised pair out, and the answer must not be "no cache".
        let odd = KvShape::new(32, 8, 80, 80);
        let ladder = kv_ladder(true, odd);
        assert_eq!(ladder, vec![KvSplit::uniform(CacheType::F16)]);
        let chosen = choose_kv_split(odd, 32_768, 2_740_000_000, 8 * GIB, true);
        assert_eq!(chosen, KvSplit::uniform(CacheType::F16));
    }

    #[test]
    fn a_model_without_flash_attention_can_still_quantise_its_keys() {
        // The bug: ozgent believed a quantised K needed flash attention, so a
        // model without it got f16 on both halves. llama.cpp restricts V, not
        // K, and the difference here is a quarter of the cache.
        let shape = qwen_shape();
        let tight = shape.bytes(32_768, KvSplit::uniform(CacheType::F16)) * 2;
        let chosen = choose_kv_split(shape, 32_768, 2_740_000_000, tight, false);
        assert!(chosen.k.is_quantized(), "K should be quantised: {chosen:?}");
        assert_eq!(chosen.v, CacheType::F16, "V cannot be, without flash attention");
    }

    #[test]
    fn a_quantised_value_cache_is_never_chosen_without_flash_attention() {
        // llama.cpp returns a null context for this, so it must be unreachable
        // however tight memory gets.
        let shape = qwen_shape();
        for free in [64 * 1024 * 1024, GIB, 8 * GIB] {
            let chosen = choose_kv_split(shape, 32_768, 2_740_000_000, free, false);
            assert!(!chosen.v.is_quantized(), "free={free}: {chosen:?}");
            assert!(kv_split_allowed(chosen, false, shape));
        }
    }

    #[test]
    fn a_head_that_does_not_divide_the_block_size_is_not_quantised() {
        // With flash attention on, llama.cpp checks head width against the
        // block size and returns null if it does not divide. 80 does not.
        let odd = KvShape::new(32, 8, 80, 80);
        let chosen = choose_kv_split(odd, 32_768, 2_740_000_000, 64 * 1024 * 1024, true);
        assert!(kv_split_allowed(chosen, true, odd), "{chosen:?}");
    }

    #[test]
    fn the_mixed_pair_beats_a_uniform_one_of_the_same_size() {
        // q8_0/q4_0 is 13 bits against q5_1/q5_1 at 12 — close enough in size
        // to be a fair trade, and much better where it matters.
        let shape = qwen_shape();
        let mixed = shape.bytes(32_768, KvSplit { k: CacheType::Q8_0, v: CacheType::Q4_0 });
        let uniform = shape.bytes(32_768, KvSplit::uniform(CacheType::Q5_1));
        assert!(mixed < uniform * 110 / 100, "mixed {mixed} vs uniform {uniform}");
    }

    #[test]
    fn a_window_is_sized_against_both_halves_not_the_wider_one() {
        // Sizing everything against the wider half was the old approximation
        // and it under-sized the window whenever the halves differed.
        let shape = qwen_shape();
        let split = KvSplit { k: CacheType::Q8_0, v: CacheType::Q4_0 };
        let budget = shape.bytes(16_384, split);
        let fitted = fit_context_split(shape, 65_536, split, budget);
        assert!(fitted >= 16_000 && fitted <= 16_384, "got {fitted}");
        let pessimistic = fit_context(shape.total(), 65_536, CacheType::Q8_0, budget);
        assert!(fitted > pessimistic, "{fitted} should beat the widest-half guess {pessimistic}");
    }

    #[test]
    fn kv_size_matches_the_hand_computed_figure() {
        // 32 layers * 4 heads * (256+256) = 65536 elements = 128 KiB/token at f16.
        assert_eq!(qwen_elements(), 65_536);
        assert_eq!(kv_bytes(qwen_elements(), 1, CacheType::F16), 128 * 1024);
        // 4096 tokens of f16 is 512 MiB, which is what the GPU actually showed.
        assert_eq!(kv_bytes(qwen_elements(), 4096, CacheType::F16), 512 * 1024 * 1024);
    }

    #[test]
    fn quantised_types_keep_their_fractional_scale() {
        // q8_0 is 8.5 bits, not 8: truncating the scale would report 256 MiB.
        let q8 = kv_bytes(qwen_elements(), 4096, CacheType::Q8_0);
        assert_eq!(q8, 272 * 1024 * 1024);
        assert!(q8 > kv_bytes(qwen_elements(), 4096, CacheType::F16) / 2);
    }

    #[test]
    fn short_context_prefers_f16_for_speed() {
        // 5 GiB free after a 2.7 GB model: f16 fits easily and the cache is a
        // small fraction of the weight traffic, so there is nothing to buy by
        // quantising it.
        let t = choose_kv_type(qwen_elements(), 4096, 2_740_000_000, 5 * GIB, true);
        assert_eq!(t, CacheType::F16);
    }

    #[test]
    fn long_context_prefers_q8_even_when_f16_would_fit() {
        // At 32k the cache is read more per token than the weights are, so
        // halving it is a speed win. Given ample VRAM the fit check would have
        // happily returned f16 — this asserts the traffic rule overrides it.
        let free = 40 * GIB;
        assert!(kv_bytes(qwen_elements(), 32_768, CacheType::F16) <= free / 100 * 70);
        let t = choose_kv_type(qwen_elements(), 32_768, 2_740_000_000, free, true);
        assert_eq!(t, CacheType::Q8_0);
    }

    #[test]
    fn a_tight_card_steps_down_until_the_cache_fits() {
        // 2 GiB left cannot hold 16k of f16 (2 GiB exactly, before the 70%
        // share), so it must degrade to something that genuinely fits rather
        // than refusing to run.
        let free = 2 * GIB;
        assert!(kv_bytes(qwen_elements(), 16_384, CacheType::F16) > free / 100 * 70);
        let t = choose_kv_type(qwen_elements(), 16_384, 2_740_000_000, free, true);
        assert!(t.is_quantized(), "expected a quantised cache, got {t:?}");
        assert!(kv_bytes(qwen_elements(), 16_384, t) <= free / 100 * 70);
    }

    #[test]
    fn an_impossible_fit_still_names_the_smallest_cache() {
        // Nothing fits. Returning the smallest is the only chance of loading,
        // and is more useful than a panic.
        let t = choose_kv_type(qwen_elements(), 262_144, 2_740_000_000, 64 * 1024 * 1024, true);
        assert_eq!(t, CacheType::Q4_0);
    }

    #[test]
    fn choice_never_exceeds_the_vram_share_when_any_option_fits() {
        // Sweep contexts and card sizes: whenever the chosen type is not the
        // last-resort fallback, it must genuinely fit the budget.
        for ctx in [512u32, 4096, 16_384, 65_536] {
            for free_gib in [1u64, 2, 4, 8, 24] {
                let free = free_gib * GIB;
                let t = choose_kv_type(qwen_elements(), ctx, 2_740_000_000, free, true);
                if t != CacheType::Q4_0 {
                    assert!(
                        kv_bytes(qwen_elements(), ctx, t) <= free / 100 * 70,
                        "ctx {ctx} on {free_gib} GiB chose {t:?}, which does not fit"
                    );
                }
            }
        }
    }

    #[test]
    fn without_flash_attention_the_cache_stays_unquantised() {
        // llama.cpp cannot run a quantised K cache without flash attention.
        // Even at a context long enough that the traffic rule would otherwise
        // demand q8_0, the answer must stay f16.
        let t = choose_kv_type(qwen_elements(), 32_768, 2_740_000_000, 40 * GIB, false);
        assert_eq!(t, CacheType::F16);
        assert!(!t.is_quantized());
    }

    #[test]
    fn auto_is_the_default_and_round_trips_as_a_name() {
        assert_eq!(CacheType::default(), CacheType::Auto);
        assert_eq!("auto".parse::<CacheType>().unwrap(), CacheType::Auto);
    }

    #[test]
    fn a_model_with_no_weights_reported_does_not_force_quantisation() {
        // weight_bytes == 0 means "unknown"; the traffic rule needs a ratio and
        // must not fire on a divide-by-nothing.
        let t = choose_kv_type(qwen_elements(), 4096, 0, 8 * GIB, true);
        assert_eq!(t, CacheType::F16);
    }

    use super::*;

    #[test]
    fn a_context_that_fits_is_left_exactly_alone() {
        let e = qwen_elements();
        let budget = kv_bytes(e, 32_768, CacheType::F16) * 2;
        assert_eq!(fit_context(e, 32_768, CacheType::F16, budget), 32_768);
    }

    #[test]
    fn a_million_token_window_is_cut_to_what_memory_holds() {
        // The failure this exists for: a model advertising a 1M training
        // context on a 24 GB card. The cache alone would be hundreds of
        // gigabytes, and llama.cpp answers that with a null pointer.
        let e = qwen_elements();
        let budget = 16 * 1024 * 1024 * 1024;
        let fitted = fit_context(e, 1_048_576, CacheType::F16, budget);

        assert!(fitted < 1_048_576, "the request must not survive unchanged");
        assert!(fitted >= MIN_CONTEXT);
        assert!(
            kv_bytes(e, fitted, CacheType::F16) <= budget,
            "the whole point is that the result fits",
        );
    }

    #[test]
    fn the_fitted_context_is_the_largest_that_fits() {
        // Shrinking further than necessary costs the user memory they have.
        let e = qwen_elements();
        let budget = 8 * 1024 * 1024 * 1024;
        let fitted = fit_context(e, 512 * 1024, CacheType::Q8_0, budget);
        assert!(kv_bytes(e, fitted, CacheType::Q8_0) <= budget);
        assert!(
            kv_bytes(e, fitted + CONTEXT_GRAIN, CacheType::Q8_0) > budget,
            "one more grain should not have fitted",
        );
    }

    #[test]
    fn a_fitted_context_is_a_round_number() {
        let e = qwen_elements();
        let fitted = fit_context(e, 1_000_000, CacheType::F16, 3 * 1024 * 1024 * 1024);
        assert_eq!(fitted % CONTEXT_GRAIN, 0, "got {fitted}");
    }

    #[test]
    fn an_unmeasurable_budget_does_not_shrink_anything() {
        // Clamping on a guess would shrink windows that would have worked.
        let e = qwen_elements();
        assert_eq!(fit_context(e, 262_144, CacheType::F16, 0), 262_144);
    }

    #[test]
    fn a_hopeless_budget_still_yields_a_usable_floor() {
        let e = qwen_elements();
        assert_eq!(fit_context(e, 131_072, CacheType::F16, 1024), MIN_CONTEXT);
    }

    #[test]
    fn the_budget_follows_the_layers() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let all_gpu = kv_budget(24 * GIB, 64 * GIB, 32, 32);
        let all_cpu = kv_budget(24 * GIB, 64 * GIB, 0, 32);
        let half = kv_budget(24 * GIB, 64 * GIB, 16, 32);

        assert!(all_cpu > all_gpu, "the larger pool should give the larger budget");
        assert!(half > all_gpu && half < all_cpu, "a split model draws on both");
        assert!(all_gpu < 24 * GIB, "compute buffers need room too");
    }

    #[test]
    fn cache_type_parsing_and_sizing() {
        assert_eq!("q8_0".parse::<CacheType>().unwrap(), CacheType::Q8_0);
        assert!(CacheType::Q8_0.is_quantized());
        assert!(!CacheType::F16.is_quantized());
        assert!(CacheType::Q8_0.bits() < CacheType::F16.bits());
    }

    #[test]
    fn moe_offload_parsing() {
        assert_eq!("auto".parse::<MoeOffload>().unwrap(), MoeOffload::AUTO);
        assert_eq!("all".parse::<MoeOffload>().unwrap(), MoeOffload::ALL);
        assert_eq!("12".parse::<MoeOffload>().unwrap(), MoeOffload::Layers(12));
        assert!("some".parse::<MoeOffload>().is_err());
    }

    #[test]
    fn speculative_config_round_trips() {
        for s in [
            Speculative::Off,
            Speculative::Mtp,
            Speculative::Ngram,
            Speculative::Auto,
            Speculative::Draft { model: "qwen4:0.5b".into(), gpu_layers: Some(99) },
        ] {
            let j = serde_json::to_string(&s).unwrap();
            assert_eq!(serde_json::from_str::<Speculative>(&j).unwrap(), s);
        }
    }

    #[test]
    fn moe_offload_round_trips_through_toml() {
        #[derive(Serialize, Deserialize)]
        struct W { v: MoeOffload }
        for v in [MoeOffload::AUTO, MoeOffload::OFF, MoeOffload::ALL, MoeOffload::Layers(7)] {
            let s = toml::to_string(&W { v }).unwrap();
            assert_eq!(toml::from_str::<W>(&s).unwrap().v, v, "round trip failed for {s}");
        }
    }

    #[test]
    fn a_backend_without_mixed_kernels_is_never_handed_a_mixed_pair() {
        let shape = KvShape::new(32, 8, 128, 128);
        for budget in [64 * 1024 * 1024, 512 * 1024 * 1024, 8 * GIB] {
            let chosen = choose_kv_split_for(shape, 32_768, 2_740_000_000, budget, true, false);
            assert!(flash_has_kernel(chosen, false), "{chosen:?} at {budget}");
        }
        let ladder = kv_ladder_for(true, shape, false);
        assert!(ladder.iter().all(|s| s.k == s.v), "{ladder:?}");
        assert!(ladder.contains(&KvSplit::uniform(CacheType::Q4_0)));
    }

    #[test]
    fn a_backend_with_every_kernel_keeps_the_full_ladder() {
        let shape = KvShape::new(32, 8, 128, 128);
        assert_eq!(kv_ladder_for(true, shape, true), kv_ladder(true, shape));
        assert!(kv_ladder(true, shape).iter().any(|s| s.k != s.v));
    }

    #[test]
    fn the_native_kernels_are_exactly_the_same_type_ones() {
        let q = |k, v| KvSplit { k, v };
        assert!(flash_has_kernel(q(CacheType::Q8_0, CacheType::Q8_0), false));
        assert!(flash_has_kernel(q(CacheType::Q4_0, CacheType::Q4_0), false));
        assert!(flash_has_kernel(q(CacheType::F16, CacheType::F16), false));
        assert!(!flash_has_kernel(q(CacheType::Q8_0, CacheType::Q4_0), false));
        assert!(!flash_has_kernel(q(CacheType::Q5_1, CacheType::Q5_1), false));
    }

}
