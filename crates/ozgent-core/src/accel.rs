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

/// Fraction of free VRAM the KV cache may claim.
///
/// The rest is compute buffers and llama.cpp scratch, which are not predictable
/// from metadata; an over-tight fit fails at load rather than degrading.
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
}
