//! Layered runtime options.
//!
//! Every knob is an `Option`, so a layer that says nothing overrides nothing.
//! Layers merge lowest-precedence first:
//!
//! ```text
//! built-in defaults  <  configs/config.toml  <  model manifest  <  CLI flags
//! ```
//!
//! [`Options::resolve`] collapses the merged stack into concrete values.

use crate::accel::{CacheType, MoeOffload, PrefixReuse, Speculative, SpeculativeTuning};
use serde::{Deserialize, Serialize};

/// How many transformer layers to place on the GPU.
///
/// Serialises as either the string `"auto"` / `"off"` or a bare integer, so
/// `gpu_layers = 20`, `gpu_layers = "off"` and `gpu_layers = "auto"` are all
/// valid in `config.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GpuLayers {
    Count(u32),
    #[serde(with = "gpu_keyword")]
    Keyword(GpuKeyword),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKeyword {
    /// Offload as many layers as the device can hold.
    Auto,
    /// Pure CPU inference. Equivalent to `Count(0)`.
    Off,
}

impl GpuLayers {
    pub const AUTO: Self = Self::Keyword(GpuKeyword::Auto);
    pub const OFF: Self = Self::Keyword(GpuKeyword::Off);

    /// Translate into the integer llama.cpp expects, where a very large value
    /// means "everything". `Auto` is resolved upstream against real VRAM; if it
    /// reaches here unresolved we ask for all layers and let llama.cpp clamp.
    pub fn to_llama(self) -> i32 {
        match self {
            Self::Count(n) => n.min(i32::MAX as u32) as i32,
            Self::Keyword(GpuKeyword::Off) => 0,
            Self::Keyword(GpuKeyword::Auto) => i32::MAX,
        }
    }

    pub fn is_cpu_only(self) -> bool {
        matches!(self, Self::Count(0) | Self::Keyword(GpuKeyword::Off))
    }
}

impl std::str::FromStr for GpuLayers {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "max" | "all" => Ok(Self::AUTO),
            "off" | "none" | "cpu" => Ok(Self::OFF),
            other => other
                .parse::<u32>()
                .map(Self::Count)
                .map_err(|_| format!("expected a layer count, \"auto\", or \"off\"; got {other:?}")),
        }
    }
}

impl std::fmt::Display for GpuLayers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count(n) => write!(f, "{n}"),
            Self::Keyword(GpuKeyword::Auto) => f.write_str("auto"),
            Self::Keyword(GpuKeyword::Off) => f.write_str("off"),
        }
    }
}

mod gpu_keyword {
    use super::GpuKeyword;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &GpuKeyword, s: S) -> Result<S::Ok, S::Error> {
        match v {
            GpuKeyword::Auto => "auto",
            GpuKeyword::Off => "off",
        }
        .serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<GpuKeyword, D::Error> {
        let s = String::deserialize(d)?;
        match s.to_ascii_lowercase().as_str() {
            "auto" | "max" | "all" => Ok(GpuKeyword::Auto),
            "off" | "none" | "cpu" => Ok(GpuKeyword::Off),
            other => Err(serde::de::Error::custom(format!(
                "expected \"auto\" or \"off\", got {other:?}"
            ))),
        }
    }
}

/// Whether a reasoning model should be allowed to think.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    /// Think if the model advertises the capability.
    #[default]
    Auto,
    /// Force thinking on, and render the reasoning.
    On,
    /// Suppress reasoning so the model behaves like a plain chat model.
    Off,
}

/// How long a reasoning model may think before it must answer.
///
/// Enforced as a token budget on the reasoning block rather than asked for in
/// the prompt: a model told to "think briefly" frequently does not, while a
/// model whose `</think>` is written for it has no choice.
///
/// The block is closed properly rather than cut, so the output stays
/// well-formed — but the *thought* is still interrupted, and a model stopped
/// halfway answers from an argument it had not finished making. The budgets
/// below are therefore backstops against a reasoning loop, not a style
/// control. Measured against Qwen3.5-4B on a two-train word problem, the old
/// budgets bound on every single run: low stopped it at 257 tokens of 256,
/// medium at 1025 of 1024. Left alone the same model wanted 1177, and ollama
/// on the same weights ran past 4000. Medium is the default, so out of the box
/// ozgent was cutting every answer's reasoning roughly in half.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Enough to plan a couple of steps.
    Low,
    /// Room to work through a problem. The default.
    #[default]
    Medium,
    /// Effectively unbounded; the model stops when it is done.
    High,
}

impl ReasoningEffort {
    /// Tokens the reasoning block may spend; zero means no limit at all.
    ///
    /// `High` is genuinely unbounded — the model stops when it is done, and
    /// the only remaining limits are `max_tokens` and the context window. A
    /// number here, however large, is a promise to interrupt a model that was
    /// still making progress.
    ///
    /// `Low` still means low, but 256 tokens was not "brief", it was
    /// mid-sentence. These leave room to finish a thought.
    pub fn budget(self) -> u32 {
        match self {
            Self::Low => 2048,
            Self::Medium => 8192,
            Self::High => 0,
        }
    }

    /// Whether this level lets the model decide for itself when to stop.
    pub fn is_unbounded(self) -> bool {
        self.budget() == 0
    }
}

impl std::str::FromStr for ReasoningEffort {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low" | "min" | "minimal" => Ok(Self::Low),
            "medium" | "med" | "default" => Ok(Self::Medium),
            "high" | "max" => Ok(Self::High),
            other => Err(format!("unknown reasoning effort {other:?}; expected low, medium, or high")),
        }
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        })
    }
}

impl std::str::FromStr for ThinkingMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "true" | "yes" => Ok(Self::On),
            "off" | "false" | "no" => Ok(Self::Off),
            other => Err(format!("expected auto, on, or off; got {other:?}")),
        }
    }
}

/// One layer of the options stack. Absent fields defer to lower layers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    // --- model loading ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<GpuLayers>,
    /// Accepts `8192` or `"8k"`; see `crate::tokens`.
    #[serde(
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::tokens::deserialize_optional"
    )]
    pub context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    /// Physical micro-batch. `None` leaves llama.cpp's default.
    pub ubatch: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_gpu: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mmap: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_mlock: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attention: Option<bool>,

    // --- acceleration ---
    /// Keep routed experts of the first N layers in system RAM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_moe: Option<MoeOffload>,
    /// A control-vector GGUF to steer generation with, and how hard.
    ///
    /// Steering happens inside the forward pass, so unlike a system prompt it
    /// costs no context and the model cannot decide to ignore it.
    pub control_vector: Option<std::path::PathBuf>,
    pub control_strength: Option<f32>,
    /// Quantisation of the K cache. Requires flash attention when quantised.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_k: Option<CacheType>,
    /// Quantisation of the V cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_v: Option<CacheType>,
    /// Speculative decoding strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative: Option<Speculative>,
    /// Shared speculative tuning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculative_tuning: Option<SpeculativeTuning>,
    /// KV-cache reuse across turns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_reuse: Option<PrefixReuse>,

    // --- sampling ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    /// Accepts `2048` or `"2k"`; 0 means until the model stops.
    #[serde(
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::tokens::deserialize_optional"
    )]
    pub max_tokens: Option<u32>,

    // --- session ---
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingMode>,
    /// How long a reasoning model may think. Ignored when thinking is off.
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<bool>,
}

impl Options {
    /// Overlay `higher` on top of `self`, field by field.
    pub fn merge(mut self, higher: &Options) -> Self {
        macro_rules! take {
            ($($f:ident),* $(,)?) => {$(
                if higher.$f.is_some() { self.$f = higher.$f.clone(); }
            )*};
        }
        // Every field must be listed. One left out is silently dropped from
        // every merge — the flag parses, the config accepts it, and nothing
        // happens. `ubatch` and `reasoning_effort` were both added to the
        // struct and forgotten here, and both looked like features that did
        // not work rather than like a bug in this list.
        take!(
            gpu_layers, context_length, batch_size, ubatch, threads, main_gpu, use_mmap, use_mlock,
            flash_attention, cpu_moe, control_vector, control_strength,
            cache_type_k, cache_type_v, speculative,
            speculative_tuning, prefix_reuse, temperature, top_p, top_k, min_p, repeat_penalty, repeat_last_n,
            seed, max_tokens, system_prompt, thinking, reasoning_effort, tools,
        );
        self
    }

    /// Collapse into concrete values, filling any remaining gaps with built-in
    /// defaults.
    pub fn resolve(&self) -> Resolved {
        Resolved {
            gpu_layers: self.gpu_layers.unwrap_or(GpuLayers::AUTO),
            context_length: self.context_length.unwrap_or(4096),
            batch_size: self.batch_size.unwrap_or(512),
            ubatch: self.ubatch,
            // 0 lets llama.cpp pick based on the physical core count.
            threads: self.threads.unwrap_or(0),
            main_gpu: self.main_gpu.unwrap_or(0),
            use_mmap: self.use_mmap.unwrap_or(true),
            use_mlock: self.use_mlock.unwrap_or(false),
            flash_attention: self.flash_attention.unwrap_or(true),
            cpu_moe: self.cpu_moe.unwrap_or(MoeOffload::AUTO),
            control_vector: self.control_vector.clone(),
            // 1.0 applies the vector as trained. Not clamped, because the
            // usable range depends entirely on the vector: measured against a
            // deliberately meaningless direction, output drifted at 0.02-0.05,
            // restructured at 0.1 and collapsed by 0.3. A trained direction
            // tolerates far more, so a fixed ceiling would be wrong either way.
            control_strength: self.control_strength.unwrap_or(1.0),
            // q8_0 halves cache VRAM for no measurable quality cost, which is
            // the difference between a usable and an unusable context on a
            // small card. Flash attention (on by default) makes it legal.
            cache_type_k: self.cache_type_k.unwrap_or(CacheType::Auto),
            cache_type_v: self.cache_type_v.unwrap_or(CacheType::Auto),
            speculative: self.speculative.clone().unwrap_or_default(),
            speculative_tuning: self.speculative_tuning.clone().unwrap_or_default(),
            prefix_reuse: self.prefix_reuse.unwrap_or_default(),
            temperature: self.temperature.unwrap_or(0.8),
            top_p: self.top_p.unwrap_or(0.95),
            top_k: self.top_k.unwrap_or(40),
            min_p: self.min_p.unwrap_or(0.05),
            repeat_penalty: self.repeat_penalty.unwrap_or(1.1),
            repeat_last_n: self.repeat_last_n.unwrap_or(64),
            seed: self.seed,
            // 0 means "until EOS or the context fills".
            max_tokens: self.max_tokens.unwrap_or(0),
            system_prompt: self.system_prompt.clone(),
            thinking: self.thinking.unwrap_or_default(),
            reasoning_effort: self.reasoning_effort.unwrap_or_default(),
            tools: self.tools.unwrap_or(true),
        }
    }
}

impl Resolved {
    /// Take the settings that belong to a single turn from `src`.
    ///
    /// A server holds one session across many requests, so the turn's own
    /// sampling has to replace the previous turn's or the first caller's
    /// temperature, seed and reasoning budget stand for everyone after them.
    ///
    /// Load-time settings are deliberately not among these. Layer placement,
    /// context length, cache types, expert offload and the control vector are
    /// fixed when the weights are loaded and cannot change under a live KV
    /// cache; copying them here would claim a change that never happened.
    pub fn adopt_per_turn(&mut self, src: &Self) {
        self.temperature = src.temperature;
        self.top_p = src.top_p;
        self.top_k = src.top_k;
        self.min_p = src.min_p;
        self.repeat_penalty = src.repeat_penalty;
        self.repeat_last_n = src.repeat_last_n;
        self.seed = src.seed;
        self.max_tokens = src.max_tokens;
        self.system_prompt = src.system_prompt.clone();
        self.thinking = src.thinking;
        self.reasoning_effort = src.reasoning_effort;
        self.tools = src.tools;
        self.speculative = src.speculative.clone();
        self.speculative_tuning = src.speculative_tuning.clone();
        self.prefix_reuse = src.prefix_reuse;
    }

    /// llama.cpp cannot use a quantised K cache without flash attention.
    /// Rather than fail at load time, report whether the combination the user
    /// asked for had to be adjusted.
    pub fn kv_needs_flash_attention(&self) -> bool {
        self.cache_type_k.is_quantized() && !self.flash_attention
    }
}

/// Fully-resolved options with no remaining ambiguity.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub gpu_layers: GpuLayers,
    pub context_length: u32,
    pub batch_size: u32,
    pub ubatch: Option<u32>,
    pub threads: u32,
    pub main_gpu: u32,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub flash_attention: bool,
    pub cpu_moe: MoeOffload,
    pub control_vector: Option<std::path::PathBuf>,
    pub control_strength: f32,
    pub cache_type_k: CacheType,
    pub cache_type_v: CacheType,
    pub speculative: Speculative,
    pub speculative_tuning: SpeculativeTuning,
    pub prefix_reuse: PrefixReuse,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: u32,
    pub seed: Option<u32>,
    pub max_tokens: u32,
    pub system_prompt: Option<String>,
    pub thinking: ThinkingMode,
    pub reasoning_effort: ReasoningEffort,
    pub tools: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_option_survives_a_merge() {
        // Guards the explicit field list in `merge`: a field added to Options
        // and forgotten there is silently dropped, which looks exactly like a
        // flag that does nothing.
        let higher = Options {
            ubatch: Some(256),
            reasoning_effort: Some(ReasoningEffort::Low),
            ..Default::default()
        };
        let merged = Options::default().merge(&higher);
        assert_eq!(merged.ubatch, Some(256), "ubatch was dropped by merge");
        assert_eq!(
            merged.reasoning_effort,
            Some(ReasoningEffort::Low),
            "reasoning_effort was dropped by merge"
        );
    }

    #[test]
    fn accel_defaults_are_the_fast_ones() {
        let r = Options::default().resolve();
        // Auto, not a fixed type: the right cache depends on how much VRAM is
        // left after the weights and on how long the context is. See
        // `accel::choose_kv_type`.
        assert_eq!(r.cache_type_k, CacheType::Auto, "KV cache is sized at load time");
        assert_eq!(r.cpu_moe, MoeOffload::AUTO);
        assert_eq!(r.speculative, Speculative::Auto);
        assert!(r.flash_attention);
        assert!(!r.kv_needs_flash_attention(), "defaults must be self-consistent");
    }

    #[test]
    fn quantised_kv_without_flash_attention_is_flagged() {
        // Explicitly asking for a quantised cache with flash attention off is
        // a real conflict and must be reported.
        let o = Options {
            flash_attention: Some(false),
            cache_type_k: Some(CacheType::Q8_0),
            ..Default::default()
        };
        assert!(o.resolve().kv_needs_flash_attention());

        // The default is `auto`, which resolves to f16 in that situation
        // rather than conflicting, so it must not be flagged.
        let auto = Options { flash_attention: Some(false), ..Default::default() };
        assert!(!auto.resolve().kv_needs_flash_attention());

        let ok = Options {
            flash_attention: Some(false),
            cache_type_k: Some(CacheType::F16),
            ..Default::default()
        };
        assert!(!ok.resolve().kv_needs_flash_attention());
    }

    #[test]
    fn accel_fields_participate_in_merge() {
        let low = Options { cpu_moe: Some(MoeOffload::OFF), ..Default::default() };
        let high = Options { cpu_moe: Some(MoeOffload::Layers(4)), ..Default::default() };
        assert_eq!(low.merge(&high).cpu_moe, Some(MoeOffload::Layers(4)));
    }

    #[test]
    fn higher_layer_wins_and_absent_fields_defer() {
        let low = Options { temperature: Some(0.1), top_k: Some(10), ..Default::default() };
        let high = Options { temperature: Some(0.9), ..Default::default() };
        let merged = low.merge(&high);
        assert_eq!(merged.temperature, Some(0.9));
        assert_eq!(merged.top_k, Some(10), "unset field must not clobber");
    }

    #[test]
    fn gpu_layers_accepts_keywords_and_counts() {
        assert_eq!("off".parse::<GpuLayers>().unwrap(), GpuLayers::OFF);
        assert_eq!("auto".parse::<GpuLayers>().unwrap(), GpuLayers::AUTO);
        assert_eq!("24".parse::<GpuLayers>().unwrap(), GpuLayers::Count(24));
        assert!("twenty".parse::<GpuLayers>().is_err());
    }

    #[test]
    fn cpu_only_detection() {
        assert!(GpuLayers::OFF.is_cpu_only());
        assert!(GpuLayers::Count(0).is_cpu_only());
        assert!(!GpuLayers::Count(1).is_cpu_only());
        assert!(!GpuLayers::AUTO.is_cpu_only());
        assert_eq!(GpuLayers::OFF.to_llama(), 0);
    }

    #[test]
    fn gpu_layers_round_trips_through_toml() {
        #[derive(Serialize, Deserialize)]
        struct W { v: GpuLayers }
        for v in [GpuLayers::AUTO, GpuLayers::OFF, GpuLayers::Count(33)] {
            let s = toml::to_string(&W { v }).unwrap();
            assert_eq!(toml::from_str::<W>(&s).unwrap().v, v, "round trip failed for {s}");
        }
    }

    #[test]
    fn every_per_turn_option_is_adopted() {
        // The sibling of `every_option_survives_a_merge`. A per-turn field
        // left out of `adopt_per_turn` means one client's request silently
        // runs under another client's setting, which is invisible until two
        // people share a model.
        let mut dst = Options::default().resolve();
        let src = Resolved {
            temperature: 0.123,
            top_p: 0.456,
            top_k: 7,
            min_p: 0.089,
            repeat_penalty: 1.23,
            repeat_last_n: 99,
            seed: Some(4242),
            max_tokens: 777,
            system_prompt: Some("per turn".into()),
            thinking: ThinkingMode::Off,
            reasoning_effort: ReasoningEffort::High,
            tools: !Options::default().resolve().tools,
            ..Options::default().resolve()
        };
        dst.adopt_per_turn(&src);

        assert_eq!(dst.temperature, 0.123, "temperature");
        assert_eq!(dst.top_p, 0.456, "top_p");
        assert_eq!(dst.top_k, 7, "top_k");
        assert_eq!(dst.min_p, 0.089, "min_p");
        assert_eq!(dst.repeat_penalty, 1.23, "repeat_penalty");
        assert_eq!(dst.repeat_last_n, 99, "repeat_last_n");
        assert_eq!(dst.seed, Some(4242), "seed");
        assert_eq!(dst.max_tokens, 777, "max_tokens");
        assert_eq!(dst.system_prompt.as_deref(), Some("per turn"), "system_prompt");
        assert_eq!(dst.thinking, ThinkingMode::Off, "thinking");
        assert_eq!(dst.reasoning_effort, ReasoningEffort::High, "reasoning_effort");
        assert_eq!(dst.tools, src.tools, "tools");
    }

    #[test]
    fn adopting_a_turn_leaves_load_time_settings_alone() {
        // These cannot change under a live KV cache. Copying them would report
        // a context length or layer split the loaded model does not have.
        let mut dst = Resolved { context_length: 8192, threads: 6, ..Options::default().resolve() };
        let src = Resolved { context_length: 32768, threads: 1, ..Options::default().resolve() };
        dst.adopt_per_turn(&src);
        assert_eq!(dst.context_length, 8192, "context length must survive");
        assert_eq!(dst.threads, 6, "thread count must survive");
    }

    #[test]
    fn high_effort_never_interrupts_the_model() {
        // A budget is a promise to cut a model off. At the level whose whole
        // meaning is "think as long as you need", there must not be one.
        assert!(ReasoningEffort::High.is_unbounded());
        assert_eq!(ReasoningEffort::High.budget(), 0);
    }

    #[test]
    fn the_lower_levels_still_bound_and_still_rank() {
        assert!(ReasoningEffort::Low.budget() > 0);
        assert!(ReasoningEffort::Medium.budget() > ReasoningEffort::Low.budget());
        // Room to finish a sentence: the old 256 stopped Qwen3.5 mid-clause.
        assert!(ReasoningEffort::Low.budget() >= 1024);
    }
}
