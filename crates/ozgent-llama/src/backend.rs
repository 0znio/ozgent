//! Backend initialisation and device selection.
//!
//! llama.cpp registers whichever backends were compiled in, then reports the
//! devices each one found. ozgent surfaces that directly rather than guessing:
//! `ozgent doctor` can say *which* backend is being used and how much VRAM is
//! free, which is the first thing anyone needs when a model is unexpectedly
//! slow.

use ozgent_core::{GpuLayers, accel::MoeOffload};

/// A compute device llama.cpp can place layers on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    pub index: usize,
    /// Short name, e.g. `CUDA0`.
    pub name: String,
    /// Human description, e.g. `NVIDIA GeForce RTX 5050 Laptop GPU`.
    pub description: String,
    /// Which backend owns it: `CUDA`, `Vulkan`, `Metal`, `CPU`.
    pub backend: String,
    pub memory_total: usize,
    pub memory_free: usize,
}

impl Device {
    pub fn is_gpu(&self) -> bool {
        !self.backend.eq_ignore_ascii_case("CPU")
    }

    pub fn free_gib(&self) -> f64 {
        self.memory_free as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn total_gib(&self) -> f64 {
        self.memory_total as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// How many layers fit, given per-layer cost and what must be reserved.
///
/// Kept separate from any llama.cpp call so the arithmetic — which is where
/// out-of-memory bugs actually live — is testable without a GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitEstimate {
    pub layers: u32,
    /// Experts to evict, when the model is a mixture of experts.
    pub cpu_moe: Option<u32>,
    pub bytes_used: u64,
}

/// Decide how much of a model to offload.
///
/// The order matters. Evicting routed experts frees far more VRAM per lost
/// token/sec than dropping whole layers does, because experts are most of the
/// weight but only a fraction of the work per token — so experts go first, and
/// layers are only dropped once that is exhausted.
///
/// `expert_bytes_per_layer` is the portion of each layer that is routed
/// experts, and is zero for a dense model. `overhead_bytes` covers the KV
/// cache and compute buffers, which must stay resident.
pub fn fit_to_vram(
    free_bytes: u64,
    total_layers: u32,
    bytes_per_layer: u64,
    expert_bytes_per_layer: u64,
    overhead_bytes: u64,
) -> FitEstimate {
    // Leave headroom: llama.cpp allocates scratch beyond what we can predict,
    // and an over-tight fit fails at load time rather than degrading.
    let budget = free_bytes.saturating_sub(overhead_bytes).saturating_mul(92) / 100;

    if bytes_per_layer == 0 || total_layers == 0 {
        return FitEstimate { layers: 0, cpu_moe: None, bytes_used: 0 };
    }

    let all_layers_cost = bytes_per_layer * total_layers as u64;
    if all_layers_cost <= budget {
        return FitEstimate { layers: total_layers, cpu_moe: None, bytes_used: all_layers_cost };
    }

    // Try keeping every layer resident but evicting experts from as few as
    // possible, which is the smallest loss that still fits.
    if expert_bytes_per_layer > 0 {
        let dense_cost = bytes_per_layer - expert_bytes_per_layer;
        for evicted in 1..=total_layers {
            let cost = dense_cost * evicted as u64
                + bytes_per_layer * (total_layers - evicted) as u64;
            if cost <= budget {
                return FitEstimate {
                    layers: total_layers,
                    cpu_moe: Some(evicted),
                    bytes_used: cost,
                };
            }
        }
    }

    // Nothing else for it: drop whole layers.
    let layers = (budget / bytes_per_layer).min(total_layers as u64) as u32;
    FitEstimate {
        layers,
        cpu_moe: expert_bytes_per_layer.gt(&0).then_some(total_layers),
        bytes_used: bytes_per_layer * layers as u64,
    }
}

/// Resolve `auto` settings against the hardware actually present.
pub fn resolve_auto(
    gpu_layers: GpuLayers,
    cpu_moe: MoeOffload,
    device: Option<&Device>,
    total_layers: u32,
    bytes_per_layer: u64,
    expert_bytes_per_layer: u64,
    overhead_bytes: u64,
) -> (u32, u32) {
    let explicit_layers = match gpu_layers {
        GpuLayers::Count(n) => Some(n.min(total_layers)),
        GpuLayers::Keyword(ozgent_core::options::GpuKeyword::Off) => Some(0),
        GpuLayers::Keyword(ozgent_core::options::GpuKeyword::Auto) => None,
    };

    // With no GPU, or with the GPU switched off, nothing is offloaded and
    // expert eviction is meaningless.
    let Some(device) = device.filter(|d| d.is_gpu()) else {
        return (0, 0);
    };
    if explicit_layers == Some(0) {
        return (0, 0);
    }

    let estimate = fit_to_vram(
        device.memory_free as u64,
        total_layers,
        bytes_per_layer,
        expert_bytes_per_layer,
        overhead_bytes,
    );

    let layers = explicit_layers.unwrap_or(estimate.layers);
    let moe = match cpu_moe {
        MoeOffload::Layers(n) => n,
        MoeOffload::Keyword(ozgent_core::accel::MoeKeyword::Off) => 0,
        MoeOffload::Keyword(ozgent_core::accel::MoeKeyword::All) => total_layers,
        MoeOffload::Keyword(ozgent_core::accel::MoeKeyword::Auto) => {
            estimate.cpu_moe.unwrap_or(0)
        }
    };
    (layers, moe)
}

#[cfg(feature = "llama")]
mod real {
    use super::Device;

    /// Enumerate every device the compiled-in backends found.
    pub fn devices() -> Vec<Device> {
        // Listing devices brings the backends up, which prints CUDA's banner;
        // routed into ozgent's log like everything else llama.cpp says.
        crate::llamalog::capture();
        llama_cpp_2::list_llama_ggml_backend_devices()
            .into_iter()
            .map(|d| Device {
                index: d.index,
                name: d.name,
                description: d.description,
                backend: d.backend,
                memory_total: d.memory_total,
                memory_free: d.memory_free,
            })
            .collect()
    }

    /// The GPU with the most free memory, or `None` when only CPU is present.
    pub fn best_gpu() -> Option<Device> {
        devices()
            .into_iter()
            .filter(Device::is_gpu)
            .max_by_key(|d| d.memory_free)
    }

    pub fn supports_gpu_offload() -> bool {
        llama_cpp_2::llama_backend::LlamaBackend::init()
            .map(|b| b.supports_gpu_offload())
            .unwrap_or(false)
    }
}

#[cfg(feature = "llama")]
pub use real::{best_gpu, devices, supports_gpu_offload};

#[cfg(not(feature = "llama"))]
pub fn devices() -> Vec<Device> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn everything_fits_when_there_is_room() {
        let f = fit_to_vram(24 * GIB, 32, 200 * 1024 * 1024, 0, GIB);
        assert_eq!(f.layers, 32);
        assert_eq!(f.cpu_moe, None, "no need to evict experts when it all fits");
    }

    #[test]
    fn experts_are_evicted_before_layers_are_dropped() {
        // A dense-heavy MoE that does not fit whole: the right answer keeps
        // every layer's attention on the GPU and moves experts out.
        let per_layer = 600 * 1024 * 1024;
        let expert = 500 * 1024 * 1024;
        let f = fit_to_vram(8 * GIB, 32, per_layer, expert, GIB);

        assert_eq!(f.layers, 32, "layers should stay resident");
        assert!(f.cpu_moe.is_some(), "experts should be evicted instead");
        assert!(f.cpu_moe.unwrap() > 0 && f.cpu_moe.unwrap() <= 32);
    }

    #[test]
    fn the_smallest_sufficient_eviction_is_chosen() {
        // Throughput against cpu_moe is V-shaped, so over-evicting is pure
        // loss. The estimate must not evict more than necessary.
        let per_layer = 400 * 1024 * 1024;
        let expert = 300 * 1024 * 1024;
        let f = fit_to_vram(8 * GIB, 32, per_layer, expert, GIB);
        let evicted = f.cpu_moe.expect("should evict");

        let dense = per_layer - expert;
        let one_less = dense * (evicted - 1) as u64 + per_layer * (32 - evicted + 1) as u64;
        let budget = (8 * GIB - GIB) * 92 / 100;
        assert!(one_less > budget, "evicting {} would have sufficed", evicted - 1);
    }

    #[test]
    fn layers_are_dropped_only_when_eviction_is_not_enough() {
        // A dense model has no experts to evict.
        let f = fit_to_vram(2 * GIB, 32, 500 * 1024 * 1024, 0, GIB);
        assert!(f.layers < 32, "must offload fewer layers");
        assert_eq!(f.cpu_moe, None);
    }

    #[test]
    fn headroom_is_reserved_rather_than_filling_vram_exactly() {
        let free = 8 * GIB;
        let f = fit_to_vram(free, 100, 100 * 1024 * 1024, 0, 0);
        assert!(
            f.bytes_used < free,
            "an exact fit fails at load time; got {} of {free}",
            f.bytes_used
        );
    }

    #[test]
    fn no_gpu_means_nothing_is_offloaded() {
        let (layers, moe) = resolve_auto(
            GpuLayers::AUTO, MoeOffload::AUTO, None, 32, GIB, 0, GIB,
        );
        assert_eq!((layers, moe), (0, 0));
    }

    #[test]
    fn a_cpu_device_is_not_treated_as_a_gpu() {
        let cpu = Device {
            index: 0,
            name: "CPU".into(),
            description: "host".into(),
            backend: "CPU".into(),
            memory_total: 64 * GIB as usize,
            memory_free: 32 * GIB as usize,
        };
        assert!(!cpu.is_gpu());
        let (layers, _) = resolve_auto(
            GpuLayers::AUTO, MoeOffload::AUTO, Some(&cpu), 32, GIB, 0, GIB,
        );
        assert_eq!(layers, 0, "the CPU backend must not be offloaded to");
    }

    #[test]
    fn gpu_layers_off_wins_over_available_hardware() {
        let gpu = Device {
            index: 0,
            name: "CUDA0".into(),
            description: "big card".into(),
            backend: "CUDA".into(),
            memory_total: 24 * GIB as usize,
            memory_free: 24 * GIB as usize,
        };
        let (layers, moe) = resolve_auto(
            GpuLayers::OFF, MoeOffload::AUTO, Some(&gpu), 32, GIB, 0, GIB,
        );
        assert_eq!((layers, moe), (0, 0), "--no-gpu must mean no GPU");
    }

    #[test]
    fn an_explicit_layer_count_is_respected_and_clamped() {
        let gpu = Device {
            index: 0,
            name: "CUDA0".into(),
            description: "card".into(),
            backend: "CUDA".into(),
            memory_total: 24 * GIB as usize,
            memory_free: 24 * GIB as usize,
        };
        let (layers, _) = resolve_auto(
            GpuLayers::Count(10), MoeOffload::OFF, Some(&gpu), 32, GIB / 4, 0, GIB,
        );
        assert_eq!(layers, 10);

        let (clamped, _) = resolve_auto(
            GpuLayers::Count(999), MoeOffload::OFF, Some(&gpu), 32, GIB / 4, 0, GIB,
        );
        assert_eq!(clamped, 32, "cannot offload more layers than the model has");
    }

    #[test]
    fn moe_all_evicts_every_expert() {
        let gpu = Device {
            index: 0,
            name: "CUDA0".into(),
            description: "card".into(),
            backend: "CUDA".into(),
            memory_total: 8 * GIB as usize,
            memory_free: 8 * GIB as usize,
        };
        let (_, moe) = resolve_auto(
            GpuLayers::AUTO, MoeOffload::ALL, Some(&gpu), 48, GIB / 8, GIB / 16, GIB,
        );
        assert_eq!(moe, 48);
    }

    #[test]
    fn zero_sized_models_do_not_divide_by_zero() {
        let f = fit_to_vram(8 * GIB, 0, 0, 0, 0);
        assert_eq!(f.layers, 0);
    }
}

/// What of a model to put on the GPU, decided against the memory that is
/// actually free right now.
///
/// Read from the driver at load time rather than tracked, which matters once
/// more than one model can be resident: the figure then accounts for the other
/// models, and for anything else on the card — a game, a notebook, a second
/// ozgent. Bookkeeping of our own would drift from all three.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plan {
    /// Layers to offload. Every layer when the model fits.
    pub layers: u32,
    /// Layers to evict routed experts from. Zero on a dense model.
    pub experts: u32,
    pub total_layers: u32,
    /// Free device memory the decision was made against.
    pub free_bytes: u64,
}

impl Plan {
    /// Everything on the GPU, for when there is nothing to weigh up.
    fn wide_open(total_layers: u32, free_bytes: u64) -> Self {
        // `u32::MAX` rather than `total_layers`: llama.cpp clamps it, and a
        // file whose tensor table could not be read reports zero layers, which
        // must not become "offload nothing".
        Self { layers: u32::MAX, experts: 0, total_layers, free_bytes }
    }

    pub fn for_model(path: &std::path::Path, opts: &ozgent_core::Resolved) -> Self {
        let layout = crate::layout::read(path).unwrap_or_default();
        let device = best_gpu();
        let free = device.as_ref().map(|d| d.memory_free as u64).unwrap_or(0);

        // No GPU, or a file we could not measure. Hand it to llama.cpp as
        // before rather than inventing a number from nothing.
        let Some(device) = device.filter(Device::is_gpu) else {
            return Self::wide_open(layout.layers, free);
        };
        if layout.layers == 0 || layout.bytes_per_layer == 0 {
            return Self::wide_open(layout.layers, free);
        }

        // The KV cache and compute buffers stay resident whatever else is
        // evicted, so they come off the budget before any weight does.
        //
        // Which cache, though, is decided later — and `auto` picks the
        // *smallest type that fits* when memory is tight, down to q4_0. So
        // assuming f16 here is not the conservative choice it looks like: it
        // doubles the overhead against a cache the runtime would have
        // quantised, and on a card that already holds another model it ate the
        // entire budget and offloaded nothing. Measured: a second 4B model got
        // 0 layers where 14 fit.
        //
        // q8_0 is what `auto` actually lands on under pressure, and pressure is
        // the only regime where this number changes the answer — with room to
        // spare every layer fits whatever the cache costs.
        let assumed = match opts.cache_type_k {
            ozgent_core::accel::CacheType::Auto => ozgent_core::accel::CacheType::Q8_0,
            explicit => explicit,
        };
        let overhead =
            ozgent_core::accel::kv_bytes(layout.kv_elements_per_token, opts.context_length, assumed);
        let (layers, experts) = resolve_auto(
            opts.gpu_layers,
            opts.cpu_moe,
            Some(&device),
            layout.layers,
            layout.bytes_per_layer,
            layout.expert_bytes_per_layer,
            overhead,
        );
        Self { layers, experts, total_layers: layout.layers, free_bytes: free }
    }

    /// Whether this is the whole model on the GPU.
    pub fn is_full(&self) -> bool {
        self.total_layers == 0 || self.layers >= self.total_layers
    }

    /// How much of the model lands on the GPU, 0 to 1. What a caller uses to
    /// decide whether a load is worth doing or whether to make room first.
    pub fn share(&self) -> f32 {
        if self.total_layers == 0 {
            return 1.0;
        }
        (self.layers.min(self.total_layers) as f32) / (self.total_layers as f32)
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;

    fn plan(layers: u32, total: u32) -> Plan {
        Plan { layers, experts: 0, total_layers: total, free_bytes: 0 }
    }

    #[test]
    fn a_model_that_fits_is_reported_as_fully_offloaded() {
        assert!(plan(32, 32).is_full());
        assert_eq!(plan(32, 32).share(), 1.0);
        // llama.cpp is handed u32::MAX when there is nothing to weigh up, and
        // clamps it itself; that is still the whole model.
        assert!(Plan::wide_open(32, 0).is_full());
    }

    #[test]
    fn a_partial_offload_reports_the_share_that_landed_on_the_gpu() {
        assert!(!plan(16, 32).is_full());
        assert_eq!(plan(16, 32).share(), 0.5);
        assert_eq!(plan(0, 32).share(), 0.0);
    }

    #[test]
    fn a_file_we_could_not_measure_offloads_everything_rather_than_nothing() {
        // layout::read failing reports zero layers. Treating that as "offload
        // nothing" would silently move a working model onto the CPU.
        let unknown = Plan::wide_open(0, 0);
        assert!(unknown.is_full());
        assert_eq!(unknown.share(), 1.0);
        assert_eq!(unknown.layers, u32::MAX, "hand it to llama.cpp as before");
    }

    #[test]
    fn an_empty_gpu_still_takes_the_whole_model() {
        // The regression this guards: making `auto` mean "what fits" must not
        // make the ordinary single-model case offload less than it used to.
        let est = fit_to_vram(8 << 30, 32, 100 << 20, 0, 1 << 30);
        assert_eq!(est.layers, 32, "3.2 GB of layers into 8 GB free");
        assert_eq!(est.cpu_moe, None);
    }

    #[test]
    fn a_gpu_with_another_model_on_it_takes_what_is_left() {
        // 8 GB card, 3.9 GB already used by another model, 1 GB of overhead:
        // some layers fit, not all, and the answer is a number rather than a
        // failure.
        let est = fit_to_vram(4 << 30, 32, 200 << 20, 0, 1 << 30);
        assert!(est.layers > 0 && est.layers < 32, "got {}", est.layers);
    }

    #[test]
    fn a_full_gpu_offloads_nothing_rather_than_overcommitting() {
        let est = fit_to_vram(512 << 20, 32, 200 << 20, 0, 1 << 30);
        assert_eq!(est.layers, 0, "overhead alone does not fit");
    }
}
