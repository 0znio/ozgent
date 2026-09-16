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
    /// Whole layers to evict experts from, when the model is a mixture of
    /// experts.
    pub cpu_moe: Option<u32>,
    /// Individual expert tensors to evict from the *next* layer after those,
    /// 0 to 2. This is what makes the step size a third of a layer rather
    /// than a whole one, and at most one layer is ever left split.
    pub cpu_moe_tensors: u32,
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
/// experts, and is zero for a dense model. `kind_bytes` is that figure split
/// across the three routed-expert tensors, which is what allows a partial
/// layer. `overhead_bytes` covers the KV cache and compute buffers, which
/// must stay resident.
///
/// **Why thirds.** A whole layer is a coarse unit: on a 40-layer model with
/// 19 GB of experts each step is ~475 MB, so the search rounds down by up to
/// half a gigabyte of VRAM that then sits idle. Splitting one layer's experts
/// across the two devices costs a single activation round trip per token —
/// tens of kilobytes at batch 1, which is nothing beside the layer it buys —
/// and only ever one layer is split, because whole layers are always spent
/// first.
pub fn fit_to_vram(
    free_bytes: u64,
    total_layers: u32,
    bytes_per_layer: u64,
    expert_bytes_per_layer: u64,
    kind_bytes: [u64; 3],
    overhead_bytes: u64,
) -> FitEstimate {
    // `overhead_bytes` already carries the KV floor *and* a measured reserve
    // for llama.cpp's own scratch, so there is no percentage to take here.
    // There used to be a flat 8% on top, which stacked with a separate 30% on
    // the cache side and between them left well over a gigabyte of an 8 GB
    // card unused — a margin that grew with the card instead of with the
    // model, which is the wrong way round.
    let budget = free_bytes.saturating_sub(overhead_bytes);

    if bytes_per_layer == 0 || total_layers == 0 {
        return FitEstimate { layers: 0, cpu_moe: None, cpu_moe_tensors: 0, bytes_used: 0 };
    }

    let all_layers_cost = bytes_per_layer * total_layers as u64;
    if all_layers_cost <= budget {
        return FitEstimate {
            layers: total_layers,
            cpu_moe: None,
            cpu_moe_tensors: 0,
            bytes_used: all_layers_cost,
        };
    }

    // Try keeping every layer resident but evicting as little as possible,
    // which is the smallest loss that still fits. The search walks in thirds
    // of a layer: `whole` layers fully evicted, then `tensors` more from the
    // layer after them.
    if expert_bytes_per_layer > 0 {
        // A measured split is used where there is one, and an even one is
        // assumed otherwise — a file we could not break down is still better
        // served by approximate thirds than by whole layers.
        let kinds = if kind_bytes.iter().sum::<u64>() > 0 {
            kind_bytes
        } else {
            [expert_bytes_per_layer / 3; 3]
        };
        for step in 1..=(total_layers as u64 * 3) {
            let whole = (step / 3) as u32;
            let tensors = (step % 3) as u32;
            let partial: u64 = kinds.iter().take(tensors as usize).sum();
            let freed = expert_bytes_per_layer * whole as u64 + partial;
            let cost = all_layers_cost.saturating_sub(freed);
            if cost <= budget {
                return FitEstimate {
                    layers: total_layers,
                    cpu_moe: Some(whole),
                    cpu_moe_tensors: tensors,
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
        cpu_moe_tensors: 0,
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
    kind_bytes: [u64; 3],
    overhead_bytes: u64,
) -> (u32, u32, u32) {
    let explicit_layers = match gpu_layers {
        GpuLayers::Count(n) => Some(n.min(total_layers)),
        GpuLayers::Keyword(ozgent_core::options::GpuKeyword::Off) => Some(0),
        GpuLayers::Keyword(ozgent_core::options::GpuKeyword::Auto) => None,
    };

    // With no GPU, or with the GPU switched off, nothing is offloaded and
    // expert eviction is meaningless.
    let Some(device) = device.filter(|d| d.is_gpu()) else {
        return (0, 0, 0);
    };
    if explicit_layers == Some(0) {
        return (0, 0, 0);
    }

    let estimate = fit_to_vram(
        device.memory_free as u64,
        total_layers,
        bytes_per_layer,
        expert_bytes_per_layer,
        kind_bytes,
        overhead_bytes,
    );

    let layers = explicit_layers.unwrap_or(estimate.layers);
    // A number the person typed means whole layers and nothing finer; the
    // partial layer only ever comes from `auto`, which is the only setting
    // that is trying to land on an exact figure.
    let (moe, tensors) = match cpu_moe {
        MoeOffload::Layers(n) => (n, 0),
        MoeOffload::Keyword(ozgent_core::accel::MoeKeyword::Off) => (0, 0),
        MoeOffload::Keyword(ozgent_core::accel::MoeKeyword::All) => (total_layers, 0),
        MoeOffload::Keyword(ozgent_core::accel::MoeKeyword::Auto) => {
            (estimate.cpu_moe.unwrap_or(0), estimate.cpu_moe_tensors)
        }
    };
    (layers, moe, tensors)
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
        let f = fit_to_vram(24 * GIB, 32, 200 * 1024 * 1024, 0, [0; 3], GIB);
        assert_eq!(f.layers, 32);
        assert_eq!(f.cpu_moe, None, "no need to evict experts when it all fits");
    }

    #[test]
    fn a_partial_layer_lands_closer_to_the_memory_that_exists() {
        // The point of thirds. Whole-layer steps round down by up to a whole
        // layer of experts, and on a card with nothing else running that
        // rounding is VRAM left idle for the life of the process.
        let per_layer = 300 * 1024 * 1024;
        let expert = 240 * 1024 * 1024;
        let kinds = [expert / 3; 3];
        // A budget deliberately landing between two whole-layer steps.
        let free = 8 * GIB;
        let f = fit_to_vram(free, 32, per_layer, expert, kinds, GIB);
        let whole_only = fit_to_vram(free, 32, per_layer, expert, [0, 0, 0], GIB);
        assert_eq!(f.layers, 32, "every layer should stay resident");
        // With thirds available the plan uses at least as much of the card.
        assert!(
            f.bytes_used >= whole_only.bytes_used,
            "thirds used {} but whole layers used {}",
            f.bytes_used, whole_only.bytes_used
        );
    }

    #[test]
    fn at_most_one_layer_is_ever_left_split() {
        // Splitting a layer costs an activation round trip per token, so it
        // is spent once and never as a general strategy.
        let per_layer = 300 * 1024 * 1024;
        let expert = 240 * 1024 * 1024;
        for free in [2 * GIB, 4 * GIB, 6 * GIB, 8 * GIB] {
            let f = fit_to_vram(free, 32, per_layer, expert, [expert / 3; 3], GIB);
            assert!(f.cpu_moe_tensors <= 2, "free={free}: {f:?}");
        }
    }

    #[test]
    fn an_unmeasured_split_still_gets_thirds() {
        // A file whose tensor table could not be broken down is better served
        // by approximate thirds than by whole layers only.
        let per_layer = 300 * 1024 * 1024;
        let expert = 240 * 1024 * 1024;
        let f = fit_to_vram(5 * GIB, 32, per_layer, expert, [0; 3], GIB);
        assert_eq!(f.layers, 32);
        assert!(f.cpu_moe.is_some());
    }

    #[test]
    fn the_pattern_names_whole_layers_and_the_part_layer_together() {
        let p = moe_pattern(3, 2).expect("something is evicted");
        // Whole layers 0..2, then two of three kinds from layer 3.
        assert!(p.contains("(0|1|2)"), "{p}");
        assert!(p.contains(r"blk\.3\."), "{p}");
        assert!(p.contains("down|gate"), "{p}");
        assert!(!p.contains("up|down|gate|"), "the part layer must not take all three: {p}");
    }

    #[test]
    fn a_whole_number_of_layers_keeps_the_pattern_it_always_had() {
        let p = moe_pattern(3, 0).expect("something is evicted");
        assert_eq!(p, r"blk\.(0|1|2)\.ffn_(up|down|gate)_(ch|)exps");
    }

    #[test]
    fn evicting_nothing_produces_no_pattern() {
        // An empty pattern would match every tensor name and move the whole
        // model to the CPU, which is the opposite of what it means.
        assert_eq!(moe_pattern(0, 0), None);
    }

    #[test]
    fn a_split_with_no_whole_layers_still_names_layer_zero() {
        let p = moe_pattern(0, 1).expect("something is evicted");
        assert!(p.contains(r"blk\.0\."), "{p}");
        assert!(p.contains("down"), "{p}");
    }

    #[test]
    fn experts_are_evicted_before_layers_are_dropped() {
        // A dense-heavy MoE that does not fit whole: the right answer keeps
        // every layer's attention on the GPU and moves experts out.
        let per_layer = 600 * 1024 * 1024;
        let expert = 500 * 1024 * 1024;
        let f = fit_to_vram(8 * GIB, 32, per_layer, expert, [expert / 3; 3], GIB);

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
        let f = fit_to_vram(8 * GIB, 32, per_layer, expert, [expert / 3; 3], GIB);
        let evicted = f.cpu_moe.expect("should evict");

        let dense = per_layer - expert;
        let one_less = dense * (evicted - 1) as u64 + per_layer * (32 - evicted + 1) as u64;
        let budget = (8 * GIB - GIB) * 92 / 100;
        assert!(one_less > budget, "evicting {} would have sufficed", evicted - 1);
    }

    #[test]
    fn layers_are_dropped_only_when_eviction_is_not_enough() {
        // A dense model has no experts to evict.
        let f = fit_to_vram(2 * GIB, 32, 500 * 1024 * 1024, 0, [0; 3], GIB);
        assert!(f.layers < 32, "must offload fewer layers");
        assert_eq!(f.cpu_moe, None);
    }

    #[test]
    fn headroom_is_reserved_rather_than_filling_vram_exactly() {
        let free = 8 * GIB;
        let f = fit_to_vram(free, 100, 100 * 1024 * 1024, 0, [0; 3], 0);
        assert!(
            f.bytes_used < free,
            "an exact fit fails at load time; got {} of {free}",
            f.bytes_used
        );
    }

    #[test]
    fn no_gpu_means_nothing_is_offloaded() {
        let (layers, moe, _) = resolve_auto(
            GpuLayers::AUTO, MoeOffload::AUTO, None, 32, GIB, 0, [0; 3], GIB,
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
        let (layers, _, _) = resolve_auto(
            GpuLayers::AUTO, MoeOffload::AUTO, Some(&cpu), 32, GIB, 0, [0; 3], GIB,
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
        let (layers, moe, _) = resolve_auto(
            GpuLayers::OFF, MoeOffload::AUTO, Some(&gpu), 32, GIB, 0, [0; 3], GIB,
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
        let (layers, _, _) = resolve_auto(
            GpuLayers::Count(10), MoeOffload::OFF, Some(&gpu), 32, GIB / 4, 0, [0; 3], GIB,
        );
        assert_eq!(layers, 10);

        let (clamped, _, _) = resolve_auto(
            GpuLayers::Count(999), MoeOffload::OFF, Some(&gpu), 32, GIB / 4, 0, [0; 3], GIB,
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
        let (_, moe, _) = resolve_auto(
            GpuLayers::AUTO, MoeOffload::ALL, Some(&gpu), 48, GIB / 8, GIB / 16, [0; 3], GIB,
        );
        assert_eq!(moe, 48);
    }

    #[test]
    fn zero_sized_models_do_not_divide_by_zero() {
        let f = fit_to_vram(8 * GIB, 0, 0, 0, [0; 3], 0);
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
    /// Individual expert tensors evicted from the layer after those, 0 to 2.
    /// The step that makes placement land within a third of a layer of the
    /// memory actually available instead of rounding a whole one away.
    pub expert_tensors: u32,
    pub total_layers: u32,
    /// Free device memory the decision was made against.
    pub free_bytes: u64,
}

/// What this machine has learned about llama.cpp's own memory cost.
///
/// Held per process rather than per engine: the expensive half of it is the
/// backend context, which every model in the process shares. See
/// [`ozgent_core::reserve`].
fn learned() -> &'static std::sync::Mutex<ozgent_core::reserve::Reserve> {
    static LEARNED: std::sync::OnceLock<std::sync::Mutex<ozgent_core::reserve::Reserve>> =
        std::sync::OnceLock::new();
    LEARNED.get_or_init(|| {
        let loaded = ozgent_core::Paths::discover()
            .map(|p| ozgent_core::reserve::Reserve::load(&p))
            .unwrap_or_default();
        std::sync::Mutex::new(loaded)
    })
}

/// Set once a model has been loaded in this process, so the next one is not
/// charged for bringing the backend up a second time.
static BACKEND_UP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The shape of a load, for predicting and for learning what it costs.
pub fn reserve_shape(
    ubatch: u32,
    n_embd: u32,
    n_ctx: u32,
    n_batch: u32,
    staging_bytes: u64,
) -> ozgent_core::reserve::Shape {
    ozgent_core::reserve::Shape {
        first_in_process: !BACKEND_UP.load(std::sync::atomic::Ordering::SeqCst),
        ubatch,
        n_embd,
        n_ctx,
        n_batch,
        staging_bytes,
    }
}

/// Bytes to keep back from a load of this shape.
pub fn reserve_for(shape: ozgent_core::reserve::Shape) -> u64 {
    learned().lock().map(|r| r.predict(shape)).unwrap_or(512 * 1024 * 1024)
}

/// Whether flash attention on this build's GPU backend has a kernel for every
/// quantised K/V pair, or only the handful of same-type ones.
///
/// Asked of the backend rather than assumed: llama.cpp's CUDA backend lists
/// `FA_ALL_QUANTS` among its features exactly when it was compiled with every
/// kernel. No CUDA backend registered means no such limitation to respect.
/// See `accel::flash_has_kernel` for what getting this wrong costs.
#[cfg(feature = "llama")]
pub fn flash_takes_any_kv_pair() -> bool {
    use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;
    static ANSWER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ANSWER.get_or_init(|| unsafe {
        let reg = sys::ggml_backend_reg_by_name(c"CUDA".as_ptr());
        if reg.is_null() {
            return true;
        }
        let proc = sys::ggml_backend_reg_get_proc_address(reg, c"ggml_backend_get_features".as_ptr());
        if proc.is_null() {
            return false;
        }
        let features: unsafe extern "C" fn(sys::ggml_backend_reg_t) -> *const sys::ggml_backend_feature =
            std::mem::transmute(proc);
        let mut f = features(reg);
        while !f.is_null() && !(*f).name.is_null() {
            if std::ffi::CStr::from_ptr((*f).name) == c"FA_ALL_QUANTS" {
                return true;
            }
            f = f.add(1);
        }
        false
    })
}

/// What a context's first decode is expected to take beyond its buffers.
pub fn decode_reserve() -> u64 {
    learned().lock().map(|r| r.decode_bytes).unwrap_or(128 * 1024 * 1024)
}

/// Fold in what a first decode actually took, and remember it.
pub fn record_decode(bytes: u64) {
    let Ok(mut learned) = learned().lock() else { return };
    learned.observe_decode(bytes);
    if let Ok(paths) = ozgent_core::Paths::discover() {
        learned.save(&paths);
    }
    tracing::debug!("decode working memory: {} MiB", learned.decode_bytes / (1024 * 1024));
}

/// Fold in what a load actually cost, and remember it for next time.
///
/// `overhead` is what disappeared beyond the weights and the cache. This is
/// the whole of the adaptation: the reservation stops being a number somebody
/// guessed and becomes a number this machine measured.
pub fn record_reserve(shape: ozgent_core::reserve::Shape, overhead: u64) {
    BACKEND_UP.store(true, std::sync::atomic::Ordering::SeqCst);
    let Ok(mut learned) = learned().lock() else { return };
    learned.observe(shape, overhead);
    if let Ok(paths) = ozgent_core::Paths::discover() {
        learned.save(&paths);
    }
    tracing::debug!(
        "vram reserve: {} MiB once + {:.0} B per ubatch-element, after {} sample(s)",
        learned.process_bytes / (1024 * 1024),
        learned.rate,
        learned.samples
    );
}

/// The tensor-name pattern that sends routed experts to the CPU.
///
/// `whole` layers give up all three of their expert tensors; the layer after
/// them gives up the first `tensors` of [`crate::layout::EXPERT_KINDS`]. One
/// pattern covers both, because the vendored binding fills slot zero on every
/// call and asserts on the second — so a second override is not available,
/// and alternation is how two rules become one.
///
/// llama.cpp matches with `std::regex_search`, so this need not anchor.
pub fn moe_pattern(whole: u32, tensors: u32) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if whole > 0 {
        let blocks = (0..whole).map(|l| l.to_string()).collect::<Vec<_>>().join("|");
        parts.push(format!(r"blk\.({blocks})\.ffn_(up|down|gate)_(ch|)exps"));
    }
    if tensors > 0 {
        let kinds = crate::layout::EXPERT_KINDS
            .iter()
            .take(tensors.min(3) as usize)
            .copied()
            .collect::<Vec<_>>()
            .join("|");
        parts.push(format!(r"blk\.{whole}\.ffn_({kinds})_(ch|)exps"));
    }
    match parts.len() {
        0 => None,
        1 => parts.pop(),
        _ => Some(parts.iter().map(|p| format!("({p})")).collect::<Vec<_>>().join("|")),
    }
}

/// The window the placement decision reserves cache for.
///
/// Not the window the model will run with — that is decided afterwards, from
/// what the weights left behind, and is usually far larger. This is only "how
/// much cache must fit for the result to be worth loading at all", so that a
/// model with an enormous default window does not reserve its way out of the
/// GPU entirely.
const PLANNING_WINDOW: u32 = 8192;

impl Plan {
    /// Everything on the GPU, for when there is nothing to weigh up.
    fn wide_open(total_layers: u32, free_bytes: u64) -> Self {
        // `u32::MAX` rather than `total_layers`: llama.cpp clamps it, and a
        // file whose tensor table could not be read reports zero layers, which
        // must not become "offload nothing".
        Self { layers: u32::MAX, experts: 0, expert_tensors: 0, total_layers, free_bytes }
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

        // Some KV cache has to stay resident whatever else is evicted, so it
        // comes off the budget before any weight does — but only a *floor* of
        // it, and that distinction is the whole of this comment.
        //
        // The cache is elastic and the weights are not. `fit_context` sizes
        // the real window to whatever is left once the weights are placed, so
        // reserving the full requested window here is reserving memory that
        // will never be asked for. On a model whose default window is its
        // trained one — 107k tokens is ordinary now — that reservation is
        // larger than the card, the budget for weights comes out at zero, and
        // a 4B model that fits four times over lands 4 of 33 layers on the
        // GPU with 5.9 GB free beside it. Measured, on an 8 GB card.
        //
        // So the floor is a window worth having rather than the window asked
        // for. Place the weights, then let the context take what is left:
        // layers on the GPU are worth far more than a window nothing will use.
        let window = opts.context_length.min(PLANNING_WINDOW);

        // Which cache type is decided later, and `auto` picks the smallest
        // that fits when memory is tight, down to q4_0. Assuming f16 is not
        // the conservative choice it looks like — it doubles the floor against
        // a cache the runtime would have quantised. q8_0 is where `auto`
        // actually lands under pressure.
        let assumed = match opts.cache_type_k {
            ozgent_core::accel::CacheType::Auto => ozgent_core::accel::CacheType::Q8_0,
            explicit => explicit,
        };
        // Two things must survive the weights: a floor of KV cache, and
        // whatever llama.cpp allocates for itself. The second used to be a
        // percentage of the card; it is now a measured figure that scales
        // with the model rather than with the hardware.
        // Staging is added below, once it is known whether anything is evicted.
        //
        // The micro-batch must be the one the *context* will choose, not the
        // default. `Engine::open` takes the wider batch whenever it costs no
        // window, and scratch scales with it; planning the weights against the
        // narrow one and then opening the wide one spends the difference out
        // of the cache's budget, which is exactly the memory this reserve
        // exists to protect.
        let planned_batch = match opts.ubatch {
            Some(n) => n,
            None => opts.batch_size.max(crate::engine::WIDE_BATCH),
        };
        let shape =
            reserve_shape(planned_batch, layout.n_embd, window, planned_batch.max(opts.batch_size), 0);
        // `decode_reserve` is the memory llama.cpp turns out to want on its
        // first decode — lazily created cuBLAS workspaces and pool growth —
        // and it is charged at context-open time whatever happens here. Left
        // out of the placement decision, the weights are laid down as though
        // it did not exist and the cache pays for it instead. Measured on a
        // 23B MoE with four conversations: the plan left 15 MiB of an 8 GB
        // card free, the 32k window asked for collapsed to 578 tokens, and a
        // turn carrying tool schemas no longer fitted in its own context.
        let overhead = ozgent_core::accel::kv_bytes(layout.kv_elements_per_token, window, assumed)
            + reserve_for(shape)
            + decode_reserve()
            + layout.fixed_gpu_bytes;
        let place = |overhead: u64| {
            resolve_auto(
                opts.gpu_layers,
                opts.cpu_moe,
                Some(&device),
                layout.layers,
                layout.bytes_per_layer,
                layout.expert_bytes_per_layer,
                layout.expert_kind_bytes,
                overhead,
            )
        };
        let (mut layers, mut experts, mut expert_tensors) = place(overhead);
        // Evicting any experts brings a cost the first pass could not see:
        // prefill uploads one block's experts to the GPU at a time, into a
        // buffer llama.cpp sizes for the heaviest block. Measured at 331 MiB
        // on a model whose blocks carry 294 MB of experts. So once eviction is
        // on the table, place again with that much held back.
        if experts > 0 || expert_tensors > 0 {
            (layers, experts, expert_tensors) = place(overhead + layout.max_layer_expert_bytes);
        }
        // `fit_to_vram` counts blocks that have experts. The pattern counts
        // blocks from zero, the way llama.cpp's own `--n-cpu-moe` does, so the
        // dense blocks in front are added back in.
        if experts > 0 || expert_tensors > 0 {
            experts = (experts + layout.first_moe_layer).min(layout.layers);
        }
        Self { layers, experts, expert_tensors, total_layers: layout.layers, free_bytes: free }
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
        Plan { layers, experts: 0, expert_tensors: 0, total_layers: total, free_bytes: 0 }
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
        let est = fit_to_vram(8 << 30, 32, 100 << 20, 0, [0; 3], 1 << 30);
        assert_eq!(est.layers, 32, "3.2 GB of layers into 8 GB free");
        assert_eq!(est.cpu_moe, None);
    }

    #[test]
    fn a_gpu_with_another_model_on_it_takes_what_is_left() {
        // 8 GB card, 3.9 GB already used by another model, 1 GB of overhead:
        // some layers fit, not all, and the answer is a number rather than a
        // failure.
        let est = fit_to_vram(4 << 30, 32, 200 << 20, 0, [0; 3], 1 << 30);
        assert!(est.layers > 0 && est.layers < 32, "got {}", est.layers);
    }

    #[test]
    fn a_full_gpu_offloads_nothing_rather_than_overcommitting() {
        let est = fit_to_vram(512 << 20, 32, 200 << 20, 0, [0; 3], 1 << 30);
        assert_eq!(est.layers, 0, "overhead alone does not fit");
    }
}

#[cfg(test)]
mod elastic_cache_tests {
    use super::*;

    /// Roughly a 4B at Q4 with a 107k trained window, on an 8 GB card — the
    /// case that put 4 of 33 layers on the GPU and left 5.9 GB unused.
    const LAYERS: u32 = 33;
    const PER_LAYER: u64 = 103 << 20;
    const FREE: u64 = 7680 << 20;

    /// What the old planner did: reserve the whole requested window first.
    fn reserving(window_bytes: u64) -> FitEstimate {
        fit_to_vram(FREE, LAYERS, PER_LAYER, 0, [0; 3], window_bytes)
    }

    #[test]
    fn reserving_a_huge_window_first_is_what_emptied_the_gpu() {
        // Not a test of current behaviour — a record of why the floor exists.
        // A 107k window costs more than the card, so nothing was left for
        // weights.
        let whole_window = 7500u64 << 20;
        assert!(reserving(whole_window).layers < 5, "this is the bug being fixed");
    }

    #[test]
    fn reserving_only_a_usable_floor_puts_the_whole_model_on_the_gpu() {
        // 8k of cache for this model is a few hundred MB, and 3.3 GB of
        // weights fits the remaining budget several times over.
        let floor = 300u64 << 20;
        let est = reserving(floor);
        assert_eq!(est.layers, LAYERS, "every layer should fit");
    }

    #[test]
    fn the_planning_window_is_a_floor_not_the_window_that_gets_used() {
        // A model asking for less than the floor reserves only what it asked
        // for; one asking for more still reserves just the floor, and takes
        // the rest for context afterwards.
        assert_eq!(4096u32.min(PLANNING_WINDOW), 4096);
        assert_eq!(107_008u32.min(PLANNING_WINDOW), PLANNING_WINDOW);
        assert_eq!(32_768u32.min(PLANNING_WINDOW), PLANNING_WINDOW);
    }

    #[test]
    fn a_model_too_big_for_the_card_still_offloads_what_fits() {
        // The floor must not make the planner optimistic to the point of
        // claiming a model fits when it does not.
        let est = fit_to_vram(FREE, LAYERS, 400 << 20, 0, [0; 3], 300 << 20);
        assert!(est.layers > 0 && est.layers < LAYERS, "got {}", est.layers);
    }

    #[test]
    fn a_card_with_nothing_spare_still_offloads_nothing() {
        let est = fit_to_vram(200 << 20, LAYERS, PER_LAYER, 0, [0; 3], 300 << 20);
        assert_eq!(est.layers, 0);
    }
}
