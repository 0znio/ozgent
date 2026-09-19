//! What a GGUF weighs, per layer and per expert.
//!
//! Deciding how much of a model fits in VRAM needs two numbers the model file
//! knows and llama.cpp's loaded handle does not expose: how many bytes an
//! average block costs, and how much of that is routed experts. Experts are the
//! interesting part — in a mixture-of-experts model they are most of the weight
//! but only a fraction of the work per token, so evicting them to system RAM
//! buys far more VRAM per lost token/sec than dropping whole layers does.
//!
//! Read with `no_alloc`, so only the header and tensor table are touched. The
//! weights themselves are never mapped, which matters when the file is 20 GB.

use std::ffi::{CStr, CString};
use std::path::Path;

use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;

/// Per-layer byte costs taken from a GGUF's tensor table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Layout {
    /// Average bytes per transformer block.
    pub bytes_per_layer: u64,
    /// Of that, the bytes belonging to routed experts. Zero for a dense model.
    pub expert_bytes_per_layer: u64,
    /// The same figure split by which of the three routed-expert tensors it
    /// belongs to, in the order they are evicted: down, gate, up.
    ///
    /// Kept apart because they are not equal. `_K` quantisations routinely
    /// store `ffn_down_exps` at a higher precision than the other two, so a
    /// third of the layer is the wrong answer by as much as 20% — and the
    /// whole point of measuring at tensor granularity is to stop rounding.
    pub expert_kind_bytes: [u64; 3],
    /// Blocks found in the file.
    pub layers: u32,
    /// KV elements stored per token across all layers, for sizing the cache
    /// that has to stay resident whatever else is evicted.
    pub kv_elements_per_token: u64,
    /// Blocks that keep a cache growing with the context. Fewer than
    /// `layers` on a hybrid model; see [`caching_layers`].
    pub caching_layers: u32,
    /// The model's embedding width, which together with the micro-batch
    /// decides how large llama.cpp's scratch allocations are. Zero when the
    /// file does not say.
    pub n_embd: u32,
    /// The context the model was trained for, from its own metadata. Zero
    /// when the file does not say. Asking for more than this is not a longer
    /// memory, it is a model reading positions it has never seen.
    pub context_train: u32,
    /// The first block that has routed experts. Many mixture-of-experts
    /// models open with dense blocks, and an eviction pattern that counts
    /// from block zero frees nothing for those.
    pub first_moe_layer: u32,
    /// The routed-expert bytes of the heaviest single block. During prefill
    /// llama.cpp uploads one block's evicted experts to the GPU at a time, so
    /// this much has to stay free beside everything else.
    pub max_layer_expert_bytes: u64,
    /// Tensors outside the blocks that llama.cpp keeps on the GPU — the output
    /// head and its norm. The input embedding stays in host memory and is not
    /// counted.
    pub fixed_gpu_bytes: u64,
    /// Recurrent state per sequence, averaged over every block: what one
    /// conversation's linear-attention state costs for each block placed on
    /// the GPU. Zero for a model without one. See [`recurrent_bytes`].
    pub recurrent_bytes_per_layer: u64,
    /// Multi-token-prediction blocks appended past the main stack: a draft
    /// head the model carries for speculating with itself. Zero on most
    /// models. See [`crate::mtp`].
    pub nextn_layers: u32,
}

/// The routed-expert tensors, in the order eviction spends them.
///
/// Order among them barely matters — they are within a fifth of each other in
/// size and each costs the same single activation round trip when it is the
/// one that splits a layer. What matters is that it is *fixed*, so that a
/// plan and the regex built from it cannot disagree about which was meant.
pub const EXPERT_KINDS: [&str; 3] = ["down", "gate", "up"];

impl Layout {
    pub fn is_moe(&self) -> bool {
        self.expert_bytes_per_layer > 0
    }

    /// Bytes freed by evicting `tensors` individual expert tensors, counting
    /// from the start of [`EXPERT_KINDS`].
    pub fn expert_prefix_bytes(&self, tensors: u32) -> u64 {
        self.expert_kind_bytes.iter().take(tensors as usize).sum()
    }
}

/// Measure `path` without loading its weights.
///
/// Returns `None` when the file cannot be read as GGUF, which is not worth an
/// error: the caller falls back to llama.cpp's own placement, exactly as
/// before.
pub fn read(path: &Path) -> Option<Layout> {
    let c_path = CString::new(path.to_string_lossy().as_bytes()).ok()?;
    let params = sys::gguf_init_params { no_alloc: true, ctx: std::ptr::null_mut() };

    // SAFETY: `c_path` outlives the call; the handle is freed on every path.
    let gguf = unsafe { sys::gguf_init_from_file(c_path.as_ptr(), params) };
    if gguf.is_null() {
        return None;
    }
    let mut layout = scan(gguf);
    if let Some(l) = layout.as_mut() {
        l.kv_elements_per_token = kv_elements(gguf, l.layers);
        l.caching_layers = string_key(gguf, "general.architecture")
            .map(|arch| caching_layers(gguf, &arch, l.layers))
            .unwrap_or(l.layers);
        l.context_train = context_train(gguf);
        l.recurrent_bytes_per_layer = string_key(gguf, "general.architecture")
            .map(|arch| recurrent_bytes(gguf, &arch, l.layers, l.caching_layers) / l.layers.max(1) as u64)
            .unwrap_or(0);
        l.n_embd = string_key(gguf, "general.architecture")
            .and_then(|arch| u32_key(gguf, &format!("{arch}.embedding_length")))
            .unwrap_or(0);
        l.nextn_layers = string_key(gguf, "general.architecture")
            .and_then(|arch| u32_key(gguf, &format!("{arch}.nextn_predict_layers")))
            .unwrap_or(0);
    }
    unsafe { sys::gguf_free(gguf) };
    layout
}

/// Whether `path` is an embedding model rather than one to chat with.
///
/// Read from the file, not guessed from the name: an embedding model's GGUF
/// declares how its token vectors are pooled into one (`<arch>.pooling_type`)
/// and a generative model's does not. Only the header is read.
pub fn is_embedding(path: &Path) -> bool {
    let Ok(c_path) = CString::new(path.to_string_lossy().as_bytes()) else { return false };
    let params = sys::gguf_init_params { no_alloc: true, ctx: std::ptr::null_mut() };
    // SAFETY: `c_path` outlives the call; the handle is freed below.
    let gguf = unsafe { sys::gguf_init_from_file(c_path.as_ptr(), params) };
    if gguf.is_null() {
        return false;
    }
    let pooled = string_key(gguf, "general.architecture")
        .and_then(|arch| find(gguf, &format!("{arch}.pooling_type")))
        .is_some();
    unsafe { sys::gguf_free(gguf) };
    pooled
}

fn scan(gguf: *mut sys::gguf_context) -> Option<Layout> {
    let count = unsafe { sys::gguf_get_n_tensors(gguf) };
    if count <= 0 {
        return None;
    }

    let mut block_bytes = 0u64;
    let mut expert_bytes = 0u64;
    let mut kind_bytes = [0u64; 3];
    let mut highest_block: i64 = -1;
    let mut per_block_experts: std::collections::BTreeMap<i64, u64> = Default::default();
    let mut fixed_gpu_bytes = 0u64;

    for i in 0..count {
        let name = unsafe { CStr::from_ptr(sys::gguf_get_tensor_name(gguf, i)) }
            .to_string_lossy()
            .into_owned();
        let size = unsafe { sys::gguf_get_tensor_size(gguf, i) } as u64;

        // Only per-block tensors scale with layer count. The rest are paid
        // once — but the output head and its norm still sit on the GPU, and
        // leaving them out of every figure was 262 MB of GLM-4.7-Flash that the
        // planner placed experts into. The input embedding is looked up on the
        // host, so it alone is left out.
        let Some(rest) = name.strip_prefix("blk.") else {
            if !name.starts_with("token_embd") {
                fixed_gpu_bytes += size;
            }
            continue;
        };
        let Some((index, tail)) = rest.split_once('.') else { continue };
        let Ok(index) = index.parse::<i64>() else { continue };
        highest_block = highest_block.max(index);
        block_bytes += size;

        // The routed-expert tensors, named the same way llama.cpp's own
        // --n-cpu-moe targets them. `ffn_*_shexp` is the *shared* expert, which
        // every token uses and which therefore must stay resident.
        if is_routed_expert(tail) {
            expert_bytes += size;
            *per_block_experts.entry(index).or_default() += size;
            if let Some(k) = EXPERT_KINDS.iter().position(|k| tail.starts_with(&format!("ffn_{k}_"))) {
                kind_bytes[k] += size;
            }
        }
    }

    let layers = u32::try_from(highest_block + 1).ok()?;
    if layers == 0 {
        return None;
    }
    // Averaged over the blocks that have experts, not over every block. A
    // dense leading block diluted the average and, worse, was the first block
    // the eviction pattern named — credited with a full block of experts that
    // did not exist.
    let moe_layers = per_block_experts.len().max(1) as u64;
    Some(Layout {
        bytes_per_layer: block_bytes / layers as u64,
        expert_bytes_per_layer: expert_bytes / moe_layers,
        expert_kind_bytes: kind_bytes.map(|b| b / moe_layers),
        first_moe_layer: per_block_experts.keys().next().map_or(0, |&i| i as u32),
        max_layer_expert_bytes: per_block_experts.values().copied().max().unwrap_or(0),
        fixed_gpu_bytes,
        recurrent_bytes_per_layer: 0,
        layers,
        kv_elements_per_token: 0,
        caching_layers: layers,
        n_embd: 0,
        context_train: 0,
        nextn_layers: 0,
    })
}

/// The context length the model was trained for.
///
/// Read from the file rather than from a loaded handle so a caller can ask
/// before paying to load twenty gigabytes — which is exactly what a settings
/// page needs in order to bound its slider.
fn context_train(gguf: *mut sys::gguf_context) -> u32 {
    let Some(arch) = string_key(gguf, "general.architecture") else { return 0 };
    u32_key(gguf, &format!("{arch}.context_length")).unwrap_or(0)
}

/// KV elements per token, from the architecture's own declared widths.
///
/// Not derivable from `n_embd / n_head`: Qwen3.5 has 16 heads over an
/// embedding width of 2560, giving 160, while its true key length is 256 —
/// which would mis-size the cache by 60%.
fn kv_elements(gguf: *mut sys::gguf_context, layers: u32) -> u64 {
    let Some(arch) = string_key(gguf, "general.architecture") else { return 0 };
    let heads_kv = u32_values(gguf, &format!("{arch}.attention.head_count_kv"));
    let embd = u32_key(gguf, &format!("{arch}.embedding_length")).unwrap_or(0);
    let heads = u32_key(gguf, &format!("{arch}.attention.head_count")).unwrap_or(0);
    let fallback = if heads > 0 { embd / heads } else { 0 };
    let k = u32_key(gguf, &format!("{arch}.attention.key_length")).unwrap_or(fallback);
    let v = u32_key(gguf, &format!("{arch}.attention.value_length")).unwrap_or(fallback);
    // Multi-head latent attention caches one compressed latent per layer and
    // no values at all: llama.cpp allocates no V tensor when a model declares
    // its MLA widths. Pricing a V half anyway doubled the estimate for
    // GLM-4.7-Flash — 848 MiB predicted where llama.cpp allocated 449 — and
    // that half a gigabyte came straight off the budget experts are placed in.
    let v = if is_mla(gguf, &arch) { 0 } else { v };
    // Only the layers that actually keep a cache are priced. See
    // [`caching_layers`] — on a model that interleaves linear attention this
    // is a quarter of them, and pricing all of them over-reserves by 4x.
    let caching = caching_layers(gguf, &arch, layers);
    fold_kv(caching, &heads_kv, k, v)
}

/// Whether this model uses multi-head latent attention, by llama.cpp's own
/// test: both MLA widths declared.
pub fn is_mla(gguf: *mut sys::gguf_context, arch: &str) -> bool {
    u32_key(gguf, &format!("{arch}.attention.key_length_mla")).is_some_and(|n| n > 0)
        && u32_key(gguf, &format!("{arch}.attention.value_length_mla")).is_some_and(|n| n > 0)
}

/// How many blocks keep a cache that grows with the context.
///
/// A hybrid model runs linear attention on most of its layers and full
/// attention on the rest. The linear ones hold a fixed-size recurrent state —
/// real memory, but the same amount at one token as at a hundred thousand —
/// so they contribute nothing to a *per-token* figure.
///
/// Some models say which layers those are by declaring `head_count_kv` as an
/// array, and [`fold_kv`] already handles that. Qwen3.5 does not: it declares
/// a scalar `head_count_kv = 4` and puts the pattern in a separate key. Taking
/// the scalar at face value priced all 33 of a 4B's blocks when only 9 of them
/// cache anything, reserving 2176 MiB where 550 was needed — and that 1.6 GB
/// came straight off the budget the weights were placed against.
///
/// The rule is llama.cpp's own, from `models/qwen35.cpp`:
///
/// ```text
/// is_recr[i] = (i < n_layer) && ((i + 1) % full_attn_interval != 0)
/// ```
///
/// which also says the appended MTP blocks are attention-only, and so do
/// cache. Anything that declares no interval is not hybrid and prices every
/// layer, exactly as before.
fn caching_layers(gguf: *mut sys::gguf_context, arch: &str, layers: u32) -> u32 {
    // An explicit per-layer list wins over any derived pattern, the same
    // precedence llama.cpp applies.
    let explicit = u32_values(gguf, &format!("{arch}.attention.recurrent_layers"));
    if explicit.len() == layers as usize {
        return explicit.iter().filter(|&&recurrent| recurrent == 0).count() as u32;
    }

    let interval = u32_key(gguf, &format!("{arch}.full_attention_interval")).unwrap_or(0);
    if interval <= 1 {
        return layers;
    }
    // The multi-token-prediction blocks sit beyond the main stack and are
    // never recurrent.
    let nextn = u32_key(gguf, &format!("{arch}.nextn_predict_layers")).unwrap_or(0);
    let main = layers.saturating_sub(nextn);
    let full = (0..main).filter(|i| (i + 1) % interval == 0).count() as u32;
    (full + nextn).max(1)
}

/// Recurrent state one sequence holds across the whole model, in bytes.
///
/// Fixed in size — the same at one token as at a million — and so invisible to
/// any per-token figure, but not small: 748 MB for five conversations of a
/// 27B hybrid. Never counted, it let placement put 63 of 64 blocks on an 8 GB
/// card, after which no context of any size would open. It lives beside its
/// block, so a block sent to the CPU takes its state with it, which is why it
/// is priced per block.
///
/// llama.cpp's own sizes (`n_embd_r`, `n_embd_s` in `llama-hparams.cpp`), in
/// f32, which is the type it keeps them in:
///
/// ```text
/// conv  = (ssm.conv_kernel - 1) * (ssm.inner_size + 2 * ssm.group_count * ssm.state_size)
/// state = ssm.state_size * ssm.inner_size
/// ```
///
/// Checked against the allocation that failed: 47 blocks on the card, five
/// sequences, 817,152 floats each, 768,122,880 bytes exactly. Architectures
/// that size their state another way (RWKV, Kimi, MiniMax) are not priced and
/// stay at zero, which is where every model was before.
fn recurrent_bytes(gguf: *mut sys::gguf_context, arch: &str, layers: u32, caching: u32) -> u64 {
    let key = |k: &str| u32_key(gguf, &format!("{arch}.ssm.{k}")).unwrap_or(0) as u64;
    let (d_conv, d_inner, n_group, d_state) =
        (key("conv_kernel"), key("inner_size"), key("group_count"), key("state_size"));
    if d_inner == 0 || d_state == 0 {
        return 0;
    }
    let conv = d_conv.saturating_sub(1) * (d_inner + 2 * n_group * d_state);
    let state = d_state * d_inner;
    let recurrent_layers = layers.saturating_sub(caching) as u64;
    recurrent_layers * (conv + state) * 4
}

/// Turn declared KV-head counts into elements cached per token.
///
/// A hybrid model declares `head_count_kv` per layer rather than once: Ling
/// 3.0 alternates three Kimi-Delta layers to every latent-attention one, so
/// its array reads `[0, 0, 0, 1, ...]`. The zeros are not missing data. Those
/// layers keep a fixed-size recurrent state instead of a cache, and a state
/// that does not grow with the context contributes nothing to a per-token
/// figure. Multiplying the layer count by any single entry would be wrong in
/// both directions — 24x by the ones, zero by the zeros.
fn fold_kv(layers: u32, heads_kv: &[u32], k: u32, v: u32) -> u64 {
    if k == 0 || v == 0 {
        return 0;
    }
    match heads_kv {
        [] => 0,
        // Uniform attention: every layer caches the same amount.
        &[uniform] => ozgent_core::accel::kv_elements_per_token(layers, uniform, k, v),
        per_layer => {
            // Already summed across layers, so there is exactly one layer's
            // worth of that many heads left to price.
            let total: u32 = per_layer.iter().sum();
            ozgent_core::accel::kv_elements_per_token(1, total, k, v)
        }
    }
}

/// Locate a metadata key by name.
fn find(gguf: *mut sys::gguf_context, key: &str) -> Option<i64> {
    let c = CString::new(key).ok()?;
    let index = unsafe { sys::gguf_find_key(gguf, c.as_ptr()) };
    (index >= 0).then_some(index)
}

/// Read one unsigned value, or `None` if the key is absent or not a scalar.
///
/// The type check is not defensive tidiness. `gguf_get_val_u32` asserts on a
/// mismatch, and a failed `GGML_ASSERT` aborts the process — so reading a
/// hybrid model's array-valued `head_count_kv` as a scalar core-dumped ozgent
/// before llama.cpp ever saw the file. Nothing here is recoverable at the
/// call site; it has to be avoided.
fn u32_key(gguf: *mut sys::gguf_context, key: &str) -> Option<u32> {
    let index = find(gguf, key)?;
    scalar_u32(gguf, index)
}

/// Read a key that may be either one value or one per layer.
///
/// Returns an empty vector for a missing key or an element type this cannot
/// read, both of which the caller treats as "unknown" rather than zero.
fn u32_values(gguf: *mut sys::gguf_context, key: &str) -> Vec<u32> {
    let Some(index) = find(gguf, key) else { return Vec::new() };
    if unsafe { sys::gguf_get_kv_type(gguf, index) } != sys::GGUF_TYPE_ARRAY {
        return scalar_u32(gguf, index).into_iter().collect();
    }
    let element = unsafe { sys::gguf_get_arr_type(gguf, index) };
    if !matches!(element, sys::GGUF_TYPE_UINT32 | sys::GGUF_TYPE_INT32) {
        return Vec::new();
    }
    let n = unsafe { sys::gguf_get_arr_n(gguf, index) };
    let data = unsafe { sys::gguf_get_arr_data(gguf, index) } as *const i32;
    if n == 0 || data.is_null() {
        return Vec::new();
    }
    // SAFETY: the array is `n` elements of a 4-byte integer type, checked
    // above, and lives in the gguf context the caller still holds.
    let raw = unsafe { std::slice::from_raw_parts(data, n) };
    raw.iter().map(|&e| u32::try_from(e).unwrap_or(0)).collect()
}

fn scalar_u32(gguf: *mut sys::gguf_context, index: i64) -> Option<u32> {
    match unsafe { sys::gguf_get_kv_type(gguf, index) } {
        sys::GGUF_TYPE_UINT32 => Some(unsafe { sys::gguf_get_val_u32(gguf, index) }),
        sys::GGUF_TYPE_INT32 => {
            u32::try_from(unsafe { sys::gguf_get_val_i32(gguf, index) }).ok()
        }
        _ => None,
    }
}

fn string_key(gguf: *mut sys::gguf_context, key: &str) -> Option<String> {
    let index = find(gguf, key)?;
    if unsafe { sys::gguf_get_kv_type(gguf, index) } != sys::GGUF_TYPE_STRING {
        return None;
    }
    let raw = unsafe { sys::gguf_get_val_str(gguf, index) };
    if raw.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned())
}

/// Whether a per-block tensor name is a routed expert.
fn is_routed_expert(tail: &str) -> bool {
    // `_exps` marks the routed stack; `_shexp` marks the shared expert, which
    // is dense in practice and must not be evicted.
    matches!(
        tail,
        "ffn_up_exps.weight"
            | "ffn_down_exps.weight"
            | "ffn_gate_exps.weight"
            | "ffn_up_chexps.weight"
            | "ffn_down_chexps.weight"
            | "ffn_gate_chexps.weight"
    ) || (tail.contains("_exps") && !tail.contains("shexp"))
}

#[cfg(test)]
mod tests {
    /// The rule from llama.cpp's `models/qwen35.cpp`, in isolation.
    fn caching(layers: u32, interval: u32, nextn: u32) -> u32 {
        if interval <= 1 {
            return layers;
        }
        let main = layers.saturating_sub(nextn);
        let full = (0..main).filter(|i| (i + 1) % interval == 0).count() as u32;
        (full + nextn).max(1)
    }

    #[test]
    fn a_hybrid_model_prices_only_the_layers_that_cache() {
        // Qwen3.5-4B: 33 blocks, one of them multi-token prediction, full
        // attention every fourth. Pricing all 33 reserved 2176 MiB where 550
        // was needed, and that 1.6 GB came off the weights' budget.
        assert_eq!(caching(33, 4, 1), 9);
    }

    #[test]
    fn the_prediction_blocks_cache_even_though_they_sit_outside_the_stack() {
        // `is_recr[i] = (i < n_layer) && ...` — the appended blocks fail the
        // first clause, so they are attention and do cache.
        assert_eq!(caching(33, 4, 1) - caching(32, 4, 0), 1);
    }

    #[test]
    fn a_dense_model_is_untouched_by_any_of_this() {
        // No interval declared means not hybrid, and every layer is priced
        // exactly as it always was.
        assert_eq!(caching(32, 0, 0), 32);
        assert_eq!(caching(32, 1, 0), 32);
    }

    #[test]
    fn a_hybrid_model_never_prices_zero_layers() {
        // An interval longer than the stack would otherwise round to nothing,
        // and a cache of zero bytes is a window of infinity.
        assert!(caching(4, 64, 0) >= 1);
    }

    use super::*;

    #[test]
    fn routed_experts_are_recognised() {
        assert!(is_routed_expert("ffn_up_exps.weight"));
        assert!(is_routed_expert("ffn_down_exps.weight"));
        assert!(is_routed_expert("ffn_gate_chexps.weight"));
    }

    #[test]
    fn the_shared_expert_is_not_evictable() {
        // Every token goes through it, so moving it to RAM costs the full
        // penalty on every token rather than a fraction of them.
        assert!(!is_routed_expert("ffn_up_shexp.weight"));
        assert!(!is_routed_expert("ffn_down_shexp.weight"));
    }

    #[test]
    fn dense_tensors_are_not_experts() {
        for t in ["attn_q.weight", "attn_norm.weight", "ffn_up.weight", "ffn_down.weight"] {
            assert!(!is_routed_expert(t), "{t}");
        }
    }

    #[test]
    fn a_dense_layout_reports_no_experts() {
        let l = Layout { bytes_per_layer: 1000, layers: 32, caching_layers: 32, ..Default::default() };
        assert!(!l.is_moe());
    }

    #[test]
    fn a_uniform_model_prices_every_layer() {
        // 32 layers * 4 heads * (256 + 256).
        assert_eq!(fold_kv(32, &[4], 256, 256), 65_536);
    }

    #[test]
    fn a_hybrid_model_prices_only_its_caching_layers() {
        // Ling 3.0 tiny: 24 layers, one latent-attention layer in every four,
        // 576-wide keys against 128-wide values.
        let per_layer: Vec<u32> = (0..24).map(|i| u32::from(i % 4 == 3)).collect();
        assert_eq!(fold_kv(24, &per_layer, 576, 128), 6 * (576 + 128));
    }

    #[test]
    fn the_layer_count_does_not_scale_a_per_layer_array() {
        // The bug this guards: treating entry zero as uniform reads the whole
        // model as cache-free, and treating a one as uniform overcounts 4x.
        let per_layer = [0, 0, 0, 1];
        assert_ne!(fold_kv(4, &per_layer, 64, 64), 0);
        assert!(fold_kv(4, &per_layer, 64, 64) < fold_kv(4, &[1], 64, 64));
    }

    #[test]
    fn an_unreadable_width_is_unknown_rather_than_zero_cost() {
        assert_eq!(fold_kv(32, &[4], 0, 256), 0);
        assert_eq!(fold_kv(32, &[], 256, 256), 0);
    }
}
