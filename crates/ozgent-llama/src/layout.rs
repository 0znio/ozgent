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
    /// Blocks found in the file.
    pub layers: u32,
    /// KV elements stored per token across all layers, for sizing the cache
    /// that has to stay resident whatever else is evicted.
    pub kv_elements_per_token: u64,
    /// The context the model was trained for, from its own metadata. Zero
    /// when the file does not say. Asking for more than this is not a longer
    /// memory, it is a model reading positions it has never seen.
    pub context_train: u32,
}

impl Layout {
    pub fn is_moe(&self) -> bool {
        self.expert_bytes_per_layer > 0
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
        l.context_train = context_train(gguf);
    }
    unsafe { sys::gguf_free(gguf) };
    layout
}

fn scan(gguf: *mut sys::gguf_context) -> Option<Layout> {
    let count = unsafe { sys::gguf_get_n_tensors(gguf) };
    if count <= 0 {
        return None;
    }

    let mut block_bytes = 0u64;
    let mut expert_bytes = 0u64;
    let mut highest_block: i64 = -1;

    for i in 0..count {
        let name = unsafe { CStr::from_ptr(sys::gguf_get_tensor_name(gguf, i)) }
            .to_string_lossy()
            .into_owned();
        let size = unsafe { sys::gguf_get_tensor_size(gguf, i) } as u64;

        // Only per-block tensors scale with layer count; embeddings and the
        // output head are paid once and belong to neither figure.
        let Some(rest) = name.strip_prefix("blk.") else { continue };
        let Some((index, tail)) = rest.split_once('.') else { continue };
        let Ok(index) = index.parse::<i64>() else { continue };
        highest_block = highest_block.max(index);
        block_bytes += size;

        // The routed-expert tensors, named the same way llama.cpp's own
        // --n-cpu-moe targets them. `ffn_*_shexp` is the *shared* expert, which
        // every token uses and which therefore must stay resident.
        if is_routed_expert(tail) {
            expert_bytes += size;
        }
    }

    let layers = u32::try_from(highest_block + 1).ok()?;
    if layers == 0 {
        return None;
    }
    Some(Layout {
        bytes_per_layer: block_bytes / layers as u64,
        expert_bytes_per_layer: expert_bytes / layers as u64,
        layers,
        kv_elements_per_token: 0,
        context_train: 0,
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
    fold_kv(layers, &heads_kv, k, v)
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
        let l = Layout { bytes_per_layer: 1000, expert_bytes_per_layer: 0, layers: 32, kv_elements_per_token: 0, context_train: 0 };
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
