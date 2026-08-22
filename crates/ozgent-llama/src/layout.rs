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
    })
}

/// KV elements per token, from the architecture's own declared widths.
///
/// Not derivable from `n_embd / n_head`: Qwen3.5 has 16 heads over an
/// embedding width of 2560, giving 160, while its true key length is 256 —
/// which would mis-size the cache by 60%.
fn kv_elements(gguf: *mut sys::gguf_context, layers: u32) -> u64 {
    let Some(arch) = string_key(gguf, "general.architecture") else { return 0 };
    let head_kv = u32_key(gguf, &format!("{arch}.attention.head_count_kv")).unwrap_or(0);
    let embd = u32_key(gguf, &format!("{arch}.embedding_length")).unwrap_or(0);
    let heads = u32_key(gguf, &format!("{arch}.attention.head_count")).unwrap_or(0);
    let fallback = if heads > 0 { embd / heads } else { 0 };
    let k = u32_key(gguf, &format!("{arch}.attention.key_length")).unwrap_or(fallback);
    let v = u32_key(gguf, &format!("{arch}.attention.value_length")).unwrap_or(fallback);
    if head_kv == 0 || k == 0 || v == 0 {
        return 0;
    }
    ozgent_core::accel::kv_elements_per_token(layers, head_kv, k, v)
}

fn u32_key(gguf: *mut sys::gguf_context, key: &str) -> Option<u32> {
    let c = CString::new(key).ok()?;
    let index = unsafe { sys::gguf_find_key(gguf, c.as_ptr()) };
    if index < 0 {
        return None;
    }
    Some(unsafe { sys::gguf_get_val_u32(gguf, index) })
}

fn string_key(gguf: *mut sys::gguf_context, key: &str) -> Option<String> {
    let c = CString::new(key).ok()?;
    let index = unsafe { sys::gguf_find_key(gguf, c.as_ptr()) };
    if index < 0 {
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
        let l = Layout { bytes_per_layer: 1000, expert_bytes_per_layer: 0, layers: 32, kv_elements_per_token: 0 };
        assert!(!l.is_moe());
    }
}
