//! Pin the pages of host-side experts where they already are, in the mmap.
//!
//! During prefill llama.cpp copies each block's host-side experts to the GPU.
//! From ordinary pageable memory CUDA cannot copy directly: it stages the
//! bytes through a pinned bounce buffer first, which is slower and cannot run
//! beside anything else. Placing the experts in CUDA's pinned host buffer
//! instead measured 484 tok/s against 423 on GLM-4.7-Flash — but that copies
//! them out of the file into memory the kernel can never reclaim, and loses
//! what mmap is for: sharing the page cache, and not paying a copy at load.
//!
//! This keeps the mmap and pins its pages in place. llama.cpp maps the whole
//! file once, read-only, so a tensor's address is the mapping's start plus the
//! tensor's offset in the file — both readable without reaching into
//! llama.cpp: `/proc/self/maps` for the first, the GGUF tensor table for the
//! second. Only the experts that stay on the host are pinned. Everything on the
//! GPU already has its own copy there, and pinning it would lock RAM for no
//! transfer at all.

use crate::layout::{EXPERT_KINDS, is_routed_expert};
use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;
use std::ffi::{CStr, CString};
use std::path::Path;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn ggml_backend_cuda_register_host_buffer(buffer: *mut std::ffi::c_void, size: usize) -> bool;
}

/// A read-only file mapping of `path` in this process.
struct Mapping {
    start: usize,
    end: usize,
    offset: usize,
}

fn mappings(path: &Path) -> Vec<Mapping> {
    let Ok(canon) = std::fs::canonicalize(path) else { return Vec::new() };
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else { return Vec::new() };
    maps.lines()
        .filter_map(|line| {
            // start-end perms offset dev inode path
            let mut f = line.split_whitespace();
            let range = f.next()?;
            let _perms = f.next()?;
            let offset = usize::from_str_radix(f.next()?, 16).ok()?;
            let _dev = f.next()?;
            let _inode = f.next()?;
            let file = f.collect::<Vec<_>>().join(" ");
            if Path::new(&file) != canon {
                return None;
            }
            let (a, b) = range.split_once('-')?;
            Some(Mapping {
                start: usize::from_str_radix(a, 16).ok()?,
                end: usize::from_str_radix(b, 16).ok()?,
                offset,
            })
        })
        .collect()
}

/// File ranges of the routed-expert tensors that stay on the host: every
/// block before `whole`, and the first `tensors` kinds of block `whole`.
fn host_expert_ranges(path: &Path, whole: u32, tensors: u32) -> Vec<(usize, usize)> {
    let Ok(c_path) = CString::new(path.to_string_lossy().as_bytes()) else { return Vec::new() };
    let params = sys::gguf_init_params { no_alloc: true, ctx: std::ptr::null_mut() };
    // SAFETY: the path outlives the call; the handle is freed below.
    let gguf = unsafe { sys::gguf_init_from_file(c_path.as_ptr(), params) };
    if gguf.is_null() {
        return Vec::new();
    }
    let base = unsafe { sys::gguf_get_data_offset(gguf) };
    let mut out = Vec::new();
    for i in 0..unsafe { sys::gguf_get_n_tensors(gguf) } {
        let name = unsafe { CStr::from_ptr(sys::gguf_get_tensor_name(gguf, i)) }.to_string_lossy();
        let Some((block, tail)) = name.strip_prefix("blk.").and_then(|r| r.split_once('.')) else {
            continue;
        };
        let Ok(block) = block.parse::<u32>() else { continue };
        if !is_routed_expert(tail) {
            continue;
        }
        let evicted = block < whole
            || (block == whole
                && EXPERT_KINDS
                    .iter()
                    .take(tensors as usize)
                    .any(|k| tail.starts_with(&format!("ffn_{k}_"))));
        if evicted {
            let start = base + unsafe { sys::gguf_get_tensor_offset(gguf, i) };
            out.push((start, start + unsafe { sys::gguf_get_tensor_size(gguf, i) }));
        }
    }
    unsafe { sys::gguf_free(gguf) };
    out.sort_unstable();
    // Tensors of one block sit side by side in the file; one registration per
    // run rather than per tensor.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (s, e) in out {
        match merged.last_mut() {
            Some(last) if s <= last.1 + 4096 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    merged
}

/// Pin the host-side experts of `path`. Returns the bytes pinned.
pub fn pin_host_experts(path: &Path, whole: u32, tensors: u32) -> u64 {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (path, whole, tensors);
        0
    }
    #[cfg(feature = "cuda")]
    {
        // ggml only registers when asked to, process-wide.
        if std::env::var_os("GGML_CUDA_REGISTER_HOST").is_none() {
            // SAFETY: set once, before any thread of ours reads the environment
            // for this variable.
            unsafe { std::env::set_var("GGML_CUDA_REGISTER_HOST", "1") };
        }
        let maps = mappings(path);
        let page = 4096usize;
        let mut pinned = 0u64;
        for (fs, fe) in host_expert_ranges(path, whole, tensors) {
            let Some(m) = maps.iter().find(|m| fs >= m.offset && fe <= m.offset + (m.end - m.start))
            else {
                continue;
            };
            let a = (m.start + (fs - m.offset)) & !(page - 1);
            let b = (m.start + (fe - m.offset) + page - 1) & !(page - 1);
            // SAFETY: the range lies inside a live read-only mapping of the
            // model file, which llama.cpp keeps for the model's lifetime.
            if unsafe { ggml_backend_cuda_register_host_buffer(a as *mut _, b - a) } {
                pinned += (b - a) as u64;
            }
        }
        pinned
    }
}
