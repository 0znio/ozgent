# Why this crate is vendored

The crate is `llama-cpp-sys-2` 0.1.154's build glue around **upstream llama.cpp
v0.4.1** (tag `v0.4.1`, commit `391fac1`, released 2026-09-14) — newer than any
published crate, which trail upstream by weeks. v0.4.1 already contains
everything ozgent had patched in by hand before: `bailingmoe3` (Ling 3.0), the
recurrent rollback ring, and the NextN hooks. It also brings the DFlash /
DSpark drafter architectures.

## What is copied

`cmake/`, `CMakeLists.txt`, `common/`, `ggml/`, `include/`, `src/`,
`tools/mtmd/`, `vendor/`, `LICENSE`, `pocs/` and `convert_hf_to_gguf.py` from
the release tarball, unmodified. Nothing else is built.

## What had to change around it

- `wrapper_common.cpp`: `common_fit_params` gained an `extra` model argument
  (passed as `nullptr`), and `json_schema_to_grammar` now takes llama.cpp's
  own `common_json` rather than nlohmann's.
- `vendor/llama-cpp-2`: `llama_sampler_init_penalties` takes `n_vocab` (for
  backend-side sampling), and `llama_sampler_init_dry` no longer takes
  `n_ctx_train`.
- `crates/ozgent-mtmd-sys`: mtmd names bitmaps by a SHA-256 from
  `vendor/hash`, which upstream links as a library of its own. Compiled here
  with the languages upstream's CMake gives them — `sha1.c` as C++, `sha256.c`
  and `xxhash.c` as C, because `hash.cpp` includes them inside `extern "C"`.

## One local change to llama.cpp itself

`ggml/src/ggml-cpu/arch/x86/quants.c` gains an AVX2 `ggml_vec_dot_q2_0_q8_0`,
and the x86 line aliasing it to the generic version is removed from
`arch-fallback.h`. Upstream ships CUDA kernels for Q2_0 but only the scalar
loop on x86, so any Ternary Bonsai layer that does not fit on the card ran at
scalar speed. The kernel extracts the four 2-bit planes with one per-64-bit
shift and transposes the activations to match; measured 5.3x faster (207 vs
1100 ns on a 5120-wide row, Zen 5) and equal to the scalar result to float
rounding (2.3e-4 relative, over 2000 random rows of valid Q8_0 data).
Re-apply both edits after an upgrade until upstream has its own.

## Upgrading again

Copy the same directories from a newer release over `llama.cpp/`, rebuild, and
fix whatever the three places above report. Then measure decode against
`llama-bench` on the same GGUF: ozgent should match it within a few percent.

## Upstream tuning knobs that were measured and left off

llama.cpp carries several opt-in environment switches. All were measured on this
machine (RTX 5050 Laptop, 8 GB, sm_120, CUDA 13.3), interleaved and repeated,
against GLM-4.7-Flash with host-side experts and Qwen3.5-4B wholly resident:

| switch | what it does | result |
|---|---|---|
| `GGML_CUDA_GRAPH_OPT=1` | runs independent Q/K/V projections on concurrent streams | no change on either model |
| `GGML_CUDA_REGISTER_HOST=1` | `cudaHostRegister`s the mmap'd weights so host->device runs at full PCIe speed | GLM prefill +4%, decode flat |
| `GGML_OP_OFFLOAD_MIN_BATCH` | batch width above which weights are uploaded rather than computed on the CPU | not worth moving from its default of 32 |

The first is reported upstream as +17-27% on a 4090/5090; it does nothing here.
Worth re-measuring on a card with bandwidth to spare, which this one has not.

## The CUDA toolkit trap does not apply here

There is a widely repeated claim that Blackwell builds must use CUDA 12.8,
because 13.x segfaults the MMQ kernel and forces a cuBLAS fallback that is ~5x
slower at prompt processing. Checked rather than assumed: `cuobjdump` shows this
build carries real `sm_120a`/`sm_121a` SASS, `GGML_CUDA_FORCE_CUBLAS` is `OFF`
in the cache, and `ggml_cuda_should_use_mmq` returns true unconditionally on
this compute capability via `turing_mma_available`. We are on the fast path
under CUDA 13.3.
