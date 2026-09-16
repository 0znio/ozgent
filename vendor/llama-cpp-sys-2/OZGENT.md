# Why this crate is vendored

Upstream `llama-cpp-sys-2` 0.1.154 — the newest published version as of
2026-08-22 — ships a llama.cpp that knows `bailingmoe` and `bailingmoe2` but
not `bailingmoe3`, the architecture Ling 3.0 uses. Loading a Ling GGUF fails
with `unknown model architecture: 'bailingmoe3'`.

This is an unmodified copy of the 0.1.154 crate plus the C++ half of
[llama.cpp PR #26608](https://github.com/ggml-org/llama.cpp/pull/26608)
(merged upstream 2026-08-17, commit `3733366720`).

## What was taken

Only `src/llama-*` and `src/models/*` from that PR. The `gguf-py/` and
`conversion/` changes are the HF-to-GGUF converter, which nothing here runs —
the GGUF already exists — and `tests/` is not built by this crate.

## What had to be adapted

The PR is written against a newer llama.cpp than 0.1.154 vendors, so seven
hunks needed hand-fitting. Each is a context mismatch, not a behaviour change:

- **`LLM_KV_KDA_SAFE_GATE` / `LLM_KV_KDA_GATE_LOWER_BOUND`** did not exist.
  This tree's Kimi Linear predates both, so the enum entries, their GGUF
  spellings, the `hparams.kda_gate_lower_bound` field, and the model-saver
  round-trip were added alongside the existing `LLM_KV_KDA_HEAD_DIM`.
- **`llm_arch_is_hybrid` / `llm_arch_supports_sm_tensor`** list different
  architectures here, so `LLM_ARCH_BAILINGMOE3` was inserted by hand.
- **`mtp_on_hybrid_qwen`** is called `mtp_on_hybrid_qwen35` in this tree; the
  new arch went into that condition instead.
- **`add_kv<std::vector<float>>`** is already instantiated here, so the PR's
  copy was dropped as a duplicate.
- **`ml.load_mtp`** does not exist in this loader. Upstream uses it to decline
  the MTP block; here `qwen35moe` — the only other hybrid MTP arch — loads it
  unconditionally, so `bailingmoe3` now does too.

`src/CMakeLists.txt` globs `models/*.cpp`, so the new arch file is picked up
with no build-system change.

## When to delete this

As soon as a published `llama-cpp-sys-2` vendors a llama.cpp at or past
`3733366720`. Drop the `[patch.crates-io]` line in the workspace root, delete
this directory, and rebuild. Nothing else in ozgent depends on the patch —
`vendor/llama-cpp-2` is a separate and still-needed patch, for mtmd.

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
