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
