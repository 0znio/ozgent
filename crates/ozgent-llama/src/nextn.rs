//! The multi-token-prediction head, bound by hand.
//!
//! Qwen3.5 and its relatives append a trained NextN block past the end of the
//! main stack — `nextn_predict_layers` in the GGUF says how many — which
//! predicts the token *after* the one being decoded. It is a drafter the model
//! already carries: no second set of weights, no VRAM, and a draft drawn from
//! the model's own distribution rather than from repetition in the context.
//! That is the case n-gram drafting cannot serve. Measured on a 4B, ordinary
//! prose produced exactly zero n-gram drafts.
//!
//! **Why these are declared by hand.** They live in llama.cpp's
//! `src/llama-ext.h`, a staging header whose own comment says "breaking
//! changes and C++ are allowed… everything here should be considered WIP". It
//! is not wrapped in `extern "C"`, so the symbols carry C++ mangling, and it
//! includes `<map>`, so running bindgen over it drags the whole standard
//! library in. Naming four exported symbols explicitly is smaller, has no
//! build-time cost, and fails loudly at link time rather than subtly at
//! runtime if llama.cpp renames one.
//!
//! The mangling is Itanium ABI, which is what GCC and Clang emit on Linux and
//! macOS — the platforms ozgent builds for. MSVC would need different names,
//! and would fail to link rather than misbehave.

use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;

unsafe extern "C" {
    /// Ask a context to emit NextN embeddings alongside its logits.
    ///
    /// `masked` selects whether they come back for every token in the batch or
    /// only those with logits requested.
    #[link_name = "_Z26llama_set_embeddings_nextnP13llama_contextbb"]
    fn set_embeddings_nextn(ctx: *mut sys::llama_context, value: bool, masked: bool);

    /// The NextN embedding row for the `i`th output of the last decode.
    #[link_name = "_Z30llama_get_embeddings_nextn_ithP13llama_contexti"]
    fn get_embeddings_nextn_ith(ctx: *mut sys::llama_context, i: i32) -> *mut f32;

    /// Which appended NextN block to run, for models carrying more than one.
    #[link_name = "_Z28llama_set_nextn_layer_offsetP13llama_contexti"]
    fn set_nextn_layer_offset(ctx: *mut sys::llama_context, offset: i32);
}

/// Turn NextN embeddings on or off for a context.
///
/// # Safety
/// `ctx` must be a live llama.cpp context.
pub unsafe fn set_enabled(ctx: *mut sys::llama_context, on: bool, masked: bool) {
    unsafe { set_embeddings_nextn(ctx, on, masked) }
}

/// Choose which trained NextN head runs. Zero is the first, which is all a
/// model declaring `nextn_predict_layers = 1` has.
///
/// # Safety
/// `ctx` must be a live llama.cpp context.
pub unsafe fn set_head(ctx: *mut sys::llama_context, offset: i32) {
    unsafe { set_nextn_layer_offset(ctx, offset) }
}

/// Copy out the NextN embedding row for one output position.
///
/// Returns `None` when the context produced none — the feature is off, or the
/// decode requested no logits at that position. A null row is the documented
/// answer rather than a failure, so it is not an error here either.
///
/// # Safety
/// `ctx` must be a live llama.cpp context and `n_embd` its true row width, or
/// this reads past the end of llama.cpp's buffer.
pub unsafe fn embedding(ctx: *mut sys::llama_context, i: i32, n_embd: usize) -> Option<Vec<f32>> {
    let row = unsafe { get_embeddings_nextn_ith(ctx, i) };
    if row.is_null() || n_embd == 0 {
        return None;
    }
    Some(unsafe { std::slice::from_raw_parts(row, n_embd) }.to_vec())
}
