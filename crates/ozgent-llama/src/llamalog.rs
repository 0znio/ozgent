//! Keep llama.cpp's last words.
//!
//! llama.cpp explains a failure through its log callback and then returns a
//! bare null pointer, so silencing that log — which every clean CLI wants —
//! turns "unknown model architecture: 'bailingmoe3'" into "null result from
//! llama cpp". The callback installed here keeps error-level lines in a small
//! ring instead of printing them, so a failed call can quote the real reason
//! while a successful one stays as quiet as before.

use std::collections::VecDeque;
use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;

/// How many error lines to keep. llama.cpp reports a failure in two or three
/// lines — the throw site, then the loader, then the entry point — and the
/// first is the informative one, so the ring has to outlast the summary.
const KEEP: usize = 8;

static ERRORS: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
/// llama.cpp logs a line in fragments, ending with the newline. Held here
/// until the line is complete so a reason is never split in half.
static PARTIAL: Mutex<String> = Mutex::new(String::new());
/// Whether the line currently being assembled began at error level. A `CONT`
/// fragment carries no level of its own, so without this the progress dots
/// that continue an info line would be filed as errors.
static IN_ERROR: AtomicBool = AtomicBool::new(false);

/// Route llama.cpp's output into the ring rather than onto stderr.
pub fn capture() {
    // SAFETY: the callback holds no borrowed state and takes no user data.
    unsafe { sys::llama_log_set(Some(sink), std::ptr::null_mut()) };
}

/// Drop anything held from an earlier call, so a reason cannot be misattributed.
pub fn clear() {
    if let Ok(mut e) = ERRORS.lock() {
        e.clear();
    }
    if let Ok(mut p) = PARTIAL.lock() {
        p.clear();
    }
    IN_ERROR.store(false, Ordering::Relaxed);
}

/// The most useful error line llama.cpp logged, if it logged one.
///
/// Prefers the earliest retained line: the loader's own summary ("failed to
/// load model") says nothing the caller does not already know, while the
/// line above it names the architecture, the tensor, or the file.
pub fn reason() -> Option<String> {
    let errors = ERRORS.lock().ok()?;
    errors.iter().find(|l| is_informative(l)).or_else(|| errors.front()).cloned()
}

/// Whether a line adds anything to "this call returned null".
fn is_informative(line: &str) -> bool {
    const EMPTY: [&str; 3] =
        ["failed to load model", "error loading model", "failed to load the model"];
    !EMPTY.iter().any(|generic| line.trim_end_matches(['.', ':']).ends_with(generic))
}

unsafe extern "C" fn sink(
    level: sys::ggml_log_level,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    if text.is_null() {
        return;
    }
    // A CONT fragment continues the previous line and carries no level of its
    // own, so it inherits the decision already made about that line.
    if level != sys::GGML_LOG_LEVEL_CONT {
        IN_ERROR.store(level == sys::GGML_LOG_LEVEL_ERROR, Ordering::Relaxed);
    }
    if !IN_ERROR.load(Ordering::Relaxed) {
        return;
    }
    // SAFETY: llama.cpp always passes a NUL-terminated string.
    let chunk = unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned();

    let Ok(mut partial) = PARTIAL.lock() else { return };
    partial.push_str(&chunk);
    while let Some(end) = partial.find('\n') {
        let line: String = partial.drain(..=end).collect();
        push(line.trim().to_string());
    }
    // A line this long is not a message llama.cpp meant to print; drop it
    // rather than let a missing newline grow the buffer without bound.
    if partial.len() > 4096 {
        partial.clear();
    }
}

fn push(line: String) {
    if line.is_empty() {
        return;
    }
    let Ok(mut errors) = ERRORS.lock() else { return };
    // Keep the first lines, not the last: llama.cpp names the cause where it
    // throws and only summarises on the way back out.
    if errors.len() < KEEP {
        errors.push_back(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_generic_summary_is_not_a_reason() {
        assert!(!is_informative("llama_model_load: error loading model"));
        assert!(!is_informative("common_init_from_params: failed to load model"));
    }

    #[test]
    fn the_line_that_names_the_cause_is_a_reason() {
        assert!(is_informative("error loading model architecture: unknown model architecture: 'bailingmoe3'"));
        assert!(is_informative("llama_model_load: error loading model: tensor 'blk.0.attn_q.weight' not found"));
    }
}
