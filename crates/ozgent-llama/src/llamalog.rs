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

/// Whether llama.cpp said flash attention resolved on, off, or said nothing.
///
/// It announces the answer — "Flash Attention enabled", or "not supported, set
/// to disabled" — and ozgent used to throw the line away, because the sink
/// keeps error level only. So the KV policy *guessed* at a feature llama.cpp
/// had already reported, and found out it had guessed wrong by having a
/// context refused. Reading the answer is strictly better than inferring it
/// from a failure.
static FLASH: Mutex<Option<bool>> = Mutex::new(None);

/// What llama.cpp decided about flash attention for the last context, if it
/// said. `None` means it has not been resolved yet in this process.
pub fn flash_attention() -> Option<bool> {
    FLASH.lock().ok().and_then(|f| *f)
}

/// Compute buffers llama.cpp reported for the last context it built, in bytes.
///
/// It prints these itself — "CUDA0 compute buffer size = 1234.56 MiB" — and
/// ozgent was modelling the same quantity from first principles and getting it
/// wrong. Reading the number llama.cpp already published is strictly better
/// than deriving it: it needs no assumption about which buffers exist, it
/// follows llama.cpp's own changes, and it is exact.
static COMPUTE_BYTES: Mutex<u64> = Mutex::new(0);
static RS_BYTES: Mutex<u64> = Mutex::new(0);

/// Total compute buffer bytes since the last [`clear`].
pub fn compute_buffers() -> u64 {
    COMPUTE_BYTES.lock().map(|b| *b).unwrap_or(0)
}

/// Record a resolution line. Public for tests; called from the sink.
pub fn note_line(line: &str) {
    // "resolve_fused_ops: Flash Attention enabled" / "... not supported, set
    // to disabled". Matched on both halves so an unrelated line mentioning
    // flash attention cannot flip it.
    if !line.contains("Flash Attention") {
        return;
    }
    let verdict = if line.contains("not supported") || line.contains("disabled") {
        Some(false)
    } else if line.contains("enabled") {
        Some(true)
    } else {
        None
    };
    if let (Some(v), Ok(mut f)) = (verdict, FLASH.lock()) {
        *f = Some(v);
    }
}

/// Add up a "compute buffer size = N MiB" line, whichever device it names.
fn note_buffer(line: &str) {
    if let Some(rest) = line.split_once("RS buffer size =").map(|(_, r)| r) {
        if let Some(mib) = rest.split_whitespace().next().and_then(|n| n.parse::<f64>().ok()) {
            if let Ok(mut total) = RS_BYTES.lock() {
                *total += (mib * 1024.0 * 1024.0) as u64;
            }
        }
        return;
    }
    let Some(rest) = line.split_once("compute buffer size =").map(|(_, r)| r) else { return };
    let Some(mib) = rest.split_whitespace().next().and_then(|n| n.parse::<f64>().ok()) else {
        return;
    };
    if let Ok(mut total) = COMPUTE_BYTES.lock() {
        *total += (mib * 1024.0 * 1024.0) as u64;
    }
}

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
    if let Ok(mut b) = COMPUTE_BYTES.lock() {
        *b = 0;
    }
    if let Ok(mut b) = RS_BYTES.lock() {
        *b = 0;
    }
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
    // SAFETY: llama.cpp always passes a NUL-terminated string.
    let chunk = unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned();
    if std::env::var_os("OZ_LLAMA_LOG").is_some() {
        eprint!("{chunk}");
    }
    // Read at every level, because the answer arrives at info or warn — this
    // is the one non-error line worth keeping.
    note_line(&chunk);
    note_buffer(&chunk);
    if !IN_ERROR.load(Ordering::Relaxed) {
        return;
    }

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
    // Also logged, not only kept.
    //
    // These are retained so a null context can be explained afterwards, but
    // some of them precede a `GGML_ABORT` — and then there is no afterwards.
    // A crash whose one explanatory line was held in a buffer nobody lived to
    // read is the worst version of this: the log showed the abort and not the
    // reason for it.
    tracing::error!(target: "llama", "{line}");
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

#[cfg(test)]
mod flash_verdict_tests {
    use super::*;

    fn read(line: &str) -> Option<bool> {
        if let Ok(mut f) = FLASH.lock() {
            *f = None;
        }
        note_line(line);
        flash_attention()
    }

    #[test]
    fn llama_cpps_verdict_is_read_rather_than_guessed_at() {
        // One test, not four: the verdict is process-global, so separate
        // tests would each be clearing the state the others are reading.
        // Under `cargo test --workspace` that failed about one run in three.
        assert_eq!(read("resolve_fused_ops: Flash Attention enabled\n"), Some(true));

        // The line that used to be discarded, leaving the cache policy to
        // find out by having a context refused.
        assert_eq!(
            read("resolve_fused_ops: Flash Attention not supported, set to disabled\n"),
            Some(false)
        );

        // Nothing said yet is not the same as "off".
        assert_eq!(read("llama_model_loader: loaded meta data\n"), None);

        // And a line that merely mentions the feature decides nothing.
        assert_eq!(read("Flash Attention is a thing that exists\n"), None);
    }

    #[test]
    fn compute_buffer_lines_are_added_up() {
        // Read at info level, which the sink keeps for this one purpose: the
        // figure ozgent used to model from first principles and get wrong by
        // a factor of eight.
        if let Ok(mut b) = COMPUTE_BYTES.lock() {
            *b = 0;
        }
        note_buffer("llama_context:      CUDA0 compute buffer size =   548.01 MiB\n");
        note_buffer("llama_context:  CUDA_Host compute buffer size =    20.01 MiB\n");
        let total = compute_buffers();
        assert!(total > 560 * 1024 * 1024 && total < 572 * 1024 * 1024, "{total}");
    }
}
