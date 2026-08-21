//! Vision and audio input, through llama.cpp's `mtmd` library.
//!
//! A vision model is two files: the language model, and a projector that turns
//! pixels into embeddings the language model can attend to. ozgent already
//! downloaded both, but until now only the first was ever loaded — images were
//! parsed out of the prompt and then silently dropped.
//!
//! mtmd does the work: it decodes the image, runs the projector, and writes the
//! resulting embeddings straight into the llama context's KV cache. That last
//! part is why this cannot be expressed as tokens — an image becomes embeddings,
//! not token ids, so it has to be evaluated rather than tokenised.
//!
//! The unit of work is a *chunk*: `mtmd_tokenize` splits a prompt containing
//! media markers into alternating text and media chunks, and
//! `mtmd_helper_eval_chunks` evaluates them in order, advancing `n_past` past
//! everything it wrote.

use std::ffi::{CStr, CString};
use std::path::Path;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use ozgent_mtmd_sys as sys;

/// A loaded multimodal projector.
///
/// Owns a `mtmd_context`, which borrows the model it was built against — hence
/// the lifetime: the projector must not outlive the model.
pub struct Projector<'a> {
    ctx: *mut sys::mtmd_context,
    model: std::marker::PhantomData<&'a LlamaModel>,
}

// The context is only ever used from the one thread that owns the session.
unsafe impl Send for Projector<'_> {}

#[derive(Debug, thiserror::Error)]
pub enum MtmdError {
    #[error("loading the projector {path}: mtmd refused the file")]
    Load { path: String },
    #[error("{0}")]
    Invalid(String),
    #[error("this model's projector handles {supported}, not {kind}")]
    Unsupported { kind: &'static str, supported: &'static str },
    #[error("could not decode the attached file; is it a supported format?")]
    Bitmap,
    #[error("splitting the prompt into media chunks failed (code {0})")]
    Tokenize(i32),
    #[error("evaluating media chunks failed (code {0})")]
    Eval(i32),
}

/// One piece of media, as raw file bytes.
///
/// Images and audio travel the same path: mtmd detects the format from the
/// bytes, so nothing here needs to know which it is.
#[derive(Clone)]
pub struct Media {
    pub bytes: Vec<u8>,
}

// Printing an image's bytes is never useful, so the size stands in for them.
impl std::fmt::Debug for Media {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Media({} bytes)", self.bytes.len())
    }
}

impl<'a> Projector<'a> {
    /// Load `mmproj` and bind it to `model`.
    pub fn load(
        mmproj: &Path,
        model: &'a LlamaModel,
        use_gpu: bool,
        threads: i32,
    ) -> Result<Self, MtmdError> {
        let path = CString::new(mmproj.to_string_lossy().as_bytes())
            .map_err(|_| MtmdError::Invalid("the projector path contains a NUL byte".into()))?;

        let mut params = unsafe { sys::mtmd_context_params_default() };
        params.use_gpu = use_gpu;
        params.print_timings = false;
        params.n_threads = threads.max(1);

        // SAFETY: `path` outlives the call, and `model` outlives `self` by the
        // lifetime parameter.
        let ctx = unsafe { sys::mtmd_init_from_file(path.as_ptr(), model.as_ptr(), params) };
        if ctx.is_null() {
            return Err(MtmdError::Load { path: mmproj.display().to_string() });
        }
        Ok(Self { ctx, model: std::marker::PhantomData })
    }

    /// Whether this projector handles images at all; some are audio-only.
    pub fn supports_vision(&self) -> bool {
        unsafe { sys::mtmd_support_vision(self.ctx) }
    }

    /// Whether this projector handles audio.
    pub fn supports_audio(&self) -> bool {
        unsafe { sys::mtmd_support_audio(self.ctx) }
    }

    /// The marker that stands in for a piece of media in the prompt text.
    ///
    /// One marker per image, positioned where the image belongs in the
    /// conversation, because position is what the model attends over.
    pub fn marker(&self) -> &'static str {
        // SAFETY: mtmd returns a pointer to a static string.
        unsafe { CStr::from_ptr(sys::mtmd_default_marker()) }
            .to_str()
            .unwrap_or("<__media__>")
    }

    /// Evaluate `text` and its `images` into `context`, starting at `n_past`.
    ///
    /// `text` must contain exactly one [`marker`](Self::marker) per image.
    /// Returns the new `n_past`: everything up to it is now resident in the KV
    /// cache, images included.
    pub fn eval(
        &self,
        context: &mut LlamaContext,
        text: &str,
        images: &[Media],
        sources: &[ozgent_core::ImageSource],
        n_past: i32,
        n_batch: i32,
        want_logits: bool,
    ) -> Result<i32, MtmdError> {
        let c_text = CString::new(text)
            .map_err(|_| MtmdError::Invalid("the prompt contains a NUL byte".into()))?;

        // Bitmaps and chunks are C-owned; the guards below free them on every
        // exit path, including the error returns in between.
        // Check the capability first: mtmd's own failure for an audio file on a
        // vision-only projector is "could not read it", which sends the reader
        // looking for a corrupt file rather than the real answer.
        let supported = match (self.supports_vision(), self.supports_audio()) {
            (true, true) => "images and audio",
            (true, false) => "images",
            (false, true) => "audio",
            (false, false) => "nothing",
        };
        for source in sources {
            let is_audio = crate::vision::looks_like_audio(source);
            let ok = if is_audio { self.supports_audio() } else { self.supports_vision() };
            if !ok {
                return Err(MtmdError::Unsupported {
                    kind: if is_audio { "audio" } else { "images" },
                    supported,
                });
            }
        }

        let mut bitmaps = Vec::with_capacity(images.len());
        for image in images {
            let wrapper = unsafe {
                sys::mtmd_helper_bitmap_init_from_buf(
                    self.ctx,
                    image.bytes.as_ptr(),
                    image.bytes.len(),
                    false,
                )
            };
            if wrapper.bitmap.is_null() {
                for b in &bitmaps {
                    unsafe { sys::mtmd_bitmap_free(*b) };
                }
                return Err(MtmdError::Bitmap);
            }
            bitmaps.push(wrapper.bitmap);
        }

        let chunks = unsafe { sys::mtmd_input_chunks_init() };
        let result = self.eval_inner(
            context, &c_text, &bitmaps, chunks, n_past, n_batch, want_logits,
        );

        unsafe { sys::mtmd_input_chunks_free(chunks) };
        for b in bitmaps {
            unsafe { sys::mtmd_bitmap_free(b) };
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_inner(
        &self,
        context: &mut LlamaContext,
        text: &CString,
        bitmaps: &[*mut sys::mtmd_bitmap],
        chunks: *mut sys::mtmd_input_chunks,
        n_past: i32,
        n_batch: i32,
        want_logits: bool,
    ) -> Result<i32, MtmdError> {
        let input = sys::mtmd_input_text {
            text: text.as_ptr(),
            text_len: text.as_bytes().len(),
            // The prompt arrives already rendered through the chat template,
            // so special tokens are text to be honoured, not escaped.
            add_special: false,
            parse_special: true,
        };

        let mut handles: Vec<*const sys::mtmd_bitmap> =
            bitmaps.iter().map(|b| *b as *const _).collect();

        let code = unsafe {
            sys::mtmd_tokenize(
                self.ctx,
                chunks,
                &input,
                handles.as_mut_ptr(),
                handles.len(),
            )
        };
        if code != 0 {
            return Err(MtmdError::Tokenize(code));
        }

        let mut new_n_past: i32 = n_past;
        let code = unsafe {
            sys::mtmd_helper_eval_chunks(
                self.ctx,
                context.as_ptr(),
                chunks,
                n_past,
                0,
                n_batch.max(1),
                want_logits,
                &mut new_n_past,
            )
        };
        if code != 0 {
            return Err(MtmdError::Eval(code));
        }
        Ok(new_n_past)
    }
}

impl Drop for Projector<'_> {
    fn drop(&mut self) {
        // SAFETY: `ctx` came from `mtmd_init_from_file` and is freed once.
        unsafe { sys::mtmd_free(self.ctx) };
    }
}

/// Read every source into the bytes mtmd decodes.
///
/// Message construction deliberately does no I/O, so paths and URLs arrive
/// unread and are resolved here — at prompt-build time, where a failure can
/// still be reported to the user before the model is asked anything.
pub fn load_media(sources: &[ozgent_core::ImageSource]) -> Result<Vec<Media>, MtmdError> {
    let mut out = Vec::with_capacity(sources.len());
    for source in sources {
        let bytes = match source {
            ozgent_core::ImageSource::Bytes { bytes, .. } => bytes.clone(),
            ozgent_core::ImageSource::Path { path } => std::fs::read(path).map_err(|e| {
                MtmdError::Invalid(format!("cannot read {}: {e}", path.display()))
            })?,
            // Fetching is the caller's job: it needs an HTTP client and, in an
            // async front end, must not block the runtime.
            ozgent_core::ImageSource::Url { url } => {
                return Err(MtmdError::Invalid(format!(
                    "{url} must be downloaded before it can be used"
                )));
            }
        };
        if bytes.is_empty() {
            return Err(MtmdError::Invalid("an image was empty".into()));
        }
        out.push(Media { bytes });
    }
    Ok(out)
}

/// Place one media marker per image at the start of `text`.
///
/// Markers lead rather than trail because a question almost always refers back
/// to the image ("what is this?"), and the model attends over position.
pub fn with_markers(marker: &str, text: &str, count: usize) -> String {
    if count == 0 {
        return text.to_string();
    }
    let markers: String = std::iter::repeat_n(marker, count).collect();
    if text.trim().is_empty() {
        markers
    } else {
        format!("{markers}\n{text}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_with_a_nul_byte_is_refused_rather_than_truncated() {
        // CString would otherwise silently stop at the NUL, sending the model
        // half a prompt.
        let bad = "before\0after";
        assert!(CString::new(bad).is_err());
    }

    #[test]
    fn markers_lead_the_text_one_per_image() {
        assert_eq!(with_markers("<M>", "what is this?", 1), "<M>\nwhat is this?");
        assert_eq!(with_markers("<M>", "compare", 2), "<M><M>\ncompare");
        assert_eq!(with_markers("<M>", "", 1), "<M>", "an image alone needs no blank line");
        assert_eq!(with_markers("<M>", "no images", 0), "no images");
    }

    #[test]
    fn a_url_is_reported_rather_than_silently_skipped() {
        // Dropping it would leave the prompt with one fewer image than markers,
        // which mtmd rejects with a far less helpful message.
        let sources = vec![ozgent_core::ImageSource::Url { url: "https://x/y.png".into() }];
        let err = load_media(&sources).expect_err("must not succeed");
        assert!(err.to_string().contains("downloaded"), "{err}");
    }

    #[test]
    fn an_empty_image_is_refused() {
        let sources = vec![ozgent_core::ImageSource::Bytes { bytes: Vec::new(), mime: None }];
        assert!(load_media(&sources).is_err());
    }

    #[test]
    fn image_bytes_are_carried_verbatim() {
        // The decoder is stb_image inside mtmd; nothing here reinterprets the
        // bytes, so a PNG header must survive untouched.
        let png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let image = Media { bytes: png.clone() };
        assert_eq!(image.bytes, png);
    }
}
