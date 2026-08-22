//! Control vectors: steering the model by nudging its hidden states.
//!
//! A control vector is one direction per layer, added to that layer's residual
//! stream during the forward pass. Scaling it up or down moves the model along
//! a behavioural axis — terse against verbose, formal against casual — without
//! spending a single prompt token on the instruction, and without the model
//! being able to ignore it the way it ignores a system prompt.
//!
//! The file format is llama.cpp's own: a GGUF holding tensors named
//! `direction.<layer>`, each `n_embd` floats, with layers numbered from 1. That
//! is deliberate rather than convenient — it means a vector produced by
//! llama.cpp's `cvector-generator`, or shared by anyone else, loads here
//! unchanged.

use std::ffi::{CStr, CString};
use std::path::Path;

use llama_cpp_2::context::LlamaContext;
use ozgent_mtmd_sys::llama_cpp_sys_2 as sys;

#[derive(Debug, thiserror::Error)]
pub enum CvecError {
    #[error("cannot read the control vector {path}: not a GGUF file, or unreadable")]
    Open { path: String },
    #[error("{0}")]
    Invalid(String),
    #[error("the control vector describes {found} embedding dimensions, but this model has {expected}")]
    WrongWidth { found: usize, expected: usize },
    #[error("llama.cpp refused the control vector (code {0})")]
    Apply(i32),
}

/// One direction per layer, flattened.
///
/// Laid out layer-major — layer `il` occupies `[(il - 1) * n_embd ..]` — which
/// is the shape `llama_set_adapter_cvec` expects, so applying it needs no copy.
#[derive(Clone)]
pub struct ControlVector {
    data: Vec<f32>,
    n_embd: usize,
    n_layers: usize,
}

impl std::fmt::Debug for ControlVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ControlVector({} layers x {})", self.n_layers, self.n_embd)
    }
}

impl ControlVector {
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Multiply every direction by `strength`.
    ///
    /// This is the whole user-facing control: negative reverses the axis and
    /// zero is a no-op. Push it far enough and the residual stream leaves the
    /// manifold the model was trained on, at which point output collapses —
    /// but "far enough" is a property of the vector, not a constant, so
    /// nothing here clamps it.
    pub fn scaled(&self, strength: f32) -> Self {
        Self {
            data: self.data.iter().map(|v| v * strength).collect(),
            n_embd: self.n_embd,
            n_layers: self.n_layers,
        }
    }

    /// Build one directly, for tests and for vectors computed in-process.
    pub fn from_parts(data: Vec<f32>, n_embd: usize) -> Result<Self, CvecError> {
        if n_embd == 0 || data.is_empty() || data.len() % n_embd != 0 {
            return Err(CvecError::Invalid(format!(
                "{} floats do not divide into rows of {n_embd}",
                data.len()
            )));
        }
        let n_layers = data.len() / n_embd;
        Ok(Self { data, n_embd, n_layers })
    }

    /// Read a control vector from a GGUF file produced for llama.cpp.
    pub fn load(path: &Path, expect_n_embd: usize) -> Result<Self, CvecError> {
        let c_path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| CvecError::Invalid("the path contains a NUL byte".into()))?;

        // `no_alloc: false` with a context out-parameter is what makes ggml
        // read the tensor *data*, not merely the header. Without it every
        // tensor comes back with a null data pointer.
        let mut ggml_ctx: *mut sys::ggml_context = std::ptr::null_mut();
        let params = sys::gguf_init_params { no_alloc: false, ctx: &mut ggml_ctx };

        // SAFETY: `c_path` outlives the call; the two out-pointers are freed
        // below on every path.
        let gguf = unsafe { sys::gguf_init_from_file(c_path.as_ptr(), params) };
        if gguf.is_null() {
            return Err(CvecError::Open { path: path.display().to_string() });
        }

        let result = Self::read_tensors(gguf, ggml_ctx, expect_n_embd);

        unsafe {
            sys::gguf_free(gguf);
            if !ggml_ctx.is_null() {
                sys::ggml_free(ggml_ctx);
            }
        }
        result
    }

    fn read_tensors(
        gguf: *mut sys::gguf_context,
        ggml_ctx: *mut sys::ggml_context,
        expect_n_embd: usize,
    ) -> Result<Self, CvecError> {
        let count = unsafe { sys::gguf_get_n_tensors(gguf) };
        if count <= 0 {
            return Err(CvecError::Invalid("the file holds no tensors".into()));
        }

        // Collected by layer index first, because GGUF makes no promise about
        // tensor order and a vector assembled in file order would apply the
        // wrong direction to every layer.
        let mut rows: Vec<(usize, Vec<f32>)> = Vec::new();
        for i in 0..count {
            let name = unsafe { CStr::from_ptr(sys::gguf_get_tensor_name(gguf, i)) }
                .to_string_lossy()
                .into_owned();
            let Some(index) = name.strip_prefix("direction.") else {
                continue; // not ours; other tooling stores metadata alongside
            };
            let layer: usize = index.parse().map_err(|_| {
                CvecError::Invalid(format!("tensor {name:?} does not end in a layer number"))
            })?;
            if layer == 0 {
                return Err(CvecError::Invalid(
                    "layers are numbered from 1 in this format; found direction.0".into(),
                ));
            }

            let c_name = CString::new(name.as_str())
                .map_err(|_| CvecError::Invalid("a tensor name contains a NUL byte".into()))?;
            let tensor = unsafe { sys::ggml_get_tensor(ggml_ctx, c_name.as_ptr()) };
            if tensor.is_null() {
                return Err(CvecError::Invalid(format!("{name} has no data")));
            }
            let len = unsafe { sys::ggml_nelements(tensor) } as usize;
            if len != expect_n_embd {
                return Err(CvecError::WrongWidth { found: len, expected: expect_n_embd });
            }
            let ptr = unsafe { sys::ggml_get_data_f32(tensor) };
            if ptr.is_null() {
                return Err(CvecError::Invalid(format!("{name} is not f32")));
            }
            // SAFETY: `len` elements, checked above against the model's width.
            let row = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
            rows.push((layer, row));
        }

        if rows.is_empty() {
            return Err(CvecError::Invalid(
                "no `direction.N` tensors; is this a control vector?".into(),
            ));
        }

        // Layers may be sparse — a vector trained on the middle of the stack
        // only. Missing layers become zero rows, which add nothing.
        let highest = rows.iter().map(|(l, _)| *l).max().unwrap_or(0);
        let mut data = vec![0.0f32; highest * expect_n_embd];
        for (layer, row) in rows {
            let start = (layer - 1) * expect_n_embd;
            data[start..start + expect_n_embd].copy_from_slice(&row);
        }
        Ok(Self { data, n_embd: expect_n_embd, n_layers: highest })
    }

    /// Install this vector on `context`, replacing any previous one.
    pub fn apply(&self, context: &mut LlamaContext) -> Result<(), CvecError> {
        // SAFETY: `data` lives for the call, and llama.cpp copies it.
        let code = unsafe {
            sys::llama_set_adapter_cvec(
                context.as_ptr(),
                self.data.as_ptr(),
                self.data.len(),
                self.n_embd as i32,
                1,
                self.n_layers as i32,
            )
        };
        if code != 0 { Err(CvecError::Apply(code)) } else { Ok(()) }
    }
}

/// Remove any control vector from `context`.
///
/// A null pointer is how llama.cpp spells "none"; the context otherwise keeps
/// steering every later turn, including ones the user never asked to steer.
pub fn clear(context: &mut LlamaContext) -> Result<(), CvecError> {
    let code = unsafe {
        sys::llama_set_adapter_cvec(context.as_ptr(), std::ptr::null(), 0, 0, 1, 1)
    };
    if code != 0 { Err(CvecError::Apply(code)) } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_must_divide_into_rows() {
        assert!(ControlVector::from_parts(vec![1.0; 10], 4).is_err(), "10 is not a multiple of 4");
        assert!(ControlVector::from_parts(Vec::new(), 4).is_err(), "empty is not a vector");
        assert!(ControlVector::from_parts(vec![1.0; 8], 0).is_err(), "zero width is nonsense");
        let ok = ControlVector::from_parts(vec![1.0; 12], 4).expect("3 rows of 4");
        assert_eq!(ok.n_layers(), 3);
        assert_eq!(ok.n_embd(), 4);
    }

    #[test]
    fn scaling_is_linear_and_zero_is_a_no_op() {
        let v = ControlVector::from_parts(vec![2.0, -4.0, 1.0, 0.5], 2).unwrap();
        assert_eq!(v.scaled(0.5).data, vec![1.0, -2.0, 0.5, 0.25]);
        assert_eq!(v.scaled(-1.0).data, vec![-2.0, 4.0, -1.0, -0.5]);
        assert!(v.scaled(0.0).data.iter().all(|f| *f == 0.0), "zero must steer nothing");
    }

    #[test]
    fn scaling_keeps_the_shape() {
        let v = ControlVector::from_parts(vec![1.0; 12], 4).unwrap();
        let s = v.scaled(3.0);
        assert_eq!((s.n_layers(), s.n_embd()), (3, 4));
    }

    #[test]
    fn a_missing_file_is_reported_rather_than_panicking() {
        let err = ControlVector::load(Path::new("/nonexistent/cvec.gguf"), 16)
            .expect_err("must not succeed");
        assert!(matches!(err, CvecError::Open { .. }), "{err}");
    }
}
