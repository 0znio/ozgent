//! Raw bindings to llama.cpp's `mtmd` multimodal library.
//!
//! Safe wrappers live in `ozgent-llama`; this crate only builds and binds.

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Re-exported so crates that already depend on this one can reach the raw
/// llama.cpp bindings without taking a second direct dependency — which would
/// change feature resolution and force llama.cpp to rebuild.
pub use llama_cpp_sys_2;
