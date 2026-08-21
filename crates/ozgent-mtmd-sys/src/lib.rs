//! Raw bindings to llama.cpp's `mtmd` multimodal library.
//!
//! Safe wrappers live in `ozgent-llama`; this crate only builds and binds.

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
