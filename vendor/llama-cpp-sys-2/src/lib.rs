//! See [llama-cpp-2](https://crates.io/crates/llama-cpp-2) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unpredictable_function_pointer_comparisons)]
// bindgen's bitfield accessors transmute between identical integer types.
#![allow(unnecessary_transmutes)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
