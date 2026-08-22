//! The inference engine.

pub mod backend;
#[cfg(feature = "llama")]
pub mod cvec;
#[cfg(feature = "llama")]
pub mod embed;
pub mod effort;
pub mod engine;
pub mod grammar;
#[cfg(feature = "llama")]
pub mod layout;
#[cfg(feature = "llama")]
pub mod mtmd;
pub mod ngram;
pub mod thinking;
pub mod toolcall;
pub mod toolgate;
pub mod utf8;
pub mod vision;

pub use backend::{Device, FitEstimate, fit_to_vram};
pub use thinking::{Chunk, ThinkingFilter};
pub use toolcall::{Parsed, extract as extract_tool_calls};
pub use vision::{Extracted, extract};
