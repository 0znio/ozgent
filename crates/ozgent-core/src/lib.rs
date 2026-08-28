//! Shared types for ozgent: filesystem layout, configuration, model manifests,
//! and the conversation model. This crate performs no inference and spawns no
//! processes, so every other crate can depend on it freely.

pub mod accel;
pub mod chat;
pub mod config;
pub mod datetime;
pub mod manifest;
pub mod options;
pub mod paths;
pub mod registry;
pub mod tokens;

pub use accel::{CacheType, MoeOffload, PrefixReuse, Speculative, SpeculativeTuning};
pub use chat::{ImageSource, Message, Part, Role, ToolCall, ToolSpec};
pub use config::Config;
pub use options::ReasoningEffort;
pub use datetime::DateTime;
pub use manifest::{Capability, Manifest};
pub use options::{GpuLayers, Options, Resolved, ThinkingMode};
pub use paths::{ModelRef, Paths};
pub use registry::{Installed, RegistryError, installed, resolve, set_alias, validate_alias};
pub use tokens::{format_count, parse_count};
