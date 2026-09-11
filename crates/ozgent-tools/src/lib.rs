//! The Python tool runtime, as seen from Rust.
//!
//! Tools are Python functions running in a separate supervised process. This
//! crate spawns that process, performs the handshake, converts the tool
//! manifests into [`ozgent_core::ToolSpec`]s the model can be shown, and
//! multiplexes calls over the pipe.

pub mod preamble;
pub mod host;
pub mod protocol;
pub mod source;
pub mod summary;

pub use host::{HostConfig, HostError, ToolCallError, ToolHost, resolve_runtime};
pub use protocol::{PROTOCOL_VERSION, RpcError, ToolManifest};
pub use preamble::{MEDIA_RULE, first_line, tool_preamble};
pub use source::{Boxed, Shadowed, Toolbox, ToolSource};
