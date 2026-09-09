//! Tools from Model Context Protocol servers.
//!
//! An MCP server is someone else's program offering tools over JSON-RPC —
//! either as a child process talking on its own stdin and stdout, or over
//! HTTP. This crate connects to them, turns what they list into the same
//! [`ToolSpec`](ozgent_core::ToolSpec) ozgent's own Python tools produce, and
//! implements [`ToolSource`](ozgent_tools::ToolSource) so that nothing above
//! it needs to know which kind of tool it is calling.
//!
//! The one place it deliberately does *not* treat them alike is trust. A
//! Python tool declares its effect in ozgent's own tree, on this machine. An
//! MCP tool's annotations are a claim by the program that wants to be run, so
//! they are ignored unless the operator marks the server trusted; see
//! [`protocol::effect_of`].

pub mod protocol;
pub mod server;
pub mod transport;

pub use protocol::{Failure, Listed, PROTOCOL_VERSION, ServerInfo};
pub use server::{ConnectError, Server, connect_all};
