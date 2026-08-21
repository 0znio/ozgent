//! Acquiring models from Hugging Face.
//!
//! Given a repository id, ozgent picks the right GGUF among the dozens a repo
//! usually holds, finds the vision projector if there is one, downloads with
//! resume, verifies checksums, and writes a manifest — so the user never has
//! to hunt for a file URL.

pub mod download;
pub mod hf;
pub mod import;
pub mod install;
pub mod progress;
pub mod select;

pub use download::{download, human};
pub use hf::{Client, HubError, RepoInfo};
pub use import::{ImportRequest, Imported, import, suggest_reference, verify_gguf};
pub use install::{Event, Installed, PullRequest, pull};
pub use progress::{Bar, bytes as human_bytes};
pub use select::{RepoFile, Selection, SelectError, derive_ref, quant_of, select};
