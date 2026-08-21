//! The `manifest.json` that describes one model directory.
//!
//! A model directory is self-contained: the manifest names every file inside
//! it by relative path, so the directory can be moved, copied, or deleted
//! without consulting anything else.

use crate::options::Options;
use crate::paths::ModelRef;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bumped when the on-disk shape changes incompatibly.
pub const SCHEMA_VERSION: u32 = 1;

/// Filename of the manifest within a model directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// Things a model can do, which gate features at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Accepts images; requires `mmproj` to be present.
    Vision,
    /// Trained for tool calling.
    Tools,
    /// Emits reasoning traces that we can show or suppress.
    Thinking,
    /// Produces embeddings rather than chat completions.
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    pub name: String,
    pub tag: String,

    /// Short unique nickname, so `ozgent run coder` works instead of the full
    /// `name:tag`. Stored here rather than in a central index so that deleting
    /// the model directory removes the alias with it.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub alias: Option<String>,

    /// GGUF weights, relative to the model directory. Multiple entries mean a
    /// sharded model; llama.cpp is handed the first shard and finds the rest.
    pub weights: Vec<PathBuf>,

    /// Multimodal projector, required for [`Capability::Vision`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub mmproj: Option<PathBuf>,

    /// Overrides the chat template embedded in the GGUF metadata.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub chat_template: Option<PathBuf>,

    /// Default system prompt, read from this file if the user sets none.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub system_prompt: Option<PathBuf>,

    #[serde(default)]
    pub capabilities: Vec<Capability>,

    /// Quantisation label, for display only (e.g. `Q4_K_M`).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub quantization: Option<String>,

    /// Total size of all weight files, for display only.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub size_bytes: Option<u64>,

    /// Provenance of the download, so a model can be re-fetched or verified.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub source: Option<Source>,

    /// Per-model option layer, overriding `config.toml` but losing to the CLI.
    #[serde(default)]
    pub defaults: Options,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// Where it came from, e.g. `huggingface` or `url`.
    pub kind: String,
    pub uri: String,
    /// `sha256:...` digests keyed by the relative path they cover.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub digests: Vec<Digest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Digest {
    pub file: PathBuf,
    pub sha256: String,
}

impl Manifest {
    /// A minimal manifest for a single-file model, used by `ozgent import`.
    pub fn new(r: &ModelRef, weights: impl Into<PathBuf>) -> Self {
        Self {
            schema: SCHEMA_VERSION,
            name: r.name.clone(),
            tag: r.tag.clone(),
            alias: None,
            weights: vec![weights.into()],
            mmproj: None,
            chat_template: None,
            system_prompt: None,
            capabilities: Vec::new(),
            quantization: None,
            size_bytes: None,
            source: None,
            defaults: Options::default(),
        }
    }

    pub fn model_ref(&self) -> ModelRef {
        ModelRef { name: self.name.clone(), tag: self.tag.clone() }
    }

    pub fn has(&self, c: Capability) -> bool {
        self.capabilities.contains(&c)
    }

    /// Vision needs both the capability flag and an actual projector file.
    pub fn supports_vision(&self) -> bool {
        self.has(Capability::Vision) && self.mmproj.is_some()
    }

    pub fn load(dir: &Path) -> Result<Self, ManifestError> {
        let path = dir.join(MANIFEST_FILE);
        let bytes = std::fs::read(&path)
            .map_err(|e| ManifestError::Io { path: path.clone(), source: e })?;
        let m: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| ManifestError::Parse { path: path.clone(), source: e })?;
        if m.schema > SCHEMA_VERSION {
            return Err(ManifestError::FutureSchema { path, found: m.schema });
        }
        if m.weights.is_empty() {
            return Err(ManifestError::NoWeights { path });
        }
        Ok(m)
    }

    pub fn save(&self, dir: &Path) -> Result<(), ManifestError> {
        let path = dir.join(MANIFEST_FILE);
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| ManifestError::Parse { path: path.clone(), source: e })?;
        std::fs::write(&path, json).map_err(|e| ManifestError::Io { path, source: e })
    }

    /// Absolute path to the first weight shard, which is what llama.cpp loads.
    /// Absolute path to the vision projector, when this model has one.
    ///
    /// Paired with [`Manifest::supports_vision`]: the capability flag says a
    /// projector was installed, this says where it is.
    pub fn projector_path(&self, dir: &std::path::Path) -> Option<std::path::PathBuf> {
        self.mmproj.as_ref().map(|p| dir.join(p))
    }

    pub fn primary_weights(&self, dir: &Path) -> PathBuf {
        dir.join(&self.weights[0])
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("reading {path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("parsing {path}: {source}")]
    Parse { path: PathBuf, source: serde_json::Error },
    #[error("{path} uses schema {found}, newer than this build understands ({SCHEMA_VERSION}); upgrade ozgent")]
    FutureSchema { path: PathBuf, found: u32 },
    #[error("{path} lists no weight files")]
    NoWeights { path: PathBuf },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_a_directory() {
        let dir = std::env::temp_dir().join(format!("ozgent-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let r = ModelRef::parse("gemma4:12b").unwrap();
        let mut m = Manifest::new(&r, "model.gguf");
        m.capabilities = vec![Capability::Vision, Capability::Tools];
        m.mmproj = Some("mmproj.gguf".into());
        m.save(&dir).unwrap();

        let back = Manifest::load(&dir).unwrap();
        assert_eq!(back.model_ref(), r);
        assert!(back.supports_vision());
        assert_eq!(back.primary_weights(&dir), dir.join("model.gguf"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn vision_requires_a_projector() {
        let r = ModelRef::parse("gemma4:12b").unwrap();
        let mut m = Manifest::new(&r, "model.gguf");
        m.capabilities = vec![Capability::Vision];
        assert!(!m.supports_vision(), "capability without mmproj must not count");
    }
}
