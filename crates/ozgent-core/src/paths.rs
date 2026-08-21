//! Filesystem layout for ozgent.
//!
//! Everything ozgent owns lives under a single root (`~/ozgent` by default,
//! overridable with `OZGENT_HOME`). The layout is deliberately flat and
//! self-describing so a user can delete any subtree by hand and leave no
//! residue anywhere else on the system:
//!
//! ```text
//! ~/ozgent/
//!   models/<name>/<tag>/     e.g. models/gemma4/12b
//!   configs/config.toml
//!   tools/                   user-authored python tools
//!   cache/
//!   logs/
//! ```

use std::path::{Path, PathBuf};

/// Environment variable that relocates the entire ozgent root.
pub const HOME_ENV: &str = "OZGENT_HOME";

/// Resolved locations of every directory ozgent reads or writes.
#[derive(Debug, Clone)]
pub struct Paths {
    root: PathBuf,
}

impl Paths {
    /// Resolve the ozgent root, honouring `OZGENT_HOME` and falling back to
    /// `~/ozgent`. Errors only if neither is available.
    pub fn discover() -> Result<Self, PathError> {
        if let Some(raw) = std::env::var_os(HOME_ENV) {
            let root = PathBuf::from(raw);
            if root.as_os_str().is_empty() {
                return Err(PathError::EmptyHomeEnv);
            }
            return Ok(Self { root });
        }
        let home = dirs::home_dir().ok_or(PathError::NoHomeDir)?;
        Ok(Self { root: home.join("ozgent") })
    }

    /// Build a `Paths` rooted at an explicit directory. Used by tests and by
    /// the `--root` CLI flag.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub fn configs_dir(&self) -> PathBuf {
        self.root.join("configs")
    }

    pub fn config_file(&self) -> PathBuf {
        self.configs_dir().join("config.toml")
    }

    pub fn tools_dir(&self) -> PathBuf {
        self.root.join("tools")
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.root.join("logs")
    }

    /// Directory holding every artifact for one model tag, e.g.
    /// `~/ozgent/models/gemma4/12b`. Deleting it removes the tag completely;
    /// deleting its parent removes every tag of that model.
    pub fn model_dir(&self, r: &ModelRef) -> PathBuf {
        self.models_dir().join(&r.name).join(&r.tag)
    }

    /// Create every directory ozgent expects to exist. Idempotent.
    pub fn ensure(&self) -> std::io::Result<()> {
        for dir in [
            self.models_dir(),
            self.configs_dir(),
            self.tools_dir(),
            self.cache_dir(),
            self.logs_dir(),
        ] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }
}

/// A model identified as `name:tag`, e.g. `gemma4:12b`.
///
/// The colon is a *display* convention only. On disk the two halves become
/// nested directories (`models/gemma4/12b`) so that no path component ever
/// contains a character that is illegal on NTFS or awkward in a shell.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelRef {
    pub name: String,
    pub tag: String,
}

/// Tag applied when the user names a model without one.
pub const DEFAULT_TAG: &str = "latest";

impl ModelRef {
    /// Parse `name`, `name:tag`, or a registry-qualified `ns/name:tag`.
    ///
    /// A missing tag defaults to [`DEFAULT_TAG`]. Each component is validated
    /// so that a hostile or careless model name can never escape the models
    /// directory via `..` or an absolute path.
    pub fn parse(s: &str) -> Result<Self, PathError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(PathError::EmptyModelRef);
        }
        let (name, tag) = match s.rsplit_once(':') {
            Some((n, t)) => (n, t),
            None => (s, DEFAULT_TAG),
        };
        let name = validate_component(name, "name")?;
        let tag = validate_component(tag, "tag")?;
        Ok(Self { name, tag })
    }
}

impl std::fmt::Display for ModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.name, self.tag)
    }
}

impl std::str::FromStr for ModelRef {
    type Err = PathError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Reject anything that could traverse out of the models directory or collide
/// with a reserved filesystem name. Model names may contain `/` (for registry
/// namespaces like `unsloth/gemma4`), so each segment is checked separately.
fn validate_component(raw: &str, field: &'static str) -> Result<String, PathError> {
    if raw.is_empty() {
        return Err(PathError::EmptyComponent(field));
    }
    for segment in raw.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(PathError::UnsafeComponent(field, raw.to_string()));
        }
        // Windows reserves these; forbidding them everywhere keeps a model
        // directory copyable between platforms.
        let bad = |c: char| c.is_control() || matches!(c, '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|');
        if segment.starts_with(' ')
            || segment.ends_with(' ')
            || segment.ends_with('.')
            || segment.contains(bad)
        {
            return Err(PathError::UnsafeComponent(field, raw.to_string()));
        }
    }
    Ok(raw.to_string())
}

#[derive(Debug, thiserror::Error)]
pub enum PathError {
    #[error("could not determine your home directory; set {HOME_ENV}")]
    NoHomeDir,
    #[error("{HOME_ENV} is set but empty")]
    EmptyHomeEnv,
    #[error("model reference is empty")]
    EmptyModelRef,
    #[error("model {0} is empty")]
    EmptyComponent(&'static str),
    #[error("model {0} {1:?} contains characters that are unsafe in a path")]
    UnsafeComponent(&'static str, String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_name_and_tag() {
        let r = ModelRef::parse("gemma4:12b").unwrap();
        assert_eq!(r.name, "gemma4");
        assert_eq!(r.tag, "12b");
        assert_eq!(r.to_string(), "gemma4:12b");
    }

    #[test]
    fn defaults_missing_tag() {
        assert_eq!(ModelRef::parse("gemma4").unwrap().tag, DEFAULT_TAG);
    }

    #[test]
    fn keeps_namespace_in_name() {
        let r = ModelRef::parse("unsloth/gemma4:27b").unwrap();
        assert_eq!(r.name, "unsloth/gemma4");
        assert_eq!(r.tag, "27b");
    }

    #[test]
    fn maps_to_nested_directories() {
        let p = Paths::with_root("/tmp/oz");
        let r = ModelRef::parse("gemma4:12b").unwrap();
        assert_eq!(p.model_dir(&r), Path::new("/tmp/oz/models/gemma4/12b"));
    }

    #[test]
    fn rejects_traversal() {
        for bad in ["../etc:12b", "gemma4:..", "a//b:1", "gemma4:", ""] {
            assert!(ModelRef::parse(bad).is_err(), "expected {bad:?} to be rejected");
        }
    }

    #[test]
    fn model_dir_stays_within_models_dir() {
        let p = Paths::with_root("/tmp/oz");
        let r = ModelRef::parse("unsloth/gemma4:27b").unwrap();
        assert!(p.model_dir(&r).starts_with(p.models_dir()));
    }
}
