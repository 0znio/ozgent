//! Finding installed models, by `name:tag` or by alias.
//!
//! Aliases are short unique nicknames — `ozgent run coder` rather than
//! `ozgent run Qwen3-Coder-30B-A3B-Instruct:Q4_K_M`. They live in each model's
//! own manifest rather than a central index, so deleting a model directory
//! takes its alias with it and can never leave a dangling entry behind.
//!
//! Uniqueness is therefore checked by scanning the manifests. There are only
//! ever a handful of small JSON files, so this costs nothing measurable.

use crate::manifest::{MANIFEST_FILE, Manifest};
use crate::paths::{ModelRef, Paths};
use std::path::PathBuf;

/// An installed model.
#[derive(Debug, Clone)]
pub struct Installed {
    pub model: ModelRef,
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl Installed {
    /// What the user should type to run it: the alias when it has one.
    pub fn short_name(&self) -> String {
        self.manifest
            .alias
            .clone()
            .unwrap_or_else(|| self.model.to_string())
    }
}

/// Every model under `models/`.
///
/// Names may contain a registry namespace (`unsloth/gemma4`), so the walk is
/// depth-limited rather than assuming exactly two levels. A manifest that
/// fails to parse is skipped with a warning: one broken model must not hide
/// the others.
pub fn installed(paths: &Paths) -> Vec<Installed> {
    let mut out = Vec::new();
    let mut stack = vec![(paths.models_dir(), 0usize)];

    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if path.join(MANIFEST_FILE).is_file() {
                match Manifest::load(&path) {
                    Ok(manifest) => out.push(Installed {
                        model: manifest.model_ref(),
                        dir: path,
                        manifest,
                    }),
                    Err(e) => tracing::warn!("skipping {}: {e}", path.display()),
                }
            } else if depth < 3 {
                stack.push((path, depth + 1));
            }
        }
    }
    out.sort_by(|a, b| a.model.cmp(&b.model));
    out
}

/// Resolve what the user typed into an installed model.
///
/// An alias is tried first and must match exactly. Because an alias may not
/// contain `:`, and a `name:tag` always does once defaulted, the two spaces
/// cannot collide — so the order is a convenience, not a tie-break.
pub fn resolve(paths: &Paths, input: &str) -> Result<Installed, RegistryError> {
    let input = input.trim();
    if input.is_empty() {
        return Err(RegistryError::Empty);
    }

    let models = installed(paths);

    if let Some(hit) = models
        .iter()
        .find(|m| m.manifest.alias.as_deref() == Some(input))
    {
        return Ok(hit.clone());
    }

    // Fall back to an exact `name:tag`, or `name` with the default tag.
    if let Ok(reference) = ModelRef::parse(input) {
        if let Some(hit) = models.iter().find(|m| m.model == reference) {
            return Ok(hit.clone());
        }
        // `ozgent run gemma4` when only `gemma4:12b` exists is unambiguous if
        // there is exactly one tag; guessing between several would not be.
        let by_name: Vec<&Installed> =
            models.iter().filter(|m| m.model.name == reference.name).collect();
        if by_name.len() == 1 {
            return Ok(by_name[0].clone());
        }
        if by_name.len() > 1 {
            return Err(RegistryError::Ambiguous {
                input: input.to_string(),
                candidates: by_name.iter().map(|m| m.model.to_string()).collect(),
            });
        }
    }

    Err(RegistryError::NotFound {
        input: input.to_string(),
        available: models.iter().map(|m| m.short_name()).collect(),
    })
}

/// Longest alias length, for aligning a listing.
pub fn alias_width(models: &[Installed]) -> usize {
    models
        .iter()
        .filter_map(|m| m.manifest.alias.as_ref())
        .map(String::len)
        .max()
        .unwrap_or(0)
}

/// Check an alias is well-formed and unused.
///
/// `exclude` is the model being renamed, so re-setting a model's own alias is
/// not reported as a collision.
pub fn validate_alias(
    paths: &Paths,
    alias: &str,
    exclude: Option<&ModelRef>,
) -> Result<String, RegistryError> {
    let alias = alias.trim();
    if alias.is_empty() {
        return Err(RegistryError::BadAlias {
            alias: alias.to_string(),
            reason: "an alias cannot be empty".into(),
        });
    }
    if alias.len() > 64 {
        return Err(RegistryError::BadAlias {
            alias: alias.to_string(),
            reason: "an alias must be 64 characters or fewer".into(),
        });
    }
    // A colon would make the alias indistinguishable from a `name:tag`.
    if alias.contains(':') {
        return Err(RegistryError::BadAlias {
            alias: alias.to_string(),
            reason: "an alias cannot contain ':', which would look like a name:tag".into(),
        });
    }
    if alias.starts_with('-') {
        return Err(RegistryError::BadAlias {
            alias: alias.to_string(),
            reason: "an alias cannot start with '-', which would look like a flag".into(),
        });
    }
    if !alias
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(RegistryError::BadAlias {
            alias: alias.to_string(),
            reason: "an alias may contain only letters, digits, '-', '_' and '.'".into(),
        });
    }

    if let Some(owner) = installed(paths).into_iter().find(|m| {
        m.manifest.alias.as_deref() == Some(alias) && Some(&m.model) != exclude
    }) {
        return Err(RegistryError::AliasTaken {
            alias: alias.to_string(),
            owner: owner.model.to_string(),
        });
    }
    Ok(alias.to_string())
}

/// Set or clear a model's alias, after checking it is free.
pub fn set_alias(
    paths: &Paths,
    model: &ModelRef,
    alias: Option<&str>,
) -> Result<(), RegistryError> {
    let dir = paths.model_dir(model);
    let mut manifest = Manifest::load(&dir).map_err(|e| RegistryError::Manifest(e.to_string()))?;

    manifest.alias = match alias {
        Some(a) => Some(validate_alias(paths, a, Some(model))?),
        None => None,
    };
    manifest
        .save(&dir)
        .map_err(|e| RegistryError::Manifest(e.to_string()))
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("no model given")]
    Empty,

    #[error(
        "{input:?} is not installed.{}",
        if available.is_empty() {
            " No models are installed. Try: ozgent pull <huggingface-repo>".to_string()
        } else {
            format!(" Available: {}", available.join(", "))
        }
    )]
    NotFound { input: String, available: Vec<String> },

    #[error("{input:?} matches several models: {}. Give the tag too.", candidates.join(", "))]
    Ambiguous { input: String, candidates: Vec<String> },

    #[error("the alias {alias:?} is already used by {owner}")]
    AliasTaken { alias: String, owner: String },

    #[error("{alias:?} is not a usable alias: {reason}")]
    BadAlias { alias: String, reason: String },

    #[error("{0}")]
    Manifest(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Home(PathBuf);

    impl Home {
        fn new(label: &str) -> Self {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let d = std::env::temp_dir().join(format!(
                "ozgent-registry-{label}-{}-{}",
                std::process::id(),
                N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
        fn paths(&self) -> Paths {
            Paths::with_root(&self.0)
        }
        fn add(&self, reference: &str, alias: Option<&str>) -> ModelRef {
            let r = ModelRef::parse(reference).unwrap();
            let dir = self.paths().model_dir(&r);
            std::fs::create_dir_all(&dir).unwrap();
            let mut m = Manifest::new(&r, "model.gguf");
            m.alias = alias.map(str::to_string);
            m.save(&dir).unwrap();
            r
        }
    }

    impl Drop for Home {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resolves_by_alias() {
        let h = Home::new("alias");
        h.add("Qwen3-Coder-30B:Q4_K_M", Some("coder"));
        let got = resolve(&h.paths(), "coder").unwrap();
        assert_eq!(got.model.to_string(), "Qwen3-Coder-30B:Q4_K_M");
    }

    #[test]
    fn resolves_by_full_reference() {
        let h = Home::new("full");
        h.add("gemma4:12b", Some("g"));
        assert_eq!(resolve(&h.paths(), "gemma4:12b").unwrap().model.tag, "12b");
    }

    #[test]
    fn a_bare_name_resolves_when_only_one_tag_exists() {
        let h = Home::new("bare");
        h.add("gemma4:12b", None);
        assert_eq!(resolve(&h.paths(), "gemma4").unwrap().model.tag, "12b");
    }

    #[test]
    fn a_bare_name_with_several_tags_is_ambiguous_rather_than_guessed() {
        let h = Home::new("ambig");
        h.add("gemma4:12b", None);
        h.add("gemma4:27b", None);
        let err = resolve(&h.paths(), "gemma4").unwrap_err().to_string();
        assert!(err.contains("several"), "{err}");
        assert!(err.contains("12b") && err.contains("27b"), "{err}");
    }

    #[test]
    fn an_unknown_model_lists_what_is_available_by_short_name() {
        let h = Home::new("unknown");
        h.add("gemma4:12b", Some("g4"));
        let err = resolve(&h.paths(), "nope").unwrap_err().to_string();
        assert!(err.contains("g4"), "should suggest the alias: {err}");
    }

    #[test]
    fn an_empty_registry_says_how_to_get_a_model() {
        let h = Home::new("empty");
        let err = resolve(&h.paths(), "anything").unwrap_err().to_string();
        assert!(err.contains("ozgent pull"), "{err}");
    }

    #[test]
    fn a_duplicate_alias_is_refused_and_names_the_owner() {
        let h = Home::new("dup");
        h.add("gemma4:12b", Some("fast"));
        h.add("qwen3:8b", None);

        let err = validate_alias(&h.paths(), "fast", Some(&ModelRef::parse("qwen3:8b").unwrap()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("already used by"), "{err}");
        assert!(err.contains("gemma4:12b"), "should name the owner: {err}");
    }

    #[test]
    fn a_model_may_keep_its_own_alias_when_re_saved() {
        let h = Home::new("self");
        let r = h.add("gemma4:12b", Some("fast"));
        // Excluding itself, so this is not a collision.
        assert!(validate_alias(&h.paths(), "fast", Some(&r)).is_ok());
    }

    #[test]
    fn malformed_aliases_are_rejected_with_the_reason() {
        let h = Home::new("bad");
        h.add("gemma4:12b", None);
        let p = h.paths();

        for (alias, expect) in [
            ("", "empty"),
            ("has:colon", "':'"),
            ("-leading", "'-'"),
            ("has space", "only letters"),
            ("has/slash", "only letters"),
        ] {
            let err = validate_alias(&p, alias, None).unwrap_err().to_string();
            assert!(err.contains(expect), "for {alias:?}: {err}");
        }
        assert!(validate_alias(&p, &"x".repeat(65), None).is_err(), "too long");
    }

    #[test]
    fn good_aliases_are_accepted() {
        let h = Home::new("good");
        h.add("gemma4:12b", None);
        for alias in ["coder", "fast-7b", "my_model", "v1.2", "A1"] {
            assert!(validate_alias(&h.paths(), alias, None).is_ok(), "{alias} should be valid");
        }
    }

    #[test]
    fn setting_an_alias_persists_and_becomes_resolvable() {
        let h = Home::new("set");
        let r = h.add("gemma4:12b", None);
        set_alias(&h.paths(), &r, Some("quick")).unwrap();

        assert_eq!(resolve(&h.paths(), "quick").unwrap().model, r);
        // And the old reference still works.
        assert_eq!(resolve(&h.paths(), "gemma4:12b").unwrap().model, r);
    }

    #[test]
    fn an_alias_can_be_cleared_and_then_reused_elsewhere() {
        let h = Home::new("clear");
        let a = h.add("gemma4:12b", Some("fast"));
        let b = h.add("qwen3:8b", None);

        set_alias(&h.paths(), &a, None).unwrap();
        set_alias(&h.paths(), &b, Some("fast")).unwrap();
        assert_eq!(resolve(&h.paths(), "fast").unwrap().model, b);
    }

    #[test]
    fn short_name_prefers_the_alias() {
        let h = Home::new("short");
        h.add("gemma4:12b", Some("g4"));
        h.add("qwen3:8b", None);
        let models = installed(&h.paths());
        let names: Vec<String> = models.iter().map(Installed::short_name).collect();
        assert!(names.contains(&"g4".to_string()));
        assert!(names.contains(&"qwen3:8b".to_string()));
    }
}
