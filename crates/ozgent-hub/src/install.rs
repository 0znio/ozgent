//! Installing a model into `~/ozgent/models/<name>/<tag>/`.
//!
//! The directory is self-contained by design: original filenames are kept
//! (llama.cpp locates sibling shards by the `-00002-of-00005` pattern), the
//! manifest names every file by relative path, and nothing is written outside
//! it. Deleting the directory removes the model completely.

use crate::download::{Progress, download};
use crate::hf::{Client, HubError, RepoInfo};
use crate::select::{Selection, derive_ref, select};
use ozgent_core::manifest::{Capability, Manifest, Source};
use ozgent_core::{ModelRef, Paths};
use std::path::PathBuf;

/// What to install and where.
#[derive(Debug, Clone)]
pub struct PullRequest {
    /// Hugging Face repository, e.g. `unsloth/gemma-3-12b-it-GGUF`.
    pub repo_id: String,
    pub revision: String,
    /// Quantisation to take. `None` selects automatically.
    pub quant: Option<String>,
    /// Install under this `name:tag` instead of the derived one.
    pub as_ref: Option<String>,
    /// Largest acceptable download, used when `quant` is absent.
    pub budget_bytes: Option<u64>,
}

impl PullRequest {
    /// Parse `repo[:quant]`, the form the CLI accepts.
    ///
    /// A repository id always contains at most one `/`, so a colon after it is
    /// unambiguously the quantisation.
    pub fn parse(spec: &str) -> Self {
        let spec = spec.trim().trim_start_matches("hf:");
        let (repo, quant) = match spec.rsplit_once(':') {
            Some((r, q)) if !q.is_empty() && !q.contains('/') => (r, Some(q.to_string())),
            _ => (spec, None),
        };
        Self {
            repo_id: repo.to_string(),
            revision: "main".into(),
            quant,
            as_ref: None,
            budget_bytes: None,
        }
    }
}

/// The outcome of a successful pull.
#[derive(Debug, Clone)]
pub struct Installed {
    pub model: ModelRef,
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub bytes_transferred: u64,
}

/// Events worth showing the user during a pull.
pub enum Event<'a> {
    Resolved { repo: &'a RepoInfo, selection: &'a Selection, model: &'a ModelRef },
    FileStart { name: &'a str, index: usize, total: usize, size: u64 },
    FileProgress { name: &'a str, done: u64, total: u64 },
    FileDone { name: &'a str, skipped: bool },
}

/// Download and install a model.
pub async fn pull(
    client: &Client,
    paths: &Paths,
    request: &PullRequest,
    on_event: &(dyn Fn(Event<'_>) + Send + Sync),
) -> Result<Installed, HubError> {
    let repo = client.repo(&request.repo_id, &request.revision).await?;
    let selection = select(&repo.files, request.quant.as_deref(), request.budget_bytes)?;

    let reference = match &request.as_ref {
        Some(s) => s.clone(),
        None => derive_ref(&repo.id, &selection.quant),
    };
    let model = ModelRef::parse(&reference)
        .map_err(|e| HubError::Other(format!("{reference:?} is not a usable model name: {e}")))?;

    let dir = paths.model_dir(&model);
    std::fs::create_dir_all(&dir)
        .map_err(|e| HubError::Io { path: dir.display().to_string(), source: e })?;

    on_event(Event::Resolved { repo: &repo, selection: &selection, model: &model });

    // Weights first, then the projector: a partial install with weights but no
    // projector is at least a usable text model.
    let mut queue: Vec<&crate::select::RepoFile> = selection.weights.iter().collect();
    if let Some(mm) = &selection.mmproj {
        queue.push(mm);
    }

    let mut transferred = 0u64;
    for (i, file) in queue.iter().enumerate() {
        let name = file.path.rsplit('/').next().unwrap_or(&file.path).to_string();
        let dest = dir.join(&name);
        on_event(Event::FileStart { name: &name, index: i + 1, total: queue.len(), size: file.size });

        let url = client.download_url(&repo.id, &request.revision, &file.path);
        let cb_name = name.clone();
        let progress: Box<dyn Fn(u64, u64) + Send + Sync> = Box::new(move |done, total| {
            on_event(Event::FileProgress { name: &cb_name, done, total });
        });

        let moved = download(
            client.http(),
            &url,
            &dest,
            file.size,
            file.sha256.as_deref(),
            client.token(),
            Some(progress.as_ref() as Progress<'_>),
        )
        .await?;

        transferred += moved;
        on_event(Event::FileDone { name: &name, skipped: moved == 0 });
    }

    let manifest = build_manifest(&model, &repo, &selection, request);
    manifest
        .save(&dir)
        .map_err(|e| HubError::Other(format!("writing the manifest: {e}")))?;

    Ok(Installed { model, dir, manifest, bytes_transferred: transferred })
}

fn build_manifest(
    model: &ModelRef,
    repo: &RepoInfo,
    selection: &Selection,
    request: &PullRequest,
) -> Manifest {
    let mut manifest = Manifest::new(model, base_name(&selection.weights[0].path));
    manifest.weights = selection
        .weights
        .iter()
        .map(|f| PathBuf::from(base_name(&f.path)))
        .collect();
    manifest.mmproj = selection
        .mmproj
        .as_ref()
        .map(|f| PathBuf::from(base_name(&f.path)));
    manifest.quantization = Some(selection.quant.clone());
    manifest.size_bytes = Some(selection.total_bytes);

    // Vision counts only when a projector was actually installed; the tag
    // alone is not enough to run an image through the model.
    let mut capabilities = Vec::new();
    if manifest.mmproj.is_some() && repo.looks_multimodal() {
        capabilities.push(Capability::Vision);
    } else if manifest.mmproj.is_some() {
        capabilities.push(Capability::Vision);
    }
    manifest.capabilities = capabilities;

    manifest.source = Some(Source {
        kind: "huggingface".into(),
        uri: format!("{}@{}", repo.id, request.revision),
        digests: selection
            .weights
            .iter()
            .chain(selection.mmproj.iter())
            .filter_map(|f| {
                f.sha256.as_ref().map(|s| ozgent_core::manifest::Digest {
                    file: PathBuf::from(base_name(&f.path)),
                    sha256: s.clone(),
                })
            })
            .collect(),
    });

    manifest
}

fn base_name(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_bare_repository() {
        let r = PullRequest::parse("unsloth/gemma-3-12b-it-GGUF");
        assert_eq!(r.repo_id, "unsloth/gemma-3-12b-it-GGUF");
        assert_eq!(r.quant, None);
    }

    #[test]
    fn parses_a_repository_with_a_quantisation() {
        let r = PullRequest::parse("unsloth/gemma-3-12b-it-GGUF:Q4_K_M");
        assert_eq!(r.repo_id, "unsloth/gemma-3-12b-it-GGUF");
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn the_hf_prefix_is_optional() {
        let r = PullRequest::parse("hf:bartowski/Qwen3-8B-GGUF:Q6_K");
        assert_eq!(r.repo_id, "bartowski/Qwen3-8B-GGUF");
        assert_eq!(r.quant.as_deref(), Some("Q6_K"));
    }

    #[test]
    fn a_colon_inside_an_owner_is_not_mistaken_for_a_quantisation() {
        let r = PullRequest::parse("owner/name");
        assert_eq!(r.quant, None);
        assert_eq!(r.repo_id, "owner/name");
    }

    #[test]
    fn manifest_records_provenance_and_keeps_original_filenames() {
        let repo = RepoInfo {
            id: "unsloth/gemma-3-12b-it-GGUF".into(),
            revision: "main".into(),
            files: vec![],
            tags: vec!["image-text-to-text".into()],
            gated: false,
        };
        let selection = Selection {
            weights: vec![crate::select::RepoFile {
                path: "gemma-3-12b-it-Q4_K_M.gguf".into(),
                size: 100,
                sha256: Some("abc123".into()),
            }],
            mmproj: Some(crate::select::RepoFile {
                path: "mmproj-F16.gguf".into(),
                size: 50,
                sha256: None,
            }),
            quant: "Q4_K_M".into(),
            total_bytes: 150,
        };
        let model = ModelRef::parse("gemma-3-12b-it:Q4_K_M").unwrap();
        let request = PullRequest::parse("unsloth/gemma-3-12b-it-GGUF");

        let m = build_manifest(&model, &repo, &selection, &request);

        assert_eq!(m.weights[0], PathBuf::from("gemma-3-12b-it-Q4_K_M.gguf"));
        assert_eq!(m.mmproj, Some(PathBuf::from("mmproj-F16.gguf")));
        assert!(m.supports_vision(), "projector plus tag means vision works");
        assert_eq!(m.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(m.size_bytes, Some(150));

        let source = m.source.expect("provenance lets the model be re-verified");
        assert_eq!(source.kind, "huggingface");
        assert!(source.uri.contains("gemma-3-12b-it-GGUF@main"));
        assert_eq!(source.digests.len(), 1, "only the file with a digest is recorded");
    }

    #[test]
    fn a_text_only_model_gains_no_vision_capability() {
        let repo = RepoInfo {
            id: "x/y".into(),
            revision: "main".into(),
            files: vec![],
            tags: vec!["text-generation".into()],
            gated: false,
        };
        let selection = Selection {
            weights: vec![crate::select::RepoFile {
                path: "m-Q4_K_M.gguf".into(),
                size: 1,
                sha256: None,
            }],
            mmproj: None,
            quant: "Q4_K_M".into(),
            total_bytes: 1,
        };
        let m = build_manifest(
            &ModelRef::parse("m:Q4_K_M").unwrap(),
            &repo,
            &selection,
            &PullRequest::parse("x/y"),
        );
        assert!(!m.supports_vision());
        assert!(m.capabilities.is_empty());
    }
}
