//! Hugging Face repository client.
//!
//! Only three endpoints are needed: model metadata for tags, the file tree for
//! sizes and checksums, and `resolve` for the bytes themselves.

use crate::select::RepoFile;
use serde::Deserialize;

pub const HF_ENDPOINT: &str = "https://huggingface.co";

/// Environment variables holding an access token, in priority order. Gated
/// repositories (Llama, some Gemma mirrors) return 401 without one.
const TOKEN_ENVS: &[&str] = &["OZGENT_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"];

pub fn token_from_env() -> Option<String> {
    TOKEN_ENVS
        .iter()
        .find_map(|k| std::env::var(k).ok())
        .filter(|t| !t.trim().is_empty())
}

/// What a repository contains.
#[derive(Debug, Clone)]
pub struct RepoInfo {
    pub id: String,
    pub revision: String,
    pub files: Vec<RepoFile>,
    pub tags: Vec<String>,
    pub gated: bool,
}

impl RepoInfo {
    /// Whether the model accepts images, per the pipeline tags Hugging Face
    /// assigns. Confirmed separately by the presence of a projector.
    pub fn looks_multimodal(&self) -> bool {
        self.tags.iter().any(|t| {
            let t = t.to_ascii_lowercase();
            t.contains("image-text-to-text") || t.contains("multimodal") || t == "vision"
        })
    }
}

#[derive(Deserialize)]
struct ModelInfo {
    #[serde(default)]
    id: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    gated: serde_json::Value,
    #[serde(default)]
    sha: Option<String>,
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<Lfs>,
}

#[derive(Deserialize)]
struct Lfs {
    /// For LFS files this is the sha256 of the content.
    #[serde(default)]
    oid: Option<String>,
}

pub struct Client {
    http: reqwest::Client,
    endpoint: String,
    token: Option<String>,
}

impl Client {
    pub fn new() -> Result<Self, HubError> {
        Ok(Self {
            http: reqwest::Client::builder()
                .user_agent(concat!("ozgent/", env!("CARGO_PKG_VERSION")))
                // Large files over slow links must not hit a global deadline;
                // stall detection is handled per-chunk by the downloader.
                .connect_timeout(std::time::Duration::from_secs(30))
                .build()?,
            endpoint: std::env::var("HF_ENDPOINT").unwrap_or_else(|_| HF_ENDPOINT.to_string()),
            token: token_from_env(),
        })
    }

    pub fn with_token(mut self, token: Option<String>) -> Self {
        if token.is_some() {
            self.token = token;
        }
        self
    }

    fn authed(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(t) => req.bearer_auth(t),
            None => req,
        }
    }

    /// Fetch metadata and the full file listing.
    pub async fn repo(&self, repo_id: &str, revision: &str) -> Result<RepoInfo, HubError> {
        let info_url = format!("{}/api/models/{repo_id}", self.endpoint);
        let resp = self.authed(self.http.get(&info_url)).send().await?;
        let resp = self.check(resp, repo_id).await?;
        let info: ModelInfo = resp.json().await?;

        let tree_url = format!("{}/api/models/{repo_id}/tree/{revision}", self.endpoint);
        let resp = self
            .authed(self.http.get(&tree_url).query(&[("recursive", "true")]))
            .send()
            .await?;
        let resp = self.check(resp, repo_id).await?;
        let entries: Vec<TreeEntry> = resp.json().await?;

        let files = entries
            .into_iter()
            .filter(|e| e.kind == "file")
            .map(|e| RepoFile {
                path: e.path,
                size: e.size,
                sha256: e.lfs.and_then(|l| l.oid),
            })
            .collect();

        let gated = !matches!(info.gated, serde_json::Value::Bool(false));

        Ok(RepoInfo {
            id: if info.id.is_empty() { repo_id.to_string() } else { info.id },
            revision: info.sha.unwrap_or_else(|| revision.to_string()),
            files,
            tags: info.tags,
            gated,
        })
    }

    /// URL that serves a file's bytes.
    pub fn download_url(&self, repo_id: &str, revision: &str, path: &str) -> String {
        format!("{}/{repo_id}/resolve/{revision}/{path}", self.endpoint)
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    /// Turn HTTP failures into messages that say what to do next.
    async fn check(
        &self,
        resp: reqwest::Response,
        repo_id: &str,
    ) -> Result<reqwest::Response, HubError> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        Err(match status.as_u16() {
            404 => HubError::NotFound { repo: repo_id.to_string() },
            401 | 403 => HubError::Unauthorized {
                repo: repo_id.to_string(),
                has_token: self.token.is_some(),
            },
            429 => HubError::RateLimited,
            _ => HubError::Http {
                status: status.as_u16(),
                body: resp.text().await.unwrap_or_default().chars().take(300).collect(),
            },
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("repository {repo:?} not found on Hugging Face")]
    NotFound { repo: String },

    #[error(
        "access to {repo:?} was denied.{}",
        if *has_token {
            " Your token may lack access, or you may need to accept the model's licence on its Hugging Face page."
        } else {
            " It is probably a gated model: accept its licence on Hugging Face, then set HF_TOKEN."
        }
    )]
    Unauthorized { repo: String, has_token: bool },

    #[error("Hugging Face is rate-limiting; wait a moment and retry")]
    RateLimited,

    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },

    #[error("io error on {path}: {source}")]
    Io { path: String, source: std::io::Error },

    #[error("{file} is corrupt: expected sha256 {expected}, got {actual}")]
    ChecksumMismatch { file: String, expected: String, actual: String },

    #[error(transparent)]
    Select(#[from] crate::select::SelectError),

    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_urls_use_the_resolve_endpoint() {
        let c = Client::new().unwrap();
        let url = c.download_url("unsloth/gemma-3-12b-it-GGUF", "main", "mmproj-F16.gguf");
        assert!(url.ends_with("/unsloth/gemma-3-12b-it-GGUF/resolve/main/mmproj-F16.gguf"), "{url}");
    }

    #[test]
    fn gated_repo_errors_tell_the_user_what_to_do() {
        let without = HubError::Unauthorized { repo: "meta/x".into(), has_token: false };
        assert!(without.to_string().contains("HF_TOKEN"), "{without}");

        let with = HubError::Unauthorized { repo: "meta/x".into(), has_token: true };
        assert!(with.to_string().contains("licence"), "{with}");
    }

    #[test]
    fn multimodal_detection_reads_pipeline_tags() {
        let mk = |tags: &[&str]| RepoInfo {
            id: "x".into(),
            revision: "main".into(),
            files: vec![],
            tags: tags.iter().map(|s| s.to_string()).collect(),
            gated: false,
        };
        assert!(mk(&["gguf", "image-text-to-text"]).looks_multimodal());
        assert!(!mk(&["gguf", "text-generation"]).looks_multimodal());
    }
}
