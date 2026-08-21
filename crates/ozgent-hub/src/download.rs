//! Resumable file download.
//!
//! Model files are gigabytes, so an interrupted download must not start over.
//! Bytes go to a `.part` file that is only renamed into place once complete
//! and verified, so a partial file can never be mistaken for a usable model.

use crate::hf::HubError;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// Progress callback: `(downloaded, total)` in bytes.
pub type Progress<'a> = &'a (dyn Fn(u64, u64) + Send + Sync);

/// Download `url` to `dest`, resuming if a partial file exists.
///
/// Returns the number of bytes actually transferred, which is zero when the
/// file was already complete.
pub async fn download(
    http: &reqwest::Client,
    url: &str,
    dest: &Path,
    expected_size: u64,
    expected_sha256: Option<&str>,
    token: Option<&str>,
    progress: Option<Progress<'_>>,
) -> Result<u64, HubError> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| HubError::Io { path: parent.display().to_string(), source: e })?;
    }

    // Already installed and the right size: nothing to do.
    if let Ok(meta) = std::fs::metadata(dest) {
        if expected_size == 0 || meta.len() == expected_size {
            if let Some(cb) = progress {
                cb(meta.len(), meta.len());
            }
            return Ok(0);
        }
    }

    let part = part_path(dest);
    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    // A partial larger than the target means the file changed upstream.
    if expected_size > 0 && have > expected_size {
        std::fs::remove_file(&part).ok();
        have = 0;
    }

    let mut request = http.get(url);
    if let Some(t) = token {
        request = request.bearer_auth(t);
    }
    if have > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }

    let resp = request.send().await?;
    let status = resp.status();

    if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        // The server considers the file fully sent; treat the part as done.
        finish(&part, dest, expected_sha256).await?;
        return Ok(0);
    }
    if !status.is_success() {
        return Err(match status.as_u16() {
            401 | 403 => HubError::Unauthorized { repo: url.to_string(), has_token: token.is_some() },
            404 => HubError::NotFound { repo: url.to_string() },
            429 => HubError::RateLimited,
            _ => HubError::Http { status: status.as_u16(), body: String::new() },
        });
    }

    // Critical: we asked to resume but the server sent the whole file. Appending
    // would silently corrupt it, so start over instead.
    let resuming = have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    if have > 0 && !resuming {
        tracing::debug!("server ignored the range request; restarting {}", dest.display());
        std::fs::remove_file(&part).ok();
        have = 0;
    }

    let total = if expected_size > 0 {
        expected_size
    } else {
        resp.content_length().unwrap_or(0) + have
    };

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resuming)
        .truncate(!resuming)
        .open(&part)
        .await
        .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;

    let mut written = have;
    let mut transferred = 0u64;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)
            .await
            .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;
        written += chunk.len() as u64;
        transferred += chunk.len() as u64;
        if let Some(cb) = progress {
            cb(written, total);
        }
    }

    file.flush()
        .await
        .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;
    drop(file);

    finish(&part, dest, expected_sha256).await?;
    Ok(transferred)
}

/// Verify the finished part file and move it into place.
async fn finish(part: &Path, dest: &Path, expected_sha256: Option<&str>) -> Result<(), HubError> {
    if let Some(expected) = expected_sha256 {
        let actual = sha256_file(part).await?;
        if !actual.eq_ignore_ascii_case(expected) {
            // Keep nothing corrupt on disk: a bad file that looks installed is
            // worse than no file.
            std::fs::remove_file(part).ok();
            return Err(HubError::ChecksumMismatch {
                file: dest.display().to_string(),
                expected: expected.to_string(),
                actual,
            });
        }
    }
    std::fs::rename(part, dest)
        .map_err(|e| HubError::Io { path: dest.display().to_string(), source: e })
}

pub async fn sha256_file(path: &Path) -> Result<String, HubError> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut file = std::fs::File::open(&path)
            .map_err(|e| HubError::Io { path: path.display().to_string(), source: e })?;
        let mut hasher = Sha256::new();
        // 1 MiB at a time: large enough to be fast, small enough not to spike
        // memory on a multi-gigabyte file.
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| HubError::Io { path: path.display().to_string(), source: e })?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    })
    .await
    .map_err(|e| HubError::Other(e.to_string()))?
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

/// Format bytes for progress output.
pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 { format!("{bytes} B") } else { format!("{v:.1} {}", UNITS[u]) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_files_sit_beside_the_target() {
        let p = part_path(Path::new("/models/gemma/12b/model.gguf"));
        assert_eq!(p, Path::new("/models/gemma/12b/model.gguf.part"));
    }

    #[test]
    fn sizes_are_readable() {
        assert_eq!(human(512), "512 B");
        assert_eq!(human(7_300_000_000), "6.8 GB");
    }

    #[tokio::test]
    async fn hashes_a_file() {
        let dir = std::env::temp_dir().join(format!("ozgent-dl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.bin");
        std::fs::write(&path, b"abc").unwrap();

        // Known sha256 of "abc".
        assert_eq!(
            sha256_file(&path).await.unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
