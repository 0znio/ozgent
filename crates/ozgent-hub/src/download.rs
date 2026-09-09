//! Resumable file download.
//!
//! Model files are gigabytes, so an interrupted download must not start over.
//! Bytes go to a `.part` file that is only renamed into place once complete
//! and verified, so a partial file can never be mistaken for a usable model.

use crate::hf::HubError;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};

/// Connections used for one file when the server supports ranges.
///
/// Hugging Face serves a single connection at a few MB/s regardless of the
/// link, so this is not about saturating the pipe — it is about not being
/// limited by one stream. Eight is where the measured gain flattens; more
/// mostly adds requests for the CDN to refuse.
pub const DEFAULT_CONNECTIONS: usize = 8;

/// Below this, one connection is simpler and no slower: the ranged requests
/// cost more in round trips than they save.
const PARALLEL_MIN_BYTES: u64 = 32 * 1024 * 1024;

/// Size of one slice.
///
/// Deliberately much smaller than "the file divided by the connection count".
/// Slices are the unit of resume, so one slice per connection means an
/// interrupted 2.7 GB download throws away up to 340 MB of finished work; at
/// this size it throws away at most this. Smaller slices also even out the
/// tail, since a slow connection holds up one slice rather than an eighth of
/// the file.
///
/// Sized against how long a slice takes rather than how big it is: with eight
/// connections sharing a few megabytes a second, sixteen megabytes lands in
/// about twenty seconds, so that is the most a stopped download loses.
const SLICE_BYTES: u64 = 16 * 1024 * 1024;

/// Ceiling on slice count, so a very large file does not turn into thousands
/// of requests and a ledger to match.
const MAX_SLICES: usize = 1024;

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

    // Several connections at once, where the file is big enough to be worth it
    // and the size is known — a plan cannot be made without one. Tried before
    // the sequential path rather than after a failure, because the sequential
    // path succeeds; it is just slow.
    let slices = plan(expected_size, connections());
    if !slices.is_empty() {
        match download_parallel(http, url, &part, expected_size, token, &slices, progress).await {
            Ok(Attempt::Done(n)) => {
                finish(&part, dest, expected_sha256).await?;
                return Ok(n);
            }
            // The server ignored the ranges. Whatever it did write is not
            // trustworthy at an offset, so start again on one connection.
            Ok(Attempt::NoRanges) => {
                tracing::debug!("{url} does not serve ranges; using one connection");
                std::fs::remove_file(&part).ok();
                std::fs::remove_file(ledger_path(&part)).ok();
            }
            Err(e) => return Err(e),
        }
    }

    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    // A partial larger than the target means the file changed upstream.
    if expected_size > 0 && have > expected_size {
        std::fs::remove_file(&part).ok();
        have = 0;
    }

    // A part file left by a parallel run is full-length with holes in it, not
    // a prefix. Appending to that would produce a file of the right size and
    // the wrong contents, which only the checksum would catch — and only if
    // there is one.
    if ledger_path(&part).exists() {
        std::fs::remove_file(&part).ok();
        std::fs::remove_file(ledger_path(&part)).ok();
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

/// What a parallel attempt concluded.
enum Attempt {
    /// Transferred this many bytes; the part file is complete.
    Done(u64),
    /// The server would not serve ranges. Nothing was written.
    NoRanges,
}

/// Fetch a file as several ranges at once.
///
/// Each slice writes straight to its own offset in the part file, so nothing
/// is buffered and the slices never touch each other's bytes. A sidecar
/// records which slices finished, which is what makes an interrupted parallel
/// download resumable — without it, a part file full of holes would look
/// exactly like a partial sequential download and be appended to.
#[allow(clippy::too_many_arguments)]
async fn download_parallel(
    http: &reqwest::Client,
    url: &str,
    part: &Path,
    total: u64,
    token: Option<&str>,
    slices: &[(u64, u64)],
    progress: Option<Progress<'_>>,
) -> Result<Attempt, HubError> {
    let ledger = ledger_path(part);
    let already = read_done(&ledger, total, slices.len());

    // Sized up front so every slice can seek to its own offset. Also the point
    // at which a full disk is discovered, rather than at 94%.
    let file = tokio::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(part)
        .await
        .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;
    file.set_len(total)
        .await
        .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;
    drop(file);

    // Two counters, because they answer different questions. `done` drives the
    // progress bar and must include what a previous run left on disk;
    // `transferred` is what this call actually fetched, which is what the
    // caller is told and the only honest answer to "was anything resumed".
    let done: AtomicU64 = AtomicU64::new(
        already.iter().filter_map(|i| slices.get(*i)).map(|(a, b)| b - a + 1).sum(),
    );
    let transferred = AtomicU64::new(0);
    if let Some(cb) = progress {
        cb(done.load(Ordering::Relaxed), total);
    }

    // Appended under a lock as each slice lands. Flushed every time: the point
    // of the file is to survive a process that did not get to exit.
    let ledger_lock = tokio::sync::Mutex::new(());
    let write_ledger = |index: usize| {
        let ledger = ledger.clone();
        let lock = &ledger_lock;
        let header = format!("{total} {}", slices.len());
        async move {
            use std::io::Write;
            let _guard = lock.lock().await;
            let fresh = !ledger.exists();
            if let Ok(mut f) =
                std::fs::OpenOptions::new().create(true).append(true).open(&ledger)
            {
                if fresh {
                    let _ = writeln!(f, "{header}");
                }
                let _ = writeln!(f, "{index}");
                let _ = f.flush();
            }
        }
    };

    let pending: Vec<(usize, (u64, u64))> = slices
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| !already.contains(i))
        .collect();

    let unsupported = std::sync::atomic::AtomicBool::new(false);
    let done_ref = &done;
    let transferred_ref = &transferred;
    let unsupported_ref = &unsupported;

    let fetches = pending.into_iter().map(|(index, (start, end))| {
        let write_ledger = &write_ledger;
        async move {
            let mut request = http.get(url).header(
                reqwest::header::RANGE,
                format!("bytes={start}-{end}"),
            );
            if let Some(t) = token {
                request = request.bearer_auth(t);
            }
            let resp = request.send().await?;

            // Anything but 206 means the range was ignored, and writing a
            // whole-file body at an offset would corrupt everything.
            if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
                unsupported_ref.store(true, Ordering::Relaxed);
                return Ok(());
            }

            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .open(part)
                .await
                .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;
            file.seek(std::io::SeekFrom::Start(start))
                .await
                .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;

            let mut stream = resp.bytes_stream();
            let mut written = 0u64;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                // A server that sends more than asked for would run into the
                // next slice's bytes.
                let room = (end - start + 1).saturating_sub(written) as usize;
                let chunk = if chunk.len() > room { chunk.slice(..room) } else { chunk };
                if chunk.is_empty() {
                    break;
                }
                file.write_all(&chunk)
                    .await
                    .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;
                written += chunk.len() as u64;
                transferred_ref.fetch_add(chunk.len() as u64, Ordering::Relaxed);
                let so_far = done_ref.fetch_add(chunk.len() as u64, Ordering::Relaxed)
                    + chunk.len() as u64;
                if let Some(cb) = progress {
                    cb(so_far.min(total), total);
                }
            }
            file.flush()
                .await
                .map_err(|e| HubError::Io { path: part.display().to_string(), source: e })?;

            if written == end - start + 1 {
                write_ledger(index).await;
            }
            Ok::<(), HubError>(())
        }
    });

    let limit = slices.len().min(connections());
    let mut stream = futures_util::stream::iter(fetches).buffer_unordered(limit);
    while let Some(result) = stream.next().await {
        result?;
        if unsupported.load(Ordering::Relaxed) {
            return Ok(Attempt::NoRanges);
        }
    }
    if unsupported.load(Ordering::Relaxed) {
        return Ok(Attempt::NoRanges);
    }

    std::fs::remove_file(&ledger).ok();
    Ok(Attempt::Done(transferred.load(Ordering::Relaxed)))
}

fn ledger_path(part: &Path) -> PathBuf {
    let mut name = part.as_os_str().to_owned();
    name.push(".ranges");
    PathBuf::from(name)
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

/// How many connections to use, honouring the environment.
pub fn connections() -> usize {
    std::env::var("OZGENT_DOWNLOAD_CONNECTIONS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|n| n.clamp(1, 32))
        .unwrap_or(DEFAULT_CONNECTIONS)
}

/// Split `total` bytes into inclusive byte ranges, one per connection.
///
/// Returns an empty plan when the file is too small to be worth splitting, so
/// the caller has one condition to test rather than two.
///
/// Public because the layout is what a resumed download has to agree with: the
/// ledger beside a part file records the plan it belongs to, and anything
/// reasoning about a half-finished download needs the same answer.
pub fn plan(total: u64, want: usize) -> Vec<(u64, u64)> {
    if total < PARALLEL_MIN_BYTES || want <= 1 {
        return Vec::new();
    }
    // Sliced by size, not by connection count. `want` decides how many are in
    // flight at once, which is a different question and is applied by the
    // caller.
    let count = total.div_ceil(SLICE_BYTES).clamp(2, MAX_SLICES as u64) as usize;
    let chunk = total / count as u64;
    (0..count)
        .map(|i| {
            let start = chunk * i as u64;
            // The last slice takes the remainder, so the ranges always cover
            // the file exactly however the division fell.
            let end = if i + 1 == count { total - 1 } else { start + chunk - 1 };
            (start, end)
        })
        .collect()
}

/// Which slices of a plan are already on disk, from the sidecar file.
///
/// The sidecar records the plan it belongs to. A file downloaded with four
/// connections and resumed with eight has a different layout, and replaying
/// the old indices against the new plan would leave holes — so a header that
/// does not match is treated as no progress at all.
fn read_done(path: &Path, total: u64, count: usize) -> std::collections::BTreeSet<usize> {
    let empty = std::collections::BTreeSet::new();
    let Ok(text) = std::fs::read_to_string(path) else { return empty };
    let mut lines = text.lines();
    match lines.next() {
        Some(header) if header == format!("{total} {count}") => {}
        _ => return empty,
    }
    lines.filter_map(|l| l.trim().parse::<usize>().ok()).collect()
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

    // ----------------------------------------------------- the split plan

    #[test]
    fn the_slices_cover_the_file_exactly() {
        // The property everything else depends on: no gap and no overlap. A
        // gap is a hole in the file that only a checksum would notice; an
        // overlap is two connections writing the same bytes.
        for total in [40 * MB, 100 * MB, 2_600_000_000, 7_300_000_001] {
            let slices = plan(total, 8);
            assert!(!slices.is_empty(), "{total} should split");
            assert_eq!(slices[0].0, 0, "starts at the beginning");
            assert_eq!(slices.last().unwrap().1, total - 1, "ends at the last byte");
            for pair in slices.windows(2) {
                assert_eq!(pair[1].0, pair[0].1 + 1, "contiguous: {slices:?}");
            }
            let covered: u64 = slices.iter().map(|(a, b)| b - a + 1).sum();
            assert_eq!(covered, total, "{slices:?}");
        }
    }

    const MB: u64 = 1024 * 1024;

    #[test]
    fn a_small_file_is_not_split() {
        // The round trips cost more than the parallelism saves.
        assert!(plan(1024, 8).is_empty());
        assert!(plan(31 * MB, 8).is_empty());
        assert!(!plan(32 * MB, 8).is_empty());
    }

    #[test]
    fn one_connection_means_no_plan() {
        assert!(plan(500 * MB, 1).is_empty());
        assert!(plan(500 * MB, 0).is_empty());
    }

    #[test]
    fn a_slice_is_small_enough_to_be_worth_re_fetching() {
        // Slices are the unit of resume. One per connection would mean an
        // interrupted 2.7 GB download discarding up to 340 MB of finished
        // work, which is the difference between resuming and restarting.
        for (total, label) in [(2_740_937_888u64, "2.7 GB"), (7_300_000_000, "7.3 GB")] {
            let slices = plan(total, 8);
            let largest = slices.iter().map(|(a, b)| b - a + 1).max().unwrap();
            assert!(
                largest <= SLICE_BYTES * 2,
                "{label}: largest slice {} MB",
                largest / MB
            );
        }
    }

    #[test]
    fn a_huge_file_does_not_become_thousands_of_requests() {
        let slices = plan(400 * 1024 * MB, 8);
        assert!(slices.len() <= MAX_SLICES, "{} slices", slices.len());
    }

    #[test]
    fn the_slice_count_does_not_depend_on_the_connection_count() {
        // They answer different questions: how the file is cut, and how many
        // are fetched at once. Tying them together is what made resume coarse.
        assert_eq!(plan(500 * MB, 4).len(), plan(500 * MB, 16).len());
    }

    #[test]
    fn an_unknown_size_cannot_be_planned() {
        // Without a length there is nothing to divide, so the sequential path
        // has to handle it.
        assert!(plan(0, 8).is_empty());
    }

    // -------------------------------------------------------- the ledger

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ozgent-dl-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("f")
    }

    #[test]
    fn the_ledger_sits_beside_the_part_file() {
        assert_eq!(ledger_path(Path::new("/m/x.gguf.part")), Path::new("/m/x.gguf.part.ranges"));
    }

    #[test]
    fn completed_slices_are_remembered() {
        let path = scratch("ledger");
        std::fs::write(&path, "1000 4
0
2
").unwrap();
        let done = read_done(&path, 1000, 4);
        assert_eq!(done.iter().copied().collect::<Vec<_>>(), [0, 2]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn a_ledger_for_a_different_plan_is_ignored() {
        // Resuming a four-connection download with eight would replay indices
        // against a layout they do not describe, leaving holes.
        let path = scratch("mismatch");
        std::fs::write(&path, "1000 4
0
1
").unwrap();
        assert!(read_done(&path, 1000, 8).is_empty(), "different connection count");
        assert!(read_done(&path, 2000, 4).is_empty(), "different file size");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn a_missing_or_junk_ledger_means_no_progress() {
        assert!(read_done(Path::new("/nowhere/at/all.ranges"), 10, 2).is_empty());
        let path = scratch("junk");
        std::fs::write(&path, "not a header
0
").unwrap();
        assert!(read_done(&path, 10, 2).is_empty());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn the_connection_count_is_bounded() {
        // Read from the environment, so it has to survive whatever is in it.
        unsafe { std::env::set_var("OZGENT_DOWNLOAD_CONNECTIONS", "999") };
        assert_eq!(connections(), 32);
        unsafe { std::env::set_var("OZGENT_DOWNLOAD_CONNECTIONS", "0") };
        assert_eq!(connections(), 1);
        unsafe { std::env::set_var("OZGENT_DOWNLOAD_CONNECTIONS", "nonsense") };
        assert_eq!(connections(), DEFAULT_CONNECTIONS);
        unsafe { std::env::remove_var("OZGENT_DOWNLOAD_CONNECTIONS") };
        assert_eq!(connections(), DEFAULT_CONNECTIONS);
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
