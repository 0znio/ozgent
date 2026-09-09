//! The parallel downloader, against a server that really serves ranges.
//!
//! `tests/ranges.py` serves a file whose every byte is derived from its offset,
//! so a slice written to the wrong place shows up as wrong data rather than as
//! plausible zeroes. That is the failure worth catching: an assembled file of
//! exactly the right length and the wrong contents.
//!
//! Skipped when python3 is missing.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

const SIZE: u64 = 40 * 1024 * 1024; // over the threshold that triggers a split

fn expected(size: u64) -> Vec<u8> {
    (0..size).map(|i| ((i * 7 + 11) & 0xFF) as u8).collect()
}

struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn serve(extra: &[&str]) -> Option<Server> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/ranges.py");
    let mut cmd = Command::new("python3");
    cmd.arg(&script).arg("0").arg(SIZE.to_string());
    for e in extra {
        cmd.arg(e);
    }
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;

    let stdout = child.stdout.take()?;
    let mut line = String::new();
    BufReader::new(stdout).read_line(&mut line).ok()?;
    let port: u16 = line.trim().parse().ok()?;
    Some(Server { child, port })
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ozgent-par-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("model.gguf")
}

async fn fetch(port: u16, dest: &std::path::Path) -> Result<u64, ozgent_hub::HubError> {
    let http = reqwest::Client::new();
    ozgent_hub::download::download(
        &http,
        &format!("http://127.0.0.1:{port}/model.gguf"),
        dest,
        SIZE,
        None,
        None,
        None,
    )
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn a_file_fetched_in_parallel_is_byte_for_byte_correct() {
    let Some(server) = serve(&[]) else {
        eprintln!("skipping: python3 is not installed");
        return;
    };
    let dest = scratch("ok");
    let moved = fetch(server.port, &dest).await.expect("download");

    assert_eq!(moved, SIZE, "every byte accounted for");
    let got = std::fs::read(&dest).expect("the finished file");
    assert_eq!(got.len() as u64, SIZE, "length");
    // Compared whole: a slice landing at the wrong offset produces the right
    // length and the wrong bytes, which is the bug this exists to catch.
    assert!(got == expected(SIZE), "contents differ from what the server served");

    let _ = std::fs::remove_dir_all(dest.parent().unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_that_refuses_ranges_still_works() {
    // It answers every ranged request with the whole file. Writing that at an
    // offset would corrupt everything, so the downloader has to notice and
    // start over on one connection.
    let Some(server) = serve(&["--no-ranges"]) else {
        eprintln!("skipping: python3 is not installed");
        return;
    };
    let dest = scratch("noranges");
    fetch(server.port, &dest).await.expect("download");

    let got = std::fs::read(&dest).expect("the finished file");
    assert_eq!(got.len() as u64, SIZE);
    assert!(got == expected(SIZE), "contents differ");

    let _ = std::fs::remove_dir_all(dest.parent().unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn an_interrupted_parallel_download_resumes_rather_than_restarting() {
    let Some(server) = serve(&[]) else {
        eprintln!("skipping: python3 is not installed");
        return;
    };
    let dest = scratch("resume");
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let ledger = PathBuf::from(format!("{}.ranges", part.display()));

    // Stand in for a download killed after some slices landed: a full-length
    // part file with a ledger naming the slices that finished.
    //
    // The plan is asked for rather than assumed. It is not simply the
    // connection count — a 40 MB file will not be cut into eight five-megabyte
    // pieces — and a ledger describing a different layout is correctly ignored,
    // which is what a guess here would silently test instead.
    std::fs::write(&part, vec![0u8; SIZE as usize]).unwrap();
    let plan = ozgent_hub::download::plan(SIZE, ozgent_hub::download::connections());
    let slices = plan.len();
    assert!(slices > 1, "the fixture must actually split: {slices}");
    let chunk = plan[0].1 - plan[0].0 + 1;
    let truth = expected(SIZE);
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().write(true).open(&part).unwrap();
        use std::io::{Seek, SeekFrom};
        // Slice 0 is genuinely on disk; the ledger says so.
        f.seek(SeekFrom::Start(0)).unwrap();
        f.write_all(&truth[..chunk as usize]).unwrap();
    }
    std::fs::write(&ledger, format!("{SIZE} {slices}\n0\n")).unwrap();

    let moved = fetch(server.port, &dest).await.expect("download");
    assert!(moved < SIZE, "the finished slice should not be fetched again: {moved} of {SIZE}");

    let got = std::fs::read(&dest).expect("the finished file");
    assert!(got == truth, "resumed file differs");
    assert!(!ledger.exists(), "the ledger is cleared when the file completes");

    let _ = std::fs::remove_dir_all(dest.parent().unwrap());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_part_file_from_a_parallel_run_is_never_appended_to() {
    // A parallel part file is full-length with holes, not a prefix. Treating
    // it as a prefix produces a file of the right size and the wrong contents,
    // which nothing but a checksum would catch.
    let Some(server) = serve(&["--no-ranges"]) else {
        eprintln!("skipping: python3 is not installed");
        return;
    };
    let dest = scratch("holes");
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let ledger = PathBuf::from(format!("{}.ranges", part.display()));

    std::fs::write(&part, vec![0u8; SIZE as usize]).unwrap();
    std::fs::write(&ledger, "999999 4\n0\n").unwrap(); // a plan for another file

    fetch(server.port, &dest).await.expect("download");
    let got = std::fs::read(&dest).expect("the finished file");
    assert_eq!(got.len() as u64, SIZE, "not appended to the holed part file");
    assert!(got == expected(SIZE), "contents differ");

    let _ = std::fs::remove_dir_all(dest.parent().unwrap());
}
