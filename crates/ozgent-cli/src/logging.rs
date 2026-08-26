//! Where ozgent's diagnostics go.
//!
//! Two destinations, because they answer different questions. The terminal
//! shows what someone watching wants to see right now, quiet by default and
//! opened up with `-v`. The file records what happened, at a fixed level,
//! whether or not anyone was watching — which is the only thing that helps
//! when the server has been running under systemd for a week and something
//! went wrong on Tuesday.
//!
//! The file is the same for every entry point. Chat, the TUI, `serve` and the
//! one-shot commands all write to it, so an investigation starts in one place
//! rather than depending on which way ozgent happened to be started.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Roll over once the active log passes this size.
const MAX_BYTES: u64 = 8 * 1024 * 1024;

/// Rolled files kept besides the active one.
const KEEP: usize = 3;

/// The environment variable that overrides both levels.
pub const FILTER_ENV: &str = "OZGENT_LOG";

/// Level written to the file when nothing overrides it.
///
/// Info rather than warn: the file exists to explain a past event, and the
/// lines that explain one — which model loaded, how the cache was placed,
/// which tools registered — are all info. Warnings alone say something went
/// wrong without saying what led there.
const FILE_LEVEL: &str = "ozgent=info";

/// A log file that rolls over instead of growing without bound.
///
/// Rotation is by size and happens on the write that crosses the limit. A
/// daemon left alone for months should cost a bounded amount of disk, and
/// nobody should have to remember to configure logrotate for it.
#[derive(Clone)]
pub struct RollingLog {
    inner: Arc<Mutex<Rolling>>,
}

struct Rolling {
    path: PathBuf,
    file: Option<File>,
    written: u64,
}

impl RollingLog {
    /// Open `dir/ozgent.log`, appending to whatever is already there.
    ///
    /// Returns `None` when the file cannot be opened. A machine with a
    /// read-only or full disk should still be able to run ozgent; losing the
    /// log is a degradation, not a reason to refuse to start.
    pub fn open(dir: &Path) -> Option<Self> {
        std::fs::create_dir_all(dir).ok()?;
        let path = dir.join("ozgent.log");
        let file = OpenOptions::new().create(true).append(true).open(&path).ok()?;
        let written = file.metadata().map(|m| m.len()).unwrap_or(0);
        Some(Self { inner: Arc::new(Mutex::new(Rolling { path, file: Some(file), written })) })
    }
}

impl Rolling {
    fn roll(&mut self) {
        // Drop the handle first: the rename is what makes the old bytes
        // unreachable under the active name, and writing through a stale
        // handle afterwards would append to a file nothing will look at.
        self.file = None;
        for n in (1..KEEP).rev() {
            let _ = std::fs::rename(suffixed(&self.path, n), suffixed(&self.path, n + 1));
        }
        let _ = std::fs::rename(&self.path, suffixed(&self.path, 1));
        self.file = OpenOptions::new().create(true).append(true).open(&self.path).ok();
        self.written = 0;
    }
}

fn suffixed(path: &Path, n: usize) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{n}"));
    PathBuf::from(name)
}

impl Write for Rolling {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.written + buf.len() as u64 > MAX_BYTES {
            self.roll();
        }
        match self.file.as_mut() {
            Some(f) => {
                let n = f.write(buf)?;
                self.written += n as u64;
                Ok(n)
            }
            // Nowhere to put it. Report success rather than failing the write:
            // a logging error must not become the program's error.
            None => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.file.as_mut() {
            Some(f) => f.flush(),
            None => Ok(()),
        }
    }
}

/// Handle handed to the tracing layer for one write.
pub struct Handle(Arc<Mutex<Rolling>>);

impl Write for Handle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self.0.lock() {
            Ok(mut g) => g.write(buf),
            Err(_) => Ok(buf.len()),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self.0.lock() {
            Ok(mut g) => g.flush(),
            Err(_) => Ok(()),
        }
    }
}

impl<'a> MakeWriter<'a> for RollingLog {
    type Writer = Handle;
    fn make_writer(&'a self) -> Self::Writer {
        Handle(Arc::clone(&self.inner))
    }
}

/// Install the subscriber: terminal at the requested verbosity, file at info.
///
/// `logs_dir` is where the file lives; passing `None` keeps the terminal-only
/// behaviour, which is what the tests and `--help` paths want.
pub fn init(verbose: u8, logs_dir: Option<&Path>) {
    let level = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let requested = std::env::var(FILTER_ENV).ok();
    let terminal_filter = requested
        .clone()
        .unwrap_or_else(|| format!("ozgent={level}"));

    let terminal = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .without_time()
        .with_filter(EnvFilter::new(terminal_filter));

    // An explicit OZGENT_LOG governs both, so turning on debug for an
    // investigation puts the detail in the file too rather than only on a
    // terminal that may not be attached.
    let file_layer = logs_dir.and_then(RollingLog::open).map(|writer| {
        tracing_subscriber::fmt::layer()
            .with_writer(writer)
            .with_ansi(false)
            .with_target(true)
            .with_filter(EnvFilter::new(
                requested.unwrap_or_else(|| FILE_LEVEL.to_string()),
            ))
    });

    tracing_subscriber::registry().with(terminal).with(file_layer).init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_keeps_the_log_bounded() {
        let dir = std::env::temp_dir().join(format!("ozgent-log-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log = RollingLog::open(&dir).expect("open");

        // Enough to cross the limit several times over.
        let chunk = vec![b'x'; 512 * 1024];
        for _ in 0..40 {
            let mut h = log.make_writer();
            h.write_all(&chunk).expect("write");
        }

        let active = dir.join("ozgent.log");
        assert!(active.exists(), "the active log must survive rotation");
        assert!(
            active.metadata().expect("meta").len() <= MAX_BYTES,
            "the active log must stay under the limit"
        );
        // Never more than the active file plus the ones we promised to keep.
        let files = std::fs::read_dir(&dir).expect("read").count();
        assert!(files <= KEEP + 1, "rotation kept {files} files");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_log_directory_that_cannot_be_opened_is_not_fatal() {
        // /proc is real and not writable, which is the shape of the problem:
        // a read-only or full disk must degrade to no file, not to no ozgent.
        assert!(RollingLog::open(Path::new("/proc/ozgent-nope")).is_none());
    }

    #[test]
    fn rolled_names_are_ordered() {
        let p = Path::new("/tmp/ozgent.log");
        assert_eq!(suffixed(p, 1), PathBuf::from("/tmp/ozgent.log.1"));
        assert_eq!(suffixed(p, 3), PathBuf::from("/tmp/ozgent.log.3"));
    }
}
