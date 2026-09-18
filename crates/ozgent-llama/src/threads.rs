//! How many CPU threads a decode should use, found by measuring.
//!
//! There is no right constant, and it cannot be found by comparing runs.
//! Measured on one laptop (8 cores, 16 threads), the same setting loaded
//! twice on a 35B mixture-of-experts model ran at 38.9 and 48.4 ms a token:
//! power and thermal state and what the page cache holds move a whole run by
//! more than threads ever do. Compared inside one run, pass by pass, the
//! differences are real and model-specific: a 27B ternary model with nine
//! layers on the CPU was 9% faster on 6 or 8 threads than on llama.cpp's
//! default of 4, while the 35B was the same on 4, 6 and 8. Every hardware
//! thread — 16 — is never a candidate: two threads on one core share its
//! memory bandwidth and fight for it.
//!
//! The tuner takes turns: each decode pass runs on the next candidate count,
//! round robin, so a context growing through the measurement slows every
//! candidate alike instead of whichever came last. After enough rounds the
//! fastest is kept. It measures again every so often, because the machine
//! that was idle when the model loaded may not be now.
//!
//! Candidates never exceed the physical cores other programs are not already
//! using, read from `/proc/stat`: a browser compiling something should not
//! have ozgent's threads land on top of it.

use std::time::Duration;

/// Decode passes timed per candidate before choosing.
const SAMPLES: u32 = 16;

/// Decode passes run on the default before anything is timed. Right after a
/// load the weights are still arriving from disk into the page cache, and
/// passes then are slow for reasons that have nothing to do with threads:
/// measured on a cold 35B, 58 ms a token against 38 once warm.
const WARMUP: u32 = 64;

/// Passes skipped after a switch before timing starts, so the first pass on
/// a new count — which pays for waking its threads — is not held against it.
const SETTLE: u32 = 1;

/// Decode passes between measurements once settled.
const RETUNE_EVERY: u64 = 4096;

/// A candidate must beat the current count by this share to replace it.
/// Below that the difference is noise, and switching for noise just moves the
/// threads around.
const MARGIN: f64 = 0.03;

#[derive(Debug, Clone)]
enum Phase {
    /// Not yet timing; `left` passes to go.
    Warming { left: u32 },
    /// Taking turns. `turn` counts passes since the round robin began.
    Measuring { turn: u32 },
    /// Using `best` until `left` more passes have run.
    Settled { left: u64 },
}

/// Chooses the decode thread count from timed passes.
#[derive(Debug, Clone)]
pub struct Tuner {
    candidates: Vec<u32>,
    /// Summed pass time and passes timed, per candidate.
    totals: Vec<(f64, u32)>,
    best: usize,
    phase: Phase,
}

impl Tuner {
    /// A tuner over thread counts up to `available` physical cores.
    pub fn new(available: u32) -> Self {
        let candidates = candidates(available);
        let best = candidates.len() - 1;
        let mut tuner = Tuner {
            totals: vec![(0.0, 0); candidates.len()],
            candidates,
            best,
            phase: Phase::Warming { left: WARMUP },
        };
        if tuner.candidates.len() == 1 {
            tuner.phase = Phase::Settled { left: RETUNE_EVERY };
        }
        tuner
    }

    /// The count the next decode pass should run on.
    pub fn current(&self) -> u32 {
        match self.phase {
            Phase::Measuring { turn } => {
                self.candidates[(turn / (SETTLE + 1)) as usize % self.candidates.len()]
            }
            Phase::Settled { .. } | Phase::Warming { .. } => self.candidates[self.best],
        }
    }

    /// Whether the tuner has yet to settle on a count.
    pub fn measuring(&self) -> bool {
        matches!(self.phase, Phase::Measuring { .. } | Phase::Warming { .. })
    }

    /// Record how long a decode pass took on [`Tuner::current`].
    pub fn record(&mut self, took: Duration) {
        match &mut self.phase {
            Phase::Warming { left } => {
                *left = left.saturating_sub(1);
                if *left == 0 {
                    self.phase = Phase::Measuring { turn: 0 };
                }
            }
            Phase::Measuring { turn } => {
                let at = (*turn / (SETTLE + 1)) as usize % self.candidates.len();
                // Each candidate gets SETTLE warm-up passes, then one timed.
                if *turn % (SETTLE + 1) == SETTLE {
                    let slot = &mut self.totals[at];
                    slot.0 += took.as_secs_f64();
                    slot.1 += 1;
                }
                *turn += 1;
                if self.totals.iter().all(|(_, n)| *n >= SAMPLES) {
                    self.choose();
                }
            }
            Phase::Settled { left } => {
                *left = left.saturating_sub(1);
                if *left == 0 {
                    self.restart(None);
                }
            }
        }
    }

    /// Measure again, over the physical cores now `available` when given.
    pub fn restart(&mut self, available: Option<u32>) {
        let keep = self.candidates[self.best];
        if let Some(a) = available {
            self.candidates = candidates(a);
        }
        self.totals = vec![(0.0, 0); self.candidates.len()];
        // Whatever the last choice was stays the choice unless beaten.
        self.best = self
            .candidates
            .iter()
            .position(|c| *c == keep)
            .unwrap_or(self.candidates.len() - 1);
        self.phase = if self.candidates.len() > 1 {
            Phase::Measuring { turn: 0 }
        } else {
            Phase::Settled { left: RETUNE_EVERY }
        };
    }

    fn choose(&mut self) {
        let mean = |i: usize| self.totals[i].0 / self.totals[i].1.max(1) as f64;
        let fastest = (0..self.candidates.len())
            .min_by(|a, b| mean(*a).total_cmp(&mean(*b)))
            .unwrap_or(self.best);
        // The incumbent keeps its place unless clearly beaten; among ties the
        // smaller count wins, since it leaves cores for everything else.
        if mean(fastest) < mean(self.best) * (1.0 - MARGIN) {
            self.best = fastest;
        }
        let quickest = mean(self.best);
        if let Some(fewer) = (0..self.best)
            .find(|i| mean(*i) <= quickest * (1.0 + MARGIN / 2.0))
        {
            self.best = fewer;
        }
        tracing::info!(
            "decode threads: {} ({})",
            self.candidates[self.best],
            (0..self.candidates.len())
                .map(|i| format!("{} → {:.1} ms", self.candidates[i], mean(i) * 1000.0))
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.phase = Phase::Settled { left: RETUNE_EVERY };
    }
}

/// The counts worth trying on `available` physical cores: half, three
/// quarters and all of them. Fewer than half was slower on every model
/// measured, and more than the physical cores slower still.
fn candidates(available: u32) -> Vec<u32> {
    let a = available.max(1);
    let mut out = vec![(a / 2).max(1), (a * 3 / 4).max(1), a];
    out.dedup();
    out
}

/// Physical cores on this machine: hardware threads sharing a core counted
/// once. Falls back to half the logical count, which is right for every
/// desktop and laptop with SMT, and then to the logical count.
pub fn physical_cores() -> u32 {
    let mut cores = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let topology = entry.path().join("topology");
            let read = |f: &str| std::fs::read_to_string(topology.join(f)).ok().map(|s| s.trim().to_string());
            if let (Some(package), Some(core)) = (read("physical_package_id"), read("core_id")) {
                cores.insert((package, core));
            }
        }
    }
    if !cores.is_empty() {
        return cores.len() as u32;
    }
    let logical = std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4);
    (logical / 2).max(1)
}

/// CPU time used on the whole machine and by this process, in clock ticks,
/// at one moment. Two of these give what everything else was doing between.
#[derive(Debug, Clone, Copy)]
pub struct Usage {
    total: u64,
    busy: u64,
    mine: u64,
}

impl Usage {
    pub fn now() -> Option<Usage> {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        let line = stat.lines().next()?;
        let fields: Vec<u64> = line.split_whitespace().skip(1).filter_map(|v| v.parse().ok()).collect();
        if fields.len() < 5 {
            return None;
        }
        // user nice system idle iowait irq softirq steal …
        let idle = fields[3] + fields.get(4).copied().unwrap_or(0);
        let total: u64 = fields.iter().take(8).sum();
        let own = std::fs::read_to_string("/proc/self/stat").ok()?;
        // Fields after the command name, which is in parentheses and may
        // contain spaces: utime and stime are the 12th and 13th after it.
        let after = own.rsplit_once(')')?.1;
        let parts: Vec<&str> = after.split_whitespace().collect();
        let mine = parts.get(11)?.parse::<u64>().ok()? + parts.get(12)?.parse::<u64>().ok()?;
        Some(Usage { total, busy: total - idle, mine })
    }

    /// Cores other processes kept busy between `earlier` and this sample.
    pub fn others_since(&self, earlier: &Usage, logical: u32) -> f64 {
        let span = self.total.saturating_sub(earlier.total);
        if span == 0 {
            return 0.0;
        }
        let busy = self.busy.saturating_sub(earlier.busy);
        let mine = self.mine.saturating_sub(earlier.mine);
        busy.saturating_sub(mine) as f64 / span as f64 * logical as f64
    }
}

/// Physical cores free for ozgent, given that other programs kept `others`
/// logical threads busy.
///
/// Another program's thread may sit on a core's second hardware thread and
/// leave most of the core free, so busy logical threads count as half a core
/// each, and only whole cores are subtracted.
pub fn available(physical: u32, others: f64) -> u32 {
    let taken = (others / 2.0).floor() as u32;
    physical.saturating_sub(taken).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: f64) -> Duration {
        Duration::from_secs_f64(n / 1000.0)
    }

    /// Feed passes whose time depends on the thread count until settled.
    fn run(tuner: &mut Tuner, cost: impl Fn(u32) -> f64) {
        for _ in 0..10_000 {
            if !tuner.measuring() {
                return;
            }
            let t = tuner.current();
            tuner.record(ms(cost(t)));
        }
        panic!("never settled");
    }

    #[test]
    fn candidates_are_half_three_quarters_and_all() {
        assert_eq!(candidates(8), vec![4, 6, 8]);
        assert_eq!(candidates(2), vec![1, 2]);
        assert_eq!(candidates(1), vec![1]);
    }

    #[test]
    fn the_fastest_count_is_kept() {
        // A MoE with experts on the CPU: more threads are genuinely faster.
        let mut t = Tuner::new(8);
        run(&mut t, |n| 40.0 + 40.0 / n as f64);
        assert_eq!(t.current(), 8);
    }

    #[test]
    fn a_tie_goes_to_fewer_threads() {
        // Memory-bound: 4, 6 and 8 threads within noise of one another.
        let mut t = Tuner::new(8);
        run(&mut t, |n| 88.0 + (n as f64 * 0.01));
        assert_eq!(t.current(), 4, "no reason to hold cores the model cannot use");
    }

    #[test]
    fn it_measures_again_after_a_while() {
        let mut t = Tuner::new(8);
        run(&mut t, |_| 10.0);
        for _ in 0..RETUNE_EVERY {
            t.record(ms(10.0));
        }
        assert!(t.measuring());
    }

    #[test]
    fn a_busy_machine_offers_fewer_cores() {
        assert_eq!(available(8, 0.3), 8);
        assert_eq!(available(8, 4.0), 6);
        assert_eq!(available(8, 40.0), 1);
    }

    #[test]
    fn this_machine_reports_its_cores() {
        assert!(physical_cores() >= 1);
        assert!(Usage::now().is_some());
    }
}
