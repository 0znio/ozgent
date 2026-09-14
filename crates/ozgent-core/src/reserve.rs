//! How much VRAM to keep back from the weights and the cache.
//!
//! Something has to be reserved: llama.cpp allocates compute buffers and
//! scratch that no amount of reading a GGUF will predict, and a fit that is
//! too tight fails at allocation rather than degrading. The question is what
//! shape that reservation has.
//!
//! It used to be a percentage — thirty percent of whatever was free. That is
//! the wrong shape twice over. It is wrong *downwards* on a small card, where
//! thirty percent of five gigabytes is a gigabyte and a half of an eight
//! gigabyte card left permanently idle; and it is wrong *upwards* on a large
//! one, where the same rule throws away seven gigabytes of a 24 GB card for a
//! cost that did not grow at all. Scratch scales with how wide the model is
//! and how many tokens are in flight. It does not scale with how much memory
//! somebody happens to own.
//!
//! So it is two terms, learned from what actually happens:
//!
//! * **A per-process cost**, paid once when the CUDA context comes up and
//!   shared by every model loaded after it. Charging this to each model is
//!   how a pool of three ends up reserving three context's worth of memory
//!   that only exists once.
//! * **A per-model cost**, which does scale — with the micro-batch and the
//!   model's width, the two things that decide how big an activation is.
//!
//! Both start at a deliberately generous prior and are corrected by
//! measurement: every load reports what it actually cost, and the estimate
//! converges on this machine rather than on an average of machines.

use serde::{Deserialize, Serialize};

/// What a fresh install assumes before it has seen a single load.
///
/// Chosen to be roughly right for CUDA on a consumer card and to err large:
/// the cost of over-reserving is a slightly smaller window, and the cost of
/// under-reserving is a model that does not load.
const PRIOR_PROCESS_BYTES: u64 = 400 * 1024 * 1024;

/// Bytes of scratch per (context token × batch token), the prior.
///
/// Measured, not guessed. llama.cpp reports its own compute buffers, and on a
/// 4B at a 82,432-token window with a 2048 batch it printed **3157 MiB** —
/// which works out at just under 20 bytes per (context x batch) token pair.
///
/// The first version of this model had no context term at all and a prior of
/// 24 bytes per (micro-batch x embedding) element, about 400 MB. It was wrong
/// by a factor of eight at long windows, which is exactly how a 262,144-token
/// ask came back as 65,536: the planner said the cache fitted, llama.cpp
/// disagreed, and the retreat threw away three quarters of the context. The
/// term that dominates is the one that scales with the window.
const PRIOR_RATE: f64 = 20.0;

/// How much to inflate the estimate while it is still young.
///
/// A prediction from two observations should not be trusted like one from
/// fifty. The margin starts at a third and decays towards nothing, so a new
/// install is safe and a used one is accurate.
fn margin(samples: u32) -> f64 {
    1.0 + 0.33 / (1.0 + samples as f64 / 4.0)
}

/// The learned model of what llama.cpp costs beyond its weights and cache.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Reserve {
    /// Bytes the backend costs once per process, independent of any model.
    pub process_bytes: u64,
    /// Bytes per (n_ctx × n_batch), the part that scales — and the part that
    /// dominates everything else at any window worth having.
    pub rate: f64,
    /// Observations behind those two figures.
    pub samples: u32,
}

impl Default for Reserve {
    fn default() -> Self {
        Self { process_bytes: PRIOR_PROCESS_BYTES, rate: PRIOR_RATE, samples: 0 }
    }
}

/// What a load is about to ask of the device, for predicting and for learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Whether this is the first model in this process, and so the one that
    /// pays for bringing the backend up.
    pub first_in_process: bool,
    /// The physical micro-batch, in tokens.
    pub ubatch: u32,
    /// The model's embedding width.
    pub n_embd: u32,
    /// The window being opened. Some of llama.cpp's scratch scales with it
    /// rather than with the model — see [`mask_bytes`].
    pub n_ctx: u32,
    /// The *logical* batch. The mask is sized against this and not against
    /// the micro-batch, which is four times smaller by default and was why a
    /// 262,144-token window still failed to allocate after being accounted
    /// for: 2.1 GB of mask predicted as 536 MB.
    pub n_batch: u32,
}

impl Reserve {
    /// The quantity the scaling term is measured against.
    fn work(shape: Shape) -> f64 {
        shape.n_ctx.max(1) as f64 * shape.n_batch.max(1) as f64
    }

    /// Bytes to keep back for a load of this shape.
    pub fn predict(&self, shape: Shape) -> u64 {
        let scaling = self.rate * Self::work(shape);
        let once = if shape.first_in_process { self.process_bytes as f64 } else { 0.0 };
        ((once + scaling) * margin(self.samples)) as u64
    }

    /// Fold in what a load actually cost.
    ///
    /// `overhead` is the memory that disappeared beyond the weights and the
    /// cache — measured as free-before minus free-after minus what was asked
    /// for. A negative figure means the prediction was already generous and
    /// carries no information, so it is dropped rather than averaged in.
    pub fn observe(&mut self, shape: Shape, overhead: u64) {
        let scaling = Self::work(shape);
        // Each observation moves the estimate part of the way rather than
        // replacing it: one unusual load — a game starting mid-download, a
        // driver reporting late — should not throw the figure away.
        let weight = 0.3;
        if shape.first_in_process {
            // This load paid for both, so the fixed part is whatever the
            // scaling part does not explain.
            let implied = overhead as f64 - self.rate * scaling;
            if implied > 0.0 {
                self.process_bytes =
                    (self.process_bytes as f64 * (1.0 - weight) + implied * weight) as u64;
            }
        } else {
            // Nothing fixed to pay, so all of it is the scaling part.
            let implied = overhead as f64 / scaling;
            if implied > 0.0 {
                self.rate = self.rate * (1.0 - weight) + implied * weight;
            }
        }
        self.samples = self.samples.saturating_add(1);
    }

    /// Read what was learned on this machine, or start from the prior.
    ///
    /// A missing or unreadable file is not an error: the prior is a working
    /// answer, and refusing to load a model because a cache file was corrupt
    /// would be a far worse failure than a slightly wrong reservation.
    pub fn load(paths: &crate::Paths) -> Self {
        std::fs::read_to_string(Self::path(paths))
            .ok()
            .and_then(|t| serde_json::from_str::<Self>(&t).ok())
            .filter(|r| r.rate.is_finite() && r.rate > 0.0)
            .unwrap_or_default()
    }

    /// Write it back. Failure is ignored for the same reason.
    pub fn save(&self, paths: &crate::Paths) {
        let path = Self::path(paths);
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }

    fn path(paths: &crate::Paths) -> std::path::PathBuf {
        paths.cache_dir().join("vram.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: Shape = Shape { first_in_process: true, ubatch: 512, n_embd: 2560, n_ctx: 32768, n_batch: 2048 };
    const LATER: Shape = Shape { first_in_process: false, ubatch: 512, n_embd: 2560, n_ctx: 32768, n_batch: 2048 };

    #[test]
    fn a_later_model_is_not_charged_for_the_backend_again() {
        // The bug a per-model figure would have: three models in the pool
        // reserving three CUDA contexts, when the process only ever paid for
        // one. That is gigabytes of memory refused for no reason.
        let r = Reserve::default();
        assert!(
            r.predict(SHAPE) > r.predict(LATER),
            "the first load pays for the backend and later ones do not"
        );
    }

    #[test]
    fn the_estimate_scales_with_the_work_not_with_the_card() {
        // The percentage it replaces got this exactly backwards: it grew with
        // how much memory somebody owned and not at all with what was asked.
        let r = Reserve::default();
        let small = Shape { n_ctx: 4096, ..LATER };
        let large = Shape { n_ctx: 32768, ..LATER };
        assert!(r.predict(large) > r.predict(small) * 4);
    }

    #[test]
    fn the_prior_is_near_what_llama_cpp_actually_reported() {
        // 3157 MiB of compute buffers at 82,432 ctx with a 2048 batch, read
        // off llama.cpp's own log on an 8 GB card. The first version of this
        // model was wrong by a factor of eight here, which is how a 262,144
        // ask ended up opening at 65,536.
        let r = Reserve { samples: 100, ..Default::default() };
        let measured = Shape {
            first_in_process: false,
            ubatch: 512,
            n_embd: 2560,
            n_ctx: 82_432,
            n_batch: 2048,
        };
        let predicted = r.predict(measured) as f64;
        let actual = 3157.0 * 1024.0 * 1024.0;
        assert!(
            (predicted / actual - 1.0).abs() < 0.25,
            "predicted {predicted:.0} against a measured {actual:.0}"
        );
    }

    #[test]
    fn a_bigger_batch_reserves_more() {
        let r = Reserve::default();
        let small = Shape { n_batch: 128, ..LATER };
        let big = Shape { n_batch: 2048, ..LATER };
        assert!(r.predict(big) > r.predict(small));
    }

    #[test]
    fn measurement_moves_the_estimate_towards_the_truth() {
        let mut r = Reserve::default();
        let before = r.predict(LATER);
        // Consistently cheaper than the prior expects.
        for _ in 0..20 {
            r.observe(LATER, before / 4);
        }
        let after = r.predict(LATER);
        assert!(after < before, "{after} should be below {before}");
        assert!(after > 0, "it must not collapse to nothing");
    }

    #[test]
    fn a_young_estimate_is_more_generous_than_an_old_one() {
        // Two observations should not be trusted like fifty.
        let young = Reserve { samples: 0, ..Default::default() };
        let old = Reserve { samples: 100, ..Default::default() };
        assert!(young.predict(LATER) > old.predict(LATER));
    }

    #[test]
    fn an_impossible_observation_is_ignored_rather_than_averaged_in() {
        // Free memory can move for reasons that have nothing to do with us —
        // another process starting, a driver reporting late. An overhead that
        // comes out at or below zero carries no information.
        let mut r = Reserve::default();
        let rate = r.rate;
        r.observe(LATER, 0);
        assert_eq!(r.rate, rate, "a zero observation must not drag the rate down");
    }

    #[test]
    fn a_corrupt_cache_file_is_not_a_failure() {
        let dir = std::env::temp_dir().join(format!("ozgent-reserve-{}", std::process::id()));
        let paths = crate::Paths::with_root(&dir);
        std::fs::create_dir_all(paths.cache_dir()).unwrap();
        std::fs::write(paths.cache_dir().join("vram.json"), "{ not json").unwrap();
        assert_eq!(Reserve::load(&paths), Reserve::default());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn what_was_learned_survives_a_restart() {
        let dir = std::env::temp_dir().join(format!("ozgent-reserve-rt-{}", std::process::id()));
        let paths = crate::Paths::with_root(&dir);
        let mut r = Reserve::default();
        for _ in 0..10 {
            r.observe(LATER, 50 * 1024 * 1024);
        }
        r.save(&paths);
        let back = Reserve::load(&paths);
        assert_eq!(back.samples, r.samples);
        assert_eq!(back.process_bytes, r.process_bytes);
        // JSON is decimal, so the rate returns near enough rather than equal.
        assert!((back.rate - r.rate).abs() < 1e-6, "{} vs {}", back.rate, r.rate);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
