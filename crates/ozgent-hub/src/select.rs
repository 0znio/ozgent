//! Choosing which files to download from a repository.
//!
//! A GGUF repo is a menu, not a package: `unsloth/gemma-3-12b-it-GGUF` holds
//! 29 quantisations of the same model plus three vision projectors. Picking
//! correctly is the difference between a 4 GB download that runs and a 23 GB
//! one that does not fit.
//!
//! All of this is pure so it can be tested against real repository listings
//! without touching the network.

/// One file in a repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
    /// LFS object id, which for GGUF files is the sha256 of the content.
    pub sha256: Option<String>,
}

impl RepoFile {
    pub fn is_gguf(&self) -> bool {
        self.path.to_ascii_lowercase().ends_with(".gguf")
    }

    /// Vision projectors are GGUF files but are never the main weights.
    pub fn is_mmproj(&self) -> bool {
        let name = file_name(&self.path).to_ascii_lowercase();
        name.starts_with("mmproj") || name.contains("mmproj")
    }
}

/// Quantisation names, longest first so `Q4_K_M` wins over `Q4_K` and
/// `Q4`. Order within a length does not matter.
const QUANTS: &[&str] = &[
    "Q2_K_XL", "Q3_K_XL", "Q4_K_XL", "Q5_K_XL", "Q6_K_XL", "Q8_K_XL",
    "IQ2_XXS", "IQ3_XXS", "IQ4_NL", "IQ4_XS", "IQ1_M", "IQ1_S", "IQ2_XS",
    "IQ2_M", "IQ2_S", "IQ3_XS", "IQ3_M", "IQ3_S",
    "Q2_K_L", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S",
    "Q5_K_M", "Q5_K_S", "TQ1_0", "TQ2_0",
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q6_K", "Q8_0", "Q2_K",
    "BF16", "F16", "F32",
];

/// The quantisation a filename encodes, e.g. `Q4_K_M`.
///
/// Repos are inconsistent — `model-Q4_K_M.gguf`, `Model.Q4_K_M.gguf`,
/// `model-UD-Q4_K_XL.gguf`, lowercase variants — so matching is on token
/// boundaries within the stem rather than a fixed position.
pub fn quant_of(path: &str) -> Option<String> {
    let stem = file_name(path).trim_end_matches(".gguf").trim_end_matches(".GGUF");
    let upper = stem.to_ascii_uppercase();

    let mut best: Option<(usize, &str)> = None;
    for q in QUANTS {
        if let Some(pos) = find_token(&upper, q) {
            // Prefer the longest match, then the latest, since the quant is
            // conventionally the trailing part of the name.
            let better = match best {
                None => true,
                Some((bp, bq)) => q.len() > bq.len() || (q.len() == bq.len() && pos > bp),
            };
            if better {
                best = Some((pos, q));
            }
        }
    }
    best.map(|(_, q)| q.to_string())
}

/// Find `needle` in `haystack` bounded by non-alphanumeric characters, so
/// `F16` does not match inside `BF16`.
fn find_token(haystack: &str, needle: &str) -> Option<usize> {
    let bytes = haystack.as_bytes();
    let mut from = 0;
    while let Some(rel) = haystack[from..].find(needle) {
        let start = from + rel;
        let end = start + needle.len();
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let after_ok = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return Some(start);
        }
        from = start + 1;
    }
    None
}

/// Byte range of the `-00002-of-00005` segment inside a stem.
///
/// The segment is not always at the end: `Model-Q4_K_M-00001-of-00009` is the
/// common layout, but tools also emit the quantisation after the shard count.
fn shard_bounds(stem: &str) -> Option<(usize, usize)> {
    let idx = stem.rfind("-of-")?;

    let after = &stem[idx + 4..];
    let total_len = after.chars().take_while(char::is_ascii_digit).count();
    if total_len == 0 {
        return None;
    }

    let head = &stem[..idx];
    let part_len = head.chars().rev().take_while(char::is_ascii_digit).count();
    if part_len == 0 {
        return None;
    }
    let part_start = head.len() - part_len;
    // Include the dash that introduces the segment, when there is one.
    let start = part_start.saturating_sub(usize::from(head[..part_start].ends_with('-')));

    Some((start, idx + 4 + total_len))
}

/// Shard position, for `name-00002-of-00005.gguf`.
fn shard_of(path: &str) -> Option<(u32, u32)> {
    let stem = file_name(path).trim_end_matches(".gguf");
    let (start, end) = shard_bounds(stem)?;
    let segment = stem[start..end].trim_start_matches('-');
    let (part, total) = segment.split_once("-of-")?;
    Some((part.parse().ok()?, total.parse().ok()?))
}

/// Remove the shard segment so every shard of one model groups together.
fn shard_group(path: &str) -> String {
    let stem = file_name(path).trim_end_matches(".gguf");
    match shard_bounds(stem) {
        Some((start, end)) => format!("{}{}", &stem[..start], &stem[end..]),
        None => stem.to_string(),
    }
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// What to download.
#[derive(Debug, Clone, PartialEq)]
pub struct Selection {
    /// Weight shards, in order. A single-file model has one entry.
    pub weights: Vec<RepoFile>,
    pub mmproj: Option<RepoFile>,
    pub quant: String,
    pub total_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum SelectError {
    #[error("no .gguf files found in this repository")]
    NoGguf,
    #[error("quantisation {requested:?} not found. Available: {}", available.join(", "))]
    QuantNotFound { requested: String, available: Vec<String> },
}

/// Preference order when nothing is requested and there is no size budget.
///
/// `Q4_K_M` first: it is the long-standing quality-per-byte sweet spot, and
/// picking the largest available by default would hand someone a 23 GB BF16.
const PREFERRED: &[&str] = &["Q4_K_M", "Q4_K_XL", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q4_0", "Q3_K_M", "Q8_0"];

/// Pick weights and a projector.
///
/// `requested` names a quantisation explicitly. When absent and `budget_bytes`
/// is given, the largest quantisation that fits is chosen; otherwise the
/// preference list decides.
pub fn select(
    files: &[RepoFile],
    requested: Option<&str>,
    budget_bytes: Option<u64>,
) -> Result<Selection, SelectError> {
    let gguf: Vec<&RepoFile> = files.iter().filter(|f| f.is_gguf()).collect();
    if gguf.is_empty() {
        return Err(SelectError::NoGguf);
    }

    let (projectors, weights): (Vec<&RepoFile>, Vec<&RepoFile>) =
        gguf.into_iter().partition(|f| f.is_mmproj());

    // Group shards, keyed by quantisation.
    let mut groups: std::collections::BTreeMap<String, Vec<&RepoFile>> = Default::default();
    for f in &weights {
        let key = quant_of(&f.path).unwrap_or_else(|| shard_group(&f.path));
        groups.entry(key).or_default().push(f);
    }
    if groups.is_empty() {
        return Err(SelectError::NoGguf);
    }

    for set in groups.values_mut() {
        set.sort_by_key(|f| shard_of(&f.path).map(|(p, _)| p).unwrap_or(0));
    }

    let mut available: Vec<String> = groups.keys().cloned().collect();
    available.sort();

    let chosen = match requested {
        Some(want) => {
            let want_upper = want.to_ascii_uppercase();
            groups
                .keys()
                .find(|k| k.eq_ignore_ascii_case(&want_upper))
                .cloned()
                .ok_or_else(|| SelectError::QuantNotFound {
                    requested: want.to_string(),
                    available: available.clone(),
                })?
        }
        None => auto_pick(&groups, budget_bytes),
    };

    let set = &groups[&chosen];
    let weights: Vec<RepoFile> = set.iter().map(|f| (*f).clone()).collect();
    let weight_bytes: u64 = weights.iter().map(|f| f.size).sum();

    // Match the projector's precision to the weights where possible: an F32
    // projector on a Q4 model wastes memory for no quality gain.
    let mmproj = pick_mmproj(&projectors).cloned();
    let total_bytes = weight_bytes + mmproj.as_ref().map_or(0, |m| m.size);

    Ok(Selection { weights, mmproj, quant: chosen, total_bytes })
}

fn auto_pick(
    groups: &std::collections::BTreeMap<String, Vec<&RepoFile>>,
    budget_bytes: Option<u64>,
) -> String {
    if let Some(budget) = budget_bytes {
        // Largest that fits, since quality rises with size.
        let mut candidates: Vec<(&String, u64)> = groups
            .iter()
            .map(|(k, v)| (k, v.iter().map(|f| f.size).sum::<u64>()))
            .filter(|(_, size)| *size <= budget)
            .collect();
        candidates.sort_by_key(|(_, size)| *size);
        if let Some((k, _)) = candidates.last() {
            return (*k).clone();
        }
        // Nothing fits: take the smallest and let the caller warn.
        let mut all: Vec<(&String, u64)> = groups
            .iter()
            .map(|(k, v)| (k, v.iter().map(|f| f.size).sum::<u64>()))
            .collect();
        all.sort_by_key(|(_, size)| *size);
        return all.first().map(|(k, _)| (*k).clone()).unwrap_or_default();
    }

    for want in PREFERRED {
        if let Some(k) = groups.keys().find(|k| k.eq_ignore_ascii_case(want)) {
            return k.clone();
        }
    }
    groups.keys().next().cloned().unwrap_or_default()
}

/// Prefer F16 for the projector: F32 doubles the size for no practical gain,
/// and BF16 is not universally supported by older builds.
fn pick_mmproj<'a>(projectors: &[&'a RepoFile]) -> Option<&'a RepoFile> {
    for want in ["F16", "BF16", "F32"] {
        if let Some(p) = projectors
            .iter()
            .find(|p| quant_of(&p.path).as_deref() == Some(want))
        {
            return Some(p);
        }
    }
    projectors.first().copied()
}

/// Derive an ozgent `name:tag` from a repository id and quantisation.
///
/// `unsloth/gemma-3-12b-it-GGUF` + `Q4_K_M` becomes `gemma-3-12b-it:Q4_K_M`:
/// the owner and the `-GGUF` suffix carry no information once installed.
pub fn derive_ref(repo_id: &str, quant: &str) -> String {
    let name = repo_id.rsplit('/').next().unwrap_or(repo_id);
    let mut name = name.to_string();
    for suffix in ["-GGUF", "-gguf", ".GGUF", ".gguf"] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            name = stripped.to_string();
            break;
        }
    }
    format!("{}:{}", name, quant)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(path: &str, size: u64) -> RepoFile {
        RepoFile { path: path.into(), size, sha256: None }
    }

    /// The real listing of unsloth/gemma-3-12b-it-GGUF, trimmed.
    fn gemma_repo() -> Vec<RepoFile> {
        vec![
            f("README.md", 100),
            f("config.json", 1665),
            f("gemma-3-12b-it-BF16.gguf", 23_540_151_520),
            f("gemma-3-12b-it-IQ4_XS.gguf", 6_550_964_576),
            f("gemma-3-12b-it-Q2_K.gguf", 4_768_221_536),
            f("gemma-3-12b-it-Q4_K_M.gguf", 7_300_000_000),
            f("gemma-3-12b-it-Q8_0.gguf", 12_500_000_000),
            f("gemma-3-12b-it-UD-Q4_K_XL.gguf", 7_600_000_000),
            f("mmproj-F16.gguf", 850_000_000),
            f("mmproj-F32.gguf", 1_700_000_000),
            f("mmproj-BF16.gguf", 850_000_000),
        ]
    }

    #[test]
    fn extracts_quantisation_from_real_filenames() {
        for (path, want) in [
            ("gemma-3-12b-it-Q4_K_M.gguf", "Q4_K_M"),
            ("gemma-3-12b-it-UD-Q4_K_XL.gguf", "Q4_K_XL"),
            ("gemma-3-12b-it-IQ4_XS.gguf", "IQ4_XS"),
            ("Meta-Llama-3-8B.Q5_K_S.gguf", "Q5_K_S"),
            ("model-q4_0.gguf", "Q4_0"),
            ("mmproj-F16.gguf", "F16"),
            ("mmproj-BF16.gguf", "BF16"),
            ("qwen3-30b-a3b-Q6_K.gguf", "Q6_K"),
        ] {
            assert_eq!(quant_of(path).as_deref(), Some(want), "for {path}");
        }
    }

    #[test]
    fn f16_does_not_match_inside_bf16() {
        // A naive substring search returns F16 here and picks the wrong file.
        assert_eq!(quant_of("mmproj-BF16.gguf").as_deref(), Some("BF16"));
    }

    #[test]
    fn a_filename_with_no_quantisation_yields_none() {
        assert_eq!(quant_of("model.gguf"), None);
        assert_eq!(quant_of("ggml-vocab.gguf"), None);
    }

    #[test]
    fn picks_a_sensible_default_rather_than_the_largest() {
        let s = select(&gemma_repo(), None, None).unwrap();
        assert_eq!(s.quant, "Q4_K_M", "must not default to a 23 GB BF16");
        assert_eq!(s.weights.len(), 1);
    }

    #[test]
    fn an_explicit_quantisation_is_honoured_case_insensitively() {
        let s = select(&gemma_repo(), Some("q8_0"), None).unwrap();
        assert_eq!(s.quant, "Q8_0");
        assert!(s.weights[0].path.contains("Q8_0"));
    }

    #[test]
    fn an_unknown_quantisation_lists_what_is_available() {
        let err = select(&gemma_repo(), Some("Q9_MEGA"), None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Q9_MEGA"), "{msg}");
        assert!(msg.contains("Q4_K_M"), "should list real options: {msg}");
    }

    #[test]
    fn a_budget_picks_the_largest_that_fits() {
        // 8 GB card: Q4_K_M (7.3 GB) fits, Q8_0 (12.5) and BF16 do not.
        let s = select(&gemma_repo(), None, Some(7_500_000_000)).unwrap();
        assert_eq!(s.quant, "Q4_K_M");

        let bigger = select(&gemma_repo(), None, Some(13_000_000_000)).unwrap();
        assert_eq!(bigger.quant, "Q8_0", "with room, take the better quant");
    }

    #[test]
    fn a_tiny_budget_falls_back_to_the_smallest() {
        let s = select(&gemma_repo(), None, Some(1_000_000)).unwrap();
        assert_eq!(s.quant, "Q2_K", "must still return something actionable");
    }

    #[test]
    fn the_vision_projector_is_found_and_never_chosen_as_weights() {
        let s = select(&gemma_repo(), None, None).unwrap();
        let mm = s.mmproj.expect("gemma 3 is multimodal");
        assert!(mm.path.contains("mmproj"));
        assert!(
            !s.weights.iter().any(|w| w.path.contains("mmproj")),
            "a projector must never be selected as the main weights"
        );
    }

    #[test]
    fn f16_projector_is_preferred_over_f32() {
        let s = select(&gemma_repo(), None, None).unwrap();
        assert_eq!(
            s.mmproj.unwrap().path,
            "mmproj-F16.gguf",
            "F32 doubles the size for no practical gain"
        );
    }

    #[test]
    fn sharded_weights_are_grouped_and_ordered() {
        // The layout large repos actually use: quantisation, then shard.
        let files = vec![
            f("DeepSeek-R1-Q4_K_M-00003-of-00003.gguf", 3),
            f("DeepSeek-R1-Q4_K_M-00001-of-00003.gguf", 1),
            f("DeepSeek-R1-Q4_K_M-00002-of-00003.gguf", 2),
        ];
        let s = select(&files, Some("Q4_K_M"), None).unwrap();
        assert_eq!(s.weights.len(), 3, "all shards must be selected");
        let order: Vec<&str> = s.weights.iter().map(|w| w.path.as_str()).collect();
        assert_eq!(order[0], "DeepSeek-R1-Q4_K_M-00001-of-00003.gguf", "shard 1 must be first");
        assert_eq!(order[2], "DeepSeek-R1-Q4_K_M-00003-of-00003.gguf");
        assert_eq!(s.total_bytes, 6, "size is the sum of every shard");
    }

    #[test]
    fn shards_are_recognised_in_either_filename_layout() {
        for path in [
            "DeepSeek-R1-Q4_K_M-00002-of-00009.gguf",
            "model-00002-of-00009-Q4_K_M.gguf",
            "plain-00002-of-00009.gguf",
        ] {
            assert_eq!(shard_of(path), Some((2, 9)), "for {path}");
        }
        assert_eq!(shard_of("model-Q4_K_M.gguf"), None, "unsharded files have no position");
    }

    #[test]
    fn unquantised_shards_still_group_together() {
        let files = vec![
            f("model-00002-of-00002.gguf", 2),
            f("model-00001-of-00002.gguf", 1),
        ];
        let s = select(&files, None, None).unwrap();
        assert_eq!(s.weights.len(), 2, "both shards belong to one model");
        assert!(s.weights[0].path.contains("00001"), "and in order");
    }

    #[test]
    fn total_bytes_includes_the_projector() {
        let s = select(&gemma_repo(), Some("Q4_K_M"), None).unwrap();
        assert_eq!(s.total_bytes, 7_300_000_000 + 850_000_000);
    }

    #[test]
    fn a_repo_with_no_gguf_is_rejected_clearly() {
        let files = vec![f("README.md", 1), f("pytorch_model.bin", 2)];
        assert!(matches!(select(&files, None, None), Err(SelectError::NoGguf)));
    }

    #[test]
    fn derives_a_clean_model_reference() {
        assert_eq!(derive_ref("unsloth/gemma-3-12b-it-GGUF", "Q4_K_M"), "gemma-3-12b-it:Q4_K_M");
        assert_eq!(derive_ref("bartowski/Qwen3-8B-GGUF", "Q6_K"), "Qwen3-8B:Q6_K");
        assert_eq!(derive_ref("someone/plain-model", "Q4_0"), "plain-model:Q4_0");
    }

    #[test]
    fn the_derived_reference_is_a_valid_model_ref() {
        // It becomes a directory path, so it must survive validation.
        let s = derive_ref("unsloth/gemma-3-12b-it-GGUF", "Q4_K_M");
        let parsed = ozgent_core::ModelRef::parse(&s).expect("must be a legal ref");
        assert_eq!(parsed.name, "gemma-3-12b-it");
        assert_eq!(parsed.tag, "Q4_K_M");
    }
}
