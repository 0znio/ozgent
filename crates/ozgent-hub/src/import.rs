//! Registering a GGUF file that is already on disk.
//!
//! The file may live anywhere — a downloads folder, an external drive, a
//! directory shared with another runtime — so importing links rather than
//! copies where it can. A hard link costs no space and leaves the original
//! path working, which matters when the file is 20 GB.

use crate::hf::HubError;
use crate::select::quant_of;
use ozgent_core::manifest::{Capability, Manifest, Source};
use ozgent_core::{ModelRef, Paths};
use std::path::{Path, PathBuf};

/// Every GGUF file begins with these four bytes.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

#[derive(Debug, Clone)]
pub struct ImportRequest {
    /// Where to install it, as `name:tag`.
    pub reference: String,
    pub weights: PathBuf,
    pub mmproj: Option<PathBuf>,
    /// Copy instead of hard-linking. Needed across filesystems.
    pub copy: bool,
}

#[derive(Debug, Clone)]
pub struct Imported {
    pub model: ModelRef,
    pub dir: PathBuf,
    pub manifest: Manifest,
    /// True when the bytes were duplicated rather than linked.
    pub copied: bool,
}

/// Check the file really is a GGUF before installing it.
///
/// Registering a mislabelled file would fail much later, inside llama.cpp,
/// with a far less useful message.
pub fn verify_gguf(path: &Path) -> Result<(), HubError> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .map_err(|e| HubError::Io { path: path.display().to_string(), source: e })?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| HubError::Io { path: path.display().to_string(), source: e })?;

    if &magic != GGUF_MAGIC {
        return Err(HubError::Other(format!(
            "{} is not a GGUF file (it begins with {:?}, expected \"GGUF\"). \
             Safetensors and PyTorch checkpoints must be converted first.",
            path.display(),
            String::from_utf8_lossy(&magic)
        )));
    }
    Ok(())
}

/// Install a local GGUF into the models directory.
pub fn import(paths: &Paths, request: &ImportRequest) -> Result<Imported, HubError> {
    verify_gguf(&request.weights)?;
    if let Some(mm) = &request.mmproj {
        verify_gguf(mm)?;
    }

    let model = ModelRef::parse(&request.reference).map_err(|e| {
        HubError::Other(format!("{:?} is not a usable model name: {e}", request.reference))
    })?;

    let dir = paths.model_dir(&model);
    if dir.join(ozgent_core::manifest::MANIFEST_FILE).is_file() {
        return Err(HubError::Other(format!(
            "{model} is already installed at {}. Remove it first with: ozgent rm {model}",
            dir.display()
        )));
    }
    std::fs::create_dir_all(&dir)
        .map_err(|e| HubError::Io { path: dir.display().to_string(), source: e })?;

    let (weights_name, copied) = place(&request.weights, &dir, request.copy)?;
    let mmproj_name = match &request.mmproj {
        Some(p) => Some(place(p, &dir, request.copy)?.0),
        None => None,
    };

    let mut manifest = Manifest::new(&model, &weights_name);
    manifest.mmproj = mmproj_name.as_ref().map(PathBuf::from);
    manifest.quantization = quant_of(&weights_name);
    manifest.size_bytes = std::fs::metadata(&request.weights).ok().map(|m| m.len());
    if manifest.mmproj.is_some() {
        manifest.capabilities.push(Capability::Vision);
    }
    manifest.source = Some(Source {
        kind: "local".into(),
        // Record where it came from, so a later `show` can explain the origin.
        uri: request
            .weights
            .canonicalize()
            .unwrap_or_else(|_| request.weights.clone())
            .display()
            .to_string(),
        digests: Vec::new(),
    });

    manifest
        .save(&dir)
        .map_err(|e| HubError::Other(format!("writing the manifest: {e}")))?;

    Ok(Imported { model, dir, manifest, copied })
}

/// Link or copy `src` into `dir`, returning the file name used.
///
/// A hard link is tried first: it is instant and costs no extra space. It only
/// works within one filesystem, so a cross-device failure falls back to a copy
/// rather than being reported as an error.
fn place(src: &Path, dir: &Path, force_copy: bool) -> Result<(String, bool), HubError> {
    let name = src
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model.gguf".into());
    let dest = dir.join(&name);

    if dest.exists() {
        return Ok((name, false));
    }

    if !force_copy && std::fs::hard_link(src, &dest).is_ok() {
        return Ok((name, false));
    }

    std::fs::copy(src, &dest)
        .map_err(|e| HubError::Io { path: dest.display().to_string(), source: e })?;
    Ok((name, true))
}

/// Suggest a `name:tag` for a GGUF path when the user gives none.
///
/// `~/dl/Qwen3-8B-Q4_K_M.gguf` becomes `Qwen3-8B:Q4_K_M`; a file with no
/// recognisable quantisation gets the `latest` tag.
pub fn suggest_reference(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());

    match quant_of(&format!("{stem}.gguf")) {
        Some(quant) => {
            // Strip the quantisation and any separator left behind.
            let lower = stem.to_ascii_lowercase();
            let q_lower = quant.to_ascii_lowercase();
            let name = match lower.rfind(&q_lower) {
                Some(i) => stem[..i].trim_end_matches(['-', '.', '_']).to_string(),
                None => stem.clone(),
            };
            let name = if name.is_empty() { "model".to_string() } else { name };
            format!("{name}:{quant}")
        }
        None => format!("{stem}:latest"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dir(PathBuf);
    impl Dir {
        fn new(label: &str) -> Self {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let d = std::env::temp_dir().join(format!(
                "ozgent-import-{label}-{}-{}",
                std::process::id(),
                N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
        fn gguf(&self, name: &str) -> PathBuf {
            let p = self.0.join(name);
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(&[3, 0, 0, 0]);
            std::fs::write(&p, bytes).unwrap();
            p
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_non_gguf_file_is_rejected_with_an_explanation() {
        let d = Dir::new("bad");
        let p = d.0.join("weights.safetensors");
        std::fs::write(&p, b"NOTGGUF here").unwrap();

        let err = verify_gguf(&p).unwrap_err().to_string();
        assert!(err.contains("not a GGUF"), "{err}");
        assert!(err.contains("converted"), "should say what to do: {err}");
    }

    #[test]
    fn a_real_gguf_header_passes() {
        let d = Dir::new("good");
        assert!(verify_gguf(&d.gguf("m.gguf")).is_ok());
    }

    #[test]
    fn a_missing_file_reports_its_path() {
        let err = verify_gguf(Path::new("/nonexistent/x.gguf")).unwrap_err().to_string();
        assert!(err.contains("/nonexistent/x.gguf"), "{err}");
    }

    #[test]
    fn importing_creates_the_directory_and_manifest() {
        let src = Dir::new("src");
        let home = Dir::new("home");
        let paths = Paths::with_root(&home.0);
        let weights = src.gguf("Qwen3-8B-Q4_K_M.gguf");

        let out = import(
            &paths,
            &ImportRequest {
                reference: "Qwen3-8B:Q4_K_M".into(),
                weights: weights.clone(),
                mmproj: None,
                copy: false,
            },
        )
        .unwrap();

        assert_eq!(out.model.to_string(), "Qwen3-8B:Q4_K_M");
        assert!(out.dir.ends_with("models/Qwen3-8B/Q4_K_M"));
        assert!(out.dir.join("Qwen3-8B-Q4_K_M.gguf").exists(), "weights must be placed");
        assert!(out.dir.join("manifest.json").exists());
        assert_eq!(out.manifest.quantization.as_deref(), Some("Q4_K_M"));
        // The original must still work; importing is not a move.
        assert!(weights.exists(), "the source file must not be consumed");
    }

    #[test]
    fn a_hard_link_avoids_duplicating_the_bytes() {
        let src = Dir::new("link-src");
        let home = Dir::new("link-home");
        let paths = Paths::with_root(&home.0);
        let weights = src.gguf("m-Q4_0.gguf");

        let out = import(
            &paths,
            &ImportRequest {
                reference: "m:Q4_0".into(),
                weights: weights.clone(),
                mmproj: None,
                copy: false,
            },
        )
        .unwrap();

        // Same inode means one copy of the data on disk.
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let a = std::fs::metadata(&weights).unwrap();
            let b = std::fs::metadata(out.dir.join("m-Q4_0.gguf")).unwrap();
            if a.dev() == b.dev() {
                assert_eq!(a.ino(), b.ino(), "should be hard-linked, not copied");
            }
        }
    }

    #[test]
    fn copy_mode_duplicates_rather_than_links() {
        let src = Dir::new("copy-src");
        let home = Dir::new("copy-home");
        let paths = Paths::with_root(&home.0);
        let weights = src.gguf("m-Q8_0.gguf");

        let out = import(
            &paths,
            &ImportRequest {
                reference: "m:Q8_0".into(),
                weights,
                mmproj: None,
                copy: true,
            },
        )
        .unwrap();
        assert!(out.copied, "copy was requested");
    }

    #[test]
    fn a_projector_enables_vision() {
        let src = Dir::new("mm-src");
        let home = Dir::new("mm-home");
        let paths = Paths::with_root(&home.0);

        let out = import(
            &paths,
            &ImportRequest {
                reference: "gemma:Q4_K_M".into(),
                weights: src.gguf("gemma-Q4_K_M.gguf"),
                mmproj: Some(src.gguf("mmproj-F16.gguf")),
                copy: false,
            },
        )
        .unwrap();

        assert!(out.manifest.supports_vision());
        assert!(out.dir.join("mmproj-F16.gguf").exists());
    }

    #[test]
    fn importing_over_an_existing_model_is_refused() {
        let src = Dir::new("dup-src");
        let home = Dir::new("dup-home");
        let paths = Paths::with_root(&home.0);
        let req = ImportRequest {
            reference: "m:Q4_0".into(),
            weights: src.gguf("m-Q4_0.gguf"),
            mmproj: None,
            copy: false,
        };

        import(&paths, &req).unwrap();
        let err = import(&paths, &req).unwrap_err().to_string();
        assert!(err.contains("already installed"), "{err}");
        assert!(err.contains("ozgent rm"), "should say how to proceed: {err}");
    }

    #[test]
    fn a_reference_is_suggested_from_the_filename() {
        for (file, want) in [
            ("Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B:Q4_K_M"),
            ("gemma-3-12b-it-Q8_0.gguf", "gemma-3-12b-it:Q8_0"),
            ("Meta-Llama-3.Q5_K_S.gguf", "Meta-Llama-3:Q5_K_S"),
            ("mystery.gguf", "mystery:latest"),
        ] {
            assert_eq!(suggest_reference(Path::new(file)), want, "for {file}");
        }
    }

    #[test]
    fn a_suggested_reference_is_always_parseable() {
        for file in ["Qwen3-8B-Q4_K_M.gguf", "weird name.gguf", "x.gguf"] {
            let r = suggest_reference(Path::new(file));
            assert!(ModelRef::parse(&r).is_ok(), "{r:?} from {file} must be valid");
        }
    }
}
