//! Builds llama.cpp's `mtmd` multimodal library and binds it.
//!
//! `llama-cpp-sys-2` configures llama.cpp with `LLAMA_BUILD_TOOLS=OFF`, and
//! mtmd lives under `tools/`, so the vision code ships in the vendored tree but
//! is never compiled. Rather than fork that crate, this compiles the same
//! sources against the headers it already vendored and links them alongside the
//! `llama` and `ggml` static libraries it produced — so there is exactly one
//! copy of llama.cpp in the build.

use std::path::{Path, PathBuf};

fn main() {
    let source = llama_cpp_source().expect(
        "cannot locate the vendored llama.cpp sources. \
         Build llama-cpp-sys-2 first, or set LLAMA_CPP_SOURCE to its llama.cpp directory.",
    );
    let mtmd = source.join("tools/mtmd");
    assert!(
        mtmd.join("mtmd.cpp").is_file(),
        "no mtmd sources under {}",
        mtmd.display()
    );

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(source.join("include"))
        .include(source.join("src"))
        .include(source.join("ggml/include"))
        .include(source.join("ggml/src"))
        .include(&mtmd)
        .include(source.join("vendor"))
        // The vendored miniaudio and stb headers are third-party; their
        // warnings are not ours to fix and would drown out anything real.
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-cast-qual")
        .warnings(false);

    for file in mtmd_sources(&mtmd) {
        build.file(file);
    }
    build.compile("mtmd");

    // mtmd's symbols resolve against the llama and ggml libraries that
    // llama-cpp-sys-2 already emitted link directives for; nothing to add here.
    generate_bindings(&mtmd, &source);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_SOURCE");
    println!("cargo:source={}", source.display());
}

/// Every mtmd translation unit except the ones that build executables.
fn mtmd_sources(mtmd: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut push_dir = |dir: PathBuf| {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "cpp") {
                    let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    // `mtmd-cli` has a `main`, and the deprecation shim exists
                    // only to print a message from a retired binary.
                    if name != "mtmd-cli.cpp" && name != "deprecation-warning.cpp" {
                        out.push(path);
                    }
                }
            }
        }
    };
    push_dir(mtmd.to_path_buf());
    push_dir(mtmd.join("models"));
    // `debug/` is a standalone CLI harness that pulls in `common/arg.h`; the
    // debug entry points it declares are defined in mtmd.cpp, so only the
    // header is needed and it is already on the include path.
    out.sort();
    out
}

fn generate_bindings(mtmd: &Path, source: &Path) {
    let bindings = bindgen::Builder::default()
        .header(mtmd.join("mtmd.h").to_string_lossy())
        .header(mtmd.join("mtmd-helper.h").to_string_lossy())
        .clang_arg(format!("-I{}", mtmd.display()))
        .clang_arg(format!("-I{}", source.join("include").display()))
        .clang_arg(format!("-I{}", source.join("ggml/include").display()))
        .allowlist_function("mtmd_.*")
        .allowlist_type("mtmd_.*")
        .allowlist_var("MTMD_.*")
        // llama's own types come from llama-cpp-sys-2; binding them again
        // would produce two incompatible definitions of the same struct.
        .blocklist_type("llama_.*")
        .blocklist_function("llama_.*")
        // Every llama type mtmd's headers mention must come from the one crate
        // that defines them, or the same struct exists twice and nothing links.
        .raw_line(
            "use llama_cpp_sys_2::{llama_batch, llama_context, llama_flash_attn_type, \
             llama_model, llama_pos, llama_seq_id, llama_token};",
        )
        .derive_debug(true)
        .generate()
        .expect("generating mtmd bindings");

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("writing mtmd bindings");
}

/// Find the llama.cpp tree `llama-cpp-sys-2` vendored.
fn llama_cpp_source() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("LLAMA_CPP_SOURCE") {
        return Some(PathBuf::from(explicit));
    }

    // The workspace patches `llama-cpp-sys-2` to `vendor/`, so the tree in
    // this repository is by definition the one being compiled. Checked first
    // because it is the only lookup that cannot fail for a reason outside this
    // repository: the two below depend on cmake having installed its package
    // config in a particular place, or on a copy of the crate happening to be
    // unpacked in the registry — and a machine that has never built the
    // unpatched crate has no such copy. That is exactly a fresh clone, which
    // is to say every clone but the author's.
    if let Some(vendored) = vendored_source() {
        return Some(vendored);
    }

    // `llama-cpp-sys-2` declares `links = "llama"`, so its metadata reaches us
    // as DEP_LLAMA_*. Its cmake build directory records where it configured
    // from, which is the vendored source we need.
    if let Ok(ggml_dir) = std::env::var("DEP_LLAMA_GGML_CMAKE_DIR") {
        let mut dir = PathBuf::from(ggml_dir);
        for _ in 0..6 {
            let cache = dir.join("CMakeCache.txt");
            if let Ok(text) = std::fs::read_to_string(&cache) {
                for line in text.lines() {
                    if let Some(path) = line.strip_prefix("CMAKE_HOME_DIRECTORY:INTERNAL=") {
                        let candidate = PathBuf::from(path.trim());
                        if candidate.join("tools/mtmd/mtmd.cpp").is_file() {
                            return Some(candidate);
                        }
                    }
                }
            }
            if !dir.pop() {
                break;
            }
        }
    }

    // Last resort: the crate is unpacked in the registry next to our own build.
    let registry = home_registry()?;
    let mut best: Option<PathBuf> = None;
    for entry in std::fs::read_dir(registry).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("llama-cpp-sys-2-") {
            let candidate = entry.path().join("llama.cpp");
            if candidate.join("tools/mtmd/mtmd.cpp").is_file() {
                best = Some(candidate);
            }
        }
    }
    best
}

/// `vendor/llama-cpp-sys-2/llama.cpp`, found by walking up from this crate.
///
/// Walked rather than hard-coded as `../../vendor/...` so that moving this
/// crate within the workspace does not silently break the build.
fn vendored_source() -> Option<PathBuf> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    for dir in manifest.ancestors().take(6) {
        let candidate = dir.join("vendor/llama-cpp-sys-2/llama.cpp");
        if candidate.join("tools/mtmd/mtmd.cpp").is_file() {
            return Some(candidate);
        }
    }
    None
}

fn home_registry() -> Option<PathBuf> {
    let home = std::env::var("CARGO_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".cargo")))?;
    let src = home.join("registry/src");
    std::fs::read_dir(&src).ok()?.flatten().next().map(|e| e.path())
}
