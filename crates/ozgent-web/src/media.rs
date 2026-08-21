//! Storing the images a conversation was given.
//!
//! Attachments have to survive a reload or the transcript stops making sense —
//! a question about "this image" with no image is not a conversation. The bytes
//! go on disk under `~/ozgent/media` and only their file names are stored in
//! SQLite, because a database is a poor place for megabytes of PNG.
//!
//! What is stored is *not* what the model saw. The model is given the original
//! upload at full fidelity; the copy kept for redisplay is downscaled, since
//! its only job from then on is to be looked at by a person. That keeps a
//! phone photo from costing twelve megabytes a turn without ever putting a
//! degraded image in front of the model.

use std::path::{Path, PathBuf};

/// Longest edge of a stored copy, in pixels.
///
/// Comfortably above what vision encoders resolve — Qwen-VL family models tile
/// at a few hundred pixels — so a person can still read text in a screenshot,
/// while a 4000px camera image stops being stored at 4000px.
const MAX_EDGE: u32 = 1400;

/// JPEG quality for photographic content. High enough that compression
/// artefacts stay invisible at normal viewing size.
const QUALITY: u8 = 85;

/// Files this large are re-encoded; below it, the original is kept verbatim.
const REENCODE_ABOVE: usize = 256 * 1024;

/// How long an attachment is kept before being swept up.
pub const RETENTION_DAYS: u64 = 30;

pub fn dir(paths: &ozgent_core::Paths) -> PathBuf {
    paths.root().join("media")
}

/// Write one attachment, returning the file name to store against the message.
///
/// Small images are written untouched: re-encoding a 40 KB PNG icon as JPEG
/// would make it larger and blurrier at once.
pub fn store(
    paths: &ozgent_core::Paths,
    bytes: &[u8],
    mime: Option<&str>,
) -> std::io::Result<String> {
    let root = dir(paths);
    std::fs::create_dir_all(&root)?;

    let (data, extension) = match shrink(bytes, mime) {
        Some(jpeg) => (jpeg, "jpg"),
        None => (bytes.to_vec(), extension_for(mime, bytes)),
    };

    let name = format!("{}.{extension}", token());
    std::fs::write(root.join(&name), data)?;
    Ok(name)
}

/// Downscale and re-encode, or `None` to keep the original.
fn shrink(bytes: &[u8], mime: Option<&str>) -> Option<Vec<u8>> {
    // An animated GIF loses its animation through a still decode, which is a
    // worse outcome than the disk it saves.
    if mime.is_some_and(|m| m.contains("gif")) {
        return None;
    }

    let decoded = image::load_from_memory(bytes).ok()?;
    let (w, h) = (decoded.width(), decoded.height());
    let oversized = w.max(h) > MAX_EDGE;

    // Two independent reasons to re-encode, and file size alone is not enough:
    // a six-megapixel gradient can compress to under the byte threshold while
    // still being far larger than anything needs to be redisplayed at.
    if !oversized && bytes.len() <= REENCODE_ABOVE {
        return None;
    }

    let scaled = if oversized {
        decoded.resize(MAX_EDGE, MAX_EDGE, image::imageops::FilterType::Lanczos3)
    } else {
        decoded
    };

    let mut out = std::io::Cursor::new(Vec::new());
    // Alpha cannot survive JPEG, so flatten first: a transparent background
    // would otherwise come back as noise.
    let rgb = image::DynamicImage::ImageRgb8(scaled.to_rgb8());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, QUALITY)
        .encode_image(&rgb)
        .ok()?;

    let encoded = out.into_inner();
    // When the goal was to bound the pixel dimensions, take the result even if
    // it encodes larger — a synthetic gradient is tiny as PNG and bulky as
    // JPEG, but storing it at 3000px was never the intent. When the only
    // reason was file size, a bigger "compressed" copy means the original was
    // already the better choice.
    if oversized || encoded.len() < bytes.len() {
        Some(encoded)
    } else {
        None
    }
}

fn extension_for(mime: Option<&str>, bytes: &[u8]) -> &'static str {
    if let Some(m) = mime {
        for (needle, ext) in [("png", "png"), ("jpeg", "jpg"), ("jpg", "jpg"), ("webp", "webp"), ("gif", "gif")] {
            if m.contains(needle) {
                return ext;
            }
        }
    }
    // Fall back to the magic bytes, since a browser does not always say.
    match bytes {
        [0x89, b'P', b'N', b'G', ..] => "png",
        [0xff, 0xd8, 0xff, ..] => "jpg",
        [b'G', b'I', b'F', ..] => "gif",
        _ => "bin",
    }
}

/// A file name component that cannot collide or escape the media directory.
fn token() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let boxed = Box::new(0u8);
    let addr = Box::into_raw(boxed) as usize;
    // SAFETY: reclaimed immediately; only the address was wanted.
    unsafe { drop(Box::from_raw(addr as *mut u8)) };
    format!("{nanos:x}{addr:x}")
}

/// Reject anything that is not a plain file name.
///
/// The name arrives from the database, but it is used to build a path served
/// over HTTP, so it is validated at the point of use rather than trusted.
pub fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')
        && !name.contains("..")
}

/// Delete attachments older than [`RETENTION_DAYS`].
///
/// Run at startup rather than on a timer: a local server is started often
/// enough, and a background sweep is one more thing to get wrong.
pub fn sweep(paths: &ozgent_core::Paths) -> std::io::Result<usize> {
    let root = dir(paths);
    if !root.exists() {
        return Ok(0);
    }
    let cutoff = std::time::SystemTime::now()
        .checked_sub(std::time::Duration::from_secs(RETENTION_DAYS * 24 * 60 * 60))
        .unwrap_or(std::time::UNIX_EPOCH);

    let mut removed = 0;
    for entry in std::fs::read_dir(&root)?.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let stale = meta
            .modified()
            .map(|m| m < cutoff)
            .unwrap_or(false);
        if stale && std::fs::remove_file(entry.path()).is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Read one stored attachment.
pub fn read(paths: &ozgent_core::Paths, name: &str) -> Option<(Vec<u8>, &'static str)> {
    if !safe_name(name) {
        return None;
    }
    let path: PathBuf = dir(paths).join(name);
    let bytes = std::fs::read(&path).ok()?;
    Some((bytes, content_type(&path)))
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_that_could_escape_the_media_directory_is_refused() {
        // These arrive from the database but end up in a filesystem path.
        assert!(safe_name("abc123.png"));
        assert!(!safe_name("../../etc/passwd"));
        assert!(!safe_name("a/b.png"));
        assert!(!safe_name(""));
        assert!(!safe_name(&"x".repeat(200)));
    }

    #[test]
    fn a_small_image_is_kept_byte_for_byte() {
        // Re-encoding a small PNG as JPEG makes it larger and blurrier at once.
        let small = vec![0u8; 1000];
        assert!(shrink(&small, Some("image/png")).is_none());
    }

    #[test]
    fn an_animated_gif_is_never_re_encoded() {
        // A still decode would silently drop the animation.
        let big = vec![7u8; REENCODE_ABOVE + 1];
        assert!(shrink(&big, Some("image/gif")).is_none());
    }

    #[test]
    fn a_large_photo_is_downscaled_but_stays_legible() {
        let wide = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(3000, 2000, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        }));
        let mut original = std::io::Cursor::new(Vec::new());
        wide.write_to(&mut original, image::ImageFormat::Png).unwrap();
        let original = original.into_inner();

        let shrunk = shrink(&original, Some("image/png")).expect("a 3000px image should shrink");
        // Deliberately not asserting fewer bytes here: a smooth gradient is
        // tiny as PNG and bulky as JPEG. The guarantee for an oversized image
        // is its dimensions, not its size — see the photo test below.
        let decoded = image::load_from_memory(&shrunk).expect("must still decode");
        assert_eq!(decoded.width().max(decoded.height()), MAX_EDGE, "longest edge capped");
        assert!(
            decoded.width().min(decoded.height()) > 600,
            "aspect ratio kept and still large enough to read: {}x{}",
            decoded.width(),
            decoded.height()
        );
    }

    #[test]
    fn a_bulky_photograph_actually_gets_smaller() {
        // Noise, which is what a real photograph looks like to an encoder: PNG
        // cannot compress it, so the stored copy must genuinely shrink.
        let noisy = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(2200, 1600, |x, y| {
            let n = (x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503)) as u8;
            image::Rgb([n, n.wrapping_add(97), n.wrapping_mul(3)])
        }));
        let mut original = std::io::Cursor::new(Vec::new());
        noisy.write_to(&mut original, image::ImageFormat::Png).unwrap();
        let original = original.into_inner();

        let shrunk = shrink(&original, Some("image/png")).expect("should re-encode");
        assert!(
            shrunk.len() < original.len() / 2,
            "a photo should shrink substantially: {} -> {}",
            original.len(),
            shrunk.len()
        );
        let decoded = image::load_from_memory(&shrunk).expect("must still decode");
        assert_eq!(decoded.width().max(decoded.height()), MAX_EDGE);
    }

    #[test]
    fn the_extension_falls_back_to_the_magic_bytes() {
        assert_eq!(extension_for(None, &[0x89, b'P', b'N', b'G', 0, 0]), "png");
        assert_eq!(extension_for(None, &[0xff, 0xd8, 0xff, 0]), "jpg");
        assert_eq!(extension_for(Some("image/webp"), &[]), "webp");
    }

    #[test]
    fn names_do_not_collide() {
        let names: std::collections::HashSet<String> = (0..200).map(|_| token()).collect();
        assert_eq!(names.len(), 200, "every attachment needs its own file");
    }
}
