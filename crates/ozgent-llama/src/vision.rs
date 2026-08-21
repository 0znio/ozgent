//! Finding images in what the user typed.
//!
//! The user should be able to drop a file on the terminal, paste a URL, or
//! type a path, and have it become an image in the prompt without any flag.
//! Terminals hand those over in several shapes, so the detector accounts for:
//!
//! - drag-and-drop, which arrives quoted (`'/a/b.png'`) or backslash-escaped
//!   (`/a/b\ c.png`) depending on the terminal
//! - `file://` URLs, which is what Wayland and GTK apps paste
//! - `~` paths
//! - `http(s)` URLs and `data:` URIs
//!
//! Extraction is deliberately conservative: only recognised image extensions
//! count, so a message that merely mentions a path is left alone.

use ozgent_core::ImageSource;
use std::path::PathBuf;

/// Extensions llama.cpp's multimodal path can decode.
pub const IMAGE_EXTENSIONS: &[&str] =
    &["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "tif"];

/// Audio formats mtmd decodes, for models with an audio projector.
///
/// Handled through exactly the same path as images: mtmd detects the format
/// from the bytes, so the only thing that differs is which extensions get
/// picked out of a pasted line.
const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"];

/// What was found in a line of user input.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    /// The message with image references removed and whitespace tidied.
    pub text: String,
    /// Images, in the order they appeared.
    pub images: Vec<ImageSource>,
}

impl Extracted {
    pub fn has_images(&self) -> bool {
        !self.images.is_empty()
    }
}

/// Pull image references out of a line of input.
pub fn extract(input: &str) -> Extracted {
    let mut images = Vec::new();
    let mut kept: Vec<String> = Vec::new();

    for token in tokenize(input) {
        match classify(&token.value) {
            Some(source) => images.push(source),
            None => kept.push(token.raw),
        }
    }

    Extracted { text: kept.join(" ").trim().to_string(), images }
}

struct Token {
    /// The original text, so non-image tokens survive verbatim.
    raw: String,
    /// Unquoted and unescaped, for classification.
    value: String,
}

/// Split on whitespace, honouring quotes and backslash escapes.
fn tokenize(input: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut raw = String::new();
    let mut value = String::new();
    let mut quote: Option<char> = None;
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\\' if quote.is_none() => {
                raw.push(ch);
                // A backslash-escaped space is part of a dragged path, not a
                // separator.
                if let Some(&next) = chars.peek() {
                    raw.push(next);
                    value.push(next);
                    chars.next();
                }
            }
            '\'' | '"' => {
                raw.push(ch);
                match quote {
                    Some(q) if q == ch => quote = None,
                    Some(_) => value.push(ch),
                    None => quote = Some(ch),
                }
            }
            c if c.is_whitespace() && quote.is_none() => {
                if !raw.is_empty() {
                    out.push(Token { raw: std::mem::take(&mut raw), value: std::mem::take(&mut value) });
                }
            }
            c => {
                raw.push(c);
                value.push(c);
            }
        }
    }
    if !raw.is_empty() {
        out.push(Token { raw, value });
    }
    out
}

/// Decide whether one token names an image.
fn classify(token: &str) -> Option<ImageSource> {
    let trimmed = token.trim_matches(|c: char| matches!(c, ',' | ';' | ')' | '(' | '>' | '<'));
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.starts_with("data:image/") || trimmed.starts_with("data:audio/") {
        return decode_data_uri(trimmed);
    }

    if let Some(rest) = trimmed.strip_prefix("file://") {
        let path = percent_decode(rest.strip_prefix("localhost").unwrap_or(rest));
        return has_media_extension(&path).then(|| ImageSource::Path { path: PathBuf::from(path) });
    }

    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        // Strip a query string before testing the extension, since image CDNs
        // append cache-busting parameters.
        let bare = trimmed.split(['?', '#']).next().unwrap_or(trimmed);
        return has_media_extension(bare)
            .then(|| ImageSource::Url { url: trimmed.to_string() });
    }

    // Anything left is a candidate filesystem path. Requiring the file to
    // exist keeps a mention of "diagram.png" in prose from becoming an image.
    if has_media_extension(trimmed) {
        let expanded = expand_tilde(trimmed);
        if expanded.is_file() {
            return Some(ImageSource::Path { path: expanded });
        }
    }
    None
}

/// Whether a path names something mtmd can decode.
fn has_media_extension(s: &str) -> bool {
    let Some((_, ext)) = s.rsplit_once('.') else { return false };
    let ext = ext.to_ascii_lowercase();
    IMAGE_EXTENSIONS.contains(&ext.as_str()) || AUDIO_EXTENSIONS.contains(&ext.as_str())
}

/// Whether this source is audio rather than a picture.
///
/// Only used to tell the user which capability their model is missing; mtmd
/// itself decides from the bytes.
pub fn looks_like_audio(source: &ozgent_core::ImageSource) -> bool {
    let name = match source {
        ozgent_core::ImageSource::Path { path } => path.to_string_lossy().to_string(),
        ozgent_core::ImageSource::Url { url } => url.clone(),
        ozgent_core::ImageSource::Bytes { mime, .. } => {
            return mime.as_deref().is_some_and(|m| m.starts_with("audio/"));
        }
    };
    name.rsplit_once('.')
        .map(|(_, e)| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

fn expand_tilde(s: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = dirs_home() {
            return home.join(rest);
        }
    }
    PathBuf::from(s)
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn decode_data_uri(uri: &str) -> Option<ImageSource> {
    let (meta, payload) = uri.split_once(',')?;
    if !meta.ends_with(";base64") {
        return None;
    }
    let mime = meta.strip_prefix("data:")?.strip_suffix(";base64")?.to_string();
    let bytes = base64_decode(payload)?;
    Some(ImageSource::Bytes { bytes, mime: Some(mime) })
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut acc = 0u32;
    let mut bits = 0u8;
    let mut out = Vec::new();
    for ch in s.bytes() {
        if ch == b'=' || ch.is_ascii_whitespace() {
            continue;
        }
        let v = CHARS.iter().position(|&c| c == ch)? as u32;
        acc = acc << 6 | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Creates a real image file, since path detection requires existence.
    ///
    /// Each fixture gets its own directory: tests run in parallel in one
    /// process, so a shared directory lets one test's cleanup delete another
    /// test's file.
    struct Fixture {
        dir: PathBuf,
        path: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "ozgent-vision-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(name);
            std::fs::write(&path, b"\x89PNG\r\n\x1a\n").unwrap();
            Self { dir, path }
        }
        fn str(&self) -> String {
            self.path.display().to_string()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn finds_a_plain_path_and_keeps_the_rest_of_the_message() {
        let f = Fixture::new("shot.png");
        let got = extract(&format!("what is in {} exactly?", f.str()));

        assert_eq!(got.images.len(), 1);
        assert_eq!(got.text, "what is in exactly?");
        match &got.images[0] {
            ImageSource::Path { path } => assert_eq!(path, &f.path),
            other => panic!("expected a path, got {other:?}"),
        }
    }

    #[test]
    fn handles_drag_and_drop_quoting() {
        let f = Fixture::new("my shot.png");
        let got = extract(&format!("describe '{}'", f.str()));
        assert_eq!(got.images.len(), 1, "quoted path with a space must be found");
        assert_eq!(got.text, "describe");
    }

    #[test]
    fn handles_backslash_escaped_spaces() {
        let f = Fixture::new("my shot.png");
        let escaped = f.str().replace(' ', "\\ ");
        let got = extract(&format!("describe {escaped}"));
        assert_eq!(got.images.len(), 1, "escaped path must be found");
        assert_eq!(got.text, "describe");
    }

    #[test]
    fn handles_file_urls_with_percent_encoding() {
        let f = Fixture::new("my shot.png");
        let encoded = f.str().replace(' ', "%20");
        let got = extract(&format!("look at file://{encoded}"));
        assert_eq!(got.images.len(), 1);
        match &got.images[0] {
            ImageSource::Path { path } => assert_eq!(path, &f.path),
            other => panic!("expected a path, got {other:?}"),
        }
    }

    #[test]
    fn finds_http_urls_including_ones_with_query_strings() {
        let got = extract("compare https://example.com/a.png and https://cdn.test/b.jpg?w=800");
        assert_eq!(got.images.len(), 2);
        assert_eq!(got.text, "compare and");
        match &got.images[1] {
            ImageSource::Url { url } => assert!(url.ends_with("?w=800"), "query must be kept: {url}"),
            other => panic!("expected a url, got {other:?}"),
        }
    }

    #[test]
    fn decodes_data_uris() {
        // "hello" in base64.
        let got = extract("data:image/png;base64,aGVsbG8=");
        assert_eq!(got.images.len(), 1);
        match &got.images[0] {
            ImageSource::Bytes { bytes, mime } => {
                assert_eq!(bytes, b"hello");
                assert_eq!(mime.as_deref(), Some("image/png"));
            }
            other => panic!("expected bytes, got {other:?}"),
        }
    }

    #[test]
    fn a_mentioned_but_nonexistent_path_stays_as_text() {
        let got = extract("the file diagram.png shows the layout");
        assert!(got.images.is_empty(), "must not invent an image from prose");
        assert_eq!(got.text, "the file diagram.png shows the layout");
    }

    #[test]
    fn non_image_files_are_ignored() {
        let f = Fixture::new("notes.txt");
        let got = extract(&format!("read {}", f.str()));
        assert!(got.images.is_empty(), "only image extensions count");
    }

    #[test]
    fn ordinary_messages_are_untouched() {
        let text = "explain how mixture of experts routing works";
        let got = extract(text);
        assert!(got.images.is_empty());
        assert_eq!(got.text, text);
    }

    #[test]
    fn tilde_paths_are_expanded() {
        let home = std::env::var("HOME").unwrap();
        let dir = PathBuf::from(&home).join(".ozgent-vision-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.png");
        std::fs::write(&path, b"x").unwrap();

        let got = extract("see ~/.ozgent-vision-test/t.png");
        assert_eq!(got.images.len(), 1, "~ must expand");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn several_images_keep_their_order() {
        let a = Fixture::new("a.png");
        let b = Fixture::new("b.jpg");
        let got = extract(&format!("{} then {}", a.str(), b.str()));
        assert_eq!(got.images.len(), 2);
        match (&got.images[0], &got.images[1]) {
            (ImageSource::Path { path: p1 }, ImageSource::Path { path: p2 }) => {
                assert!(p1.ends_with("a.png"));
                assert!(p2.ends_with("b.jpg"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn trailing_punctuation_does_not_break_a_url() {
        let got = extract("see https://example.com/a.png, then stop");
        assert_eq!(got.images.len(), 1, "a trailing comma must not defeat detection");
    }
}
