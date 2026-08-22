//! Embeddings and similarity.
//!
//! The real embedder will be a small GGUF model run through llama.cpp. This
//! module defines the seam for it and ships a dependency-free fallback so
//! memory works before any embedding model is installed.

/// Turns text into a vector. Implemented by llama.cpp at runtime.
pub trait Embedder: Send + Sync {
    fn dimensions(&self) -> usize;

    fn embed(&self, text: &str) -> Vec<f32>;

    /// Embed a batch. Overridden by backends where batching is cheaper.
    fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// So a boxed embedder can be passed wherever an `&impl Embedder` is expected.
///
/// Without this, choosing the embedder at runtime forces every call site to
/// name a concrete type, which is the opposite of what the trait is for.
impl<T: Embedder + ?Sized> Embedder for Box<T> {
    fn dimensions(&self) -> usize {
        (**self).dimensions()
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        (**self).embed(text)
    }

    fn embed_batch(&self, texts: &[String]) -> Vec<Vec<f32>> {
        (**self).embed_batch(texts)
    }
}

/// A lexical embedder using the hashing trick.
///
/// Words are hashed into a fixed number of buckets and the resulting vector is
/// L2-normalised, so cosine similarity reflects weighted word overlap. It
/// cannot recognise a paraphrase that shares no vocabulary — that is what the
/// real model is for — but it is deterministic, needs no download, and gives
/// the vector half of hybrid retrieval something to work with out of the box.
pub struct HashingEmbedder {
    dims: usize,
    /// Also hash adjacent word pairs, which captures a little word order.
    bigrams: bool,
}

impl Default for HashingEmbedder {
    fn default() -> Self {
        Self { dims: 256, bigrams: true }
    }
}

impl HashingEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims: dims.max(16), bigrams: true }
    }
}

impl Embedder for HashingEmbedder {
    fn dimensions(&self) -> usize {
        self.dims
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dims];
        let words: Vec<&str> = tokenize(text);

        for w in &words {
            let (idx, sign) = bucket(w, self.dims);
            v[idx] += sign;
        }
        if self.bigrams {
            for pair in words.windows(2) {
                let joined = format!("{} {}", pair[0], pair[1]);
                let (idx, sign) = bucket(&joined, self.dims);
                // Bigrams are corroborating evidence, not primary signal.
                v[idx] += sign * 0.5;
            }
        }

        normalize(&mut v);
        v
    }
}

/// Lowercase word tokens, dropping punctuation and single characters.
fn tokenize(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-')
        .filter(|w| w.len() > 1)
        .collect()
}

/// Map a token to a bucket and a sign. The signed hash keeps unrelated tokens
/// from systematically inflating similarity when they collide.
fn bucket(token: &str, dims: usize) -> (usize, f32) {
    let h = fnv1a(token.to_ascii_lowercase().as_bytes());
    let idx = (h % dims as u64) as usize;
    let sign = if h & (1 << 63) == 0 { 1.0 } else { -1.0 };
    (idx, sign)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Scale to unit length so cosine similarity is a plain dot product.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity, in `[-1, 1]`.
///
/// Vectors are normalised on the way in, so this is a dot product; the guard
/// covers vectors that arrived from elsewhere unnormalised.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f32::EPSILON);
    (dot / denom).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_deterministic() {
        let e = HashingEmbedder::default();
        assert_eq!(e.embed("the cat sat"), e.embed("the cat sat"));
    }

    #[test]
    fn embeddings_are_unit_length() {
        let e = HashingEmbedder::default();
        let v = e.embed("some ordinary sentence about databases");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit length, got {norm}");
    }

    #[test]
    fn related_text_scores_higher_than_unrelated() {
        let e = HashingEmbedder::default();
        let q = e.embed("how do I configure the gpu layers");
        let near = e.embed("set the gpu layers option to configure offload");
        let far = e.embed("my favourite dessert is lemon tart");

        assert!(
            cosine(&q, &near) > cosine(&q, &far),
            "related {:.3} should beat unrelated {:.3}",
            cosine(&q, &near),
            cosine(&q, &far)
        );
    }

    #[test]
    fn identical_text_is_maximally_similar() {
        let e = HashingEmbedder::default();
        let a = e.embed("mixture of experts routing");
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_text_does_not_panic() {
        let e = HashingEmbedder::default();
        let v = e.embed("");
        assert_eq!(v.len(), e.dimensions());
        assert_eq!(cosine(&v, &v), 0.0, "a zero vector has no direction");
    }

    #[test]
    fn mismatched_dimensions_score_zero_rather_than_panicking() {
        assert_eq!(cosine(&[1.0, 0.0], &[1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn batch_matches_individual_embedding() {
        let e = HashingEmbedder::default();
        let texts = vec!["first one".to_string(), "second one".to_string()];
        let batch = e.embed_batch(&texts);
        assert_eq!(batch[0], e.embed(&texts[0]));
        assert_eq!(batch[1], e.embed(&texts[1]));
    }
}
