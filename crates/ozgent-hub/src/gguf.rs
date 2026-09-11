//! Reading a GGUF file's metadata from its first bytes, without the rest.
//!
//! What a model needs in memory beyond its weights is its KV cache, and the
//! cache's size per token is written in the header: layers, key/value heads,
//! and key/value widths. Hugging Face serves byte ranges, so those few
//! kilobytes can be read before anyone commits to downloading gigabytes —
//! which is how the Models page can say "Q4_K_M: 64k context fits" rather
//! than finding out after the download, as other runners do.
//!
//! Only the header is parsed, and only as far as needed: the metadata comes
//! in file order, and the architecture keys come before the tokenizer's large
//! vocabulary arrays, so reading stops as soon as everything wanted is in.

use std::collections::HashMap;

/// What the header says that decides memory.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Shape {
    pub architecture: String,
    pub layers: u32,
    /// Key/value heads. Some architectures store one per layer; llama.cpp
    /// sizes by the first, and so does this, so the estimate matches the load.
    pub kv_heads: u32,
    pub key_length: u32,
    pub value_length: u32,
    /// The context it was trained for.
    pub context_train: u32,
}

impl Shape {
    /// KV cache elements per token, the way the engine counts them.
    pub fn kv_elements_per_token(&self) -> u64 {
        ozgent_core::accel::kv_elements_per_token(self.layers, self.kv_heads, self.key_length, self.value_length)
    }
}

/// Why a header could not be read.
#[derive(Debug, PartialEq)]
pub enum ShapeError {
    /// More bytes are needed; try again with at least this many.
    Short(usize),
    NotGguf,
    Malformed(String),
}

#[derive(Debug, Clone)]
enum Value {
    Uint(u64),
    Int(i64),
    Str(String),
    /// Only integer arrays are kept; they are small. Others are skipped.
    Ints(Vec<i64>),
    Other,
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ShapeError> {
        let end = self.at.checked_add(n).ok_or_else(|| ShapeError::Malformed("length overflows".into()))?;
        if end > self.buf.len() {
            // Ask for comfortably more, so a second fetch is rarely followed
            // by a third.
            return Err(ShapeError::Short((end * 2).max(self.buf.len() * 2)));
        }
        let s = &self.buf[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, ShapeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ShapeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, ShapeError> {
        let n = self.u64()?;
        if n > 1 << 30 {
            return Err(ShapeError::Malformed("a string longer than a gigabyte".into()));
        }
        Ok(String::from_utf8_lossy(self.take(n as usize)?).into_owned())
    }
    /// One value of GGUF type `t`.
    fn value(&mut self, t: u32) -> Result<Value, ShapeError> {
        Ok(match t {
            0 => Value::Uint(self.take(1)?[0] as u64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Uint(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Uint(self.u32()? as u64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => {
                self.take(4)?;
                Value::Other
            }
            7 => {
                self.take(1)?;
                Value::Other
            }
            8 => Value::Str(self.string()?),
            9 => {
                let inner = self.u32()?;
                let n = self.u64()?;
                if n > 1 << 28 {
                    return Err(ShapeError::Malformed("an array with too many elements".into()));
                }
                let mut ints = Vec::new();
                for _ in 0..n {
                    match self.value(inner)? {
                        Value::Uint(u) => ints.push(u as i64),
                        Value::Int(i) => ints.push(i),
                        _ => {}
                    }
                }
                if matches!(inner, 0..=5 | 10 | 11) { Value::Ints(ints) } else { Value::Other }
            }
            10 => Value::Uint(self.u64()?),
            11 => Value::Int(self.u64()? as i64),
            12 => {
                self.take(8)?;
                Value::Other
            }
            t => return Err(ShapeError::Malformed(format!("unknown value type {t}"))),
        })
    }
}

/// Read the memory-deciding keys from the start of a GGUF file.
pub fn read_shape(buf: &[u8]) -> Result<Shape, ShapeError> {
    let mut r = Reader { buf, at: 0 };
    if r.take(4)? != b"GGUF" {
        return Err(ShapeError::NotGguf);
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(ShapeError::Malformed(format!("GGUF version {version}")));
    }
    let _tensors = r.u64()?;
    let kv_count = r.u64()?;

    let mut kv: HashMap<String, Value> = HashMap::new();
    let mut arch = String::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        // The vocabulary comes after the architecture keys and is megabytes
        // of strings; once the essentials are in, it is never read.
        if key.starts_with("tokenizer.") && !arch.is_empty() && has(&kv, &arch, &ESSENTIAL) {
            break;
        }
        let t = r.u32()?;
        let value = r.value(t)?;
        if key == "general.architecture" {
            if let Value::Str(s) = &value {
                arch = s.clone();
            }
        }
        if key.starts_with("general.") || (!arch.is_empty() && key.starts_with(&format!("{arch}."))) {
            kv.insert(key, value);
        }
        if !arch.is_empty() && has(&kv, &arch, &ESSENTIAL) && has(&kv, &arch, &WIDTHS) {
            break;
        }
    }
    if arch.is_empty() {
        return Err(ShapeError::Malformed("no general.architecture".into()));
    }
    let uint = |k: &str| -> Option<u32> {
        match kv.get(&format!("{arch}.{k}"))? {
            Value::Uint(u) => u32::try_from(*u).ok(),
            Value::Int(i) => u32::try_from(*i).ok(),
            Value::Ints(v) => v.first().and_then(|i| u32::try_from(*i).ok()),
            _ => None,
        }
    };
    let layers = uint("block_count").ok_or_else(|| ShapeError::Malformed("no block_count".into()))?;
    let heads = uint("attention.head_count").unwrap_or(0);
    let kv_heads = uint("attention.head_count_kv").unwrap_or(heads);
    let head_dim = uint("embedding_length").zip(Some(heads)).map(|(e, h)| e / h.max(1)).unwrap_or(0);
    Ok(Shape {
        layers,
        kv_heads,
        key_length: uint("attention.key_length").unwrap_or(head_dim),
        value_length: uint("attention.value_length").unwrap_or(head_dim),
        context_train: uint("context_length").unwrap_or(0),
        architecture: arch,
    })
}

/// Keys every usable header has.
const ESSENTIAL: [&str; 5] =
    ["block_count", "context_length", "embedding_length", "attention.head_count", "attention.head_count_kv"];
/// Keys some headers have; without them the width is derived, as llama.cpp does.
const WIDTHS: [&str; 2] = ["attention.key_length", "attention.value_length"];

fn has(kv: &HashMap<String, Value>, arch: &str, keys: &[&str]) -> bool {
    keys.iter().all(|k| kv.contains_key(&format!("{arch}.{k}")))
}

/// Fetch just enough of a remote GGUF to read its shape.
pub async fn fetch_shape(client: &crate::Client, repo: &str, revision: &str, path: &str) -> Result<Shape, crate::HubError> {
    let url = client.download_url(repo, revision, path);
    // The keys wanted sit in the first kilobyte or two of every header seen;
    // 64 KB leaves room for a long model card in `general.*` before them.
    let mut want: usize = 64 * 1024;
    for _ in 0..4 {
        let mut req = client.http().get(&url).header("Range", format!("bytes=0-{}", want - 1));
        if let Some(t) = client.token() {
            req = req.bearer_auth(t);
        }
        let res = req
            .timeout(std::time::Duration::from_secs(20))
            .send()
            .await?
            .error_for_status()?;
        let bytes = res.bytes().await?;
        match read_shape(&bytes) {
            Ok(shape) => return Ok(shape),
            Err(ShapeError::Short(more)) if bytes.len() >= want => want = more.min(16 << 20),
            Err(ShapeError::Short(_)) => {
                return Err(crate::HubError::Other(format!("{path} ended inside its header")));
            }
            Err(ShapeError::NotGguf) => return Err(crate::HubError::Other(format!("{path} is not a GGUF file"))),
            Err(ShapeError::Malformed(m)) => return Err(crate::HubError::Other(format!("{path}: {m}"))),
        }
    }
    Err(crate::HubError::Other(format!("{path}: the header is larger than 16 MB")))
}

/// How much context fits on a GPU with `vram` bytes beside `weights` bytes of
/// model, choosing the KV cache type the way the engine does at load.
///
/// `None` when the weights alone do not fit — then some layers run on the CPU
/// and the window is bounded by system memory instead.
pub fn context_that_fits(shape: &Shape, weights: u64, vram: u64) -> Option<(u32, ozgent_core::accel::CacheType)> {
    // What the desktop and llama.cpp's own buffers take, before any model.
    let usable = vram / 100 * 90;
    let free = usable.checked_sub(weights)?;
    let elements = shape.kv_elements_per_token();
    let target = if shape.context_train > 0 { shape.context_train } else { 32_768 };
    let kind = ozgent_core::accel::choose_kv_type(elements, target, weights, free, true);
    let budget = ozgent_core::accel::kv_budget(free, 0, 1, 1);
    let ctx = ozgent_core::accel::fit_context(elements, target, kind, budget);
    Some((ctx, kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv_str(out: &mut Vec<u8>, key: &str, val: &str) {
        out.extend((key.len() as u64).to_le_bytes());
        out.extend(key.as_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend((val.len() as u64).to_le_bytes());
        out.extend(val.as_bytes());
    }
    fn kv_u32(out: &mut Vec<u8>, key: &str, val: u32) {
        out.extend((key.len() as u64).to_le_bytes());
        out.extend(key.as_bytes());
        out.extend(4u32.to_le_bytes());
        out.extend(val.to_le_bytes());
    }

    /// A header shaped like Qwen3-8B's: 36 layers, 8 KV heads of 128.
    fn header() -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(9u64.to_le_bytes());
        kv_str(&mut b, "general.architecture", "qwen3");
        kv_u32(&mut b, "qwen3.block_count", 36);
        kv_u32(&mut b, "qwen3.context_length", 40_960);
        kv_u32(&mut b, "qwen3.embedding_length", 4096);
        kv_u32(&mut b, "qwen3.attention.head_count", 32);
        kv_u32(&mut b, "qwen3.attention.head_count_kv", 8);
        kv_u32(&mut b, "qwen3.attention.key_length", 128);
        kv_u32(&mut b, "qwen3.attention.value_length", 128);
        // A vocabulary array, which must not need to be read.
        let key = "tokenizer.ggml.tokens";
        b.extend((key.len() as u64).to_le_bytes());
        b.extend(key.as_bytes());
        b.extend(9u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        b.extend(1_000_000u64.to_le_bytes());
        b
    }

    #[test]
    fn the_shape_is_read_before_the_vocabulary() {
        let s = read_shape(&header()).unwrap();
        assert_eq!(s.architecture, "qwen3");
        assert_eq!((s.layers, s.kv_heads, s.key_length, s.value_length), (36, 8, 128, 128));
        assert_eq!(s.context_train, 40_960);
        assert_eq!(s.kv_elements_per_token(), 36 * 8 * 256);
    }

    #[test]
    fn a_cut_off_header_asks_for_more() {
        let h = header();
        assert!(matches!(read_shape(&h[..40]), Err(ShapeError::Short(_))));
        assert_eq!(read_shape(b"PK\x03\x04rest"), Err(ShapeError::NotGguf));
    }

    #[test]
    fn a_bigger_file_leaves_less_room_for_context() {
        let s = read_shape(&header()).unwrap();
        let gib = 1u64 << 30;
        let (small, _) = context_that_fits(&s, 5 * gib, 8 * gib).unwrap();
        let (large, _) = context_that_fits(&s, 7 * gib, 8 * gib).unwrap_or((0, ozgent_core::accel::CacheType::F16));
        assert!(small > large, "{small} vs {large}");
        assert!(small <= 40_960, "never more than it was trained for");
        assert!(context_that_fits(&s, 9 * gib, 8 * gib).is_none(), "the weights alone do not fit");
    }
}
