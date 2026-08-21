//! Incremental UTF-8 assembly for detokenised bytes.
//!
//! A token is a byte sequence, not a character. Multi-byte characters —
//! anything non-ASCII, and every emoji — are routinely split across two or
//! three tokens, so decoding each token independently produces replacement
//! characters. Bytes are therefore buffered and only released once they form
//! complete characters.

/// Accumulates bytes and yields complete UTF-8 as it becomes available.
#[derive(Debug, Default)]
pub struct Utf8Buffer {
    pending: Vec<u8>,
}

impl Utf8Buffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add bytes and take whatever is now decodable.
    ///
    /// At most three bytes are ever held back, since that is the longest
    /// incomplete prefix a UTF-8 character can have.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();

        // Loop, because dropping one invalid byte can expose more decodable
        // text behind it; a single pass would leave that text stuck.
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.pending[..good]));
                    match e.error_len() {
                        // Genuinely invalid: drop the offending byte, emit a
                        // replacement, and keep going.
                        Some(len) => {
                            self.pending.drain(..good + len);
                            out.push('\u{FFFD}');
                        }
                        // Truncated: the rest of the character has not arrived.
                        None => {
                            self.pending.drain(..good);
                            return out;
                        }
                    }
                }
            }
        }
    }

    /// Flush at end of stream, replacing any trailing incomplete character.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_passes_straight_through() {
        let mut b = Utf8Buffer::new();
        assert_eq!(b.push(b"hello"), "hello");
        assert!(b.is_empty());
    }

    #[test]
    fn a_character_split_across_pushes_is_reassembled() {
        // "日" is E6 97 A5; a token boundary can fall anywhere inside it.
        let mut b = Utf8Buffer::new();
        assert_eq!(b.push(&[0xE6]), "", "must not emit a partial character");
        assert_eq!(b.push(&[0x97]), "");
        assert_eq!(b.push(&[0xA5]), "日");
    }

    #[test]
    fn an_emoji_split_across_four_pushes_is_reassembled() {
        let bytes = "🎉".as_bytes().to_vec();
        let mut b = Utf8Buffer::new();
        let mut out = String::new();
        for byte in bytes {
            out.push_str(&b.push(&[byte]));
        }
        assert_eq!(out, "🎉");
    }

    #[test]
    fn complete_characters_are_released_while_a_partial_is_held() {
        let mut b = Utf8Buffer::new();
        let mut input = b"ok ".to_vec();
        input.push(0xE6); // start of a split character
        assert_eq!(b.push(&input), "ok ", "the ASCII prefix must not wait");
        assert!(!b.is_empty());
    }

    #[test]
    fn invalid_bytes_do_not_stall_the_stream() {
        let mut b = Utf8Buffer::new();
        let out = b.push(&[0xFF, b'a']);
        assert!(out.contains('a'), "a genuinely invalid byte must not block later text");
    }

    #[test]
    fn finish_flushes_a_trailing_partial() {
        let mut b = Utf8Buffer::new();
        assert_eq!(b.push(&[0xE6]), "");
        assert!(!b.finish().is_empty(), "truncated output should still surface");
        assert!(b.is_empty());
    }

    #[test]
    fn a_long_mixed_stream_round_trips_byte_by_byte() {
        let text = "Hello 世界 🎉 café — done.";
        let mut b = Utf8Buffer::new();
        let mut out = String::new();
        for byte in text.as_bytes() {
            out.push_str(&b.push(&[*byte]));
        }
        out.push_str(&b.finish());
        assert_eq!(out, text);
    }
}
