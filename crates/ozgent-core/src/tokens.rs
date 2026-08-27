//! Token counts written the way people say them.
//!
//! Context lengths are powers of two with four or five zeros on the end, and
//! nobody reads `131072` at a glance. Every other tool in this space takes
//! `128k`, so ozgent does too — on the command line and in `config.toml`
//! alike, since a setting that only one of the two accepts is a trap.
//!
//! `k` and `m` are the binary multipliers, not the decimal ones: `8k` is
//! 8192, because that is the number the user meant when they typed it.

use serde::{Deserialize, Deserializer};

/// Parse a token count, accepting a plain number or a `k`/`m` suffix.
///
/// Case-insensitive, tolerant of `8kb`/`8K`, and accepts a fraction where it
/// lands on a whole number of tokens (`1.5k` is 1536). Rejects anything that
/// would silently become zero, since a zero context is not a thing a user
/// asks for and is a very confusing way to fail.
pub fn parse_count(raw: &str) -> Result<u32, String> {
    let text = raw.trim();
    if text.is_empty() {
        return Err("expected a token count, e.g. 8192, 8k, or 128k".to_string());
    }
    // Underscores group digits the way Rust literals do; some users type them.
    let text: String = text.chars().filter(|c| *c != '_').collect();
    let lower = text.to_ascii_lowercase();

    // Longest suffix first, so `kb` is not read as `k` with a trailing `b`.
    let (digits, multiplier) = ["kib", "mib", "kb", "mb", "k", "m"]
        .iter()
        .find_map(|suffix| {
            let head = lower.strip_suffix(suffix)?;
            let scale = if suffix.starts_with('m') { 1024 * 1024 } else { 1024 };
            Some((head, scale))
        })
        .unwrap_or((lower.as_str(), 1));

    let digits = digits.trim();
    let value: f64 = digits
        .parse()
        .map_err(|_| format!("{raw:?} is not a token count; try 8192, 8k, or 128k"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{raw:?} is not a token count"));
    }

    let scaled = value * multiplier as f64;
    // A fraction that does not land on a whole token is a typo, not a
    // rounding request: `1.3k` would be 1331.2, and quietly picking one of
    // the neighbours is worse than saying so.
    if scaled.fract() != 0.0 {
        return Err(format!("{raw:?} is not a whole number of tokens"));
    }
    if scaled > u32::MAX as f64 {
        return Err(format!("{raw:?} is larger than any model's context"));
    }
    let scaled = scaled as u32;
    if scaled == 0 && multiplier != 1 {
        return Err(format!("{raw:?} is zero tokens"));
    }
    Ok(scaled)
}

/// Serde adapter so `config.toml` accepts `8192` and `"8k"` alike.
///
/// Written against `Option<u32>` rather than a newtype on purpose: changing
/// the field's type would ripple through every layer of the options stack,
/// and each layer that forgot to follow would silently drop the setting.
pub fn deserialize_optional<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Written {
        Number(u64),
        Text(String),
    }

    let Some(written) = Option::<Written>::deserialize(deserializer)? else {
        return Ok(None);
    };
    match written {
        Written::Number(n) => u32::try_from(n)
            .map(Some)
            .map_err(|_| serde::de::Error::custom(format!("{n} is larger than any context"))),
        Written::Text(s) => parse_count(&s).map(Some).map_err(serde::de::Error::custom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_numbers_pass_through() {
        assert_eq!(parse_count("8192"), Ok(8192));
        assert_eq!(parse_count("0"), Ok(0), "zero is how max_tokens says unbounded");
    }

    #[test]
    fn k_and_m_are_binary() {
        // The whole point: 8k must be the context length people actually set,
        // not 8000, which no model has.
        assert_eq!(parse_count("8k"), Ok(8192));
        assert_eq!(parse_count("32k"), Ok(32768));
        assert_eq!(parse_count("128k"), Ok(131_072));
        assert_eq!(parse_count("1m"), Ok(1_048_576));
    }

    #[test]
    fn case_and_common_suffixes_are_accepted() {
        for text in ["8K", "8k", "8kb", "8KB", "8kib", " 8k "] {
            assert_eq!(parse_count(text), Ok(8192), "for {text:?}");
        }
    }

    #[test]
    fn underscores_group_digits() {
        assert_eq!(parse_count("131_072"), Ok(131_072));
    }

    #[test]
    fn a_fraction_is_allowed_only_when_it_is_whole() {
        assert_eq!(parse_count("1.5k"), Ok(1536));
        assert!(parse_count("1.3k").is_err(), "1331.2 tokens is not a number the user meant");
    }

    #[test]
    fn nonsense_is_rejected_rather_than_read_as_zero() {
        for text in ["", "  ", "lots", "8g", "-1", "k"] {
            assert!(parse_count(text).is_err(), "{text:?} must not parse");
        }
    }

    #[test]
    fn the_error_shows_what_to_type_instead() {
        let err = parse_count("huge").unwrap_err();
        assert!(err.contains("8k"), "{err}");
    }

    #[test]
    fn oversized_counts_are_refused() {
        assert!(parse_count("9999m").is_err());
    }

    #[test]
    fn config_accepts_both_spellings() {
        #[derive(Deserialize)]
        struct Holder {
            #[serde(default, deserialize_with = "deserialize_optional")]
            ctx: Option<u32>,
        }

        let number: Holder = toml::from_str("ctx = 8192").unwrap();
        let text: Holder = toml::from_str(r#"ctx = "8k""#).unwrap();
        let missing: Holder = toml::from_str("").unwrap();

        assert_eq!(number.ctx, Some(8192));
        assert_eq!(text.ctx, Some(8192), "the two spellings must agree");
        assert_eq!(missing.ctx, None);
    }

    #[test]
    fn a_bad_config_value_names_the_problem() {
        #[derive(Debug, Deserialize)]
        struct Holder {
            #[serde(default, deserialize_with = "deserialize_optional")]
            ctx: Option<u32>,
        }
        let err = toml::from_str::<Holder>(r#"ctx = "enormous""#).unwrap_err();
        assert!(err.to_string().contains("token count"), "{err}");
    }
}
