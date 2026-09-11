//! The admin password, and the random tokens that stand in for it.
//!
//! The password is stored as an Argon2id hash in PHC form
//! (`$argon2id$v=19$m=…,t=…,p=…$salt$hash`): salted per password, and
//! deliberately expensive in memory as well as time, so a copied config file
//! is not a password. The parameters are carried inside the string, so a hash
//! written by an older ozgent with different costs still verifies.

use rand_core::{OsRng, RngCore};
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::{Algorithm, Argon2, Params, Version};

/// Shortest admin password accepted.
pub const MIN_PASSWORD: usize = 8;

/// Argon2id with OWASP's first recommended setting: 19 MiB, two passes, one
/// lane. About 50 ms here — slow for someone guessing, unnoticeable for
/// someone logging in.
fn hasher() -> Argon2<'static> {
    let params = Params::new(19 * 1024, 2, 1, None).expect("valid Argon2 parameters");
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
}

/// Why a password was not accepted.
pub fn check_strength(password: &str) -> Result<(), String> {
    let n = password.chars().count();
    if n < MIN_PASSWORD {
        return Err(format!("use at least {MIN_PASSWORD} characters ({n} given)"));
    }
    if password.trim() != password {
        return Err("it starts or ends with a space, which is easy to mistype later".into());
    }
    Ok(())
}

/// Hash a password for storing.
pub fn hash_password(password: &str) -> Result<String, String> {
    check_strength(password)?;
    let salt = SaltString::generate(&mut OsRng);
    hasher()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| format!("hashing the password: {e}"))
}

/// Whether a stored value is a hash this module can check against.
///
/// Anything else — a password typed straight into `config.toml`, say — is not
/// quietly accepted as one: that would be a plain-text password pretending to
/// be protected.
pub fn is_hash(stored: &str) -> bool {
    PasswordHash::new(stored.trim()).is_ok_and(|h| h.algorithm.as_str() == "argon2id")
}

/// Whether `password` matches a stored hash. Constant-time in the comparison;
/// the cost is the hash itself.
pub fn verify_password(password: &str, stored: &str) -> bool {
    let Ok(parsed) = PasswordHash::new(stored.trim()) else { return false };
    if parsed.algorithm.as_str() != "argon2id" {
        return false;
    }
    // Verified with the parameters the hash carries, not today's defaults.
    Argon2::default().verify_password(password.as_bytes(), &parsed).is_ok()
}

/// A random token, as hex, from the operating system's generator.
pub fn random_token(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    OsRng.fill_bytes(&mut buf);
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

/// Compare two secrets without leaking where they first differ.
pub fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hash_verifies_its_password_and_nothing_else() {
        let h = hash_password("correct horse").unwrap();
        assert!(h.starts_with("$argon2id$v=19$m=19456,t=2,p=1$"), "{h}");
        assert!(is_hash(&h));
        assert!(verify_password("correct horse", &h));
        assert!(!verify_password("correct horse ", &h));
        assert!(!verify_password("Correct horse", &h));
    }

    #[test]
    fn the_same_password_hashes_differently_each_time() {
        // Salted: two operators with one password do not share a hash.
        assert_ne!(hash_password("password1").unwrap(), hash_password("password1").unwrap());
    }

    #[test]
    fn a_plain_password_in_the_file_is_not_a_hash() {
        assert!(!is_hash("hunter22"));
        assert!(!verify_password("hunter22", "hunter22"));
        assert!(!is_hash(""));
    }

    #[test]
    fn weak_passwords_are_refused() {
        assert!(hash_password("short").is_err());
        assert!(check_strength(" padded password").is_err());
        assert!(check_strength("long enough").is_ok());
    }

    #[test]
    fn tokens_are_random_hex() {
        let a = random_token(32);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, random_token(32));
        assert!(same(&a, &a.clone()));
        assert!(!same(&a, &random_token(32)));
        assert!(!same("ab", "abc"));
    }
}
