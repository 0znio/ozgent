//! `[web.access]`: who may reach the server, and what a key lets them do.
//!
//! Lives under `[web]` on purpose. The general settings endpoint neither shows
//! nor accepts `[web]` (see `ozgent-web`'s `public`), so nothing a browser tab,
//! an API caller or a tool can reach is able to widen these rules. They change
//! through the admin page, behind the admin password, or by editing the file.
//!
//! Three questions, answered in this order for every connection:
//!
//! 1. **May this address connect at all?** [`AccessConfig::admits`]: a deny
//!    list, and in `allowlist` mode an allow list, of IPv4 and IPv6 addresses
//!    and CIDR ranges. This machine is always admitted unless denied by name,
//!    so an operator cannot lock themselves out of their own server.
//! 2. **Who is it?** This machine's own page (the local token), a signed-in
//!    admin, or a caller with an API key. Decided in `ozgent-web::access`.
//! 3. **What may it do?** A key carries [`Scope`]s; see [`Grant`].

use serde::{Deserialize, Serialize};
use std::net::IpAddr;

/// Whether addresses are admitted by default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetMode {
    /// Everyone may connect except the addresses in `deny`.
    #[default]
    Open,
    /// Only the addresses in `allow` may connect (and this machine).
    Allowlist,
}

/// What an API key may be used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// The model: `/v1/chat/completions`, `/v1/messages`, `/v1/completions`,
    /// `/v1/embeddings`, `/v1/models`, with the caller's own tools. The caller
    /// runs those; nothing executes here.
    Inference,
    /// ozgent's own tools, run on this machine — only those whose effect is
    /// `read`, and only where the tool policy does not deny them.
    Tools,
    /// Also tools that change files or data. Lifts the question the policy
    /// would ask; never lifts a `deny`, nor any limit the tool itself enforces
    /// (its root, its protected directories).
    ToolsWrite,
    /// Also tools that run programs, on the same terms. Commands still run in
    /// the sandbox and only if `[tools.config.permissions] shell` allows them.
    ToolsExecute,
    /// ozgent's `@agents`.
    Agents,
}

impl Scope {
    pub const ALL: [Scope; 5] =
        [Scope::Inference, Scope::Tools, Scope::ToolsWrite, Scope::ToolsExecute, Scope::Agents];

    pub fn describe(self) -> &'static str {
        match self {
            Scope::Inference => "use the models, with your own tools",
            Scope::Tools => "use ozgent's read-only tools",
            Scope::ToolsWrite => "use tools that change files or data",
            Scope::ToolsExecute => "use tools that run programs",
            Scope::Agents => "use ozgent's agents",
        }
    }
}

/// One API key. The key itself is never stored: only its SHA-256, which is
/// enough for a 256-bit random secret and cheap enough to check on every
/// request (an Argon2 hash would add 50 ms to each one).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKeyEntry {
    /// Public identifier, shown in lists and logs: `ozk_` and 8 hex digits,
    /// which are also the key's first characters.
    pub id: String,
    /// What the operator called it.
    pub name: String,
    /// Hex SHA-256 of the whole key.
    pub hash: String,
    pub scopes: Vec<Scope>,
    /// Unix seconds.
    #[serde(default)]
    pub created: i64,
    /// A disabled key is refused but kept, so its name still explains old logs.
    #[serde(default)]
    pub disabled: bool,
}

/// `[web.access]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AccessConfig {
    pub mode: NetMode,
    /// Addresses or CIDR ranges, IPv4 or IPv6: `192.168.1.0/24`, `2001:db8::/32`,
    /// `203.0.113.7`. Consulted only in `allowlist` mode.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
    /// Always refused, in either mode, and before anything else.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub deny: Vec<String>,
    /// Reverse proxies whose `X-Forwarded-For` is believed. Anyone else's is
    /// ignored: the header is whatever the sender wants it to be.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub trusted_proxies: Vec<String>,
    /// Host names this server answers to beyond `localhost` and bare IP
    /// addresses — a LAN name, or the name a reverse proxy forwards. A request
    /// naming any other host is refused, which is what stops a web page from
    /// reaching this server by pointing its own domain at 127.0.0.1.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hosts: Vec<String>,
    /// `/v1` from this machine without a key. On by default so local programs
    /// keep working; it still requires a valid `Host` and refuses browsers
    /// calling cross-site. Off: every API caller needs a key.
    pub local_api_open: bool,
    /// Requests per minute from one address. `0` is unlimited.
    pub requests_per_minute: u32,
    /// Open connections from one address at once. `0` is unlimited.
    pub max_connections_per_address: u32,
    /// Largest request body, in MiB, except model uploads, which stream.
    pub max_body_mb: u32,
    /// Failed keys or tokens from one address before it is refused for
    /// `lockout_minutes`.
    pub max_auth_failures: u32,
    pub lockout_minutes: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<ApiKeyEntry>,
}

impl Default for AccessConfig {
    fn default() -> Self {
        Self {
            mode: NetMode::Open,
            allow: Vec::new(),
            deny: Vec::new(),
            trusted_proxies: Vec::new(),
            hosts: Vec::new(),
            local_api_open: true,
            requests_per_minute: 600,
            max_connections_per_address: 64,
            max_body_mb: 48,
            max_auth_failures: 10,
            lockout_minutes: 15,
            keys: Vec::new(),
        }
    }
}

/// Why an address was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    Denied(String),
    NotAllowed,
}

impl AccessConfig {
    /// Whether `ip` may connect.
    pub fn admits(&self, ip: IpAddr) -> Result<(), Refusal> {
        let ip = canonical(ip);
        if let Some(rule) = self.deny.iter().find(|r| rule_matches(r, ip)) {
            return Err(Refusal::Denied(rule.clone()));
        }
        match self.mode {
            NetMode::Open => Ok(()),
            // This machine is always let in: an allowlist that forgot it would
            // lock the operator out of the page that fixes it.
            NetMode::Allowlist if ip.is_loopback() => Ok(()),
            NetMode::Allowlist if self.allow.iter().any(|r| rule_matches(r, ip)) => Ok(()),
            NetMode::Allowlist => Err(Refusal::NotAllowed),
        }
    }

    /// Whether `ip` is a proxy whose forwarding header is believed.
    pub fn trusts_proxy(&self, ip: IpAddr) -> bool {
        let ip = canonical(ip);
        self.trusted_proxies.iter().any(|r| rule_matches(r, ip))
    }

    /// Every rule that does not parse, with the reason. The admin page refuses
    /// to save a list that has any, rather than silently ignoring a typo in a
    /// deny list — which would be a rule that looks like it protects something.
    pub fn invalid_rules(&self) -> Vec<String> {
        self.allow
            .iter()
            .chain(&self.deny)
            .chain(&self.trusted_proxies)
            .filter_map(|r| parse_rule(r).err().map(|e| format!("{r:?}: {e}")))
            .collect()
    }

    /// The key whose digest is `digest`, if it is enabled.
    pub fn key_by_digest(&self, digest: &str) -> Option<&ApiKeyEntry> {
        self.keys.iter().find(|k| !k.disabled && crate::secret::same(&k.hash, digest))
    }
}

/// An IPv4 address carried in IPv6 (`::ffff:1.2.3.4`) is that IPv4 address.
/// Without this, a dual-stack listener would see an IPv4 client in IPv6 form
/// and an IPv4 deny rule would never match it.
pub fn canonical(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(IpAddr::V6(v6)),
        v4 => v4,
    }
}

/// A parsed rule: an address and how many leading bits must match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Net {
    pub addr: IpAddr,
    pub prefix: u8,
}

/// Parse `1.2.3.4`, `1.2.3.0/24`, `2001:db8::1`, `2001:db8::/32`, `[::1]`.
pub fn parse_rule(rule: &str) -> Result<Net, String> {
    let rule = rule.trim();
    let (addr, prefix) = match rule.split_once('/') {
        Some((a, p)) => (a, Some(p)),
        None => (rule, None),
    };
    let addr = addr.trim_start_matches('[').trim_end_matches(']');
    let addr: IpAddr = addr.parse().map_err(|_| "not an IPv4 or IPv6 address".to_string())?;
    let addr = canonical(addr);
    let max = if addr.is_ipv4() { 32 } else { 128 };
    let prefix = match prefix {
        None => max,
        Some(p) => {
            let n: u8 = p.parse().map_err(|_| format!("{p:?} is not a prefix length"))?;
            if n > max {
                return Err(format!("/{n} is longer than an address ({max} bits)"));
            }
            n
        }
    };
    Ok(Net { addr, prefix })
}

impl Net {
    pub fn contains(&self, ip: IpAddr) -> bool {
        match (self.addr, canonical(ip)) {
            (IpAddr::V4(net), IpAddr::V4(ip)) => {
                let mask = if self.prefix == 0 { 0 } else { u32::MAX << (32 - self.prefix) };
                (u32::from(net) & mask) == (u32::from(ip) & mask)
            }
            (IpAddr::V6(net), IpAddr::V6(ip)) => {
                let mask = if self.prefix == 0 { 0 } else { u128::MAX << (128 - self.prefix) };
                (u128::from(net) & mask) == (u128::from(ip) & mask)
            }
            _ => false,
        }
    }
}

fn rule_matches(rule: &str, ip: IpAddr) -> bool {
    parse_rule(rule).is_ok_and(|net| net.contains(ip))
}

/// What an API caller may do, from the scopes of the key it presented.
///
/// `None` in a request means the caller is this machine's owner, who is
/// governed by the tool policy alone, exactly as before keys existed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Grant {
    pub key: String,
    pub inference: bool,
    pub tools: bool,
    pub write: bool,
    pub execute: bool,
    pub agents: bool,
}

impl Grant {
    pub fn from_scopes(key: &str, scopes: &[Scope]) -> Self {
        let has = |s| scopes.contains(&s);
        Self {
            key: key.to_string(),
            inference: has(Scope::Inference),
            tools: has(Scope::Tools) || has(Scope::ToolsWrite) || has(Scope::ToolsExecute),
            write: has(Scope::ToolsWrite),
            execute: has(Scope::ToolsExecute),
            agents: has(Scope::Agents),
        }
    }

    /// Everything, for the single key `ozgent serve --api-key` is started with,
    /// which has always meant "this caller may do what the server can".
    pub fn full(key: &str) -> Self {
        Self::from_scopes(key, &Scope::ALL)
    }

    /// Whether a tool with this effect may run for this caller at all.
    pub fn permits(&self, effect: crate::permission::Effect) -> bool {
        use crate::permission::Effect;
        match effect {
            Effect::Read => self.tools,
            Effect::Write => self.write,
            Effect::Execute => self.execute,
            // A tool that will not say what it does is treated as the worst
            // thing it could be.
            Effect::Unknown => self.write && self.execute,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn rules_match_both_families_and_mapped_addresses() {
        let net = parse_rule("192.168.1.0/24").unwrap();
        assert!(net.contains(ip("192.168.1.77")));
        assert!(!net.contains(ip("192.168.2.1")));
        // A dual-stack socket reports an IPv4 client like this.
        assert!(net.contains(ip("::ffff:192.168.1.77")));
        let v6 = parse_rule("2001:db8::/32").unwrap();
        assert!(v6.contains(ip("2001:db8:1::5")));
        assert!(!v6.contains(ip("2001:db9::5")));
        assert!(!v6.contains(ip("192.168.1.1")));
        assert!(parse_rule("[::1]").unwrap().contains(ip("::1")));
        assert!(parse_rule("0.0.0.0/0").unwrap().contains(ip("8.8.8.8")));
        assert!(parse_rule("::/0").unwrap().contains(ip("2606:4700::1111")));
    }

    #[test]
    fn bad_rules_are_named_not_ignored() {
        for bad in ["1.2.3.4/33", "::/129", "example.com", "1.2.3", "1.2.3.4/x", ""] {
            assert!(parse_rule(bad).is_err(), "{bad}");
        }
        let c = AccessConfig { deny: vec!["10.0.0.0/8".into(), "nope".into()], ..Default::default() };
        assert_eq!(c.invalid_rules().len(), 1);
    }

    #[test]
    fn deny_wins_and_open_admits_the_rest() {
        let c = AccessConfig { deny: vec!["203.0.113.0/24".into(), "2001:db8:bad::/48".into()], ..Default::default() };
        assert!(c.admits(ip("203.0.113.9")).is_err());
        assert!(c.admits(ip("::ffff:203.0.113.9")).is_err());
        assert!(c.admits(ip("2001:db8:bad:1::1")).is_err());
        assert!(c.admits(ip("198.51.100.1")).is_ok());
    }

    #[test]
    fn an_allowlist_admits_only_its_ranges_and_this_machine() {
        let c = AccessConfig {
            mode: NetMode::Allowlist,
            allow: vec!["192.168.1.0/24".into()],
            deny: vec!["192.168.1.66".into()],
            ..Default::default()
        };
        assert!(c.admits(ip("192.168.1.5")).is_ok());
        assert!(c.admits(ip("192.168.1.66")).is_err(), "deny beats allow");
        assert_eq!(c.admits(ip("10.0.0.1")), Err(Refusal::NotAllowed));
        assert!(c.admits(ip("127.0.0.1")).is_ok());
        assert!(c.admits(ip("::1")).is_ok());
        // Unless this machine is denied by name, which is a deliberate act.
        let locked = AccessConfig { deny: vec!["127.0.0.1".into()], ..c };
        assert!(locked.admits(ip("127.0.0.1")).is_err());
    }

    #[test]
    fn scopes_cap_effects() {
        use crate::permission::Effect;
        let g = Grant::from_scopes("k", &[Scope::Inference, Scope::Tools]);
        assert!(g.permits(Effect::Read));
        assert!(!g.permits(Effect::Write));
        assert!(!g.permits(Effect::Execute));
        assert!(!g.permits(Effect::Unknown));
        let all = Grant::full("k");
        assert!(all.permits(Effect::Unknown));
        // Write implies being able to use tools at all.
        assert!(Grant::from_scopes("k", &[Scope::ToolsWrite]).tools);
    }

    #[test]
    fn the_table_round_trips() {
        let c = AccessConfig {
            mode: NetMode::Allowlist,
            allow: vec!["10.0.0.0/8".into()],
            keys: vec![ApiKeyEntry {
                id: "ozk_12345678".into(),
                name: "script".into(),
                hash: "ab".repeat(32),
                scopes: vec![Scope::Inference, Scope::ToolsExecute],
                created: 1,
                disabled: false,
            }],
            ..Default::default()
        };
        let text = toml::to_string(&c).unwrap();
        assert!(text.contains("tools_execute"), "{text}");
        let back: AccessConfig = toml::from_str(&text).unwrap();
        assert_eq!(back, c);
    }
}
