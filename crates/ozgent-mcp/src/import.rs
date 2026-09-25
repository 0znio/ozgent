//! Reading the MCP settings other clients use.
//!
//! Almost every MCP server documents itself as a block of JSON for Claude
//! Desktop, VS Code or Cursor:
//!
//! ```json
//! { "mcpServers": { "files": { "command": "npx", "args": ["-y", "…"], "env": { … } } } }
//! ```
//!
//! This reads those so the block can be pasted as it is. The layouts differ
//! only a little — `mcpServers` (Claude Desktop, Cursor, Windsurf), `servers`
//! (VS Code), a `type` field or not — and VS Code's file allows comments and
//! trailing commas. Anything that cannot be carried over is reported by
//! entry, never dropped quietly: a server that half-arrives is worse than one
//! that says why it did not.

use std::collections::BTreeMap;

use ozgent_core::mcp::{self, Server};
use serde_json::Value;

/// One server read from a pasted configuration.
#[derive(Debug, Clone)]
pub struct Imported {
    pub name: String,
    /// The entry, or why it cannot be used.
    pub server: Result<Server, String>,
    /// Things it said that ozgent does not use, for the person to know.
    pub notes: Vec<String>,
}

/// Read a pasted configuration: the whole file, the servers map, or one
/// server's object (named `fallback_name`).
pub fn parse(text: &str, fallback_name: Option<&str>) -> Result<Vec<Imported>, String> {
    let cleaned = strip_jsonc(text);
    let value: Value = serde_json::from_str(cleaned.trim())
        .map_err(|e| format!("that is not JSON ({e}); paste the block exactly as the server's page shows it"))?;
    let Value::Object(top) = value else {
        return Err("expected a JSON object like {\"mcpServers\": {…}}".into());
    };

    // Where the servers are: under a known key, a single server, or the map
    // itself.
    let servers: serde_json::Map<String, Value> = if let Some(map) = ["mcpServers", "servers", "mcp_servers"]
        .iter()
        .find_map(|k| top.get(*k).and_then(Value::as_object))
    {
        map.clone()
    } else if let Some(map) = top.get("mcp").and_then(|m| m.get("servers")).and_then(Value::as_object) {
        map.clone()
    } else if looks_like_server(&Value::Object(top.clone())) {
        let name = fallback_name.map(str::to_string).filter(|n| !n.trim().is_empty()).ok_or(
            "this is one server without a name around it; give it a name, or paste it as {\"name\": {…}}",
        )?;
        let mut one = serde_json::Map::new();
        one.insert(name, Value::Object(top));
        one
    } else if !top.is_empty() && top.values().all(looks_like_server) {
        top
    } else {
        return Err("no servers found: expected \"mcpServers\" or \"servers\" with one entry per server".into());
    };

    if servers.is_empty() {
        return Err("the configuration lists no servers".into());
    }
    Ok(servers.into_iter().map(|(name, entry)| entry_to_server(&name, &entry)).collect())
}

fn looks_like_server(v: &Value) -> bool {
    v.get("command").is_some() || v.get("url").is_some() || v.get("serverUrl").is_some()
}

fn strings(v: Option<&Value>, what: &str) -> Result<Vec<String>, String> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(items)) => items
            .iter()
            .map(|x| match x {
                Value::String(s) => Ok(s.clone()),
                Value::Number(n) => Ok(n.to_string()),
                Value::Bool(b) => Ok(b.to_string()),
                _ => Err(format!("`{what}` must be a list of strings")),
            })
            .collect(),
        Some(_) => Err(format!("`{what}` must be a list of strings")),
    }
}

fn map(v: Option<&Value>, what: &str) -> Result<BTreeMap<String, String>, String> {
    match v {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(Value::Object(m)) => m
            .iter()
            .map(|(k, v)| match v {
                Value::String(s) => Ok((k.clone(), s.clone())),
                Value::Number(n) => Ok((k.clone(), n.to_string())),
                Value::Bool(b) => Ok((k.clone(), b.to_string())),
                _ => Err(format!("`{what}.{k}` must be a string")),
            })
            .collect(),
        Some(_) => Err(format!("`{what}` must be an object of names and values")),
    }
}

/// A value that refers to something only the other client can fill in.
fn placeholder(s: &str) -> Option<&'static str> {
    if s.contains("${input:") {
        Some("uses a VS Code input (`${input:…}`); put the value itself in instead")
    } else if s.contains("${env:") || s.contains("${workspaceFolder") || s.contains("${userHome") {
        Some("uses a variable only VS Code fills in (`${…}`); put the value itself in instead")
    } else {
        None
    }
}

fn entry_to_server(name: &str, entry: &Value) -> Imported {
    let mut notes = Vec::new();
    let server = (|| -> Result<Server, String> {
        mcp::valid_name(name).map_err(|e| format!("{e} (rename it in the JSON)"))?;
        let obj = entry.as_object().ok_or("each server must be an object")?;
        let kind = obj.get("type").or_else(|| obj.get("transport")).and_then(Value::as_str).unwrap_or("");
        let url = obj.get("url").or_else(|| obj.get("serverUrl")).and_then(Value::as_str);
        let command = obj.get("command").and_then(Value::as_str);

        let mut server = Server {
            enabled: !obj.get("disabled").and_then(Value::as_bool).unwrap_or(false),
            ..Default::default()
        };
        if !server.enabled {
            notes.push("marked disabled; added switched off".into());
        }
        match (command, url) {
            (Some(command), None) => {
                if !kind.is_empty() && kind != "stdio" {
                    return Err(format!("says type {kind:?} but gives a command"));
                }
                server.command = Some(command.to_string());
                server.args = strings(obj.get("args"), "args")?;
                server.env = map(obj.get("env"), "env")?;
                if let Some(cwd) = obj.get("cwd").and_then(Value::as_str) {
                    server.cwd = Some(cwd.into());
                }
            }
            (None, Some(url)) => {
                if kind == "sse" {
                    return Err("uses the older SSE transport, which ozgent does not speak; \
                                use the server's Streamable HTTP address if it has one"
                        .into());
                }
                if url.ends_with("/sse") && kind.is_empty() {
                    notes.push("its address ends in /sse, the older transport; if it fails to connect, look for its /mcp address".into());
                }
                let local = ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|p| url.starts_with(p));
                if !url.starts_with("https://") && !local {
                    return Err("a remote server's URL must be https (http only for this machine)".into());
                }
                server.url = Some(url.to_string());
                server.headers = map(obj.get("headers"), "headers")?;
            }
            (Some(_), Some(_)) => return Err("gives both a command and a URL".into()),
            (None, None) => return Err("gives neither a command nor a URL".into()),
        }

        for value in server.args.iter().chain(server.env.values()).chain(server.headers.values()) {
            if let Some(why) = placeholder(value) {
                return Err(why.into());
            }
            if value.contains(['\0', '\n', '\r']) {
                return Err("a value has a line break in it".into());
            }
        }
        for (key, _) in server.env.iter().chain(server.headers.iter()) {
            if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                return Err(format!("{key:?} is not a valid variable or header name"));
            }
        }
        for ignored in ["envFile", "autoApprove", "alwaysAllow", "oauth", "auth"] {
            if obj.contains_key(ignored) {
                notes.push(format!("`{ignored}` is not used by ozgent"));
            }
        }
        if let Some(t) = obj.get("timeout").and_then(Value::as_u64) {
            // Claude Desktop and others give milliseconds.
            server.timeout_seconds = if t > 3600 { (t / 1000).clamp(1, 3600) } else { t.clamp(1, 3600) };
        }
        server.source = Some("pasted JSON".into());
        server.transport().map_err(|e| e.to_string())?;
        Ok(server)
    })();
    Imported { name: name.to_string(), server, notes }
}

/// JSON with comments and trailing commas — VS Code's `mcp.json` — made into
/// JSON, leaving strings alone.
pub fn strip_jsonc(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let (mut i, mut in_string, mut escaped) = (0, false, false);
    while i < chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
                i += 1;
            }
            '/' if chars.get(i + 1) == Some(&'/') => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '/' if chars.get(i + 1) == Some(&'*') => {
                i += 2;
                while i + 1 < chars.len() && !(chars[i] == '*' && chars[i + 1] == '/') {
                    i += 1;
                }
                i += 2;
            }
            ',' => {
                // A trailing comma: the next thing that is not space closes.
                let mut j = i + 1;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                if !matches!(chars.get(j), Some('}') | Some(']')) {
                    out.push(c);
                }
                i += 1;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vs_code_layout_with_a_local_browser_service() {
        // The block as camofox-mcp's page gives it.
        let text = r#"{
          "servers": {
            "camofox": {
              "type": "stdio",
              "command": "npx",
              "args": ["-y", "camofox-mcp@latest"],
              "env": { "CAMOFOX_URL": "http://localhost:9377" }
            }
          }
        }"#;
        let got = parse(text, None).unwrap();
        assert_eq!(got.len(), 1);
        let s = got[0].server.as_ref().unwrap();
        assert_eq!(got[0].name, "camofox");
        assert_eq!(s.command.as_deref(), Some("npx"));
        assert_eq!(s.args, ["-y", "camofox-mcp@latest"]);
        assert_eq!(s.env["CAMOFOX_URL"], "http://localhost:9377");
    }

    #[test]
    fn claude_desktop_layout_with_comments_and_trailing_commas() {
        let text = r#"{
          // what the docs paste
          "mcpServers": {
            "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp",], },
            /* a remote one */
            "docs": { "url": "https://example.com/mcp", "headers": { "Authorization": "Bearer x" } },
          },
        }"#;
        let got = parse(text, None).unwrap();
        let mut names: Vec<&str> = got.iter().map(|g| g.name.as_str()).collect();
        names.sort();
        assert_eq!(names, ["docs", "files"]);
        assert!(got.iter().all(|g| g.server.is_ok()), "{got:?}");
        let docs = got.iter().find(|g| g.name == "docs").unwrap();
        assert_eq!(docs.server.as_ref().unwrap().headers["Authorization"], "Bearer x");
    }

    #[test]
    fn what_cannot_come_over_says_why_and_the_rest_still_does() {
        let text = r#"{ "mcpServers": {
            "old": { "type": "sse", "url": "https://x.test/sse" },
            "vscode": { "command": "npx", "env": { "TOKEN": "${input:token}" } },
            "plain": { "url": "http://example.com/mcp" },
            "bad name!": { "command": "x" },
            "good": { "command": "uvx", "args": ["mcp-server-time"], "disabled": true }
        } }"#;
        let got = parse(text, None).unwrap();
        let by = |n: &str| got.iter().find(|g| g.name == n).unwrap();
        assert!(by("old").server.as_ref().unwrap_err().contains("SSE"));
        assert!(by("vscode").server.as_ref().unwrap_err().contains("input"));
        assert!(by("plain").server.as_ref().unwrap_err().contains("https"));
        assert!(by("bad name!").server.is_err());
        let good = by("good").server.as_ref().unwrap();
        assert!(!good.enabled, "a disabled entry arrives switched off");
    }

    #[test]
    fn one_bare_server_needs_a_name() {
        let text = r#"{ "command": "npx", "args": ["-y", "pkg"] }"#;
        assert!(parse(text, None).is_err());
        let got = parse(text, Some("pkg")).unwrap();
        assert_eq!(got[0].name, "pkg");
    }

    #[test]
    fn strings_that_look_like_comments_are_kept() {
        assert_eq!(strip_jsonc(r#"{"u": "http://x//y", "a": [1,2,]}"#), r#"{"u": "http://x//y", "a": [1,2]}"#);
    }
}
