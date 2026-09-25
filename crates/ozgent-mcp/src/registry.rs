//! Installing servers: from the official MCP Registry, or a package by name.
//!
//! The registry (<https://registry.modelcontextprotocol.io>) lists servers as
//! `server.json` documents: npm or PyPI packages to run over stdio, container
//! images, and remote endpoints. This turns one of those into a
//! [`Server`](ozgent_core::mcp::Server) entry, and nothing else: whatever
//! runs is built here from what the registry says, never from a command line
//! a page sent, so a page can ask for "this server, with these values" but
//! cannot ask for a program of its own choosing.
//!
//! The registry is open — anyone can publish under a namespace they own — so
//! an entry is someone's claim about their own code. That is why installs
//! show the exact command first, link the source repository, and are
//! sandboxed by default.

use std::collections::BTreeMap;
use std::time::Duration;

use ozgent_core::mcp::Server;
use serde::Serialize;
use serde_json::Value;

pub const REGISTRY_URL: &str = "https://registry.modelcontextprotocol.io";

/// Longest value accepted for any one input.
const MAX_VALUE: usize = 4096;

/// One server in the registry, at one version.
#[derive(Debug, Clone, Serialize)]
pub struct Listing {
    pub name: String,
    pub title: Option<String>,
    pub description: String,
    pub version: String,
    pub repository: Option<String>,
    pub website: Option<String>,
    /// Ways to run it: each package, then each remote endpoint.
    pub options: Vec<Choice>,
}

/// One way to run a listed server.
#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    /// `npm`, `pypi`, `oci`, `remote`, or whatever else the registry says.
    pub kind: String,
    /// Package name, image, or URL.
    pub identifier: String,
    pub version: Option<String>,
    /// `stdio`, `streamable-http` or `sse`.
    pub transport: String,
    /// The program that runs it here: `npx`, `uvx`, `docker`. None for a
    /// remote endpoint.
    pub runner: Option<String>,
    pub supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why_not: Option<String>,
    pub inputs: Vec<Input>,
    /// Arguments that are fixed, not asked for, in order.
    #[serde(skip)]
    fixed: Vec<Arg>,
    #[serde(skip)]
    runtime_fixed: Vec<String>,
}

/// Something the person installing has to (or may) fill in.
#[derive(Debug, Clone, Serialize)]
pub struct Input {
    /// Unique within a choice: `env:NAME`, `header:NAME` or `arg:N`.
    pub key: String,
    /// What it is called: the variable, header or flag name.
    pub name: String,
    pub description: String,
    pub required: bool,
    pub secret: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub choices: Vec<String>,
}

/// A package argument, in the order the package takes them.
#[derive(Debug, Clone)]
enum Arg {
    /// Always passed, as given.
    Fixed { flag: Option<String>, value: String },
    /// Filled from the input with this key; left out when empty and optional.
    Asked { flag: Option<String>, key: String },
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(8))
        .timeout(Duration::from_secs(15))
        .user_agent(concat!("ozgent/", env!("CARGO_PKG_VERSION")))
        .build()
        .unwrap_or_default()
}

/// Search the registry. Returns a page of the latest versions, and the
/// cursor for the next page.
pub async fn search(query: &str, cursor: Option<&str>) -> Result<(Vec<Listing>, Option<String>), String> {
    let mut params = vec![("version", "latest".to_string()), ("limit", "30".to_string())];
    let query = query.trim();
    if !query.is_empty() {
        params.push(("search", query.chars().take(100).collect()));
    }
    if let Some(c) = cursor.filter(|c| !c.is_empty()) {
        params.push(("cursor", c.chars().take(300).collect()));
    }
    let url = format!("{REGISTRY_URL}/v0/servers");
    // Its search is sometimes slow: twenty seconds is not unusual.
    let body: Value = get(client().get(&url).query(&params).timeout(Duration::from_secs(30))).await?;
    let mut out = Vec::new();
    for entry in body["servers"].as_array().into_iter().flatten() {
        if !active(entry) {
            continue;
        }
        if let Some(listing) = parse(&entry["server"]) {
            out.push(listing);
        }
    }
    let next = body["metadata"]["nextCursor"].as_str().map(str::to_string);
    Ok((out, next))
}

/// One server at one version (`latest` for the newest).
pub async fn fetch(name: &str, version: &str) -> Result<Listing, String> {
    let mut url = reqwest::Url::parse(REGISTRY_URL).map_err(|e| e.to_string())?;
    // Each part as one path segment, so the `/` in a server's name is
    // encoded rather than read as a path separator.
    url.path_segments_mut()
        .map_err(|_| "bad registry URL".to_string())?
        .extend(["v0", "servers", name, "versions", version]);
    let body: Value = get(client().get(url)).await?;
    if !active(&body) {
        return Err(format!("{name} is marked deprecated or removed in the registry"));
    }
    parse(&body["server"]).ok_or_else(|| format!("the registry's entry for {name} did not parse"))
}

/// One GET, tried twice: the registry now and then answers with an empty
/// body, and a person searching should not see an error for a blip.
async fn get(request: reqwest::RequestBuilder) -> Result<Value, String> {
    let retry = request.try_clone();
    match get_once(request).await {
        // Not after a timeout: some searches are slow on the registry's side
        // every time, and trying again only doubles the wait.
        Err(e) if !e.contains("no such server") && !e.contains("in time") => match retry {
            Some(again) => {
                tokio::time::sleep(Duration::from_millis(400)).await;
                get_once(again).await
            }
            None => Err(e),
        },
        other => other,
    }
}

async fn get_once(request: reqwest::RequestBuilder) -> Result<Value, String> {
    let response = request.send().await.map_err(|e| {
        if e.is_timeout() {
            "the MCP registry did not answer in time. Its search is slow for some words: try a shorter or different one".to_string()
        } else {
            format!("could not reach the MCP registry: {e}")
        }
    })?;
    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err("the registry has no such server or version".into());
    }
    if !status.is_success() {
        return Err(format!("the MCP registry answered {status}"));
    }
    response.json().await.map_err(|e| format!("the registry's answer did not parse: {e}"))
}

fn active(entry: &Value) -> bool {
    let status = entry["_meta"]["io.modelcontextprotocol.registry/official"]["status"].as_str();
    status.is_none_or(|s| s == "active")
}

fn text(v: &Value) -> Option<String> {
    v.as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Read a `server.json` document.
pub fn parse(server: &Value) -> Option<Listing> {
    let name = text(&server["name"])?;
    let mut options = Vec::new();
    for package in server["packages"].as_array().into_iter().flatten() {
        options.push(package_choice(package));
    }
    for remote in server["remotes"].as_array().into_iter().flatten() {
        options.push(remote_choice(remote));
    }
    Some(Listing {
        name,
        title: text(&server["title"]),
        description: text(&server["description"]).unwrap_or_default(),
        version: text(&server["version"]).unwrap_or_else(|| "latest".into()),
        repository: text(&server["repository"]["url"]),
        website: text(&server["websiteUrl"]),
        options,
    })
}

fn input_from(v: &Value, key: String, name: String) -> Input {
    Input {
        key,
        name,
        description: text(&v["description"]).unwrap_or_default(),
        required: v["isRequired"].as_bool().unwrap_or(false),
        secret: v["isSecret"].as_bool().unwrap_or(false),
        default: text(&v["default"]),
        choices: v["choices"]
            .as_array()
            .map(|c| c.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
            .unwrap_or_default(),
    }
}

fn package_choice(p: &Value) -> Choice {
    let kind = text(&p["registryType"]).unwrap_or_else(|| "unknown".into());
    let identifier = text(&p["identifier"]).unwrap_or_default();
    let version = text(&p["version"]);
    let transport = text(&p["transport"]["type"]).unwrap_or_else(|| "stdio".into());
    let runner = match kind.as_str() {
        "npm" => Some("npx"),
        "pypi" => Some("uvx"),
        "oci" => Some("docker"),
        _ => None,
    };
    let mut why_not = None;
    if runner.is_none() {
        why_not = Some(format!("{kind} packages cannot be installed from ozgent"));
    } else if transport != "stdio" {
        why_not = Some(format!(
            "this package runs its own web server ({transport}); only packages that talk over stdio can be installed"
        ));
    } else if let Err(e) = check_identifier(&kind, &identifier) {
        why_not = Some(e);
    } else if let Some(v) = &version {
        if let Err(e) = check_version(v) {
            why_not = Some(e);
        }
    }

    let mut inputs = Vec::new();
    for env in p["environmentVariables"].as_array().into_iter().flatten() {
        let Some(name) = text(&env["name"]) else { continue };
        if let Some(value) = text(&env["value"]) {
            // A fixed value is still an input, prefilled: some entries put
            // the real value there, others a placeholder like `{token}`.
            let mut input = input_from(env, format!("env:{name}"), name);
            input.default = Some(value);
            inputs.push(input);
        } else {
            inputs.push(input_from(env, format!("env:{name}"), name.clone()));
        }
    }

    let mut fixed = Vec::new();
    for (i, arg) in p["packageArguments"].as_array().into_iter().flatten().enumerate() {
        let flag = match arg["type"].as_str() {
            Some("named") => match text(&arg["name"]) {
                Some(n) if n.starts_with('-') => Some(n),
                Some(n) => Some(format!("--{n}")),
                None => {
                    why_not.get_or_insert_with(|| "a named argument has no name".into());
                    continue;
                }
            },
            _ => None,
        };
        if let Some(flag) = &flag {
            if !valid_flag(flag) {
                why_not.get_or_insert_with(|| format!("the argument {flag:?} is not a plain flag"));
                continue;
            }
        }
        let templated = arg["variables"].is_object();
        match text(&arg["value"]) {
            Some(value) if !templated => fixed.push(Arg::Fixed { flag, value }),
            _ => {
                let key = format!("arg:{i}");
                let label = flag.clone().unwrap_or_else(|| {
                    text(&arg["valueHint"]).or_else(|| text(&arg["name"])).unwrap_or_else(|| format!("argument {}", i + 1))
                });
                let mut input = input_from(arg, key.clone(), label);
                if templated {
                    input.default = text(&arg["value"]);
                }
                inputs.push(input);
                fixed.push(Arg::Asked { flag, key });
            }
        }
    }

    let runtime_fixed: Vec<String> = p["runtimeArguments"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| text(&a["value"]))
        .filter(|v| valid_flag(v) || v.chars().all(|c| c.is_ascii_alphanumeric() || "-_=.".contains(c)))
        .collect();

    let supported = why_not.is_none();
    Choice {
        kind,
        identifier,
        version,
        transport,
        runner: runner.map(str::to_string),
        supported,
        why_not,
        inputs,
        fixed,
        runtime_fixed,
    }
}

fn remote_choice(r: &Value) -> Choice {
    let transport = text(&r["type"]).unwrap_or_default();
    let url = text(&r["url"]).unwrap_or_default();
    let why_not = if transport != "streamable-http" {
        Some(format!("this endpoint uses {transport}; ozgent speaks Streamable HTTP"))
    } else if url.contains('{') {
        Some("this endpoint's address has parts to fill in, which ozgent cannot do yet".into())
    } else if !url.starts_with("https://") {
        Some("only https endpoints can be installed".into())
    } else {
        None
    };
    let inputs = r["headers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|h| {
            let name = text(&h["name"])?;
            let mut input = input_from(h, format!("header:{name}"), name);
            if let Some(v) = text(&h["value"]) {
                input.default = Some(v);
            }
            Some(input)
        })
        .collect();
    Choice {
        kind: "remote".into(),
        identifier: url,
        version: None,
        transport,
        runner: None,
        supported: why_not.is_none(),
        why_not,
        inputs,
        fixed: Vec::new(),
        runtime_fixed: Vec::new(),
    }
}

fn valid_flag(flag: &str) -> bool {
    let bare = flag.trim_start_matches('-');
    flag.len() - bare.len() <= 2
        && !bare.is_empty()
        && bare.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
}

fn check_identifier(kind: &str, id: &str) -> Result<(), String> {
    let ok = match kind {
        // npm: optional @scope/, then a lowercase name.
        "npm" => {
            let (scope, name) = match id.strip_prefix('@').and_then(|s| s.split_once('/')) {
                Some((scope, name)) => (Some(scope), name),
                None => (None, id),
            };
            let part = |s: &str| {
                !s.is_empty()
                    && !s.starts_with(['.', '_', '-'])
                    && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "-._~".contains(c))
            };
            scope.is_none_or(part) && part(name) && id.len() <= 214
        }
        "pypi" => {
            !id.is_empty()
                && id.len() <= 100
                && id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
                && id.chars().all(|c| c.is_ascii_alphanumeric() || "-._".contains(c))
        }
        "oci" => {
            !id.is_empty()
                && id.len() <= 255
                && id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
                && id.chars().all(|c| c.is_ascii_alphanumeric() || "-._/:@".contains(c))
        }
        _ => false,
    };
    if ok { Ok(()) } else { Err(format!("{id:?} is not a valid {kind} package name")) }
}

fn check_version(v: &str) -> Result<(), String> {
    let ok = !v.is_empty()
        && v.len() <= 64
        && v.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && v.chars().all(|c| c.is_ascii_alphanumeric() || "-._+".contains(c));
    if ok { Ok(()) } else { Err(format!("{v:?} is not a version")) }
}

fn check_value(key: &str, value: &str) -> Result<(), String> {
    if value.len() > MAX_VALUE {
        return Err(format!("{key} is longer than {MAX_VALUE} characters"));
    }
    if value.contains(['\0', '\n', '\r']) {
        return Err(format!("{key} may not contain line breaks"));
    }
    Ok(())
}

/// Whether the program a choice needs is installed on this machine.
pub fn runner_present(choice: &Choice) -> bool {
    choice.runner.as_deref().is_none_or(|r| which(r).is_some())
}

/// Where `program` is on `PATH`.
pub fn which(program: &str) -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(program)).find(|p| p.is_file())
}

/// Build the server entry for one choice of a listing, from the values the
/// person filled in. Values for keys the choice does not ask for are refused,
/// as are missing required ones.
pub fn plan(listing: &Listing, choice: usize, values: &BTreeMap<String, String>) -> Result<Server, String> {
    let c = listing.options.get(choice).ok_or("no such way to run it")?;
    if !c.supported {
        return Err(c.why_not.clone().unwrap_or_else(|| "this cannot be installed".into()));
    }
    let known: std::collections::HashSet<&str> = c.inputs.iter().map(|i| i.key.as_str()).collect();
    if let Some(stray) = values.keys().find(|k| !known.contains(k.as_str())) {
        return Err(format!("{stray} is not something this server asks for"));
    }
    let mut filled: BTreeMap<&str, String> = BTreeMap::new();
    for input in &c.inputs {
        let value = values
            .get(&input.key)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .or_else(|| input.default.clone().filter(|_| !input.secret));
        match value {
            Some(v) => {
                check_value(&input.name, &v)?;
                if !input.choices.is_empty() && !input.choices.contains(&v) {
                    return Err(format!("{} must be one of {}", input.name, input.choices.join(", ")));
                }
                filled.insert(input.key.as_str(), v);
            }
            None if input.required => return Err(format!("{} is required", input.name)),
            None => {}
        }
    }

    let mut server = Server {
        description: Some(listing.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| listing.description.clone()))
            .filter(|d| !d.is_empty())
            .map(|d| d.chars().take(200).collect()),
        source: Some(format!("registry:{}@{}", listing.name, listing.version)),
        ..Default::default()
    };
    for input in &c.inputs {
        let Some(v) = filled.get(input.key.as_str()) else { continue };
        if let Some(name) = input.key.strip_prefix("env:") {
            server.env.insert(name.to_string(), v.clone());
        } else if let Some(name) = input.key.strip_prefix("header:") {
            server.headers.insert(name.to_string(), v.clone());
        }
    }

    if c.kind == "remote" {
        server.url = Some(c.identifier.clone());
        return Ok(server);
    }

    let mut args = Vec::new();
    match c.kind.as_str() {
        "npm" => {
            args.push("-y".to_string());
            args.extend(c.runtime_fixed.iter().filter(|a| *a != "-y" && *a != "--yes").cloned());
            args.push(match c.version.as_deref() {
                Some(v) if v != "latest" => format!("{}@{v}", c.identifier),
                _ => c.identifier.clone(),
            });
        }
        "pypi" => {
            args.extend(c.runtime_fixed.iter().cloned());
            args.push(match c.version.as_deref() {
                Some(v) if v != "latest" => format!("{}=={v}", c.identifier),
                _ => c.identifier.clone(),
            });
        }
        "oci" => {
            args.extend(["run", "-i", "--rm"].map(str::to_string));
            for name in server.env.keys() {
                args.push("-e".into());
                args.push(name.clone());
            }
            args.extend(c.runtime_fixed.iter().cloned());
            args.push(match c.version.as_deref() {
                Some(v) if v != "latest" && !c.identifier.contains(':') => format!("{}:{v}", c.identifier),
                _ => c.identifier.clone(),
            });
        }
        other => return Err(format!("{other} packages cannot be installed from ozgent")),
    }
    for arg in &c.fixed {
        match arg {
            Arg::Fixed { flag, value } => {
                args.extend(flag.iter().cloned());
                args.push(value.clone());
            }
            Arg::Asked { flag, key } => {
                if let Some(v) = filled.get(key.as_str()) {
                    args.extend(flag.iter().cloned());
                    args.push(v.clone());
                }
            }
        }
    }
    server.command = c.runner.clone();
    server.args = args;
    Ok(server)
}

/// A package by name, not from the registry: `npm` runs with `npx`, `pypi`
/// with `uvx`.
pub fn plan_package(kind: &str, package: &str, version: Option<&str>, args: &[String]) -> Result<Server, String> {
    check_identifier(kind, package)?;
    if let Some(v) = version {
        check_version(v)?;
    }
    for a in args {
        check_value("an argument", a)?;
    }
    let (runner, spec) = match (kind, version) {
        ("npm", Some(v)) => ("npx", format!("{package}@{v}")),
        ("npm", None) => ("npx", package.to_string()),
        ("pypi", Some(v)) => ("uvx", format!("{package}=={v}")),
        ("pypi", None) => ("uvx", package.to_string()),
        _ => return Err(format!("{kind} packages cannot be installed; use npm or pypi")),
    };
    let mut all = Vec::new();
    if runner == "npx" {
        all.push("-y".to_string());
    }
    all.push(spec.clone());
    all.extend(args.iter().cloned());
    Ok(Server {
        command: Some(runner.into()),
        args: all,
        source: Some(format!("{kind}:{spec}")),
        ..Default::default()
    })
}

/// Folders named in a server's arguments: `server-filesystem ~/notes` is
/// asking for `~/notes`, so a sandboxed server may use the folders its own
/// command line names. Only folders that exist, and never `/` or the whole
/// home folder.
pub fn folders_in_args(args: &[String]) -> Vec<std::path::PathBuf> {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let mut out = Vec::new();
    for arg in args {
        // `--root=/x` as well as `/x`.
        let value = arg.split_once('=').map(|(_, v)| v).unwrap_or(arg);
        let path = match (value.strip_prefix("~/"), &home) {
            (Some(rest), Some(h)) => h.join(rest),
            _ => std::path::PathBuf::from(value),
        };
        if !path.is_absolute() || path.parent().is_none() || home.as_ref().is_some_and(|h| &path == h) {
            continue;
        }
        if path.is_dir() && !out.contains(&path) {
            out.push(path);
        }
    }
    out
}

/// The command line a server entry runs, for showing before it is saved.
/// Secret values are never part of it: they go in the environment or
/// headers, which are listed by name only.
pub fn preview(server: &Server) -> String {
    let quote = |s: &str| {
        if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./@=:+,".contains(c)) {
            s.to_string()
        } else {
            format!("'{}'", s.replace('\'', "'\\''"))
        }
    };
    if let Some(url) = &server.url {
        let mut out = format!("connect to {url}");
        if !server.headers.is_empty() {
            out.push_str(&format!(" with headers {}", server.headers.keys().cloned().collect::<Vec<_>>().join(", ")));
        }
        return out;
    }
    let mut out = server.command.clone().unwrap_or_default();
    for a in &server.args {
        out.push(' ');
        out.push_str(&quote(a));
    }
    if !server.env.is_empty() {
        out.push_str(&format!("  (environment: {})", server.env.keys().cloned().collect::<Vec<_>>().join(", ")));
    }
    out
}

/// A local name for a listed server: the last part of its registry name,
/// cleaned to what a tool prefix may contain.
pub fn suggested_name(registry_name: &str) -> String {
    let last = registry_name.rsplit('/').next().unwrap_or(registry_name);
    let last = last.strip_prefix("mcp-server-").or_else(|| last.strip_prefix("server-")).unwrap_or(last);
    let last = last.strip_suffix("-mcp-server").or_else(|| last.strip_suffix("-mcp")).unwrap_or(last);
    let clean: String = last
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c.to_ascii_lowercase() } else { '-' })
        .take(24)
        .collect();
    let clean = clean.trim_matches('-').to_string();
    if clean.is_empty() { "server".into() } else { clean }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn npm_entry() -> Value {
        json!({
            "name": "io.github.example/files",
            "title": "Files",
            "description": "Read files",
            "version": "1.2.0",
            "repository": { "url": "https://github.com/example/files" },
            "packages": [{
                "registryType": "npm",
                "identifier": "@example/files-mcp",
                "version": "1.2.0",
                "runtimeHint": "npx",
                "transport": { "type": "stdio" },
                "runtimeArguments": [{ "type": "positional", "value": "-y" }],
                "packageArguments": [
                    { "type": "named", "name": "root", "isRequired": true, "description": "Folder" },
                    { "type": "named", "name": "--verbose", "value": "true" }
                ],
                "environmentVariables": [
                    { "name": "FILES_TOKEN", "isRequired": true, "isSecret": true },
                    { "name": "FILES_MODE", "default": "read", "choices": ["read", "write"] }
                ]
            }, {
                "registryType": "npm",
                "identifier": "@example/files-mcp",
                "version": "1.2.0",
                "transport": { "type": "sse", "url": "http://127.0.0.1:{port}/sse" }
            }],
            "remotes": [{
                "type": "streamable-http",
                "url": "https://files.example.com/mcp",
                "headers": [{ "name": "Authorization", "isRequired": true, "isSecret": true }]
            }, { "type": "sse", "url": "https://files.example.com/sse" }]
        })
    }

    fn values(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn an_npm_package_becomes_a_pinned_npx_command() {
        let listing = parse(&npm_entry()).unwrap();
        assert_eq!(listing.options.len(), 4);
        let s = plan(&listing, 0, &values(&[("arg:0", "/home/me/notes"), ("env:FILES_TOKEN", "t0k")])).unwrap();
        assert_eq!(s.command.as_deref(), Some("npx"));
        assert_eq!(s.args, ["-y", "@example/files-mcp@1.2.0", "--root", "/home/me/notes", "--verbose", "true"]);
        assert_eq!(s.env["FILES_TOKEN"], "t0k");
        assert_eq!(s.env["FILES_MODE"], "read", "defaults fill in");
        assert_eq!(s.source.as_deref(), Some("registry:io.github.example/files@1.2.0"));
        assert!(!preview(&s).contains("t0k"), "a secret is never in the preview: {}", preview(&s));
    }

    #[test]
    fn required_inputs_and_stray_values_are_refused() {
        let listing = parse(&npm_entry()).unwrap();
        assert!(plan(&listing, 0, &values(&[("env:FILES_TOKEN", "t")])).unwrap_err().contains("--root"));
        let stray = plan(&listing, 0, &values(&[("arg:0", "/x"), ("env:FILES_TOKEN", "t"), ("env:LD_PRELOAD", "/evil.so")]));
        assert!(stray.unwrap_err().contains("LD_PRELOAD"));
        let bad_choice = plan(&listing, 0, &values(&[("arg:0", "/x"), ("env:FILES_TOKEN", "t"), ("env:FILES_MODE", "admin")]));
        assert!(bad_choice.is_err());
        let newline = plan(&listing, 0, &values(&[("arg:0", "/x\n--evil"), ("env:FILES_TOKEN", "t")]));
        assert!(newline.is_err());
    }

    #[test]
    fn what_ozgent_cannot_run_says_why() {
        let listing = parse(&npm_entry()).unwrap();
        assert!(!listing.options[1].supported, "a package with its own web server");
        assert!(listing.options[1].why_not.as_deref().unwrap().contains("sse"));
        assert!(listing.options[2].supported, "streamable-http remote");
        assert!(!listing.options[3].supported, "sse remote");
        assert!(plan(&listing, 1, &BTreeMap::new()).is_err());
    }

    #[test]
    fn a_remote_becomes_a_url_with_headers() {
        let listing = parse(&npm_entry()).unwrap();
        let s = plan(&listing, 2, &values(&[("header:Authorization", "Bearer abc")])).unwrap();
        assert_eq!(s.url.as_deref(), Some("https://files.example.com/mcp"));
        assert_eq!(s.headers["Authorization"], "Bearer abc");
        assert!(s.command.is_none() && s.transport().is_ok());
    }

    #[test]
    fn a_package_name_that_could_be_a_flag_is_refused() {
        let mut entry = npm_entry();
        entry["packages"][0]["identifier"] = json!("--registry=https://evil.test");
        let listing = parse(&entry).unwrap();
        assert!(!listing.options[0].supported);
        assert!(plan_package("npm", "-e", None, &[]).is_err());
        assert!(plan_package("pypi", "mcp-server-time", Some("2025.1.0"), &[]).is_ok());
        assert!(plan_package("npm", "@modelcontextprotocol/server-filesystem", None, &["/tmp".into()]).is_ok());
        assert!(check_version("--pre").is_err());
    }

    #[test]
    fn pypi_runs_with_uvx() {
        let s = plan_package("pypi", "mcp-server-time", Some("2025.1.0"), &["--local-timezone".into(), "UTC".into()]).unwrap();
        assert_eq!(s.command.as_deref(), Some("uvx"));
        assert_eq!(s.args, ["mcp-server-time==2025.1.0", "--local-timezone", "UTC"]);
    }

    #[test]
    fn folders_named_in_arguments_are_found() {
        let dir = std::env::temp_dir();
        let found = folders_in_args(&["-y".into(), "pkg".into(), dir.display().to_string(), format!("--root={}", dir.display()), "/".into(), "/no/such/dir".into()]);
        assert_eq!(found, vec![dir]);
    }

    #[test]
    fn names_are_suggested_from_the_registry_name() {
        assert_eq!(suggested_name("io.github.bytedance/mcp-server-filesystem"), "filesystem");
        assert_eq!(suggested_name("com.example/Weather Tool"), "weather-tool");
        assert!(ozgent_core::mcp::valid_name(&suggested_name("x/@@@")).is_ok());
    }
}
