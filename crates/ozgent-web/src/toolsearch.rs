//! Tools described on request, found with `find_tools`.
//!
//! Every tool described to the model up front costs context before anyone
//! has said anything, and past a point it costs accuracy too: with over a
//! hundred MCP tools the descriptions took 38,896 tokens of a 64k window, and
//! a 4B model fumbled its first call. Published results agree — models pick
//! the right tool more often from a few candidates than from a long list.
//!
//! So past a budget, the tools of MCP servers are not described up front.
//! The model gets one tool, `find_tools`, whose description names each such
//! server and what it does; calling it returns the full description of the
//! best matches, and those can then be called by name.
//!
//! **The prompt's head never changes because of this.** ozgent serves a
//! conversation from its cache as long as the start of the prompt is the
//! same, and the tool descriptions sit at the start: narrowing the list per
//! turn would re-read the whole conversation every turn (docs/tools.md
//! measures it). Here the head is fixed — ozgent's own tools, the servers set
//! to load always, and `find_tools` — and what the model looks up arrives as
//! a tool result, at the end, where it costs nothing to what is cached.

use std::collections::{BTreeMap, HashMap};

use ozgent_core::mcp::Load;
use ozgent_core::{Config, ToolSpec};
use serde_json::{Value, json};

use crate::worker::Tools;

pub const FIND_TOOLS: &str = "find_tools";

/// Tool descriptions, in tokens, carried up front before MCP servers set to
/// load automatically are moved behind `find_tools`. Roughly ozgent's own
/// tools and a server or two: under it, nothing changes.
pub const BUDGET_TOKENS: usize = 4000;

/// How many tools one search returns in full.
const RESULTS: usize = 6;

/// The tools a turn describes up front, and the ones it can call.
pub struct Split {
    /// In the prompt: rendered by the template or the preamble.
    pub declared: Vec<ToolSpec>,
    /// Callable but described only on request.
    pub deferred: Vec<ToolSpec>,
}

/// A rough token count for a tool's description: its JSON, a character in
/// four. Close enough to decide whether a list is large.
pub fn tokens(spec: &ToolSpec) -> usize {
    (spec.name.len() + spec.description.len() + spec.input_schema.to_string().len()) / 4
}

fn server_of(tools: &Tools, name: &str) -> Option<String> {
    tools.host.source_of(name).and_then(|o| o.strip_prefix("mcp:").map(str::to_string))
}

/// Decide what this turn describes up front.
pub fn split(offered: Vec<ToolSpec>, tools: &Tools, config: &Config) -> Split {
    let total: usize = offered.iter().map(tokens).sum();
    let over = total > BUDGET_TOKENS;
    let mut declared = Vec::new();
    let mut deferred = Vec::new();
    for spec in offered {
        let load = server_of(tools, &spec.name)
            .and_then(|s| config.mcp.servers.get(&s).map(|srv| srv.load));
        let defer = match load {
            // ozgent's own tools, the scheduler, a caller's: always up front.
            None => false,
            Some(Load::Always) => false,
            Some(Load::OnRequest) => true,
            Some(Load::Auto) => over,
        };
        if defer { deferred.push(spec) } else { declared.push(spec) }
    }
    if !deferred.is_empty() {
        declared.push(find_tools_spec(&deferred, tools, config));
    }
    Split { declared, deferred }
}

/// How much of `find_tools`' description the directory of tools described on
/// request may take, in tokens. It is in the prompt's head, paid once and
/// cached, but it is window a conversation cannot use.
pub const DIRECTORY_TOKENS: usize = 4000;

/// `find_tools`, whose description is a directory of every tool described on
/// request, as detailed as fits in [`DIRECTORY_TOKENS`]:
///
/// 1. each tool's exact name and what it does, in one line;
/// 2. else each server, what it is, and its tools' names;
/// 3. else each server and what it is;
/// 4. else the servers' names.
///
/// The model chooses tools well from names and purposes — measured, a 4B
/// picked the right one of 129 in every case from the first form — and badly
/// from a summary it had to guess names from. What costs context is the
/// parameters: 119 tools' full descriptions were ~36,000 tokens, the first
/// form ~2,300. Past a few hundred tools even that does not fit, and the
/// lookup made for each message (see [`lookup`]) does the choosing; the
/// directory is then there so the model knows what exists to ask for.
/// Built from the tools alone, in a fixed order, so it is the same every turn.
pub fn find_tools_spec(deferred: &[ToolSpec], tools: &Tools, config: &Config) -> ToolSpec {
    let mut by_server: BTreeMap<String, Vec<&ToolSpec>> = BTreeMap::new();
    for spec in deferred {
        by_server.entry(server_of(tools, &spec.name).unwrap_or_default()).or_default().push(spec);
    }
    for specs in by_server.values_mut() {
        specs.sort_by(|a, b| a.name.cmp(&b.name));
    }
    let about = |server: &str| {
        server_about(tools, config, server).map(|d| format!(" — {d}")).unwrap_or_default()
    };
    let directory = directory(&by_server, &about);
    ToolSpec {
        name: FIND_TOOLS.to_string(),
        description: format!(
            "More tools, listed here without their parameters. To use one, call find_tools with what \
             you need to do (or its name) to get its parameters, then call it by name. These are real \
             tools you can call:\n{directory}"
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "What you need to do, or a tool's name from the list." },
                "server": { "type": "string", "description": "Only this server's tools. Optional." }
            },
            "required": ["query"]
        }),
        output_schema: None,
        effect: ozgent_core::permission::Effect::Read,
    }
}

/// What a server is, in a line: its `description` if one was given, else
/// what it said about itself in the handshake ("Stealth browser for AI
/// agents. Create a session, open pages…"), cut to its first two sentences.
/// A server pasted in without a description was otherwise just a name, and a
/// name does not say "this is a browser".
pub fn server_about(tools: &Tools, config: &Config, server: &str) -> Option<String> {
    let given = config.mcp.servers.get(server).and_then(|s| s.description.clone()).filter(|d| !d.trim().is_empty());
    let said = || {
        tools.mcp.iter().find(|s| s.name == server).map(|s| s.instructions.clone()).filter(|i| !i.trim().is_empty())
    };
    given.or_else(said).map(|text| brief(&text, 24))
}

/// The first two sentences of `text`, at most `limit` words.
fn brief(text: &str, limit: usize) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut end = text.len();
    let mut seen = 0;
    for (i, c) in text.char_indices() {
        if matches!(c, '.' | '!' | '?') && text[i + 1..].starts_with(' ') {
            seen += 1;
            if seen == 2 {
                end = i + 1;
                break;
            }
        }
    }
    let words: Vec<&str> = text[..end].split(' ').collect();
    if words.len() > limit { format!("{}…", words[..limit].join(" ")) } else { text[..end].trim_end_matches('.').to_string() }
}

/// The most detailed directory that fits [`DIRECTORY_TOKENS`].
fn directory(by_server: &BTreeMap<String, Vec<&ToolSpec>>, about: &dyn Fn(&str) -> String) -> String {
    let fits = |text: &str| text.len() / 4 <= DIRECTORY_TOKENS;
    let tools_named = |specs: &[&ToolSpec], server: &str| {
        specs.iter().map(|s| s.name.strip_prefix(&format!("{server}_")).unwrap_or(&s.name).to_string()).collect::<Vec<_>>()
    };
    let every_tool: String = by_server
        .iter()
        .flat_map(|(server, specs)| {
            std::iter::once(format!("{server}{}:", about(server)))
                .chain(specs.iter().map(|s| format!("  {}: {}", s.name, one_line(&s.description))))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if fits(&every_tool) {
        return every_tool;
    }
    // Names without the server's prefix, which the header gives once.
    let with_names: String = by_server
        .iter()
        .map(|(server, specs)| {
            format!("{server}{} — tools {server}_…: {}", about(server), tools_named(specs, server).join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n");
    if fits(&with_names) {
        return with_names;
    }
    let servers: String = by_server
        .iter()
        .map(|(server, specs)| format!("{server} ({} tools){}", specs.len(), about(server)))
        .collect::<Vec<_>>()
        .join("\n");
    if fits(&servers) {
        return servers;
    }
    let total: usize = by_server.values().map(Vec::len).sum();
    let mut names = String::new();
    let mut shown = 0;
    for (server, specs) in by_server {
        let item = format!("{server} ({}), ", specs.len());
        if !fits(&format!("{names}{item}")) {
            break;
        }
        names.push_str(&item);
        shown += 1;
    }
    let rest = by_server.len() - shown;
    let tail = if rest > 0 { format!(" and {rest} more servers") } else { String::new() };
    format!("{total} tools from these servers: {}{tail}", names.trim_end_matches(", "))
}

/// The first sentence of a description, at most fourteen words.
fn one_line(description: &str) -> String {
    let text = description.split_whitespace().collect::<Vec<_>>().join(" ");
    let first = text
        .char_indices()
        .find(|(i, c)| matches!(c, '.' | '!' | '?') && text[i + 1..].starts_with(' '))
        .map(|(i, _)| &text[..i])
        .unwrap_or(&text);
    let words: Vec<&str> = first.split(' ').collect();
    if words.len() > 14 { format!("{}…", words[..14].join(" ")) } else { first.trim_end_matches('.').to_string() }
}

/// A call to a tool described on request whose schema asks for parameters
/// it did not give: the answer is its full description, so the model can
/// call it again properly instead of the server failing on it.
pub fn missing_parameters(spec: &ToolSpec, arguments: &Value) -> Option<Value> {
    let required = spec.input_schema.get("required")?.as_array()?;
    let missing: Vec<&str> = required
        .iter()
        .filter_map(Value::as_str)
        .filter(|r| arguments.get(*r).is_none_or(Value::is_null))
        .collect();
    if missing.is_empty() {
        return None;
    }
    Some(json!({
        "error": format!("missing required parameters: {}. Call {} again with its parameters below.", missing.join(", "), spec.name),
        "name": spec.name,
        "description": spec.description,
        "parameters": spec.input_schema,
    }))
}

fn words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric()) {
        if raw.is_empty() {
            continue;
        }
        // camelCase and snake_case both split into words.
        let mut word = String::new();
        let mut last_lower = false;
        for c in raw.chars() {
            if c.is_uppercase() && last_lower && !word.is_empty() {
                out.push(std::mem::take(&mut word));
            }
            last_lower = c.is_lowercase();
            word.extend(c.to_lowercase());
        }
        if !word.is_empty() {
            out.push(word);
        }
    }
    out.into_iter().filter(|w| w.len() > 1 && !STOP.contains(&w.as_str())).map(|w| stem(&w)).collect()
}

const STOP: &[&str] = &[
    "the", "and", "for", "with", "from", "into", "this", "that", "use", "using", "you", "your", "can",
    "will", "are", "was", "its", "not", "all", "any", "one", "get", "of", "to", "in", "on", "or", "an", "is",
    "it", "be", "by", "as", "at", "if", "me", "my", "do", "need", "want", "please",
];

/// Crude, but it makes the forms people write meet the forms descriptions
/// use: "files"/"file", "reading"/"read", "created"/"create"/"creating",
/// "directories"/"directory".
fn stem(w: &str) -> String {
    let mut base = w.to_string();
    if let Some(b) = base.strip_suffix("ies").filter(|b| b.len() >= 2) {
        base = format!("{b}y");
    } else if ["sses", "xes", "ches", "shes", "zes"].iter().any(|s| base.ends_with(s)) {
        base.truncate(base.len() - 2);
    } else if base.ends_with('s') && !base.ends_with("ss") && base.len() > 3 {
        base.pop();
    } else if let Some(b) = base.strip_suffix("ing").filter(|b| b.len() >= 3) {
        base = b.to_string();
    } else if let Some(b) = base.strip_suffix("ed").filter(|b| b.len() >= 3) {
        base = b.to_string();
    }
    // A final e comes and goes with the suffix ("create", "creat-ing").
    if base.len() > 3 && base.ends_with('e') {
        base.pop();
    }
    base
}

/// Words people use for what tool descriptions call something else. Only
/// for the query: a description says "directory", a person says "folder".
const SYNONYMS: &[(&str, &[&str])] = &[
    ("folder", &["directory"]),
    ("directory", &["folder"]),
    ("site", &["page", "url", "web"]),
    ("website", &["page", "url", "web"]),
    ("webpage", &["page", "url", "web"]),
    ("page", &["url", "web"]),
    ("link", &["url", "href"]),
    ("open", &["navigate", "browse"]),
    ("visit", &["navigate", "browse"]),
    ("go", &["navigate"]),
    ("browse", &["navigate"]),
    ("video", &["youtube"]),
    ("transcript", &["caption", "subtitle"]),
    ("picture", &["screenshot", "image"]),
    ("image", &["screenshot"]),
    ("rss", &["feed"]),
    ("feed", &["rss"]),
    ("clock", &["time"]),
    ("timezone", &["zone", "time"]),
    ("stock", &["price", "quote", "ticker"]),
    ("delete", &["remove"]),
    ("remove", &["delete"]),
    ("make", &["create"]),
    ("new", &["create"]),
    ("look", &["search", "find"]),
];

fn expand(query: &[String]) -> Vec<String> {
    let mut out = query.to_vec();
    for w in query {
        if let Some((_, more)) = SYNONYMS.iter().find(|(k, _)| stem(k) == *w) {
            for m in *more {
                let m = stem(m);
                if !out.contains(&m) {
                    out.push(m);
                }
            }
        }
    }
    out
}

fn document(spec: &ToolSpec) -> Vec<String> {
    let mut text = format!("{} {} {}", spec.name, spec.name, spec.description);
    if let Some(props) = spec.input_schema.get("properties").and_then(Value::as_object) {
        for (key, p) in props {
            text.push(' ');
            text.push_str(key);
            if let Some(d) = p.get("description").and_then(Value::as_str) {
                text.push(' ');
                text.push_str(d);
            }
        }
    }
    words(&text)
}

/// The deferred tools best matching `query`, best first. `server_of` names
/// the MCP server a tool comes from, for the optional filter.
pub fn search<'a>(
    deferred: &'a [ToolSpec],
    server_of: impl Fn(&str) -> Option<String>,
    query: &str,
    server: Option<&str>,
) -> Vec<&'a ToolSpec> {
    let query = clean_query(query);
    let pool: Vec<&ToolSpec> = deferred
        .iter()
        .filter(|s| server.is_none_or(|want| server_of(&s.name).as_deref() == Some(want)))
        .collect();
    if words(&query).is_empty() {
        return pool.into_iter().take(RESULTS).collect();
    }
    let mut found = ranked(&pool, &server_of, &|_| String::new(), &query);
    found.truncate(RESULTS);
    found
}

/// Tools ranked for `query` by keywords (BM25) and by meaning (the embedding
/// model, when one is installed), the two lists fused by reciprocal rank, as
/// ozgent's memory recall does.
///
/// Measured on 2,797 tools from 308 MCP servers (the MCP-Zero set) with 260
/// requests written by a model, the right tool was among the first three
/// for 52% of requests by keywords, 57% by meaning and 62% fused; a fitting
/// tool, judged, more often still (docs/tools.md has the table). Routing to a
/// server first, as MCP-Zero does, did worse at every size: servers overlap,
/// and a request's words name a server's topic more often than its tools.
fn ranked<'a>(
    pool: &[&'a ToolSpec],
    server_of: &dyn Fn(&str) -> Option<String>,
    about: &dyn Fn(&str) -> String,
    query: &str,
) -> Vec<&'a ToolSpec> {
    // Each ranking takes part with its first fifty; past that a match adds
    // noise, not recall.
    const DEPTH: usize = 50;
    let keyword = scored_pool(pool, server_of, about, query);
    let meaning = semantic(pool, query).unwrap_or_default();
    let mut fused: HashMap<&str, (f64, &ToolSpec)> = HashMap::new();
    for (rank, (_, spec)) in keyword.iter().filter(|(score, _)| *score > 0.0).take(DEPTH).enumerate() {
        fused.entry(spec.name.as_str()).or_insert((0.0, spec)).0 += 1.0 / (60.0 + rank as f64);
    }
    for (rank, (_, spec)) in meaning.iter().take(DEPTH).enumerate() {
        fused.entry(spec.name.as_str()).or_insert((0.0, spec)).0 += 1.0 / (60.0 + rank as f64);
    }
    let mut ranked: Vec<(f64, &ToolSpec)> = fused.into_values().collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.name.cmp(&b.1.name)));
    ranked.into_iter().map(|(_, s)| s).collect()
}

/// An MCP server that is a web browser, and the tools that make it one.
///
/// Found by what its tools say they do, not by its name: camofox, Playwright
/// and ghostcloak call the same steps different things.
#[derive(Debug, Clone, PartialEq)]
pub struct Browser {
    pub server: String,
    /// Starts a session or tab, for a browser that wants one first.
    pub start: Option<String>,
    /// Goes to a URL.
    pub open: String,
    /// Reads the page as text.
    pub read: String,
}

impl Browser {
    pub fn tools(&self) -> Vec<&str> {
        self.start.iter().map(String::as_str).chain([self.open.as_str(), self.read.as_str()]).collect()
    }
}

/// The browser among these tools, if one server has a tool that goes to a
/// URL and one that reads the page's text.
pub fn find_browser(specs: &[ToolSpec], server_of: &dyn Fn(&str) -> Option<String>) -> Option<Browser> {
    // The name's words, a blank, then what it says it does: the blank lets a
    // check read the name alone.
    let text = |s: &ToolSpec| format!("{}  {}", s.name.replace('_', " "), one_line(&s.description)).to_lowercase();
    let mut servers: Vec<String> = specs.iter().filter_map(|s| server_of(&s.name)).collect();
    servers.sort();
    servers.dedup();
    for server in servers {
        let own: Vec<&ToolSpec> = specs.iter().filter(|s| server_of(&s.name).as_deref() == Some(&server)).collect();
        let pick = |score: &dyn Fn(&str) -> i32| {
            own.iter()
                .map(|s| (score(&text(s)), s.name.clone()))
                .filter(|(n, _)| *n > 0)
                .max_by_key(|(n, name)| (*n, std::cmp::Reverse(name.len())))
                .map(|(_, name)| name)
        };
        let open = pick(&|t| {
            let url = t.contains("url");
            let goes = ["navigate", " open", "go to", "visit", "load"].iter().any(|w| t.contains(w));
            if url && goes { 2 + t.contains("navigate") as i32 + t.contains(" open") as i32 } else { 0 }
        });
        let read = pick(&|t| {
            let page = t.contains("page");
            let text = ["text", "snapshot", "markdown", "content"].iter().any(|w| t.contains(w));
            // A screenshot is a picture, not the page's text — judged by the
            // tool's name: Playwright's snapshot calls itself "better than
            // screenshot".
            let picture = t.split(' ').take_while(|w| !w.is_empty()).any(|w| w == "screenshot");
            if page && text && !picture { 2 + t.contains("snapshot") as i32 + t.contains("text") as i32 } else { 0 }
        });
        let start = pick(&|t| {
            let thing = t.contains("session") || t.contains(" tab");
            let makes = ["create", "new", "launch", "start"].iter().any(|w| t.contains(w));
            (thing && makes) as i32
        });
        if let (Some(open), Some(read)) = (open, read) {
            if open != read {
                return Some(Browser { server, start: start.filter(|s| *s != open), open, read });
            }
        }
    }
    None
}

/// How to reach the web when ozgent's own `web_search` is not on offer.
#[derive(Debug, Clone, PartialEq)]
pub enum WebBridge {
    /// An MCP tool that searches the web itself.
    Search(String),
    /// A browser, which can search by opening a search engine's page.
    Browser(Browser),
}

impl WebBridge {
    /// The tools a web request should have described in full.
    pub fn tools(&self) -> Vec<&str> {
        match self {
            Self::Search(tool) => vec![tool.as_str()],
            Self::Browser(b) => b.tools(),
        }
    }

    /// Said once, in the prompt's head. Nothing is said when there is no way
    /// to the web at all: then the model saying it cannot search is right.
    pub fn hint(&self) -> String {
        match self {
            Self::Search(tool) => format!(
                "There is no web_search tool here; to search the web, use {tool}."
            ),
            Self::Browser(b) => browser_hint(b),
        }
    }
}

/// The way to the web among `specs` when `web_search` itself is not one of
/// them: an MCP tool that searches the web, else a browser, else none.
pub fn web_bridge(specs: &[ToolSpec], server_of: impl Fn(&str) -> Option<String>) -> Option<WebBridge> {
    if specs.iter().any(|s| s.name == "web_search") {
        return None;
    }
    let search = specs
        .iter()
        .filter(|s| server_of(&s.name).is_some())
        .filter(|s| {
            let t = format!("{} {}", s.name.replace('_', " "), one_line(&s.description)).to_lowercase();
            t.contains("search") && ["web", "internet", "online", "search engine"].iter().any(|w| t.contains(w))
                && !t.contains("page") && !t.contains("history")
        })
        .min_by_key(|s| s.name.len());
    if let Some(tool) = search {
        return Some(WebBridge::Search(tool.name.clone()));
    }
    find_browser(specs, &server_of).map(WebBridge::Browser)
}

/// Said once, in the prompt's head, when there is no `web_search` but there
/// is a browser: a small model told only that it has "ghostcloak_page_open"
/// answered "I don't have access to the web" as often as it used it.
pub fn browser_hint(b: &Browser) -> String {
    let start = b.start.as_ref().map(|s| format!("{s}, then ")).unwrap_or_default();
    format!(
        "There is no web_search tool here, but there is a web browser, {server}. To read a site the \
         user names, {start}{open} with its address, then {read}. To search the web, open \
         https://duckduckgo.com/html/?q=<search words> the same way and read the results; open a \
         result to read it.",
        server = b.server,
        open = b.open,
        read = b.read,
    )
}

/// Whether a message is asking for something from the web: a stand-in for
/// `web_search` is ranked among the tools, as the lookup ranks them, and a
/// message it comes near the top for is taken to want the web. Measured the
/// same way as every other match, so it is right when the lookup would have
/// been, and costs one more document in a ranking already being made.
pub fn wants_web(pool: &[ToolSpec], server_of: impl Fn(&str) -> Option<String>, message: &str) -> bool {
    let query = clean_query(message);
    if words(&query).is_empty() {
        return false;
    }
    let probe = ToolSpec {
        name: WEB_PROBE.to_string(),
        description: "Search the web for current information: news, facts, prices, people, places, \
                      products, anything online. Look it up on the internet and read web pages."
            .to_string(),
        input_schema: json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        output_schema: None,
        effect: ozgent_core::permission::Effect::Read,
    };
    let mut all: Vec<&ToolSpec> = pool.iter().collect();
    all.push(&probe);
    let ranked = ranked(&all, &server_of, &|_| String::new(), &query);
    ranked.iter().take(LOOKUP_FULL).any(|s| s.name == WEB_PROBE)
}

const WEB_PROBE: &str = "search_the_web";

/// What a lookup made for a message found.
pub struct Lookup<'a> {
    /// Described in full: callable at once.
    pub full: Vec<&'a ToolSpec>,
    /// Named, with what they do: the next candidates, one line each.
    pub related: Vec<&'a ToolSpec>,
}

/// For a message, before the model sees it: the tools most likely to serve
/// it.
///
/// Made for every message that has words to search by, not only those that
/// seem to need a tool. Whether a message needs one cannot be read from the
/// scores: with a thousand tools, "write me a haiku" matched a poetry tool as
/// well as real requests matched theirs (the no-tool messages' best scores
/// overlapped the requests' by keywords and by meaning alike). What it costs
/// a message that needs nothing is a few hundred tokens at the end of the
/// prompt, and measured, the answers to such messages did not change.
///
/// The first few come in full; the next ones by name and purpose, so a right
/// tool ranked fifth is still in front of the model — it chose well from
/// names and purposes — and costs a line rather than a schema.
pub fn lookup<'a>(
    deferred: &'a [ToolSpec],
    server_of: impl Fn(&str) -> Option<String>,
    about: impl Fn(&str) -> String,
    message: &str,
) -> Lookup<'a> {
    let query = clean_query(message);
    if words(&query).is_empty() {
        return Lookup { full: Vec::new(), related: Vec::new() };
    }
    let pool: Vec<&ToolSpec> = deferred.iter().collect();
    let mut found = ranked(&pool, &server_of, &about, &query);
    found.truncate(LOOKUP_FULL + LOOKUP_RELATED);
    let related = found.split_off(found.len().min(LOOKUP_FULL));
    tracing::info!(
        "tool lookup {query:?}: {} (then {})",
        found.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "),
        related.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ")
    );
    Lookup { full: found, related }
}

/// How many tools a lookup describes in full, and how many more it names.
pub const LOOKUP_FULL: usize = 3;
pub const LOOKUP_RELATED: usize = 7;

/// A message as a search: without the time stamp ozgent puts on it, links,
/// paths and quoted code, which match everything and mean nothing here.
pub fn clean_query(message: &str) -> String {
    let mut out = Vec::new();
    for token in message.split_whitespace() {
        let t = token.trim_matches(|c: char| "()[]{}<>\"'`,.;:!?".contains(c));
        let is_stamp = t.chars().all(|c| c.is_ascii_digit() || c == ':') && t.contains(':');
        let is_link = t.contains("://") || t.starts_with("www.");
        let is_path = t.starts_with('/') || t.starts_with("~/") || t.matches('/').count() >= 2;
        if !(is_stamp || is_link || is_path) {
            out.push(token);
        }
    }
    out.join(" ")
}

/// Tool descriptions' vectors, by the text embedded, so each is embedded once
/// per install rather than per message.
static VECTORS: std::sync::Mutex<Option<HashMap<String, Vec<f32>>>> = std::sync::Mutex::new(None);

fn tool_text(spec: &ToolSpec) -> String {
    let words = spec.name.replace('_', " ");
    format!("{words}: {}", spec.description.chars().take(600).collect::<String>())
}

/// Tools ranked by meaning, or `None` without an embedding model.
fn semantic<'a>(pool: &[&'a ToolSpec], query: &str) -> Option<Vec<(f32, &'a ToolSpec)>> {
    let embedder = crate::worker::tool_embedder()?;
    let model = embedder.embedding_model()?;
    let keys: Vec<String> = pool.iter().map(|s| vector_key(&model, s)).collect();
    // Normally embedded already, in the background when the tools started
    // (see `warm`); this is for a tool that arrived since.
    embed_missing(embedder, &model, pool.iter().copied())?;
    let q = query_vector(embedder, &model, query)?;
    let cache = VECTORS.lock().unwrap_or_else(|e| e.into_inner());
    let map = cache.as_ref()?;
    let mut ranked: Vec<(f32, &ToolSpec)> = pool
        .iter()
        .zip(&keys)
        .filter_map(|(s, k)| map.get(k).map(|v| (cosine(&q, v), *s)))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    Some(ranked)
}

/// A tool's vector is keyed by the model that made it as well as the text:
/// another model's vectors cannot be compared with this one's.
fn vector_key(model: &str, spec: &ToolSpec) -> String {
    format!("{model}\u{0}{}", tool_text(spec))
}

/// Embed the tools that have no vector yet, a few at a time. Small batches,
/// because a chat model being loaded waits for the embedding model's current
/// job before it takes the card (see `Membership::admit`).
fn embed_missing<'a>(
    embedder: &crate::worker::Worker,
    model: &str,
    specs: impl Iterator<Item = &'a ToolSpec>,
) -> Option<()> {
    let missing: Vec<(String, String)> = {
        let cache = VECTORS.lock().unwrap_or_else(|e| e.into_inner());
        specs
            .map(|s| (vector_key(model, s), tool_text(s)))
            .filter(|(k, _)| cache.as_ref().is_none_or(|c| !c.contains_key(k)))
            .collect()
    };
    for chunk in missing.chunks(WARM_BATCH) {
        let texts: Vec<String> = chunk.iter().map(|(_, t)| t.clone()).collect();
        let vectors = embedder.embed_as(ozgent_llama::embed::Role::Document, texts).ok()?;
        let mut cache = VECTORS.lock().unwrap_or_else(|e| e.into_inner());
        let map = cache.get_or_insert_with(HashMap::new);
        for ((key, _), v) in chunk.iter().zip(vectors) {
            map.insert(key.clone(), v);
        }
    }
    Some(())
}

const WARM_BATCH: usize = 8;

/// Embed every tool described on request, ahead of the first message that
/// would need it, and keep the vectors on disk. A vector depends only on the
/// model and the tool's text, so each is made once, ever: done by the first
/// message, on the CPU beside the chat model, 51 tools took 9.2 s of that
/// message's time. Blocking.
pub fn warm(specs: &[ToolSpec], store: &std::path::Path) {
    let Some(embedder) = crate::worker::tool_embedder() else { return };
    let Some(model) = embedder.embedding_model() else { return };
    let started = std::time::Instant::now();
    let loaded = load_vectors(store);
    let before = VECTORS.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, HashMap::len);
    if embed_missing(embedder, &model, specs.iter()).is_none() {
        return;
    }
    let cache = VECTORS.lock().unwrap_or_else(|e| e.into_inner());
    let made = cache.as_ref().map_or(0, HashMap::len).saturating_sub(before);
    if made > 0 {
        if let Some(map) = cache.as_ref() {
            save_vectors(store, map);
        }
    }
    tracing::info!(
        "tool lookup: {} tools ready in {} ms ({loaded} vectors from disk, {made} made)",
        specs.len(),
        started.elapsed().as_millis()
    );
}

/// Read saved vectors into the cache. Returns how many were read. A file
/// that does not parse is ignored and rewritten: it is a cache.
fn load_vectors(store: &std::path::Path) -> usize {
    let Ok(bytes) = std::fs::read(store) else { return 0 };
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let slice = bytes.get(at..at + n)?;
        at += n;
        Some(slice)
    };
    let mut read = Vec::new();
    loop {
        let Some(len) = take(4) else { break };
        let len = u32::from_le_bytes(len.try_into().unwrap_or_default()) as usize;
        let (Some(key), Some(dim)) = (take(len).map(|k| String::from_utf8_lossy(k).into_owned()), take(4)) else { break };
        let dim = u32::from_le_bytes(dim.try_into().unwrap_or_default()) as usize;
        let Some(raw) = take(dim * 4) else { break };
        let v: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap_or_default())).collect();
        read.push((key, v));
    }
    let n = read.len();
    let mut cache = VECTORS.lock().unwrap_or_else(|e| e.into_inner());
    let map = cache.get_or_insert_with(HashMap::new);
    for (k, v) in read {
        map.entry(k).or_insert(v);
    }
    n
}

/// Write the cache, whole, beside the old file and then over it.
fn save_vectors(store: &std::path::Path, map: &HashMap<String, Vec<f32>>) {
    let mut out = Vec::new();
    for (key, v) in map {
        out.extend((key.len() as u32).to_le_bytes());
        out.extend(key.as_bytes());
        out.extend((v.len() as u32).to_le_bytes());
        for x in v {
            out.extend(x.to_le_bytes());
        }
    }
    let partial = store.with_extension("partial");
    if let Some(dir) = store.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if std::fs::write(&partial, out).is_ok() {
        let _ = std::fs::rename(&partial, store);
    }
}

/// Where tool vectors are kept.
pub fn vector_store(paths: &ozgent_core::Paths) -> std::path::PathBuf {
    paths.cache_dir().join("tool-vectors.bin")
}

/// [`warm`] on a thread of its own, for the MCP tools of `tools`.
pub fn warm_in_background(tools: &Tools, paths: &ozgent_core::Paths) {
    let store = vector_store(paths);
    let specs: Vec<ToolSpec> = tools
        .host
        .tools()
        .iter()
        .filter(|s| server_of(tools, &s.name).is_some())
        .cloned()
        .collect();
    if specs.is_empty() {
        return;
    }
    let _ = std::thread::Builder::new().name("ozgent-tool-vectors".into()).spawn(move || warm(&specs, &store));
}

/// The message's vector, made once per message: the lookup and the check
/// for a web request both rank by it, and on the CPU each embedding of it
/// cost a few hundred milliseconds.
fn query_vector(embedder: &crate::worker::Worker, model: &str, query: &str) -> Option<Vec<f32>> {
    let key = format!("{model}\u{0}{query}");
    if let Some((_, v)) = QUERIES.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| *k == key) {
        return Some(v.clone());
    }
    let v = embedder.embed_as(ozgent_llama::embed::Role::ToolQuery, vec![query.to_string()]).ok()?.pop()?;
    let mut recent = QUERIES.lock().unwrap_or_else(|e| e.into_inner());
    if recent.len() >= 8 {
        recent.pop_front();
    }
    recent.push_back((key, v.clone()));
    Some(v)
}

static QUERIES: std::sync::Mutex<std::collections::VecDeque<(String, Vec<f32>)>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn scored_pool<'a>(
    pool: &[&'a ToolSpec],
    server_of: &dyn Fn(&str) -> Option<String>,
    about: &dyn Fn(&str) -> String,
    query: &str,
) -> Vec<(f64, &'a ToolSpec)> {
    let owned: Vec<ToolSpec> = pool.iter().map(|s| (*s).clone()).collect();
    let ranked = scored(&owned, server_of, about, query, None);
    ranked
        .into_iter()
        .filter_map(|(score, s)| pool.iter().find(|p| p.name == s.name).map(|p| (score, *p)))
        .collect()
}

fn scored<'a>(
    deferred: &'a [ToolSpec],
    server_of: &dyn Fn(&str) -> Option<String>,
    about: &dyn Fn(&str) -> String,
    query: &str,
    server: Option<&str>,
) -> Vec<(f64, &'a ToolSpec)> {
    let pool: Vec<&ToolSpec> = deferred
        .iter()
        .filter(|s| server.is_none_or(|want| server_of(&s.name).as_deref() == Some(want)))
        .collect();
    let q = expand(&words(query));
    if q.is_empty() {
        return pool.into_iter().map(|s| (0.0, s)).collect();
    }
    let docs: Vec<Vec<String>> = pool
        .iter()
        .map(|s| {
            let mut d = document(s);
            if let Some(server) = server_of(&s.name) {
                d.extend(words(&about(&server)));
            }
            d
        })
        .collect();
    let n = docs.len().max(1) as f64;
    let avg = docs.iter().map(Vec::len).sum::<usize>() as f64 / n;
    let mut df: HashMap<&str, usize> = HashMap::new();
    for d in &docs {
        let mut seen: Vec<&str> = d.iter().map(String::as_str).collect();
        seen.sort_unstable();
        seen.dedup();
        for w in seen {
            *df.entry(w).or_default() += 1;
        }
    }
    let (k1, b) = (1.2, 0.75);
    let mut scored: Vec<(f64, usize)> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let len = d.len() as f64;
            let score: f64 = q
                .iter()
                .map(|term| {
                    let tf = d.iter().filter(|w| *w == term).count() as f64;
                    if tf == 0.0 {
                        return 0.0;
                    }
                    let n_t = *df.get(term.as_str()).unwrap_or(&0) as f64;
                    let idf = ((n - n_t + 0.5) / (n_t + 0.5) + 1.0).ln();
                    idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len / avg.max(1.0)))
                })
                .sum();
            (score, i)
        })
        .filter(|(s, _)| *s > 0.0)
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().map(|(score, i)| (score, pool[i])).collect()
}

/// A call to a tool that does not exist, as a search: the words of the
/// invented name and of the arguments' names and short values.
/// `github_create_issue {"title": …}` finds `github_issue_write`.
pub fn guess_as_query(name: &str, arguments: &Value) -> String {
    let mut parts = vec![name.replace(['_', '-', '.'], " ")];
    if let Some(args) = arguments.as_object() {
        for (key, value) in args {
            parts.push(key.replace('_', " "));
            if let Some(v) = value.as_str().filter(|v| v.len() <= 60) {
                parts.push(v.to_string());
            }
        }
    }
    parts.join(" ")
}

/// The answer to a call to a tool that does not exist: the closest real
/// ones, in full, so the next round can call one of them properly.
pub fn not_a_tool(found: &[&ToolSpec]) -> String {
    if found.is_empty() {
        return "there is no tool by that name. Use find_tools to look for one.".into();
    }
    let answer = json!(found
        .iter()
        .map(|s| json!({ "name": s.name, "description": s.description, "parameters": s.input_schema }))
        .collect::<Vec<_>>());
    format!("there is no tool by that name. The closest ones, which you can call by their exact name:\n{answer}")
}

/// What `find_tools` answers: the matches in full, as the model would have
/// seen them up front, and any further candidates by name and purpose.
pub fn answer(found: &[&ToolSpec], related: &[&ToolSpec], query: &str) -> Value {
    if found.is_empty() {
        return json!({
            "found": [],
            "note": format!("No tool matches {query:?}. Try other words, or name the server."),
        });
    }
    let mut answer = json!({
        "found": found.iter().map(|s| json!({
            "name": s.name,
            "description": s.description,
            "parameters": s.input_schema,
        })).collect::<Vec<_>>(),
        "note": "Call any of these by name now, with arguments matching its parameters.",
    });
    if !related.is_empty() {
        answer["also"] = json!(related.iter().map(|s| format!("{}: {}", s.name, one_line(&s.description))).collect::<Vec<_>>());
        answer["note"] = json!(
            "Call any of these by name now, with arguments matching its parameters. The tools under \"also\" \
             can be called too; find_tools gives their parameters."
        );
    }
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, description: &str) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: description.into(),
            input_schema: json!({"type": "object", "properties": {}}),
            output_schema: None,
            effect: ozgent_core::permission::Effect::Unknown,
        }
    }

    #[test]
    fn words_split_names_and_stem() {
        assert_eq!(words("files_read_text_file"), ["fil", "read", "text", "fil"]);
        assert_eq!(words("getCurrentTime in Tokyo"), ["current", "tim", "tokyo"]);
        for (a, b) in [("create", "created"), ("create", "creating"), ("directory", "directories"), ("search", "searches")] {
            assert_eq!(stem(a), stem(b), "{a} / {b}");
        }
    }

    fn rank(pool: &[ToolSpec], query: &str) -> Vec<String> {
        let server = |n: &str| n.split('_').next().map(str::to_string);
        search(pool, server, query, None).into_iter().map(|s| s.name.clone()).collect()
    }

    #[test]
    fn the_right_tool_ranks_first_on_plain_words() {
        let pool = vec![
            spec("time_convert_time", "Convert time between timezones."),
            spec("time_get_current_time", "Get current time in a specific timezone."),
            spec("fetch_fetch_robots", "Fetch and parse the robots.txt for a given origin."),
            spec("files_list_directory", "Get a detailed listing of all files and directories in a specified path."),
            spec("camofox_create_tab", "Create a new browser tab and optionally navigate to a URL."),
        ];
        assert_eq!(rank(&pool, "what are the robots rules of a site")[0], "fetch_fetch_robots");
        assert_eq!(rank(&pool, "list the files in a directory")[0], "files_list_directory");
        assert_eq!(rank(&pool, "open a new browser tab")[0], "camofox_create_tab");
        assert_eq!(rank(&pool, "the current time in Tokyo")[0], "time_get_current_time");
        let server = |n: &str| n.split('_').next().map(str::to_string);
        let only = search(&pool, server, "", Some("time"));
        assert_eq!(only.len(), 2, "an empty query lists one server's tools");
    }

    #[test]
    fn a_query_loses_stamps_links_and_paths() {
        assert_eq!(
            clean_query("[01:46] List the files in /home/me/notes and read https://example.com/a, please"),
            "List the files in and read please"
        );
    }

    #[test]
    fn one_line_keeps_the_first_sentence_within_fourteen_words() {
        assert_eq!(one_line("Get current time. Then more."), "Get current time");
        assert_eq!(one_line("Read v1.2 files.\n  Details."), "Read v1.2 files");
        let long = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen";
        assert!(one_line(long).ends_with("fourteen…"));
    }

    #[test]
    fn missing_parameters_are_answered_with_the_schema() {
        let mut s = spec("files_read_file", "Read a file.");
        s.input_schema = json!({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]});
        let answer = missing_parameters(&s, &json!({})).expect("path is missing");
        assert!(answer["error"].as_str().unwrap().contains("path"));
        assert_eq!(answer["parameters"]["required"][0], "path");
        assert!(missing_parameters(&s, &json!({"path": "/a"})).is_none());
        assert!(missing_parameters(&s, &json!({"path": null})).is_some());
    }

    #[test]
    fn the_directory_shrinks_to_fit() {
        let many: Vec<ToolSpec> = (0..40)
            .flat_map(|s| {
                (0..25).map(move |t| {
                    spec(&format!("srv{s}_tool{t}"), "Does a fairly specific thing to a fairly specific kind of record.")
                })
            })
            .collect();
        fn group(specs: &[ToolSpec]) -> BTreeMap<String, Vec<&ToolSpec>> {
            let mut by: BTreeMap<String, Vec<&ToolSpec>> = BTreeMap::new();
            for s in specs {
                by.entry(s.name.split('_').next().unwrap().to_string()).or_default().push(s);
            }
            by
        }
        let none = |_: &str| String::new();
        let small = directory(&group(&many[..20]), &none);
        assert!(small.contains("srv0_tool3: Does a fairly"), "few tools: each one described");
        let big = directory(&group(&many), &none);
        assert!(big.len() / 4 <= DIRECTORY_TOKENS);
        assert!(big.contains("srv39"), "every server still named: {}", &big[..200]);
        assert!(!big.contains("Does a fairly"), "a thousand tools: no per-tool descriptions");
    }

    #[test]
    fn an_invented_call_searches_by_its_name_and_arguments() {
        let q = guess_as_query("github_create_issue", &json!({"title": "Crash on start", "body": "x".repeat(200)}));
        assert_eq!(q, "github create issue body title Crash on start");
        let pool = vec![
            spec("github_issue_write", "Create or update an issue in a GitHub repository."),
            spec("time_get_current_time", "Get current time in a specific timezone."),
        ];
        assert_eq!(rank(&pool, &q)[0], "github_issue_write");
        assert!(not_a_tool(&[&pool[0]]).contains("github_issue_write"));
    }

    fn named(names: &[(&str, &str)]) -> Vec<ToolSpec> {
        names.iter().map(|(n, d)| spec(n, d)).collect()
    }

    fn by_prefix(n: &str) -> Option<String> {
        n.split('_').next().filter(|p| *p != "web" && *p != "read").map(str::to_string)
    }

    #[test]
    fn a_browser_is_found_by_what_its_tools_do_whatever_they_are_called() {
        let ghost = named(&[
            ("ghostcloak_session_create", "Create a new browsing session: launches the engine with a fresh identity."),
            ("ghostcloak_page_open", "Navigate to a URL in an existing session. Returns a page_id."),
            ("ghostcloak_page_snapshot", "Extract the visible text content of a page as plain text."),
            ("ghostcloak_page_screenshot", "Capture a PNG screenshot of a page."),
            ("ghostcloak_page_click", "Click an element by CSS selector."),
        ]);
        let b = find_browser(&ghost, &by_prefix).expect("a browser");
        assert_eq!(b.tools(), ["ghostcloak_session_create", "ghostcloak_page_open", "ghostcloak_page_snapshot"]);

        let playwright = named(&[
            ("browser_browser_navigate", "Navigate to a URL"),
            ("browser_browser_snapshot", "Capture accessibility snapshot of the current page, this is better than screenshot"),
            ("browser_browser_click", "Perform click on a web page"),
        ]);
        let b = find_browser(&playwright, &by_prefix).expect("a browser");
        assert_eq!((b.open.as_str(), b.read.as_str(), b.start.as_deref()), ("browser_browser_navigate", "browser_browser_snapshot", None));

        let files = named(&[("files_read_file", "Read a file."), ("files_list_directory", "List a directory.")]);
        assert!(find_browser(&files, &by_prefix).is_none());
    }

    #[test]
    fn a_search_tool_is_preferred_and_nothing_is_said_without_a_way_to_the_web() {
        let mut tools = named(&[
            ("ghostcloak_page_open", "Navigate to a URL in an existing session."),
            ("ghostcloak_page_snapshot", "Extract the visible text content of a page as plain text."),
        ]);
        let bridge = web_bridge(&tools, by_prefix).expect("the browser");
        assert!(bridge.hint().contains("duckduckgo"), "{}", bridge.hint());

        tools.push(spec("brave_web_search", "Search the web with the Brave search engine."));
        assert_eq!(web_bridge(&tools, by_prefix), Some(WebBridge::Search("brave_web_search".into())));

        tools.push(spec("web_search", "Search the web."));
        assert!(web_bridge(&tools, by_prefix).is_none(), "ozgent's own web_search needs no note");

        let offline = named(&[("files_read_file", "Read a file.")]);
        assert!(web_bridge(&offline, by_prefix).is_none(), "no way to the web: nothing is claimed");
    }

    #[test]
    fn a_message_asking_for_the_web_is_told_apart_from_one_that_is_not() {
        let pool = named(&[
            ("ghostcloak_page_click", "Click an element by CSS selector."),
            ("files_write_file", "Write text to a file."),
            ("time_get_current_time", "Get the current time in a timezone."),
        ]);
        assert!(wants_web(&pool, by_prefix, "what's the latest news about Nvidia's earnings?"));
        assert!(wants_web(&pool, by_prefix, "search the web for the price of a Raspberry Pi 5"));
        assert!(!wants_web(&pool, by_prefix, "write a haiku about rain"));
        assert!(!wants_web(&pool, by_prefix, "save these notes to a file called ideas.txt"));
    }

    #[test]
    fn a_servers_own_words_are_cut_to_two_sentences() {
        assert_eq!(
            brief("Stealth browser for AI agents. Create a session, open pages, take snapshots. Identities are coherent.", 24),
            "Stealth browser for AI agents. Create a session, open pages, take snapshots"
        );
    }

    #[test]
    fn the_answer_carries_full_descriptions() {
        let s = spec("time_get_current_time", "Get current time.");
        let a = answer(&[&s], &[], "time");
        assert_eq!(a["found"][0]["name"], "time_get_current_time");
        assert!(a["found"][0]["parameters"].is_object());
        assert!(answer(&[], &[], "zzz")["note"].as_str().unwrap().contains("No tool"));
    }
}
