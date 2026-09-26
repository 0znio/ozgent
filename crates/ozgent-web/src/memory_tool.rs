//! `memory`: the model looking things up in what ozgent has kept.
//!
//! A conversation carries its last few messages and, since the tool results
//! of the turn before (see `turn::carried_material`); everything older is
//! still in the database — every message, every page a tool read, every
//! remembered fact — searchable by words and by meaning. This tool is how
//! the model reaches it when what it needs is not in view: before searching
//! the web again for what it read an hour ago, or saying it does not know
//! what the user told it last week.
//!
//! **Related, not only matching.** A search finds what matches, then follows
//! what that points at: the names and numbers the best matches mention are
//! searched for in turn, and a remembered fact leads to the facts about its
//! value (`project → uses → Postgres`, then what is known about Postgres) —
//! a knowledge graph's reach, one hop at a time, answered with short
//! excerpts and ids rather than a graph poured into the prompt. `read` then
//! gives any one in full.
//!
//! **Fast.** One database connection, kept; the query embedded once, on the
//! GPU when the embedding model is there (~35 ms); the second hop and the
//! fact links are full-text lookups, well under a millisecond. Nothing is
//! written on the path of a reply except what `remember` is asked to keep.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use ozgent_core::ToolSpec;
use ozgent_memory::store::Scope as FactScope;
use ozgent_memory::{Embedder, HashingEmbedder, OwnerKind, Retriever, Store, StoredMessage};
use serde_json::{Value, json};

pub const NAME: &str = "memory";

/// Whose memory a turn may search.
#[derive(Debug, Clone, PartialEq)]
pub struct Reach {
    /// The conversation being answered.
    pub conversation: i64,
    /// Whether other conversations are open to it too: for the person at
    /// this machine, not for a chat on a messaging channel or a program
    /// holding an API key, who may see only their own conversation.
    pub everywhere: bool,
}

/// How many results a search gives by default, and at most.
const RESULTS: usize = 6;
const MAX_RESULTS: usize = 12;
/// Characters of each result shown, and of the best one: enough of it to
/// quote from without a `read`.
const SNIPPET: usize = 320;
const TOP_SNIPPET: usize = 800;
/// Characters of one `read`, a page of a long item.
const PAGE: usize = 6000;

/// Where `memory` finds the database: set once when the worker starts.
static ROOT: OnceLock<PathBuf> = OnceLock::new();
static STORE: Mutex<Option<Store>> = Mutex::new(None);
/// The last search in each conversation, so a `read` that names no part
/// can open the part of a long item that search was about.
static LAST_QUERY: Mutex<Option<HashMap<i64, String>>> = Mutex::new(None);

pub fn set_root(root: PathBuf) {
    let _ = ROOT.set(root);
}

pub fn spec(everywhere: bool) -> ToolSpec {
    let mut properties = json!({
        "action": {
            "type": "string",
            "enum": ["search", "read", "remember", "forget"],
            "description": "search: find what is kept about something. read: one item in full, by id. remember: keep a fact. forget: drop a fact."
        },
        "query": { "type": "string", "description": "For search: what to look for, in plain words." },
        "id": { "type": "string", "description": "For read or forget: an id from a search, like m42 or f7." },
        "part": { "type": "integer", "description": "For read: which part of a long item, from 1." },
        "fact": { "type": "string", "description": "For remember: the fact itself, written to stand alone — 'the user prefers Reuters' or 'subject | property | value'." },
        "only_here": { "type": "boolean", "description": "For remember: true if it matters only in this conversation. Default: remembered everywhere." }
    });
    if everywhere {
        properties["everywhere"] = json!({ "type": "boolean", "description": "For search: also look in the user's other conversations." });
    }
    ToolSpec {
        name: NAME.to_string(),
        description: "What this conversation already holds but is not in view: earlier messages, everything \
                      tools returned before (pages read, search results) and remembered facts. Search it \
                      before searching the web again or saying you don't know; it also finds indirectly \
                      related items. read gives one in full; remember keeps a fact for later. Fast."
            .to_string(),
        input_schema: json!({ "type": "object", "properties": properties, "required": ["action"] }),
        output_schema: None,
        effect: ozgent_core::permission::Effect::Read,
    }
}

/// Answer one call. Blocking; runs on the inference thread like `find_tools`.
pub fn call(reach: &Reach, arguments: &Value) -> Result<Value, String> {
    let started = std::time::Instant::now();
    let action = arguments.get("action").and_then(Value::as_str).unwrap_or("search");
    let mut guard = STORE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let root = ROOT.get().ok_or("memory is not available here")?;
        *guard = Some(Store::open(root.join("ozgent.db")).map_err(|e| format!("opening memory: {e}"))?);
    }
    let store = guard.as_ref().expect("just opened");
    let text = |key: &str| arguments.get(key).and_then(Value::as_str).map(str::trim).unwrap_or("").to_string();
    let result = match action {
        "search" => {
            let query = text("query");
            if query.is_empty() {
                return Err("search needs a query".into());
            }
            let limit = arguments.get("limit").and_then(Value::as_u64).map_or(RESULTS, |n| (n as usize).clamp(1, MAX_RESULTS));
            let everywhere = reach.everywhere && arguments.get("everywhere").and_then(Value::as_bool).unwrap_or(false);
            LAST_QUERY.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(reach.conversation, query.clone());
            search(store, reach, &query, limit, everywhere)
        }
        "read" => {
            let about = LAST_QUERY.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&reach.conversation).cloned());
            let part = arguments.get("part").and_then(Value::as_u64).map(|p| p as usize);
            read(store, reach, &text("id"), part, about.as_deref())
        }
        "remember" => {
            // Everywhere by default, for the person at this machine: what they
            // ask to be remembered is about them. A chat on a messaging
            // channel keeps what it is told to itself, so nobody there can
            // put words into the owner's other conversations.
            let only_here = arguments.get("only_here").and_then(Value::as_bool).unwrap_or(false)
                || arguments.get("about_the_user").and_then(Value::as_bool) == Some(false);
            remember(store, reach, &text("fact"), reach.everywhere && !only_here)
        }
        "forget" => forget(store, reach, &text("id")),
        other => Err(format!("no action {other:?}: use search, read, remember or forget")),
    };
    tracing::info!("memory {action}: {} ms", started.elapsed().as_millis());
    result
}

/// Whether a message points back at something from earlier — "the article
/// you read earlier", "what did you find last time", "that report" — and so
/// is worth a memory search before the model answers. A small model rarely
/// thinks to look: asked to quote a page it read two turns before, it
/// fetched the page again.
pub fn points_back(message: &str) -> bool {
    let m = message.to_lowercase();
    const CUES: &[&str] = &[
        "earlier", "previously", "before,", "last time", "the other day", "remember", "recall", "we discussed",
        "we talked", "you said", "you told", "you mentioned", "you found", "you read", "you wrote", "you fetched",
        "you looked", "you searched", "you gave", "you showed", "i told you", "i said", "i mentioned", "above",
        "at the start", "from before",
    ];
    const THAT: &[&str] = &["article", "page", "link", "site", "report", "result", "results", "list", "table", "summary", "search", "email", "file", "news"];
    CUES.iter().any(|c| m.contains(c))
        || ["that ", "those ", "the same "].iter().any(|d| THAT.iter().any(|n| m.contains(&format!("{d}{n}"))))
}

/// A search made for the model, for a message that points back: the found
/// items if there are any, as the tool itself would answer. `None` when
/// nothing is kept that matches.
pub fn recall_for(reach: &Reach, message: &str) -> Option<Value> {
    if !points_back(message) {
        return None;
    }
    let query: String = message.chars().take(300).collect();
    let found = call(reach, &json!({ "action": "search", "query": query, "everywhere": false })).ok()?;
    found.get("results").and_then(Value::as_array).filter(|r| !r.is_empty())?;
    Some(found)
}

fn embedder() -> Box<dyn Embedder> {
    match crate::worker::tool_embedder() {
        Some(worker) => Box::new(crate::memory::MemoryEmbedder::new(worker.clone())),
        None => Box::new(HashingEmbedder::default()),
    }
}

/// One thing found, before it is shown.
struct Found {
    key: String,
    score: f32,
    /// How it was reached: the query itself, or through something that did.
    via: &'static str,
}

fn search(store: &Store, reach: &Reach, query: &str, limit: usize, everywhere: bool) -> Result<Value, String> {
    let embedder = embedder();
    let mut found: HashMap<String, Found> = HashMap::new();
    let mut add = |key: String, score: f32, via: &'static str| {
        let entry = found.entry(key.clone()).or_insert(Found { key, score: 0.0, via });
        if score > entry.score {
            entry.score = score;
            entry.via = if entry.via == "direct" { "direct" } else { via };
        }
    };

    // Hop one: words and meaning together, over this conversation's messages
    // (tool results among them) and the facts that apply here.
    let first = Retriever::new(store, embedder.as_ref())
        .search(reach.conversation, query, 24)
        .map_err(|e| e.to_string())?;
    let top: f32 = first.first().map_or(1.0, |h| h.score.max(f32::EPSILON));
    let mut seeds: Vec<String> = Vec::new();
    let asked = terms(query);
    // The retriever ranks facts and messages each on their own and merges
    // the ranks, so with one fact on file that fact came first of the facts
    // — and as high as the best message — whatever it said: "the user
    // prefers Reuters" was handed back for a question about a Raspberry Pi.
    // A fact must share a word with the question to count as a match here;
    // facts reached through what matched are added below.
    let first: Vec<_> = first
        .into_iter()
        .filter(|h| h.kind != OwnerKind::Fact || terms(&h.text).iter().any(|w| asked.contains(w)))
        .collect();
    for (rank, hit) in first.iter().enumerate() {
        let key = match hit.kind {
            OwnerKind::Fact => format!("f{}", hit.id),
            _ => format!("m{}", hit.id),
        };
        add(key, hit.score / top, "direct");
        if rank < 3 {
            seeds.push(hit.text.clone());
        }
    }

    // Hop two: what the best matches are about — the names, figures and
    // repeated words they share that the query did not say — searched for in
    // turn. Full text only, so it costs no second embedding.
    let related = related_terms(&seeds, &asked);
    if !related.is_empty() {
        let expanded = format!("{} {}", query, related.join(" "));
        let second = store.search_in_conversation(reach.conversation, &expanded, 12).map_err(|e| e.to_string())?;
        for (rank, m) in second.iter().enumerate() {
            add(format!("m{}", m.id), 0.5 / (1.0 + rank as f32 * 0.25), "related");
        }
    }

    // Facts as a graph: those naming what was asked or found, then the facts
    // about what those point at.
    let facts = store.facts_for(reach.conversation).map_err(|e| e.to_string())?;
    let wanted: HashSet<String> = asked.iter().chain(related.iter()).cloned().collect();
    let mut linked: Vec<String> = Vec::new();
    for f in &facts {
        let words: HashSet<String> = terms(&f.text).into_iter().collect();
        if words.iter().any(|w| wanted.contains(w)) {
            add(format!("f{}", f.id), 0.6, "related");
            if let Some(t) = &f.triple {
                linked.push(t.value.to_lowercase());
            }
        }
    }
    for f in &facts {
        if let Some(t) = &f.triple {
            if linked.iter().any(|v| t.subject.to_lowercase() == *v) {
                add(format!("f{}", f.id), 0.45, "linked");
            }
        }
    }

    // Other conversations, by words, for the person at this machine only.
    if everywhere {
        let hits = store.search_messages_in(query, 10, true).map_err(|e| e.to_string())?;
        for (rank, h) in hits.iter().filter(|h| h.conversation_id != reach.conversation).enumerate() {
            add(format!("m{}", h.message_id), 0.55 / (1.0 + rank as f32 * 0.2), "another conversation");
        }
    }

    // Not the question being answered: it matches its own words best of all,
    // and a model handed it back read its own question as the answer.
    let asking: Option<String> = store
        .recent_messages(reach.conversation, 1)
        .ok()
        .and_then(|m| m.into_iter().find(|m| m.role == "user"))
        .map(|m| format!("m{}", m.id));
    let mut ranked: Vec<Found> = found.into_values().filter(|f| Some(&f.key) != asking.as_ref()).collect();
    ranked.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.key.cmp(&b.key)));
    ranked.truncate(limit);
    // The question's own words, in its order, so a phrase from it can be
    // found as it was written; then what the matches were about.
    let focus: Vec<String> = focus_words(query).into_iter().chain(related.iter().cloned()).collect();
    let results: Vec<Value> = ranked
        .iter()
        .enumerate()
        .filter_map(|(i, f)| describe(store, reach, &f.key, f.via, &focus, if i == 0 { TOP_SNIPPET } else { SNIPPET }))
        .collect();
    if results.is_empty() {
        return Ok(json!({ "results": [], "note": format!("Nothing kept matches {query:?}.") }));
    }
    Ok(json!({
        "results": results,
        "note": "Excerpts. read an id for the whole item."
    }))
}

/// A result as the model sees it: what it is, where it came from, when,
/// and the part of it about the question.
fn describe(store: &Store, reach: &Reach, key: &str, via: &str, focus: &[String], width: usize) -> Option<Value> {
    let (kind, id) = key.split_at(1);
    let id: i64 = id.parse().ok()?;
    let mut out = if kind == "f" {
        let f = store.get_fact(id).ok()??;
        json!({
            "id": key,
            "kind": if f.scope == FactScope::User { "fact about the user" } else { "fact" },
            "text": clip(&f.text, SNIPPET),
        })
    } else {
        let m = store.get_message(id).ok()??;
        let mut v = json!({
            "id": key,
            "kind": kind_of(&m),
            "when": when(m.created_at),
            "text": snippet(&m.content, focus, width),
        });
        if let Some(from) = tool_label(&m) {
            v["from"] = json!(from);
        }
        if m.conversation_id != reach.conversation {
            let title = store.get_conversation(m.conversation_id).ok().flatten().map(|c| c.title).unwrap_or_default();
            v["conversation"] = json!(title);
        }
        v
    };
    if via != "direct" {
        out["found"] = json!(via);
    }
    Some(out)
}

fn read(store: &Store, reach: &Reach, id: &str, part: Option<usize>, about: Option<&str>) -> Result<Value, String> {
    let (kind, number) = id.split_at(id.len().min(1));
    let number: i64 = number.parse().map_err(|_| format!("{id:?} is not an id from a search (like m42 or f7)"))?;
    if kind == "f" {
        let f = store.get_fact(number).map_err(|e| e.to_string())?.ok_or("no such fact")?;
        if f.scope != FactScope::User && f.conversation_id != Some(reach.conversation) && !reach.everywhere {
            return Err("that fact belongs to another conversation".into());
        }
        return Ok(json!({ "id": id, "kind": "fact", "text": f.text }));
    }
    let m = store.get_message(number).map_err(|e| e.to_string())?.ok_or("no such item")?;
    if m.conversation_id != reach.conversation && !reach.everywhere {
        return Err("that item belongs to another conversation".into());
    }
    let chars: Vec<char> = m.content.chars().collect();
    let parts = chars.len().div_ceil(PAGE).max(1);
    // No part named: the one the last search was about — the Raspberry Pi 5
    // was 8,600 characters into a page whose first part a model read and
    // concluded it was not mentioned.
    let part = match (part, about) {
        (Some(p), _) => p,
        // The part holding the place in the whole text that answers it best —
        // not the part with the most mentions, which for a long article is
        // its references.
        (None, Some(q)) if parts > 1 => {
            let whole: String = chars.iter().collect();
            best_anchor(&whole, &focus_words(q), SNIPPET).map_or(1, |at| at / PAGE + 1)
        }
        _ => 1,
    }
    .clamp(1, parts);
    let text: String = chars[(part - 1) * PAGE..(part * PAGE).min(chars.len())].iter().collect();
    let mut out = json!({ "id": id, "kind": kind_of(&m), "when": when(m.created_at), "text": text });
    if let Some(from) = tool_label(&m) {
        out["from"] = json!(from);
    }
    if parts > 1 {
        out["part"] = json!(format!("{part} of {parts}"));
        out["note"] = json!("A part of a longer item; read it again with another part for the rest.");
    }
    Ok(out)
}

fn remember(store: &Store, reach: &Reach, fact: &str, about_the_user: bool) -> Result<Value, String> {
    let fact = standalone(fact);
    if fact.is_empty() {
        return Err("remember needs a fact".into());
    }
    let mut candidates = ozgent_memory::parse_extraction(&fact);
    if candidates.is_empty() {
        return Err("that does not read as a fact to keep".into());
    }
    for c in &mut candidates {
        c.scope = if about_the_user { FactScope::User } else { FactScope::Conversation };
    }
    let ids = ozgent_memory::store_candidates(store, reach.conversation, None, &candidates).map_err(|e| e.to_string())?;
    // Embedded now, a few milliseconds a fact, so it is found by meaning as
    // well as by its words from the next message on.
    let embedder = embedder();
    for (id, c) in ids.iter().zip(&candidates) {
        let v = embedder.embed(&c.text);
        if v.iter().any(|x| *x != 0.0) {
            let _ = store.put_embedding(OwnerKind::Fact, *id, &v);
        }
    }
    Ok(match ids.as_slice() {
        [] => json!({ "kept": false, "note": "Already known." }),
        ids => json!({ "kept": true, "ids": ids.iter().map(|i| format!("f{i}")).collect::<Vec<_>>() }),
    })
}

/// A fact as it should be kept: without the time stamp ozgent puts on
/// messages, without "remember that", and about "the user" rather than "I".
/// Models pass the request through as it was typed often enough —
/// "[14:09] Remember that I prefer Reuters" — that it is fixed here.
fn standalone(fact: &str) -> String {
    let mut text = fact.trim();
    while let Some(rest) = text.strip_prefix('[').and_then(|r| r.split_once(']')).map(|(_, r)| r.trim_start()) {
        text = rest;
    }
    let lower = text.to_lowercase();
    for lead in ["please remember that ", "please remember ", "remember that ", "remember: ", "remember ", "note that "] {
        if lower.starts_with(lead) {
            text = text[lead.len()..].trim_start();
            break;
        }
    }
    let mut out = text.trim_end_matches(['.', '!']).to_string();
    for (from, to) in [("I am ", "The user is "), ("I'm ", "The user is "), ("I ", "The user "), ("My ", "The user's ")] {
        if let Some(rest) = out.strip_prefix(from) {
            let rest = rest.to_string();
            // "I prefer" → "The user prefers": the verb agrees with its new subject.
            let rest = if to == "The user " {
                match rest.split_once(' ') {
                    Some((verb, tail)) if !verb.ends_with('s') && verb.chars().all(char::is_alphabetic) => format!("{verb}s {tail}"),
                    _ => rest,
                }
            } else {
                rest
            };
            out = format!("{to}{rest}");
            break;
        }
    }
    out
}

fn forget(store: &Store, reach: &Reach, id: &str) -> Result<Value, String> {
    let number: i64 = id
        .strip_prefix('f')
        .and_then(|n| n.parse().ok())
        .ok_or("only facts can be forgotten; pass an id like f7")?;
    let f = store.get_fact(number).map_err(|e| e.to_string())?.ok_or("no such fact")?;
    if f.conversation_id != Some(reach.conversation) && !(f.scope == FactScope::User && reach.everywhere) {
        return Err("that fact is not this conversation's to forget".into());
    }
    store.delete_fact(number).map_err(|e| e.to_string())?;
    Ok(json!({ "forgotten": id }))
}

fn kind_of(m: &StoredMessage) -> &'static str {
    match m.role.as_str() {
        "tool" => "tool result",
        "assistant" => "your earlier reply",
        "user" => "the user's message",
        _ => "message",
    }
}

/// `ghostcloak_page_snapshot https://…`: which tool, and what it was given
/// that says most about the result.
fn tool_label(m: &StoredMessage) -> Option<String> {
    let call: Value = serde_json::from_str(m.tool_calls.as_deref()?).ok()?;
    let name = call.get("name")?.as_str()?.to_string();
    let args = call.get("arguments").and_then(Value::as_object);
    let lead = args.and_then(|a| {
        ["url", "query", "path", "command", "symbol", "q"]
            .iter()
            .find_map(|k| a.get(*k).and_then(Value::as_str).map(str::to_string))
    });
    Some(match lead {
        Some(lead) => format!("{name} {}", clip(&lead, 120)),
        None => name,
    })
}

fn when(unix: i64) -> String {
    chrono_like(unix)
}

/// `2026-09-26 11:02 UTC`, without a date library.
fn chrono_like(unix: i64) -> String {
    let days = unix.div_euclid(86_400);
    let secs = unix.rem_euclid(86_400);
    // Civil from days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", secs / 3600, (secs % 3600) / 60)
}

const STOP: &[&str] = &[
    "the", "and", "for", "with", "from", "into", "this", "that", "what", "which", "when", "where", "who",
    "about", "have", "has", "had", "was", "were", "are", "you", "your", "can", "could", "would", "should",
    "will", "there", "their", "them", "they", "then", "than", "also", "just", "more", "most", "some", "any",
    "all", "not", "but", "our", "out", "its", "his", "her", "she", "him", "how", "why", "did", "does",
    "get", "got", "tell", "said", "say", "been", "being", "over", "under", "after", "before", "these",
    "those", "such", "very", "much", "many", "other", "only", "each", "here", "like", "make", "made",
];

/// The words of a text worth searching by, lowercased and lightly stemmed,
/// so "prefer" meets "prefers" and "released" meets "release".
fn terms(text: &str) -> Vec<String> {
    let mut out: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2 || w.chars().any(|c| c.is_ascii_digit()))
        .map(str::to_lowercase)
        .filter(|w| !STOP.contains(&w.as_str()))
        .map(|w| stem(&w))
        .collect();
    out.sort();
    out.dedup();
    out
}

fn stem(w: &str) -> String {
    if w.chars().any(|c| c.is_ascii_digit()) || w.len() <= 4 {
        return w.to_string();
    }
    for suffix in ["ing", "ies", "ed", "es", "s"] {
        if let Some(base) = w.strip_suffix(suffix) {
            if base.len() >= 3 {
                return if suffix == "ies" { format!("{base}y") } else { base.to_string() };
            }
        }
    }
    w.to_string()
}

/// What the best matches are about that the query did not say: capitalised
/// names and figures first, then words they repeat. At most six.
fn related_terms(seeds: &[String], asked: &[String]) -> Vec<String> {
    let mut score: HashMap<String, (usize, bool)> = HashMap::new();
    for text in seeds {
        let head: String = text.chars().take(4000).collect();
        let mut first = true;
        for raw in head.split(|c: char| !c.is_alphanumeric()) {
            if raw.len() < 3 && !raw.chars().any(|c| c.is_ascii_digit()) {
                continue;
            }
            let lower = raw.to_lowercase();
            if STOP.contains(&lower.as_str()) || asked.contains(&lower) {
                continue;
            }
            let name = raw.chars().next().is_some_and(char::is_uppercase) && !first
                || raw.chars().any(|c| c.is_ascii_digit()) && raw.len() >= 3;
            let entry = score.entry(lower).or_insert((0, false));
            entry.0 += 1;
            entry.1 |= name;
            first = false;
        }
    }
    let mut ranked: Vec<(String, usize, bool)> =
        score.into_iter().filter(|(_, (n, name))| *name || *n >= 3).map(|(w, (n, name))| (w, n, name)).collect();
    ranked.sort_by(|a, b| (b.2, b.1).cmp(&(a.2, a.1)).then(a.0.cmp(&b.0)));
    ranked.into_iter().take(6).map(|(w, _, _)| w).collect()
}

/// The part of `text` about `focus`, `width` characters of it.
///
/// The window with the most weight of matches, where a word counts for less
/// the more often the text repeats it — "raspberry" is on every line of an
/// article about the Raspberry Pi and says nothing about which line is about
/// the Pi 5 — and a pair of words from the question, in order, counts most.
fn snippet(text: &str, focus: &[String], width: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = flat.chars().collect();
    if chars.len() <= width {
        return flat;
    }
    let start = best_anchor(&flat, focus, width).map_or(0, |at| at.saturating_sub(width / 2));
    let end = (start + width).min(chars.len());
    let start = end.saturating_sub(width);
    let body: String = chars[start..end].iter().collect();
    format!("{}{}{}", if start > 0 { "…" } else { "" }, body.trim(), if end < chars.len() { "…" } else { "" })
}

/// Where in `text` (a character position) the question is best answered:
/// the match with the most weight within `width` around it. `None` when
/// nothing in it matches.
fn best_anchor(text: &str, focus: &[String], width: usize) -> Option<usize> {
    let lower: String = text.to_lowercase();
    let mut hits: Vec<(usize, f32)> = Vec::new();
    let mut find = |needle: &str, weight_of: &dyn Fn(usize) -> f32| {
        if needle.is_empty() {
            return;
        }
        let at: Vec<usize> = lower.match_indices(needle).map(|(b, _)| b).take(400).collect();
        let w = weight_of(at.len());
        for b in at {
            hits.push((lower[..b].chars().count(), w));
        }
    };
    for word in focus.iter().take(16) {
        find(word, &|n| 1.0 / (1.0 + n as f32));
    }
    let words: Vec<&str> = focus.iter().map(String::as_str).collect();
    for pair in words.windows(2) {
        find(&format!("{} {}", pair[0], pair[1]), &|n| 3.0 / (1.0 + n as f32));
    }
    if hits.is_empty() {
        return None;
    }
    hits.sort_by_key(|h| h.0);
    let half = width / 2;
    let mut best = (0.0f32, hits[0].0);
    let (mut lo, mut hi, mut sum) = (0, 0, 0.0f32);
    for &(h, _) in &hits {
        while hi < hits.len() && hits[hi].0 <= h + half {
            sum += hits[hi].1;
            hi += 1;
        }
        while hits[lo].0 + half < h {
            sum -= hits[lo].1;
            lo += 1;
        }
        if sum > best.0 {
            best = (sum, h);
        }
    }
    Some(best.1)
}

/// The words of a question as written, lowercased, for finding where a text
/// answers it: phrases have to meet the text as it is.
fn focus_words(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !STOP.contains(w) && (w.len() > 1 || w.chars().all(|c| c.is_ascii_digit())))
        .map(str::to_string)
        .collect()
}

fn clip(text: &str, width: usize) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(width) {
        Some((at, _)) => format!("{}…", &flat[..at]),
        None => flat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(messages: &[(&str, &str, Option<&str>)]) -> (Store, i64, Vec<i64>) {
        let store = Store::open_in_memory().unwrap();
        let conversation = store.create_conversation("test", None).unwrap();
        let ids = messages
            .iter()
            .map(|(role, content, call)| store.append_message_full(conversation, role, content, None, *call, None, 0).unwrap())
            .collect();
        (store, conversation, ids)
    }

    #[test]
    fn a_search_finds_what_a_tool_read_and_says_where_it_came_from() {
        let (store, c, _) = store_with(&[
            ("user", "get the latest news from bloomberg and reuters", None),
            ("tool", "Reuters World: Xi Jinping meets Donald Trump in Seoul; tariffs on chips paused for 90 days.",
             Some(r#"{"name":"ghostcloak_page_snapshot","arguments":{"url":"https://www.reuters.com/world"}}"#)),
            ("assistant", "Here are the headlines.", None),
        ]);
        let reach = Reach { conversation: c, everywhere: false };
        let out = search(&store, &reach, "tariffs on chips", 6, false).unwrap();
        let first = &out["results"][0];
        assert_eq!(first["kind"], "tool result", "{out}");
        assert!(first["from"].as_str().unwrap().contains("reuters.com"), "{out}");
        assert!(first["text"].as_str().unwrap().contains("tariffs"), "{out}");
    }

    #[test]
    fn a_search_reaches_what_its_matches_point_at() {
        let (store, c, _) = store_with(&[
            ("user", "which database does Orion use?", None),
            ("assistant", "Orion stores everything in Postgres 16 on the Hetzner box.", None),
            ("user", "unrelated chatter about lunch", None),
            ("assistant", "Postgres 16 needs the new pg_hba format after the upgrade; Hetzner charges extra for backups.", None),
        ]);
        let reach = Reach { conversation: c, everywhere: false };
        let out = search(&store, &reach, "Orion", 6, false).unwrap();
        let texts: Vec<&str> = out["results"].as_array().unwrap().iter().filter_map(|r| r["text"].as_str()).collect();
        assert!(texts.iter().any(|t| t.contains("pg_hba")), "reached through Postgres/Hetzner: {out}");
    }

    #[test]
    fn remembered_facts_link_to_the_facts_about_their_values() {
        let (store, c, _) = store_with(&[("user", "hello", None)]);
        let reach = Reach { conversation: c, everywhere: false };
        remember(&store, &reach, "Orion | database | Postgres", false).unwrap();
        remember(&store, &reach, "Postgres | backup schedule | nightly at 02:00", false).unwrap();
        let out = search(&store, &reach, "what database does Orion use", 6, false).unwrap();
        let texts: Vec<String> = out["results"].as_array().unwrap().iter().filter_map(|r| r["text"].as_str().map(str::to_string)).collect();
        assert!(texts.iter().any(|t| t.contains("nightly")), "one hop through Postgres: {out}");
    }

    #[test]
    fn a_long_item_is_read_in_parts_and_other_conversations_stay_shut() {
        let long = "word ".repeat(3000);
        let (store, c, ids) = store_with(&[("tool", &long, Some(r#"{"name":"fetch_url","arguments":{"url":"https://x.test"}}"#))]);
        let other = store.create_conversation("other", None).unwrap();
        let theirs = store.append_message(other, "user", "private", 0).unwrap();
        let reach = Reach { conversation: c, everywhere: false };
        let id = format!("m{}", ids[0]);
        let first = read(&store, &reach, &id, Some(1), None).unwrap();
        assert_eq!(first["part"], "1 of 3");
        assert!(read(&store, &reach, &format!("m{theirs}"), None, None).is_err(), "not this conversation's");
    }

    #[test]
    fn a_read_opens_the_part_the_search_was_about() {
        let page = format!("{}The Raspberry Pi 5 was released in October 2023.{}", "filler text ".repeat(800), " more".repeat(800));
        let (store, c, ids) = store_with(&[("tool", &page, Some(r#"{"name":"fetch_url","arguments":{"url":"https://w.test"}}"#))]);
        let reach = Reach { conversation: c, everywhere: false };
        let out = read(&store, &reach, &format!("m{}", ids[0]), None, Some("Raspberry Pi 5 release")).unwrap();
        assert!(out["text"].as_str().unwrap().contains("October 2023"), "{}", out["part"]);
        assert_ne!(out["part"], "1 of 3");
    }

    #[test]
    fn the_question_being_answered_is_not_a_result() {
        let (store, c, _) = store_with(&[
            ("tool", "Raspberry Pi 5 released October 2023", Some(r#"{"name":"fetch_url","arguments":{}}"#)),
            ("assistant", "ok", None),
            ("user", "what did the article say about the Raspberry Pi 5?", None),
        ]);
        let out = search(&store, &Reach { conversation: c, everywhere: false }, "what did the article say about the Raspberry Pi 5?", 6, false).unwrap();
        assert!(out["results"].as_array().unwrap().iter().all(|r| r["kind"] != "the user's message"), "{out}");
    }

    #[test]
    fn an_unrelated_fact_is_not_a_match_just_for_being_the_only_one() {
        let (store, c, _) = store_with(&[("tool", "The Raspberry Pi 5 was released in October 2023.", Some(r#"{"name":"fetch_url","arguments":{}}"#))]);
        let reach = Reach { conversation: c, everywhere: true };
        remember(&store, &reach, "The user prefers Reuters over Bloomberg", true).unwrap();
        let out = search(&store, &reach, "what does the article say about the Raspberry Pi 5", 6, false).unwrap();
        let kinds: Vec<&str> = out["results"].as_array().unwrap().iter().filter_map(|r| r["kind"].as_str()).collect();
        assert_eq!(kinds.first(), Some(&"tool result"), "{out}");
        assert!(!out.to_string().contains("Reuters"), "{out}");
    }

    #[test]
    fn an_excerpt_is_taken_from_where_the_question_is_answered() {
        let page = format!(
            "{} The Raspberry Pi 5 (2023) features a 2.4 GHz quad-core Cortex-A76 CPU. {}",
            "The Raspberry Pi 2 and the Raspberry Pi Zero were earlier Raspberry Pi boards. ".repeat(30),
            "The Raspberry Pi Foundation sells Raspberry Pi computers. ".repeat(30)
        );
        let focus: Vec<String> = ["raspberry", "pi", "5"].iter().map(|s| s.to_string()).collect();
        let cut = snippet(&page, &focus, 320);
        assert!(cut.contains("Pi 5 (2023)"), "{cut}");
    }

    #[test]
    fn a_fact_is_kept_standing_alone() {
        assert_eq!(standalone("[14:09] Remember that I prefer Reuters over Bloomberg as a news source."), "The user prefers Reuters over Bloomberg as a news source");
        assert_eq!(standalone("my editor is Helix"), "my editor is Helix", "only a leading capital My is rewritten");
        assert_eq!(standalone("My editor is Helix"), "The user's editor is Helix");
        assert_eq!(standalone("Orion | database | Postgres"), "Orion | database | Postgres");
    }

    #[test]
    fn what_the_owner_asks_to_remember_is_known_in_other_conversations() {
        let (store, c, _) = store_with(&[("user", "hi", None)]);
        let reach = Reach { conversation: c, everywhere: true };
        remember(&store, &reach, "I prefer Reuters over Bloomberg", true).unwrap();
        let other = store.create_conversation("new", None).unwrap();
        let there = Reach { conversation: other, everywhere: true };
        let out = search(&store, &there, "which news source do I prefer", 6, false).unwrap();
        assert!(out["results"][0]["text"].as_str().unwrap().contains("Reuters"), "{out}");
    }

    #[test]
    fn a_message_pointing_back_is_told_apart_from_one_that_does_not() {
        for yes in ["In that Wikipedia article you read earlier, what does it say?", "what did you find last time?", "summarise that report", "Remember what I said about Postgres?"] {
            assert!(points_back(yes), "{yes}");
        }
        for no in ["Write a haiku about rain.", "What is 17 times 23?", "Read https://example.com", "get the latest news from reuters"] {
            assert!(!points_back(no), "{no}");
        }
    }

    #[test]
    fn a_date_is_written_plainly() {
        assert_eq!(chrono_like(1_790_402_520), "2026-09-26 06:02 UTC");
        assert_eq!(chrono_like(951_782_400), "2000-02-29 00:00 UTC", "a leap day");
    }
}
