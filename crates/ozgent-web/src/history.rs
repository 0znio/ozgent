//! Searching, rewinding and exporting conversations.
//!
//! The three things any chat application has that ozgent did not. They are
//! grouped because they are one idea seen from three sides: a conversation is
//! a durable, addressable document, not a stream that scrolls away.
//!
//! * **Search** goes across conversations, not within one. The question people
//!   actually have is "where did I talk about the deploy script", and nobody
//!   remembers which thread it was in.
//! * **Rewind** is what "regenerate this reply" and "edit my message and send
//!   it again" both are underneath: drop everything from a point and carry on.
//!   One operation rather than two, so the two cannot disagree about what
//!   happens to facts learned from the messages that go.
//! * **Export** writes the conversation out as Markdown, because a local-first
//!   program should never be the only thing that can read your data.

use axum::extract::{Path, Query, State as AxumState};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::api::ApiError;
use crate::state::State;

/// How many search results to return when the caller does not say.
const RESULTS: i64 = 30;
/// The most any caller may ask for, so a typo cannot ask for a million rows.
const MAX_RESULTS: i64 = 200;
/// How much of a matching message to show around the match.
const SNIPPET: usize = 240;

pub fn router(state: State) -> Router {
    Router::new()
        .route("/api/search", get(search))
        .route("/api/conversations/{id}/rewind", post(rewind))
        .route("/api/conversations/{id}/export", get(export))
        .with_state(state)
}

// ---------------------------------------------------------------- search

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<i64>,
}

#[derive(Serialize)]
struct Hit {
    conversation: i64,
    uuid: String,
    title: String,
    seq: i64,
    role: String,
    /// The matching text, shortened around the match.
    snippet: String,
    created_at: i64,
}

async fn search(
    AxumState(state): AxumState<State>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<Hit>>, ApiError> {
    let limit = query.limit.unwrap_or(RESULTS).clamp(1, MAX_RESULTS);
    let store = state.store.lock().unwrap();
    let hits = store
        .search_messages(&query.q, limit)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .into_iter()
        .map(|h| Hit {
            snippet: snippet(&h.content, &query.q),
            conversation: h.conversation_id,
            uuid: h.conversation_uuid,
            title: h.title,
            seq: h.seq,
            role: h.role,
            created_at: h.created_at,
        })
        .collect();
    Ok(Json(hits))
}

/// A window of the message around the first word of the query that appears.
///
/// Showing the first 240 characters would be useless for a match in a long
/// message — the whole point of a result is seeing *why* it matched.
fn snippet(text: &str, query: &str) -> String {
    let text = text.trim();
    if text.chars().count() <= SNIPPET {
        return text.to_string();
    }
    let lower = text.to_lowercase();
    let at = query
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .find_map(|word| lower.find(&word.to_lowercase()))
        .unwrap_or(0);

    // Counted in characters, not bytes: a byte window would split a multi-byte
    // character and produce invalid text.
    let chars: Vec<char> = text.chars().collect();
    let at = text[..at].chars().count();
    let start = at.saturating_sub(SNIPPET / 3);
    let end = (start + SNIPPET).min(chars.len());
    let body: String = chars[start..end].iter().collect();

    let mut out = String::new();
    if start > 0 {
        out.push('…');
    }
    out.push_str(body.trim());
    if end < chars.len() {
        out.push('…');
    }
    out
}

// ---------------------------------------------------------------- rewind

#[derive(Deserialize)]
struct Rewind {
    /// The first message to drop. Everything from here on goes.
    seq: i64,
}

#[derive(Serialize)]
struct Rewound {
    removed: usize,
    /// The text of the user message that was dropped, when the rewind started
    /// at one. That is what "edit and resend" puts back in the composer, and
    /// finding it here means the page does not have to hold it.
    message: Option<String>,
}

async fn rewind(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
    Json(body): Json<Rewind>,
) -> Result<Json<Rewound>, ApiError> {
    if body.seq < 0 {
        return Err(ApiError::bad_request("a sequence number cannot be negative"));
    }
    let store = state.store.lock().unwrap();
    if store
        .get_conversation(id)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .is_none()
    {
        return Err(ApiError::not_found(format!("no conversation {id}")));
    }
    // Read before the delete: afterwards there is nothing left to read.
    let message = store
        .messages(id)
        .map_err(|e| ApiError::internal(e.to_string()))?
        .into_iter()
        .find(|m| m.seq == body.seq && m.role == "user")
        .map(|m| m.content);

    let removed = store
        .truncate_conversation(id, body.seq)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(Rewound { removed, message }))
}

// ---------------------------------------------------------------- export

async fn export(
    AxumState(state): AxumState<State>,
    Path(id): Path<i64>,
) -> Result<impl IntoResponse, ApiError> {
    let (title, markdown) = {
        let store = state.store.lock().unwrap();
        let conversation = store
            .get_conversation(id)
            .map_err(|e| ApiError::internal(e.to_string()))?
            .ok_or_else(|| ApiError::not_found(format!("no conversation {id}")))?;
        let messages = store.messages(id).map_err(|e| ApiError::internal(e.to_string()))?;
        (conversation.title.clone(), to_markdown(&conversation, &messages))
    };

    Ok((
        [
            ("content-type", "text/markdown; charset=utf-8".to_string()),
            (
                "content-disposition",
                format!("attachment; filename=\"{}.md\"", file_name(&title)),
            ),
        ],
        markdown,
    ))
}

/// A conversation as a Markdown document.
///
/// Written to be read by a person and by anything that reads Markdown, which
/// rules out a JSON dump. Tool calls are listed rather than expanded: the
/// arguments to a search are rarely what anyone wants from a transcript, but
/// knowing a search happened explains where an answer came from.
pub fn to_markdown(
    conversation: &ozgent_memory::Conversation,
    messages: &[ozgent_memory::StoredMessage],
) -> String {
    let mut out = String::new();
    let title = conversation.title.trim();
    out.push_str(&format!("# {}\n\n", if title.is_empty() { "Conversation" } else { title }));

    let started = ozgent_core::DateTime::from_unix(conversation.created_at);
    out.push_str(&format!("*{} — {} message", started.iso_date(), messages.len()));
    if messages.len() != 1 {
        out.push('s');
    }
    if let Some(model) = &conversation.model {
        out.push_str(&format!(", {model}"));
    }
    out.push_str("*\n");

    for message in messages {
        // A tool result is the machinery of a turn, not part of the
        // conversation; the call it answers is already summarised below.
        if message.role == "tool" {
            continue;
        }
        let who = match message.role.as_str() {
            "user" => "You",
            "assistant" => "ozgent",
            "system" => "System",
            other => other,
        };
        out.push_str(&format!("\n## {who}\n\n"));

        if let Some(calls) = message.tool_calls.as_deref().and_then(named_tools) {
            if !calls.is_empty() {
                out.push_str(&format!("*Used {}*\n\n", calls.join(", ")));
            }
        }
        let text = message.content.trim();
        out.push_str(if text.is_empty() { "*(no text)*" } else { text });
        out.push('\n');
    }
    out
}

/// The tools a stored turn used, named, in the order they were called.
fn named_tools(raw: &str) -> Option<Vec<String>> {
    let activity: Vec<serde_json::Value> = serde_json::from_str(raw).ok()?;
    let mut names: Vec<String> = activity
        .iter()
        .filter(|a| a["kind"] == "tool")
        .filter_map(|a| a["name"].as_str().map(str::to_string))
        .collect();
    names.dedup();
    Some(names)
}

/// A title as a file name: no separators, no surprises, never empty.
fn file_name(title: &str) -> String {
    let mut cleaned = String::new();
    for c in title.trim().chars() {
        let c = if c.is_alphanumeric() { c } else { '-' };
        // Runs collapse, so "Q3 report: draft" is not "Q3-report--draft".
        if c == '-' && cleaned.ends_with('-') {
            continue;
        }
        cleaned.push(c);
        if cleaned.chars().count() >= 60 {
            break;
        }
    }
    let cleaned = cleaned.trim_matches('-').to_string();
    if cleaned.is_empty() { "conversation".into() } else { cleaned }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ozgent_memory::{Conversation, StoredMessage};

    fn message(seq: i64, role: &str, text: &str) -> StoredMessage {
        StoredMessage {
            id: seq,
            conversation_id: 1,
            seq,
            role: role.into(),
            content: text.into(),
            thinking: None,
            tool_calls: None,
            tool_call_id: None,
            media: None,
            tokens: 0,
            created_at: 1_755_648_000,
        }
    }

    fn conversation(title: &str) -> Conversation {
        Conversation {
            id: 1,
            uuid: "u".into(),
            title: title.into(),
            model: Some("gemma4:12b".into()),
            created_at: 1_755_648_000,
            updated_at: 1_755_648_000,
            message_count: 2,
        }
    }

    // ------------------------------------------------------- snippets

    #[test]
    fn a_short_message_is_shown_whole() {
        assert_eq!(snippet("the deploy script", "deploy"), "the deploy script");
    }

    #[test]
    fn a_long_message_is_shown_around_the_match_not_from_the_start() {
        // The whole point of a result is seeing why it matched.
        let text = format!("{} the deploy script broke {}", "x ".repeat(300), "y ".repeat(300));
        let out = snippet(&text, "deploy script");
        assert!(out.contains("deploy script"), "{out}");
        assert!(out.chars().count() <= SNIPPET + 2, "{} chars", out.chars().count());
        assert!(out.starts_with('…') && out.ends_with('…'), "{out}");
    }

    #[test]
    fn a_snippet_never_splits_a_character() {
        // A byte window would cut a multi-byte character in half and produce
        // text no browser can render.
        let text = "日本語のテキスト".repeat(80);
        let out = snippet(&text, "テキスト");
        assert!(out.chars().count() <= SNIPPET + 2);
        assert!(!out.is_empty());
    }

    #[test]
    fn a_query_that_is_not_in_the_text_still_produces_a_snippet() {
        // FTS stems, so "deploying" can match "deploy" and the literal word
        // may genuinely not be there.
        let text = "a".repeat(1_000);
        let out = snippet(&text, "deploying");
        assert!(!out.is_empty());
        assert!(out.chars().count() <= SNIPPET + 2);
    }

    // ------------------------------------------------------- export

    #[test]
    fn a_conversation_exports_as_readable_markdown() {
        let md = to_markdown(
            &conversation("the deploy script"),
            &[message(0, "user", "why is it slow"), message(1, "assistant", "the retry budget")],
        );
        assert!(md.starts_with("# the deploy script\n"), "{md}");
        assert!(md.contains("## You\n\nwhy is it slow"), "{md}");
        assert!(md.contains("## ozgent\n\nthe retry budget"), "{md}");
        assert!(md.contains("2025-08-20"), "the date is there: {md}");
        assert!(md.contains("gemma4:12b"), "the model is there: {md}");
    }

    #[test]
    fn an_export_says_which_tools_a_turn_used() {
        // Without it an answer full of current facts looks like it was
        // invented rather than looked up.
        let mut m = message(1, "assistant", "four results");
        m.tool_calls = Some(
            r#"[{"kind":"tool","name":"web_search"},{"kind":"tool","name":"fetch_url"}]"#.into(),
        );
        let md = to_markdown(&conversation("t"), &[m]);
        assert!(md.contains("*Used web_search, fetch_url*"), "{md}");
    }

    #[test]
    fn tool_results_are_left_out_of_an_export() {
        // They are the machinery of a turn, not part of the conversation.
        let md = to_markdown(
            &conversation("t"),
            &[message(0, "user", "hi"), message(1, "tool", "{\"results\": []}")],
        );
        assert!(!md.contains("results"), "{md}");
    }

    #[test]
    fn an_untitled_or_empty_conversation_still_exports() {
        let md = to_markdown(&conversation(""), &[]);
        assert!(md.starts_with("# Conversation"), "{md}");
        assert!(md.contains("0 messages"), "{md}");
    }

    #[test]
    fn a_turn_with_no_text_is_marked_rather_than_left_blank() {
        let md = to_markdown(&conversation("t"), &[message(0, "assistant", "  ")]);
        assert!(md.contains("*(no text)*"), "{md}");
    }

    #[test]
    fn one_message_is_not_pluralised() {
        let md = to_markdown(&conversation("t"), &[message(0, "user", "hi")]);
        assert!(md.contains("1 message,"), "{md}");
        assert!(!md.contains("1 messages"), "{md}");
    }

    // ------------------------------------------------------- file names

    #[test]
    fn a_title_becomes_a_safe_file_name() {
        assert_eq!(file_name("the deploy script"), "the-deploy-script");
        assert_eq!(file_name("  spaced  out  "), "spaced-out");
        assert_eq!(file_name("Q3 report: draft"), "Q3-report-draft");
    }

    #[test]
    fn a_title_cannot_escape_the_download_folder_or_break_the_header() {
        // The title is user text and goes straight into a Content-Disposition.
        for hostile in ["../../etc/passwd", "a/b/c", "a\"b", "a\nb", "..", "/"] {
            let name = file_name(hostile);
            assert!(!name.contains('/'), "{hostile:?} -> {name}");
            assert!(!name.contains('"'), "{hostile:?} -> {name}");
            assert!(!name.contains('\n'), "{hostile:?} -> {name}");
            assert!(!name.contains(".."), "{hostile:?} -> {name}");
        }
    }

    #[test]
    fn a_title_with_nothing_usable_in_it_still_gives_a_file_name() {
        for empty in ["", "   ", "///", "!!!"] {
            assert_eq!(file_name(empty), "conversation", "for {empty:?}");
        }
    }

    #[test]
    fn a_very_long_title_is_shortened() {
        assert!(file_name(&"word ".repeat(100)).chars().count() <= 60);
    }
}
