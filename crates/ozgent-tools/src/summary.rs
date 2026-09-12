//! One line about a tool result, for the row that shows it.
//!
//! Shared by every front end, so a call reads the same in the terminal, the
//! browser, a chat app and an agent's trace. Knows the shapes the built-in
//! tools return; for anything else it says nothing and the caller falls back
//! to its own generic summary.

/// A sentence about a result whose shape is recognised, or `None`.
pub fn describe(value: &serde_json::Value) -> Option<String> {
    let obj = value.as_object()?;
    let action = obj.get("action").and_then(|a| a.as_str()).unwrap_or("");

    // The scheduler. Worth recognising above everything else because the
    // result is not information the model looked up — it is a thing that now
    // exists and will happen later, and every surface should say so plainly.
    if let Some(job) = obj.get("job").and_then(|j| j.as_str()) {
        let flag = |key: &str| obj.get(key).and_then(|v| v.as_bool()).unwrap_or(false);
        let next = obj.get("next_run").and_then(|n| n.as_str());
        let verb = if flag("scheduled") {
            "Job scheduled"
        } else if flag("changed") {
            "Job changed"
        } else if flag("deleted") {
            "Job deleted"
        } else if flag("queued") {
            "Job queued"
        } else if obj.get("paused").and_then(|p| p.as_bool()) == Some(true) {
            "Job paused"
        } else if obj.get("paused").and_then(|p| p.as_bool()) == Some(false) {
            "Job resumed"
        } else {
            ""
        };
        if !verb.is_empty() {
            return Some(match next {
                Some(next) => format!("{verb} — {job} · runs {next}"),
                None => format!("{verb} — {job}"),
            });
        }
    }
    if let Some(jobs) = obj.get("jobs").and_then(|j| j.as_array()) {
        return Some(match jobs.len() {
            0 => "nothing is scheduled".to_string(),
            1 => "1 scheduled job".to_string(),
            n => format!("{n} scheduled jobs"),
        });
    }

    // list_dir
    if let (Some(entries), Some(path)) =
        (obj.get("entries").and_then(|e| e.as_array()), obj.get("path").and_then(|p| p.as_str()))
    {
        // Nested listings put directories and their contents in one array, so
        // the count is of lines rather than of files; said as "entries" for
        // that reason rather than rounded up into a claim about files.
        let n = entries.len();
        let more = obj.get("truncated").and_then(|t| t.as_bool()).unwrap_or(false);
        let name = path.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(path);
        return Some(format!(
            "{n} entr{} in {name}{}",
            if n == 1 { "y" } else { "ies" },
            if more { ", truncated" } else { "" }
        ));
    }

    // yahoo_finance
    if let Some(quotes) = obj.get("quotes").and_then(|q| q.as_array()) {
        let parts: Vec<String> = quotes
            .iter()
            .take(3)
            .map(|q| {
                let symbol = q.get("symbol").and_then(|s| s.as_str()).unwrap_or("?");
                let price = q.get("price").and_then(|p| p.as_f64());
                let change = q.get("change_percent").and_then(|c| c.as_f64());
                let currency = q.get("currency").and_then(|c| c.as_str()).unwrap_or("");
                match (price, change) {
                    (Some(p), Some(c)) => format!("{symbol} {p} {currency} ({c:+.2}%)").replace("  ", " "),
                    (Some(p), None) => format!("{symbol} {p} {currency}"),
                    _ => symbol.to_string(),
                }
            })
            .collect();
        let more = quotes.len().saturating_sub(3);
        let tail = if more > 0 { format!(" and {more} more") } else { String::new() };
        return Some(format!("{}{tail}", parts.join(", ")));
    }
    if let Some(summary) = obj.get("summary").filter(|_| action == "history") {
        let symbol = obj.get("symbol").and_then(|s| s.as_str()).unwrap_or("?");
        let range = obj.get("range").and_then(|r| r.as_str()).unwrap_or("");
        let change = summary.get("change_percent").and_then(|c| c.as_f64());
        let bars = summary.get("bars").and_then(|b| b.as_u64()).unwrap_or(0);
        return Some(match change {
            Some(c) => format!("{symbol} {range}: {c:+.2}% over {bars} bars"),
            None => format!("{symbol} {range}: {bars} bars"),
        });
    }
    if action == "technicals" {
        let symbol = obj.get("symbol").and_then(|s| s.as_str()).unwrap_or("?");
        let mut parts = vec![format!("{symbol} technicals")];
        if let Some(r) = obj.get("rsi_14").and_then(|r| r.as_f64()) {
            parts.push(format!("RSI {r:.0}"));
        }
        if let Some(p) = value.pointer("/price_vs/sma200").and_then(|p| p.as_f64()) {
            parts.push(format!("{p:+.1}% vs 200-day"));
        }
        return Some(parts.join(" · "));
    }
    if action == "fundamentals" {
        let symbol = obj.get("symbol").and_then(|s| s.as_str()).unwrap_or("?");
        let sections = obj.keys().filter(|k| *k != "action" && *k != "symbol").count();
        return Some(format!("fundamentals for {symbol} · {sections} sections"));
    }
    if let Some(symbols) = obj.get("symbols").and_then(|s| s.as_array()) {
        let names: Vec<&str> =
            symbols.iter().filter_map(|s| s.get("symbol").and_then(|x| x.as_str())).take(4).collect();
        return Some(if names.is_empty() {
            "no matching symbols".to_string()
        } else {
            format!("found {}", names.join(", "))
        });
    }

    // fetch_url
    if let (Some(host), Some(text)) = (
        obj.get("host").and_then(|h| h.as_str()),
        obj.get("text").and_then(|t| t.as_str()),
    ) {
        let chars = text.chars().count();
        let focused = if obj.contains_key("focused_on") { " · focused" } else { "" };
        return Some(format!("{host} · {chars} chars{focused}"));
    }

    // reddit
    if let Some(source) = obj.get("source").and_then(|s| s.as_str()).filter(|s| s.starts_with("reddit")) {
        if let Some(comments) = obj.get("comments").and_then(|c| c.as_array()) {
            let n = comments.len();
            return Some(format!("{n} comment{} from {source}", if n == 1 { "" } else { "s" }));
        }
        if let Some(results) = obj.get("results").and_then(|r| r.as_array()) {
            let n = results.len();
            let place = obj
                .get("subreddit")
                .and_then(|s| s.as_str())
                .map(|s| format!(" in r/{s}"))
                .unwrap_or_default();
            return Some(format!("{n} post{}{place} from {source}", if n == 1 { "" } else { "s" }));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_quote_says_the_price_and_the_move() {
        let v = json!({"action": "quote", "quotes": [
            {"symbol": "NVDA", "price": 218.36, "change_percent": -2.374, "currency": "USD"}]});
        assert_eq!(describe(&v).unwrap(), "NVDA 218.36 USD (-2.37%)");
    }

    #[test]
    fn history_says_the_change_over_the_range() {
        let v = json!({"action": "history", "symbol": "NVDA", "range": "6mo",
                       "summary": {"change_percent": 17.38, "bars": 127}});
        assert_eq!(describe(&v).unwrap(), "NVDA 6mo: +17.38% over 127 bars");
    }

    #[test]
    fn reddit_says_how_much_and_from_where() {
        let v = json!({"action": "search", "source": "reddit feeds", "subreddit": "stocks",
                       "results": [{}, {}]});
        assert_eq!(describe(&v).unwrap(), "2 posts in r/stocks from reddit feeds");
        let c = json!({"action": "comments", "source": "reddit api", "comments": [{}]});
        assert_eq!(describe(&c).unwrap(), "1 comment from reddit api");
    }

    #[test]
    fn a_fetched_page_says_where_and_how_much() {
        let v = json!({"host": "www.infoworld.com", "text": "abcd", "kind": "text/html"});
        assert_eq!(describe(&v).unwrap(), "www.infoworld.com · 4 chars");
    }

    #[test]
    fn an_unknown_shape_is_left_to_the_caller() {
        assert!(describe(&json!({"path": "x"})).is_none());
        assert!(describe(&json!("text")).is_none());
    }

    #[test]
    fn technicals_are_summarised_by_their_headline_readings() {
        let v = serde_json::json!({
            "action": "technicals", "symbol": "NVDA", "rsi_14": 61.7,
            "price_vs": {"sma200": 12.34},
        });
        assert_eq!(describe(&v).as_deref(), Some("NVDA technicals · RSI 62 · +12.3% vs 200-day"));
    }
}

#[cfg(test)]
mod scheduler_tests {
    use super::describe;
    use serde_json::json;

    #[test]
    fn a_scheduled_job_reads_as_a_thing_that_now_exists() {
        // Not "the tool returned an object": a job was created and will run.
        let out = describe(&json!({
            "scheduled": true, "job": "pre-market-brief", "next_run": "in 14 hours",
            "summary": "every weekday at 09:20"
        }))
        .expect("the scheduler must be recognised");
        assert!(out.starts_with("Job scheduled — pre-market-brief"), "{out}");
        assert!(out.contains("in 14 hours"), "{out}");
    }

    #[test]
    fn every_change_to_a_job_says_which_change_it_was() {
        let cases = [
            (json!({"changed": true, "job": "brief"}), "Job changed"),
            (json!({"deleted": true, "job": "brief"}), "Job deleted"),
            (json!({"queued": true, "job": "brief"}), "Job queued"),
            (json!({"paused": true, "job": "brief"}), "Job paused"),
            (json!({"paused": false, "job": "brief"}), "Job resumed"),
        ];
        for (value, expected) in cases {
            let out = describe(&value).unwrap_or_else(|| panic!("not recognised: {value}"));
            assert!(out.starts_with(expected), "{value} gave {out:?}");
            assert!(out.contains("brief"), "{out}");
        }
    }

    #[test]
    fn listing_jobs_counts_them_rather_than_naming_every_one() {
        assert_eq!(describe(&json!({ "count": 0, "jobs": [] })).unwrap(), "nothing is scheduled");
        assert_eq!(describe(&json!({ "jobs": [{}] })).unwrap(), "1 scheduled job");
        assert_eq!(describe(&json!({ "jobs": [{}, {}, {}] })).unwrap(), "3 scheduled jobs");
    }

    #[test]
    fn showing_one_job_is_left_to_the_generic_summary() {
        // `show` returns the job's settings, which is ordinary tool output —
        // nothing happened, so announcing that something did would be wrong.
        let out = describe(&json!({ "job": "brief", "asks": "a brief", "paused": null }));
        assert_eq!(out, None);
    }

    #[test]
    fn a_result_from_another_tool_that_happens_to_have_a_job_key_is_not_claimed() {
        assert_eq!(describe(&json!({ "job": "something", "unrelated": 1 })), None);
    }
}

#[cfg(test)]
mod listing_tests {
    use super::describe;
    use serde_json::json;

    #[test]
    fn a_listing_says_how_much_is_in_where() {
        // It used to fall through to the key-list fallback and read
        // "entries, note, path, truncated", which says nothing.
        let out = describe(&json!({
            "path": "/home/someone/code/ozgent/docs",
            "entries": ["a.md", "b.md", "c.md"],
        }))
        .unwrap();
        assert_eq!(out, "3 entries in docs");
    }

    #[test]
    fn one_entry_is_not_pluralised() {
        let out = describe(&json!({ "path": "/tmp/x", "entries": ["only.txt"] })).unwrap();
        assert_eq!(out, "1 entry in x");
    }

    #[test]
    fn a_truncated_listing_says_so() {
        let out = describe(&json!({
            "path": "/big", "entries": ["a"], "truncated": true
        }))
        .unwrap();
        assert!(out.ends_with("truncated"), "{out}");
    }

    #[test]
    fn a_listing_of_nothing_still_reads_as_a_sentence() {
        let out = describe(&json!({ "path": "/empty", "entries": [] })).unwrap();
        assert_eq!(out, "0 entries in empty");
    }

    #[test]
    fn a_path_with_no_directory_part_is_used_whole() {
        let out = describe(&json!({ "path": ".", "entries": ["a"] })).unwrap();
        assert_eq!(out, "1 entry in .");
    }
}
