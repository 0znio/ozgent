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
}
