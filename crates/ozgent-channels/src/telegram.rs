//! Telegram, over the Bot API.
//!
//! Long-polling rather than a webhook, deliberately: a webhook needs a public
//! HTTPS endpoint, which means a domain, a certificate and an inbound hole in
//! whatever network the machine is on. Polling needs an outbound connection and
//! nothing else, which is the difference between "works on a laptop behind a
//! router" and "works if you own a server".
//!
//! Two Telegram behaviours shape the code more than anything else:
//!
//! * **A message with markup it dislikes is rejected whole.** Not stripped —
//!   rejected, with a 400. So every send has a plain-text retry behind it, and
//!   losing the formatting beats losing the reply.
//! * **Edits are rate-limited and refuse to be no-ops.** Editing a message to
//!   the text it already has is an error, not a success, so the last text sent
//!   per message is remembered and an unchanged edit is skipped here rather
//!   than counted against the limit.

use std::collections::HashMap;
use std::time::Duration;

use ozgent_core::ImageSource;
use ozgent_core::permission::Choice;
use tokio::sync::mpsc::{Sender, UnboundedReceiver};

use crate::chat::{ANSWERS, Command, Inbound, Msg, Question};
use crate::markup::{Flavour, render};
use crate::split::{TELEGRAM_LIMIT, split};

const API: &str = "https://api.telegram.org";

/// How long the server holds a poll open with nothing to say.
///
/// Long polls are what make this cheap: one request every 30 seconds when idle
/// instead of a request per second, and a message arrives the moment it is
/// sent rather than up to a poll interval later.
const POLL_SECONDS: u64 = 30;

/// Images larger than this are not downloaded.
///
/// A vision model resizes to a few hundred pixels anyway, so the only thing a
/// larger download buys is memory and time — and the sender is remote, so the
/// size is not this machine's choice to trust.
const MAX_IMAGE_BYTES: u64 = 12 * 1024 * 1024;

pub struct Telegram {
    token: String,
    http: reqwest::Client,
}

impl Telegram {
    pub fn new(token: String) -> Self {
        Self {
            token,
            http: reqwest::Client::builder()
                // Longer than the long poll, or every idle poll is a timeout.
                .timeout(Duration::from_secs(POLL_SECONDS + 30))
                .build()
                .unwrap_or_default(),
        }
    }

    fn url(&self, method: &str) -> String {
        format!("{API}/bot{}/{method}", self.token)
    }

    /// Call a Bot API method, returning its `result`.
    async fn call(&self, method: &str, body: serde_json::Value) -> Result<serde_json::Value, ApiError> {
        let response = self
            .http
            .post(self.url(method))
            .json(&body)
            .send()
            .await
            .map_err(|e| ApiError::Transport(e.to_string()))?;

        let status = response.status();
        let payload: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ApiError::Transport(format!("unreadable reply from {method}: {e}")))?;

        if payload.get("ok").and_then(|v| v.as_bool()) == Some(true) {
            return Ok(payload.get("result").cloned().unwrap_or(serde_json::Value::Null));
        }

        let description = payload
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("no description")
            .to_string();
        let retry_after = payload
            .get("parameters")
            .and_then(|p| p.get("retry_after"))
            .and_then(|v| v.as_u64());
        Err(ApiError::Api { status: status.as_u16(), description, retry_after })
    }

    /// Who the token belongs to. The first thing done, so a bad token is a
    /// clear message at startup instead of a poll loop that never yields.
    pub async fn identify(&self) -> anyhow::Result<String> {
        let me = self.call("getMe", serde_json::json!({})).await?;
        let name = me
            .get("username")
            .and_then(|v| v.as_str())
            .map(|u| format!("@{u}"))
            .unwrap_or_else(|| "the bot".to_string());
        Ok(name)
    }
}

/// Someone who sent the bot a setup code.
#[derive(Debug, Clone)]
pub struct Claimant {
    pub id: String,
    pub username: Option<String>,
    pub name: String,
    pub chat: String,
}

impl Telegram {
    /// Wait for a private message containing `code`, and say who sent it.
    ///
    /// How the terminal setup learns the owner's user id without asking them
    /// to look it up: they send the code shown on their own screen, so the
    /// sender is known to be the person at this machine. Everything else that
    /// arrives meanwhile is ignored. `None` when the time runs out.
    pub async fn wait_for_code(&self, code: &str, within: Duration) -> anyhow::Result<Option<Claimant>> {
        let deadline = tokio::time::Instant::now() + within;
        let mut offset: Option<i64> = None;
        let code = code.to_ascii_uppercase();
        while tokio::time::Instant::now() < deadline {
            let left = deadline.saturating_duration_since(tokio::time::Instant::now()).as_secs();
            let body = serde_json::json!({
                "timeout": left.clamp(1, 20),
                "offset": offset,
                "allowed_updates": ["message"],
            });
            let updates = match self.call("getUpdates", body).await {
                Ok(v) => v,
                Err(ApiError::Api { status: 409, .. }) => anyhow::bail!(
                    "another program is reading this bot's messages right now (a running ozgent?)"
                ),
                Err(e) => {
                    tracing::debug!("telegram setup: {e}");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            };
            for update in updates.as_array().into_iter().flatten() {
                if let Some(id) = update.get("update_id").and_then(|v| v.as_i64()) {
                    offset = Some(id + 1);
                }
                let Some(message) = update.get("message") else { continue };
                let private = message.pointer("/chat/type").and_then(|v| v.as_str()) == Some("private");
                let text = message.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if !private || !text.to_ascii_uppercase().contains(&code) {
                    continue;
                }
                let Some(from) = message.get("from") else { continue };
                let found = Claimant {
                    id: from.get("id").and_then(|v| v.as_i64()).map(|i| i.to_string()).unwrap_or_default(),
                    username: from.get("username").and_then(|v| v.as_str()).map(|u| format!("@{u}")),
                    name: from.get("first_name").and_then(|v| v.as_str()).unwrap_or("you").to_string(),
                    chat: message.pointer("/chat/id").and_then(|v| v.as_i64()).map(|i| i.to_string()).unwrap_or_default(),
                };
                // Acknowledge what was read, so the gateway does not answer
                // the setup message when it starts.
                let _ = self
                    .call("getUpdates", serde_json::json!({ "offset": offset, "timeout": 0 }))
                    .await;
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// Send one plain message, outside the gateway.
    pub async fn say(&self, chat: &str, text: &str) -> anyhow::Result<()> {
        self.call("sendMessage", serde_json::json!({ "chat_id": chat, "text": text })).await?;
        Ok(())
    }
}

/// What went wrong with a Bot API call.
#[derive(Debug)]
enum ApiError {
    Transport(String),
    Api { status: u16, description: String, retry_after: Option<u64> },
}

impl ApiError {
    /// Whether the markup was the problem, so a plain-text retry is worth it.
    fn is_markup(&self) -> bool {
        matches!(self, Self::Api { description, .. } if description.contains("parse entities"))
    }

    /// An edit that changed nothing. Telegram calls this an error; it is not.
    fn is_unchanged(&self) -> bool {
        matches!(self, Self::Api { description, .. } if description.contains("message is not modified"))
    }

    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Api { retry_after: Some(s), .. } => Some(Duration::from_secs(*s)),
            _ => None,
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transport(e) => write!(f, "{e}"),
            Self::Api { status, description, .. } => write!(f, "telegram {status}: {description}"),
        }
    }
}

impl std::error::Error for ApiError {}

/// Run the channel until the command side is dropped.
///
/// Receiving and sending are separate tasks because a long poll blocks for up
/// to half a minute: sharing one task would mean a reply waiting on a poll that
/// has nothing to deliver.
pub async fn run(
    token: String,
    tx: Sender<Inbound>,
    rx: UnboundedReceiver<Command>,
) -> anyhow::Result<()> {
    let api = std::sync::Arc::new(Telegram::new(token));
    let who = api.identify().await.map_err(|e| {
        anyhow::anyhow!(
            "{e}\n\nTelegram refused the bot token. Get a new one from @BotFather and set it \
             with `ozgent gateway telegram token`, or on /admin."
        )
    })?;
    let _ = tx.send(Inbound::Ready { who }).await;

    let sender = tokio::spawn(send_loop(api.clone(), rx, tx.clone()));
    let result = poll_loop(api, tx).await;
    sender.abort();
    result
}

/// Ask for updates forever, and turn each into an [`Inbound`].
async fn poll_loop(api: std::sync::Arc<Telegram>, tx: Sender<Inbound>) -> anyhow::Result<()> {
    // `None` asks for whatever is queued; afterwards, one past the last update
    // seen, which is also what acknowledges it so it is not delivered again.
    let mut offset: Option<i64> = None;
    // Backoff for a network that has gone away, so a laptop that closed its
    // lid does not spend the night hammering a dead connection.
    let mut backoff = Duration::from_secs(1);

    loop {
        let body = serde_json::json!({
            "timeout": POLL_SECONDS,
            "offset": offset,
            // Anything else — edits, channel posts, members joining — would be
            // delivered, acknowledged and thrown away, so it is not asked for.
            "allowed_updates": ["message", "callback_query"],
        });

        let updates = match api.call("getUpdates", body).await {
            Ok(v) => {
                backoff = Duration::from_secs(1);
                v
            }
            Err(e) => {
                // 409 has one cause and one fix, and the generic message for it
                // sends people to the wrong place.
                if let ApiError::Api { status: 409, .. } = &e {
                    let _ = tx
                        .send(Inbound::Failed {
                            reason: "another program is already polling this bot token, or a \
                                     webhook is set for it. Stop the other one, or clear the \
                                     webhook, then start ozgent again."
                                .into(),
                        })
                        .await;
                    return Ok(());
                }
                tracing::warn!("telegram: {e}");
                tokio::time::sleep(e.retry_after().unwrap_or(backoff)).await;
                backoff = (backoff * 2).min(Duration::from_secs(60));
                continue;
            }
        };

        let Some(list) = updates.as_array() else { continue };
        for update in list {
            if let Some(id) = update.get("update_id").and_then(|v| v.as_i64()) {
                offset = Some(id + 1);
            }
            for inbound in read_update(&api, update).await {
                // A full channel means the gateway is not keeping up. Blocking
                // here is the right answer: it stops acknowledging updates,
                // and Telegram holds them for us.
                if tx.send(inbound).await.is_err() {
                    return Ok(());
                }
            }
        }
    }
}

/// Turn one update into zero or more inbound events.
async fn read_update(api: &Telegram, update: &serde_json::Value) -> Vec<Inbound> {
    if let Some(query) = update.get("callback_query") {
        return read_callback(api, query).await.into_iter().collect();
    }
    let Some(message) = update.get("message") else { return Vec::new() };
    read_message(api, message)
        .await
        .map(|m| Inbound::Message(Box::new(m)))
        .into_iter()
        .collect()
}

async fn read_callback(api: &Telegram, query: &serde_json::Value) -> Option<Inbound> {
    let data = query.get("data").and_then(|v| v.as_str()).unwrap_or("");
    let chat = query
        .get("message")
        .and_then(|m| m.get("chat"))
        .and_then(|c| c.get("id"))
        .and_then(|v| v.as_i64())
        .map(|i| i.to_string())?;

    let (token, choice) = read_callback_data(data)?;

    // Telegram shows a spinner on the button until this is answered, so it
    // goes out before anything slower happens.
    if let Some(id) = query.get("id").and_then(|v| v.as_str()) {
        let _ = api
            .call("answerCallbackQuery", serde_json::json!({ "callback_query_id": id }))
            .await;
    }

    // The gateway holds the token→call-id mapping and resolves it.
    Some(Inbound::Answer { chat, token, choice })
}

/// Read `<token>|<n>` from a button's callback data.
///
/// Telegram caps callback data at 64 bytes, which a tool call id can exceed,
/// so the button carries the gateway's own token for the question instead.
fn read_callback_data(data: &str) -> Option<(u64, Choice)> {
    let (token, index) = data.split_once('|')?;
    let token: u64 = token.parse().ok()?;
    let index: usize = index.parse().ok()?;
    let (_, choice) = ANSWERS.get(index)?;
    Some((token, *choice))
}

async fn read_message(api: &Telegram, message: &serde_json::Value) -> Option<Msg> {
    let chat = message.get("chat")?;
    let chat_id = chat.get("id")?.as_i64()?.to_string();
    let group = chat
        .get("type")
        .and_then(|v| v.as_str())
        .is_some_and(|t| t != "private");

    let from = message.get("from")?;
    let sender_id = from.get("id")?.as_i64()?.to_string();
    let handle = from.get("username").and_then(|v| v.as_str()).map(str::to_string);
    let name = from
        .get("first_name")
        .and_then(|v| v.as_str())
        .unwrap_or("someone")
        .to_string();

    // A photo arrives with its caption in `caption`; a plain message uses
    // `text`. Neither is present for a sticker or a location, which is a
    // message with nothing to answer.
    let text = message
        .get("text")
        .or_else(|| message.get("caption"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let images = read_photos(api, message).await;
    if text.trim().is_empty() && images.is_empty() {
        return None;
    }

    // Never true here: a Telegram bot has an identity of its own and cannot
    // be the person messaging it. Only WhatsApp links a person's account.
    Some(Msg { chat: chat_id, sender_id, handle, name, text, images, group, own: false })
}

/// Download the photo on a message, at the largest size within the cap.
///
/// Telegram sends every size it has; the last is the largest. Picking the
/// largest that fits beats picking the last and then refusing it.
async fn read_photos(api: &Telegram, message: &serde_json::Value) -> Vec<ImageSource> {
    let Some(sizes) = message.get("photo").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let best = sizes
        .iter()
        .filter(|s| s.get("file_size").and_then(|v| v.as_u64()).unwrap_or(0) <= MAX_IMAGE_BYTES)
        .max_by_key(|s| s.get("file_size").and_then(|v| v.as_u64()).unwrap_or(0));
    let Some(file_id) = best.and_then(|s| s.get("file_id")).and_then(|v| v.as_str()) else {
        return Vec::new();
    };

    match download(api, file_id).await {
        Ok(bytes) => vec![ImageSource::Bytes { bytes, mime: Some("image/jpeg".into()) }],
        Err(e) => {
            tracing::warn!("telegram: downloading a photo: {e}");
            Vec::new()
        }
    }
}

async fn download(api: &Telegram, file_id: &str) -> anyhow::Result<Vec<u8>> {
    let file = api
        .call("getFile", serde_json::json!({ "file_id": file_id }))
        .await?;
    let path = file
        .get("file_path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("no file_path in the reply"))?;
    let bytes = api
        .http
        .get(format!("{API}/file/bot{}/{path}", api.token))
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    Ok(bytes.to_vec())
}

/// State the sending side keeps, all of it keyed by the gateway's token.
#[derive(Default)]
struct Sent {
    /// The Telegram message a token stands for.
    message: HashMap<u64, i64>,
    /// The last text sent for a token, so an unchanged edit is not attempted.
    text: HashMap<u64, String>,
}

async fn send_loop(
    api: std::sync::Arc<Telegram>,
    mut rx: UnboundedReceiver<Command>,
    tx: Sender<Inbound>,
) {
    let mut sent = Sent::default();

    while let Some(command) = rx.recv().await {
        match command {
            Command::Typing { chat } => {
                let _ = api
                    .call(
                        "sendChatAction",
                        serde_json::json!({ "chat_id": chat, "action": "typing" }),
                    )
                    .await;
            }

            Command::Post { chat, token, markdown } => {
                // Only the last chunk keeps the token: it is the one a later
                // revision would rewrite, and the earlier ones are finished.
                let chunks = split(&markdown, TELEGRAM_LIMIT);
                let last = chunks.len().saturating_sub(1);
                for (i, chunk) in chunks.into_iter().enumerate() {
                    match post(&api, &chat, &chunk, None).await {
                        Ok(id) if i == last => {
                            sent.message.insert(token, id);
                            sent.text.insert(token, chunk);
                        }
                        Ok(_) => {}
                        Err(e) => tracing::warn!("telegram: sending: {e}"),
                    }
                }
            }

            Command::Revise { chat, token, markdown } => {
                let Some(&message_id) = sent.message.get(&token) else {
                    continue;
                };
                // Revising is only ever a live update of a growing reply. Past
                // one message's worth there is nothing useful to edit — the
                // final text is posted in full when the turn ends — so an
                // over-long revision is left for that.
                if markdown.chars().count() > TELEGRAM_LIMIT {
                    continue;
                }
                if sent.text.get(&token).is_some_and(|t| *t == markdown) {
                    continue;
                }
                match edit(&api, &chat, message_id, &markdown, None).await {
                    Ok(()) => {
                        sent.text.insert(token, markdown);
                    }
                    Err(e) if e.is_unchanged() => {}
                    Err(e) => tracing::debug!("telegram: revising: {e}"),
                }
            }

            Command::Ask { chat, token, question } => {
                let text = render(&question_text(&question), Flavour::TelegramHtml);
                let keyboard = keyboard(token);
                match post(&api, &chat, &text, Some(keyboard)).await {
                    Ok(id) => {
                        sent.message.insert(token, id);
                        sent.text.insert(token, text);
                    }
                    Err(e) => {
                        // The question could not be shown, so nobody will ever
                        // answer it. Saying so is better than the turn quietly
                        // timing out five minutes later.
                        tracing::warn!("telegram: asking: {e}");
                        let _ = tx
                            .send(Inbound::Answer { chat: chat.clone(), token, choice: Choice::Deny })
                            .await;
                    }
                }
            }

            Command::Settle { chat, token, markdown } => {
                let Some(&message_id) = sent.message.get(&token) else {
                    continue;
                };
                let text = render(&markdown, Flavour::TelegramHtml);
                // An empty keyboard is what removes the buttons; omitting the
                // field leaves them there to be tapped a second time.
                match edit(&api, &chat, message_id, &text, Some(serde_json::json!({ "inline_keyboard": [] }))).await {
                    Ok(()) | Err(_) => {}
                }
                sent.message.remove(&token);
                sent.text.remove(&token);
            }

            Command::Logout => {}
        }
    }
}

/// The text of a permission question.
fn question_text(q: &Question) -> String {
    let what = match q.effect {
        ozgent_core::permission::Effect::Write => "wants to change something",
        ozgent_core::permission::Effect::Execute => "wants to run a program",
        ozgent_core::permission::Effect::Read => "wants to look something up",
        _ => "wants to do something it has not described",
    };
    let mut out = format!("**{}** {what}.", q.tool);
    if !q.detail.trim().is_empty() {
        out.push_str(&format!("\n\n```\n{}\n```", q.detail.trim()));
    }
    out.push_str("\n\nAllow it?");
    out
}

/// The four answers as buttons, carrying the question's token.
fn keyboard(token: u64) -> serde_json::Value {
    let row: Vec<serde_json::Value> = ANSWERS
        .iter()
        .enumerate()
        .map(|(i, (word, _))| {
            serde_json::json!({
                "text": label(word),
                "callback_data": format!("{token}|{i}"),
            })
        })
        .collect();
    serde_json::json!({ "inline_keyboard": [row] })
}

fn label(word: &str) -> String {
    match word {
        "yes" => "✓ Yes".into(),
        "session" => "Yes, this session".into(),
        "always" => "Always".into(),
        _ => "✗ No".into(),
    }
}

/// Send a message, falling back to plain text if Telegram dislikes the markup.
async fn post(
    api: &Telegram,
    chat: &str,
    html: &str,
    keyboard: Option<serde_json::Value>,
) -> Result<i64, ApiError> {
    let body = |text: &str, mode: Option<&str>| {
        let mut b = serde_json::json!({ "chat_id": chat, "text": text });
        if let Some(mode) = mode {
            b["parse_mode"] = mode.into();
        }
        if let Some(k) = &keyboard {
            b["reply_markup"] = k.clone();
        }
        b
    };

    let sent = match retrying(api, "sendMessage", body(html, Some("HTML"))).await {
        Ok(v) => v,
        Err(e) if e.is_markup() => {
            // Losing the formatting beats losing the reply. This is the one
            // failure worth a second attempt with different content.
            tracing::debug!("telegram: markup rejected, sending as text: {e}");
            retrying(api, "sendMessage", body(&strip(html), None)).await?
        }
        Err(e) => return Err(e),
    };
    sent.get("message_id")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| ApiError::Transport("sendMessage returned no message_id".into()))
}

async fn edit(
    api: &Telegram,
    chat: &str,
    message_id: i64,
    html: &str,
    keyboard: Option<serde_json::Value>,
) -> Result<(), ApiError> {
    let body = |text: &str, mode: Option<&str>| {
        let mut b = serde_json::json!({
            "chat_id": chat,
            "message_id": message_id,
            "text": text,
        });
        if let Some(mode) = mode {
            b["parse_mode"] = mode.into();
        }
        if let Some(k) = &keyboard {
            b["reply_markup"] = k.clone();
        }
        b
    };
    match retrying(api, "editMessageText", body(html, Some("HTML"))).await {
        Ok(_) => Ok(()),
        Err(e) if e.is_markup() => {
            retrying(api, "editMessageText", body(&strip(html), None)).await?;
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Call a method, waiting once if Telegram asks us to slow down.
///
/// Once, not repeatedly: a second 429 means the sending rate is wrong rather
/// than unlucky, and queueing behind an unbounded retry would delay every
/// later message in the same chat.
async fn retrying(
    api: &Telegram,
    method: &str,
    body: serde_json::Value,
) -> Result<serde_json::Value, ApiError> {
    match api.call(method, body.clone()).await {
        Err(e) => match e.retry_after() {
            Some(wait) => {
                tracing::debug!("telegram: asked to wait {}s", wait.as_secs());
                tokio::time::sleep(wait).await;
                api.call(method, body).await
            }
            None => Err(e),
        },
        ok => ok,
    }
}

/// Undo the escaping and tags, for the plain-text retry.
fn strip(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if in_tag => {}
            _ => out.push(c),
        }
    }
    out.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&amp;", "&")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_button_carries_the_token_and_the_answer_it_stands_for() {
        // Callback data is capped at 64 bytes, which a tool call id can
        // exceed, so the token is what travels.
        let k = keyboard(7);
        let row = k["inline_keyboard"][0].as_array().unwrap();
        assert_eq!(row.len(), 4);
        for (i, button) in row.iter().enumerate() {
            let data = button["callback_data"].as_str().unwrap();
            assert!(data.len() <= 64, "{data} is too long for telegram");
            assert_eq!(read_callback_data(data), Some((7, ANSWERS[i].1)));
        }
    }

    #[test]
    fn nonsense_callback_data_is_ignored_rather_than_guessed_at() {
        // It decides whether a tool runs, so anything unrecognised is not an
        // answer at all.
        for data in ["", "x|0", "7", "7|9", "7|-1", "|", "7|0|0"] {
            assert_eq!(read_callback_data(data), None, "{data:?}");
        }
    }

    #[test]
    fn stripping_recovers_the_text_from_the_markup() {
        // The plain-text retry has to undo both the tags and the escaping, or
        // the fallback shows `&lt;div&gt;` to the reader.
        assert_eq!(strip("<b>bold</b> and <i>it</i>"), "bold and it");
        assert_eq!(strip("a &lt;div&gt; &amp; more"), "a <div> & more");
        assert_eq!(strip("<a href=\"https://x.test\">link</a>"), "link");
    }

    #[test]
    fn the_ampersand_is_unescaped_last() {
        // Otherwise `&amp;lt;` becomes `<` instead of `&lt;`.
        assert_eq!(strip("&amp;lt;"), "&lt;");
    }

    #[test]
    fn a_question_says_what_the_tool_would_do() {
        let q = Question {
            id: "c1".into(),
            tool: "run_command".into(),
            effect: ozgent_core::permission::Effect::Execute,
            detail: "git status".into(),
        };
        let text = question_text(&q);
        assert!(text.contains("run_command"), "{text}");
        assert!(text.contains("run a program"), "{text}");
        assert!(text.contains("git status"), "{text}");

        // And renders as valid Telegram markup.
        let html = render(&text, Flavour::TelegramHtml);
        assert_eq!(html.matches("<b>").count(), html.matches("</b>").count());
        assert!(html.contains("<pre>"), "{html}");
    }

    #[test]
    fn a_question_with_no_arguments_yet_still_reads_as_a_question() {
        let q = Question {
            id: "c1".into(),
            tool: "write_file".into(),
            effect: ozgent_core::permission::Effect::Write,
            detail: String::new(),
        };
        let text = question_text(&q);
        assert!(text.ends_with("Allow it?"), "{text}");
        assert!(!text.contains("```"), "{text}");
    }

    #[test]
    fn an_unknown_effect_is_described_as_unknown_rather_than_harmless() {
        let q = Question {
            id: "c1".into(),
            tool: "mystery".into(),
            effect: ozgent_core::permission::Effect::Unknown,
            detail: String::new(),
        };
        assert!(question_text(&q).contains("has not described"), "{}", question_text(&q));
    }
}
