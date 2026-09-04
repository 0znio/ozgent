//! SQLite persistence for conversations, messages, facts, and embeddings.
//!
//! One file, `~/ozgent/ozgent.db`, shared by the TUI and the web UI so both
//! see the same history. SQLite is compiled in rather than linked against the
//! system copy, which guarantees FTS5 is present and keeps ozgent's promise
//! that deleting its directory leaves nothing behind.

use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;

/// Bumped whenever the schema changes; [`Store::migrate`] steps up to it.
pub const SCHEMA_VERSION: i64 = 6;

pub struct Store {
    db: Connection,
}

/// A stored conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Conversation {
    pub id: i64,
    /// Stable public identifier, safe to put in a URL.
    ///
    /// Row ids are sequential and leak how many conversations exist; a link
    /// also has to survive a database that was rebuilt or merged.
    pub uuid: String,
    pub title: String,
    pub model: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub message_count: i64,
}

/// A messaging chat bound to a conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelChat {
    /// `telegram`, `whatsapp`, and so on.
    pub channel: String,
    /// The provider's own identifier for the chat, as text: Telegram's is a
    /// 64-bit integer and WhatsApp's is a JID, so neither type fits both.
    pub chat_id: String,
    pub conversation_id: i64,
    /// A human label — the sender's name or number — so `ozgent channel list`
    /// shows who a chat belongs to rather than an opaque id.
    pub display: String,
    pub last_seen_at: i64,
}

/// A stored workflow. `definition` is the JSON the canvas edits.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredFlow {
    pub id: i64,
    /// Stable public identifier, safe to put in a URL.
    pub uuid: String,
    pub name: String,
    pub definition: String,
    /// Whether triggers other than a manual run may fire.
    pub enabled: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

/// One time a workflow ran. `record` is the JSON step-by-step detail.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRun {
    pub id: i64,
    pub flow_id: i64,
    pub trigger: String,
    pub status: String,
    pub record: String,
    pub ms: i64,
    pub created_at: i64,
}

/// A stored message. `seq` orders messages within their conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredMessage {
    pub id: i64,
    pub conversation_id: i64,
    pub seq: i64,
    pub role: String,
    pub content: String,
    pub thinking: Option<String>,
    pub tool_calls: Option<String>,
    pub tool_call_id: Option<String>,
    /// JSON array of file names under the media directory.
    pub media: Option<String>,
    pub tokens: i64,
    pub created_at: i64,
}

/// Where a fact applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// True only inside one conversation.
    Conversation,
    /// True of the user generally, across conversations.
    User,
}

impl Scope {
    fn as_str(self) -> &'static str {
        match self {
            Self::Conversation => "conversation",
            Self::User => "user",
        }
    }
    fn parse(s: &str) -> Self {
        match s {
            "user" => Self::User,
            _ => Self::Conversation,
        }
    }
}

/// A durable statement extracted from conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    pub id: i64,
    pub conversation_id: Option<i64>,
    pub scope: Scope,
    pub text: String,
    pub source_message_id: Option<i64>,
    /// Pinned facts are always in context, never subject to retrieval.
    pub pinned: bool,
    /// The statement decomposed, when it decomposes: who, which property,
    /// what value. Present together or not at all. This is what makes
    /// supersession decidable — two facts about the same subject and relation
    /// state the same property, so the later one replaces the earlier.
    pub triple: Option<Triple>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// A fact reduced to the property it asserts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Triple {
    pub subject: String,
    pub relation: String,
    pub value: String,
}

impl Triple {
    /// The key two facts must share to be about the same thing.
    ///
    /// Compared case- and space-insensitively, because an extractor writes
    /// "Editor" one turn and "editor" the next and means the same property.
    pub fn key(&self) -> (String, String) {
        (normalise_key(&self.subject), normalise_key(&self.relation))
    }
}

fn normalise_key(s: &str) -> String {
    s.trim().to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// What an embedding belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnerKind {
    Message,
    Fact,
}

impl OwnerKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::Fact => "fact",
        }
    }
}

impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let db = Connection::open(path)?;
        Self::init(db)
    }

    pub fn open_in_memory() -> Result<Self, StoreError> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(db: Connection) -> Result<Self, StoreError> {
        db.execute_batch(
            // WAL lets the web UI read while the TUI writes.
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             PRAGMA synchronous = NORMAL;",
        )?;
        let mut store = Self { db };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&mut self) -> Result<(), StoreError> {
        let current: i64 = self
            .db
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap_or(0);

        if current > SCHEMA_VERSION {
            return Err(StoreError::FutureSchema { found: current, ours: SCHEMA_VERSION });
        }
        if current == SCHEMA_VERSION {
            return Ok(());
        }

        if current < 1 {
            self.db.execute_batch(SCHEMA_V1)?;
        }
        if current < 2 {
            self.db.execute_batch(SCHEMA_V2)?;
            self.backfill_uuids()?;
        }
        if current < 3 {
            self.db.execute_batch(SCHEMA_V3)?;
        }
        if current < 4 {
            self.db.execute_batch(SCHEMA_V4)?;
        }
        if current < 5 {
            self.db.execute_batch(SCHEMA_V5)?;
        }
        if current < 6 {
            self.db.execute_batch(SCHEMA_V6)?;
        }

        self.db
            .pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(())
    }

    pub fn raw(&self) -> &Connection {
        &self.db
    }

    // ------------------------------------------------------ conversations

    pub fn create_conversation(
        &self,
        title: &str,
        model: Option<&str>,
    ) -> Result<i64, StoreError> {
        let now = now();
        self.db.execute(
            "INSERT INTO conversations (uuid, title, model, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)",
            params![new_uuid(), title, model, now],
        )?;
        Ok(self.db.last_insert_rowid())
    }

    /// Record the media files that arrived with a message.
    ///
    /// Written after the message so a failed upload cannot orphan a row; the
    /// bytes live on disk and only their names are stored.
    pub fn set_message_media(&self, message_id: i64, files: &[String]) -> Result<(), StoreError> {
        let json = serde_json::to_string(files).unwrap_or_else(|_| "[]".into());
        self.db.execute(
            "UPDATE messages SET media = ?1 WHERE id = ?2",
            params![json, message_id],
        )?;
        Ok(())
    }

    /// Find a conversation by its public identifier.
    pub fn conversation_by_uuid(&self, uuid: &str) -> Result<Option<Conversation>, StoreError> {
        self.db
            .query_row(
                "SELECT id FROM conversations WHERE uuid = ?1",
                params![uuid],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
            .map(|id| self.get_conversation(id))
            .transpose()
            .map(Option::flatten)
    }

    /// Give every pre-existing row an identifier.
    fn backfill_uuids(&self) -> Result<(), StoreError> {
        let ids: Vec<i64> = {
            let mut stmt = self
                .db
                .prepare("SELECT id FROM conversations WHERE uuid IS NULL OR uuid = ''")?;
            let rows = stmt.query_map([], |r| r.get(0))?;
            rows.collect::<Result<_, _>>()?
        };
        for id in ids {
            self.db.execute(
                "UPDATE conversations SET uuid = ?1 WHERE id = ?2",
                params![new_uuid(), id],
            )?;
        }
        Ok(())
    }

    pub fn get_conversation(&self, id: i64) -> Result<Option<Conversation>, StoreError> {
        Ok(self
            .db
            .query_row(
                "SELECT c.id, c.uuid, c.title, c.model, c.created_at, c.updated_at,
                        (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id)
                 FROM conversations c WHERE c.id = ?1",
                params![id],
                row_to_conversation,
            )
            .optional()?)
    }

    /// Most recently updated first, which is the order both UIs want.
    pub fn list_conversations(&self, limit: i64) -> Result<Vec<Conversation>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT c.id, c.uuid, c.title, c.model, c.created_at, c.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id)
             FROM conversations c ORDER BY c.updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], row_to_conversation)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Conversations that have at least one message, newest first.
    ///
    /// A conversation with nothing in it is a placeholder the user never
    /// filled, and showing it in a picker is offering to reopen nothing.
    /// Filtered in SQL rather than after the fact, so `limit` counts rows the
    /// caller can actually use.
    pub fn list_active_conversations(&self, limit: i64) -> Result<Vec<Conversation>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT c.id, c.uuid, c.title, c.model, c.created_at, c.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS n
             FROM conversations c WHERE n > 0
             ORDER BY c.updated_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit], row_to_conversation)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// How many conversations hold no messages at all.
    pub fn empty_conversation_count(&self) -> Result<i64, StoreError> {
        Ok(self.db.query_row(
            "SELECT COUNT(*) FROM conversations c WHERE NOT EXISTS
               (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)",
            [],
            |r| r.get(0),
        )?)
    }

    /// Delete every conversation holding no messages, returning how many went.
    ///
    /// `keep` is spared whatever its state — it is the one the caller is
    /// sitting in, and deleting the row underneath a live chat would strand
    /// every message written afterwards against a conversation that is gone.
    pub fn delete_empty_conversations(&self, keep: Option<i64>) -> Result<usize, StoreError> {
        let removed = self.db.execute(
            "DELETE FROM conversations WHERE id IS NOT ?1 AND NOT EXISTS
               (SELECT 1 FROM messages m WHERE m.conversation_id = conversations.id)",
            params![keep],
        )?;
        Ok(removed)
    }

    pub fn rename_conversation(&self, id: i64, title: &str) -> Result<(), StoreError> {
        self.db.execute(
            "UPDATE conversations SET title = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, title, now()],
        )?;
        Ok(())
    }

    /// Deletes the conversation and, by cascade, its messages, facts, and
    /// their embeddings.
    pub fn delete_conversation(&self, id: i64) -> Result<(), StoreError> {
        self.db
            .execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
        Ok(())
    }

    // ----------------------------------------------------- channel chats

    /// The conversation a messaging chat is bound to, if it still exists.
    ///
    /// Returns `None` both when the chat has never been seen and when the
    /// conversation it pointed at has been deleted, because the caller does the
    /// same thing in either case: start a new one.
    pub fn channel_conversation(
        &self,
        channel: &str,
        chat_id: &str,
    ) -> Result<Option<i64>, StoreError> {
        let found = self
            .db
            .query_row(
                "SELECT conversation_id FROM channel_chats WHERE channel = ?1 AND chat_id = ?2",
                params![channel, chat_id],
                |r| r.get::<_, i64>(0),
            )
            .optional()?;
        Ok(found)
    }

    /// Point a chat at a conversation, replacing any earlier binding.
    pub fn bind_channel_chat(
        &self,
        channel: &str,
        chat_id: &str,
        conversation_id: i64,
        display: &str,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO channel_chats (channel, chat_id, conversation_id, display, last_seen_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT (channel, chat_id) DO UPDATE SET
                 conversation_id = excluded.conversation_id,
                 display         = excluded.display,
                 last_seen_at    = excluded.last_seen_at",
            params![channel, chat_id, conversation_id, display, now()],
        )?;
        Ok(())
    }

    /// Forget a chat's binding, so its next message starts a new conversation.
    ///
    /// The conversation itself is left alone: someone asking their assistant to
    /// start fresh is not asking to erase what was said.
    pub fn unbind_channel_chat(&self, channel: &str, chat_id: &str) -> Result<bool, StoreError> {
        let n = self.db.execute(
            "DELETE FROM channel_chats WHERE channel = ?1 AND chat_id = ?2",
            params![channel, chat_id],
        )?;
        Ok(n > 0)
    }

    /// Every chat bound on a channel, most recently active first.
    pub fn channel_chats(&self, channel: &str) -> Result<Vec<ChannelChat>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT chat_id, conversation_id, display, last_seen_at
               FROM channel_chats WHERE channel = ?1 ORDER BY last_seen_at DESC",
        )?;
        let rows = stmt
            .query_map(params![channel], |r| {
                Ok(ChannelChat {
                    channel: channel.to_string(),
                    chat_id: r.get(0)?,
                    conversation_id: r.get(1)?,
                    display: r.get(2)?,
                    last_seen_at: r.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ---------------------------------------------------------- workflows

    pub fn create_flow(&self, name: &str, definition: &str) -> Result<StoredFlow, StoreError> {
        let now = now();
        let uuid = new_uuid();
        self.db.execute(
            "INSERT INTO flows (uuid, name, definition, enabled, created_at, updated_at)
             VALUES (?1, ?2, ?3, 0, ?4, ?4)",
            params![uuid, name, definition, now],
        )?;
        let id = self.db.last_insert_rowid();
        Ok(StoredFlow {
            id,
            uuid,
            name: name.to_string(),
            definition: definition.to_string(),
            enabled: false,
            created_at: now,
            updated_at: now,
        })
    }

    pub fn list_flows(&self) -> Result<Vec<StoredFlow>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT id, uuid, name, definition, enabled, created_at, updated_at
               FROM flows ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], read_flow)?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn get_flow(&self, id: i64) -> Result<Option<StoredFlow>, StoreError> {
        Ok(self
            .db
            .query_row(
                "SELECT id, uuid, name, definition, enabled, created_at, updated_at
                   FROM flows WHERE id = ?1",
                params![id],
                read_flow,
            )
            .optional()?)
    }

    pub fn flow_by_uuid(&self, uuid: &str) -> Result<Option<StoredFlow>, StoreError> {
        Ok(self
            .db
            .query_row(
                "SELECT id, uuid, name, definition, enabled, created_at, updated_at
                   FROM flows WHERE uuid = ?1",
                params![uuid],
                read_flow,
            )
            .optional()?)
    }

    pub fn save_flow(
        &self,
        id: i64,
        name: &str,
        definition: &str,
        enabled: bool,
    ) -> Result<(), StoreError> {
        self.db.execute(
            "UPDATE flows SET name = ?2, definition = ?3, enabled = ?4, updated_at = ?5
              WHERE id = ?1",
            params![id, name, definition, enabled, now()],
        )?;
        Ok(())
    }

    pub fn delete_flow(&self, id: i64) -> Result<(), StoreError> {
        self.db.execute("DELETE FROM flows WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Every flow whose triggers are allowed to fire.
    pub fn enabled_flows(&self) -> Result<Vec<StoredFlow>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT id, uuid, name, definition, enabled, created_at, updated_at
               FROM flows WHERE enabled = 1 ORDER BY id",
        )?;
        let rows = stmt.query_map([], read_flow)?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Record a run, and forget the oldest once there are more than `keep`.
    ///
    /// Trimmed on write rather than by a sweep: a flow on a one-minute schedule
    /// writes half a million rows a year, and the interesting ones are always
    /// the most recent.
    pub fn add_run(
        &self,
        flow_id: i64,
        trigger: &str,
        status: &str,
        record: &str,
        ms: i64,
        keep: i64,
    ) -> Result<i64, StoreError> {
        self.db.execute(
            "INSERT INTO flow_runs (flow_id, trigger, status, record, ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![flow_id, trigger, status, record, ms, now()],
        )?;
        let id = self.db.last_insert_rowid();
        self.db.execute(
            "DELETE FROM flow_runs WHERE flow_id = ?1 AND id NOT IN
                 (SELECT id FROM flow_runs WHERE flow_id = ?1 ORDER BY id DESC LIMIT ?2)",
            params![flow_id, keep],
        )?;
        Ok(id)
    }

    pub fn runs_for(&self, flow_id: i64, limit: i64) -> Result<Vec<StoredRun>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT id, flow_id, trigger, status, record, ms, created_at
               FROM flow_runs WHERE flow_id = ?1 ORDER BY id DESC LIMIT ?2",
        )?;
        let rows =
            stmt.query_map(params![flow_id, limit], read_run)?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn get_run(&self, id: i64) -> Result<Option<StoredRun>, StoreError> {
        Ok(self
            .db
            .query_row(
                "SELECT id, flow_id, trigger, status, record, ms, created_at
                   FROM flow_runs WHERE id = ?1",
                params![id],
                read_run,
            )
            .optional()?)
    }

    // ----------------------------------------------------------- messages

    /// Append a message, assigning the next sequence number.
    pub fn append_message(
        &self,
        conversation_id: i64,
        role: &str,
        content: &str,
        tokens: i64,
    ) -> Result<i64, StoreError> {
        self.append_message_full(conversation_id, role, content, None, None, None, tokens)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_message_full(
        &self,
        conversation_id: i64,
        role: &str,
        content: &str,
        thinking: Option<&str>,
        tool_calls: Option<&str>,
        tool_call_id: Option<&str>,
        tokens: i64,
    ) -> Result<i64, StoreError> {
        let now = now();
        let seq: i64 = self.db.query_row(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE conversation_id = ?1",
            params![conversation_id],
            |r| r.get(0),
        )?;

        self.db.execute(
            "INSERT INTO messages
               (conversation_id, seq, role, content, thinking, tool_calls, tool_call_id,
                tokens, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                conversation_id, seq, role, content, thinking, tool_calls, tool_call_id,
                tokens, now
            ],
        )?;
        let id = self.db.last_insert_rowid();

        self.db.execute(
            "UPDATE conversations SET updated_at = ?2 WHERE id = ?1",
            params![conversation_id, now],
        )?;
        Ok(id)
    }

    /// All messages in order.
    pub fn messages(&self, conversation_id: i64) -> Result<Vec<StoredMessage>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT id, conversation_id, seq, role, content, thinking, tool_calls,
                    tool_call_id, tokens, created_at, media
             FROM messages WHERE conversation_id = ?1 ORDER BY seq",
        )?;
        let rows = stmt.query_map(params![conversation_id], row_to_message)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// The last `n` messages, still in ascending order.
    pub fn recent_messages(
        &self,
        conversation_id: i64,
        n: i64,
    ) -> Result<Vec<StoredMessage>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT id, conversation_id, seq, role, content, thinking, tool_calls,
                    tool_call_id, tokens, created_at, media
             FROM messages WHERE conversation_id = ?1
             ORDER BY seq DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![conversation_id, n], row_to_message)?;
        let mut out: Vec<StoredMessage> = rows.collect::<Result<_, _>>()?;
        out.reverse();
        Ok(out)
    }

    pub fn get_message(&self, id: i64) -> Result<Option<StoredMessage>, StoreError> {
        Ok(self
            .db
            .query_row(
                "SELECT id, conversation_id, seq, role, content, thinking, tool_calls,
                        tool_call_id, tokens, created_at
                 FROM messages WHERE id = ?1",
                params![id],
                row_to_message,
            )
            .optional()?)
    }

    pub fn message_count(&self, conversation_id: i64) -> Result<i64, StoreError> {
        Ok(self.db.query_row(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?1",
            params![conversation_id],
            |r| r.get(0),
        )?)
    }

    // -------------------------------------------------------------- facts

    pub fn add_fact(
        &self,
        conversation_id: Option<i64>,
        scope: Scope,
        text: &str,
        source_message_id: Option<i64>,
    ) -> Result<i64, StoreError> {
        self.add_fact_with_triple(conversation_id, scope, text, source_message_id, None)
    }

    /// Store a fact, optionally with the property it asserts.
    pub fn add_fact_with_triple(
        &self,
        conversation_id: Option<i64>,
        scope: Scope,
        text: &str,
        source_message_id: Option<i64>,
        triple: Option<&Triple>,
    ) -> Result<i64, StoreError> {
        let now = now();
        self.db.execute(
            "INSERT INTO facts
               (conversation_id, scope, text, source_message_id, pinned, created_at, updated_at,
                subject, relation, value)
             VALUES (?1, ?2, ?3, ?4, 0, ?5, ?5, ?6, ?7, ?8)",
            params![
                conversation_id,
                scope.as_str(),
                text,
                source_message_id,
                now,
                triple.map(|t| t.subject.as_str()),
                triple.map(|t| t.relation.as_str()),
                triple.map(|t| t.value.as_str()),
            ],
        )?;
        Ok(self.db.last_insert_rowid())
    }

    /// The live fact stating the same property of the same subject, if any.
    ///
    /// This is the whole of conflict detection, and it is deliberately not a
    /// similarity test. A contradiction and a duplicate read almost alike, so
    /// similarity cannot separate them; an exact match on subject and relation
    /// can. Whether the new statement agrees with the old one does not matter
    /// — either way the later one is what the user last said.
    pub fn live_fact_for_property(
        &self,
        conversation_id: Option<i64>,
        triple: &Triple,
    ) -> Result<Option<Fact>, StoreError> {
        let (subject, relation) = triple.key();
        let mut stmt = self.db.prepare(
            "SELECT id, conversation_id, scope, text, source_message_id, pinned,
                    created_at, updated_at, subject, relation, value
             FROM facts
             WHERE superseded_by IS NULL
               AND subject IS NOT NULL
               AND lower(trim(subject))  = ?1
               AND lower(trim(relation)) = ?2
               AND (conversation_id IS ?3 OR scope = 'user')
             ORDER BY updated_at DESC
             LIMIT 1",
        )?;
        Ok(stmt
            .query_map(params![subject, relation, conversation_id], row_to_fact)?
            .next()
            .transpose()?)
    }

    /// Facts visible to a conversation: its own, plus every user-scoped fact.
    /// Superseded facts are excluded, so a corrected fact never resurfaces.
    pub fn facts_for(&self, conversation_id: i64) -> Result<Vec<Fact>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT id, conversation_id, scope, text, source_message_id, pinned,
                    created_at, updated_at, subject, relation, value
             FROM facts
             WHERE superseded_by IS NULL
               AND (conversation_id = ?1 OR scope = 'user')
             ORDER BY pinned DESC, updated_at DESC",
        )?;
        let rows = stmt.query_map(params![conversation_id], row_to_fact)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn pinned_facts(&self, conversation_id: i64) -> Result<Vec<Fact>, StoreError> {
        Ok(self
            .facts_for(conversation_id)?
            .into_iter()
            .filter(|f| f.pinned)
            .collect())
    }

    pub fn set_pinned(&self, fact_id: i64, pinned: bool) -> Result<(), StoreError> {
        self.db.execute(
            "UPDATE facts SET pinned = ?2, updated_at = ?3 WHERE id = ?1",
            params![fact_id, pinned as i64, now()],
        )?;
        Ok(())
    }

    /// Mark `old` as replaced by `new`, so only the correction is retrieved.
    pub fn supersede_fact(&self, old: i64, new: i64) -> Result<(), StoreError> {
        self.db.execute(
            "UPDATE facts SET superseded_by = ?2, updated_at = ?3 WHERE id = ?1",
            params![old, new, now()],
        )?;
        Ok(())
    }

    pub fn get_fact(&self, id: i64) -> Result<Option<Fact>, StoreError> {
        Ok(self
            .db
            .query_row(
                "SELECT id, conversation_id, scope, text, source_message_id, pinned,
                        created_at, updated_at, subject, relation, value
                 FROM facts WHERE id = ?1",
                params![id],
                row_to_fact,
            )
            .optional()?)
    }

    pub fn delete_fact(&self, id: i64) -> Result<(), StoreError> {
        self.db.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
        Ok(())
    }

    // --------------------------------------------------------- embeddings

    /// Store a vector for a message or fact, replacing any previous one.
    pub fn put_embedding(
        &self,
        kind: OwnerKind,
        owner_id: i64,
        vector: &[f32],
    ) -> Result<(), StoreError> {
        self.db.execute(
            "INSERT INTO embeddings (owner_kind, owner_id, dim, vec)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(owner_kind, owner_id) DO UPDATE SET dim = ?3, vec = ?4",
            params![kind.as_str(), owner_id, vector.len() as i64, encode_vec(vector)],
        )?;
        Ok(())
    }

    pub fn get_embedding(
        &self,
        kind: OwnerKind,
        owner_id: i64,
    ) -> Result<Option<Vec<f32>>, StoreError> {
        let blob: Option<Vec<u8>> = self
            .db
            .query_row(
                "SELECT vec FROM embeddings WHERE owner_kind = ?1 AND owner_id = ?2",
                params![kind.as_str(), owner_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(blob.map(|b| decode_vec(&b)))
    }

    /// Every embedding of a kind within one conversation.
    ///
    /// A personal history is thousands of vectors at most, so a full scan with
    /// cosine in Rust beats taking on a vector-index extension.
    pub fn embeddings_in_conversation(
        &self,
        kind: OwnerKind,
        conversation_id: i64,
    ) -> Result<Vec<(i64, Vec<f32>)>, StoreError> {
        let sql = match kind {
            OwnerKind::Message => {
                "SELECT e.owner_id, e.vec FROM embeddings e
                 JOIN messages m ON m.id = e.owner_id
                 WHERE e.owner_kind = 'message' AND m.conversation_id = ?1"
            }
            OwnerKind::Fact => {
                "SELECT e.owner_id, e.vec FROM embeddings e
                 JOIN facts f ON f.id = e.owner_id
                 WHERE e.owner_kind = 'fact' AND f.superseded_by IS NULL
                   AND (f.conversation_id = ?1 OR f.scope = 'user')"
            }
        };
        let mut stmt = self.db.prepare(sql)?;
        let rows = stmt.query_map(params![conversation_id], |r| {
            let id: i64 = r.get(0)?;
            let blob: Vec<u8> = r.get(1)?;
            Ok((id, decode_vec(&blob)))
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    /// Messages in a conversation that have no embedding yet.
    pub fn messages_missing_embeddings(
        &self,
        conversation_id: i64,
    ) -> Result<Vec<StoredMessage>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT m.id, m.conversation_id, m.seq, m.role, m.content, m.thinking,
                    m.tool_calls, m.tool_call_id, m.tokens, m.created_at, m.media
             FROM messages m
             LEFT JOIN embeddings e
               ON e.owner_kind = 'message' AND e.owner_id = m.id
             WHERE m.conversation_id = ?1 AND e.owner_id IS NULL
             ORDER BY m.seq",
        )?;
        let rows = stmt.query_map(params![conversation_id], row_to_message)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }
}

// ------------------------------------------------------------------ rows

fn row_to_conversation(r: &rusqlite::Row<'_>) -> rusqlite::Result<Conversation> {
    Ok(Conversation {
        id: r.get(0)?,
        // Rows written before the v2 migration are backfilled, but a NULL read
        // here should degrade rather than fail a whole listing.
        uuid: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
        title: r.get(2)?,
        model: r.get(3)?,
        created_at: r.get(4)?,
        updated_at: r.get(5)?,
        message_count: r.get(6)?,
    })
}

fn row_to_message(r: &rusqlite::Row<'_>) -> rusqlite::Result<StoredMessage> {
    Ok(StoredMessage {
        id: r.get(0)?,
        conversation_id: r.get(1)?,
        seq: r.get(2)?,
        role: r.get(3)?,
        content: r.get(4)?,
        thinking: r.get(5)?,
        tool_calls: r.get(6)?,
        tool_call_id: r.get(7)?,
        tokens: r.get(8)?,
        created_at: r.get(9)?,
        media: r.get(10).ok().flatten(),
    })
}

fn row_to_fact(r: &rusqlite::Row<'_>) -> rusqlite::Result<Fact> {
    let scope: String = r.get(2)?;
    Ok(Fact {
        id: r.get(0)?,
        conversation_id: r.get(1)?,
        scope: Scope::parse(&scope),
        text: r.get(3)?,
        source_message_id: r.get(4)?,
        pinned: r.get::<_, i64>(5)? != 0,
        created_at: r.get(6)?,
        updated_at: r.get(7)?,
        triple: match (
            r.get::<_, Option<String>>(8)?,
            r.get::<_, Option<String>>(9)?,
            r.get::<_, Option<String>>(10)?,
        ) {
            (Some(subject), Some(relation), Some(value)) => {
                Some(Triple { subject, relation, value })
            }
            _ => None,
        },
    })
}

/// Vectors are stored as little-endian f32, which is compact and lets the
/// cosine scan read them back without parsing.
fn encode_vec(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn decode_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn read_flow(r: &rusqlite::Row<'_>) -> rusqlite::Result<StoredFlow> {
    Ok(StoredFlow {
        id: r.get(0)?,
        uuid: r.get(1)?,
        name: r.get(2)?,
        definition: r.get(3)?,
        enabled: r.get::<_, i64>(4)? != 0,
        created_at: r.get(5)?,
        updated_at: r.get(6)?,
    })
}

fn read_run(r: &rusqlite::Row<'_>) -> rusqlite::Result<StoredRun> {
    Ok(StoredRun {
        id: r.get(0)?,
        flow_id: r.get(1)?,
        trigger: r.get(2)?,
        status: r.get(3)?,
        record: r.get(4)?,
        ms: r.get(5)?,
        created_at: r.get(6)?,
    })
}

const SCHEMA_V1: &str = r#"
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL DEFAULT '',
    model       TEXT,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

CREATE TABLE messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    thinking        TEXT,
    tool_calls      TEXT,
    tool_call_id    TEXT,
    tokens          INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    UNIQUE (conversation_id, seq)
);
CREATE INDEX idx_messages_conversation ON messages(conversation_id, seq);

-- External-content FTS index: the text lives in `messages`, and FTS stores
-- only the inverted index, so the content is never duplicated.
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    content    = 'messages',
    content_rowid = 'id',
    tokenize   = 'porter unicode61'
);

CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TABLE facts (
    id                INTEGER PRIMARY KEY,
    conversation_id   INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    scope             TEXT    NOT NULL DEFAULT 'conversation',
    text              TEXT    NOT NULL,
    source_message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
    pinned            INTEGER NOT NULL DEFAULT 0,
    superseded_by     INTEGER REFERENCES facts(id) ON DELETE SET NULL,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL
);
CREATE INDEX idx_facts_conversation ON facts(conversation_id);

CREATE VIRTUAL TABLE facts_fts USING fts5(
    text,
    content    = 'facts',
    content_rowid = 'id',
    tokenize   = 'porter unicode61'
);
CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE embeddings (
    owner_kind TEXT    NOT NULL,
    owner_id   INTEGER NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB    NOT NULL,
    PRIMARY KEY (owner_kind, owner_id)
) WITHOUT ROWID;

-- SQLite will not cascade into a WITHOUT ROWID table across a join, so
-- embeddings are cleaned up explicitly when their owner disappears.
CREATE TRIGGER embeddings_gc_message AFTER DELETE ON messages BEGIN
    DELETE FROM embeddings WHERE owner_kind = 'message' AND owner_id = old.id;
END;
CREATE TRIGGER embeddings_gc_fact AFTER DELETE ON facts BEGIN
    DELETE FROM embeddings WHERE owner_kind = 'fact' AND owner_id = old.id;
END;
"#;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("database uses schema {found}, newer than this build understands ({ours}); upgrade ozgent")]
    FutureSchema { found: i64, ours: i64 },
}

/// The v2 schema step: a public identifier for every conversation.
/// The v3 step: attachments belong to the message that carried them.
///
/// The bytes live on disk under `~/ozgent/media`; only the file names are
/// stored here, because a database is a poor place for megabytes of PNG.
const SCHEMA_V3: &str = "
ALTER TABLE messages ADD COLUMN media TEXT;
";

/// The v4 step: facts may carry a subject-relation-value triple.
///
/// Two things need this and they are the same thing. Supersession needs a key
/// that says "this states the same property of the same entity as that one",
/// which similarity cannot decide — a contradiction and a duplicate look
/// alike. And a knowledge graph needs edges, which is what a triple is.
///
/// The columns are nullable on purpose. Not every fact decomposes, and one
/// that does not is still worth keeping as prose; it simply does not
/// participate in supersession or traversal.
const SCHEMA_V4: &str = "
ALTER TABLE facts ADD COLUMN subject  TEXT;
ALTER TABLE facts ADD COLUMN relation TEXT;
ALTER TABLE facts ADD COLUMN value    TEXT;
CREATE INDEX idx_facts_triple ON facts(subject, relation) WHERE subject IS NOT NULL;
CREATE INDEX idx_facts_value  ON facts(value)             WHERE value   IS NOT NULL;
";

const SCHEMA_V2: &str = "
ALTER TABLE conversations ADD COLUMN uuid TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS conversations_uuid ON conversations(uuid);
";

/// The v5 step: a messaging chat is bound to a conversation.
///
/// Someone messaging ozgent from Telegram or WhatsApp is having one continuing
/// conversation, not a series of unrelated questions, so the chat has to map to
/// a stable `conversations` row — that is what gives a channel the same recent
/// window, retrieval and pinned facts every other surface gets, and what makes
/// a chat readable afterwards in the web interface.
///
/// The mapping is deliberately its own table rather than a column on
/// `conversations`: a conversation may be reached from more than one place over
/// its life, and `ON DELETE CASCADE` means deleting the conversation in the web
/// interface unbinds the chat, which then starts a fresh one on the next
/// message rather than writing into a hole.
const SCHEMA_V5: &str = "
CREATE TABLE channel_chats (
    channel         TEXT    NOT NULL,
    chat_id         TEXT    NOT NULL,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    display         TEXT    NOT NULL DEFAULT '',
    last_seen_at    INTEGER NOT NULL,
    PRIMARY KEY (channel, chat_id)
);
CREATE INDEX idx_channel_chats_conversation ON channel_chats(conversation_id);
";

/// The v6 step: workflows, and a record of every time one ran.
///
/// The definition is stored as JSON rather than decomposed into tables. A flow
/// is edited as a whole by the canvas, saved as a whole, and never queried by
/// its parts — so tables of nodes and edges would buy nothing and cost a
/// migration every time a node kind gains a setting.
///
/// Runs are the opposite: they are queried, by flow and by recency, so they get
/// their own rows. The step-by-step detail inside one is JSON for the same
/// reason as the definition.
const SCHEMA_V6: &str = "
CREATE TABLE flows (
    id          INTEGER PRIMARY KEY,
    uuid        TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    definition  TEXT    NOT NULL,
    -- Whether triggers other than `run this now` may fire. Off by default:
    -- a half-drawn flow on a schedule would start running as it was being
    -- built.
    enabled     INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE UNIQUE INDEX flows_uuid ON flows(uuid);

CREATE TABLE flow_runs (
    id          INTEGER PRIMARY KEY,
    flow_id     INTEGER NOT NULL REFERENCES flows(id) ON DELETE CASCADE,
    trigger     TEXT    NOT NULL,
    status      TEXT    NOT NULL,
    record      TEXT    NOT NULL,
    ms          INTEGER NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE INDEX idx_flow_runs_flow ON flow_runs(flow_id, id DESC);
";

/// A random UUID v4.
///
/// Hand-rolled rather than pulling in a crate for sixteen bytes: the only
/// requirements are the version/variant bits and enough entropy that two
/// conversations never collide.
fn new_uuid() -> String {
    let mut bytes = [0u8; 16];
    // Two independent sources so a coarse clock cannot produce a duplicate:
    // the nanosecond timestamp, and the address of a fresh allocation.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let entropy = {
        let boxed = Box::new(0u8);
        let addr = Box::into_raw(boxed) as usize;
        // SAFETY: reclaimed immediately; only its address was wanted.
        unsafe { drop(Box::from_raw(addr as *mut u8)) };
        addr as u128
    };
    let mixed = nanos ^ (entropy.rotate_left(64)) ^ (std::process::id() as u128) << 96;
    bytes.copy_from_slice(&mixed.to_be_bytes());

    // Stir, so adjacent timestamps do not produce adjacent-looking ids.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, b) in bytes.iter_mut().enumerate() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        *b = (hash >> ((i % 8) * 8)) as u8;
    }

    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1
    let h = |r: &[u8]| r.iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!(
        "{}-{}-{}-{}-{}",
        h(&bytes[0..4]), h(&bytes[4..6]), h(&bytes[6..8]), h(&bytes[8..10]), h(&bytes[10..16])
    )
}

#[cfg(test)]
mod channel_tests {
    use super::*;

    fn store() -> Store {
        Store::open_in_memory().expect("opening an in-memory store")
    }

    #[test]
    fn a_chat_remembers_which_conversation_it_belongs_to() {
        let s = store();
        let c = s.create_conversation("from telegram", None).unwrap();
        s.bind_channel_chat("telegram", "4242", c, "Ada").unwrap();
        assert_eq!(s.channel_conversation("telegram", "4242").unwrap(), Some(c));
    }

    #[test]
    fn chats_do_not_cross_between_channels() {
        // Telegram chat ids and WhatsApp JIDs are different namespaces that
        // can collide as text; the same id on two channels is two people.
        let s = store();
        let a = s.create_conversation("a", None).unwrap();
        let b = s.create_conversation("b", None).unwrap();
        s.bind_channel_chat("telegram", "1", a, "").unwrap();
        s.bind_channel_chat("whatsapp", "1", b, "").unwrap();
        assert_eq!(s.channel_conversation("telegram", "1").unwrap(), Some(a));
        assert_eq!(s.channel_conversation("whatsapp", "1").unwrap(), Some(b));
    }

    #[test]
    fn rebinding_moves_the_chat_rather_than_failing() {
        // What `/new` from a chat does: the next message must land somewhere
        // else, and the row is a primary key, so this has to be an upsert.
        let s = store();
        let first = s.create_conversation("first", None).unwrap();
        let second = s.create_conversation("second", None).unwrap();
        s.bind_channel_chat("telegram", "1", first, "Ada").unwrap();
        s.bind_channel_chat("telegram", "1", second, "Ada").unwrap();
        assert_eq!(s.channel_conversation("telegram", "1").unwrap(), Some(second));
        assert_eq!(s.channel_chats("telegram").unwrap().len(), 1);
    }

    #[test]
    fn deleting_the_conversation_unbinds_the_chat() {
        // Otherwise a conversation deleted in the web interface leaves the
        // chat pointing at a row that is gone, and the next message from that
        // person fails its foreign key instead of starting fresh.
        let s = store();
        let c = s.create_conversation("gone", None).unwrap();
        s.bind_channel_chat("telegram", "1", c, "").unwrap();
        s.delete_conversation(c).unwrap();
        assert_eq!(s.channel_conversation("telegram", "1").unwrap(), None);
        assert!(s.channel_chats("telegram").unwrap().is_empty());
    }

    #[test]
    fn unbinding_keeps_what_was_said() {
        let s = store();
        let c = s.create_conversation("kept", None).unwrap();
        s.append_message(c, "user", "hello", 0).unwrap();
        s.bind_channel_chat("telegram", "1", c, "").unwrap();

        assert!(s.unbind_channel_chat("telegram", "1").unwrap());
        assert!(!s.unbind_channel_chat("telegram", "1").unwrap(), "already gone");
        assert_eq!(s.channel_conversation("telegram", "1").unwrap(), None);
        assert_eq!(s.messages(c).unwrap().len(), 1, "the conversation survives");
    }

    #[test]
    fn chats_are_listed_most_recently_active_first() {
        let s = store();
        let a = s.create_conversation("a", None).unwrap();
        let b = s.create_conversation("b", None).unwrap();
        s.bind_channel_chat("telegram", "old", a, "Ada").unwrap();
        s.bind_channel_chat("telegram", "new", b, "Grace").unwrap();
        // `now()` has second resolution, so order the two explicitly rather
        // than depending on the clock ticking between two inserts.
        s.raw()
            .execute(
                "UPDATE channel_chats SET last_seen_at = 1 WHERE chat_id = 'old'",
                [],
            )
            .unwrap();

        let listed = s.channel_chats("telegram").unwrap();
        assert_eq!(listed[0].chat_id, "new");
        assert_eq!(listed[0].display, "Grace");
        assert_eq!(listed[1].chat_id, "old");
    }
}

#[cfg(test)]
mod flow_tests {
    use super::*;

    fn store() -> Store {
        Store::open_in_memory().expect("opening an in-memory store")
    }

    #[test]
    fn a_flow_is_stored_and_read_back_whole() {
        let s = store();
        let definition = r#"{"name":"daily","nodes":[],"edges":[]}"#;
        let made = s.create_flow("daily", definition).unwrap();

        let found = s.get_flow(made.id).unwrap().unwrap();
        assert_eq!(found.definition, definition);
        assert!(!found.enabled, "a new flow does not fire on its own");
        assert_eq!(s.flow_by_uuid(&made.uuid).unwrap().unwrap().id, made.id);
    }

    #[test]
    fn a_new_flow_is_not_enabled() {
        // A half-drawn flow on a schedule would start running as it is built.
        let s = store();
        s.create_flow("draft", "{}").unwrap();
        assert!(s.enabled_flows().unwrap().is_empty());
    }

    #[test]
    fn enabling_is_what_puts_a_flow_in_front_of_the_scheduler() {
        let s = store();
        let f = s.create_flow("nightly", "{}").unwrap();
        s.save_flow(f.id, "nightly", "{\"nodes\":[]}", true).unwrap();

        let enabled = s.enabled_flows().unwrap();
        assert_eq!(enabled.len(), 1);
        assert_eq!(enabled[0].definition, "{\"nodes\":[]}");
    }

    #[test]
    fn runs_are_listed_newest_first() {
        let s = store();
        let f = s.create_flow("f", "{}").unwrap();
        for i in 0..3 {
            s.add_run(f.id, "t", "ok", &format!("{{\"n\":{i}}}"), 10, 50).unwrap();
        }
        let runs = s.runs_for(f.id, 10).unwrap();
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].record, "{\"n\":2}");
    }

    #[test]
    fn only_the_most_recent_runs_are_kept() {
        // A flow on a one-minute schedule writes half a million rows a year,
        // and the interesting ones are always the newest.
        let s = store();
        let f = s.create_flow("f", "{}").unwrap();
        for i in 0..12 {
            s.add_run(f.id, "t", "ok", &format!("{{\"n\":{i}}}"), 1, 5).unwrap();
        }
        let runs = s.runs_for(f.id, 100).unwrap();
        assert_eq!(runs.len(), 5);
        assert_eq!(runs[0].record, "{\"n\":11}");
        assert_eq!(runs[4].record, "{\"n\":7}", "the oldest kept");
    }

    #[test]
    fn trimming_one_flows_runs_leaves_anothers_alone() {
        let s = store();
        let a = s.create_flow("a", "{}").unwrap();
        let b = s.create_flow("b", "{}").unwrap();
        for _ in 0..10 {
            s.add_run(a.id, "t", "ok", "{}", 1, 2).unwrap();
        }
        s.add_run(b.id, "t", "ok", "{}", 1, 2).unwrap();

        assert_eq!(s.runs_for(a.id, 100).unwrap().len(), 2);
        assert_eq!(s.runs_for(b.id, 100).unwrap().len(), 1);
    }

    #[test]
    fn deleting_a_flow_takes_its_runs_with_it() {
        let s = store();
        let f = s.create_flow("f", "{}").unwrap();
        let run = s.add_run(f.id, "t", "ok", "{}", 1, 50).unwrap();
        s.delete_flow(f.id).unwrap();

        assert!(s.get_flow(f.id).unwrap().is_none());
        assert!(s.get_run(run).unwrap().is_none(), "orphaned run rows");
    }

    #[test]
    fn flows_are_listed_most_recently_edited_first() {
        let s = store();
        let older = s.create_flow("older", "{}").unwrap();
        let newer = s.create_flow("newer", "{}").unwrap();
        // `now()` has second resolution, so order the two explicitly rather
        // than depending on the clock ticking between two inserts.
        s.raw()
            .execute("UPDATE flows SET updated_at = 1 WHERE id = ?1", params![older.id])
            .unwrap();

        let listed = s.list_flows().unwrap();
        assert_eq!(listed[0].id, newer.id);
        assert_eq!(listed[1].id, older.id);
    }
}
