//! SQLite persistence for conversations, messages, facts, and embeddings.
//!
//! One file, `~/ozgent/ozgent.db`, shared by the TUI and the web UI so both
//! see the same history. SQLite is compiled in rather than linked against the
//! system copy, which guarantees FTS5 is present and keeps ozgent's promise
//! that deleting its directory leaves nothing behind.

use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;

/// Bumped whenever the schema changes; [`Store::migrate`] steps up to it.
pub const SCHEMA_VERSION: i64 = 3;

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
    pub created_at: i64,
    pub updated_at: i64,
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
        let now = now();
        self.db.execute(
            "INSERT INTO facts
               (conversation_id, scope, text, source_message_id, pinned, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, 0, ?5, ?5)",
            params![conversation_id, scope.as_str(), text, source_message_id, now],
        )?;
        Ok(self.db.last_insert_rowid())
    }

    /// Facts visible to a conversation: its own, plus every user-scoped fact.
    /// Superseded facts are excluded, so a corrected fact never resurfaces.
    pub fn facts_for(&self, conversation_id: i64) -> Result<Vec<Fact>, StoreError> {
        let mut stmt = self.db.prepare(
            "SELECT id, conversation_id, scope, text, source_message_id, pinned,
                    created_at, updated_at
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
                        created_at, updated_at
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

const SCHEMA_V2: &str = "
ALTER TABLE conversations ADD COLUMN uuid TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS conversations_uuid ON conversations(uuid);
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
