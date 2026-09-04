//! Conversation memory.
//!
//! Keeps a conversation answerable about its own distant past without resending
//! it. Storage is one SQLite file shared by every front end; recall is lexical
//! and vector search fused by reciprocal rank; and what reaches the model is
//! assembled in tiers under an explicit token budget.

pub mod context;
pub mod embed;
pub mod facts;
pub mod retrieve;
pub mod store;

pub use context::{AssembledContext, Budget, ContextBuilder, estimate_tokens};
pub use embed::{Embedder, HashingEmbedder, cosine};
pub use facts::{Candidate, parse_extraction, store_candidates};
pub use retrieve::{Hit, Retriever};
pub use store::{
    ChannelChat, Conversation, Fact, OwnerKind, Scope, StoredFlow, StoredMessage, StoredRun,
    Store, StoreError,
};
