//! The requirement this crate exists for: a long conversation must stay
//! answerable about its own distant past without resending all of it.

use ozgent_memory::{
    Budget, ContextBuilder, Embedder, HashingEmbedder, OwnerKind, Scope, Store,
};

/// Build a conversation with `needles` planted at known positions, padded out
/// to `total` messages of unrelated filler.
fn conversation_with_needles(
    store: &Store,
    embedder: &dyn Embedder,
    needles: &[(usize, &str)],
    total: usize,
) -> i64 {
    let conv = store.create_conversation("long chat", Some("gemma4:12b")).unwrap();

    let filler = [
        "Let's move on to the next topic.",
        "That makes sense, thanks for explaining.",
        "Could you expand on that a little?",
        "Understood. What about the other approach?",
        "Right, I'll keep that in mind for later.",
        "Sounds reasonable to me overall.",
    ];

    for i in 0..total {
        let planted = needles.iter().find(|(pos, _)| *pos == i);
        let (role, text) = match planted {
            Some((_, text)) => ("user", (*text).to_string()),
            None => (
                if i % 2 == 0 { "user" } else { "assistant" },
                format!("{} (turn {i})", filler[i % filler.len()]),
            ),
        };

        let id = store.append_message(conv, role, &text, 0).unwrap();
        store
            .put_embedding(OwnerKind::Message, id, &embedder.embed(&text))
            .unwrap();
    }
    conv
}

#[test]
fn recalls_a_fact_from_the_start_of_a_long_conversation() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();

    let conv = conversation_with_needles(
        &store,
        &embedder,
        &[(4, "My postgres database is called warehouse_prod and it runs on port 6544.")],
        120,
    );

    let ctx = ContextBuilder::new(&store, &embedder)
        .with_budget(Budget { recent_messages: 8, ..Default::default() })
        .build(conv, "what port does my postgres database run on?")
        .unwrap();

    // The needle is 116 messages back, far outside the recent window.
    assert!(
        ctx.recent.iter().all(|m| m.seq != 4),
        "the answer must not be in the recent window, or the test proves nothing"
    );
    assert!(
        ctx.retrieved.iter().any(|h| h.text.contains("6544")),
        "the port must be recalled. retrieved: {:?}",
        ctx.retrieved.iter().map(|h| &h.text).collect::<Vec<_>>()
    );

    // And it must be cheap: nothing like the whole conversation.
    assert!(ctx.messages_elided > 100, "most messages should be left out");
    assert!(
        ctx.recent.len() + ctx.retrieved.len() < 20,
        "context should stay small, got {} items",
        ctx.recent.len() + ctx.retrieved.len()
    );
}

#[test]
fn exact_identifiers_are_found_where_embeddings_would_blur_them() {
    // This is what the lexical half of hybrid retrieval buys: an error code
    // has no semantic neighbourhood, so vector search alone tends to miss it.
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();

    let conv = conversation_with_needles(
        &store,
        &embedder,
        &[(2, "The deploy failed with error ERR_QUOTA_7731 on the staging cluster.")],
        100,
    );

    let ctx = ContextBuilder::new(&store, &embedder)
        .build(conv, "what was ERR_QUOTA_7731 about?")
        .unwrap();

    let hit = ctx
        .retrieved
        .iter()
        .find(|h| h.text.contains("ERR_QUOTA_7731"))
        .expect("the exact identifier must be recalled");
    assert!(hit.lexical_rank.is_some(), "lexical search should be what finds it");
}

#[test]
fn retrieval_finds_wording_that_differs_from_the_question() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();

    let conv = conversation_with_needles(
        &store,
        &embedder,
        &[(6, "I decided to offload the mixture of experts layers to system RAM.")],
        90,
    );

    let ctx = ContextBuilder::new(&store, &embedder)
        .build(conv, "which layers did I offload to RAM?")
        .unwrap();

    assert!(
        ctx.retrieved.iter().any(|h| h.text.contains("mixture of experts")),
        "should recall despite different phrasing: {:?}",
        ctx.retrieved.iter().map(|h| &h.text).collect::<Vec<_>>()
    );
}

#[test]
fn the_recent_window_is_never_duplicated_in_recall() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();

    let conv = conversation_with_needles(
        &store,
        &embedder,
        &[(58, "The magic phrase is xylophone parachute.")],
        60,
    );

    let ctx = ContextBuilder::new(&store, &embedder)
        .with_budget(Budget { recent_messages: 8, ..Default::default() })
        .build(conv, "what is the magic phrase?")
        .unwrap();

    let in_window: Vec<i64> = ctx.recent.iter().map(|m| m.id).collect();
    for hit in &ctx.retrieved {
        if hit.kind == OwnerKind::Message {
            assert!(
                !in_window.contains(&hit.id),
                "message {} appears both verbatim and as a recalled excerpt",
                hit.id
            );
        }
    }
}

#[test]
fn pinned_facts_are_always_present_regardless_of_the_question() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();
    let conv = conversation_with_needles(&store, &embedder, &[], 40);

    let fact = store
        .add_fact(Some(conv), Scope::User, "The user's name is Arfan.", None)
        .unwrap();
    store.set_pinned(fact, true).unwrap();

    // A question with no lexical or semantic relation to the fact.
    let ctx = ContextBuilder::new(&store, &embedder)
        .build(conv, "explain quicksort partitioning")
        .unwrap();

    assert!(
        ctx.pinned.iter().any(|f| f.text.contains("Arfan")),
        "pinned facts must not depend on retrieval"
    );

    let rendered = ctx.to_messages(Some("You are helpful."));
    assert!(
        rendered[0].text_content().contains("Arfan"),
        "pinned facts must reach the prompt: {}",
        rendered[0].text_content()
    );
}

#[test]
fn a_corrected_fact_replaces_the_original() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();
    let conv = conversation_with_needles(&store, &embedder, &[], 20);

    let old = store
        .add_fact(Some(conv), Scope::User, "The user's GPU is an RTX 3060.", None)
        .unwrap();
    let new = store
        .add_fact(Some(conv), Scope::User, "The user's GPU is an RTX 5050.", None)
        .unwrap();
    store.supersede_fact(old, new).unwrap();

    for f in [old, new] {
        store
            .put_embedding(
                OwnerKind::Fact,
                f,
                &embedder.embed(&store.get_fact(f).unwrap().unwrap().text),
            )
            .unwrap();
    }

    let ctx = ContextBuilder::new(&store, &embedder)
        .build(conv, "which GPU do I have?")
        .unwrap();

    let all: String = ctx.retrieved.iter().map(|h| h.text.as_str()).collect();
    assert!(all.contains("5050"), "the correction must be retrievable: {all:?}");
    assert!(!all.contains("3060"), "the superseded fact must not resurface: {all:?}");
}

#[test]
fn assembly_respects_the_token_budget() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();

    let conv = store.create_conversation("wordy", None).unwrap();
    for i in 0..60 {
        let text = format!("{} paragraph {i}", "a lot of text about databases ".repeat(20));
        let id = store.append_message(conv, "user", &text, 0).unwrap();
        store
            .put_embedding(OwnerKind::Message, id, &embedder.embed(&text))
            .unwrap();
    }

    let budget = Budget {
        total: 900,
        reserve_for_reply: 200,
        recent_messages: 20,
        max_retrieved: 10,
    };
    let ctx = ContextBuilder::new(&store, &embedder)
        .with_budget(budget.clone())
        .build(conv, "tell me about databases")
        .unwrap();

    assert!(
        ctx.tokens_used <= budget.usable(),
        "used {} tokens, budget was {}",
        ctx.tokens_used,
        budget.usable()
    );
    assert!(!ctx.recent.is_empty(), "at least the newest turn must survive");
}

#[test]
fn a_tight_budget_still_keeps_the_newest_turn() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();

    let conv = store.create_conversation("tight", None).unwrap();
    for i in 0..10 {
        store
            .append_message(conv, "user", &format!("{} {i}", "long ".repeat(200)), 0)
            .unwrap();
    }

    let ctx = ContextBuilder::new(&store, &embedder)
        .with_budget(Budget { total: 300, reserve_for_reply: 100, recent_messages: 8, max_retrieved: 2 })
        .build(conv, "what did I say?")
        .unwrap();

    assert_eq!(ctx.recent.len(), 1, "only the newest message can fit");
    assert!(ctx.recent[0].content.contains(" 9"), "and it must be the newest one");
}

#[test]
fn deleting_a_conversation_removes_its_messages_facts_and_vectors() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();
    let conv = conversation_with_needles(&store, &embedder, &[], 15);

    let fact = store.add_fact(Some(conv), Scope::Conversation, "some fact here", None).unwrap();
    store.put_embedding(OwnerKind::Fact, fact, &embedder.embed("some fact here")).unwrap();

    let embeddings_before: i64 = store
        .raw()
        .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
        .unwrap();
    assert!(embeddings_before > 0);

    store.delete_conversation(conv).unwrap();

    assert_eq!(store.message_count(conv).unwrap(), 0);
    assert!(store.get_conversation(conv).unwrap().is_none());
    let embeddings_after: i64 = store
        .raw()
        .query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))
        .unwrap();
    assert_eq!(embeddings_after, 0, "orphaned vectors would leak disk space forever");
}

#[test]
fn conversations_and_messages_round_trip() {
    let store = Store::open_in_memory().unwrap();
    let conv = store.create_conversation("test chat", Some("gemma4:12b")).unwrap();

    store.append_message(conv, "user", "first", 3).unwrap();
    store.append_message(conv, "assistant", "second", 4).unwrap();

    let msgs = store.messages(conv).unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].seq, 0);
    assert_eq!(msgs[1].seq, 1, "sequence numbers must be assigned in order");
    assert_eq!(msgs[1].role, "assistant");

    let listed = store.list_conversations(10).unwrap();
    assert_eq!(listed[0].id, conv);
    assert_eq!(listed[0].message_count, 2);
    assert_eq!(listed[0].model.as_deref(), Some("gemma4:12b"));
}

#[test]
fn a_query_full_of_fts_operators_does_not_error() {
    // User questions contain FTS5 syntax by accident all the time.
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();
    let conv = conversation_with_needles(&store, &embedder, &[(1, "config uses key: value pairs")], 20);

    for hostile in [
        "what about a AND b?",
        "the col:value syntax",
        "why did prefix* fail",
        "he said \"quoted thing\" earlier",
        "NEAR(a b) means what",
        "*",
        "",
    ] {
        let result = ContextBuilder::new(&store, &embedder).build(conv, hostile);
        assert!(result.is_ok(), "query {hostile:?} errored: {:?}", result.err());
    }
}

#[test]
fn an_empty_conversation_produces_empty_context() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();
    let conv = store.create_conversation("fresh", None).unwrap();

    let ctx = ContextBuilder::new(&store, &embedder).build(conv, "hello").unwrap();
    assert!(ctx.recent.is_empty());
    assert!(ctx.retrieved.is_empty());
    assert!(!ctx.used_recall());
    assert_eq!(ctx.messages_elided, 0);
    assert!(ctx.to_messages(None).is_empty(), "nothing to say means no messages");
}

#[test]
fn rendered_context_separates_recall_from_actual_turns() {
    let store = Store::open_in_memory().unwrap();
    let embedder = HashingEmbedder::default();
    let conv = conversation_with_needles(
        &store,
        &embedder,
        &[(3, "The staging server is at 10.0.4.19.")],
        50,
    );

    let ctx = ContextBuilder::new(&store, &embedder)
        .build(conv, "what is the staging server address?")
        .unwrap();
    let rendered = ctx.to_messages(Some("You are ozgent."));

    // Recalled text must arrive as system context, never forged as a turn the
    // user did not just say.
    let system = rendered
        .iter()
        .find(|m| matches!(m.role, ozgent_core::Role::System))
        .expect("a system message should carry the recall");
    assert!(system.text_content().contains("10.0.4.19"));
    assert!(system.text_content().contains("You are ozgent."));

    assert!(
        rendered[1..].iter().all(|m| !matches!(m.role, ozgent_core::Role::System)),
        "only one system message should be synthesised"
    );
}

#[test]
fn every_conversation_gets_a_usable_public_id() {
    // The web UI puts this in a URL, so it must be unique, stable, and shaped
    // like a UUID v4 — a bare row id would leak how many conversations exist.
    let store = Store::open_in_memory().unwrap();
    let mut seen = std::collections::HashSet::new();

    for i in 0..50 {
        let id = store.create_conversation(&format!("chat {i}"), None).unwrap();
        let conversation = store.get_conversation(id).unwrap().expect("just created");
        let uuid = conversation.uuid;

        assert_eq!(uuid.len(), 36, "not a uuid: {uuid}");
        let parts: Vec<&str> = uuid.split('-').collect();
        assert_eq!(parts.iter().map(|p| p.len()).collect::<Vec<_>>(), vec![8, 4, 4, 4, 12]);
        assert!(uuid.chars().all(|c| c.is_ascii_hexdigit() || c == '-'), "{uuid}");
        assert_eq!(parts[2].as_bytes()[0], b'4', "version nibble: {uuid}");
        assert!(matches!(parts[3].as_bytes()[0], b'8' | b'9' | b'a' | b'b'), "variant: {uuid}");
        assert!(seen.insert(uuid.clone()), "duplicate id {uuid} on iteration {i}");
    }
}

#[test]
fn a_conversation_can_be_found_by_its_public_id() {
    let store = Store::open_in_memory().unwrap();
    let id = store.create_conversation("findable", None).unwrap();
    let uuid = store.get_conversation(id).unwrap().unwrap().uuid;

    let found = store.conversation_by_uuid(&uuid).unwrap().expect("should resolve");
    assert_eq!(found.id, id);
    assert_eq!(found.title, "findable");

    assert!(
        store.conversation_by_uuid("00000000-0000-4000-8000-000000000000").unwrap().is_none(),
        "an unknown id must be absent, not an error"
    );
}

#[test]
fn empty_conversations_are_hidden_from_the_picker() {
    let store = Store::open_in_memory().unwrap();
    let empty = store.create_conversation("", Some("gemma4:12b")).unwrap();
    let used = store.create_conversation("real", Some("gemma4:12b")).unwrap();
    store.append_message(used, "user", "hello", 0).unwrap();

    let listed = store.list_active_conversations(50).unwrap();
    let ids: Vec<i64> = listed.iter().map(|c| c.id).collect();

    assert_eq!(ids, vec![used], "only the one with messages is offerable");
    assert_eq!(store.list_conversations(50).unwrap().len(), 2, "both still exist");
    assert!(ids.iter().all(|id| *id != empty));
}

#[test]
fn pruning_removes_the_empty_ones_and_spares_the_current() {
    let store = Store::open_in_memory().unwrap();
    let stale_a = store.create_conversation("", None).unwrap();
    let stale_b = store.create_conversation("", None).unwrap();
    let current = store.create_conversation("", None).unwrap();
    let used = store.create_conversation("real", None).unwrap();
    store.append_message(used, "user", "hello", 0).unwrap();

    assert_eq!(store.empty_conversation_count().unwrap(), 3);

    let removed = store.delete_empty_conversations(Some(current)).unwrap();
    assert_eq!(removed, 2, "the two stale ones, not the one being used");

    let left: Vec<i64> = store.list_conversations(50).unwrap().iter().map(|c| c.id).collect();
    assert!(left.contains(&current), "the live conversation must survive");
    assert!(left.contains(&used), "a conversation with messages is never empty");
    assert!(!left.contains(&stale_a) && !left.contains(&stale_b));
}

#[test]
fn pruning_with_nothing_to_keep_clears_them_all() {
    // `keep = None` has to mean "spare nothing", not "spare NULL" — a plain
    // `id != ?1` against NULL matches no row and would delete nothing.
    let store = Store::open_in_memory().unwrap();
    store.create_conversation("", None).unwrap();
    store.create_conversation("", None).unwrap();

    assert_eq!(store.delete_empty_conversations(None).unwrap(), 2);
    assert_eq!(store.empty_conversation_count().unwrap(), 0);
}
