//! Where tools come from, and how they are merged.
//!
//! ozgent's own tools are Python functions in a supervised process. An MCP
//! server is someone else's program offering more. Everything above this
//! layer — the model prompt, the tool-call grammar, the permission rules, the
//! terminal, the web interface, the messaging channels — must not care which
//! is which, or each of them acquires a second code path that can drift.
//!
//! So a [`Toolbox`] merges them and presents the same surface [`ToolHost`]
//! already did. What it adds beyond concatenation is the one thing that goes
//! wrong when several sources exist: two of them offering the same name. That
//! is resolved once, here, and reported — rather than being decided per call by
//! whichever lookup happened to run first.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use ozgent_core::ToolSpec;
use serde_json::Value;

use crate::host::{ToolCallError, ToolHost};

/// A future returned across a trait object.
pub type Boxed<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Somewhere tools come from, other than the Python worker.
pub trait ToolSource: Send + Sync {
    /// Where these tools are from, for `ozgent tools` and for a log line.
    /// Conventionally `mcp:<server>`.
    fn origin(&self) -> &str;

    /// What this source offers, already named as the model will see it.
    fn tools(&self) -> &[ToolSpec];

    /// Run one of them. `name` is the model-facing name from [`Self::tools`].
    ///
    /// `approved` says a person read this call and allowed it, and carries the
    /// same meaning it does for the Python worker: it lifts the *inner*
    /// boundaries a source keeps for calls made with nobody looking. A source
    /// with no such boundaries ignores it.
    fn call<'a>(
        &'a self,
        name: &'a str,
        arguments: Value,
        approved: bool,
    ) -> Boxed<'a, Result<Value, ToolCallError>>;

    /// Stop, if there is anything to stop.
    fn shutdown<'a>(&'a self) -> Boxed<'a, ()> {
        Box::pin(async {})
    }
}

/// Which source a tool came from.
#[derive(Debug, Clone, PartialEq)]
enum Owner {
    Python,
    Source(usize),
}

/// Every tool available, from every source.
pub struct Toolbox {
    python: Option<ToolHost>,
    sources: Vec<Arc<dyn ToolSource>>,
    specs: Vec<ToolSpec>,
    owner: HashMap<String, Owner>,
    /// Tools that were dropped because something already had the name.
    shadowed: Vec<Shadowed>,
}

/// A tool that could not be offered because its name was taken.
#[derive(Debug, Clone, PartialEq)]
pub struct Shadowed {
    pub name: String,
    /// Where the one that lost came from.
    pub from: String,
    /// Where the one that kept the name came from.
    pub kept: String,
}

impl Toolbox {
    /// Merge the Python worker and any other sources.
    ///
    /// The Python worker wins a name clash, and earlier sources beat later
    /// ones. The order is not arbitrary: ozgent's own tools are the ones a
    /// person configured on this machine, and a remote server should not be
    /// able to take a name out from under `write_file` by offering its own.
    pub fn new(python: Option<ToolHost>, sources: Vec<Arc<dyn ToolSource>>) -> Self {
        let mut specs: Vec<ToolSpec> = Vec::new();
        let mut owner: HashMap<String, Owner> = HashMap::new();
        let mut shadowed = Vec::new();
        let mut origin_of: HashMap<String, String> = HashMap::new();

        if let Some(host) = &python {
            for spec in host.tools() {
                owner.insert(spec.name.clone(), Owner::Python);
                origin_of.insert(spec.name.clone(), "python".to_string());
                specs.push(spec.clone());
            }
        }

        for (index, source) in sources.iter().enumerate() {
            for spec in source.tools() {
                if let Some(kept) = origin_of.get(&spec.name) {
                    shadowed.push(Shadowed {
                        name: spec.name.clone(),
                        from: source.origin().to_string(),
                        kept: kept.clone(),
                    });
                    continue;
                }
                owner.insert(spec.name.clone(), Owner::Source(index));
                origin_of.insert(spec.name.clone(), source.origin().to_string());
                specs.push(spec.clone());
            }
        }

        specs.sort_by(|a, b| a.name.cmp(&b.name));
        Self { python, sources, specs, owner, shadowed }
    }

    /// A toolbox with only the Python worker, which is the common case.
    pub fn from_python(python: ToolHost) -> Self {
        Self::new(Some(python), Vec::new())
    }

    pub fn tools(&self) -> &[ToolSpec] {
        &self.specs
    }

    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.specs.iter().find(|t| t.name == name)
    }

    /// Where a tool came from: a file path for a Python tool, `mcp:<server>`
    /// for one from a server.
    pub fn source_of(&self, name: &str) -> Option<String> {
        match self.owner.get(name)? {
            Owner::Python => self.python.as_ref()?.source_of(name).map(str::to_string),
            Owner::Source(i) => Some(self.sources.get(*i)?.origin().to_string()),
        }
    }

    /// Names that were dropped because something already had them.
    pub fn shadowed(&self) -> &[Shadowed] {
        &self.shadowed
    }

    /// The Python worker, for the things only it can answer.
    pub fn python(&self) -> Option<&ToolHost> {
        self.python.as_ref()
    }

    /// Everything that failed to load, from every source.
    pub fn load_errors(&self) -> Vec<String> {
        let mut out: Vec<String> =
            self.python.as_ref().map(|h| h.load_errors().to_vec()).unwrap_or_default();
        for s in &self.shadowed {
            out.push(format!(
                "{} from {} is not offered: {} already has that name",
                s.name, s.from, s.kept
            ));
        }
        out
    }

    /// Invoke a tool with no claim that anyone approved it.
    ///
    /// Defaults to unapproved for the same reason [`ToolHost::call`] does:
    /// forgetting to say a person allowed the call means a source's own
    /// boundaries apply, which is the harmless mistake. The harmful one would
    /// need someone to write `true`.
    pub async fn call(&self, name: &str, arguments: Value) -> Result<Value, ToolCallError> {
        self.call_approved(name, arguments, false).await
    }

    /// Invoke a tool, saying whether a person authorised this particular call.
    pub async fn call_approved(
        &self,
        name: &str,
        arguments: Value,
        approved: bool,
    ) -> Result<Value, ToolCallError> {
        match self.owner.get(name) {
            Some(Owner::Python) => {
                let host = self
                    .python
                    .as_ref()
                    .ok_or_else(|| ToolCallError::Transport("the tool worker is gone".into()))?;
                host.call_approved(name, arguments, approved).await
            }
            Some(Owner::Source(i)) => {
                let source = self
                    .sources
                    .get(*i)
                    .ok_or_else(|| ToolCallError::Transport("that tool source is gone".into()))?;
                source.call(name, arguments, approved).await
            }
            // Reported the way the Python worker reports it, so the model gets
            // one wording for "no such tool" however the toolbox is made up.
            None => Err(ToolCallError::Failed {
                name: name.to_string(),
                error: crate::protocol::RpcError {
                    code: -32601,
                    message: format!("there is no tool called {name}"),
                    data: None,
                },
            }),
        }
    }

    /// Shut every source down.
    pub async fn shutdown(&self) {
        for source in &self.sources {
            source.shutdown().await;
        }
        if let Some(host) = &self.python {
            host.shutdown().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ozgent_core::permission::Effect;

    struct Fake {
        origin: String,
        tools: Vec<ToolSpec>,
        answer: Value,
    }

    fn spec(name: &str, effect: Effect) -> ToolSpec {
        ToolSpec {
            name: name.into(),
            description: String::new(),
            input_schema: serde_json::json!({ "type": "object" }),
            output_schema: None,
            effect,
        }
    }

    fn fake(origin: &str, names: &[&str]) -> Arc<dyn ToolSource> {
        Arc::new(Fake {
            origin: origin.into(),
            tools: names.iter().map(|n| spec(n, Effect::Unknown)).collect(),
            answer: serde_json::json!({ "from": origin }),
        })
    }

    impl ToolSource for Fake {
        fn origin(&self) -> &str {
            &self.origin
        }
        fn tools(&self) -> &[ToolSpec] {
            &self.tools
        }
        fn call<'a>(
            &'a self,
            _name: &'a str,
            _arguments: Value,
            _approved: bool,
        ) -> Boxed<'a, Result<Value, ToolCallError>> {
            Box::pin(async move { Ok(self.answer.clone()) })
        }
    }

    #[tokio::test]
    async fn tools_from_every_source_are_offered_together() {
        let box_ = Toolbox::new(None, vec![fake("mcp:a", &["one"]), fake("mcp:b", &["two"])]);
        let names: Vec<&str> = box_.tools().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["one", "two"]);
        assert_eq!(box_.source_of("two").as_deref(), Some("mcp:b"));
    }

    #[tokio::test]
    async fn a_call_reaches_the_source_that_offered_it() {
        let box_ = Toolbox::new(None, vec![fake("mcp:a", &["one"]), fake("mcp:b", &["two"])]);
        let out = box_.call_approved("two", serde_json::json!({}), false).await.unwrap();
        assert_eq!(out, serde_json::json!({ "from": "mcp:b" }));
    }

    #[tokio::test]
    async fn the_first_source_to_claim_a_name_keeps_it() {
        // Whichever lookup happened to run first would otherwise decide, per
        // call, which of two identically-named tools ran.
        let box_ = Toolbox::new(None, vec![fake("mcp:a", &["same"]), fake("mcp:b", &["same"])]);
        assert_eq!(box_.tools().len(), 1);
        assert_eq!(box_.source_of("same").as_deref(), Some("mcp:a"));

        let out = box_.call_approved("same", serde_json::json!({}), false).await.unwrap();
        assert_eq!(out, serde_json::json!({ "from": "mcp:a" }));
    }

    #[test]
    fn a_dropped_tool_is_reported_rather_than_silently_missing() {
        // Otherwise a server is configured, appears to load, and one of its
        // tools simply is not there.
        let box_ = Toolbox::new(None, vec![fake("mcp:a", &["same"]), fake("mcp:b", &["same"])]);
        assert_eq!(
            box_.shadowed(),
            [Shadowed { name: "same".into(), from: "mcp:b".into(), kept: "mcp:a".into() }]
        );
        assert!(box_.load_errors()[0].contains("mcp:b"));
    }

    #[tokio::test]
    async fn calling_something_that_does_not_exist_reads_like_the_worker_says_it() {
        let box_ = Toolbox::new(None, Vec::new());
        let err = box_.call_approved("nope", serde_json::json!({}), false).await.unwrap_err();
        assert!(err.for_model().contains("no tool called nope"), "{}", err.for_model());
    }

    #[test]
    fn tools_are_listed_in_a_stable_order() {
        // The list goes into the prompt. Reordering it between runs would
        // change the prompt, and with it every cached prefix.
        let a = Toolbox::new(None, vec![fake("mcp:z", &["zebra", "apple"]), fake("mcp:a", &["mango"])]);
        let names: Vec<&str> = a.tools().iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["apple", "mango", "zebra"]);
    }

    #[test]
    fn an_empty_toolbox_is_valid_and_offers_nothing() {
        let box_ = Toolbox::new(None, Vec::new());
        assert!(box_.tools().is_empty());
        assert!(box_.get("anything").is_none());
        assert!(box_.load_errors().is_empty());
    }
}
