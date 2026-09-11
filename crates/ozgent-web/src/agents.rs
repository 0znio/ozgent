//! Agents, as the HTTP APIs see them.
//!
//! An API client can reach an agent two ways, and both have to work without
//! the client knowing agents exist:
//!
//! * **By mention.** `@stock-guru` in the latest user message, exactly as in
//!   the chat. Only the latest message: a mention three turns back in the
//!   history the client resends has already been answered.
//! * **By model name.** Agents are listed in `/v1/models` as `@name`, so a
//!   client with a model picker can select one and every message goes to it.
//!   It runs on the server's default model.
//!
//! Neither protocol has anywhere to put "an agent is working, here is what it
//! did", so that goes where both clients already show a model's working: the
//! reasoning stream. A client that ignores reasoning still gets the report as
//! the answer, which is the part that matters.

use ozgent_core::{Agent, AgentCatalog};

use crate::state::State;
use crate::worker::Event;

/// What an API request asked for, once agents are accounted for.
pub struct Resolved {
    /// The model that will actually generate.
    pub model: ozgent_core::Installed,
    /// Agents to run, in order. Empty for an ordinary request.
    pub agents: Vec<Agent>,
}

/// Why a request naming an agent cannot be served.
pub enum Refusal {
    /// No such model or agent. Carries the names that do exist.
    NotFound(String),
    /// Something the caller must change.
    BadRequest(String),
}

/// Work out which model runs and which agents, if any, it runs as.
///
/// `enabled` is the request's `ozgent_agents` field: `false` switches mention
/// detection off for a client whose users type `@` for other reasons. A model
/// named `@agent` is an explicit choice and is honoured either way.
pub fn resolve(
    state: &State,
    model: &str,
    latest_user_text: &str,
    enabled: bool,
) -> Result<Resolved, Refusal> {
    let catalog = AgentCatalog::load(&state.paths);

    if let Some(name) = model.strip_prefix('@') {
        let agent = catalog.get(name).cloned().ok_or_else(|| {
            let known: Vec<String> = catalog.all().iter().map(|a| format!("@{}", a.name)).collect();
            Refusal::NotFound(format!("no agent {model:?}. Agents: {}", known.join(", ")))
        })?;
        let base = state.config.lock().unwrap().default_model.clone().ok_or_else(|| {
            Refusal::BadRequest(format!(
                "{model} runs on the default model, and none is set. \
                 Run `ozgent default <model>`, or name a model and write {model} in the message"
            ))
        })?;
        let model = ozgent_core::resolve(&state.paths, &base)
            .map_err(|e| Refusal::NotFound(format!("the default model {base:?}: {e}")))?;
        // A mention in the message can add agents after the one selected.
        let mut agents = vec![agent];
        if enabled {
            for more in catalog.mentioned(latest_user_text) {
                if agents.len() < ozgent_core::agents::MAX_PER_MESSAGE
                    && !agents.iter().any(|a| a.name == more.name)
                {
                    agents.push(more.clone());
                }
            }
        }
        return Ok(Resolved { model, agents });
    }

    let model = ozgent_core::resolve(&state.paths, model).map_err(|_| {
        let mut names: Vec<String> = ozgent_core::installed(&state.paths)
            .iter()
            .map(|m| m.manifest.alias.clone().unwrap_or_else(|| m.model.to_string()))
            .collect();
        names.extend(catalog.all().iter().map(|a| format!("@{}", a.name)));
        Refusal::NotFound(format!("no model {model:?}. Available: {}", names.join(", ")))
    })?;
    let agents = if enabled {
        catalog.mentioned(latest_user_text).into_iter().cloned().collect()
    } else {
        Vec::new()
    };
    Ok(Resolved { model, agents })
}

/// Agents as `/v1/models` entries, so a model picker offers them.
pub fn as_models(state: &State) -> Vec<serde_json::Value> {
    let base = state.config.lock().unwrap().default_model.clone();
    AgentCatalog::load(&state.paths)
        .all()
        .iter()
        .map(|a| {
            serde_json::json!({
                "id": format!("@{}", a.name),
                "object": "model",
                "type": "model",
                "created": 0,
                "created_at": "1970-01-01T00:00:00Z",
                "owned_by": "ozgent-agent",
                "display_name": format!("@{} — {}", a.name, a.definition.description),
                "description": a.definition.description,
                "agent": true,
                "base_model": base,
                "tools": a.definition.tools,
                "capabilities": ["completion", "agent"],
            })
        })
        .collect()
}

/// A human-readable account of what agents are doing, for a reasoning stream.
///
/// Tool calls made *outside* an agent are left alone: a request that asked for
/// server tools has always been answered without narrating them, and changing
/// that would put text into streams that did not ask for it.
#[derive(Default)]
pub struct Trace {
    inside: Option<String>,
    /// Arguments by call id, so a result can say what it was a result of.
    pending: std::collections::HashMap<String, String>,
}

impl Trace {
    /// The line this event adds to the trace, if any.
    pub fn line(&mut self, event: &Event) -> Option<String> {
        match event {
            Event::AgentStart { name, tools, missing, .. } => {
                self.inside = Some(name.clone());
                let mut line = format!("@{name} is working on this");
                if !tools.is_empty() {
                    line.push_str(&format!(" · tools: {}", tools.join(", ")));
                }
                if !missing.is_empty() {
                    line.push_str(&format!(" · unavailable: {}", missing.join(", ")));
                }
                Some(format!("{line}\n"))
            }
            Event::AgentEnd { name, ok, ms, calls, .. } => {
                self.inside = None;
                let how = if *ok { "finished" } else { "produced no report" };
                let plural = if *calls == 1 { "" } else { "s" };
                Some(format!(
                    "@{name} {how} · {calls} tool call{plural} · {:.1}s\n\n",
                    *ms as f64 / 1000.0
                ))
            }
            Event::ToolCall { id, name, arguments } if self.inside.is_some() => {
                let args = brief(arguments);
                self.pending.insert(id.clone(), args.clone());
                Some(format!("→ {name} {args}\n"))
            }
            Event::ToolResult { name, ok, summary, ms, id, .. } if self.inside.is_some() => {
                // A result with no call before it is a refusal, or a tool the
                // agent was not given; say which tool it was about.
                let head = if self.pending.remove(id).is_some() {
                    String::new()
                } else {
                    format!("→ {name}\n")
                };
                let mark = if *ok { "✓" } else { "✗" };
                Some(format!("{head}  {mark} {summary} · {ms} ms\n"))
            }
            _ => None,
        }
    }

    /// Whether an agent is running, so its reasoning can be labelled as such.
    pub fn in_agent(&self) -> bool {
        self.inside.is_some()
    }
}

/// Arguments as `key=value` pairs on one short line.
fn brief(arguments: &serde_json::Value) -> String {
    let Some(map) = arguments.as_object() else { return String::new() };
    map.iter()
        .map(|(k, v)| {
            let v = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let v: String = v.chars().take(60).collect();
            format!("{k}={v}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The structured form of an event, for clients that want to draw an agent
/// view of their own. Carried in an `ozgent` field the protocols do not
/// define, which every client that does not know it simply ignores.
pub fn structured(event: &Event) -> Option<serde_json::Value> {
    match event {
        Event::AgentStart { .. }
        | Event::AgentEnd { .. }
        | Event::ToolCall { .. }
        | Event::ToolResult { .. } => serde_json::to_value(event).ok().map(|mut v| {
            // A tool's full result can be large; the trace and the model
            // already have it. The summary is what a view shows.
            if let Some(obj) = v.as_object_mut() {
                obj.remove("detail");
            }
            v
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trace_narrates_an_agent_and_nothing_outside_it() {
        let mut t = Trace::default();
        let outside = Event::ToolCall {
            id: "call_0".into(),
            name: "web_search".into(),
            arguments: serde_json::json!({"query": "x"}),
        };
        assert!(t.line(&outside).is_none(), "a plain turn's tools are not narrated");

        let start = Event::AgentStart {
            name: "stock-guru".into(),
            description: "d".into(),
            tools: vec!["yahoo_finance".into()],
            missing: vec!["reddit".into()],
        };
        assert_eq!(
            t.line(&start).unwrap(),
            "@stock-guru is working on this · tools: yahoo_finance · unavailable: reddit\n"
        );
        let call = Event::ToolCall {
            id: "stock-guru_call_0".into(),
            name: "yahoo_finance".into(),
            arguments: serde_json::json!({"action": "quote", "symbol": "NVDA"}),
        };
        assert_eq!(t.line(&call).unwrap(), "→ yahoo_finance action=quote symbol=NVDA\n");
        let result = Event::ToolResult {
            id: "stock-guru_call_0".into(),
            name: "yahoo_finance".into(),
            ok: true,
            summary: "1 quote".into(),
            ms: 320,
            detail: serde_json::json!({}),
        };
        assert_eq!(t.line(&result).unwrap(), "  ✓ 1 quote · 320 ms\n");
        let refused = Event::ToolResult {
            id: "stock-guru_call_1".into(),
            name: "write_file".into(),
            ok: false,
            summary: "write_file is not one of your tools".into(),
            ms: 0,
            detail: serde_json::json!({}),
        };
        assert!(t.line(&refused).unwrap().starts_with("→ write_file\n  ✗"));
        let end = Event::AgentEnd { name: "stock-guru".into(), ok: true, ms: 12_340, calls: 1, rounds: 2 };
        assert_eq!(t.line(&end).unwrap(), "@stock-guru finished · 1 tool call · 12.3s\n\n");
        assert!(!t.in_agent());
    }

    #[test]
    fn the_structured_form_leaves_out_the_bulky_result() {
        let v = structured(&Event::ToolResult {
            id: "c".into(),
            name: "n".into(),
            ok: true,
            summary: "s".into(),
            ms: 1,
            detail: serde_json::json!({"huge": "x"}),
        })
        .unwrap();
        assert_eq!(v["type"], "tool_result");
        assert!(v.get("detail").is_none());
        assert!(structured(&Event::Answer { text: "hi".into() }).is_none());
    }
}
