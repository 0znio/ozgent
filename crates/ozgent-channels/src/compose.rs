//! Turning a turn's events into the text of one chat message.
//!
//! A terminal redraws and a browser has a DOM; a chat app has one message that
//! can be rewritten a limited number of times. So the reply is *composed* — the
//! whole message is rebuilt from accumulated state after every event, and the
//! caller decides how often to actually send it.
//!
//! Tool activity is shown from the moment the model commits to a call rather
//! than when the call completes. Without that the chat simply stops: a model
//! writing a file spends the entire call generating `content`, and a minute of
//! silence on a phone reads as a dropped connection. The same reason the
//! terminal and the web interface show it early.

use std::collections::HashMap;

use ozgent_core::permission::Effect;
use ozgent_web::worker::Event;

/// How far a tool call has got.
#[derive(Debug, Clone, PartialEq)]
pub enum Stage {
    /// Named, and still being written by the model.
    Writing,
    /// Waiting for the person to allow it.
    Waiting,
    /// Allowed, and running.
    Running,
    Done { ok: bool, summary: String, ms: u64 },
    /// Refused, or never asked because the policy says no.
    Refused,
}

#[derive(Debug, Clone)]
struct Activity {
    /// The call id, once there is one. Absent while the call is still being
    /// written, and different from the id an early permission question
    /// carries, which is why results are matched by id *then* by name.
    id: Option<String>,
    name: String,
    effect: Effect,
    stage: Stage,
}

/// Accumulates a turn and renders it as markdown.
pub struct Composer {
    activities: Vec<Activity>,
    answer: String,
    error: Option<String>,
    /// Whether the model has produced reasoning but no answer yet.
    thinking: bool,
    /// What each tool does, so the "still being written" line can say what is
    /// being written. Supplied by the caller from the tool host.
    effects: HashMap<String, Effect>,
}

impl Composer {
    pub fn new(effects: HashMap<String, Effect>) -> Self {
        Self {
            activities: Vec::new(),
            answer: String::new(),
            error: None,
            thinking: false,
            effects,
        }
    }

    /// The answer text as it stands, without any activity lines.
    pub fn answer(&self) -> &str {
        self.answer.trim()
    }

    pub fn failed(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Whether anything has arrived that a reader would notice.
    pub fn started(&self) -> bool {
        !self.activities.is_empty() || !self.answer.trim().is_empty() || self.error.is_some()
    }

    pub fn absorb(&mut self, event: &Event) {
        match event {
            Event::Thinking { .. } => self.thinking = true,
            Event::Answer { text } => {
                self.answer.push_str(text);
                self.thinking = false;
            }
            Event::ToolCallStarted { name } => {
                let effect = self.effect_of(name);
                self.activities.push(Activity {
                    id: None,
                    name: name.clone(),
                    effect,
                    stage: Stage::Writing,
                });
            }
            Event::Permission { name, effect, .. } => {
                // The early question carries `early-<name>` rather than the
                // call id, so this is matched by name on purpose.
                match self.open_by_name(name) {
                    Some(a) => {
                        a.effect = *effect;
                        a.stage = Stage::Waiting;
                    }
                    None => self.activities.push(Activity {
                        id: None,
                        name: name.clone(),
                        effect: *effect,
                        stage: Stage::Waiting,
                    }),
                }
            }
            Event::ToolCall { id, name, .. } => {
                let effect = self.effect_of(name);
                match self.open_by_name(name) {
                    Some(a) => {
                        a.id = Some(id.clone());
                        a.stage = Stage::Running;
                    }
                    None => self.activities.push(Activity {
                        id: Some(id.clone()),
                        name: name.clone(),
                        effect,
                        stage: Stage::Running,
                    }),
                }
            }
            Event::ToolResult { id, name, ok, summary, ms, .. } => {
                // By id first, then by name: the line may have been opened
                // by an early permission question, which carries a different
                // id from the call that eventually runs.
                let found = self
                    .activities
                    .iter()
                    .rposition(|a| a.id.as_deref() == Some(id.as_str()))
                    .or_else(|| {
                        self.activities
                            .iter()
                            .rposition(|a| a.name == *name && !matches!(a.stage, Stage::Done { .. }))
                    });
                let stage = Stage::Done {
                    ok: *ok,
                    summary: first_line(summary),
                    ms: *ms,
                };
                match found {
                    Some(i) => {
                        // A call that never reached `Running` was never
                        // allowed to run: the worker announces `ToolCall` only
                        // for the calls that survived permission. So a failure
                        // on a line still waiting is a refusal, and saying
                        // "failed" would suggest something broke. This is read
                        // from the shape of the stream rather than from the
                        // wording of the message, which is a detail of the
                        // refusal text and free to change.
                        let never_ran =
                            matches!(self.activities[i].stage, Stage::Waiting | Stage::Writing);
                        self.activities[i].stage = if !*ok && never_ran {
                            Stage::Refused
                        } else {
                            stage
                        };
                    }
                    None => self.activities.push(Activity {
                        id: Some(id.clone()),
                        name: name.clone(),
                        effect: Effect::default(),
                        stage,
                    }),
                }
            }
            Event::Error { message } => self.error = Some(message.clone()),
            _ => {}
        }
    }

    /// Mark the tool a question was about as refused.
    ///
    /// The worker reports a refusal as an ordinary failed `ToolResult`, which
    /// is right for the model but reads wrong in a chat — "failed" suggests
    /// something broke. The gateway knows it was a refusal because it delivered
    /// the answer, so it says so here.
    pub fn refused(&mut self, name: &str) {
        if let Some(a) = self.open_by_name(name) {
            a.stage = Stage::Refused;
        }
    }

    fn effect_of(&self, name: &str) -> Effect {
        self.effects.get(name).copied().unwrap_or_default()
    }

    /// The most recent call of this name that has not finished.
    fn open_by_name(&mut self, name: &str) -> Option<&mut Activity> {
        self.activities
            .iter_mut()
            .rev()
            .find(|a| a.name == name && !matches!(a.stage, Stage::Done { .. } | Stage::Refused))
    }

    /// The whole message, as markdown.
    ///
    /// Never empty: a chat app refuses an empty message, and an edit to one
    /// would be a failed request rather than a cleared screen.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for a in &self.activities {
            out.push_str(&line(a));
            out.push('\n');
        }
        let answer = self.answer.trim();
        if !answer.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(answer);
        }
        if let Some(e) = &self.error {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(&format!("⚠️ {e}"));
        }
        if out.trim().is_empty() {
            out = if self.thinking { "⏳ thinking…".into() } else { "⏳ …".into() };
        }
        out.trim_end().to_string()
    }
}

fn line(a: &Activity) -> String {
    let name = format!("`{}`", a.name);
    match &a.stage {
        // The wording is the point: the person is waiting on generation they
        // cannot see, and "preparing" alone does not say that.
        Stage::Writing => match a.effect {
            Effect::Write => format!("⏳ {name} — generating what to write…"),
            Effect::Execute => format!("⏳ {name} — preparing the command…"),
            _ => format!("⏳ {name} — preparing…"),
        },
        Stage::Waiting => format!("❔ {name} — waiting for your answer"),
        Stage::Running => format!("⏳ {name} — running…"),
        Stage::Done { ok: true, summary, ms } => {
            let took = duration(*ms);
            if summary.is_empty() {
                format!("✓ {name} — {took}")
            } else {
                format!("✓ {name} — {summary} · {took}")
            }
        }
        Stage::Done { ok: false, summary, .. } => {
            if summary.is_empty() {
                format!("✗ {name} — failed")
            } else {
                format!("✗ {name} — {summary}")
            }
        }
        Stage::Refused => format!("✗ {name} — not allowed"),
    }
}

fn duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        format!("{:.1}s", ms as f64 / 1000.0)
    }
}

/// A tool's own one-line summary, bounded so a long one cannot push the reply
/// off the top of a phone screen.
fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or("").trim();
    if line.chars().count() <= 80 {
        return line.to_string();
    }
    let cut: String = line.chars().take(79).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn effects() -> HashMap<String, Effect> {
        HashMap::from([
            ("write_file".to_string(), Effect::Write),
            ("run_command".to_string(), Effect::Execute),
            ("web_search".to_string(), Effect::Read),
        ])
    }

    fn composer() -> Composer {
        Composer::new(effects())
    }

    fn started(name: &str) -> Event {
        Event::ToolCallStarted { name: name.into() }
    }
    fn call(id: &str, name: &str) -> Event {
        Event::ToolCall { id: id.into(), name: name.into(), arguments: serde_json::json!({}) }
    }
    fn result(id: &str, name: &str, ok: bool, summary: &str, ms: u64) -> Event {
        Event::ToolResult {
            id: id.into(),
            name: name.into(),
            ok,
            summary: summary.into(),
            ms,
            detail: serde_json::Value::Null,
        }
    }

    #[test]
    fn a_message_is_never_empty() {
        // Every chat app rejects an empty message, so an edit to one is a
        // failed API call rather than a blank screen.
        assert!(!composer().render().is_empty());
    }

    #[test]
    fn a_tool_is_visible_before_its_arguments_exist() {
        // The dwell this removes: a model writing a file generates the whole
        // file before the call can be parsed.
        let mut c = composer();
        c.absorb(&started("write_file"));
        let out = c.render();
        assert!(out.contains("write_file"), "{out}");
        assert!(out.contains("generating what to write"), "{out}");
    }

    #[test]
    fn the_waiting_wording_depends_on_what_the_tool_does() {
        let mut c = composer();
        c.absorb(&started("run_command"));
        assert!(c.render().contains("preparing the command"), "{}", c.render());

        let mut c = composer();
        c.absorb(&started("web_search"));
        assert!(c.render().contains("preparing…"), "{}", c.render());
    }

    #[test]
    fn one_call_produces_one_line_through_its_whole_life() {
        // The bug this pins is a call appearing three times — once when
        // announced, once when asked about, once when run.
        let mut c = composer();
        c.absorb(&started("write_file"));
        c.absorb(&Event::Permission {
            id: "early-write_file".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({}),
            effect: Effect::Write,
        });
        c.absorb(&call("c1", "write_file"));
        c.absorb(&result("c1", "write_file", true, "wrote poem.txt", 40));

        let out = c.render();
        assert_eq!(out.matches("write_file").count(), 1, "{out}");
        assert!(out.contains("✓"), "{out}");
        assert!(out.contains("wrote poem.txt"), "{out}");
    }

    #[test]
    fn an_early_question_is_matched_by_name_because_its_id_differs() {
        // `permit` asks early under `early-<name>`; the real call gets a
        // different id. Matching on id alone would show two lines.
        let mut c = composer();
        c.absorb(&started("write_file"));
        c.absorb(&Event::Permission {
            id: "early-write_file".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({}),
            effect: Effect::Write,
        });
        assert!(c.render().contains("waiting for your answer"), "{}", c.render());
        assert_eq!(c.render().matches("write_file").count(), 1);
    }

    #[test]
    fn several_tools_in_one_turn_each_keep_their_own_line() {
        // A batch is announced in full before any of it is awaited; a
        // composer that tracked only "the current call" would land the first
        // result on the wrong line.
        let mut c = composer();
        c.absorb(&call("c1", "web_search"));
        c.absorb(&call("c2", "fetch_url"));
        c.absorb(&result("c2", "fetch_url", true, "fetched", 10));

        let out = c.render();
        assert!(out.contains("⏳ `web_search`"), "{out}");
        assert!(out.contains("✓ `fetch_url`"), "{out}");
    }

    #[test]
    fn the_same_tool_called_twice_gets_two_lines() {
        let mut c = composer();
        c.absorb(&call("c1", "web_search"));
        c.absorb(&result("c1", "web_search", true, "4 results", 900));
        c.absorb(&call("c2", "web_search"));
        assert_eq!(c.render().matches("web_search").count(), 2, "{}", c.render());
    }

    #[test]
    fn a_refusal_says_so_rather_than_reporting_a_failure() {
        let mut c = composer();
        c.absorb(&started("run_command"));
        c.refused("run_command");
        let out = c.render();
        assert!(out.contains("not allowed"), "{out}");
        assert!(!out.contains("failed"), "{out}");
    }

    #[test]
    fn a_declined_call_reads_as_refused_and_not_as_broken() {
        // The worker reports a refusal as a failed result, and the only thing
        // separating it from a tool that genuinely broke is that it was never
        // announced as running.
        let mut c = composer();
        c.absorb(&started("write_file"));
        c.absorb(&Event::Permission {
            id: "early-write_file".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({}),
            effect: Effect::Write,
        });
        c.absorb(&result("c1", "write_file", false, "The user declined to run write_file.", 0));

        let out = c.render();
        assert!(out.contains("not allowed"), "{out}");
        assert!(!out.contains("declined to run"), "the model's wording is not the reader's: {out}");
    }

    #[test]
    fn a_tool_that_ran_and_broke_still_reads_as_a_failure() {
        // The other half of the same rule: once a call has been announced as
        // running, a failure is a real failure and must not be softened into
        // a refusal.
        let mut c = composer();
        c.absorb(&call("c1", "fetch_url"));
        c.absorb(&result("c1", "fetch_url", false, "connection refused", 30));

        let out = c.render();
        assert!(out.contains("connection refused"), "{out}");
        assert!(!out.contains("not allowed"), "{out}");
    }

    #[test]
    fn the_answer_follows_the_activity() {
        let mut c = composer();
        c.absorb(&call("c1", "web_search"));
        c.absorb(&result("c1", "web_search", true, "4 results", 1200));
        c.absorb(&Event::Answer { text: "The answer is 42.".into() });

        let out = c.render();
        let tool_at = out.find("web_search").unwrap();
        let answer_at = out.find("The answer").unwrap();
        assert!(tool_at < answer_at, "{out}");
        assert_eq!(c.answer(), "The answer is 42.");
    }

    #[test]
    fn timings_read_as_a_person_would_say_them() {
        assert_eq!(duration(40), "40ms");
        assert_eq!(duration(1200), "1.2s");
    }

    #[test]
    fn a_long_tool_summary_cannot_push_the_reply_off_the_screen() {
        let mut c = composer();
        c.absorb(&call("c1", "web_search"));
        c.absorb(&result("c1", "web_search", true, &"x".repeat(500), 10));
        let out = c.render();
        assert!(out.lines().next().unwrap().chars().count() < 120, "{out}");
    }

    #[test]
    fn a_multi_line_summary_is_reduced_to_its_first_line() {
        assert_eq!(first_line("one\ntwo\nthree"), "one");
    }

    #[test]
    fn an_error_is_shown_with_whatever_arrived_first() {
        let mut c = composer();
        c.absorb(&Event::Answer { text: "partial".into() });
        c.absorb(&Event::Error { message: "the model stopped".into() });
        let out = c.render();
        assert!(out.contains("partial"), "{out}");
        assert!(out.contains("the model stopped"), "{out}");
        assert_eq!(c.failed(), Some("the model stopped"));
    }

    #[test]
    fn thinking_is_shown_only_until_there_is_something_better() {
        let mut c = composer();
        c.absorb(&Event::Thinking { text: "hmm".into() });
        assert!(c.render().contains("thinking"), "{}", c.render());
        assert!(!c.started(), "reasoning alone is not a reply");

        c.absorb(&Event::Answer { text: "done".into() });
        assert_eq!(c.render(), "done");
        assert!(c.started());
    }
}
