//! Running a flow.
//!
//! The engine owns the order, the data passed between steps and the record of
//! what happened. It does not own what a step *does*: calling a tool or a model
//! is behind [`Steps`], which the caller implements. So everything here — the
//! ordering, the branch semantics, what happens after a failure, how outputs
//! reach the next step — is testable without a GPU, a Python interpreter or a
//! network, which is most of what can actually go wrong.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::expr::{self, Context};
use crate::model::{Flow, Kind, NO, Node, OUT, YES};

/// What the engine needs someone else to do.
pub trait Steps {
    /// Call an installed tool.
    fn tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> impl std::future::Future<Output = Result<Value, String>> + Send;

    /// Ask the model something. `model` overrides the default when set.
    fn agent(
        &self,
        prompt: &str,
        model: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Value, String>> + Send;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Every step that was reached did what it was asked.
    Ok,
    /// At least one step failed. Steps after it were not run.
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Ran,
    /// Not reached: a condition went the other way, or something upstream
    /// failed. Recorded rather than omitted, so the canvas can grey it out and
    /// a reader can see the path that was taken.
    Skipped,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub id: String,
    pub label: String,
    pub kind: Kind,
    pub status: StepStatus,
    /// What the step produced, and what later expressions read.
    pub output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub status: Status,
    /// The trigger this run started from.
    pub trigger: String,
    pub steps: Vec<StepRecord>,
    pub ms: u64,
}

impl Run {
    pub fn failed(&self) -> Option<&StepRecord> {
        self.steps.iter().find(|s| s.status == StepStatus::Failed)
    }

    /// The outputs, keyed by step id, as later expressions would see them.
    pub fn outputs(&self) -> Context {
        self.steps
            .iter()
            .filter(|s| s.status == StepStatus::Ran)
            .map(|s| (s.id.clone(), s.output.clone()))
            .collect()
    }
}

/// Progress, as it happens.
#[derive(Debug, Clone)]
pub enum Progress {
    Started { id: String },
    Finished(Box<StepRecord>),
}

/// Somewhere to send progress, so a canvas can light steps up as they run.
pub type Reporter<'a> = dyn Fn(Progress) + Send + Sync + 'a;

/// Why a run could not be started at all.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Refused {
    #[error("this flow cannot run yet: {0}")]
    Invalid(String),
    #[error("`{0}` is not a step in this flow")]
    NoSuchTrigger(String),
    #[error("`{0}` does not start a run")]
    NotATrigger(String),
}

/// Run `flow`, starting at `trigger`.
pub async fn execute<S: Steps>(
    flow: &Flow,
    trigger: &str,
    payload: Value,
    steps: &S,
    report: Option<&Reporter<'_>>,
) -> Result<Run, Refused> {
    let problems = flow.problems();
    if !problems.is_empty() {
        let listed =
            problems.iter().map(|p| p.to_string()).collect::<Vec<_>>().join("; ");
        return Err(Refused::Invalid(listed));
    }
    let start = flow.node(trigger).ok_or_else(|| Refused::NoSuchTrigger(trigger.into()))?;
    if !start.kind.is_trigger() {
        return Err(Refused::NotATrigger(trigger.into()));
    }

    let began = Instant::now();
    let order = flow.order().map_err(|e| Refused::Invalid(e.to_string()))?;

    let mut context = Context::new();
    // Which ports each step actually emitted on. A wire is live only if its
    // source emitted on the port it leaves by, which is the whole of the
    // branch semantics.
    let mut emitted: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut records = Vec::with_capacity(order.len());
    let mut failed = false;

    for node in order {
        let reached = if node.kind.is_trigger() {
            // Only the trigger this run started from. The others are part of
            // the same flow but a different way in.
            node.id == start.id
        } else {
            flow.edges
                .iter()
                .filter(|e| e.to == node.id)
                .any(|e| emitted.get(&e.from).is_some_and(|ports| ports.contains(&e.port)))
        };

        if !reached {
            records.push(skipped(node));
            continue;
        }

        if let Some(report) = report {
            report(Progress::Started { id: node.id.clone() });
        }

        let step_began = Instant::now();
        let outcome = run_step(node, &payload, &context, steps).await;
        let ms = step_began.elapsed().as_millis() as u64;

        let record = match outcome {
            Ok(Emitted { output, port }) => {
                context.insert(node.id.clone(), output.clone());
                emitted.insert(node.id.clone(), BTreeSet::from([port]));
                StepRecord {
                    id: node.id.clone(),
                    label: node.label().to_string(),
                    kind: node.kind,
                    status: StepStatus::Ran,
                    output,
                    error: None,
                    ms,
                }
            }
            Err(message) => {
                // Nothing is recorded as emitted, so everything downstream is
                // skipped by the ordinary rule rather than by a special case.
                failed = true;
                StepRecord {
                    id: node.id.clone(),
                    label: node.label().to_string(),
                    kind: node.kind,
                    status: StepStatus::Failed,
                    output: Value::Null,
                    error: Some(message),
                    ms,
                }
            }
        };

        if let Some(report) = report {
            report(Progress::Finished(Box::new(record.clone())));
        }
        records.push(record);
    }

    Ok(Run {
        status: if failed { Status::Failed } else { Status::Ok },
        trigger: start.id.clone(),
        steps: records,
        ms: began.elapsed().as_millis() as u64,
    })
}

fn skipped(node: &Node) -> StepRecord {
    StepRecord {
        id: node.id.clone(),
        label: node.label().to_string(),
        kind: node.kind,
        status: StepStatus::Skipped,
        output: Value::Null,
        error: None,
        ms: 0,
    }
}

/// What a step produced, and which way the run leaves it.
struct Emitted {
    output: Value,
    port: String,
}

async fn run_step<S: Steps>(
    node: &Node,
    payload: &Value,
    context: &Context,
    steps: &S,
) -> Result<Emitted, String> {
    let out = |output: Value| Emitted { output, port: OUT.to_string() };

    match node.kind {
        // Whatever started the run is what a trigger produces, so
        // `{{ trigger.field }}` reads the incoming data.
        Kind::Manual | Kind::Schedule | Kind::Webhook => Ok(out(payload.clone())),

        Kind::Text => {
            let text = expr::interpolate(node.text("template"), context);
            Ok(out(json!({ "text": text })))
        }

        Kind::Agent => {
            let prompt = expr::interpolate(node.text("prompt"), context);
            if prompt.trim().is_empty() {
                return Err("this step has no prompt".into());
            }
            let model = node.param("model").and_then(|v| v.as_str()).filter(|m| !m.is_empty());
            steps.agent(&prompt, model).await.map(out)
        }

        Kind::Tool => {
            let name = node.text("tool");
            if name.is_empty() {
                return Err("this step does not say which tool to use".into());
            }
            let arguments = node
                .param("arguments")
                .map(|a| expr::fill(a, context))
                .unwrap_or_else(|| json!({}));
            steps.tool(name, arguments).await.map(out)
        }

        Kind::Condition => {
            let met = evaluate(node, context)?;
            Ok(Emitted {
                output: json!({ "met": met }),
                port: if met { YES.to_string() } else { NO.to_string() },
            })
        }
    }
}

/// Whether a condition holds.
pub fn evaluate(node: &Node, context: &Context) -> Result<bool, String> {
    let left = expr::resolve_value(node.text("left"), context);
    let right = expr::resolve_value(node.text("right"), context);
    let op = node.text("op");

    match op {
        "" | "eq" => Ok(alike(&left, &right)),
        "ne" => Ok(!alike(&left, &right)),
        "contains" => Ok(contains(&left, &right)),
        "gt" => order(&left, &right).map(|o| o.is_gt()),
        "lt" => order(&left, &right).map(|o| o.is_lt()),
        "empty" => Ok(is_empty(&left)),
        "not_empty" => Ok(!is_empty(&left)),
        other => Err(format!("`{other}` is not a comparison this understands")),
    }
}

/// Loose equality, as a workflow tool wants it.
///
/// `3` and `"3"` are equal here. That is a real choice and not an oversight: a
/// value arriving from a webhook is a string, the same value from a tool is a
/// number, and a condition that quietly went the wrong way because of which
/// side it came from would be extremely hard to see.
fn alike(a: &Value, b: &Value) -> bool {
    if a == b {
        return true;
    }
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => render(a) == render(b),
    }
}

fn order(a: &Value, b: &Value) -> Result<std::cmp::Ordering, String> {
    if let (Some(x), Some(y)) = (as_number(a), as_number(b)) {
        return x.partial_cmp(&y).ok_or_else(|| "these cannot be ordered".to_string());
    }
    Ok(render(a).cmp(&render(b)))
}

fn as_number(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str()?.trim().parse().ok())
}

fn contains(haystack: &Value, needle: &Value) -> bool {
    match haystack {
        Value::Array(items) => items.iter().any(|i| alike(i, needle)),
        Value::Object(map) => match needle.as_str() {
            Some(key) => map.contains_key(key),
            None => false,
        },
        _ => render(haystack).contains(&render(needle)),
    }
}

fn is_empty(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.trim().is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
    }
}

/// A value as text, for comparison. Strings are themselves; everything else is
/// its JSON form, so the comparison never depends on Rust's `Debug`.
fn render(v: &Value) -> String {
    match v.as_str() {
        Some(s) => s.to_string(),
        None => v.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Edge;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    /// A `Steps` that records what it was asked and answers from a script.
    #[derive(Default)]
    struct Fake {
        tools: Mutex<Vec<(String, Value)>>,
        prompts: Mutex<Vec<String>>,
        tool_answer: Option<Value>,
        agent_answer: Option<Value>,
        fail: Option<String>,
    }

    impl Steps for Fake {
        async fn tool(&self, name: &str, arguments: Value) -> Result<Value, String> {
            self.tools.lock().unwrap().push((name.to_string(), arguments));
            match &self.fail {
                Some(e) => Err(e.clone()),
                None => Ok(self.tool_answer.clone().unwrap_or(json!({ "ok": true }))),
            }
        }

        async fn agent(&self, prompt: &str, _model: Option<&str>) -> Result<Value, String> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            Ok(self.agent_answer.clone().unwrap_or(json!({ "text": "an answer" })))
        }
    }

    fn node(id: &str, kind: Kind, params: &[(&str, Value)]) -> Node {
        Node {
            id: id.into(),
            kind,
            name: String::new(),
            x: 0.0,
            y: 0.0,
            params: params.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
        }
    }

    fn edge(from: &str, to: &str, port: &str) -> Edge {
        Edge { from: from.into(), to: to.into(), port: port.into() }
    }

    fn flow(nodes: Vec<Node>, edges: Vec<Edge>) -> Flow {
        Flow { name: "t".into(), description: String::new(), nodes, edges }
    }

    fn status_of<'a>(run: &'a Run, id: &str) -> &'a StepRecord {
        run.steps.iter().find(|s| s.id == id).expect(id)
    }

    #[tokio::test]
    async fn a_step_reads_what_the_step_before_it_produced() {
        let f = flow(
            vec![
                node("t", Kind::Manual, &[]),
                node(
                    "s",
                    Kind::Tool,
                    &[("tool", json!("web_search")), ("arguments", json!({ "query": "{{ t.topic }}" }))],
                ),
                node("a", Kind::Agent, &[("prompt", json!("Summarise: {{ s.results }}"))]),
            ],
            vec![edge("t", "s", OUT), edge("s", "a", OUT)],
        );
        let fake = Fake { tool_answer: Some(json!({ "results": ["one", "two"] })), ..Default::default() };
        let run = execute(&f, "t", json!({ "topic": "otters" }), &fake, None).await.unwrap();

        assert_eq!(run.status, Status::Ok);
        assert_eq!(fake.tools.lock().unwrap()[0].1, json!({ "query": "otters" }));
        assert!(fake.prompts.lock().unwrap()[0].contains("\"one\""), "{:?}", fake.prompts);
    }

    #[tokio::test]
    async fn a_condition_runs_one_side_and_skips_the_other() {
        let f = flow(
            vec![
                node("t", Kind::Manual, &[]),
                node("c", Kind::Condition, &[("left", json!("{{ t.count }}")), ("op", json!("gt")), ("right", json!("2"))]),
                node("many", Kind::Text, &[("template", json!("plenty"))]),
                node("few", Kind::Text, &[("template", json!("not many"))]),
            ],
            vec![edge("t", "c", OUT), edge("c", "many", YES), edge("c", "few", NO)],
        );

        let run = execute(&f, "t", json!({ "count": 5 }), &Fake::default(), None).await.unwrap();
        assert_eq!(status_of(&run, "many").status, StepStatus::Ran);
        assert_eq!(status_of(&run, "few").status, StepStatus::Skipped);

        let run = execute(&f, "t", json!({ "count": 1 }), &Fake::default(), None).await.unwrap();
        assert_eq!(status_of(&run, "many").status, StepStatus::Skipped);
        assert_eq!(status_of(&run, "few").status, StepStatus::Ran);
    }

    #[tokio::test]
    async fn a_skipped_step_is_recorded_rather_than_left_out() {
        // The canvas greys it out, and a reader needs to see the path taken.
        let f = flow(
            vec![
                node("t", Kind::Manual, &[]),
                node("c", Kind::Condition, &[("left", json!("no")), ("op", json!("eq")), ("right", json!("yes"))]),
                node("never", Kind::Text, &[("template", json!("x"))]),
            ],
            vec![edge("t", "c", OUT), edge("c", "never", YES)],
        );
        let run = execute(&f, "t", json!({}), &Fake::default(), None).await.unwrap();
        assert_eq!(run.steps.len(), 3);
        assert_eq!(status_of(&run, "never").status, StepStatus::Skipped);
        assert_eq!(run.status, Status::Ok, "a road not taken is not a failure");
    }

    #[tokio::test]
    async fn a_failure_stops_what_depended_on_it_and_nothing_else() {
        let f = flow(
            vec![
                node("t", Kind::Manual, &[]),
                node("bad", Kind::Tool, &[("tool", json!("web_search"))]),
                node("after", Kind::Text, &[("template", json!("x"))]),
                node("beside", Kind::Text, &[("template", json!("y"))]),
            ],
            vec![edge("t", "bad", OUT), edge("bad", "after", OUT), edge("t", "beside", OUT)],
        );
        let fake = Fake { fail: Some("no api key".into()), ..Default::default() };
        let run = execute(&f, "t", json!({}), &fake, None).await.unwrap();

        assert_eq!(run.status, Status::Failed);
        assert_eq!(run.failed().unwrap().id, "bad");
        assert_eq!(run.failed().unwrap().error.as_deref(), Some("no api key"));
        assert_eq!(status_of(&run, "after").status, StepStatus::Skipped);
        assert_eq!(status_of(&run, "beside").status, StepStatus::Ran, "an unrelated branch");
    }

    #[tokio::test]
    async fn only_the_trigger_that_started_the_run_fires() {
        // A flow can be reachable several ways; running it manually must not
        // also fire its webhook path.
        let f = flow(
            vec![
                node("manual", Kind::Manual, &[]),
                node("hook", Kind::Webhook, &[]),
                node("m", Kind::Text, &[("template", json!("from manual"))]),
                node("h", Kind::Text, &[("template", json!("from hook"))]),
            ],
            vec![edge("manual", "m", OUT), edge("hook", "h", OUT)],
        );
        let run = execute(&f, "manual", json!({}), &Fake::default(), None).await.unwrap();
        assert_eq!(status_of(&run, "m").status, StepStatus::Ran);
        assert_eq!(status_of(&run, "h").status, StepStatus::Skipped);
        assert_eq!(status_of(&run, "hook").status, StepStatus::Skipped);
    }

    #[tokio::test]
    async fn a_step_fed_by_two_branches_runs_if_either_arrives() {
        let f = flow(
            vec![
                node("t", Kind::Manual, &[]),
                node("c", Kind::Condition, &[("left", json!("a")), ("op", json!("eq")), ("right", json!("a"))]),
                node("join", Kind::Text, &[("template", json!("done"))]),
            ],
            vec![edge("t", "c", OUT), edge("c", "join", YES), edge("c", "join", NO)],
        );
        let run = execute(&f, "t", json!({}), &Fake::default(), None).await.unwrap();
        assert_eq!(status_of(&run, "join").status, StepStatus::Ran);
    }

    #[tokio::test]
    async fn progress_is_reported_as_it_happens() {
        let f = flow(
            vec![node("t", Kind::Manual, &[]), node("x", Kind::Text, &[("template", json!("hi"))])],
            vec![edge("t", "x", OUT)],
        );
        let seen = Mutex::new(Vec::new());
        let report = |p: Progress| {
            seen.lock().unwrap().push(match p {
                Progress::Started { id } => format!("start {id}"),
                Progress::Finished(r) => format!("done {}", r.id),
            });
        };
        execute(&f, "t", json!({}), &Fake::default(), Some(&report)).await.unwrap();
        assert_eq!(
            *seen.lock().unwrap(),
            ["start t", "done t", "start x", "done x"]
        );
    }

    #[tokio::test]
    async fn a_flow_that_cannot_run_is_refused_before_anything_happens() {
        let f = flow(vec![node("a", Kind::Tool, &[("tool", json!("x"))])], vec![]);
        let fake = Fake::default();
        let refused = execute(&f, "a", json!({}), &fake, None).await.unwrap_err();
        assert!(matches!(refused, Refused::Invalid(_)), "{refused:?}");
        assert!(fake.tools.lock().unwrap().is_empty(), "nothing ran");
    }

    #[tokio::test]
    async fn a_run_cannot_be_started_from_a_step_that_is_not_a_trigger() {
        let f = flow(
            vec![node("t", Kind::Manual, &[]), node("x", Kind::Text, &[("template", json!("hi"))])],
            vec![edge("t", "x", OUT)],
        );
        assert_eq!(
            execute(&f, "x", json!({}), &Fake::default(), None).await.unwrap_err(),
            Refused::NotATrigger("x".into())
        );
        assert_eq!(
            execute(&f, "ghost", json!({}), &Fake::default(), None).await.unwrap_err(),
            Refused::NoSuchTrigger("ghost".into())
        );
    }

    #[tokio::test]
    async fn a_tool_step_with_no_tool_named_fails_with_a_readable_reason() {
        let f = flow(
            vec![node("t", Kind::Manual, &[]), node("x", Kind::Tool, &[])],
            vec![edge("t", "x", OUT)],
        );
        let run = execute(&f, "t", json!({}), &Fake::default(), None).await.unwrap();
        assert!(run.failed().unwrap().error.as_deref().unwrap().contains("which tool"));
    }

    // ------------------------------------------------------------ conditions

    fn condition(left: &str, op: &str, right: &str) -> Node {
        node(
            "c",
            Kind::Condition,
            &[("left", json!(left)), ("op", json!(op)), ("right", json!(right))],
        )
    }

    fn ctx() -> Context {
        BTreeMap::from([(
            "s".to_string(),
            json!({ "count": 3, "text": "hello world", "list": ["a", "b"], "blank": "" }),
        )])
    }

    #[test]
    fn a_number_and_the_same_number_written_as_text_are_equal() {
        // A webhook sends "3"; a tool returns 3. A condition that went the
        // wrong way depending on which is an extremely hard bug to see.
        assert_eq!(evaluate(&condition("{{ s.count }}", "eq", "3"), &ctx()), Ok(true));
        assert_eq!(evaluate(&condition("{{ s.count }}", "ne", "3"), &ctx()), Ok(false));
    }

    #[test]
    fn ordering_is_numeric_when_both_sides_are_numbers() {
        // And not lexicographic, where "10" sorts before "9".
        assert_eq!(evaluate(&condition("10", "gt", "9"), &ctx()), Ok(true));
        assert_eq!(evaluate(&condition("{{ s.count }}", "lt", "10"), &ctx()), Ok(true));
    }

    #[test]
    fn ordering_falls_back_to_text_when_it_has_to() {
        assert_eq!(evaluate(&condition("apple", "lt", "banana"), &ctx()), Ok(true));
    }

    #[test]
    fn contains_works_on_text_and_on_lists() {
        assert_eq!(evaluate(&condition("{{ s.text }}", "contains", "world"), &ctx()), Ok(true));
        assert_eq!(evaluate(&condition("{{ s.list }}", "contains", "b"), &ctx()), Ok(true));
        assert_eq!(evaluate(&condition("{{ s.list }}", "contains", "z"), &ctx()), Ok(false));
    }

    #[test]
    fn emptiness_covers_the_things_people_mean_by_it() {
        let c = ctx();
        assert_eq!(evaluate(&condition("{{ s.blank }}", "empty", ""), &c), Ok(true));
        assert_eq!(evaluate(&condition("{{ s.text }}", "empty", ""), &c), Ok(false));
        assert_eq!(evaluate(&condition("{{ s.list }}", "not_empty", ""), &c), Ok(true));
        // A reference that does not resolve stays as its own text, which is
        // not empty — so "empty" never quietly hides a broken reference.
        assert_eq!(evaluate(&condition("{{ s.missing }}", "empty", ""), &c), Ok(false));
    }

    #[test]
    fn an_unknown_comparison_is_an_error_and_not_a_silent_false() {
        let result = evaluate(&condition("a", "matches_regex", "b"), &ctx());
        assert!(result.is_err(), "{result:?}");
    }

    #[test]
    fn the_default_comparison_is_equality() {
        let n = node("c", Kind::Condition, &[("left", json!("a")), ("right", json!("a"))]);
        assert_eq!(evaluate(&n, &ctx()), Ok(true));
    }
}
