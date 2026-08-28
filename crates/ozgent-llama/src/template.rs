//! Rendering a model's own chat template.
//!
//! llama.cpp ships C++ reimplementations of the popular templates and picks
//! one by matching substrings of the Jinja source. They are approximations,
//! and the place they diverge matters most: both Qwen3.5 and Ling 3.0 end
//! their generation prompt *inside* an open `<think>` block, and neither
//! built-in writes it. The model reasons anyway — it was trained to — but
//! emits only the closing tag, so the trace arrives looking exactly like an
//! answer and the whole reasoning split collapses.
//!
//! Guessing the prefill from outside does not work either: whether a turn
//! reasons at all is a decision the template makes, from the thinking option,
//! the system message and the model family's own defaults. The only way to
//! know what the prompt should be is to run the template the model shipped.
//!
//! Rendering is best-effort. A template this cannot compile or execute falls
//! back to llama.cpp's built-in, which is what ozgent used before and is still
//! right for the many models whose templates the built-ins render exactly.


use minijinja::value::Kwargs;
use minijinja::{Environment, Value, context};
use ozgent_core::{Message, Role};

/// A compiled chat template, ready to render turns.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
    /// Whether this template documents tools to the model itself.
    ///
    /// Probed once at compile time rather than guessed from the source, and
    /// it decides something important: a template with a `tools` block tells
    /// the model its *own* call format, so ozgent's generic preamble would be
    /// a second, contradictory instruction. See [`Self::handles_tools`].
    handles_tools: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("compiling the chat template: {0}")]
    Compile(String),
    #[error("rendering the chat template: {0}")]
    Render(String),
}

/// What the caller wants of this turn, in the vocabulary templates expect.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// `None` leaves `enable_thinking` undefined so the template's own default
    /// applies — which is what "auto" means. Forcing it either way overrides a
    /// model that has a considered opinion about when to reason.
    pub enable_thinking: Option<bool>,
    /// Whether to append the assistant's opening. Cleared only to measure how
    /// long that opening is; every real render wants it.
    pub add_generation_prompt: bool,
    /// How hard to think, in the vocabulary templates use: "low", "medium",
    /// "high". Asking is better than interrupting — a model that decides for
    /// itself to reason briefly still finishes its thought, where a token
    /// budget stops it mid-argument and makes it answer from an unfinished
    /// one. Only some templates read this; for the rest it is inert and the
    /// budget remains the only control.
    pub reasoning_effort: Option<String>,
    /// Tools to describe, in the shape `apply_chat_template` passes them:
    /// `{"type": "function", "function": {name, description, parameters}}`.
    /// Empty leaves `tools` undefined, which every template treats as "none".
    pub tools: Vec<serde_json::Value>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: None,
            reasoning_effort: None,
            tools: Vec::new(),
        }
    }
}

/// One tool in the form templates expect.
///
/// The OpenAI wrapper, not the bare schema: it is what `apply_chat_template`
/// passes and therefore what the model saw during training. Qwen's template
/// renders each entry with `| tojson` straight into its `<tools>` block, so
/// the shape reaches the model verbatim.
pub fn tool_json(spec: &ozgent_core::ToolSpec) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        }
    })
}

impl ChatTemplate {
    /// Compile `source`, the raw Jinja from the GGUF.
    pub fn new(source: &str, bos: String, eos: String) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        // Templates are written against Python's Jinja2 and call Python string
        // methods on their values — `.split()`, `.rstrip()`, `.startswith()`.
        // Without this every one of them is an unknown-method error.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Python's `json.dumps` keywords, which real templates pass and
        // minijinja's own `tojson` rejects as unknown. Ling 3.0 calls
        // `tojson(ensure_ascii=False)`, and the error aborted the whole
        // template — silently, into llama.cpp's built-in, which does not open
        // the reasoning block Ling's own template ends inside.
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.add_template_owned("chat", source.to_string())
            .map_err(|e| TemplateError::Compile(e.to_string()))?;
        let mut tmpl = Self { env, bos, eos, handles_tools: false };
        tmpl.handles_tools = tmpl.probe_tools();
        Ok(tmpl)
    }

    /// Whether the template documents tools to the model.
    ///
    /// When true, passing tools makes the template write the model's *own*
    /// call format, and ozgent's generic preamble must be left out — two
    /// descriptions of two different formats is worse than either alone.
    pub fn handles_tools(&self) -> bool {
        self.handles_tools
    }

    /// Detect a `tools` block by rendering one and looking for it.
    ///
    /// Probed rather than pattern-matched on the source: the word "tools"
    /// appears in the prose of templates that have no tool support at all,
    /// and a false positive here silently removes the only instructions a
    /// model would have received.
    fn probe_tools(&self) -> bool {
        const SENTINEL: &str = "ozgent_probe_tool_name";
        let probe = [Message::user("x")];
        let opts = RenderOptions {
            tools: vec![serde_json::json!({
                "type": "function",
                "function": {
                    "name": SENTINEL,
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}},
                }
            })],
            ..Default::default()
        };
        match self.render(&probe, opts) {
            Ok(out) => out.contains(SENTINEL),
            // A template that cannot render with tools cannot be given them.
            Err(_) => false,
        }
    }

    /// Render a conversation into a prompt the model should continue.
    pub fn render(
        &self,
        messages: &[Message],
        opts: RenderOptions,
    ) -> Result<String, TemplateError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| TemplateError::Render(e.to_string()))?;

        let turns: Vec<serde_json::Value> = messages.iter().map(turn_json).collect();

        let mut ctx = context! {
            messages => Value::from_serialize(&turns),
            add_generation_prompt => opts.add_generation_prompt,
            bos_token => self.bos,
            eos_token => self.eos,
        };
        if !opts.tools.is_empty() {
            ctx = context! { tools => Value::from_serialize(&opts.tools), ..ctx };
        }
        if let Some(on) = opts.enable_thinking {
            ctx = context! { enable_thinking => on, ..ctx };
        }
        if let Some(effort) = opts.reasoning_effort.as_deref() {
            ctx = context! { reasoning_effort => effort, ..ctx };
        }

        tmpl.render(ctx).map_err(|e| TemplateError::Render(chain(&e)))
    }
}

/// One message in the shape templates read it.
///
/// Deliberately omits any key the message does not carry: templates test with
/// `is defined` and `if message.tool_calls`, so a present-but-empty key is not
/// the same as an absent one.
fn turn_json(m: &Message) -> serde_json::Value {
    let mut turn = serde_json::Map::new();
    turn.insert("role".into(), role_name(m.role).into());
    turn.insert("content".into(), m.text_content().into());

    // The model's own reasoning, under the name every template that supports
    // one uses. Qwen 3.5 reads `reasoning_content` and, finding none, writes
    // an empty `<think></think>` for the turn — which tells the model it did
    // not reason on a turn where it did. Across a tool-calling loop that
    // erases its whole chain of thought.
    if let Some(reasoning) = &m.thinking {
        if !reasoning.trim().is_empty() {
            turn.insert("reasoning_content".into(), reasoning.clone().into());
        }
    }

    if !m.tool_calls.is_empty() {
        // `arguments` stays an object rather than a JSON string. Qwen's
        // template iterates it to write one `<parameter=name>` block per
        // entry; handed a string it renders nothing and the call is silently
        // dropped from the history.
        let calls: Vec<serde_json::Value> = m
            .tool_calls
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments },
                })
            })
            .collect();
        turn.insert("tool_calls".into(), calls.into());
    }

    if let Some(id) = &m.tool_call_id {
        turn.insert("tool_call_id".into(), id.clone().into());
        // Some templates key the response off the name instead of the id.
        turn.insert("name".into(), id.clone().into());
    }

    serde_json::Value::Object(turn)
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// `tojson`, accepting the keywords `json.dumps` takes.
///
/// `ensure_ascii` is honoured rather than merely tolerated: a template that
/// asks for escaped output and receives raw UTF-8 is being given something
/// other than what it asked for, and the difference reaches the model.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent").unwrap_or(None);
    let ensure_ascii: bool = kwargs.get("ensure_ascii").unwrap_or(Some(true)).unwrap_or(true);
    kwargs.assert_all_used()?;

    let json = serde_json::to_value(&value).map_err(|e| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
    })?;
    let text = match indent {
        // Python indents with spaces; serde's pretty printer uses two, which
        // is what every template that asks for indent=2 expects.
        Some(_) => serde_json::to_string_pretty(&json),
        None => serde_json::to_string(&json),
    }
    .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))?;

    Ok(Value::from(if ensure_ascii { escape_non_ascii(&text) } else { text }))
}

/// Rewrite non-ASCII characters as `\uXXXX`, the way `json.dumps` does.
fn escape_non_ascii(text: &str) -> String {
    if text.is_ascii() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            // Astral-plane characters need the surrogate pair JSON uses.
            let mut buf = [0u16; 2];
            for unit in ch.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// Templates call this to reject a conversation they cannot represent.
fn raise_exception(message: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message))
}

/// Some templates date-stamp the system message. There is no formatting here
/// worth pulling a date library in for; the templates that call this want a
/// plausible current date, not a specific one.
fn strftime_now(_format: String) -> String {
    String::new()
}

/// minijinja reports the useful part of a failure in the error's source chain,
/// so the top-level message alone often says only "invalid operation".
fn chain(err: &minijinja::Error) -> String {
    let mut out = err.to_string();
    let mut source = std::error::Error::source(err);
    while let Some(e) = source {
        out.push_str(": ");
        out.push_str(&e.to_string());
        source = e.source();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(source: &str, opts: RenderOptions) -> String {
        ChatTemplate::new(source, "<s>".into(), "</s>".into())
            .expect("compiles")
            .render(&[Message::user("hi")], opts)
            .expect("renders")
    }

    #[test]
    fn a_generation_prompt_can_end_inside_a_reasoning_block() {
        // The shape both Qwen3.5 and Ling 3.0 use, and the one llama.cpp's
        // built-in renderers drop.
        let src = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}\
                   {% if add_generation_prompt %}<|assistant|>\n<think>\n{% endif %}";
        let out = render(src, RenderOptions::default());
        assert!(out.ends_with("<think>\n"), "{out:?}");
    }

    /// The real Qwen 3.5 template, as shipped in the GGUF.
    fn qwen_template() -> Option<ChatTemplate> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen3.5.j2");
        let src = std::fs::read_to_string(path).ok()?;
        ChatTemplate::new(&src, String::new(), "<|im_end|>".into()).ok()
    }

    fn spec(name: &str) -> ozgent_core::ToolSpec {
        ozgent_core::ToolSpec {
            name: name.into(),
            description: "Look something up.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }),
            output_schema: None,
            effect: Default::default(),
        }
    }

    fn ling_template() -> Option<ChatTemplate> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ling3.0.j2");
        let src = std::fs::read_to_string(path).ok()?;
        ChatTemplate::new(&src, String::new(), String::new()).ok()
    }

    #[test]
    fn ling_writes_its_own_call_format_too() {
        // A third vocabulary again: Ling names the function on the opener's
        // line and lists arg_key/arg_value pairs. The point of letting the
        // template write the call is that ozgent never has to know this.
        let Some(t) = ling_template() else { return };
        let messages = vec![
            Message::user("look it up"),
            Message {
                role: Role::Assistant,
                content: Vec::new(),
                thinking: Some("I should search.".into()),
                tool_calls: vec![ozgent_core::ToolCall {
                    id: "call_0".into(),
                    name: "web_search".into(),
                    arguments: serde_json::json!({"query": "llama.cpp"}),
                }],
                tool_call_id: None,
            },
        ];

        let out = t
            .render(&messages, RenderOptions { tools: vec![tool_json(&spec("web_search"))], ..Default::default() })
            .unwrap();

        assert!(out.contains("<arg_key>"), "{out}");
        assert!(out.contains("llama.cpp"), "{out}");
        assert!(!out.contains("<parameter="), "that is Qwen's format, not Ling's: {out}");
    }

    #[test]
    fn tojson_accepts_the_keywords_json_dumps_takes() {
        // Ling calls `tojson(ensure_ascii=False)`. minijinja's own filter
        // rejects the keyword, and the failure is silent: the whole template
        // is abandoned for llama.cpp's built-in, which does not open the
        // reasoning block Ling's template ends inside.
        let t = ChatTemplate::new(
            "{{ {'a': 1} | tojson(ensure_ascii=False) }}",
            String::new(),
            String::new(),
        )
        .unwrap();
        assert_eq!(t.render(&[Message::user("x")], RenderOptions::default()).unwrap(), r#"{"a":1}"#);
    }

    #[test]
    fn ensure_ascii_is_honoured_in_both_directions() {
        let raw = ChatTemplate::new(
            "{{ ['né'] | tojson(ensure_ascii=False) }}",
            String::new(),
            String::new(),
        )
        .unwrap();
        let escaped =
            ChatTemplate::new("{{ ['né'] | tojson }}", String::new(), String::new()).unwrap();

        let probe = [Message::user("x")];
        assert_eq!(raw.render(&probe, RenderOptions::default()).unwrap(), "[\"né\"]");
        // Python's default is to escape, and a template relying on it must
        // get what it asked for.
        assert_eq!(escaped.render(&probe, RenderOptions::default()).unwrap(), "[\"n\\u00e9\"]");
    }

    #[test]
    fn ling_is_detected_as_handling_tools() {
        let Some(t) = ling_template() else { return };
        assert!(t.handles_tools());
    }

    #[test]
    fn qwen_is_detected_as_handling_tools() {
        let Some(t) = qwen_template() else { return };
        assert!(t.handles_tools(), "Qwen documents tools itself");
    }

    #[test]
    fn a_template_without_a_tools_block_is_not_claimed_to_have_one() {
        // The check that keeps the preamble in place for the many models
        // whose templates cannot describe a tool at all.
        let plain = ChatTemplate::new(
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}",
            String::new(),
            String::new(),
        )
        .unwrap();
        assert!(!plain.handles_tools());
    }

    #[test]
    fn the_word_tools_in_prose_is_not_a_tools_block() {
        // Why this is probed rather than grepped: a false positive removes the
        // only tool instructions the model would have received.
        let prose = ChatTemplate::new(
            "You may use tools.{% for m in messages %}{{ m.content }}{% endfor %}",
            String::new(),
            String::new(),
        )
        .unwrap();
        assert!(!prose.handles_tools());
    }

    #[test]
    fn qwen_writes_its_own_call_format_from_structured_calls() {
        // The whole point of passing tool_calls: the template emits the format
        // the model was trained on, instead of ozgent hand-building a
        // different one and hoping the model imitates it.
        let Some(t) = qwen_template() else { return };
        let messages = vec![
            Message::user("what is up"),
            Message {
                role: Role::Assistant,
                content: Vec::new(),
                thinking: Some("I should search.".into()),
                tool_calls: vec![ozgent_core::ToolCall {
                    id: "call_0".into(),
                    name: "web_search".into(),
                    arguments: serde_json::json!({"query": "stocks", "count": 5}),
                }],
                tool_call_id: None,
            },
            Message::tool_result("call_0", "{\"results\": []}"),
        ];

        let out = t
            .render(&messages, RenderOptions { tools: vec![tool_json(&spec("web_search"))], ..Default::default() })
            .unwrap();

        assert!(out.contains("<function=web_search>"), "{out}");
        assert!(out.contains("<parameter=query>"), "{out}");
        assert!(out.contains("stocks"), "{out}");
        assert!(out.contains("<parameter=count>"), "arguments must stay an object: {out}");
        assert!(out.contains("<tool_response>"), "the result needs its own wrapper: {out}");
        assert!(out.contains("I should search."), "reasoning must survive: {out}");
    }

    #[test]
    fn the_tool_schema_reaches_the_model() {
        let Some(t) = qwen_template() else { return };
        let out = t
            .render(
                &[Message::user("hi")],
                RenderOptions { tools: vec![tool_json(&spec("web_search"))], ..Default::default() },
            )
            .unwrap();
        assert!(out.contains("web_search"), "{out}");
        assert!(out.contains("Look something up."), "the description too: {out}");
        assert!(out.contains("query"), "and the parameters: {out}");
    }

    #[test]
    fn no_tools_means_no_tools_block() {
        let Some(t) = qwen_template() else { return };
        let out = t.render(&[Message::user("hi")], RenderOptions::default()).unwrap();
        assert!(!out.contains("# Tools"), "an empty list must leave `tools` undefined: {out}");
    }

    #[test]
    fn a_turns_reasoning_reaches_the_template() {
        // Qwen 3.5 reads `reasoning_content` and, finding none, writes an
        // empty `<think></think>` for the turn. Across a tool-calling loop
        // that tells the model it never reasoned on turns where it did.
        let src = concat!(
            "{% for m in messages %}",
            "[{{ m.role }}:{{ m.reasoning_content if m.reasoning_content is defined else 'none' }}]",
            "{% endfor %}"
        );
        let tmpl = ChatTemplate::new(src, String::new(), String::new()).unwrap();

        let messages = vec![Message {
            role: Role::Assistant,
            content: vec![ozgent_core::Part::Text { text: "answer".into() }],
            thinking: Some("the reasoning".into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }];

        let out = tmpl.render(&messages, RenderOptions::default()).unwrap();
        assert!(out.contains("[assistant:the reasoning]"), "{out}");
    }

    #[test]
    fn an_empty_reasoning_block_is_not_passed_as_one() {
        let src = concat!(
            "{% for m in messages %}",
            "{{ 'yes' if m.reasoning_content is defined else 'no' }}",
            "{% endfor %}"
        );
        let tmpl = ChatTemplate::new(src, String::new(), String::new()).unwrap();

        for thinking in [None, Some(String::new()), Some("   
".to_string())] {
            let messages = vec![Message {
                role: Role::Assistant,
                content: vec![ozgent_core::Part::Text { text: "a".into() }],
                thinking,
                tool_calls: Vec::new(),
                tool_call_id: None,
            }];
            assert_eq!(tmpl.render(&messages, RenderOptions::default()).unwrap(), "no");
        }
    }

    #[test]
    fn thinking_stays_the_templates_decision_unless_asked() {
        let src = "{% if enable_thinking is defined %}forced:{{ enable_thinking }}\
                   {% else %}default{% endif %}";
        assert_eq!(render(src, RenderOptions::default()), "default");
        assert_eq!(
            render(src, RenderOptions { enable_thinking: Some(false), ..Default::default() }),
            // Capitalised, because Python's Jinja2 renders booleans that way
            // and a template that prints one must read the same here.
            "forced:False"
        );
        assert_eq!(
            render(src, RenderOptions { enable_thinking: Some(true), ..Default::default() }),
            "forced:True"
        );
    }

    #[test]
    fn effort_reaches_a_template_that_asks_for_it() {
        // Ling 3.0's shape: the template decides what each level means, which
        // is the point — the model author knows better than a token count.
        let src = "{% if reasoning_effort is defined %}effort={{ reasoning_effort }}\
                   {% else %}unset{% endif %}";
        assert_eq!(render(src, RenderOptions::default()), "unset");
        assert_eq!(
            render(src, RenderOptions { reasoning_effort: Some("high".into()), ..Default::default() }),
            "effort=high"
        );
    }

    #[test]
    fn python_string_methods_work() {
        // Templates split reasoning back out of prior assistant turns with
        // these; without pycompat every real template fails to render.
        let src = "{{ 'a</think>b'.split('</think>')[0] }}{{ '  x '.strip() }}";
        assert_eq!(render(src, RenderOptions::default()), "ax");
    }

    #[test]
    fn a_broken_template_is_an_error_not_a_panic() {
        assert!(ChatTemplate::new("{% for %}", "".into(), "".into()).is_err());
    }

    #[test]
    fn raise_exception_surfaces_the_templates_own_message() {
        let t = ChatTemplate::new(
            "{{ raise_exception('no system messages here') }}",
            "".into(),
            "".into(),
        )
        .expect("compiles");
        let err = t.render(&[Message::user("hi")], RenderOptions::default()).unwrap_err();
        assert!(err.to_string().contains("no system messages here"), "{err}");
    }
}
