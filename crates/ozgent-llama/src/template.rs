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

use std::collections::BTreeMap;

use minijinja::{Environment, Value, context};
use ozgent_core::{Message, Role};

/// A compiled chat template, ready to render turns.
pub struct ChatTemplate {
    env: Environment<'static>,
    bos: String,
    eos: String,
}

#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("compiling the chat template: {0}")]
    Compile(String),
    #[error("rendering the chat template: {0}")]
    Render(String),
}

/// What the caller wants of this turn, in the vocabulary templates expect.
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    /// `None` leaves `enable_thinking` undefined so the template's own default
    /// applies — which is what "auto" means. Forcing it either way overrides a
    /// model that has a considered opinion about when to reason.
    pub enable_thinking: Option<bool>,
    /// How hard to think, in the vocabulary templates use: "low", "medium",
    /// "high". Asking is better than interrupting — a model that decides for
    /// itself to reason briefly still finishes its thought, where a token
    /// budget stops it mid-argument and makes it answer from an unfinished
    /// one. Only some templates read this; for the rest it is inert and the
    /// budget remains the only control.
    pub reasoning_effort: Option<String>,
}

impl ChatTemplate {
    /// Compile `source`, the raw Jinja from the GGUF.
    pub fn new(source: &str, bos: String, eos: String) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        // Templates are written against Python's Jinja2 and call Python string
        // methods on their values — `.split()`, `.rstrip()`, `.startswith()`.
        // Without this every one of them is an unknown-method error.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.add_template_owned("chat", source.to_string())
            .map_err(|e| TemplateError::Compile(e.to_string()))?;
        Ok(Self { env, bos, eos })
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

        let turns: Vec<BTreeMap<&str, String>> = messages
            .iter()
            .map(|m| {
                let mut turn = BTreeMap::new();
                turn.insert("role", role_name(m.role).to_string());
                turn.insert("content", m.text_content());
                turn
            })
            .collect();

        let mut ctx = context! {
            messages => turns,
            add_generation_prompt => true,
            bos_token => self.bos,
            eos_token => self.eos,
        };
        if let Some(on) = opts.enable_thinking {
            ctx = context! { enable_thinking => on, ..ctx };
        }
        if let Some(effort) = opts.reasoning_effort.as_deref() {
            ctx = context! { reasoning_effort => effort, ..ctx };
        }

        tmpl.render(ctx).map_err(|e| TemplateError::Render(chain(&e)))
    }
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
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
