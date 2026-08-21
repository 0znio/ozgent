//! Switching constrained decoding on at the exact moment a tool call begins.
//!
//! A tool grammar cannot simply be applied to a turn. Its root requires the
//! opening marker, so constraining from the first token would force *every*
//! response to be a tool call — the model could not answer a question even if
//! it wanted to. The constraint is only correct once the model has committed
//! to calling something.
//!
//! So the gate watches the emitted text for an opening marker and, the instant
//! one completes, hands back a grammar rooted at the JSON that follows. Prose
//! is generated entirely unconstrained; only the inside of a call is forced.
//!
//! Detection is deliberately over a small rolling window rather than the whole
//! output. Scanning everything produced so far on every token would be
//! quadratic in the length of the turn, which is the cost this file exists to
//! avoid paying.

use ozgent_core::ToolSpec;

/// The rolling window kept for marker detection.
///
/// Only needs to exceed the longest opener; 32 bytes leaves generous room.
const WINDOW: usize = 32;

/// Openers whose body is a bare `{"name": ..., "arguments": ...}` object, and
/// the marker that closes them.
///
/// Deliberately a subset of [`crate::toolcall::OPENERS`]. `[TOOL_CALLS]` wraps
/// its call in an array and `<function=` carries the name inside the marker
/// itself, so both need a differently shaped root; until that exists they are
/// simply left unconstrained, which is exactly today's behaviour.
const GATED: &[(&str, Option<&str>)] = &[
    ("<tool_call>", Some("</tool_call>")),
    ("<|tool_call|>", Some("<|/tool_call|>")),
    ("<|python_tag|>", None),
];

/// Watches generated text and fires once a tool call starts.
pub struct ToolGate {
    /// Opener paired with the grammar to apply after it.
    grammars: Vec<(&'static str, String)>,
    tail: String,
    armed: bool,
    fired: bool,
}

impl ToolGate {
    /// Compile the per-opener grammars for `tools`.
    ///
    /// Done once when tools are configured, not per turn: converting a JSON
    /// schema to GBNF is pure work that does not depend on the conversation.
    pub fn compile(tools: &[ToolSpec]) -> Vec<(&'static str, String)> {
        GATED
            .iter()
            .filter_map(|(opener, closer)| {
                crate::grammar::tool_body_grammar(tools, *closer).map(|g| (*opener, g))
            })
            .collect()
    }

    /// A gate that will never fire, for turns with no tools.
    pub fn inert() -> Self {
        Self { grammars: Vec::new(), tail: String::new(), armed: false, fired: false }
    }

    /// Arm the gate with previously compiled grammars.
    pub fn new(grammars: Vec<(&'static str, String)>) -> Self {
        let armed = !grammars.is_empty();
        Self { grammars, tail: String::new(), armed, fired: false }
    }

    /// Whether the gate applied a grammar during this turn.
    pub fn fired(&self) -> bool {
        self.fired
    }

    /// Feed newly emitted text.
    ///
    /// Returns the grammar to apply when a marker has *just* completed, and
    /// `None` otherwise. The marker must land exactly at the end of the output
    /// so far: if a single token carried the marker plus the first character of
    /// the body, the grammar would start one character out of step with the
    /// text already emitted, and llama.cpp would reject the first token it saw.
    /// Declining to fire in that case costs nothing — the call is merely
    /// unconstrained, as it is today.
    pub fn observe(&mut self, text: &str) -> Option<String> {
        if !self.armed || text.is_empty() {
            return None;
        }

        self.tail.push_str(text);
        if self.tail.len() > WINDOW {
            let mut cut = self.tail.len() - WINDOW;
            while !self.tail.is_char_boundary(cut) {
                cut += 1;
            }
            self.tail.drain(..cut);
        }

        let hit = self
            .grammars
            .iter()
            .find(|(opener, _)| self.tail.ends_with(opener))
            .map(|(_, grammar)| grammar.clone());

        if hit.is_some() {
            self.armed = false;
            self.fired = true;
        }
        hit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ozgent_core::ToolSpec;
    use serde_json::json;

    fn tool() -> ToolSpec {
        ToolSpec {
            name: "web_search".into(),
            description: "search".into(),
            input_schema: json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
            output_schema: None,
        }
    }

    #[test]
    fn prose_never_fires_the_gate() {
        let mut gate = ToolGate::new(ToolGate::compile(&[tool()]));
        for chunk in ["A hash ", "table is ", "a data structure."] {
            assert!(gate.observe(chunk).is_none(), "prose must stay unconstrained");
        }
        assert!(!gate.fired());
    }

    #[test]
    fn a_completed_marker_fires_once() {
        let mut gate = ToolGate::new(ToolGate::compile(&[tool()]));
        assert!(gate.observe("Let me look. ").is_none());
        let g = gate.observe("<tool_call>").expect("marker should fire the gate");
        assert!(g.contains("web_search"), "grammar should name the tool: {g}");
        assert!(!g.contains("\"<tool_call>\""), "root must start after the marker");
        // Disarmed afterwards: a second marker in the same turn must not
        // rebuild the sampler underneath a live grammar.
        assert!(gate.observe("<tool_call>").is_none());
    }

    #[test]
    fn a_marker_that_arrives_with_body_text_declines_to_fire() {
        // One token carrying `<tool_call>{` would leave the grammar a character
        // behind the output. Falling back to unconstrained decoding is correct.
        let mut gate = ToolGate::new(ToolGate::compile(&[tool()]));
        assert!(gate.observe("<tool_call>{").is_none());
    }

    #[test]
    fn a_marker_split_across_tokens_still_fires() {
        // Markers are usually one vocabulary token, but nothing guarantees it.
        let mut gate = ToolGate::new(ToolGate::compile(&[tool()]));
        assert!(gate.observe("<tool").is_none());
        assert!(gate.observe("_call>").is_some(), "the window spans token boundaries");
    }

    #[test]
    fn the_window_stays_bounded_over_a_long_turn() {
        let mut gate = ToolGate::new(ToolGate::compile(&[tool()]));
        for _ in 0..500 {
            gate.observe("some more prose ");
        }
        assert!(gate.tail.len() <= WINDOW + 16, "window grew to {}", gate.tail.len());
    }

    #[test]
    fn multibyte_text_does_not_split_a_character() {
        let mut gate = ToolGate::new(ToolGate::compile(&[tool()]));
        for _ in 0..20 {
            // Truncating mid-codepoint would panic on the String drain.
            gate.observe("café — naïve ☃");
        }
        assert!(gate.observe("<tool_call>").is_some());
    }

    #[test]
    fn no_tools_means_an_inert_gate() {
        let mut gate = ToolGate::new(ToolGate::compile(&[]));
        assert!(gate.observe("<tool_call>").is_none());
        assert!(!gate.fired());
    }
}
