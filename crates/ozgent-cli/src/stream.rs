//! Rendering a turn that the daemon is running.
//!
//! The terminal used to do all of this itself: assemble the prompt, drive the
//! tool loop, ask about permissions, persist the reply. All of it existed a
//! second time in `ozgent-web`, and two implementations of one thing drift.
//!
//! Now the daemon does it and this draws what comes back. The event stream is
//! the browser's — unchanged, deliberately, because a translated version of it
//! for the terminal would be exactly the seam where the two start to differ
//! again.
//!
//! What this leaves the terminal responsible for is the part that is actually
//! terminal-shaped: repainting markdown as it arrives, showing a tool call as
//! a line rather than a card, and asking a permission question in a way you
//! can answer with one key.

use anyhow::Result;
use serde_json::Value;

use crate::backend::{Backend, Stream};
use crate::tui::Ui;

/// What a finished turn produced, for the status line and `/copy`.
#[derive(Debug, Default)]
pub struct Finished {
    pub answer: String,
    /// Tokens per second, when the daemon reported them.
    pub rate: Option<f64>,
    /// Context used and its size, for the gauge.
    pub used: Option<u32>,
    pub window: Option<u32>,
    /// Set when the turn ended badly. The text has already been shown.
    pub failed: bool,
}

/// Answers a permission question. Takes the screen rather than capturing it,
/// because this module already holds `&mut Ui` and a closure that captured it
/// would be a second mutable borrow of the same thing.
pub type Decide = fn(&mut Ui, &str, &Value, &str) -> Option<String>;

/// Draw a turn as it happens.
///
/// `decide` is called for each permission question and returns the choice, or
/// `None` to refuse. It is a callback rather than something this module does
/// itself because asking is the one part that needs the whole terminal — the
/// prompt, the key handling, the redraw underneath it.
pub async fn render(
    backend: &Backend,
    mut events: Stream,
    ui: &mut Ui,
    theme: &ozgent_render::Theme,
    show_thinking: bool,
    decide: Decide,
) -> Result<Finished> {
    let mut out = Finished::default();
    let mut thinking = String::new();
    // The name of the call being announced, so a result can close the line it
    // opened rather than guessing which one it belongs to.
    let mut running: Option<String> = None;
    let mut loading: Option<String> = None;

    while let Some(event) = events.next().await? {
        let kind = event["type"].as_str().unwrap_or("");
        match kind {
            // Before anything is generated, and only when a model has to be
            // brought in. A daemon that already holds it never sends this,
            // which is the whole point of asking the daemon.
            "loading" => {
                let model = event["model"].as_str().unwrap_or("the model");
                let progress = event["progress"].as_f64().unwrap_or(0.0);
                if loading.as_deref() != Some(model) {
                    loading = Some(model.to_string());
                }
                ui.begin_activity(format!(
                    "loading {model}  {}",
                    crate::chat::progress_bar(progress as f32, 24)
                ));
                ui.tick();
            }

            "ready" => {
                if loading.take().is_some() {
                    ui.settle();
                }
                out.window = event["context"].as_u64().map(|n| n as u32);
            }

            "thinking" => {
                thinking.push_str(event["text"].as_str().unwrap_or(""));
                if show_thinking {
                    ui.stream(Some(&thinking), &out.answer, false);
                }
            }

            "answer" => {
                out.answer.push_str(event["text"].as_str().unwrap_or(""));
                ui.stream(show_thinking.then_some(thinking.as_str()), &out.answer, false);
            }

            // The model has committed to a call and is still writing it. Said
            // as soon as the name is readable, because everything from the
            // opening marker is withheld from the stream — without this the
            // screen simply stops while a file is generated.
            "tool_call_started" => {
                let name = event["name"].as_str().unwrap_or("a tool").to_string();
                ui.stream(show_thinking.then_some(thinking.as_str()), &out.answer, true);
                ui.commit();
                ui.begin_activity(format!("{name} — writing the call…"));
                running = Some(name);
            }

            "tool_call" => {
                let name = event["name"].as_str().unwrap_or("a tool").to_string();
                let args = crate::chat::pretty_args(&event["arguments"]);
                if running.as_deref() != Some(name.as_str()) {
                    ui.stream(show_thinking.then_some(thinking.as_str()), &out.answer, true);
                    ui.commit();
                }
                ui.begin_activity(if args.is_empty() {
                    name.clone()
                } else {
                    format!("{name}  {args}")
                });
                running = Some(name);
            }

            "permission" => {
                let name = event["name"].as_str().unwrap_or("a tool");
                let id = event["id"].as_str().unwrap_or("");
                let effect = event["effect"].as_str().unwrap_or("unknown");
                ui.settle();
                let choice = decide(ui, name, &event["arguments"], effect);
                match choice {
                    Some(choice) => {
                        backend.decide(id, &choice).await?;
                        // Only an outright refusal ends the call; the other
                        // three all run it, and the result says what happened.
                        if choice != "deny" {
                            ui.begin_activity(name.to_string());
                        }
                    }
                    None => {
                        backend.decide(id, "deny").await?;
                    }
                }
            }

            "tool_result" => {
                let name = event["name"].as_str().unwrap_or("a tool");
                let ok = event["ok"].as_bool().unwrap_or(false);
                let ms = event["ms"].as_u64().unwrap_or(0);
                // The daemon's summary is written for a card that can show
                // the whole result underneath it; a terminal line has no
                // underneath, so the detail is summarised again here rather
                // than printed as the JSON it is.
                let summary = match (&event["detail"], ok) {
                    (serde_json::Value::Null, _) => event["summary"].as_str().unwrap_or("").to_string(),
                    (detail, true) => crate::chat::summarise_result(detail),
                    (_, false) => event["summary"].as_str().unwrap_or("failed").to_string(),
                };
                ui.settle();
                ui.say(tool_line(theme, name, ok, &summary, ms));
                running = None;
            }

            "agent_start" => {
                let name = event["name"].as_str().unwrap_or("an agent");
                let description = event["description"].as_str().unwrap_or("");
                // Its report begins a block of its own, so whatever the model
                // said before handing over is not run into it.
                if !out.answer.trim().is_empty() {
                    out.answer.push_str("\n\n");
                }
                ui.stream(show_thinking.then_some(thinking.as_str()), &out.answer, true);
                ui.commit();
                ui.say(theme.style(
                    ozgent_render::Style::dim(),
                    &format!("@{name} — {description}"),
                ));
            }

            "agent_end" => {
                let name = event["name"].as_str().unwrap_or("an agent");
                let ok = event["ok"].as_bool().unwrap_or(false);
                let calls = event["calls"].as_u64().unwrap_or(0);
                let ms = event["ms"].as_u64().unwrap_or(0);
                ui.say(theme.style(
                    ozgent_render::Style::dim(),
                    &format!(
                        "@{name} {} · {calls} call{} · {:.1}s",
                        if ok { "finished" } else { "gave no report" },
                        if calls == 1 { "" } else { "s" },
                        ms as f64 / 1000.0
                    ),
                ));
            }

            "done" => {
                out.rate = event["tokens_per_second"].as_f64().filter(|r| *r > 0.0);
                // The daemon counts the prompt it actually processed, which is
                // the honest figure for a gauge: the terminal's own guess at
                // the context used never included what retrieval added.
                let prompt = event["prompt"].as_u64().unwrap_or(0);
                let generated = event["generated"].as_u64().unwrap_or(0);
                out.used = Some((prompt + generated) as u32);
                break;
            }

            "error" => {
                let message = event["message"].as_str().unwrap_or("something went wrong");
                ui.settle();
                ui.say(theme.style(ozgent_render::Style::color(ozgent_render::Color::Red), message));
                out.failed = true;
                break;
            }

            // A kind this build does not know. Ignored rather than shown: the
            // daemon may be newer than the terminal, and an unknown event is
            // not an error.
            _ => {}
        }
    }

    // Whatever arrived, painted once more and committed, so the reply is on
    // the screen as a finished block rather than a throttled half-frame.
    if !out.answer.trim().is_empty() || !thinking.trim().is_empty() {
        ui.stream(show_thinking.then_some(thinking.as_str()), &out.answer, true);
        ui.commit();
    }
    Ok(out)
}

/// The line a finished tool call leaves behind.
fn tool_line(
    theme: &ozgent_render::Theme,
    name: &str,
    ok: bool,
    summary: &str,
    ms: u64,
) -> String {
    let took = if ms >= 1000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{ms}ms")
    };
    let mark = if ok { "✓" } else { "✗" };
    let body = if summary.is_empty() {
        format!("{mark} {name} · {took}")
    } else {
        format!("{mark} {name} — {summary} · {took}")
    };
    theme.style(
        if ok { ozgent_render::Style::dim() } else { ozgent_render::Style::color(ozgent_render::Color::Red) },
        &body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn theme() -> ozgent_render::Theme {
        ozgent_render::Theme::plain()
    }

    #[test]
    fn a_finished_call_says_what_it_did_and_how_long_it_took() {
        let line = tool_line(&theme(), "web_search", true, "4 results", 1200);
        assert!(line.contains("web_search"), "{line}");
        assert!(line.contains("4 results"), "{line}");
        assert!(line.contains("1.2s"), "{line}");
        assert!(line.starts_with('✓'), "{line}");
    }

    #[test]
    fn a_failed_call_is_marked_as_one() {
        let line = tool_line(&theme(), "run_command", false, "not allowed", 4);
        assert!(line.starts_with('✗'), "{line}");
        assert!(line.contains("4ms"), "{line}");
    }

    #[test]
    fn a_call_with_nothing_to_say_still_reads_as_a_line() {
        let line = tool_line(&theme(), "list_dir", true, "", 7);
        assert!(line.contains("list_dir"), "{line}");
        assert!(!line.contains("—"), "no empty dash: {line}");
    }

    #[test]
    fn sub_second_calls_are_shown_in_milliseconds() {
        assert!(tool_line(&theme(), "t", true, "", 999).contains("999ms"));
        assert!(tool_line(&theme(), "t", true, "", 1000).contains("1.0s"));
    }
}
