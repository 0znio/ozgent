//! Asking the browser whether a tool call may run.
//!
//! The terminal can block on `stdin` and be done with it. The browser cannot:
//! the question has to travel out on the event stream the turn is already
//! streaming, and the answer comes back as a separate HTTP request, on another
//! task, possibly from another device. This module is the meeting point.
//!
//! Two properties matter and neither is free:
//!
//! - **The inference thread must not wedge.** It is the only thread that can
//!   talk to the GPU, so a browser tab closed mid-question would otherwise
//!   take every later conversation with it. Waiting is bounded, and running
//!   out of patience is a refusal — the safe direction.
//! - **An answer must reach the call it belongs to.** A turn can ask about
//!   several tools at once, so the pending questions are keyed by the call id
//!   the client already uses to pair a result with its card.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ozgent_core::permission::Choice;

/// How long a question waits for an answer before it is treated as refused.
///
/// Long enough to walk away from the keyboard and come back; short enough that
/// a closed tab does not strand the inference thread for the rest of the day.
pub const WAIT: Duration = Duration::from_secs(300);

/// Questions asked and not yet answered, keyed by call id.
#[derive(Default)]
pub struct Pending {
    waiting: Mutex<HashMap<String, std::sync::mpsc::Sender<Choice>>>,
}

pub type SharedPending = Arc<Pending>;

impl Pending {
    /// Register a question and hand back the end that waits for its answer.
    pub fn ask(&self, id: &str) -> std::sync::mpsc::Receiver<Choice> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.waiting.lock().unwrap_or_else(|e| e.into_inner()).insert(id.to_string(), tx);
        rx
    }

    /// Deliver an answer. False when nothing was waiting for it — a reload,
    /// a double click, or an answer that arrived after the wait ran out.
    pub fn answer(&self, id: &str, choice: Choice) -> bool {
        let sender = self.waiting.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
        match sender {
            Some(tx) => tx.send(choice).is_ok(),
            None => false,
        }
    }

    /// Forget a question, whether or not it was answered.
    pub fn forget(&self, id: &str) {
        self.waiting.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
    }

    /// Refuse everything outstanding.
    ///
    /// Called when a turn ends for any other reason — the client disconnected,
    /// the model was interrupted — so a question nobody will ever see does not
    /// sit in the map until the process exits.
    pub fn abandon_all(&self) {
        self.waiting.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    pub fn outstanding(&self) -> usize {
        self.waiting.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// Wait for an answer, treating silence as a refusal.
pub fn wait(rx: std::sync::mpsc::Receiver<Choice>) -> Choice {
    rx.recv_timeout(WAIT).unwrap_or(Choice::Deny)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_answer_reaches_the_call_that_asked() {
        let pending = Pending::default();
        let rx = pending.ask("c1");
        assert!(pending.answer("c1", Choice::Once));
        assert_eq!(rx.recv().unwrap(), Choice::Once);
    }

    #[test]
    fn answers_do_not_cross_between_calls() {
        // A turn can ask about several tools at once; delivering the answer to
        // the wrong one would run a tool the user refused.
        let pending = Pending::default();
        let first = pending.ask("c1");
        let second = pending.ask("c2");

        pending.answer("c2", Choice::Deny);
        pending.answer("c1", Choice::Always);

        assert_eq!(first.recv().unwrap(), Choice::Always);
        assert_eq!(second.recv().unwrap(), Choice::Deny);
    }

    #[test]
    fn answering_twice_is_not_an_error_but_only_lands_once() {
        // A double-clicked button, or two tabs open on the same conversation.
        let pending = Pending::default();
        let rx = pending.ask("c1");
        assert!(pending.answer("c1", Choice::Once));
        assert!(!pending.answer("c1", Choice::Deny), "the second has nothing to answer");
        assert_eq!(rx.recv().unwrap(), Choice::Once);
    }

    #[test]
    fn an_unanswered_question_is_refused_rather_than_waited_on_forever() {
        // The inference thread is the only one that can reach the GPU. A tab
        // closed mid-question must not take every later conversation with it.
        let pending = Pending::default();
        let rx = pending.ask("c1");
        drop(pending);
        // The sender is gone, so this returns immediately rather than after
        // the full timeout — the case being pinned is that it returns at all,
        // and that the answer is a refusal.
        assert_eq!(wait(rx), Choice::Deny);
    }

    #[test]
    fn abandoning_clears_what_nobody_will_answer() {
        let pending = Pending::default();
        let _rx = pending.ask("c1");
        assert_eq!(pending.outstanding(), 1);
        pending.abandon_all();
        assert_eq!(pending.outstanding(), 0);
    }

    #[test]
    fn forgetting_one_leaves_the_others() {
        let pending = Pending::default();
        let _a = pending.ask("c1");
        let _b = pending.ask("c2");
        pending.forget("c1");
        assert_eq!(pending.outstanding(), 1);
        assert!(pending.answer("c2", Choice::Once));
    }
}
