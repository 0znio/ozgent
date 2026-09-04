//! Workflows: steps wired together, run on a schedule or on demand.
//!
//! The engine is deliberately ignorant of what a step *does*. It owns the
//! graph, the order, the data passed between steps and the record of what
//! happened; running a tool or a model is behind [`Steps`], which the caller
//! implements. That is what lets the whole thing be tested without a GPU, a
//! Python interpreter or a network.

pub mod expr;
pub mod engine;
pub mod model;
pub mod schedule;

pub use expr::Context;
pub use engine::{Progress, Refused, Run, Status, StepRecord, StepStatus, Steps, execute};
pub use model::{Edge, Flow, Invalid, Kind, Node};
