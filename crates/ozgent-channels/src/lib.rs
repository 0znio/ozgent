//! Reaching ozgent from a messaging app.
//!
//! A channel is a bridge between a chat app and the same engine, model
//! registry, memory store and permission rules every other surface uses — so a
//! conversation started on a phone can be picked up in the browser, and a
//! permission answered in the browser applies to a question asked from a chat.
//!
//! Two things make this different from the other surfaces and shape everything
//! here:
//!
//! * **The person is not at this machine.** They cannot see the terminal, they
//!   cannot see what a tool is about to do beyond what the message says, and
//!   they may not be the person who owns the machine at all. So the allowlist
//!   is the load-bearing part, not the transport.
//! * **Chat apps are not terminals.** There is no scrollback to redraw, edits
//!   are rate-limited, markdown does not exist, and a message has a length
//!   limit. A reply is therefore composed, throttled and split rather than
//!   streamed token by token.

pub mod chat;
pub mod command;
pub mod compose;
pub mod gateway;
pub mod live;
pub mod markup;
pub mod split;
pub mod telegram;
pub mod whatsapp;

pub use chat::{Command, Inbound, Msg, Question};
pub use markup::Flavour;
