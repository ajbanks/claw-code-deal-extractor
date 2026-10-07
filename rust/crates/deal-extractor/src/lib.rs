//! Property-deal extraction pipeline — pure-LLM extraction via the
//! claw-code `api` crate, with a JSON-Schema-driven validate→repair loop
//! where the agent owns correctness.
//!
//! The contract is the dumped TS TypeBox schema (`schemas/*.json`), not
//! hand-written serde structs. Deals travel as `serde_json::Value`.

pub mod contract;
pub mod corpus;
pub mod coverage;
pub mod documents;
pub mod extract;
pub mod pipeline;
pub mod prompts;
pub mod repair;
pub mod schema;
pub mod scoring;
pub mod validate;
