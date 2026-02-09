//! phrasegen: timing-model + scoring primitives.
//!
//! This crate is intentionally small and “dataset-format tolerant”: you can fit a
//! model from any source as long as you can produce `(phrase, digraph_dt_ms)` rows.

#![forbid(unsafe_code)]

pub mod adapt;
pub mod corrections;
pub mod data;
pub mod entropy;
pub mod generate;
pub mod import;
pub mod kgram;
pub mod model;
pub mod record;
pub mod score;
pub mod textprep;
pub mod timing;
