//! Elimination graph: [`adjacency`], [`multiplicity`], [`blocks`], and the
//! [`ingest`]ion that builds all three from CSR input.

mod adjacency;
mod blocks;
mod ingest;
mod multiplicity;

pub(crate) use adjacency::{AdjListGraph, Neighbor};
pub(crate) use blocks::{BlockLayout, BlockVertices};
pub(crate) use ingest::Ingestion;
pub(crate) use multiplicity::{EdgeCount, Multi, Single, SplitFactor};
