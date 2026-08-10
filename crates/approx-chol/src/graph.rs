//! Elimination graph: the [`adjacency`] it is eliminated on, the edge
//! [`multiplicity`] that separates AC from AC2, the [`blocks`] the vertex set splits
//! into, and the [`ingest`]ion that builds all three from CSR input.

mod adjacency;
mod blocks;
mod ingest;
mod multiplicity;

pub(crate) use adjacency::{AdjListGraph, Neighbor};
pub(crate) use blocks::{BlockLayout, BlockVertices};
pub(crate) use ingest::Ingestion;
pub(crate) use multiplicity::{EdgeCount, Multi, Single, SplitFactor};
