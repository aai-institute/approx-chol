//! Elimination graph: the [`adjacency`] it is eliminated on, the edge
//! [`multiplicity`] that separates AC from AC2, and the [`ingest`]ion that builds one
//! from CSR input.

mod adjacency;
mod ingest;
mod multiplicity;

pub(crate) use adjacency::{AdjListGraph, Neighbor};
pub(crate) use ingest::{BlockVertices, Ingestion};
pub(crate) use multiplicity::{EdgeCount, Multi, Single, SplitFactor};
