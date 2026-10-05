//! Elimination graph: [`adjacency`], [`multiplicity`], and the input's [`components`],
//! each viewed in place and built into its own graph.

mod adjacency;
mod components;
mod multiplicity;

pub(crate) use adjacency::{AdjListGraph, Neighbor};
pub(crate) use components::{Component, Components};
pub(crate) use multiplicity::{EdgeCount, Multi, Single, SplitFactor};
