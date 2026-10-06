//! Elimination graph, and the input's [`components`] viewed in place, each built as its own graph.

mod adjacency;
mod components;
mod multiplicity;

pub(crate) use adjacency::{AdjListGraph, Neighbor};
pub(crate) use components::{Component, Components};
pub(crate) use multiplicity::{EdgeCount, Multi, Single, SplitFactor};
