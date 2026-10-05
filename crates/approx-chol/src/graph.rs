//! Elimination graph: [`adjacency`], [`multiplicity`], [`blocks`], and the input's
//! [`components`], each viewed in place and built into its own graph.

mod adjacency;
mod blocks;
mod components;
mod multiplicity;

pub(crate) use adjacency::{AdjListGraph, Neighbor};
pub(crate) use blocks::BlockVertices;
pub(crate) use components::{Component, Components};
pub(crate) use multiplicity::{EdgeCount, Multi, Single, SplitFactor};
