//! Elimination graph: [`adjacency`], [`multiplicity`], [`blocks`], and the SDDM input split into [`component`]s.

mod adjacency;
mod blocks;
mod component;
mod multiplicity;

pub(crate) use adjacency::{AdjListGraph, Neighbor};
pub(crate) use blocks::{BlockLayout, BlockVertices};
pub(crate) use component::Component;
pub(crate) use multiplicity::{EdgeCount, Multi, Single, SplitFactor};
