use super::graph::EliminationGraph;
use super::multiplicity::EdgeCount;
use crate::Real;

/// Copies stored as the graph stores them, so a single-copy star pays no count or division.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StarEntry<T, C> {
    pub(super) neighbor: u32,
    pub(super) copies: C,
    pub(super) weight: T,
}

/// A pivot's deduplicated live neighborhood in clique-tree order, and what collapsing it cost each degree.
pub(super) struct Star<T, C> {
    entries: Vec<StarEntry<T, C>>,
    removed_copies: Vec<(u32, u32)>,
}

impl<T: Real, C: EdgeCount> Star<T, C> {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            removed_copies: Vec::new(),
        }
    }

    /// Merges duplicate neighbors under the split's cap, then sorts by per-copy weight.
    pub(super) fn refill(&mut self, graph: &EliminationGraph<C, T>, pivot: usize, split: C::Split) {
        todo!()
    }

    pub(super) fn entries(&self) -> &[StarEntry<T, C>] {
        &self.entries
    }

    /// Collapsed duplicates or capped copies: both are one degree decrement.
    pub(super) fn removed_copies(&self) -> &[(u32, u32)] {
        &self.removed_copies
    }
}
