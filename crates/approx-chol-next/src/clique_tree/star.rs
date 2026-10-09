use super::multiplicity::EdgeCount;
use crate::Real;

/// Copies stored as the graph stores them, so a single-copy star pays no count or division.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StarEntry<T, C> {
    pub(super) neighbor: u32,
    pub(super) copies: C,
    pub(super) weight: T,
}

/// A pivot's deduplicated neighborhood in clique-tree order, and what collapsing it cost each degree.
pub(crate) struct Star<T, C> {
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
    pub(super) fn refill(&mut self, neighbors: impl Iterator<Item = (u32, C, T)>, split: C::Split) {
        todo!()
    }

    pub(super) fn entries(&self) -> &[StarEntry<T, C>] {
        &self.entries
    }

    pub(crate) fn neighbors(&self) -> impl Iterator<Item = u32> + '_ {
        self.entries.iter().map(|entry| entry.neighbor)
    }

    /// Collapsed duplicates or capped copies: both are one degree decrement.
    pub(crate) fn removed_copies(&self) -> &[(u32, u32)] {
        &self.removed_copies
    }
}
