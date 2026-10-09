use super::copies::Copies;
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
    /// Grown to `n` on first need: above 32 entries, dedup scatters instead of sorting 16-byte entries twice.
    scatter: Vec<u32>,
}

impl<T: Real, C: Copies> Star<T, C> {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            removed_copies: Vec::new(),
            scatter: Vec::new(),
        }
    }

    /// Merges duplicate neighbors under `split`'s cap, then sorts by per-copy weight.
    pub(super) fn refill(&mut self, neighbors: impl Iterator<Item = (u32, C, T)>, split: C::Split) {
        todo!()
    }

    pub(super) fn entries(&self) -> &[StarEntry<T, C>] {
        &self.entries
    }

    /// Copies each neighbor lost: those to the pivot, and those merging or capping collapsed.
    pub(crate) fn degree_losses(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let to_pivot = (self.entries.iter()).map(|entry| (entry.neighbor, entry.copies.get()));
        to_pivot.chain(self.removed_copies.iter().copied())
    }
}
