use core::cmp::Ordering;

use super::copies::Copies;
use crate::Real;

/// Copies stored as the graph stores them, so a single-copy star pays no count or division.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StarEntry<T, C> {
    pub(super) neighbor: u32,
    pub(super) copies: C,
    pub(super) weight: T,
}

/// NaN last: `partial_cmp`'s `None` breaks the total order sorts require (1.81+ panics).
fn float_total_cmp<T: Real>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

/// Total on a deduped star, which keeps the clique tree off `sort_unstable`'s width heuristics.
fn by_weight_then_neighbor<T: Real, C>(a: &StarEntry<T, C>, b: &StarEntry<T, C>) -> Ordering {
    float_total_cmp(&a.weight, &b.weight).then_with(|| a.neighbor.cmp(&b.neighbor))
}

/// A pivot's deduplicated neighborhood in clique-tree order, and what collapsing it cost each degree.
pub(super) struct Star<T, C> {
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

    /// Neighbors already unique under one `copies`, which keeps per-copy order the raw order: one sort, no scratch.
    pub(super) fn refill_unique(&mut self, neighbors: impl Iterator<Item = (u32, T)>, copies: C) {
        self.entries.clear();
        self.removed_copies.clear();
        let entries = neighbors.map(|(neighbor, weight)| StarEntry {
            neighbor,
            copies,
            weight,
        });
        self.entries.extend(entries);
        self.entries.sort_unstable_by(by_weight_then_neighbor);
    }

    pub(super) fn entries(&self) -> &[StarEntry<T, C>] {
        &self.entries
    }

    /// Copies each neighbor lost: those to the pivot, and those merging or capping collapsed.
    pub(super) fn degree_losses(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let to_pivot = (self.entries.iter()).map(|entry| (entry.neighbor, entry.copies.get()));
        to_pivot.chain(self.removed_copies.iter().copied())
    }
}
