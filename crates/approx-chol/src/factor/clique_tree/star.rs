use core::cmp::Ordering;

use super::copies::Copies;
use super::CliqueSampler;
use crate::{Index, Real};

/// Copies stored as the graph stores them, so a single-copy star pays no count or division.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StarEntry<T, I, C> {
    pub(super) neighbor: I,
    pub(super) copies: C,
    pub(super) weight: T,
}

/// Total on a deduped star, which keeps the clique tree off `sort_unstable`'s width heuristics.
fn by_weight_then_neighbor<T: Real, I: Index, C>(
    a: &StarEntry<T, I, C>,
    b: &StarEntry<T, I, C>,
) -> Ordering {
    a.weight
        .total_cmp(&b.weight)
        .then_with(|| a.neighbor.cmp(&b.neighbor))
}

/// A function of each entry, so still a total order; `Single`'s identity makes it the raw sort.
fn by_per_copy_then_neighbor<T: Real, I: Index, C: Copies>(
    a: &StarEntry<T, I, C>,
    b: &StarEntry<T, I, C>,
) -> Ordering {
    let (x, y) = (a.copies.per_copy(a.weight), b.copies.per_copy(b.weight));
    x.total_cmp(&y).then_with(|| a.neighbor.cmp(&b.neighbor))
}

/// At or below this many entries, sorting beats the scatter path's random access.
const SCATTER_THRESHOLD: usize = 32;

/// Rests in every slot of [`CliqueSampler::scatter`] between merges.
fn unseen<I: Index>() -> I {
    I::max_value()
}

impl<T: Real, I: Index, C: Copies> CliqueSampler<T, I, C> {
    /// Merges duplicate neighbors under the sampler's `split` cap, then sorts by per-copy weight.
    pub(super) fn refill(&mut self, neighbors: impl Iterator<Item = (I, C, T)>) {
        self.entries.clear();
        self.removed_copies.clear();
        self.counts.clear();
        let entries = neighbors.map(|(neighbor, copies, weight)| StarEntry {
            neighbor,
            copies,
            weight,
        });
        self.entries.extend(entries);
        let merged = match self.entries.len() <= SCATTER_THRESHOLD {
            true => self.merge_by_sort(),
            false => self.merge_by_scatter(),
        };
        self.entries.truncate(merged);
        for (entry, &count) in self.entries.iter_mut().zip(&self.counts) {
            let (copies, dropped) = C::cap(count, self.split);
            entry.copies = copies;
            if dropped > 0 {
                self.removed_copies.push((entry.neighbor, dropped));
            }
        }
        self.entries.sort_unstable_by(by_per_copy_then_neighbor);
    }

    /// Runs of one neighbor summed in place; returns the merged count.
    fn merge_by_sort(&mut self) -> usize {
        self.entries.sort_unstable_by_key(|entry| entry.neighbor);
        let mut write = 0;
        for read in 0..self.entries.len() {
            let entry = self.entries[read];
            match write > 0 && self.entries[write - 1].neighbor == entry.neighbor {
                true => self.absorb(write - 1, entry),
                false => {
                    self.entries[write] = entry;
                    self.counts.push(entry.copies.get());
                    write += 1;
                }
            }
        }
        write
    }

    /// First-seen order, each slot back at [`unseen`] on return; returns the merged count.
    fn merge_by_scatter(&mut self) -> usize {
        let mut write = 0;
        for read in 0..self.entries.len() {
            let entry = self.entries[read];
            let slot = entry.neighbor.index();
            if slot >= self.scatter.len() {
                self.scatter.resize(slot + 1, unseen());
            }
            match self.scatter[slot] {
                position if position == unseen() => {
                    self.scatter[slot] = I::from_index(write);
                    self.entries[write] = entry;
                    self.counts.push(entry.copies.get());
                    write += 1;
                }
                position => self.absorb(position.index(), entry),
            }
        }
        for entry in &self.entries[..write] {
            self.scatter[entry.neighbor.index()] = unseen();
        }
        write
    }

    fn absorb(&mut self, position: usize, duplicate: StarEntry<T, I, C>) {
        let merged = &mut self.entries[position];
        merged.weight = merged.weight + duplicate.weight;
        self.counts[position] = self.counts[position].saturating_add(duplicate.copies.get());
    }

    /// Neighbors already unique, each at the split's initial copies, which keeps per-copy order the raw order: one sort, no scratch.
    pub(super) fn refill_unique(&mut self, neighbors: impl Iterator<Item = (I, T)>) {
        let copies = C::initial(self.split);
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

    /// Copies each neighbor lost: those to the pivot, and those merging or capping collapsed.
    pub(in crate::factor) fn degree_losses(&self) -> impl Iterator<Item = (I, u32)> + '_ {
        let to_pivot = (self.entries.iter()).map(|entry| (entry.neighbor, entry.copies.get()));
        to_pivot.chain(self.removed_copies.iter().copied())
    }
}
