use core::cmp::Ordering;

use super::copies::Copies;
use super::CliqueTreeSampler;
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

/// Searching the merged prefix beats the scatter path's random access up to here.
const SEARCH_LIMIT: usize = 8;

/// Rests in every slot of [`CliqueTreeSampler::scatter`] between merges.
fn unseen<I: Index>() -> I {
    I::max_value()
}

impl<T: Real, I: Index, C: Copies> CliqueTreeSampler<T, I, C> {
    /// Merges duplicate neighbors under the sampler's `split` cap, then sorts by per-copy weight.
    pub(super) fn refill(&mut self, neighbors: impl Iterator<Item = (I, C, T)>) {
        self.load(neighbors);
        let merged = match self.entries.len() <= SEARCH_LIMIT {
            true => self.merge_by_search(),
            false => self.merge_by_scatter(),
        };
        self.entries.truncate(merged);
        self.entries.sort_unstable_by(by_per_copy_then_neighbor);
    }

    fn load(&mut self, neighbors: impl Iterator<Item = (I, C, T)>) {
        self.entries.clear();
        let entries = neighbors.map(|(neighbor, copies, weight)| StarEntry {
            neighbor,
            copies,
            weight,
        });
        self.entries.extend(entries);
    }

    /// First-seen order, each entry searched in the merged prefix; returns the merged count.
    fn merge_by_search(&mut self) -> usize {
        let mut write = 0;
        for read in 0..self.entries.len() {
            let entry = self.entries[read];
            let seen =
                (self.entries[..write].iter()).position(|merged| merged.neighbor == entry.neighbor);
            match seen {
                Some(position) => self.absorb(position, entry),
                None => {
                    self.entries[write] = entry;
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

    /// Capped as the sum grows, which keeps the cap of the whole sum: no input exceeds the split.
    fn absorb(&mut self, position: usize, duplicate: StarEntry<T, I, C>) {
        let merged = &mut self.entries[position];
        merged.weight = merged.weight + duplicate.weight;
        let copies = merged.copies.get().saturating_add(duplicate.copies.get());
        merged.copies = self.split.cap(copies);
    }

    /// Neighbors already unique, each at the split's copies, which keeps per-copy order the raw order: one sort, no scratch.
    pub(super) fn refill_unique(&mut self, neighbors: impl Iterator<Item = (I, T)>) {
        let copies = self.split;
        self.entries.clear();
        let entries = neighbors.map(|(neighbor, weight)| StarEntry {
            neighbor,
            copies,
            weight,
        });
        self.entries.extend(entries);
        self.entries.sort_unstable_by(by_weight_then_neighbor);
    }

    /// The last star's neighbors in clique-tree order: every vertex its pivot's elimination moved.
    pub(in crate::factor) fn neighbors(&self) -> impl Iterator<Item = I> + '_ {
        self.entries.iter().map(|entry| entry.neighbor)
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::super::{CliqueTreeSampler, Copies, Multi};

    /// Merged with `merge` under a split of `k`, as `(neighbor, copies, weight bits)`.
    fn merged(
        k: u32,
        star: &[(u32, u32, f32)],
        merge: fn(&mut CliqueTreeSampler<f32, u32, Multi>) -> usize,
    ) -> Vec<(u32, u64, u32)> {
        let split = Multi::split(k).expect("AC2");
        let mut sampler = CliqueTreeSampler::new(split);
        sampler.load(
            (star.iter()).map(|&(neighbor, copies, weight)| (neighbor, split.cap(copies), weight)),
        );
        let merged = merge(&mut sampler);
        (sampler.entries[..merged].iter())
            .map(|e| (e.neighbor, u64::from(e.copies.get()), e.weight.to_bits()))
            .collect()
    }

    /// First-seen order, weights summed in input order, copies summed whole in `u64`, then capped at `k`.
    fn model(k: u32, star: &[(u32, u32, f32)]) -> Vec<(u32, u64, u32)> {
        let mut merged: Vec<(u32, u64, f32)> = Vec::new();
        for &(neighbor, copies, weight) in star {
            match merged.iter_mut().find(|seen| seen.0 == neighbor) {
                Some(seen) => (seen.1, seen.2) = (seen.1 + u64::from(copies), seen.2 + weight),
                None => merged.push((neighbor, u64::from(copies), weight)),
            }
        }
        (merged.into_iter())
            .map(|(neighbor, copies, weight)| {
                (neighbor, copies.min(u64::from(k)), weight.to_bits())
            })
            .collect()
    }

    proptest! {
        /// f32, where three duplicates' sum can depend on their order; copies within the split, as the graph keeps them, summing past it and, at 2^31, past `u32`.
        #[test]
        fn merges_follow_the_model(
            k in prop_oneof![Just(4u32), Just(1 << 31)],
            star in vec((0u32..6, prop_oneof![1u32..6, (1u32 << 31) - 2..=1 << 31], 0.1f32..10.0), 0..40),
        ) {
            let star: Vec<_> = (star.into_iter()).map(|(neighbor, copies, weight)| (neighbor, copies.min(k), weight)).collect();
            prop_assert_eq!(merged(k, &star, CliqueTreeSampler::merge_by_search), model(k, &star));
            prop_assert_eq!(merged(k, &star, CliqueTreeSampler::merge_by_scatter), model(k, &star));
        }
    }
}
