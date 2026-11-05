use core::cmp::Ordering;

use rand::rngs::Xoshiro256PlusPlus;
use rand::Rng;

use super::copies::Copies;
use crate::storage::{widen, Index, Real};

/// Above this many entries past a draw's start, it bisects; a linear scan up to 32 (256 measured +4-5%).
const LINEAR_THRESHOLD: usize = 32;

/// A pivot's neighborhood, merged and capped, sorted ascending by per-copy weight, with its prefix sums; reused across pivots.
pub(in crate::factor) struct Star<T, I, C: Copies> {
    /// The deduplicated neighborhood in clique-tree order.
    entries: Vec<StarEntry<T, I, C>>,
    /// What merging or capping collapsed, per neighbor.
    removed_copies: Vec<(I, u32)>,
    /// Grown to `n` on first need: above 32 entries, dedup scatters instead of sorting 16-byte entries twice.
    scatter: Vec<I>,
    /// Each merged entry's summed copies before the cap: `C` cannot add.
    counts: Vec<u32>,
    /// The weights' prefix sums, heaviest last, so each suffix's draw range is one subtraction; `f64`, as `f32` loses short suffixes (#275).
    cumsum: Vec<f64>,
    split: C,
}

/// The pivot's column, read off its sorted star: each neighbor's share `w_i / d` but the heaviest's, which takes what the shares leave, and the pivot's weight.
pub(in crate::factor) struct Column<'a, T, I, C> {
    heaviest: &'a StarEntry<T, I, C>,
    rest: &'a [StarEntry<T, I, C>],
    d: f64,
}

impl<T: Real, I: Index, C> Column<'_, T, I, C> {
    pub(in crate::factor) fn weight(&self) -> T {
        self.heaviest.weight
    }

    pub(in crate::factor) fn remainder(&self) -> I {
        self.heaviest.neighbor
    }

    pub(in crate::factor) fn shares(&self) -> impl Iterator<Item = (I, f64)> + '_ {
        (self.rest.iter()).map(|entry| (entry.neighbor, widen(entry.weight) / self.d))
    }
}

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

/// Rests in every slot of [`Star::scatter`] between merges.
fn unseen<I: Index>() -> I {
    I::max_value()
}

impl<T: Real, I: Index, C: Copies> Star<T, I, C> {
    /// `split` caps every merge, so one star cannot mix two.
    pub(in crate::factor) fn new(split: C) -> Self {
        Self {
            entries: Vec::new(),
            removed_copies: Vec::new(),
            scatter: Vec::new(),
            counts: Vec::new(),
            cumsum: Vec::new(),
            split,
        }
    }

    /// Merges duplicate neighbors under the `split` cap, then sorts by per-copy weight.
    pub(in crate::factor) fn merged(&mut self, neighbors: impl Iterator<Item = (I, C, T)>) {
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
            let (copies, dropped) = self.split.cap(count);
            entry.copies = copies;
            if dropped > 0 {
                self.removed_copies.push((entry.neighbor, dropped));
            }
        }
        self.entries.sort_unstable_by(by_per_copy_then_neighbor);
        self.sum();
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

    /// Neighbors already unique, as a row of `C` is, each at the split's copies, which keeps per-copy order the raw order: one sort, no scratch.
    pub(in crate::factor) fn unique(&mut self, neighbors: impl Iterator<Item = (I, T)>) {
        let copies = self.split;
        self.entries.clear();
        self.removed_copies.clear();
        let entries = neighbors.map(|(neighbor, weight)| StarEntry {
            neighbor,
            copies,
            weight,
        });
        self.entries.extend(entries);
        self.entries.sort_unstable_by(by_weight_then_neighbor);
        self.sum();
    }

    fn sum(&mut self) {
        self.cumsum.clear();
        let mut acc = 0.0;
        for entry in &self.entries {
            acc += widen(entry.weight);
            self.cumsum.push(acc);
        }
        debug_assert!(
            self.entries.is_empty() || (self.total().is_finite() && self.total() > 0.0),
            "positive weights under the diagonal bound"
        );
    }

    pub(super) fn entries(&self) -> &[StarEntry<T, I, C>] {
        &self.entries
    }

    /// `d = ω_v`, the star's whole weight.
    pub(super) fn total(&self) -> f64 {
        *self.cumsum.last().expect("a prepared star")
    }

    /// The weight past entry `i`: one subtraction, as each draw's range.
    pub(super) fn past(&self, i: usize) -> f64 {
        self.total() - self.cumsum[i]
    }

    /// An entry past `i`, drawn by weight, clamped to the last: a draw can round up to the total.
    pub(super) fn draw_after(&self, rng: &mut Xoshiro256PlusPlus, i: usize) -> usize {
        let cumsum = &self.cumsum;
        let (start, end) = (i + 1, cumsum.len());
        // 53 bits times a power of two is exact, so never 1.0.
        let u = ((rng.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
        let r = u * self.past(i) + cumsum[i];
        let k = if end - start <= LINEAR_THRESHOLD {
            let mut k = start;
            while k < end && cumsum[k] < r {
                k += 1;
            }
            k
        } else {
            cumsum[start..end].partition_point(|&c| c < r) + start
        };
        k.min(end - 1)
    }

    pub(in crate::factor) fn column(&self) -> Column<'_, T, I, C> {
        let (heaviest, rest) =
            (self.entries.split_last()).expect("a pivot of a connected graph has a neighbor");
        Column {
            heaviest,
            rest,
            d: self.total(),
        }
    }

    /// Copies each neighbor lost: those to the pivot, and those merging or capping collapsed.
    pub(in crate::factor) fn degree_losses(&self) -> impl Iterator<Item = (I, u32)> + '_ {
        let to_pivot = (self.entries.iter()).map(|entry| (entry.neighbor, entry.copies.get()));
        to_pivot.chain(self.removed_copies.iter().copied())
    }
}
