use super::ordering::{DegreeDeltas, DynamicOrdering};
use crate::graph::{AdjListGraph, EdgeCount, Neighbor};
use crate::types::Real;
use core::cmp::Ordering;

/// Copies stored as the graph stores them, so a single-copy star pays no count or division.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StarEntry<T, C> {
    pub neighbor: u32,
    pub copies: C,
    pub weight: T,
}

/// NaN last: `partial_cmp`'s `None` breaks the total order sorts require (1.81+ panics).
#[inline]
fn float_total_cmp<T: Real>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

/// Total on a deduped star, which keeps the clique tree off `sort_unstable`'s width heuristics.
#[inline]
fn by_weight_then_neighbor<T: Real, C>(a: &StarEntry<T, C>, b: &StarEntry<T, C>) -> Ordering {
    float_total_cmp(&a.weight, &b.weight).then_with(|| a.neighbor.cmp(&b.neighbor))
}

/// A pivot's deduped neighborhood in clique-tree order, and what collapsing it cost each degree.
pub(super) struct Star<T: Real, C> {
    entries: Vec<StarEntry<T, C>>,
    removed_copies: Vec<(u32, u32)>,
    /// A single-copy star sorts in place and never touches this.
    sort_scratch: Vec<(T, StarEntry<T, C>)>,
}

impl<T: Real, C: EdgeCount> Star<T, C> {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            removed_copies: Vec::new(),
            sort_scratch: Vec::new(),
        }
    }

    /// One multiplicity keeps `per_copy` monotone, so this sorts raw weights, skipping the scratch.
    pub(super) fn refill_uniform(&mut self, entries: &[(u32, T)], copies: C) {
        self.clear();
        self.entries
            .extend(entries.iter().map(|&(neighbor, weight)| StarEntry {
                neighbor,
                copies,
                weight,
            }));
        self.entries.sort_unstable_by(by_weight_then_neighbor);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.removed_copies.clear();
    }

    /// One call per unique neighbor, so the cap needs no second pass.
    fn push_capped(&mut self, neighbor: u32, weight: T, copies: u32, limit: C::Split) {
        let (copies, dropped) = C::cap(copies, limit);
        if dropped > 0 {
            self.removed_copies.push((neighbor, dropped));
        }
        self.entries.push(StarEntry {
            neighbor,
            copies,
            weight,
        });
    }

    pub(super) fn entries(&self) -> &[StarEntry<T, C>] {
        &self.entries
    }

    /// Collapsed duplicates or capped copies: both are one degree decrement, hence one ledger.
    pub(super) fn removed_copies(&self) -> &[(u32, u32)] {
        &self.removed_copies
    }

    pub(super) fn accumulate_removal_delta(&self, deltas: &mut DegreeDeltas) {
        for entry in &self.entries {
            deltas.decrease(entry.neighbor, entry.copies.get());
        }
    }

    /// Per-copy weight is precomputed: cross-multiplying in the comparator can break transitivity.
    fn sort(&mut self) {
        if self.entries.len() <= 1 {
            return;
        }
        if C::SINGLE_COPY {
            self.entries.sort_unstable_by(by_weight_then_neighbor);
            return;
        }
        self.sort_scratch.clear();
        self.sort_scratch.reserve(self.entries.len());
        for entry in &self.entries {
            self.sort_scratch
                .push((entry.copies.per_copy(entry.weight), *entry));
        }
        self.sort_scratch.sort_unstable_by(|a, b| {
            float_total_cmp(&a.0, &b.0).then_with(|| a.1.neighbor.cmp(&b.1.neighbor))
        });
        for (slot, &(_, entry)) in self.entries.iter_mut().zip(&self.sort_scratch) {
            *slot = entry;
        }
    }
}

/// Unbatched, so a merge below zero loses its excess instead of offsetting this step's fill.
fn apply_removed_copies(merged: &[(u32, u32)], ordering: &mut DynamicOrdering) {
    for &(u, n_merged) in merged {
        ordering.decrease(u as usize, n_merged);
    }
}

/// AC and AC2 share this builder; [`EdgeCount`] holds everything that differs.
pub(super) struct StarBuilder<T: Real, C: EdgeCount> {
    star: Star<T, C>,
    dedup: DedupWorkspace<T, C>,
    split: C::Split,
}

impl<T: Real, C: EdgeCount> StarBuilder<T, C> {
    pub(super) fn new(n: usize, split: C::Split) -> Self {
        Self {
            star: Star::new(),
            dedup: DedupWorkspace::new(n),
            split,
        }
    }

    pub(super) fn build_star(
        &mut self,
        graph: &mut AdjListGraph<C, T>,
        v: usize,
        ordering: &mut DynamicOrdering,
    ) -> &Star<T, C> {
        graph.live_neighbors(v, &mut self.dedup.raw);
        self.dedup.dedup(&mut self.star, self.split);
        apply_removed_copies(self.star.removed_copies(), ordering);
        &self.star
    }
}

/// At or below this many entries, sorting beats the scatter path's random access.
const SCATTER_THRESHOLD: usize = 32;

/// Slots rest at zero, so a zero `count` also marks a vertex unvisited — no separate seen-set.
struct DedupScratch<T: Real> {
    scatter: Vec<T>,
    counts: Vec<u32>,
    unique: Vec<u32>,
    n: usize,
}

impl<T: Real> DedupScratch<T> {
    fn new(n: usize) -> Self {
        Self {
            scatter: Vec::new(),
            counts: Vec::new(),
            unique: Vec::new(),
            n,
        }
    }

    fn begin_pass(&mut self) {
        if self.scatter.len() < self.n {
            self.scatter.resize(self.n, T::zero());
            self.counts.resize(self.n, 0);
        }
        self.unique.clear();
    }

    #[inline]
    fn accumulate(&mut self, to: u32, weight: T, count: u32) {
        let idx = to as usize;
        if self.counts[idx] == 0 {
            self.unique.push(to);
        }
        self.scatter[idx] = self.scatter[idx] + weight;
        self.counts[idx] = self.counts[idx].saturating_add(count);
    }

    /// First-seen order, re-zeroing each slot so the buffers are all-zero again on return.
    #[inline]
    fn drain_unique(&mut self, mut visit: impl FnMut(u32, T, u32)) {
        for index in 0..self.unique.len() {
            let vertex = self.unique[index];
            let idx = vertex as usize;
            visit(vertex, self.scatter[idx], self.counts[idx]);
            self.scatter[idx] = T::zero();
            self.counts[idx] = 0;
        }
    }
}

pub(super) struct DedupWorkspace<T: Real, C> {
    raw: Vec<Neighbor<T, C>>,
    scratch: DedupScratch<T>,
}

impl<T: Real, C: EdgeCount> DedupWorkspace<T, C> {
    pub fn new(n: usize) -> Self {
        Self {
            raw: Vec::new(),
            scratch: DedupScratch::new(n),
        }
    }

    /// The paths differ only in finding duplicates; neither caps, so their merge reports agree.
    pub(super) fn dedup(&mut self, star: &mut Star<T, C>, limit: C::Split) {
        star.clear();
        if self.raw.len() <= SCATTER_THRESHOLD {
            self.dedup_by_sort(star, limit);
        } else {
            self.dedup_by_scatter(star, limit);
        }
        star.sort();
    }

    fn dedup_by_sort(&mut self, star: &mut Star<T, C>, limit: C::Split) {
        if self.raw.is_empty() {
            return;
        }
        self.raw.sort_unstable_by_key(|n| n.to);
        let first = self.raw[0];
        let mut run = (first.to, first.fill_weight, first.count.get());
        for neighbor in &self.raw[1..] {
            if neighbor.to == run.0 {
                run.1 = run.1 + neighbor.fill_weight;
                run.2 = run.2.saturating_add(neighbor.count.get());
            } else {
                star.push_capped(run.0, run.1, run.2, limit);
                run = (neighbor.to, neighbor.fill_weight, neighbor.count.get());
            }
        }
        star.push_capped(run.0, run.1, run.2, limit);
    }

    fn dedup_by_scatter(&mut self, star: &mut Star<T, C>, limit: C::Split) {
        self.scratch.begin_pass();
        for neighbor in &self.raw {
            self.scratch
                .accumulate(neighbor.to, neighbor.fill_weight, neighbor.count.get());
        }
        self.scratch.drain_unique(|vertex, weight, copies| {
            star.push_capped(vertex, weight, copies, limit);
        });
    }
}

#[cfg(test)]
mod tests;
