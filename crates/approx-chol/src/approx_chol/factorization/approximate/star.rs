use super::ordering::{DegreeDeltas, DynamicOrdering};
use crate::graph::{AdjListGraph, EdgeCount, Neighbor};
use crate::types::{float_total_cmp, Real};
use core::cmp::Ordering;

/// Copies are stored the way the graph stores them, so a single-copy layout spends no
/// space on a count it knows and no arithmetic dividing by it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StarEntry<T, C> {
    pub neighbor: u32,
    pub copies: C,
    pub weight: T,
}

/// Ascending by weight, ties by neighbor index. Total on a deduped star, which is what
/// keeps the sampled clique tree off `sort_unstable`'s element-width heuristics.
#[inline]
fn by_weight_then_neighbor<T: Real, C>(a: &StarEntry<T, C>, b: &StarEntry<T, C>) -> Ordering {
    float_total_cmp(&a.weight, &b.weight).then_with(|| a.neighbor.cmp(&b.neighbor))
}

/// A pivot's deduped neighborhood — one entry per unique neighbor, ordered as the
/// clique-tree path eliminates them — and what collapsing it cost each neighbor's
/// degree.
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

    /// Every neighbor at the same multiplicity, the shape the standalone sampler is
    /// handed. Refills in place, so sampling a whole elimination allocates once.
    ///
    /// One multiplicity throughout makes `per_copy` order-preserving, so this orders on
    /// the raw weight and skips [`Self::sort`]'s scratch round-trip.
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

    fn push(&mut self, entry: StarEntry<T, C>) {
        self.entries.push(entry);
    }

    /// One call per unique neighbor, so the cap needs no second pass.
    fn push_capped(&mut self, neighbor: u32, weight: T, copies: u32, limit: u32) {
        let (copies, dropped) = C::cap(copies, limit);
        if dropped > 0 {
            self.removed_copies.push((neighbor, dropped));
        }
        self.push(StarEntry {
            neighbor,
            copies,
            weight,
        });
    }

    pub(super) fn entries(&self) -> &[StarEntry<T, C>] {
        &self.entries
    }

    /// Duplicates collapsed into one entry, or multiplicity the cap discarded: one
    /// degree decrement either way, which is why one ledger carries both.
    pub(super) fn removed_copies(&self) -> &[(u32, u32)] {
        &self.removed_copies
    }

    pub(super) fn accumulate_removal_delta(&self, deltas: &mut DegreeDeltas) {
        for entry in &self.entries {
            deltas.decrease(entry.neighbor, entry.copies.get());
        }
    }

    /// Ascending by the weight one copy carries, ties by neighbor index. The quotient
    /// is precomputed: cross-multiplying in the comparator can break transitivity.
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

/// Immediate, not batched through [`DegreeDeltas`], so a merge driving the estimate
/// below zero loses the excess rather than offsetting the same step's fill.
fn apply_removed_copies(merged: &[(u32, u32)], ordering: &mut DynamicOrdering) {
    for &(u, n_merged) in merged {
        ordering.decrease(u as usize, n_merged);
    }
}

/// AC and AC2 are the same builder over different edge layouts: [`EdgeCount`] holds
/// everything that differs.
pub(super) struct StarBuilder<T: Real, C: EdgeCount> {
    star: Star<T, C>,
    dedup: DedupWorkspace<T, C>,
    copies: u32,
}

impl<T: Real, C: EdgeCount> StarBuilder<T, C> {
    pub(super) fn new(n: usize, copies: u32) -> Self {
        Self {
            star: Star::new(),
            dedup: DedupWorkspace::new(n),
            copies,
        }
    }

    pub(super) fn build_star(
        &mut self,
        graph: &mut AdjListGraph<C, T>,
        v: usize,
        ordering: &mut DynamicOrdering,
    ) {
        self.dedup.collect(graph, v);
        self.dedup.dedup(&mut self.star, self.copies);
        apply_removed_copies(self.star.removed_copies(), ordering);
    }

    /// Entries are empty when the pivot had no live neighbor left.
    pub(super) fn star(&self) -> &Star<T, C> {
        &self.star
    }
}

/// At or below this many entries, sorting beats the scatter path's random access.
const SCATTER_THRESHOLD: usize = 32;

/// Every per-vertex slot is left at zero, so a zero `count` also marks a vertex
/// unvisited — no separate seen-set to keep in step with it.
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

    /// First-seen order, resetting each slot as it goes so the buffers are all-zero
    /// again when this returns.
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

    fn collect(&mut self, graph: &AdjListGraph<C, T>, v: usize) {
        graph.live_neighbors(v, &mut self.raw);
    }

    /// The two paths differ only in how they find the duplicates; neither caps, so
    /// neither can report a merge the other would not.
    pub(super) fn dedup(&mut self, star: &mut Star<T, C>, limit: u32) {
        star.clear();
        if self.raw.len() <= SCATTER_THRESHOLD {
            self.dedup_by_sort(star, limit);
        } else {
            self.dedup_by_scatter(star, limit);
        }
        star.sort();
    }

    fn dedup_by_sort(&mut self, star: &mut Star<T, C>, limit: u32) {
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

    fn dedup_by_scatter(&mut self, star: &mut Star<T, C>, limit: u32) {
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
