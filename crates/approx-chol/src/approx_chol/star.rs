use crate::graph::{EliminationGraph, Neighbor};
use crate::ordering::EliminationOrdering;
use crate::sampling::WeightedSampler;
use crate::types::float_total_cmp;
use crate::Real;
use num_traits::NumCast;

use super::clique_tree::{
    clique_tree_sample_column, clique_tree_sample_column_multi, SampledColumn,
};

pub(super) trait StarBuilderVariant<T: Real> {
    fn build_star<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    );
    fn is_empty(&self) -> bool;
    fn entries(&self) -> &[(u32, T)];
    fn sample_column<S: WeightedSampler<T>>(
        &self,
        pivot_diag: T,
        sampler: &mut S,
        column: &mut SampledColumn<T>,
    );
    fn notify_eliminated<O: EliminationOrdering<T>>(&self, ordering: &mut O, eliminated: usize);
}

/// Star neighborhood builder for standard AC factorization.
///
/// Lightweight variant: no multi-edge counts or merge limit.
pub(super) struct AcStarBuilder<T: Real> {
    /// Raw neighbor output from `live_neighbors`.
    raw: Vec<Neighbor<T>>,
    entries: Vec<(u32, T)>,
    dedup: AcDedupWorkspace<T>,
}

impl<T: Real> AcStarBuilder<T> {
    pub(super) fn new(n: usize) -> Self {
        Self {
            raw: Vec::new(),
            entries: Vec::new(),
            dedup: AcDedupWorkspace::new(n),
        }
    }
}

impl<T: Real> StarBuilderVariant<T> for AcStarBuilder<T> {
    fn build_star<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    ) {
        graph.live_neighbors(v, &mut self.raw);
        self.dedup.dedup(&mut self.raw, &mut self.entries);
        for &(u, n_merged) in self.dedup.merged_counts() {
            ordering.notify_edges_merged_n(u, n_merged);
        }
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn entries(&self) -> &[(u32, T)] {
        &self.entries
    }

    fn sample_column<S: WeightedSampler<T>>(
        &self,
        pivot_diag: T,
        sampler: &mut S,
        column: &mut SampledColumn<T>,
    ) {
        clique_tree_sample_column(&self.entries, pivot_diag, sampler, column);
    }

    fn notify_eliminated<O: EliminationOrdering<T>>(&self, ordering: &mut O, eliminated: usize) {
        ordering.notify_eliminated(eliminated, &self.entries);
    }
}

/// Star neighborhood builder for AC2 factorization.
///
/// Tracks multi-edge counts per neighbor and enforces a merge limit.
pub(super) struct Ac2StarBuilder<T: Real> {
    /// Raw neighbor output from `live_neighbors`.
    raw: Vec<Neighbor<T>>,
    entries: Vec<(u32, T)>,
    /// Multi-edge count per unique neighbor after compression.
    counts: Vec<u32>,
    /// Sum of deduplicated incident weights for the current star.
    total_weight: T,
    /// Max multi-edges kept per neighbor pair after compression.
    merge_limit: u32,
    dedup: Ac2DedupWorkspace<T>,
}

impl<T: Real> Ac2StarBuilder<T> {
    pub(super) fn new(n: usize, merge_limit: u32) -> Self {
        Self {
            raw: Vec::new(),
            entries: Vec::new(),
            counts: Vec::new(),
            total_weight: T::zero(),
            merge_limit,
            dedup: Ac2DedupWorkspace::new(n),
        }
    }
}

impl<T: Real> StarBuilderVariant<T> for Ac2StarBuilder<T> {
    fn build_star<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    ) {
        graph.live_neighbors(v, &mut self.raw);
        self.total_weight = self.dedup.dedup(
            &mut self.raw,
            &mut self.entries,
            &mut self.counts,
            self.merge_limit,
        );
        for &(u, n_merged) in self.dedup.merged_counts() {
            ordering.notify_edges_merged_n(u, n_merged);
        }
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn entries(&self) -> &[(u32, T)] {
        &self.entries
    }

    fn sample_column<S: WeightedSampler<T>>(
        &self,
        pivot_diag: T,
        sampler: &mut S,
        column: &mut SampledColumn<T>,
    ) {
        clique_tree_sample_column_multi(
            &self.entries,
            &self.counts,
            self.total_weight,
            pivot_diag,
            sampler,
            column,
        );
    }

    fn notify_eliminated<O: EliminationOrdering<T>>(&self, ordering: &mut O, _eliminated: usize) {
        debug_assert_eq!(self.entries.len(), self.counts.len());
        for (&(u, _), &count) in self.entries.iter().zip(self.counts.iter()) {
            ordering.notify_neighbor_removed_n(u, count);
        }
    }
}

#[derive(Clone, Copy)]
struct Ac2SortEntry<T: Real> {
    idx: u32,
    weight: T,
    count: u32,
    /// Precomputed avg weight (`weight / count`) used as a sort key.
    /// Avoids cross-multiplication in the comparator, which can break
    /// transitivity under floating-point rounding.
    avg_weight: T,
}

/// Neighborhoods with at most this many entries use sort-based dedup (O(d log d),
/// cache-friendly for small d). Larger neighborhoods use scatter-gather (O(d) via
/// indexed buffers, but with higher constant from random-access pattern).
const SCATTER_THRESHOLD: usize = 32;

/// Sort entries by weight (ascending), breaking ties by vertex index.
fn sort_by_weight_then_index<T: Real>(entries: &mut [(u32, T)]) {
    entries.sort_unstable_by(|a, b| float_total_cmp(&a.1, &b.1).then_with(|| a.0.cmp(&b.0)));
}

/// Shared scratch for dedup variants.
struct DedupScratch<T: Real> {
    /// `scatter[idx]` accumulates weight for vertex `idx`.
    scatter: Vec<T>,
    /// Tracks first-seen vertices for AC scatter dedup.
    scatter_seen: Vec<bool>,
    /// Tracks unique vertex indices seen in the current pass.
    unique: Vec<u32>,
    /// Number of vertices in the graph (for buffer sizing).
    n: usize,
}

impl<T: Real> DedupScratch<T> {
    fn new(n: usize) -> Self {
        Self {
            scatter: Vec::new(),
            scatter_seen: Vec::new(),
            unique: Vec::new(),
            n,
        }
    }

    fn ensure_scatter_buffers(&mut self) {
        if self.scatter.len() < self.n {
            self.scatter.resize(self.n, T::zero());
            self.scatter_seen.resize(self.n, false);
        }
    }
}

/// AC dedup workspace (weights only, tracks merged duplicate counts).
pub(super) struct AcDedupWorkspace<T: Real> {
    scratch: DedupScratch<T>,
    /// Number of duplicates merged per vertex.
    merged_counts: Vec<(u32, u32)>,
    /// Duplicate counter per vertex for scatter dedup.
    scatter_merged_counts: Vec<u32>,
}

impl<T: Real> AcDedupWorkspace<T> {
    pub fn new(n: usize) -> Self {
        Self {
            scratch: DedupScratch::new(n),
            merged_counts: Vec::new(),
            scatter_merged_counts: Vec::new(),
        }
    }

    /// `(vertex, count)` pairs merged during the last dedup call.
    pub fn merged_counts(&self) -> &[(u32, u32)] {
        &self.merged_counts
    }

    /// Deduplicate raw tuples for AC path and sort by weight ascending.
    pub fn dedup(&mut self, raw: &mut [Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        if raw.len() <= SCATTER_THRESHOLD {
            self.dedup_sort_small(raw, entries);
        } else {
            self.dedup_scatter(raw, entries);
        }
    }

    fn dedup_sort_small(&mut self, raw: &mut [Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        self.dedup_sort_core(raw, entries);
        sort_by_weight_then_index(entries);
    }

    fn dedup_sort_core(&mut self, raw: &mut [Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        self.merged_counts.clear();
        entries.clear();
        if raw.is_empty() {
            return;
        }
        if raw.len() == 1 {
            entries.push((raw[0].to, raw[0].fill_weight));
            return;
        }

        raw.sort_unstable_by_key(|n| n.to);

        let mut write = 0;
        let mut n_merged: u32 = 0;
        for read in 1..raw.len() {
            if raw[write].to == raw[read].to {
                raw[write].fill_weight = raw[write].fill_weight + raw[read].fill_weight;
                n_merged = n_merged.saturating_add(1);
            } else {
                entries.push((raw[write].to, raw[write].fill_weight));
                if n_merged > 0 {
                    self.merged_counts.push((raw[write].to, n_merged));
                }
                write += 1;
                raw[write] = raw[read];
                n_merged = 0;
            }
        }
        entries.push((raw[write].to, raw[write].fill_weight));
        if n_merged > 0 {
            self.merged_counts.push((raw[write].to, n_merged));
        }
    }

    fn dedup_scatter(&mut self, raw: &[Neighbor<T>], entries: &mut Vec<(u32, T)>) {
        self.scratch.ensure_scatter_buffers();
        self.scratch.unique.clear();
        self.merged_counts.clear();
        entries.clear();
        if self.scatter_merged_counts.len() < self.scratch.n {
            self.scatter_merged_counts.resize(self.scratch.n, 0);
        }

        for nbr in raw {
            let idx = nbr.to as usize;
            if !self.scratch.scatter_seen[idx] {
                self.scratch.scatter_seen[idx] = true;
                self.scratch.unique.push(nbr.to);
            } else {
                self.scatter_merged_counts[idx] = self.scatter_merged_counts[idx].saturating_add(1);
            }
            self.scratch.scatter[idx] = self.scratch.scatter[idx] + nbr.fill_weight;
        }

        for &idx in &self.scratch.unique {
            let idx_usize = idx as usize;
            entries.push((idx, self.scratch.scatter[idx_usize]));
            let n_merged = self.scatter_merged_counts[idx_usize];
            if n_merged > 0 {
                self.merged_counts.push((idx, n_merged));
                self.scatter_merged_counts[idx_usize] = 0;
            }
            self.scratch.scatter[idx_usize] = T::zero();
            self.scratch.scatter_seen[idx_usize] = false;
        }
        sort_by_weight_then_index(entries);
    }
}

/// AC2 dedup workspace (weights + multiplicities + merge-cap reporting).
pub(super) struct Ac2DedupWorkspace<T: Real> {
    scratch: DedupScratch<T>,
    /// Scatter buffer for multi-edge counts during scatter-gather dedup.
    scatter_counts: Vec<u32>,
    /// Compressed merge counts for AC2 merge-limit discards.
    merged_counts: Vec<(u32, u32)>,
    /// Reusable packed scratch for AC2 avg-weight sorting.
    sort_entries: Vec<Ac2SortEntry<T>>,
}

impl<T: Real> Ac2DedupWorkspace<T> {
    pub fn new(n: usize) -> Self {
        Self {
            scratch: DedupScratch::new(n),
            scatter_counts: Vec::new(),
            merged_counts: Vec::new(),
            sort_entries: Vec::new(),
        }
    }

    /// `(vertex, count)` pairs for merge-limit discards during the last dedup.
    pub fn merged_counts(&self) -> &[(u32, u32)] {
        &self.merged_counts
    }

    /// Deduplicate raw tuples for AC2 path, apply merge cap, and sort by avg-weight.
    pub fn dedup(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) -> T {
        if raw.len() <= SCATTER_THRESHOLD {
            self.dedup_sort_small(raw, entries, counts, merge_limit)
        } else {
            self.dedup_scatter(raw, entries, counts, merge_limit)
        }
    }

    fn dedup_sort_small(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) -> T {
        let total_weight = self.dedup_sort_core(raw, entries, counts);
        Self::apply_merge_limit(entries, counts, merge_limit, &mut self.merged_counts);
        self.sort_by_avg_weight(entries, counts);
        total_weight
    }

    fn dedup_sort_core(
        &mut self,
        raw: &mut [Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
    ) -> T {
        self.merged_counts.clear();
        entries.clear();
        counts.clear();
        if raw.is_empty() {
            return T::zero();
        }
        if raw.len() == 1 {
            entries.push((raw[0].to, raw[0].fill_weight));
            counts.push(raw[0].count);
            return raw[0].fill_weight;
        }

        raw.sort_unstable_by_key(|n| n.to);

        let mut write = 0;
        let mut count: u32 = raw[0].count;
        let mut total_weight = T::zero();
        for read in 1..raw.len() {
            if raw[write].to == raw[read].to {
                raw[write].fill_weight = raw[write].fill_weight + raw[read].fill_weight;
                count = count.saturating_add(raw[read].count);
            } else {
                entries.push((raw[write].to, raw[write].fill_weight));
                counts.push(count);
                total_weight = total_weight + raw[write].fill_weight;
                count = raw[read].count;
                write += 1;
                raw[write] = raw[read];
            }
        }
        entries.push((raw[write].to, raw[write].fill_weight));
        counts.push(count);
        total_weight + raw[write].fill_weight
    }

    fn dedup_scatter(
        &mut self,
        raw: &[Neighbor<T>],
        entries: &mut Vec<(u32, T)>,
        counts: &mut Vec<u32>,
        merge_limit: u32,
    ) -> T {
        self.scratch.ensure_scatter_buffers();
        self.scratch.unique.clear();
        self.merged_counts.clear();
        entries.clear();
        counts.clear();
        if self.scatter_counts.len() < self.scratch.n {
            self.scatter_counts.resize(self.scratch.n, 0);
        }

        for nbr in raw {
            let idx = nbr.to as usize;
            if self.scatter_counts[idx] == 0 {
                self.scratch.unique.push(nbr.to);
            }
            self.scratch.scatter[idx] = self.scratch.scatter[idx] + nbr.fill_weight;
            self.scatter_counts[idx] = self.scatter_counts[idx].saturating_add(nbr.count);
        }

        let mut total_weight = T::zero();
        for &idx in &self.scratch.unique {
            let idx_usize = idx as usize;
            entries.push((idx, self.scratch.scatter[idx_usize]));
            counts.push(self.scatter_counts[idx_usize]);
            total_weight = total_weight + self.scratch.scatter[idx_usize];
            self.scratch.scatter[idx_usize] = T::zero();
            self.scatter_counts[idx_usize] = 0;
        }

        Self::apply_merge_limit(entries, counts, merge_limit, &mut self.merged_counts);
        self.sort_by_avg_weight(entries, counts);
        total_weight
    }

    /// Apply merge limit: cap multi-edge counts, preserving total weight.
    fn apply_merge_limit(
        entries: &[(u32, T)],
        counts: &mut [u32],
        merge_limit: u32,
        merged_counts: &mut Vec<(u32, u32)>,
    ) {
        let limit = merge_limit;
        for i in 0..entries.len() {
            let count = counts[i];
            if count > limit {
                let discarded = count - limit;
                counts[i] = limit;
                let idx = entries[i].0;
                merged_counts.push((idx, discarded));
            }
        }
    }

    /// Sort entries by average weight ascending, then by vertex index.
    fn sort_by_avg_weight(&mut self, entries: &mut [(u32, T)], counts: &mut [u32]) {
        let len = entries.len();
        self.sort_entries.clear();
        self.sort_entries.reserve(len);
        for i in 0..len {
            let count_scalar: T = <T as NumCast>::from(counts[i]).unwrap_or(T::one());
            self.sort_entries.push(Ac2SortEntry {
                idx: entries[i].0,
                weight: entries[i].1,
                count: counts[i],
                avg_weight: entries[i].1 / count_scalar,
            });
        }

        // Sort by precomputed avg_weight. Using a precomputed key guarantees
        // transitivity (each element maps to a fixed float). The previous
        // cross-multiplication approach (a.weight * b.count vs b.weight * a.count)
        // could violate transitivity under floating-point rounding.
        self.sort_entries.sort_unstable_by(|a, b| {
            float_total_cmp(&a.avg_weight, &b.avg_weight).then_with(|| a.idx.cmp(&b.idx))
        });

        for (dst, item) in self.sort_entries.iter().enumerate() {
            entries[dst] = (item.idx, item.weight);
            counts[dst] = item.count;
        }
    }
}

#[cfg(test)]
mod tests;
