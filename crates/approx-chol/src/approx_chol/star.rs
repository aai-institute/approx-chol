use crate::graph::{AdjListGraph, EdgeCount, Multi, Neighbor, Single};
use crate::ordering::{DegreeDeltas, DynamicOrdering};
use crate::sampling::CdfSampler;
use crate::types::float_total_cmp;
use crate::Real;

use super::clique_tree::{
    clique_tree_sample_column, clique_tree_sample_column_multi, MultiStar, SampledColumn,
};

/// Apply the merge-compression degree decrease immediately (not batched through
/// [`DegreeDeltas`]), so it floors at zero *before* the step's fill/removal net
/// delta — matching the per-edge baseline, where a merge driving the estimate
/// below zero loses the excess rather than offsetting later fill.
fn apply_merged_counts(merged: &[(u32, u32)], ordering: &mut DynamicOrdering) {
    for &(u, n_merged) in merged {
        ordering.decrease(u as usize, n_merged);
    }
}

pub(super) trait StarBuilderVariant<T: Real> {
    /// Edge multiplicity storage this variant eliminates on: what makes an AC
    /// builder over a split multi-edge graph a type error rather than a mistake
    /// the caller has to avoid.
    type Count: EdgeCount;

    fn build_star(
        &mut self,
        graph: &mut AdjListGraph<Self::Count, T>,
        v: usize,
        ordering: &mut DynamicOrdering,
    );
    /// The star's neighbors; empty when the pivot had no live neighbor left.
    fn entries(&self) -> &[(u32, T)];
    fn sample_column(
        &self,
        pivot_diag: T,
        sampler: &mut CdfSampler<T>,
        column: &mut SampledColumn<T>,
    );
    /// Accumulate the degree decrease each surviving neighbor experiences from
    /// this vertex's elimination (negative deltas).
    fn accumulate_removal_delta(&self, deltas: &mut DegreeDeltas);
}

/// Star neighborhood builder for standard AC factorization.
///
/// Lightweight variant: no multi-edge counts or merge limit.
pub(super) struct AcStarBuilder<T: Real> {
    /// Raw neighbor output from `live_neighbors`.
    raw: Vec<Neighbor<T, Single>>,
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
    type Count = Single;

    fn build_star(
        &mut self,
        graph: &mut AdjListGraph<Single, T>,
        v: usize,
        ordering: &mut DynamicOrdering,
    ) {
        graph.live_neighbors(v, &mut self.raw);
        self.dedup.dedup(&mut self.raw, &mut self.entries);
        apply_merged_counts(self.dedup.merged_counts(), ordering);
    }

    fn entries(&self) -> &[(u32, T)] {
        &self.entries
    }

    fn sample_column(
        &self,
        pivot_diag: T,
        sampler: &mut CdfSampler<T>,
        column: &mut SampledColumn<T>,
    ) {
        clique_tree_sample_column(&self.entries, pivot_diag, sampler, column);
    }

    fn accumulate_removal_delta(&self, deltas: &mut DegreeDeltas) {
        // AC is the count-1 case of AC2's per-neighbor decrement below.
        for &(u, _) in &self.entries {
            deltas.decrease(u, 1);
        }
    }
}

/// Star neighborhood builder for AC2 factorization.
///
/// Tracks multi-edge counts per neighbor and enforces a merge limit.
pub(super) struct Ac2StarBuilder<T: Real> {
    /// Raw neighbor output from `live_neighbors`.
    raw: Vec<Neighbor<T, Multi>>,
    star: MultiStar<T>,
    /// Max multi-edges kept per neighbor pair after compression.
    merge_limit: u32,
    dedup: Ac2DedupWorkspace<T>,
}

impl<T: Real> Ac2StarBuilder<T> {
    pub(super) fn new(n: usize, merge_limit: u32) -> Self {
        Self {
            raw: Vec::new(),
            star: MultiStar::new(),
            merge_limit,
            dedup: Ac2DedupWorkspace::new(n),
        }
    }
}

impl<T: Real> StarBuilderVariant<T> for Ac2StarBuilder<T> {
    type Count = Multi;

    fn build_star(
        &mut self,
        graph: &mut AdjListGraph<Multi, T>,
        v: usize,
        ordering: &mut DynamicOrdering,
    ) {
        graph.live_neighbors(v, &mut self.raw);
        self.dedup
            .dedup(&mut self.raw, &mut self.star, self.merge_limit);
        apply_merged_counts(self.dedup.merged_counts(), ordering);
    }

    fn entries(&self) -> &[(u32, T)] {
        self.star.entries()
    }

    fn sample_column(
        &self,
        pivot_diag: T,
        sampler: &mut CdfSampler<T>,
        column: &mut SampledColumn<T>,
    ) {
        clique_tree_sample_column_multi(&self.star, pivot_diag, sampler, column);
    }

    fn accumulate_removal_delta(&self, deltas: &mut DegreeDeltas) {
        for (u, _, count) in self.star.iter() {
            deltas.decrease(u, count);
        }
    }
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
    pub fn dedup(&mut self, raw: &mut [Neighbor<T, Single>], entries: &mut Vec<(u32, T)>) {
        if raw.len() <= SCATTER_THRESHOLD {
            self.dedup_sort_small(raw, entries);
        } else {
            self.dedup_scatter(raw, entries);
        }
    }

    fn dedup_sort_small(&mut self, raw: &mut [Neighbor<T, Single>], entries: &mut Vec<(u32, T)>) {
        self.dedup_sort_core(raw, entries);
        sort_by_weight_then_index(entries);
    }

    fn dedup_sort_core(&mut self, raw: &mut [Neighbor<T, Single>], entries: &mut Vec<(u32, T)>) {
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

    fn dedup_scatter(&mut self, raw: &[Neighbor<T, Single>], entries: &mut Vec<(u32, T)>) {
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
}

impl<T: Real> Ac2DedupWorkspace<T> {
    pub fn new(n: usize) -> Self {
        Self {
            scratch: DedupScratch::new(n),
            scatter_counts: Vec::new(),
            merged_counts: Vec::new(),
        }
    }

    /// `(vertex, count)` pairs for merge-limit discards during the last dedup.
    pub fn merged_counts(&self) -> &[(u32, u32)] {
        &self.merged_counts
    }

    /// Deduplicate raw tuples for AC2 path, apply merge cap, and sort by avg-weight.
    pub fn dedup(
        &mut self,
        raw: &mut [Neighbor<T, Multi>],
        star: &mut MultiStar<T>,
        merge_limit: u32,
    ) {
        if raw.len() <= SCATTER_THRESHOLD {
            self.dedup_sort(raw, star);
        } else {
            self.dedup_scatter(raw, star);
        }
        star.apply_merge_limit(merge_limit, &mut self.merged_counts);
        star.sort_by_avg_weight();
    }

    fn dedup_sort(&mut self, raw: &mut [Neighbor<T, Multi>], star: &mut MultiStar<T>) {
        self.merged_counts.clear();
        star.clear();
        raw.sort_unstable_by_key(|n| n.to);
        for nbr in raw.iter() {
            star.push_or_merge(nbr.to, nbr.fill_weight, nbr.count.get());
        }
    }

    fn dedup_scatter(&mut self, raw: &[Neighbor<T, Multi>], star: &mut MultiStar<T>) {
        self.scratch.ensure_scatter_buffers();
        self.scratch.unique.clear();
        self.merged_counts.clear();
        star.clear();
        if self.scatter_counts.len() < self.scratch.n {
            self.scatter_counts.resize(self.scratch.n, 0);
        }

        for nbr in raw {
            let idx = nbr.to as usize;
            if self.scatter_counts[idx] == 0 {
                self.scratch.unique.push(nbr.to);
            }
            self.scratch.scatter[idx] = self.scratch.scatter[idx] + nbr.fill_weight;
            self.scatter_counts[idx] = self.scatter_counts[idx].saturating_add(nbr.count.get());
        }

        for &idx in &self.scratch.unique {
            let idx_usize = idx as usize;
            star.push(
                idx,
                self.scratch.scatter[idx_usize],
                self.scatter_counts[idx_usize],
            );
            self.scratch.scatter[idx_usize] = T::zero();
            self.scatter_counts[idx_usize] = 0;
        }
    }
}

#[cfg(test)]
mod tests;
