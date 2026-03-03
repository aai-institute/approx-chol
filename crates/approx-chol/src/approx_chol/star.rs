use crate::graph::{EliminationGraph, Neighbor};
use crate::ordering::EliminationOrdering;
use crate::Real;

use super::dedup::{Ac2DedupWorkspace, AcDedupWorkspace};

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
    pub fn new(n: usize) -> Self {
        Self {
            raw: Vec::new(),
            entries: Vec::new(),
            dedup: AcDedupWorkspace::new(n),
        }
    }

    /// Build a deduplicated star sorted by ascending weight.
    pub fn build_star<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    ) {
        graph.live_neighbors(v, &mut self.raw);
        self.dedup.dedup(&mut self.raw, &mut self.entries);
        for &u in self.dedup.merged() {
            ordering.notify_edges_merged(u);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[(u32, T)] {
        &self.entries
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
    /// Max multi-edges kept per neighbor pair after compression.
    merge_limit: u32,
    dedup: Ac2DedupWorkspace<T>,
}

impl<T: Real> Ac2StarBuilder<T> {
    pub fn new(n: usize, merge_limit: u32) -> Self {
        Self {
            raw: Vec::new(),
            entries: Vec::new(),
            counts: Vec::new(),
            merge_limit,
            dedup: Ac2DedupWorkspace::new(n),
        }
    }

    /// Build a deduplicated star sorted by average weight with merge-capped counts.
    pub fn build_star<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    ) {
        graph.live_neighbors(v, &mut self.raw);
        self.dedup.dedup(
            &mut self.raw,
            &mut self.entries,
            &mut self.counts,
            self.merge_limit,
        );
        for &(u, n_merged) in self.dedup.merged_counts() {
            ordering.notify_edges_merged_n(u, n_merged);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[(u32, T)] {
        &self.entries
    }

    pub fn counts(&self) -> &[u32] {
        &self.counts
    }
}
