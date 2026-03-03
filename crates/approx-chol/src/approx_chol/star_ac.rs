use crate::graph::{EliminationGraph, Neighbor};
use crate::ordering::EliminationOrdering;
use crate::Real;

use super::dedup::AcDedupWorkspace;

/// Star neighborhood for standard AC factorization (Algorithm 5, GKS 2023).
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
    /// Fill from graph: get live neighbors, deduplicate, sort by weight.
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
