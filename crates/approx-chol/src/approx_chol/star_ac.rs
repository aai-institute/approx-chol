use crate::graph::{EliminationGraph, Neighbor};
use crate::ordering::EliminationOrdering;
use crate::sampling::WeightedSampler;
use crate::Real;

use super::dedup::AcDedupWorkspace;
use super::sampled_column::{SampledColumn, StarElimination};
use super::Star;

/// Star neighborhood for standard AC factorization (Algorithm 5, GKS 2023).
///
/// Lightweight variant: no multi-edge counts or merge limit.
/// Single sample per neighbor, exact Schur complement fill weights.
pub(super) struct AcStar<S: WeightedSampler<T>, T: Real> {
    /// Raw neighbor output from `live_neighbors`.
    raw: Vec<Neighbor<T>>,
    entries: Vec<(u32, T)>,
    dedup: AcDedupWorkspace<T>,
    sampler: S,
}

impl<S: WeightedSampler<T>, T: Real> AcStar<S, T> {
    pub fn new(n: usize, sampler: S) -> Self {
        Self {
            raw: Vec::new(),
            entries: Vec::new(),
            dedup: AcDedupWorkspace::new(n),
            sampler,
        }
    }
}

impl<S: WeightedSampler<T>, T: Real> Star<T> for AcStar<S, T> {
    /// Fill from graph: get live neighbors, deduplicate, sort by weight.
    fn compress<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
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

    /// Algorithm 5 (GKS 2023): sample one column of the approximate Cholesky factor.
    fn sample(&mut self, pivot_diag: T, column: &mut SampledColumn<T>) {
        let entries = &self.entries;
        let Some(n) = column.begin_sampling(entries, pivot_diag) else {
            return;
        };

        self.sampler.prepare(entries);
        let mut elim = StarElimination::new(pivot_diag);

        for (i, &(j, w)) in entries[..n - 1].iter().enumerate() {
            let f = elim.fraction(w);
            let fill_wt = f * (T::one() - f) * elim.capacity(); // exact Schur complement

            column.neighbors.push(j);
            column.fractions.push(f);
            column.sample_fill_edges(j, 1, fill_wt, &mut self.sampler, entries, i + 1);
            elim.advance(f);
        }

        column.finalize_sampling(entries[n - 1], &elim);
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn entries(&self) -> &[(u32, T)] {
        &self.entries
    }
}
