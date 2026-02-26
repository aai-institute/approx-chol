use crate::graph::{EliminationGraph, Neighbor};
use crate::ordering::EliminationOrdering;
use crate::sampling::WeightedSampler;
use crate::Real;

use super::dedup::DedupWorkspace;
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
    dedup: DedupWorkspace<T>,
    sampler: S,
}

impl<S: WeightedSampler<T>, T: Real> AcStar<S, T> {
    pub fn new(n: usize, sampler: S) -> Self {
        Self {
            raw: Vec::new(),
            entries: Vec::new(),
            dedup: DedupWorkspace::new(n),
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
        self.dedup.dedup_ac(&mut self.raw, &mut self.entries);
        for &u in self.dedup.merged() {
            ordering.notify_edges_merged(u);
        }
    }

    /// Algorithm 5 (GKS 2023): sample one column of the approximate Cholesky factor.
    fn sample(&mut self, pivot_diag: T, column: &mut SampledColumn<T>) {
        column.clear();
        let entries = &self.entries;
        let n = entries.len();

        if n == 0 {
            column.diagonal = pivot_diag;
            return;
        }

        if n == 1 {
            column.neighbors.push(entries[0].0);
            column.fractions.push(T::one());
            column.diagonal = pivot_diag;
            return;
        }

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

        let (j_last, w_last) = entries[n - 1];
        column.neighbors.push(j_last);
        column.fractions.push(T::one());
        column.diagonal = elim.diagonal(w_last);
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn entries(&self) -> &[(u32, T)] {
        &self.entries
    }
}
