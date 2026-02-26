use crate::graph::{EliminationGraph, Neighbor};
use crate::ordering::EliminationOrdering;
use crate::sampling::WeightedSampler;
use crate::Real;
use num_traits::NumCast;

use super::dedup::DedupWorkspace;
use super::sampled_column::{SampledColumn, StarElimination};
use super::Star;

/// Star neighborhood for AC2 factorization (Algorithm 6, GKS 2023).
///
/// Tracks multi-edge counts per neighbor and enforces a merge limit.
/// Uses batched sampling with `t` samples per neighbor and avg-weight ordering.
pub(super) struct Ac2Star<S: WeightedSampler<T>, T: Real> {
    /// Raw neighbor output from `live_neighbors`.
    raw: Vec<Neighbor<T>>,
    entries: Vec<(u32, T)>,
    /// Multi-edge count per unique neighbor after compression.
    counts: Vec<u32>,
    /// Max multi-edges kept per neighbor pair after compression.
    merge_limit: u32,
    dedup: DedupWorkspace<T>,
    sampler: S,
}

impl<S: WeightedSampler<T>, T: Real> Ac2Star<S, T> {
    pub fn new(n: usize, merge_limit: u32, sampler: S) -> Self {
        Self {
            raw: Vec::new(),
            entries: Vec::new(),
            counts: Vec::new(),
            merge_limit,
            dedup: DedupWorkspace::new(n),
            sampler,
        }
    }
}

impl<S: WeightedSampler<T>, T: Real> Star<T> for Ac2Star<S, T> {
    /// Fill from graph: get live neighbors, deduplicate with counts, sort by avg weight.
    fn compress<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &mut self,
        graph: &mut G,
        v: usize,
        ordering: &mut O,
    ) {
        graph.live_neighbors(v, &mut self.raw);
        self.dedup.dedup_ac2(
            &mut self.raw,
            &mut self.entries,
            &mut self.counts,
            self.merge_limit,
        );
        for &(u, n_merged) in self.dedup.merged_counts() {
            ordering.notify_edges_merged_n(u, n_merged);
        }
    }

    /// Algorithm 6 (GKS 2023): sample one column via batched clique-tree sampling.
    fn sample(&mut self, pivot_diag: T, column: &mut SampledColumn<T>) {
        debug_assert_eq!(self.entries.len(), self.counts.len());
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

        let total_weight = entries.iter().fold(T::zero(), |a, e| a + e.1);

        if total_weight <= T::near_zero() {
            // Degenerate star: no meaningful weight to distribute.
            // Record all neighbors with fraction 1/n and preserve pivot diagonal.
            column.diagonal = pivot_diag;
            for &(j, _) in entries.iter() {
                column.neighbors.push(j);
                column.fractions.push(T::one() / NumCast::from(n).unwrap());
            }
            return;
        }

        // weight of neighbors not yet processed (j_{i+1}..j_n)
        let mut remaining = total_weight;
        // Algorithm 6 line 8: d ← w(G,v) (sum of incident edge weights, not the
        // matrix diagonal).  The fill weight formula (line 15: w_new = w̄_vi · w(G,v) / d)
        // normalizes by this d, so StarElimination capacity must be total_weight to
        // keep fractions consistent.
        let mut elim = StarElimination::new(total_weight);

        self.sampler.prepare(entries);

        for (i, (&(j, w), &t)) in entries[..n - 1].iter().zip(self.counts.iter()).enumerate() {
            remaining = remaining - w;
            let f = elim.fraction(w);
            let fill_wt =
                w * remaining / (<T as NumCast>::from(t).expect("count to scalar") * total_weight); // averaged fill weight

            column.neighbors.push(j);
            column.fractions.push(f);
            column.sample_fill_edges(j, t, fill_wt, &mut self.sampler, entries, i + 1);
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

    /// Julia AC2 parity: surviving multi-edges decrement neighbor degree once
    /// per copy (counts carry multiplicities after merge-cap).
    fn notify_eliminated<O: EliminationOrdering<T>>(&self, ordering: &mut O, _v: usize) {
        debug_assert_eq!(self.entries.len(), self.counts.len());
        for (&(u, _), &count) in self.entries.iter().zip(self.counts.iter()) {
            ordering.notify_neighbor_removed_n(u, count);
        }
    }
}
