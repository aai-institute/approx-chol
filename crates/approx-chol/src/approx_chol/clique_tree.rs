use core::cmp::Ordering;

use crate::graph::EliminationGraph;
use crate::ordering::EliminationOrdering;
use crate::sampling::{near_zero, CdfSampler, WeightedSampler};
use crate::types::Real;
use num_traits::NumCast;

/// One sampled column of the approximate Cholesky factor (Algorithm 5, GKS 2023).
///
/// Represents the result of clique-tree sampling on a star neighborhood.
/// Contains the column's diagonal entry, its non-zero neighbor indices with
/// fractional weights, and the fill edges to be inserted back into the graph.
///
/// Reusable across elimination steps (cleared at start of each sampling pass).
pub(crate) struct SampledColumn<T: Real> {
    /// Diagonal value of the factor column: `L[v,v]`.
    pub diagonal: T,
    /// Neighbor indices in the column's non-zero pattern.
    pub neighbors: Vec<u32>,
    /// Fractional weight for each neighbor: `L[neighbor, v] / L[v, v]`.
    pub fractions: Vec<T>,
    /// Fill edges `(u, w, weight)` to insert into the graph after elimination.
    fill_edges: Vec<(u32, u32, T)>,
}

impl<T: Real> SampledColumn<T> {
    pub(crate) fn new() -> Self {
        Self {
            diagonal: T::zero(),
            neighbors: Vec::new(),
            fractions: Vec::new(),
            fill_edges: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.diagonal = T::zero();
        self.neighbors.clear();
        self.fractions.clear();
        self.fill_edges.clear();
    }

    /// Initialize column sampling from a star's deduplicated neighbor list.
    ///
    /// Returns `Some(n)` when sampling should continue (`n >= 2`), otherwise
    /// writes the trivial result (`n == 0` or `n == 1`) and returns `None`.
    fn begin_sampling(&mut self, entries: &[(u32, T)], pivot_diag: T) -> Option<usize> {
        self.clear();
        match entries {
            [] => {
                self.diagonal = pivot_diag;
                None
            }
            [(j, _)] => {
                self.neighbors.push(*j);
                self.fractions.push(T::one());
                self.diagonal = pivot_diag;
                None
            }
            _ => Some(entries.len()),
        }
    }

    /// Finalize sampling with the last star neighbor (always fraction 1).
    fn finalize_sampling(&mut self, last: (u32, T), elim: &StarElimination<T>) {
        self.neighbors.push(last.0);
        self.fractions.push(T::one());
        self.diagonal = elim.diagonal(last.1);
    }

    /// Apply fill-in edges to the graph, update diagonal values, and notify ordering.
    pub(crate) fn apply_fill_in<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
        &self,
        graph: &mut G,
        diag: &mut [T],
        ordering: &mut O,
    ) {
        for &(u, w, weight) in &self.fill_edges {
            graph.add_fill_edge(u, w, weight);
            diag[u as usize] = diag[u as usize] + weight;
            diag[w as usize] = diag[w as usize] + weight;
            ordering.notify_fill_edge(u, w);
        }
    }

    /// Sample fill edges between `neighbor` and random neighbors from `entries[tail..]`.
    fn sample_fill_edges(
        &mut self,
        neighbor: u32,
        n_samples: u32,
        fill_weight: T,
        sampler: &mut impl WeightedSampler<T>,
        entries: &[(u32, T)],
        tail: usize,
    ) {
        if n_samples == 0 || fill_weight <= T::near_zero() {
            return;
        }
        let n = entries.len();
        if tail >= n {
            return;
        }
        for _ in 0..n_samples {
            if let Some(koff) = sampler.sample_from_range(tail, n) {
                let k = entries[koff].0;
                if neighbor != k {
                    self.fill_edges.push((neighbor, k, fill_weight));
                }
            }
        }
    }
}

/// Running state for sequential edge elimination on a star graph.
///
/// When eliminating pivot vertex v, its neighbors are processed sequentially
/// along a clique-tree path (GKS 2023, Algorithms 5 & 6). For each neighbor
/// j_i with edge weight w_i, the elimination fraction is
/// `f_i = w_i * scale / capacity`.
///
/// **Fields:**
/// - `scale`: cumulative product of `(1 - f_k)` for all previously processed
///   neighbors k < i. Tracks how much of the original edge weight survives
///   after earlier samplings.
/// - `capacity`: remaining weight budget, updated as `capacity *= (1 - f_i)^2`
///   after each step. Initialized differently by variant:
///   - **AC**: `pivot_diag` (the matrix diagonal entry for the pivot)
///   - **AC2**: `total_weight` (sum of incident edge weights)
///
/// After `advance(f)`, both `scale` and `capacity` shrink,
/// ensuring subsequent fractions account for weight already consumed by
/// earlier fill edges.
struct StarElimination<T = f64> {
    scale: T,
    capacity: T,
}

impl<T: Real> StarElimination<T> {
    #[inline(always)]
    fn new(capacity: T) -> Self {
        Self {
            scale: T::one(),
            capacity,
        }
    }

    #[inline(always)]
    fn fraction(&self, w: T) -> T {
        debug_assert!(self.capacity > T::epsilon());
        w * self.scale / self.capacity
    }

    #[inline(always)]
    fn capacity(&self) -> T {
        self.capacity
    }

    #[inline(always)]
    fn advance(&mut self, f: T) {
        let retain = T::one() - f;
        self.scale = self.scale * retain;
        self.capacity = self.capacity * retain * retain;
    }

    #[inline(always)]
    fn diagonal(&self, last_weight: T) -> T {
        last_weight * self.scale
    }
}

/// Clique-tree sampling for AC stars (single sample per neighbor).
pub(crate) fn clique_tree_sample_column<T: Real, S: WeightedSampler<T>>(
    entries: &[(u32, T)],
    pivot_diag: T,
    sampler: &mut S,
    column: &mut SampledColumn<T>,
) {
    let Some(n) = column.begin_sampling(entries, pivot_diag) else {
        return;
    };

    sampler.prepare(entries);
    let mut elim = StarElimination::new(pivot_diag);

    for (i, &(j, w)) in entries[..n - 1].iter().enumerate() {
        let f = elim.fraction(w);
        let fill_wt = f * (T::one() - f) * elim.capacity();
        column.neighbors.push(j);
        column.fractions.push(f);
        column.sample_fill_edges(j, 1, fill_wt, sampler, entries, i + 1);
        elim.advance(f);
    }

    column.finalize_sampling(entries[n - 1], &elim);
}

/// Clique-tree sampling for AC2 stars (multi-sample per neighbor).
pub(crate) fn clique_tree_sample_column_multi<T: Real, S: WeightedSampler<T>>(
    entries: &[(u32, T)],
    counts: &[u32],
    _total_weight: T,
    pivot_diag: T,
    sampler: &mut S,
    column: &mut SampledColumn<T>,
) {
    debug_assert_eq!(entries.len(), counts.len());
    let Some(n) = column.begin_sampling(entries, pivot_diag) else {
        return;
    };

    // Preserve pre-optimization behavior: accumulate in sorted entry order.
    let total_weight = entries.iter().fold(T::zero(), |a, e| a + e.1);
    if total_weight <= T::near_zero() {
        column.diagonal = pivot_diag;
        for &(j, _) in entries {
            column.neighbors.push(j);
            column
                .fractions
                .push(T::one() / NumCast::from(n).expect("n to scalar"));
        }
        return;
    }

    sampler.prepare(entries);
    let mut remaining = total_weight;
    let mut elim = StarElimination::new(total_weight);

    for (i, (&(j, w), &count)) in entries[..n - 1].iter().zip(counts.iter()).enumerate() {
        remaining = remaining - w;
        let f = elim.fraction(w);
        let fill_wt =
            w * remaining / (<T as NumCast>::from(count).expect("count to scalar") * total_weight);
        column.neighbors.push(j);
        column.fractions.push(f);
        column.sample_fill_edges(j, count, fill_wt, sampler, entries, i + 1);
        elim.advance(f);
    }

    column.finalize_sampling(entries[n - 1], &elim);
}

/// Sample fill edges approximating the Schur complement clique of a star.
///
/// Given an eliminated vertex with weighted neighbors `entries` and diagonal
/// `pivot_diag`, walks neighbors sorted by ascending weight and samples one
/// fill edge per neighbor to a random later neighbor (AC clique-tree,
/// Algorithm 5 in Gao-Kyng-Spielman 2023).
///
/// Produces at most `n-1` fill edges (a spanning tree on the n neighbors).
/// Each fill edge is unbiased: `E[w(i,j)] = a_i * a_j / pivot_diag`.
///
/// `entries` is sorted in place. Fill edges are appended to `out`.
pub fn clique_tree_sample<T>(
    entries: &mut [(u32, T)],
    pivot_diag: T,
    seed: u64,
    out: &mut Vec<(u32, u32, T)>,
) where
    T: num_traits::Float + Send + Sync + 'static,
{
    let n = entries.len();
    if n <= 1 {
        return;
    }

    entries.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let mut sampler = CdfSampler::<T>::new(seed);
    sampler.prepare(entries);
    let mut elim = StarElimination::new(pivot_diag);

    for i in 0..n - 1 {
        let (j, w) = entries[i];
        let f = elim.fraction(w);
        let fill_wt = f * (T::one() - f) * elim.capacity();

        if fill_wt > near_zero::<T>() {
            if let Some(koff) = sampler.sample_from_range(i + 1, n) {
                let k = entries[koff].0;
                if j != k {
                    let (lo, hi) = if j < k { (j, k) } else { (k, j) };
                    out.push((lo, hi, fill_wt));
                }
            }
        }
        elim.advance(f);
    }
}

/// Sample AC2-style fill edges for a star with multiplicity `k`.
///
/// This is the multi-edge counterpart of [`clique_tree_sample`], following the
/// AC2 sampling logic (Algorithm 6 in Gao-Kyng-Spielman 2023).
///
/// `split_merge` controls the per-neighbor multiplicity used during sampling.
/// The function emits up to `split_merge * (n - 1)` edges.
///
/// `entries` is sorted in place (ascending by weight), and fill edges are
/// appended to `out`.
pub fn clique_tree_sample_multi<T>(
    entries: &mut [(u32, T)],
    split_merge: u32,
    seed: u64,
    out: &mut Vec<(u32, u32, T)>,
) where
    T: num_traits::Float + Send + Sync + 'static,
{
    let n = entries.len();
    if n <= 1 {
        return;
    }

    if split_merge == 0 {
        return;
    }
    let t = split_merge;
    let t_scalar: T = NumCast::from(t).expect("u32 to scalar");

    entries.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let mut sampler = CdfSampler::<T>::new(seed);
    sampler.prepare(entries);

    let total_weight = entries.iter().fold(T::zero(), |acc, &(_, w)| acc + w);
    if total_weight <= near_zero::<T>() {
        return;
    }

    let mut remaining = total_weight;
    let mut elim = StarElimination::new(total_weight);

    for i in 0..n - 1 {
        let (j, w) = entries[i];
        let f = elim.fraction(w);
        remaining = remaining - w;
        let fill_wt = w * remaining / (t_scalar * total_weight);

        if fill_wt > near_zero::<T>() {
            for _ in 0..t {
                if let Some(koff) = sampler.sample_from_range(i + 1, n) {
                    let k = entries[koff].0;
                    if j != k {
                        let (lo, hi) = if j < k { (j, k) } else { (k, j) };
                        out.push((lo, hi, fill_wt));
                    }
                }
            }
        }
        elim.advance(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_on_five_neighbors() {
        let mut entries: Vec<(u32, f64)> = vec![(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let pivot_diag: f64 = entries.iter().map(|(_, w)| w).sum();
        let mut out = Vec::new();

        clique_tree_sample(&mut entries, pivot_diag, 42, &mut out);

        assert!(out.len() <= 4, "got {} edges, expected <= 4", out.len());
        for &(lo, hi, w) in &out {
            assert!(lo < hi, "edge ({lo}, {hi}) not ordered");
            assert!(w > 0.0, "edge ({lo}, {hi}) has non-positive weight {w}");
        }
    }

    #[test]
    fn empty_and_single() {
        let mut out = Vec::new();

        clique_tree_sample(&mut [], 1.0, 0, &mut out);
        assert!(out.is_empty());

        let mut entries = vec![(0u32, 5.0)];
        clique_tree_sample(&mut entries, 5.0, 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn statistical_unbiasedness() {
        let base_entries: Vec<(u32, f64)> = vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)];
        let pivot_diag: f64 = base_entries.iter().map(|(_, w)| w).sum();

        let n_trials = 50_000;
        let mut pair_total = std::collections::HashMap::<(u32, u32), f64>::new();

        for trial in 0..n_trials {
            let mut entries = base_entries.clone();
            let mut out = Vec::new();
            clique_tree_sample(&mut entries, pivot_diag, trial as u64, &mut out);
            for &(lo, hi, w) in &out {
                *pair_total.entry((lo, hi)).or_insert(0.0) += w;
            }
        }

        let weights = [1.0, 2.0, 3.0, 4.0];
        for (&(lo, hi), &total) in &pair_total {
            let avg_per_trial = total / n_trials as f64;
            let exact = weights[lo as usize] * weights[hi as usize] / pivot_diag;
            let ratio = avg_per_trial / exact;
            assert!(
                (0.3..=3.0).contains(&ratio),
                "pair ({lo},{hi}): avg_per_trial={avg_per_trial:.4}, exact={exact:.4}, ratio={ratio:.2}"
            );
        }
    }

    #[test]
    fn ac2_respects_split_merge_edge_budget() {
        let mut entries: Vec<(u32, f64)> = vec![(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let mut out = Vec::new();

        clique_tree_sample_multi(&mut entries, 2, 42, &mut out);

        assert!(out.len() <= 8, "got {} edges, expected <= 8", out.len());
        for &(lo, hi, w) in &out {
            assert!(lo < hi, "edge ({lo}, {hi}) not ordered");
            assert!(w > 0.0, "edge ({lo}, {hi}) has non-positive weight {w}");
        }
    }
}
