//! Standalone star-clique sampling for approximate Schur complement construction.

use core::cmp::Ordering;

use crate::approx_chol::sampled_column::StarElimination;
use crate::sampling::{near_zero, CdfSampler, WeightedSampler};

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
pub fn sample_star_clique<T>(
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

    // Sort by ascending weight (same order as AC factorization)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_on_five_neighbors() {
        let mut entries: Vec<(u32, f64)> = vec![(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let pivot_diag: f64 = entries.iter().map(|(_, w)| w).sum();
        let mut out = Vec::new();

        sample_star_clique(&mut entries, pivot_diag, 42, &mut out);

        // At most n-1 = 4 edges
        assert!(out.len() <= 4, "got {} edges, expected <= 4", out.len());
        // All weights positive
        for &(lo, hi, w) in &out {
            assert!(lo < hi, "edge ({lo}, {hi}) not ordered");
            assert!(w > 0.0, "edge ({lo}, {hi}) has non-positive weight {w}");
        }
    }

    #[test]
    fn empty_and_single() {
        let mut out = Vec::new();

        // Empty
        sample_star_clique(&mut [], 1.0, 0, &mut out);
        assert!(out.is_empty());

        // Single neighbor
        let mut entries = vec![(0u32, 5.0)];
        sample_star_clique(&mut entries, 5.0, 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn statistical_unbiasedness() {
        // 4 neighbors with known weights; run many trials and check that
        // E[total edge weight per trial] ≈ a_i * a_j / pivot_diag.
        // The expectation includes trials where the pair doesn't appear.
        let base_entries: Vec<(u32, f64)> = vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)];
        let pivot_diag: f64 = base_entries.iter().map(|(_, w)| w).sum(); // 10.0

        let n_trials = 50_000;
        // Accumulate total weight per edge pair across all trials
        let mut pair_total = std::collections::HashMap::<(u32, u32), f64>::new();

        for trial in 0..n_trials {
            let mut entries = base_entries.clone();
            let mut out = Vec::new();
            sample_star_clique(&mut entries, pivot_diag, trial as u64, &mut out);
            for &(lo, hi, w) in &out {
                *pair_total.entry((lo, hi)).or_insert(0.0) += w;
            }
        }

        // Check that for each pair, total / n_trials ≈ a_i * a_j / pivot_diag
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
}
