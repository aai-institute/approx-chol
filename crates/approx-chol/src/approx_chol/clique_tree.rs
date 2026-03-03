use crate::sampling::WeightedSampler;
use crate::types::Real;
use num_traits::NumCast;

use super::sampled_column::{SampledColumn, StarElimination};

/// Clique-tree sampling for AC stars (single sample per neighbor).
pub(crate) fn clique_tree_sample<T: Real, S: WeightedSampler<T>>(
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
pub(crate) fn clique_tree_sample_multi<T: Real, S: WeightedSampler<T>>(
    entries: &[(u32, T)],
    counts: &[u32],
    pivot_diag: T,
    sampler: &mut S,
    column: &mut SampledColumn<T>,
) {
    debug_assert_eq!(entries.len(), counts.len());
    let Some(n) = column.begin_sampling(entries, pivot_diag) else {
        return;
    };

    let total_weight = entries.iter().fold(T::zero(), |a, e| a + e.1);
    if total_weight <= T::near_zero() {
        column.diagonal = pivot_diag;
        for &(j, _) in entries.iter() {
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
