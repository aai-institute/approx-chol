use super::ordering::DegreeDeltas;
use super::star::{Star, StarEntry};
use crate::graph::{AdjListGraph, EdgeCount, Multi, Single, SplitFactor};
use crate::sampling::CdfSampler;
use crate::types::{count_as_scalar, Real};

/// One sampled column of the factor (Algorithm 5, GKS 2023), reused across steps.
pub(super) struct SampledColumn<T: Real> {
    /// Only appended through [`Self::push_share`], so the two can never disagree.
    neighbors: Vec<u32>,
    coefficients: Vec<T>,
    /// Named rather than weighted: only the sequence storing the shares can subtract them.
    remainder: Option<u32>,
    fill_edges: Vec<(u32, u32, T)>,
}

/// A share cannot be handed on with nothing named to absorb the remainder.
pub(super) struct ColumnShares<'a, T> {
    pub(super) neighbors: &'a [u32],
    pub(super) coefficients: &'a [T],
    pub(super) remainder: u32,
}

impl<T: Real> SampledColumn<T> {
    pub(super) fn new() -> Self {
        Self {
            neighbors: Vec::new(),
            coefficients: Vec::new(),
            remainder: None,
            fill_edges: Vec::new(),
        }
    }

    /// A column that hands on nothing, which is what an empty star leaves.
    fn reset(&mut self) {
        self.neighbors.clear();
        self.coefficients.clear();
        self.remainder = None;
        self.fill_edges.clear();
    }

    /// `share` is of the *original* pivot, so the solve never rebuilds the running product.
    #[inline]
    fn push_share(&mut self, neighbor: u32, share: T) {
        self.neighbors.push(neighbor);
        self.coefficients.push(share);
    }

    pub(super) fn shares(&self) -> Option<ColumnShares<'_, T>> {
        Some(ColumnShares {
            neighbors: &self.neighbors,
            coefficients: &self.coefficients,
            remainder: self.remainder?,
        })
    }

    /// Uniform split, no fill; only a standalone `CliqueTreeSampler` caller's weights reach here.
    fn push_uniform_shares<C>(&mut self, rest: &[StarEntry<T, C>]) {
        let fraction = T::one() / count_as_scalar::<T, _>(rest.len() + 1);
        let mut retained = T::one();
        for entry in rest {
            self.push_share(entry.neighbor, fraction * retained);
            retained = retained * (T::one() - fraction);
        }
    }

    /// Degree changes go to `deltas`: one queue move per affected neighbor, not per fill edge.
    pub(super) fn apply_fill_in_delta<C: EdgeCount>(
        &self,
        graph: &mut AdjListGraph<C, T>,
        deltas: &mut DegreeDeltas,
    ) {
        for &(u, w, weight) in &self.fill_edges {
            graph.add_fill_edge(u, w, weight);
            deltas.increase(u, 1);
            deltas.increase(w, 1);
        }
    }

    /// Generic in the count so `Single`'s literal 1 folds the copy loop away.
    fn sample_fill_edges<C: EdgeCount>(
        &mut self,
        neighbor: u32,
        copies: C,
        fill_weight: T,
        draws: &mut CdfSampler<T>,
        tail: usize,
    ) {
        if fill_weight <= T::zero() {
            return;
        }
        // The suffix does not move across the copies, so the sampler resolves it once.
        let fill_edges = &mut self.fill_edges;
        draws.sample_batch(tail, copies.get(), |k| {
            if neighbor != k {
                fill_edges.push((neighbor, k, fill_weight));
            }
        });
    }

    fn extend_ordered_fill_edges(&self, out: &mut Vec<(u32, u32, T)>) {
        out.extend(
            self.fill_edges
                .iter()
                .map(|&(u, v, w)| if u < v { (u, v, w) } else { (v, u, w) }),
        );
    }
}

/// Neighbors walk a clique-tree path, each taking `f_i = w_i * scale / capacity` of what's left.
struct StarElimination<T = f64> {
    /// Product of `(1 - f_k)` over the neighbors already taken.
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
        debug_assert!(self.capacity > T::zero());
        w * self.scale / self.capacity
    }

    /// What this neighbor takes of the original pivot, the path advanced past it.
    #[inline(always)]
    fn take(&mut self, f: T) -> T {
        let share = f * self.scale;
        let retain = T::one() - f;
        self.scale = self.scale * retain;
        self.capacity = self.capacity * retain * retain;
        share
    }
}

/// Returns the pivot: the arm sees only Laplacians, whose diagonal is the star's live sum.
pub(super) fn sample_column<T: Real, C: EdgeCount>(
    star: &Star<T, C>,
    sampler: &mut CdfSampler<T>,
    column: &mut SampledColumn<T>,
) -> T {
    let entries = star.entries();
    column.reset();
    // The last neighbor takes the remainder, so no caller derives its count from an index.
    let Some((last, rest)) = entries.split_last() else {
        return T::zero();
    };
    column.remainder = Some(last.neighbor);
    if rest.is_empty() {
        return last.weight;
    }

    // Fold in sorted order: the sum order changes the factor bit-for-bit under a fixed seed.
    let total_weight = entries
        .iter()
        .fold(T::zero(), |acc, entry| acc + entry.weight);
    // Sorted ascending puts every `f` in `[0, 1]`; a floor above zero would judge scale instead.
    if !(total_weight.is_finite() && total_weight > T::zero()) {
        column.push_uniform_shares(rest);
        return total_weight;
    }

    sampler.prepare(entries.iter().map(|entry| (entry.neighbor, entry.weight)));
    let mut elim = StarElimination::new(total_weight);
    for (i, entry) in rest.iter().enumerate() {
        let f = elim.fraction(entry.weight);
        let fill_wt = entry.copies.per_copy(f * (T::one() - f) * elim.capacity);
        column.push_share(entry.neighbor, elim.take(f));
        column.sample_fill_edges(entry.neighbor, entry.copies, fill_wt, sampler, i + 1);
    }

    last.weight * elim.scale
}

/// Fixed for the sampler's life, as [`Config::split_merge`](crate::Config::split_merge) is.
enum StarScratch<T: Real> {
    Single(Star<T, Single>),
    Multi(Star<T, Multi>, Multi),
}

/// Standalone callers promise dedup; a repeat ties with itself and under-weights the fill.
fn neighbors_are_unique<T>(entries: &[(u32, T)]) -> bool {
    let mut seen: Vec<u32> = entries.iter().map(|&(neighbor, _)| neighbor).collect();
    seen.sort_unstable();
    seen.windows(2).all(|pair| pair[0] != pair[1])
}

/// Samples a star's clique tree outside a factorization; samplers on one seed agree per star index.
pub struct CliqueTreeSampler<T: num_traits::Float + Send + Sync + 'static = f64> {
    star: StarScratch<T>,
    draws: CdfSampler<T>,
    column: SampledColumn<T>,
}

impl<T: num_traits::Float + Send + Sync + 'static> CliqueTreeSampler<T> {
    /// `split_merge` as in [`Config`](crate::Config); each star's stream derives from `seed`.
    pub fn new(seed: u64, split_merge: Option<u32>) -> Self {
        Self {
            star: match split_merge.and_then(SplitFactor::new) {
                None => StarScratch::Single(Star::new()),
                Some(k) => StarScratch::Multi(Star::new(), Multi::from(k)),
            },
            draws: CdfSampler::new(seed),
            column: SampledColumn::new(),
        }
    }

    /// Appends ≤ `k * (n-1)` fill edges for deduped `entries`; `index` picks the stream.
    pub fn sample(&mut self, index: u64, entries: &[(u32, T)], out: &mut Vec<(u32, u32, T)>) {
        debug_assert!(
            neighbors_are_unique(entries),
            "a star needs one entry per neighbor"
        );
        self.draws.restart(index);
        match &mut self.star {
            StarScratch::Single(star) => {
                star.refill_uniform(entries, Single);
                sample_column(star, &mut self.draws, &mut self.column);
            }
            StarScratch::Multi(star, copies) => {
                star.refill_uniform(entries, *copies);
                sample_column(star, &mut self.draws, &mut self.column);
            }
        }
        self.column.extend_ordered_fill_edges(out);
    }
}

/// Reports `copies`, not `split_merge`, so `Some(1)` (which selects AC) never prints as AC2.
impl<T: num_traits::Float + Send + Sync + 'static> core::fmt::Debug for CliqueTreeSampler<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let copies = match &self.star {
            StarScratch::Single(_) => 1,
            StarScratch::Multi(_, copies) => copies.get(),
        };
        f.debug_struct("CliqueTreeSampler")
            .field("seed", &self.draws.seed())
            .field("copies", &copies)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AC's budget is the spanning tree on the five neighbors, AC2's is `k` copies.
    #[test]
    fn a_sampled_star_stays_within_its_edge_budget() {
        let star: [(u32, f64); 5] = [(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let mut ac = Vec::new();
        CliqueTreeSampler::new(42, None).sample(0, &star, &mut ac);
        let mut ac2 = Vec::new();
        CliqueTreeSampler::new(42, Some(2)).sample(0, &star, &mut ac2);

        for (label, out, budget) in [("AC", ac, 4), ("AC2 k=2", ac2, 8)] {
            assert!(
                out.len() <= budget,
                "{label}: got {} edges, expected <= {budget}",
                out.len()
            );
            assert_finite_positive_ordered(&out);
        }
    }

    /// Else `Some(0)` would select AC through one public entry point and nothing through the other.
    #[test]
    fn a_split_below_two_samples_the_same_edges_as_ac() {
        let star: [(u32, f64); 5] = [(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let mut expected = Vec::new();
        CliqueTreeSampler::new(42, None).sample(0, &star, &mut expected);

        for split_merge in [Some(0), Some(1)] {
            let mut out = Vec::new();
            CliqueTreeSampler::new(42, split_merge).sample(0, &star, &mut out);
            assert_eq!(out, expected, "split_merge {split_merge:?} is not AC");
        }
    }

    #[test]
    fn empty_and_single() {
        let mut out = Vec::new();

        CliqueTreeSampler::new(0, None).sample(0, &[], &mut out);
        assert!(out.is_empty());

        CliqueTreeSampler::new(0, None).sample(0, &[(0u32, 5.0)], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn statistical_unbiasedness() {
        let base_entries: Vec<(u32, f64)> = vec![(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)];
        let pivot_diag: f64 = base_entries.iter().map(|(_, w)| w).sum();

        let n_trials = 50_000;
        let mut pair_total = std::collections::HashMap::<(u32, u32), f64>::new();
        let mut sampler = CliqueTreeSampler::new(0, None);

        for trial in 0..n_trials {
            let mut out = Vec::new();
            sampler.sample(trial as u64, &base_entries, &mut out);
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

    /// Stale scratch from an earlier star would make the same star answer differently.
    #[test]
    fn a_reused_sampler_answers_as_a_fresh_one() {
        let stars: [Vec<(u32, f64)>; 3] = [
            vec![(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)],
            vec![(7, 1.0), (8, 1.0)],
            vec![(2, 9.0), (5, 0.5), (6, 4.0), (9, 2.5)],
        ];

        for split_merge in [None, Some(3)] {
            let mut reused = CliqueTreeSampler::new(11, split_merge);
            for (index, star) in stars.iter().enumerate() {
                let mut from_reused = Vec::new();
                reused.sample(index as u64, star, &mut from_reused);

                let mut from_fresh = Vec::new();
                CliqueTreeSampler::new(11, split_merge).sample(index as u64, star, &mut from_fresh);

                assert_eq!(
                    from_reused, from_fresh,
                    "star {index} at split_merge {split_merge:?} depends on sampler history"
                );
            }
        }
    }

    /// A repeated neighbor ties with itself on both keys, so the tie-break cannot order it.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "one entry per neighbor")]
    fn a_repeated_neighbor_is_caught_in_debug() {
        let mut out = Vec::new();
        CliqueTreeSampler::new(0, None).sample(0, &[(3, 1.0), (5, 4.0), (3, 2.0)], &mut out);
    }

    fn assert_finite_positive_ordered(out: &[(u32, u32, f64)]) {
        for &(lo, hi, w) in out {
            assert!(lo < hi, "edge ({lo}, {hi}) not ordered");
            assert!(
                w.is_finite() && w > 0.0,
                "edge ({lo}, {hi}) has non-finite/non-positive weight {w}"
            );
        }
    }

    #[test]
    fn degenerate_total_weight_is_consistent_and_clean() {
        // Zero/negative/NaN/inf totals: neither path may panic or emit fill.
        let cases = [
            [(0, 0.0), (1, 0.0), (2, 0.0)],
            [(0, -1.0), (1, -2.0), (2, -3.0)],
            [(0, f64::NAN), (1, 1.0), (2, 2.0)],
            [(0, f64::INFINITY), (1, 1.0), (2, 2.0)],
        ];
        for entries in cases {
            let mut out = Vec::new();
            CliqueTreeSampler::new(7, None).sample(0, &entries, &mut out);
            assert!(out.is_empty(), "AC emitted fill for a degenerate star");
            CliqueTreeSampler::new(7, Some(3)).sample(0, &entries, &mut out);
            assert!(out.is_empty(), "AC2 emitted fill for a degenerate star");
        }
    }
}
