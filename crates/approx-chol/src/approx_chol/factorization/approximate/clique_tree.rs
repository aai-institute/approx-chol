use super::ordering::DegreeDeltas;
use super::star::{Star, StarEntry};
use crate::graph::{AdjListGraph, EdgeCount, Multi, Single, SplitFactor};
use crate::sampling::CdfSampler;
use crate::types::{count_as_scalar, float_total_cmp, near_zero, Real};

/// One sampled column of the approximate Cholesky factor (Algorithm 5, GKS 2023),
/// reused across elimination steps and cleared at the start of each sampling pass.
pub(super) struct SampledColumn<T: Real> {
    /// Diagonal value of the factor column: `L[v,v]`.
    pub diagonal: T,
    /// Neighbor indices in the column's non-zero pattern, and each one's
    /// fractional weight `L[neighbor, v] / L[v, v]`. Private and only appended
    /// through [`Self::push_neighbor`], so the two can never disagree.
    neighbors: Vec<u32>,
    fractions: Vec<T>,
    /// Fill edges `(u, w, weight)` to insert into the graph after elimination.
    fill_edges: Vec<(u32, u32, T)>,
}

impl<T: Real> SampledColumn<T> {
    pub(super) fn new() -> Self {
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

    #[inline]
    fn push_neighbor(&mut self, neighbor: u32, fraction: T) {
        self.neighbors.push(neighbor);
        self.fractions.push(fraction);
    }

    /// The column's non-zero pattern, always as the pair it is built as.
    pub(super) fn pattern(&self) -> (&[u32], &[T]) {
        (&self.neighbors, &self.fractions)
    }

    /// Initialize sampling, or write the fallback column and return `None`.
    ///
    /// `Some` only when the elimination loop should run. Otherwise the column is a
    /// uniform split with no fill — trivial for one entry or none, degenerate for a
    /// non-positive/non-finite total.
    fn begin_sampling<'a, C: Copy>(
        &mut self,
        entries: &'a [StarEntry<T, C>],
        pivot_diag: T,
    ) -> Option<Sampling<'a, T, C>> {
        self.clear();
        let fraction = match entries.split_last() {
            Some((&last, rest)) if !rest.is_empty() => {
                // Fold in entry (sorted) order: the sum order affects the factor
                // bit-for-bit under a fixed seed.
                let total_weight = entries
                    .iter()
                    .fold(T::zero(), |acc, entry| acc + entry.weight);
                // A non-positive/non-finite total has no valid fraction
                // (`f = w·scale/total` divides through zero or NaN).
                if total_weight.is_finite() && total_weight > near_zero::<T>() {
                    return Some(Sampling {
                        rest,
                        last,
                        total_weight,
                    });
                }
                T::one() / count_as_scalar::<T, _>(entries.len())
            }
            _ => T::one(),
        };

        self.diagonal = pivot_diag;
        for entry in entries {
            self.push_neighbor(entry.neighbor, fraction);
        }
        None
    }

    /// Finalize sampling with the last star neighbor (always fraction 1).
    fn finalize_sampling<C>(&mut self, last: StarEntry<T, C>, elim: &StarElimination<T>) {
        self.push_neighbor(last.neighbor, T::one());
        self.diagonal = elim.diagonal(last.weight);
    }

    /// Apply fill-in edges to the graph and update diagonal values, recording
    /// each endpoint's +1 degree change in `deltas` (the caller flushes one
    /// priority-queue move per affected neighbor, rather than one per fill edge).
    pub(super) fn apply_fill_in_delta<C: EdgeCount>(
        &self,
        graph: &mut AdjListGraph<C, T>,
        diag: &mut [T],
        deltas: &mut DegreeDeltas,
    ) {
        for &(u, w, weight) in &self.fill_edges {
            graph.add_fill_edge(u, w, weight);
            diag[u as usize] = diag[u as usize] + weight;
            diag[w as usize] = diag[w as usize] + weight;
            deltas.increase(u, 1);
            deltas.increase(w, 1);
        }
    }

    /// Sample fill edges between `neighbor` and random star neighbors past `tail`.
    fn sample_fill_edges(
        &mut self,
        neighbor: u32,
        n_samples: u32,
        fill_weight: T,
        draws: &mut CdfSampler<T>,
        tail: usize,
    ) {
        if n_samples == 0 || fill_weight <= near_zero::<T>() {
            return;
        }
        for _ in 0..n_samples {
            if let Some(k) = draws.sample_after(tail) {
                if neighbor != k {
                    self.fill_edges.push((neighbor, k, fill_weight));
                }
            }
        }
    }

    /// Append the sampled fill edges to `out` as `(lo, hi, weight)`, `lo < hi`.
    fn extend_ordered_fill_edges(&self, out: &mut Vec<(u32, u32, T)>) {
        out.extend(
            self.fill_edges
                .iter()
                .map(|&(u, v, w)| if u < v { (u, v, w) } else { (v, u, w) }),
        );
    }
}

/// A star the elimination loop can run on, split where the sampler splits it: the
/// last neighbor takes what the others leave, so it never enters the loop, and every
/// caller reads its count off `rest` rather than deriving it from an index.
struct Sampling<'a, T, C> {
    rest: &'a [StarEntry<T, C>],
    last: StarEntry<T, C>,
    total_weight: T,
}

/// Running state for sequential edge elimination on a star graph (GKS 2023,
/// Algorithms 5 & 6): neighbors are processed along a clique-tree path, each one
/// taking fraction `f_i = w_i * scale / capacity` of what earlier neighbors left.
struct StarElimination<T = f64> {
    /// Product of `(1 - f_k)` over already-processed neighbors: the share of the
    /// original edge weight that survives to this one.
    scale: T,
    /// Remaining weight budget, shrunk by `(1 - f_i)^2` per step.
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

/// Clique-tree sampling of one column (GKS 2023, Algorithms 5 and 6): each neighbor
/// takes its fraction of what earlier ones left, and samples one fill edge per copy.
///
/// Multiplicity enters in exactly two places — how many fill edges a neighbor draws,
/// and the share of the clique weight one of them carries — which is why AC and AC2
/// are this one function. AC2's per-copy fill weight is `w·remaining/(k·total)`,
/// and `remaining` is `total·scale`, the quantity [`StarElimination`] already
/// carries; spelling it that way is what leaves one formula here instead of two.
///
/// Capacity comes from the live column sum, not from `pivot_diag`: that keeps
/// `f ∈ [0, 1]` by construction, whereas a caller-maintained `diag[v]` can drift
/// below the column sum under stochastic elimination. `pivot_diag` only seeds the
/// degenerate (`n <= 1`) column.
pub(super) fn sample_column<T: Real, C: EdgeCount>(
    star: &Star<T, C>,
    pivot_diag: T,
    sampler: &mut CdfSampler<T>,
    column: &mut SampledColumn<T>,
) {
    let entries = star.entries();
    let Some(Sampling {
        rest,
        last,
        total_weight,
    }) = column.begin_sampling(entries, pivot_diag)
    else {
        return;
    };

    sampler.prepare(entries.iter().map(|entry| (entry.neighbor, entry.weight)));
    let mut elim = StarElimination::new(total_weight);

    for (i, entry) in rest.iter().enumerate() {
        let f = elim.fraction(entry.weight);
        let fill_wt = entry.copies.per_copy(f * (T::one() - f) * elim.capacity);
        column.push_neighbor(entry.neighbor, f);
        column.sample_fill_edges(entry.neighbor, entry.copies.get(), fill_wt, sampler, i + 1);
        elim.advance(f);
    }

    column.finalize_sampling(last, &elim);
}

/// Sample fill edges approximating the Schur complement clique of a star.
///
/// Given an eliminated vertex with weighted neighbors `entries`, walks neighbors
/// sorted by ascending weight and samples `split_merge` fill edges per neighbor to
/// random later neighbors (clique-tree, Algorithms 5 and 6 in Gao-Kyng-Spielman
/// 2023). `split_merge` is [`Config::split_merge`](crate::Config::split_merge): it
/// takes the same values and means the same thing, so AC is `None`, `Some(0)` and
/// `Some(1)` here too, and AC2 with multiplicity `k` is `Some(k)`.
///
/// Elimination capacity is `Σ |entries.weights|` (the live column's
/// off-diagonal sum), matching Laplacians.jl. For Laplacian inputs — where
/// the pivot's matrix diagonal equals this sum — each fill edge is unbiased:
/// `E[w(i,j)] = a_i * a_j / Σ a_k`.
///
/// Produces at most `k * (n-1)` fill edges (`k` copies of a spanning tree on the n
/// neighbors). `entries` is sorted in place. Fill edges are appended to `out`.
pub fn clique_tree_sample<T>(
    entries: &mut [(u32, T)],
    split_merge: Option<u32>,
    seed: u64,
    out: &mut Vec<(u32, u32, T)>,
) where
    T: num_traits::Float + Send + Sync + 'static,
{
    // Generic over the copies, so AC and AC2 differ only in the star they build.
    // The zero `pivot_diag` only seeds the column diagonal this discards.
    fn sample<T: Real, C: EdgeCount>(
        entries: &[(u32, T)],
        copies: C,
        seed: u64,
        out: &mut Vec<(u32, u32, T)>,
    ) {
        let mut sampler = CdfSampler::<T>::new(seed);
        let mut column = SampledColumn::new();
        sample_column(
            &Star::uniform(entries, copies),
            T::zero(),
            &mut sampler,
            &mut column,
        );
        column.extend_ordered_fill_edges(out);
    }

    entries.sort_unstable_by(|a, b| float_total_cmp(&a.1, &b.1));
    match split_merge.and_then(SplitFactor::new) {
        None => sample(entries, Single, seed, out),
        Some(k) => sample(entries, Multi::from(k), seed, out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both samplers span the same five neighbors; AC's budget is the spanning tree
    /// on them, AC2's is `k` copies of it.
    #[test]
    fn a_sampled_star_stays_within_its_edge_budget() {
        let star: [(u32, f64); 5] = [(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let mut ac = Vec::new();
        clique_tree_sample(&mut star.clone(), None, 42, &mut ac);
        let mut ac2 = Vec::new();
        clique_tree_sample(&mut star.clone(), Some(2), 42, &mut ac2);

        for (label, out, budget) in [("AC", ac, 4), ("AC2 k=2", ac2, 8)] {
            assert!(
                out.len() <= budget,
                "{label}: got {} edges, expected <= {budget}",
                out.len()
            );
            assert_finite_positive_ordered(&out);
        }
    }

    /// The raw sampler and `Config` have to agree about what a split below two
    /// means, or `Some(0)` selects AC through one public entry point and nothing
    /// through the other.
    #[test]
    fn a_split_below_two_samples_the_same_edges_as_ac() {
        let star: [(u32, f64); 5] = [(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let mut expected = Vec::new();
        clique_tree_sample(&mut star.clone(), None, 42, &mut expected);

        for split_merge in [Some(0), Some(1)] {
            let mut out = Vec::new();
            clique_tree_sample(&mut star.clone(), split_merge, 42, &mut out);
            assert_eq!(out, expected, "split_merge {split_merge:?} is not AC");
        }
    }

    #[test]
    fn empty_and_single() {
        let mut out = Vec::new();

        clique_tree_sample(&mut [], None, 0, &mut out);
        assert!(out.is_empty());

        let mut entries = vec![(0u32, 5.0)];
        clique_tree_sample(&mut entries, None, 0, &mut out);
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
            clique_tree_sample(&mut entries, None, trial as u64, &mut out);
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
        for mut entries in cases {
            let mut out = Vec::new();
            // AC only sorts `entries`, so AC2 sees the same (degenerate) star.
            clique_tree_sample(&mut entries, None, 7, &mut out);
            assert!(out.is_empty(), "AC emitted fill for a degenerate star");
            clique_tree_sample(&mut entries, Some(3), 7, &mut out);
            assert!(out.is_empty(), "AC2 emitted fill for a degenerate star");
        }
    }
}
