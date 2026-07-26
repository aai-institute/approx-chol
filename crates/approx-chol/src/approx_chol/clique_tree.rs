use crate::graph::{AdjListGraph, EdgeCount};
use crate::ordering::DegreeDeltas;
use crate::sampling::CdfSampler;
use crate::types::{count_as_scalar, float_total_cmp, Real};

/// One sampled column of the approximate Cholesky factor (Algorithm 5, GKS 2023),
/// reused across elimination steps and cleared at the start of each sampling pass.
pub(crate) struct SampledColumn<T: Real> {
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

    #[inline]
    fn push_neighbor(&mut self, neighbor: u32, fraction: T) {
        self.neighbors.push(neighbor);
        self.fractions.push(fraction);
    }

    pub(crate) fn neighbors(&self) -> &[u32] {
        &self.neighbors
    }

    pub(crate) fn fractions(&self) -> &[T] {
        &self.fractions
    }

    /// Initialize sampling, or write the fallback column and return `None`.
    ///
    /// Returns `Some((n, total_weight))` only when the elimination loop should
    /// run. Otherwise the column is a uniform split with no fill — trivial for
    /// `n <= 1`, degenerate for a non-positive/non-finite total.
    fn begin_sampling(&mut self, entries: &[(u32, T)], pivot_diag: T) -> Option<(usize, T)> {
        self.clear();
        let n = entries.len();
        let fraction = if n < 2 {
            T::one()
        } else {
            // Fold in entry (sorted) order: the sum order affects the factor
            // bit-for-bit under a fixed seed.
            let total_weight = entries.iter().fold(T::zero(), |acc, &(_, w)| acc + w);
            // A non-positive/non-finite total has no valid fraction
            // (`f = w·scale/total` divides through zero or NaN).
            if total_weight.is_finite() && total_weight > T::near_zero() {
                return Some((n, total_weight));
            }
            T::one() / count_as_scalar::<T, _>(n)
        };

        self.diagonal = pivot_diag;
        for &(neighbor, _) in entries {
            self.push_neighbor(neighbor, fraction);
        }
        None
    }

    /// Finalize sampling with the last star neighbor (always fraction 1).
    fn finalize_sampling(&mut self, last: (u32, T), elim: &StarElimination<T>) {
        self.push_neighbor(last.0, T::one());
        self.diagonal = elim.diagonal(last.1);
    }

    /// Apply fill-in edges to the graph and update diagonal values, recording
    /// each endpoint's +1 degree change in `deltas` (the caller flushes one
    /// priority-queue move per affected neighbor, rather than one per fill edge).
    pub(crate) fn apply_fill_in_delta<C: EdgeCount>(
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

    /// Sample fill edges between `neighbor` and random neighbors from `entries[tail..]`.
    fn sample_fill_edges(
        &mut self,
        neighbor: u32,
        n_samples: u32,
        fill_weight: T,
        sampler: &mut CdfSampler<T>,
        entries: &[(u32, T)],
        tail: usize,
    ) {
        if n_samples == 0 || fill_weight <= T::near_zero() {
            return;
        }
        if tail >= entries.len() {
            return;
        }
        for _ in 0..n_samples {
            if let Some(koff) = sampler.sample_suffix(tail) {
                let k = entries[koff].0;
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

/// A deduped AC2 star neighborhood: one `(neighbor, weight)` entry and one
/// multiplicity per unique neighbor.
///
/// The two arrays are only ever pushed, cleared and permuted together, so
/// nothing downstream has to check that they still agree about length or about
/// which multiplicity belongs to which neighbor.
pub(crate) struct MultiStar<T: Real> {
    entries: Vec<(u32, T)>,
    counts: Vec<u32>,
    sort_scratch: Vec<SortEntry<T>>,
}

/// Packed staging record for [`MultiStar::sort_by_avg_weight`]: permuting two
/// arrays needs one sortable element holding both halves.
#[derive(Clone, Copy)]
struct SortEntry<T: Real> {
    neighbor: u32,
    weight: T,
    count: u32,
    /// Precomputed `weight / count` sort key. Cross-multiplying in the
    /// comparator instead can break transitivity under floating-point rounding.
    avg_weight: T,
}

impl<T: Real> MultiStar<T> {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            counts: Vec::new(),
            sort_scratch: Vec::new(),
        }
    }

    /// Every neighbor at the same multiplicity — the shape
    /// [`clique_tree_sample_multi`] samples.
    fn uniform(entries: &[(u32, T)], count: u32) -> Self {
        let mut star = Self::new();
        for &(neighbor, weight) in entries {
            star.push(neighbor, weight, count);
        }
        star
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.counts.clear();
    }

    pub(super) fn push(&mut self, neighbor: u32, weight: T, count: u32) {
        self.entries.push((neighbor, weight));
        self.counts.push(count);
    }

    /// Push, or fold into the previous entry when it repeats the same neighbor.
    /// Over `neighbor`-sorted input that coalesces duplicates in one pass.
    pub(super) fn push_or_merge(&mut self, neighbor: u32, weight: T, count: u32) {
        if self.entries.last().map(|last| last.0) == Some(neighbor) {
            let last = self.entries.len() - 1;
            self.entries[last].1 = self.entries[last].1 + weight;
            self.counts[last] = self.counts[last].saturating_add(count);
        } else {
            self.push(neighbor, weight, count);
        }
    }

    pub(super) fn entries(&self) -> &[(u32, T)] {
        &self.entries
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (u32, T, u32)> + '_ {
        self.entries
            .iter()
            .zip(&self.counts)
            .map(|(&(neighbor, weight), &count)| (neighbor, weight, count))
    }

    /// Cap every multiplicity at `limit`, leaving weights untouched and
    /// reporting each discard as `(neighbor, discarded)`.
    pub(super) fn apply_merge_limit(&mut self, limit: u32, merged: &mut Vec<(u32, u32)>) {
        for (&(neighbor, _), count) in self.entries.iter().zip(self.counts.iter_mut()) {
            if *count > limit {
                merged.push((neighbor, *count - limit));
                *count = limit;
            }
        }
    }

    /// Sort by average weight ascending, breaking ties by neighbor index.
    pub(super) fn sort_by_avg_weight(&mut self) {
        self.sort_scratch.clear();
        self.sort_scratch.reserve(self.entries.len());
        for (&(neighbor, weight), &count) in self.entries.iter().zip(&self.counts) {
            self.sort_scratch.push(SortEntry {
                neighbor,
                weight,
                count,
                avg_weight: weight / count_as_scalar::<T, _>(count),
            });
        }
        self.sort_scratch.sort_unstable_by(|a, b| {
            float_total_cmp(&a.avg_weight, &b.avg_weight).then_with(|| a.neighbor.cmp(&b.neighbor))
        });
        for (i, item) in self.sort_scratch.iter().enumerate() {
            self.entries[i] = (item.neighbor, item.weight);
            self.counts[i] = item.count;
        }
    }
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
///
/// Capacity comes from the live column sum, not from `pivot_diag`: that keeps
/// `f ∈ [0, 1]` by construction, whereas a caller-maintained `diag[v]` can drift
/// below the column sum under stochastic elimination. `pivot_diag` only seeds the
/// degenerate (`n <= 1`) column.
pub(crate) fn clique_tree_sample_column<T: Real>(
    entries: &[(u32, T)],
    pivot_diag: T,
    sampler: &mut CdfSampler<T>,
    column: &mut SampledColumn<T>,
) {
    let Some((n, total_weight)) = column.begin_sampling(entries, pivot_diag) else {
        return;
    };

    sampler.prepare(entries);
    let mut elim = StarElimination::new(total_weight);

    for (i, &(j, w)) in entries[..n - 1].iter().enumerate() {
        let f = elim.fraction(w);
        let fill_wt = f * (T::one() - f) * elim.capacity();
        column.push_neighbor(j, f);
        column.sample_fill_edges(j, 1, fill_wt, sampler, entries, i + 1);
        elim.advance(f);
    }

    column.finalize_sampling(entries[n - 1], &elim);
}

/// Clique-tree sampling for AC2 stars (multi-sample per neighbor).
pub(crate) fn clique_tree_sample_column_multi<T: Real>(
    star: &MultiStar<T>,
    pivot_diag: T,
    sampler: &mut CdfSampler<T>,
    column: &mut SampledColumn<T>,
) {
    let entries = star.entries();
    let Some((n, total_weight)) = column.begin_sampling(entries, pivot_diag) else {
        return;
    };

    sampler.prepare(entries);
    let mut remaining = total_weight;
    let mut elim = StarElimination::new(total_weight);

    for (i, (j, w, count)) in star.iter().take(n - 1).enumerate() {
        remaining = remaining - w;
        let f = elim.fraction(w);
        let fill_wt = w * remaining / (count_as_scalar::<T, _>(count) * total_weight);
        column.push_neighbor(j, f);
        column.sample_fill_edges(j, count, fill_wt, sampler, entries, i + 1);
        elim.advance(f);
    }

    column.finalize_sampling(entries[n - 1], &elim);
}

/// Sample fill edges approximating the Schur complement clique of a star.
///
/// Given an eliminated vertex with weighted neighbors `entries`, walks
/// neighbors sorted by ascending weight and samples one fill edge per
/// neighbor to a random later neighbor (AC clique-tree, Algorithm 5 in
/// Gao-Kyng-Spielman 2023).
///
/// Elimination capacity is `Σ |entries.weights|` (the live column's
/// off-diagonal sum), matching Laplacians.jl. For Laplacian inputs — where
/// the pivot's matrix diagonal equals this sum — each fill edge is unbiased:
/// `E[w(i,j)] = a_i * a_j / Σ a_k`.
///
/// Produces at most `n-1` fill edges (a spanning tree on the n neighbors).
/// `entries` is sorted in place. Fill edges are appended to `out`.
pub fn clique_tree_sample<T>(entries: &mut [(u32, T)], seed: u64, out: &mut Vec<(u32, u32, T)>)
where
    T: num_traits::Float + Send + Sync + 'static,
{
    entries.sort_unstable_by(|a, b| float_total_cmp(&a.1, &b.1));
    let mut sampler = CdfSampler::<T>::new(seed);
    let mut column = SampledColumn::new();
    // pivot_diag only seeds the discarded column diagonal; capacity is derived
    // from the entry weights, so the value passed here is irrelevant.
    clique_tree_sample_column(entries, T::zero(), &mut sampler, &mut column);
    column.extend_ordered_fill_edges(out);
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
    if split_merge == 0 {
        return;
    }
    entries.sort_unstable_by(|a, b| float_total_cmp(&a.1, &b.1));
    let star = MultiStar::uniform(entries, split_merge);
    let mut sampler = CdfSampler::<T>::new(seed);
    let mut column = SampledColumn::new();
    clique_tree_sample_column_multi(&star, T::zero(), &mut sampler, &mut column);
    column.extend_ordered_fill_edges(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_on_five_neighbors() {
        let mut entries: Vec<(u32, f64)> = vec![(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let mut out = Vec::new();

        clique_tree_sample(&mut entries, 42, &mut out);

        assert!(out.len() <= 4, "got {} edges, expected <= 4", out.len());
        assert_finite_positive_ordered(&out);
    }

    #[test]
    fn empty_and_single() {
        let mut out = Vec::new();

        clique_tree_sample(&mut [], 0, &mut out);
        assert!(out.is_empty());

        let mut entries = vec![(0u32, 5.0)];
        clique_tree_sample(&mut entries, 0, &mut out);
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
            clique_tree_sample(&mut entries, trial as u64, &mut out);
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
            clique_tree_sample(&mut entries, 7, &mut out);
            assert!(out.is_empty(), "AC emitted fill for a degenerate star");
            clique_tree_sample_multi(&mut entries, 3, 7, &mut out);
            assert!(out.is_empty(), "AC2 emitted fill for a degenerate star");
        }
    }

    #[test]
    fn ac2_respects_split_merge_edge_budget() {
        let mut entries: Vec<(u32, f64)> = vec![(0, 2.0), (1, 3.0), (2, 1.0), (3, 5.0), (4, 4.0)];
        let mut out = Vec::new();

        clique_tree_sample_multi(&mut entries, 2, 42, &mut out);

        assert!(out.len() <= 8, "got {} edges, expected <= 8", out.len());
        assert_finite_positive_ordered(&out);
    }
}
