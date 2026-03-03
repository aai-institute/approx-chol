use crate::graph::EliminationGraph;
use crate::ordering::EliminationOrdering;
use crate::sampling::WeightedSampler;
use crate::types::Real;

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
    pub(super) fill_edges: Vec<(u32, u32, T)>,
}

impl<T: Real> SampledColumn<T> {
    pub fn new() -> Self {
        Self {
            diagonal: T::zero(),
            neighbors: Vec::new(),
            fractions: Vec::new(),
            fill_edges: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.diagonal = T::zero();
        self.neighbors.clear();
        self.fractions.clear();
        self.fill_edges.clear();
    }

    /// Initialize column sampling from a star's deduplicated neighbor list.
    ///
    /// Returns `Some(n)` when sampling should continue (`n >= 2`), otherwise
    /// writes the trivial result (`n == 0` or `n == 1`) and returns `None`.
    pub(super) fn begin_sampling(&mut self, entries: &[(u32, T)], pivot_diag: T) -> Option<usize> {
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
    pub(super) fn finalize_sampling(&mut self, last: (u32, T), elim: &StarElimination<T>) {
        self.neighbors.push(last.0);
        self.fractions.push(T::one());
        self.diagonal = elim.diagonal(last.1);
    }

    /// Apply fill-in edges to the graph, update diagonal values, and notify ordering.
    pub fn apply_fill_in<G: EliminationGraph<T>, O: EliminationOrdering<T>>(
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
    pub(super) fn sample_fill_edges(
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
/// After [`advance(f)`](Self::advance), both `scale` and `capacity` shrink,
/// ensuring subsequent fractions account for weight already consumed by
/// earlier fill edges.
pub(crate) struct StarElimination<T = f64> {
    scale: T,
    capacity: T,
}

impl<T: Real> StarElimination<T> {
    /// Start elimination with given initial capacity.
    ///
    /// - **AC (Algorithm 5):** pass `pivot_diag` (the matrix diagonal entry).
    /// - **AC2 (Algorithm 6):** pass `total_weight` (sum of incident edge weights),
    ///   which normalizes the fill-weight formula `w̄ · remaining / (t · d)`.
    #[inline(always)]
    pub(crate) fn new(capacity: T) -> Self {
        Self {
            scale: T::one(),
            capacity,
        }
    }

    /// Elimination fraction for edge with original weight `w`.
    #[inline(always)]
    pub(crate) fn fraction(&self, w: T) -> T {
        debug_assert!(self.capacity > T::epsilon());
        w * self.scale / self.capacity
    }

    /// Current remaining weight budget.
    #[inline(always)]
    pub(crate) fn capacity(&self) -> T {
        self.capacity
    }

    /// Advance state after eliminating with fraction `f`.
    #[inline(always)]
    pub(crate) fn advance(&mut self, f: T) {
        let retain = T::one() - f;
        self.scale = self.scale * retain;
        self.capacity = self.capacity * retain * retain;
    }

    /// Diagonal entry of the factor column.
    #[inline(always)]
    pub(crate) fn diagonal(&self, last_weight: T) -> T {
        last_weight * self.scale
    }
}
