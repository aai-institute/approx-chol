//! Elimination graph for approximate Cholesky factorization.

use crate::types::count_as_scalar;
use crate::{CsrError, CsrRef, Error, Real};

/// Named return type for [`AdjListGraph::from_sddm`].
pub(crate) struct GraphBuild<G, T: Real> {
    pub graph: G,
    pub diagonal: Vec<T>,
}

/// A neighbor entry produced by star elimination. Carries the edge's
/// multiplicity storage, so the AC path has no multiplicity field to fill in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neighbor<T, C> {
    pub to: u32,
    /// Accumulated fill weight (weight × count for AC2, just weight for AC).
    pub fill_weight: T,
    pub count: C,
}

struct BitVec {
    words: Vec<u64>,
}

impl BitVec {
    fn new(n: usize) -> Self {
        Self {
            words: vec![0u64; n.div_ceil(64)],
        }
    }

    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline]
    fn get(&self, i: usize) -> bool {
        self.words[i >> 6] & (1u64 << (i & 63)) != 0
    }
}

/// How an edge stores its multiplicity: the only thing that differs between the
/// AC and AC2 edge layouts, so it is the only thing either one defines.
///
/// `Single` is a ZST, which keeps the AC edge exactly as wide as it was before
/// the count existed (asserted in the tests below).
pub(crate) trait EdgeCount: Clone + Copy {
    fn one() -> Self;
    fn get(&self) -> u32;
    /// Fill weight this multiplicity contributes at edge weight `weight`.
    fn scale<T: Real>(&self, weight: T) -> T;
}

/// AC: every edge is a single edge, so there is nothing to store.
#[derive(Clone, Copy)]
pub(crate) struct Single;

/// AC2: a virtual multiplicity set by [`MultiEdgeGraph::mark_split_edges`].
#[derive(Clone, Copy)]
pub(crate) struct Multi(u32);

impl Multi {
    pub(crate) fn new(count: u32) -> Self {
        Self(count)
    }
}

impl EdgeCount for Single {
    #[inline]
    fn one() -> Self {
        Self
    }
    #[inline]
    fn get(&self) -> u32 {
        1
    }
    #[inline]
    fn scale<T: Real>(&self, weight: T) -> T {
        weight
    }
}

impl EdgeCount for Multi {
    #[inline]
    fn one() -> Self {
        Self(1)
    }
    #[inline]
    fn get(&self) -> u32 {
        self.0
    }
    #[inline]
    fn scale<T: Real>(&self, weight: T) -> T {
        weight * count_as_scalar::<T, _>(self.0)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Edge<T: Real, C> {
    weight: T,
    to: u32,
    rev: u32,
    count: C,
}

impl<T: Real, C: EdgeCount> Edge<T, C> {
    #[inline]
    fn new(weight: T, to: u32, rev: u32) -> Self {
        Self {
            weight,
            to,
            rev,
            count: C::one(),
        }
    }

    #[inline]
    fn fill_weight(&self) -> T {
        self.count.scale(self.weight)
    }
}

/// Adjacency-list elimination graph, generic over edge multiplicity storage.
pub(crate) struct AdjListGraph<C, T: Real> {
    /// Per-vertex adjacency list.
    adj: Vec<Vec<Edge<T, C>>>,
    /// `eliminated[v]` is `true` after `eliminate_vertex(v)` has been called.
    eliminated: BitVec,
}

/// AC path: no multi-edge tracking.
pub(crate) type SlimGraph<T> = AdjListGraph<Single, T>;

/// AC2 path: edges with virtual multi-edge counts.
pub(crate) type MultiEdgeGraph<T> = AdjListGraph<Multi, T>;

/// Keep capacity of tiny adjacency lists to reduce allocator churn, but release
/// large vectors to avoid retaining fill-heavy buffers across eliminations.
const RETAIN_ADJ_CAPACITY_MAX: usize = 64;

impl<C: EdgeCount, T: Real> AdjListGraph<C, T> {
    /// Construct from a CSR SDDM matrix.
    pub(crate) fn from_sddm(csr: CsrRef<'_, T, u32>) -> Result<GraphBuild<Self, T>, Error> {
        let n = csr.n();
        let mut adj: Vec<Vec<Edge<T, C>>> = Vec::with_capacity(n);
        for (cols, _) in csr.rows() {
            adj.push(Vec::with_capacity(cols.len()));
        }
        let mut diag = vec![T::zero(); n];
        let mut row_sums = vec![T::zero(); n];

        for (row, (cols, vals)) in csr.rows().enumerate() {
            for (&col, &val) in cols.iter().zip(vals.iter()) {
                let col_usize = col as usize;
                if row == col_usize {
                    diag[row] = diag[row] + val;
                    row_sums[row] = row_sums[row] + val;
                } else if val < T::zero() {
                    row_sums[row] = row_sums[row] + val;
                    // Build a single undirected edge per symmetric pair.
                    if row < col_usize {
                        Self::add_edge_pair(&mut adj, row, col_usize, -val);
                    }
                } else if val > T::zero() {
                    // Positive off-diagonal: outside the SDDM/Laplacian class.
                    // Reject instead of falling through — the silent drop here
                    // corrupted the factored matrix.
                    return Err(Error::PositiveOffDiagonal {
                        edge: (row, col_usize),
                    });
                }
            }
        }
        Self::build_augmented_laplacian(adj, diag, &row_sums)
    }

    /// Number of vertices (fixed at construction time).
    pub(crate) fn n(&self) -> usize {
        self.adj.len()
    }

    /// Current degree of vertex `v` (sum of multi-edge counts; includes stale entries).
    pub(crate) fn degree(&self, v: usize) -> usize {
        self.adj[v].iter().map(|e| e.count.get() as usize).sum()
    }

    /// Collect live (non-eliminated, positive-weight) neighbors of `v` into `scratch`.
    pub(crate) fn live_neighbors(&self, v: usize, scratch: &mut Vec<Neighbor<T, C>>) {
        scratch.clear();
        scratch.extend(self.adj[v].iter().filter_map(|e| {
            // Positive predicate: a NaN weight is dead (`!(w > 0)` differs from
            // `w <= 0` at NaN). if/else (not `bool::then`) keeps `fill_weight()`
            // lazy for dead edges and avoids `clippy::filter_map_bool_then`.
            if e.weight > T::zero() && !self.eliminated.get(e.to as usize) {
                Some(Neighbor {
                    to: e.to,
                    fill_weight: e.fill_weight(),
                    count: e.count,
                })
            } else {
                None
            }
        }));
    }

    /// Returns `true` if `v` has an empty adjacency list.
    pub(crate) fn is_empty(&self, v: usize) -> bool {
        self.adj[v].is_empty()
    }

    /// Mark `v` as eliminated and release its adjacency storage.
    pub(crate) fn eliminate_vertex(&mut self, v: usize) {
        self.eliminated.set(v);
        while let Some(edge) = self.adj[v].pop() {
            let u = edge.to as usize;
            if self.eliminated.get(u) {
                continue;
            }
            debug_assert!(
                (edge.rev as usize) < self.adj[u].len(),
                "reverse pointer out of bounds: rev={} but adj[{}].len()={}",
                edge.rev,
                u,
                self.adj[u].len()
            );
            Self::remove_edge_at(&mut self.adj, u, edge.rev as usize);
        }
        if self.adj[v].capacity() > RETAIN_ADJ_CAPACITY_MAX {
            self.adj[v] = Vec::new();
        }
    }

    /// Insert a symmetric fill edge between `u` and `v` with the given weight.
    pub(crate) fn add_fill_edge(&mut self, u: u32, v: u32, weight: T) {
        if u == v {
            return;
        }
        Self::add_edge_pair(&mut self.adj, u as usize, v as usize, weight);
    }
}

/// Compute (max_abs_surplus, total_surplus_sum, count_nonzero) from row sums.
fn surplus_stats<T: Real>(row_sums: &[T]) -> (T, T, usize) {
    row_sums
        .iter()
        .fold((T::zero(), T::zero(), 0usize), |(max_s, sum, cnt), &s| {
            (
                max_s.max(s.abs()),
                sum + s,
                cnt + (s.abs() > T::zero()) as usize,
            )
        })
}

/// Absolute surplus floor separating genuine diagonal dominance from a
/// Laplacian's rounding noise (scaled down for sub-unit matrices at the use site).
fn augmentation_eps<T: Real>() -> T {
    if core::mem::size_of::<T>() <= 4 {
        T::from(1e-6_f64).unwrap_or_else(T::epsilon)
    } else {
        T::from(1e-10_f64).unwrap_or_else(T::epsilon)
    }
}

impl<C: EdgeCount, T: Real> AdjListGraph<C, T> {
    #[inline]
    fn add_edge_pair(adj: &mut [Vec<Edge<T, C>>], u: usize, v: usize, weight: T) {
        // u32 reverse pointers; overflow is unreachable for tractable inputs,
        // so assert (release too) rather than truncate and corrupt removal.
        assert!(
            adj[u].len() < u32::MAX as usize && adj[v].len() < u32::MAX as usize,
            "adjacency list exceeds u32 edge capacity"
        );
        let rev_u = adj[v].len() as u32;
        let rev_v = adj[u].len() as u32;
        adj[u].push(Edge::new(weight, v as u32, rev_u));
        adj[v].push(Edge::new(weight, u as u32, rev_v));
    }

    /// Remove `adj[u][idx]` in O(1) via swap-remove and repair the moved edge's
    /// reverse pointer in its opposite adjacency list.
    fn remove_edge_at(adj: &mut [Vec<Edge<T, C>>], u: usize, idx: usize) {
        let last_idx = adj[u].len() - 1;
        adj[u].swap_remove(idx);
        if idx < last_idx {
            let moved = adj[u][idx];
            let w = moved.to as usize;
            let rev = moved.rev as usize;
            adj[w][rev].rev = idx as u32;
        }
    }

    /// Connected components among the first `n_real` vertices. Traversal follows
    /// every edge, so a ground vertex (index `>= n_real`) links the blocks it
    /// touches without being counted as its own component.
    fn count_components(adj: &[Vec<Edge<T, C>>], n_real: usize) -> usize {
        let mut visited = BitVec::new(adj.len());
        let mut stack: Vec<usize> = Vec::new();
        let mut components = 0usize;
        for start in 0..n_real {
            if visited.get(start) {
                continue;
            }
            components += 1;
            visited.set(start);
            stack.push(start);
            while let Some(v) = stack.pop() {
                for e in &adj[v] {
                    let u = e.to as usize;
                    if !visited.get(u) {
                        visited.set(u);
                        stack.push(u);
                    }
                }
            }
        }
        components
    }

    /// Build the final graph: apply Gremban augmentation if needed, then reject
    /// disconnected input (`Error::Disconnected`).
    fn build_augmented_laplacian(
        mut adj: Vec<Vec<Edge<T, C>>>,
        mut diag: Vec<T>,
        row_sums: &[T],
    ) -> Result<GraphBuild<Self, T>, Error> {
        let m = adj.len();
        let (max_surplus, surplus_sum, surplus_count) = surplus_stats(row_sums);
        // Surplus floor: absolute `augmentation_eps`, shrunk proportionally below
        // scale 1. The cap keeps a large-scale barely-PD input augmented (consumers
        // rely on it); `near_zero` rejects input the elimination can't resolve.
        let max_diag = diag.iter().fold(T::zero(), |acc, &d| acc.max(d.abs()));
        let floor = augmentation_eps::<T>() * max_diag.min(T::one());
        let needs_augmentation = max_surplus > floor && max_diag > T::near_zero();

        if needs_augmentation {
            // The ground vertex takes index `m`, so the augmented graph needs one
            // more `u32` than the input dimension. Checking that here leaves the
            // cast below infallible.
            if m >= u32::MAX as usize {
                return Err(Error::InvalidCsr(
                    CsrError::MatrixDimensionExceedsIndexType {
                        n: m.saturating_add(1),
                    },
                ));
            }
            let aux = m as u32;

            adj.push(Vec::with_capacity(surplus_count));
            diag.push(surplus_sum);

            for (row, &surplus_raw) in row_sums.iter().enumerate() {
                let surplus = surplus_raw.max(T::zero());
                if surplus > T::zero() {
                    Self::add_edge_pair(&mut adj, row, aux as usize, surplus);
                }
            }
        }

        let n = adj.len();
        // Reject before the expensive elimination: >1 component (over the real
        // vertices) means a block can't reach ground. See `Error::Disconnected`.
        let components = Self::count_components(&adj, m);
        if components > 1 {
            return Err(Error::Disconnected { components });
        }

        let eliminated = BitVec::new(n);
        Ok(GraphBuild {
            graph: AdjListGraph { adj, eliminated },
            diagonal: diag,
        })
    }
}

impl<T: Real> MultiEdgeGraph<T> {
    /// Mark each edge as `k` virtual copies at `weight / k`.
    pub(crate) fn mark_split_edges(&mut self, k: u32) {
        if k <= 1 {
            return;
        }
        let inv_k = T::one() / count_as_scalar::<T, _>(k);
        for adj_list in &mut self.adj {
            for edge in adj_list.iter_mut() {
                edge.weight = edge.weight * inv_k;
                edge.count = Multi::new(k);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AC edge must not pay for the multiplicity it does not store: `Single`
    /// is a ZST, so both layouts are what the two hand-written structs were.
    #[test]
    fn edge_layout_is_unchanged_by_the_shared_definition() {
        assert_eq!(
            size_of::<Edge<f64, Single>>(),
            size_of::<f64>() + 2 * size_of::<u32>()
        );
        assert_eq!(
            size_of::<Edge<f64, Multi>>(),
            size_of::<Edge<f64, Single>>() + size_of::<f64>(),
            "one u32 plus its alignment padding"
        );
        assert_eq!(size_of::<Single>(), 0);
    }
}
