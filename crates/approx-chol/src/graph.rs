//! Elimination graph for approximate Cholesky factorization.

use crate::{CsrError, CsrRef, Error, Real};
use num_traits::NumCast;
use std::collections::HashMap;

/// Named return type for [`EliminationGraph::from_sddm`].
pub(crate) struct GraphBuild<G, T: Real> {
    pub graph: G,
    pub diagonal: Vec<T>,
    pub components: Option<Vec<Vec<u32>>>,
}

/// A neighbor entry produced by star elimination.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neighbor<T> {
    /// Target vertex index.
    pub to: u32,
    /// Accumulated fill weight (weight × count for AC2, just weight for AC).
    pub fill_weight: T,
    /// Edge multiplicity (always 1 for AC, may be >1 for AC2).
    pub count: u32,
}

/// Contract for a mutable graph that supports vertex elimination and fill-in.
pub(crate) trait EliminationGraph<T: Real> {
    /// Construct from a CSR SDDM matrix.
    fn from_sddm(csr: CsrRef<'_, T, u32>) -> Result<GraphBuild<Self, T>, Error>
    where
        Self: Sized;

    /// Number of vertices (fixed at construction time).
    fn n(&self) -> usize;

    /// Current degree of vertex `v` (sum of multi-edge counts; includes stale entries).
    fn degree(&self, v: usize) -> usize;

    /// Collect live (non-eliminated, positive-weight) neighbors of `v` into `scratch`.
    fn live_neighbors(&self, v: usize, scratch: &mut Vec<Neighbor<T>>);

    /// Returns `true` if `v` has an empty adjacency list.
    fn is_empty(&self, v: usize) -> bool;

    /// Mark `v` as eliminated and release its adjacency storage.
    fn eliminate_vertex(&mut self, v: usize);

    /// Insert a symmetric fill edge between `u` and `v` with the given weight.
    fn add_fill_edge(&mut self, u: u32, v: u32, weight: T);
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

/// Abstraction over edge storage: slim (AC) vs multi-edge (AC2).
pub(crate) trait EdgeLike<T: Real>: Clone + Copy {
    fn new(weight: T, to: u32, rev: u32) -> Self;
    fn weight(&self) -> T;
    fn to(&self) -> u32;
    fn rev(&self) -> u32;
    fn set_rev(&mut self, rev: u32);
    /// Virtual multi-edge count. Returns 1 for slim edges.
    fn count(&self) -> u32;
    /// Total fill weight contributed by this edge (`weight * count`).
    /// For slim edges this is just `weight` (no cast/multiply).
    fn fill_weight(&self) -> T;
}

/// Slim edge for AC (no multi-edge tracking).
#[derive(Clone, Copy)]
pub(crate) struct Edge<T: Real> {
    weight: T,
    to: u32,
    rev: u32,
}

impl<T: Real> EdgeLike<T> for Edge<T> {
    #[inline]
    fn new(weight: T, to: u32, rev: u32) -> Self {
        Self { weight, to, rev }
    }
    #[inline]
    fn weight(&self) -> T {
        self.weight
    }
    #[inline]
    fn to(&self) -> u32 {
        self.to
    }
    #[inline]
    fn rev(&self) -> u32 {
        self.rev
    }
    #[inline]
    fn set_rev(&mut self, rev: u32) {
        self.rev = rev;
    }
    #[inline]
    fn count(&self) -> u32 {
        1
    }
    #[inline]
    fn fill_weight(&self) -> T {
        self.weight
    }
}

/// Multi-edge for AC2 with virtual count.
#[derive(Clone, Copy)]
pub(crate) struct MultiEdge<T: Real> {
    weight: T,
    to: u32,
    rev: u32,
    count: u32,
}

impl<T: Real> EdgeLike<T> for MultiEdge<T> {
    #[inline]
    fn new(weight: T, to: u32, rev: u32) -> Self {
        Self {
            weight,
            to,
            rev,
            count: 1,
        }
    }
    #[inline]
    fn weight(&self) -> T {
        self.weight
    }
    #[inline]
    fn to(&self) -> u32 {
        self.to
    }
    #[inline]
    fn rev(&self) -> u32 {
        self.rev
    }
    #[inline]
    fn set_rev(&mut self, rev: u32) {
        self.rev = rev;
    }
    #[inline]
    fn count(&self) -> u32 {
        self.count
    }
    #[inline]
    fn fill_weight(&self) -> T {
        // `count` is 1 (fill/fresh) or a split factor `mark_split_edges` already
        // cast to `T`, so it is always representable here; assert rather than
        // silently yielding a wrong count-1 weight on a failed cast.
        let count: T = <T as NumCast>::from(self.count)
            .expect("edge count is representable in T by construction");
        self.weight * count
    }
}

/// Adjacency-list elimination graph, generic over edge type.
pub(crate) struct AdjListGraph<E: EdgeLike<T>, T: Real> {
    /// Per-vertex adjacency list.
    adj: Vec<Vec<E>>,
    /// `eliminated[v]` is `true` after `eliminate_vertex(v)` has been called.
    eliminated: BitVec,
    _marker: core::marker::PhantomData<T>,
}

/// AC path: slim edges, no multi-edge tracking.
pub(crate) type SlimGraph<T> = AdjListGraph<Edge<T>, T>;

/// AC2 path: edges with virtual multi-edge counts.
pub(crate) type MultiEdgeGraph<T> = AdjListGraph<MultiEdge<T>, T>;

/// Keep capacity of tiny adjacency lists to reduce allocator churn, but release
/// large vectors to avoid retaining fill-heavy buffers across eliminations.
const RETAIN_ADJ_CAPACITY_MAX: usize = 64;

impl<E: EdgeLike<T>, T: Real> EliminationGraph<T> for AdjListGraph<E, T> {
    fn from_sddm(csr: CsrRef<'_, T, u32>) -> Result<GraphBuild<Self, T>, Error> {
        let n = csr.n();
        if n > u32::MAX as usize {
            return Err(Error::InvalidCsr(
                CsrError::MatrixDimensionExceedsIndexType { n },
            ));
        }
        let mut adj: Vec<Vec<E>> = Vec::with_capacity(n);
        for row in 0..n {
            let (cols, _) = csr.try_row(row)?;
            adj.push(Vec::with_capacity(cols.len()));
        }
        let mut diag = vec![T::zero(); n];
        let mut off_diagonals: HashMap<(usize, usize), (Option<T>, Option<T>)> = HashMap::new();
        let mut edge_order = Vec::new();
        let mut upper_edges = Vec::new();

        for (row, diagonal) in diag.iter_mut().enumerate() {
            let (cols, vals) = csr.try_row(row)?;
            let row_start = csr.row_ptrs()[row] as usize;
            for (offset, (&col, &val)) in cols.iter().zip(vals.iter()).enumerate() {
                if !val.is_finite() {
                    return Err(Error::NonFiniteValue {
                        position: row_start + offset,
                    });
                }
                let col_usize = col as usize;
                debug_assert!(
                    col_usize < n,
                    "CSR column index {col_usize} out of bounds (n={n})"
                );
                if row == col_usize {
                    *diagonal = *diagonal + val;
                } else if val < T::zero() {
                    if row < col_usize {
                        upper_edges.push((row, col_usize, -val));
                    }
                    let (lo, hi, side) = if row < col_usize {
                        (row, col_usize, 0)
                    } else {
                        (col_usize, row, 1)
                    };
                    let pair = off_diagonals.entry((lo, hi)).or_insert_with(|| {
                        edge_order.push((lo, hi));
                        (None, None)
                    });
                    let slot = if side == 0 { &mut pair.0 } else { &mut pair.1 };
                    *slot = Some(slot.unwrap_or_else(T::zero) + val);
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
        let mut row_sums = diag.clone();
        let mut row_scales: Vec<T> = diag.iter().map(|value| value.abs()).collect();
        for (row, col) in edge_order {
            let (upper, lower) = off_diagonals
                .remove(&(row, col))
                .expect("edge order and symmetry map are built together");
            let (Some(upper), Some(lower)) = (upper, lower) else {
                return Err(Error::Asymmetric { edge: (row, col) });
            };
            if upper != lower {
                return Err(Error::Asymmetric { edge: (row, col) });
            }
            row_sums[row] = row_sums[row] + upper;
            row_sums[col] = row_sums[col] + upper;
            row_scales[row] = row_scales[row] + upper.abs();
            row_scales[col] = row_scales[col] + upper.abs();
        }
        for (row, col, weight) in upper_edges {
            Self::add_edge_pair(&mut adj, row, col, weight);
        }
        let tolerance = augmentation_eps::<T>();
        for row in 0..n {
            let row_tolerance = tolerance * row_scales[row];
            if row_sums[row] < -row_tolerance {
                return Err(Error::NotDiagonallyDominant { row });
            }
            if row_sums[row].abs() <= row_tolerance {
                row_sums[row] = T::zero();
            }
        }
        Self::build_augmented_laplacian(adj, diag, &row_sums)
    }

    fn n(&self) -> usize {
        self.adj.len()
    }

    fn degree(&self, v: usize) -> usize {
        self.adj[v].iter().map(|e| e.count() as usize).sum()
    }

    fn live_neighbors(&self, v: usize, scratch: &mut Vec<Neighbor<T>>) {
        scratch.clear();
        scratch.extend(self.adj[v].iter().filter_map(|e| {
            // Positive predicate: a NaN weight is dead (`!(w > 0)` differs from
            // `w <= 0` at NaN). if/else (not `bool::then`) keeps `fill_weight()`
            // lazy for dead edges and avoids `clippy::filter_map_bool_then`.
            if e.weight() > T::zero() && !self.eliminated.get(e.to() as usize) {
                Some(Neighbor {
                    to: e.to(),
                    fill_weight: e.fill_weight(),
                    count: e.count(),
                })
            } else {
                None
            }
        }));
    }

    fn is_empty(&self, v: usize) -> bool {
        self.adj[v].is_empty()
    }

    fn eliminate_vertex(&mut self, v: usize) {
        self.eliminated.set(v);
        while let Some(edge) = self.adj[v].pop() {
            let u = edge.to() as usize;
            if self.eliminated.get(u) {
                continue;
            }
            debug_assert!(
                (edge.rev() as usize) < self.adj[u].len(),
                "reverse pointer out of bounds: rev={} but adj[{}].len()={}",
                edge.rev(),
                u,
                self.adj[u].len()
            );
            Self::remove_edge_at(&mut self.adj, u, edge.rev() as usize);
        }
        if self.adj[v].capacity() > RETAIN_ADJ_CAPACITY_MAX {
            self.adj[v] = Vec::new();
        }
    }

    fn add_fill_edge(&mut self, u: u32, v: u32, weight: T) {
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

impl<E: EdgeLike<T>, T: Real> AdjListGraph<E, T> {
    pub(crate) fn dense_principal(&self, diagonal: &[T], vertices: &[u32]) -> (Vec<T>, Vec<u32>) {
        let pivot_vertices = if vertices.is_empty() {
            Vec::new()
        } else {
            vertices[..vertices.len() - 1].to_vec()
        };
        let m = pivot_vertices.len();
        let local_of: HashMap<u32, usize> = pivot_vertices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();
        let mut matrix = vec![T::zero(); m * m];
        for (local, &global) in pivot_vertices.iter().enumerate() {
            matrix[local * m + local] = diagonal[global as usize];
            for edge in &self.adj[global as usize] {
                if let Some(&other) = local_of.get(&edge.to()) {
                    matrix[local * m + other] = matrix[local * m + other] - edge.weight();
                }
            }
        }
        (matrix, pivot_vertices)
    }

    pub(crate) fn into_components(
        self,
        diagonal: Vec<T>,
        components: Vec<Vec<u32>>,
    ) -> Vec<(Self, Vec<T>, Vec<u32>)> {
        let mut result = Vec::with_capacity(components.len());
        let mut local_of = vec![usize::MAX; self.adj.len()];
        for vertices in components {
            for (local, &global) in vertices.iter().enumerate() {
                local_of[global as usize] = local;
            }
            let mut adjacency = vec![Vec::new(); vertices.len()];
            for (local_u, &global_u) in vertices.iter().enumerate() {
                for edge in &self.adj[global_u as usize] {
                    let local_v = local_of[edge.to() as usize];
                    if local_v != usize::MAX && local_u < local_v {
                        Self::add_edge_pair(&mut adjacency, local_u, local_v, edge.weight());
                    }
                }
            }
            let local_diagonal = vertices
                .iter()
                .map(|&vertex| diagonal[vertex as usize])
                .collect();
            for &global in &vertices {
                local_of[global as usize] = usize::MAX;
            }
            result.push((
                Self {
                    eliminated: BitVec::new(vertices.len()),
                    adj: adjacency,
                    _marker: core::marker::PhantomData,
                },
                local_diagonal,
                vertices,
            ));
        }
        result
    }

    #[inline]
    fn add_edge_pair(adj: &mut [Vec<E>], u: usize, v: usize, weight: T) {
        // u32 reverse pointers; overflow is unreachable for tractable inputs,
        // so assert (release too) rather than truncate and corrupt removal.
        assert!(
            adj[u].len() < u32::MAX as usize && adj[v].len() < u32::MAX as usize,
            "adjacency list exceeds u32 edge capacity"
        );
        let rev_u = adj[v].len() as u32;
        let rev_v = adj[u].len() as u32;
        adj[u].push(E::new(weight, v as u32, rev_u));
        adj[v].push(E::new(weight, u as u32, rev_v));
    }

    /// Remove `adj[u][idx]` in O(1) via swap-remove and repair the moved edge's
    /// reverse pointer in its opposite adjacency list.
    fn remove_edge_at(adj: &mut [Vec<E>], u: usize, idx: usize) {
        let last_idx = adj[u].len() - 1;
        adj[u].swap_remove(idx);
        if idx < last_idx {
            let moved = adj[u][idx];
            let w = moved.to() as usize;
            let rev = moved.rev() as usize;
            adj[w][rev].set_rev(idx as u32);
        }
    }

    /// Connected components among the first `n_real` vertices. Traversal follows
    /// every edge, so a ground vertex (index `>= n_real`) links the blocks it
    /// touches without being counted as its own component.
    fn components(adj: &[Vec<E>], n_real: usize) -> Vec<Vec<u32>> {
        let mut visited = BitVec::new(adj.len());
        let mut stack: Vec<usize> = Vec::new();
        let mut components = Vec::new();
        for start in 0..n_real {
            if visited.get(start) {
                continue;
            }
            let mut component = Vec::new();
            visited.set(start);
            stack.push(start);
            while let Some(v) = stack.pop() {
                component.push(v as u32);
                for e in &adj[v] {
                    let u = e.to() as usize;
                    if !visited.get(u) {
                        visited.set(u);
                        stack.push(u);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components
    }

    /// Build the final graph, apply Gremban augmentation, and retain component
    /// labels only when block dispatch is required.
    fn build_augmented_laplacian(
        mut adj: Vec<Vec<E>>,
        mut diag: Vec<T>,
        row_sums: &[T],
    ) -> Result<GraphBuild<Self, T>, Error> {
        let m = adj.len();
        let (max_surplus, surplus_sum, surplus_count) = surplus_stats(row_sums);
        let needs_augmentation = max_surplus > T::zero();

        if needs_augmentation {
            if m >= u32::MAX as usize {
                return Err(Error::InvalidCsr(
                    CsrError::MatrixDimensionExceedsIndexType {
                        n: m.saturating_add(1),
                    },
                ));
            }
            let aux = u32::try_from(m).map_err(|_| {
                Error::InvalidCsr(CsrError::MatrixDimensionExceedsIndexType { n: m })
            })?;

            // Add augmentation vertex adjacency list
            adj.push(Vec::with_capacity(surplus_count));

            // Extend diagonal
            diag.push(surplus_sum);

            for (row, &surplus_raw) in row_sums.iter().enumerate() {
                let surplus = surplus_raw.max(T::zero());
                if surplus > T::zero() {
                    Self::add_edge_pair(&mut adj, row, aux as usize, surplus);
                }
            }
        }

        let n = adj.len();
        let components = Self::components(&adj, m);
        let components = if components.len() > 1 {
            Some(components)
        } else {
            None
        };

        let eliminated = BitVec::new(n);
        Ok(GraphBuild {
            graph: AdjListGraph {
                adj,
                eliminated,
                _marker: core::marker::PhantomData,
            },
            diagonal: diag,
            components,
        })
    }
}

impl<T: Real> MultiEdgeGraph<T> {
    /// Mark each edge as `k` virtual copies at `weight / k`.
    pub(crate) fn mark_split_edges(&mut self, k: u32) {
        if k <= 1 {
            return;
        }
        let Some(k_scalar) = <T as NumCast>::from(k) else {
            return;
        };
        let inv_k = T::one() / k_scalar;
        for adj_list in &mut self.adj {
            for edge in adj_list.iter_mut() {
                edge.weight = edge.weight * inv_k;
                edge.count = k;
            }
        }
    }
}
