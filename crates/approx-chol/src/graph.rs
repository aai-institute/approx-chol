//! Elimination graph for approximate Cholesky factorization.

use crate::{CsrRef, Error, Real};
use num_traits::NumCast;

/// Named return type for [`EliminationGraph::from_sddm`].
pub(crate) struct GraphBuild<G, T: Real> {
    pub graph: G,
    pub diagonal: Vec<T>,
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
    fn live_neighbors(&mut self, v: usize, scratch: &mut Vec<Neighbor<T>>);

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
        assert!(n <= u32::MAX as usize, "graph size exceeds u32::MAX");
        let mut adj: Vec<Vec<E>> = Vec::with_capacity(n);
        for row in 0..n {
            let (cols, _) = csr.try_row(row)?;
            adj.push(Vec::with_capacity(cols.len()));
        }
        let mut diag = vec![T::zero(); n];
        let mut row_sums = vec![T::zero(); n];

        for row in 0..n {
            let (cols, vals) = csr.try_row(row)?;
            for (&col, &val) in cols.iter().zip(vals.iter()) {
                let col_usize = col as usize;
                debug_assert!(
                    col_usize < n,
                    "CSR column index {col_usize} out of bounds (n={n})"
                );
                if row == col_usize {
                    diag[row] = diag[row] + val;
                    row_sums[row] = row_sums[row] + val;
                } else if val < T::zero() {
                    row_sums[row] = row_sums[row] + val;
                    // Build a single undirected edge per symmetric pair.
                    if row < col_usize {
                        Self::add_edge_pair(&mut adj, row, col_usize, -val);
                    }
                }
            }
        }

        Ok(Self::build_augmented_laplacian(adj, diag, &row_sums))
    }

    fn n(&self) -> usize {
        self.adj.len()
    }

    fn degree(&self, v: usize) -> usize {
        self.adj[v].iter().map(|e| e.count() as usize).sum()
    }

    fn live_neighbors(&mut self, v: usize, scratch: &mut Vec<Neighbor<T>>) {
        scratch.clear();
        scratch.extend(self.adj[v].iter().filter_map(|e| {
            let count_scalar = <T as NumCast>::from(e.count())?;
            (e.weight() > T::zero() && !self.eliminated.get(e.to() as usize)).then_some(Neighbor {
                to: e.to(),
                fill_weight: e.weight() * count_scalar,
                count: e.count(),
            })
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

/// Small epsilon for Laplacian augmentation, scaled to floating-point precision.
fn augmentation_epsilon<T: Real>() -> T {
    if core::mem::size_of::<T>() <= 4 {
        T::from(1e-6_f64).unwrap_or_else(T::epsilon)
    } else {
        T::from(1e-10_f64).unwrap_or_else(T::epsilon)
    }
}

impl<E: EdgeLike<T>, T: Real> AdjListGraph<E, T> {
    #[inline]
    fn add_edge_pair(adj: &mut [Vec<E>], u: usize, v: usize, weight: T) {
        debug_assert!(adj[v].len() < u32::MAX as usize);
        debug_assert!(adj[u].len() < u32::MAX as usize);
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

    /// Build the final graph, adding Gremban augmentation if needed.
    fn build_augmented_laplacian(
        mut adj: Vec<Vec<E>>,
        mut diag: Vec<T>,
        row_sums: &[T],
    ) -> GraphBuild<Self, T> {
        assert!(
            adj.len() <= u32::MAX as usize,
            "graph size exceeds u32::MAX"
        );
        let m = adj.len();
        let (max_surplus, surplus_sum, surplus_count) = surplus_stats(row_sums);
        let aug_eps = augmentation_epsilon::<T>();
        let needs_augmentation = max_surplus >= aug_eps;

        if needs_augmentation {
            let aux = m as u32;

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
        let eliminated = BitVec::new(n);
        GraphBuild {
            graph: AdjListGraph {
                adj,
                eliminated,
                _marker: core::marker::PhantomData,
            },
            diagonal: diag,
        }
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
