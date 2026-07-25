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

/// One stored off-diagonal, bucketed by its lower index so that index is implied.
struct DirectedOffDiagonal<T: Real> {
    hi: u32,
    /// Whether the entry was stored above the diagonal, needed to compare the two
    /// triangles for symmetry.
    upper: bool,
    value: T,
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
    fn reindex(self, to: u32, rev: u32) -> Self;
    /// `k` virtual copies at `weight * inv_k`; slim edges are unchanged.
    fn split(self, inv_k: T, k: u32) -> Self;
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
    fn reindex(self, to: u32, rev: u32) -> Self {
        Self {
            weight: self.weight,
            to,
            rev,
        }
    }
    #[inline]
    fn split(self, _inv_k: T, _k: u32) -> Self {
        self
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
    fn reindex(self, to: u32, rev: u32) -> Self {
        Self {
            weight: self.weight,
            to,
            rev,
            count: self.count,
        }
    }
    #[inline]
    fn split(self, inv_k: T, k: u32) -> Self {
        Self {
            weight: self.weight * inv_k,
            count: k,
            ..self
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

        if Self::columns_strictly_ascending(&csr)? {
            return Self::from_canonical_sddm(csr, adj, diag);
        }

        // Pair up the two stored triangles by bucketing every off-diagonal on its
        // lower index. Counting the buckets first, then scattering, keeps this
        // O(nnz) — a comparison sort over all entries dominated ingestion.
        let mut bucket_ends = vec![0u32; n + 1];
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
                } else if val != T::zero() {
                    bucket_ends[row.min(col_usize) + 1] += 1;
                }
            }
        }
        for index in 0..n {
            bucket_ends[index + 1] += bucket_ends[index];
        }
        let mut cursors = bucket_ends[..n].to_vec();
        let mut off_diagonals: Vec<DirectedOffDiagonal<T>> = Vec::new();
        off_diagonals.resize_with(bucket_ends[n] as usize, || DirectedOffDiagonal {
            hi: 0,
            upper: false,
            value: T::zero(),
        });
        for row in 0..n {
            let (cols, vals) = csr.try_row(row)?;
            for (&col, &val) in cols.iter().zip(vals.iter()) {
                let col_usize = col as usize;
                if row == col_usize || val == T::zero() {
                    continue;
                }
                let lo = row.min(col_usize);
                let slot = &mut cursors[lo];
                off_diagonals[*slot as usize] = DirectedOffDiagonal {
                    hi: row.max(col_usize) as u32,
                    upper: row < col_usize,
                    value: val,
                };
                *slot += 1;
            }
        }

        let mut row_sums = diag.clone();
        let mut row_scales: Vec<T> = diag.iter().map(|value| value.abs()).collect();
        for row in 0..n {
            let bucket =
                &mut off_diagonals[bucket_ends[row] as usize..bucket_ends[row + 1] as usize];
            // Buckets hold one vertex's stored neighbours, so this is a sort over
            // the degree rather than over nnz. Stable, so duplicates sum in order.
            bucket.sort_by_key(|entry| entry.hi);
            let mut index = 0;
            while index < bucket.len() {
                let col = bucket[index].hi as usize;
                let mut upper = T::zero();
                let mut lower = T::zero();
                while index < bucket.len() && bucket[index].hi as usize == col {
                    if bucket[index].upper {
                        upper = upper + bucket[index].value;
                    } else {
                        lower = lower + bucket[index].value;
                    }
                    index += 1;
                }
                if !approximately_equal(upper, lower) {
                    return Err(Error::Asymmetric { edge: (row, col) });
                }
                if upper > T::zero() {
                    return Err(Error::PositiveOffDiagonal { edge: (row, col) });
                }
                if upper < T::zero() {
                    row_sums[row] = row_sums[row] + upper;
                    row_sums[col] = row_sums[col] + upper;
                    row_scales[row] = row_scales[row] + upper.abs();
                    row_scales[col] = row_scales[col] + upper.abs();
                    Self::add_edge_pair(&mut adj, row, col, -upper);
                }
            }
        }
        Self::augment(adj, diag, row_sums, &row_scales)
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

fn approximately_equal<T: Real>(left: T, right: T) -> bool {
    if left == right {
        return true;
    }
    let ulps = T::from(8.0).unwrap_or_else(T::one);
    let scale = left.abs().max(right.abs());
    (left - right).abs() <= ulps * T::epsilon() * scale
}

impl<E: EdgeLike<T>, T: Real> AdjListGraph<E, T> {
    /// Clamp each row's surplus and apply Gremban augmentation.
    fn augment(
        adj: Vec<Vec<E>>,
        diag: Vec<T>,
        mut row_sums: Vec<T>,
        row_scales: &[T],
    ) -> Result<GraphBuild<Self, T>, Error> {
        let tolerance = augmentation_eps::<T>();
        for row in 0..row_sums.len() {
            // Every check below succeeds on an infinite deficit (`-inf < -inf`).
            if !row_sums[row].is_finite() || !row_scales[row].is_finite() {
                return Err(Error::NonFiniteRow { row });
            }
            let row_tolerance = tolerance * row_scales[row];
            if row_sums[row] < -row_tolerance {
                return Err(Error::NotDiagonallyDominant { row });
            }
            // The cap stops the relative arm swallowing real dominance at large
            // scale; both ends are pinned by `*_scale_*` ingestion tests.
            let augmentation_floor = row_tolerance.min(T::epsilon().sqrt());
            if row_sums[row] < T::zero() || row_sums[row].abs() <= augmentation_floor {
                row_sums[row] = T::zero();
            }
        }
        Self::build_augmented_laplacian(adj, diag, &row_sums)
    }

    /// Strictly ascending columns imply no duplicate entries, which lets
    /// [`Self::from_canonical_sddm`] pair the stored triangles without reordering.
    /// Scans indices only, never values.
    fn columns_strictly_ascending(csr: &CsrRef<'_, T, u32>) -> Result<bool, Error> {
        for row in 0..csr.n() {
            let (cols, _) = csr.try_row(row)?;
            if cols.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Ingest canonical CSR — the shape scipy produces and the only one within
    /// emits — by walking each row once and claiming each upper-triangle entry's
    /// mirror through a monotone per-row cursor. Accumulates into `row_sums` in the
    /// same order as the bucketed path, so both agree bit for bit.
    fn from_canonical_sddm(
        csr: CsrRef<'_, T, u32>,
        mut adj: Vec<Vec<E>>,
        mut diag: Vec<T>,
    ) -> Result<GraphBuild<Self, T>, Error> {
        let n = csr.n();
        let row_ptrs = csr.row_ptrs();
        let col_indices = csr.col_indices();
        let values = csr.values();

        // Read the diagonal by search rather than a full pass, so the off-diagonal
        // accumulation below can start from it as the bucketed path does.
        for (row, diagonal) in diag.iter_mut().enumerate() {
            let (cols, vals) = csr.try_row(row)?;
            let offset = cols.partition_point(|&col| (col as usize) < row);
            if cols.get(offset) == Some(&(row as u32)) {
                let value = vals[offset];
                if !value.is_finite() {
                    return Err(Error::NonFiniteValue {
                        position: row_ptrs[row] as usize + offset,
                    });
                }
                *diagonal = value;
            }
        }
        let mut row_sums = diag.clone();
        let mut row_scales: Vec<T> = diag.iter().map(|value| value.abs()).collect();

        let mut cursors: Vec<u32> = row_ptrs[..n].to_vec();
        for row in 0..n {
            let row_end = row_ptrs[row + 1];
            let mut cursor = cursors[row];
            // Anything still below the diagonal was never claimed as a mirror, so
            // its counterpart above the diagonal is missing.
            while cursor < row_end {
                let col = col_indices[cursor as usize] as usize;
                if col >= row {
                    break;
                }
                let value = values[cursor as usize];
                if !value.is_finite() {
                    return Err(Error::NonFiniteValue {
                        position: cursor as usize,
                    });
                }
                if value != T::zero() {
                    return Err(Error::Asymmetric { edge: (col, row) });
                }
                cursor += 1;
            }
            if cursor < row_end && col_indices[cursor as usize] as usize == row {
                cursor += 1;
            }
            cursors[row] = cursor;

            while cursor < row_end {
                let col = col_indices[cursor as usize] as usize;
                let upper = values[cursor as usize];
                if !upper.is_finite() {
                    return Err(Error::NonFiniteValue {
                        position: cursor as usize,
                    });
                }
                cursor += 1;
                if upper == T::zero() {
                    continue;
                }
                let lower = Self::claim_mirror(col, row, &csr, &mut cursors)?;
                if !approximately_equal(upper, lower) {
                    return Err(Error::Asymmetric { edge: (row, col) });
                }
                if upper > T::zero() {
                    return Err(Error::PositiveOffDiagonal { edge: (row, col) });
                }
                row_sums[row] = row_sums[row] + upper;
                row_sums[col] = row_sums[col] + upper;
                row_scales[row] = row_scales[row] + upper.abs();
                row_scales[col] = row_scales[col] + upper.abs();
                Self::add_edge_pair(&mut adj, row, col, -upper);
            }
        }
        Self::augment(adj, diag, row_sums, &row_scales)
    }

    /// Consume `row`'s entry at `col`, treating stored zeros as absent and
    /// returning zero when it is missing.
    fn claim_mirror(
        row: usize,
        col: usize,
        csr: &CsrRef<'_, T, u32>,
        cursors: &mut [u32],
    ) -> Result<T, Error> {
        let row_ptrs = csr.row_ptrs();
        let col_indices = csr.col_indices();
        let values = csr.values();
        let row_end = row_ptrs[row + 1];
        let mut cursor = cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            let at = col_indices[cursor as usize] as usize;
            if at > col {
                break;
            }
            let value = values[cursor as usize];
            if !value.is_finite() {
                return Err(Error::NonFiniteValue {
                    position: cursor as usize,
                });
            }
            cursor += 1;
            if at == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                cursors[row] = cursor;
                return Err(Error::Asymmetric { edge: (at, row) });
            }
        }
        cursors[row] = cursor;
        Ok(found)
    }

    /// The anchor-deleted principal submatrix, row-major, or `None` when `m * m`
    /// scalars do not fit — a dispatch signal, not an input error.
    pub(crate) fn dense_principal(
        &self,
        diagonal: &[T],
        vertices: &[u32],
    ) -> Option<(Vec<T>, Vec<u32>)> {
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
        let matrix_len = m.checked_mul(m)?;
        let mut matrix = Vec::new();
        matrix.try_reserve_exact(matrix_len).ok()?;
        matrix.resize(matrix_len, T::zero());
        for (local, &global) in pivot_vertices.iter().enumerate() {
            matrix[local * m + local] = diagonal[global as usize];
            for edge in &self.adj[global as usize] {
                if let Some(&other) = local_of.get(&edge.to()) {
                    matrix[local * m + other] = matrix[local * m + other] - edge.fill_weight();
                }
            }
        }
        Some((matrix, pivot_vertices))
    }

    pub(crate) fn extract_component(
        &self,
        diagonal: &[T],
        vertices: &[u32],
        local_of: &mut [usize],
    ) -> (Self, Vec<T>) {
        debug_assert_eq!(local_of.len(), self.adj.len());
        for (local, &global) in vertices.iter().enumerate() {
            local_of[global as usize] = local;
        }
        let mut adjacency = vec![Vec::new(); vertices.len()];
        for (local_u, &global_u) in vertices.iter().enumerate() {
            for &edge in &self.adj[global_u as usize] {
                let local_v = local_of[edge.to() as usize];
                if local_v != usize::MAX && local_u < local_v {
                    Self::add_reindexed_edge_pair(&mut adjacency, local_u, local_v, edge);
                }
            }
        }
        for &global in vertices {
            local_of[global as usize] = usize::MAX;
        }
        let local_diagonal = vertices
            .iter()
            .map(|&vertex| diagonal[vertex as usize])
            .collect();
        (
            Self {
                eliminated: BitVec::new(vertices.len()),
                adj: adjacency,
                _marker: core::marker::PhantomData,
            },
            local_diagonal,
        )
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

    fn add_reindexed_edge_pair(adj: &mut [Vec<E>], u: usize, v: usize, edge: E) {
        assert!(
            adj[u].len() < u32::MAX as usize && adj[v].len() < u32::MAX as usize,
            "adjacency list exceeds u32 edge capacity"
        );
        let rev_u = adj[v].len() as u32;
        let rev_v = adj[u].len() as u32;
        adj[u].push(edge.reindex(v as u32, rev_u));
        adj[v].push(edge.reindex(u as u32, rev_v));
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

    /// Connected components among the first `n_real` vertices, or `None` when the
    /// graph is connected. Traversal follows every edge, so a ground vertex (index
    /// `>= n_real`) links the blocks it touches without being counted as its own
    /// component.
    fn components(adj: &[Vec<E>], n_real: usize) -> Option<Vec<Vec<u32>>> {
        let n = adj.len();
        let mut visited = BitVec::new(n);
        let mut stack: Vec<usize> = Vec::new();
        // One traversal reaches every vertex iff the graph is connected, so the
        // common case never materializes or sorts a component list.
        let mut reached = 0usize;
        if n_real > 0 {
            visited.set(0);
            stack.push(0);
            while let Some(v) = stack.pop() {
                reached += 1;
                for e in &adj[v] {
                    let u = e.to() as usize;
                    if !visited.get(u) {
                        visited.set(u);
                        stack.push(u);
                    }
                }
            }
        }
        if reached == n {
            return None;
        }

        let mut visited = BitVec::new(n);
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
        Some(components)
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

impl<E: EdgeLike<T>, T: Real> AdjListGraph<E, T> {
    /// Mark each edge as `k` virtual copies at `weight / k`; no-op for slim edges.
    /// Approximate path only: `weight / k` underflows a subnormal to zero, which
    /// `fill_weight`'s `weight * count` cannot recover, so the exact dense
    /// assembly must see an unsplit graph.
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
                *edge = edge.split(inv_k, k);
            }
        }
    }
}
