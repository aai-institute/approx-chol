//! Sparse storage shapes every stage reads.

use std::sync::OnceLock;

use crate::Real;

/// Finite, and zero or at least `MIN_POSITIVE/EPSILON`: the 8-bit margin keeps every product out of subnormals (#163).
pub(crate) fn in_range<T: Real>(x: T) -> bool {
    x.is_finite() && (x == T::zero() || x.abs() >= T::min_positive_value() / T::epsilon())
}

/// Fill edges `(lo, hi, weight)`, `lo < hi`, as an elimination emits them.
pub(crate) type Fill<T> = Vec<(u32, u32, T)>;

/// Compressed sparse rows.
#[derive(Clone, Debug)]
pub(crate) struct Rows<T> {
    ptrs: Vec<u32>,
    indices: Vec<u32>,
    values: Vec<T>,
    n_cols: usize,
}

impl<T: Real> Rows<T> {
    /// Panics unless `ptrs` ascends from `0` to the entry count and every index is below `n_cols`.
    pub(crate) fn new(ptrs: Vec<u32>, indices: Vec<u32>, values: Vec<T>, n_cols: usize) -> Self {
        todo!()
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.ptrs.len() - 1
    }

    pub(crate) fn n_cols(&self) -> usize {
        self.n_cols
    }

    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        let span = self.ptrs[i] as usize..self.ptrs[i + 1] as usize;
        (self.indices[span.clone()].iter().map(|&j| j as usize)).zip(&self.values[span])
    }

    fn transpose(&self) -> Self {
        todo!()
    }

    /// `x_ij ← r_i x_ij c_j`; `None` if a product leaves [`in_range`] or reaches zero: no entry is stored zero.
    fn scaled(self, r: &[T], c: &[T]) -> Option<Self> {
        todo!()
    }
}

/// A symmetric off-diagonal's strict upper triangle by rows: each edge once.
#[derive(Clone, Debug)]
pub(crate) struct UpperAdjacency<T> {
    rows: Rows<T>,
}

impl<T: Real> UpperAdjacency<T> {
    /// Panics unless square with every column above its row and no stored zero.
    pub(crate) fn new(rows: Rows<T>) -> Self {
        todo!()
    }

    pub(crate) fn n(&self) -> usize {
        self.rows.n_rows()
    }

    /// Counting-sorts the tasks' chunks by `lo` without flattening (~10% at 1M edges), sums duplicates, drops zeros.
    pub(crate) fn from_edges(n: usize, chunks: Vec<Fill<T>>) -> Self {
        todo!()
    }

    /// `Σ_j |x_ij|` over both triangles.
    pub(crate) fn abs_row_sums(&self) -> Vec<T> {
        todo!()
    }

    pub(crate) fn nonpositive(&self) -> bool {
        todo!()
    }

    /// Row `i`'s neighbors above it.
    pub(crate) fn upper(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        self.rows.row(i)
    }

    /// `F X F`; `None` if an entry leaves [`in_range`].
    pub(crate) fn scaled(self, f: &[T]) -> Option<Self> {
        let rows = self.rows.scaled(f, f)?;
        Some(Self { rows })
    }
}

/// The biadjacency `C` of a factor-width-2 view, cancelled cells dropped: rows are `A`'s left vertices.
#[derive(Clone, Debug)]
pub(crate) struct Biadjacency<T> {
    c: Rows<T>,
}

impl<T: Real> Biadjacency<T> {
    /// Panics on a stored zero: a cancelled cell joins nothing.
    pub(crate) fn new(c: Rows<T>) -> Self {
        todo!()
    }

    pub(crate) fn n(&self) -> usize {
        self.c.n_rows() + self.c.n_cols()
    }

    /// Each stored cell as an edge of `A`: left row `i` is vertex `i`, right column `j` is `n_left + j`.
    pub(crate) fn joins(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let n_left = self.c.n_rows();
        (0..n_left).flat_map(move |i| self.c.row(i).map(move |(j, _)| (i, n_left + j)))
    }

    /// One component's block, its larger side as rows, remapped through `position`; returns its local order.
    pub(crate) fn block(&self, vertices: &[u32], position: &[u32]) -> (Box<[u32]>, CrossBlock<T>) {
        todo!()
    }
}

/// One component's `C` of `[D₁ C; Cᵀ D₂]`, rows the eliminated side, `Cᵀ` built on first read.
#[derive(Clone, Debug)]
pub(crate) struct CrossBlock<T> {
    c: Rows<T>,
    ct: OnceLock<Rows<T>>,
}

impl<T: Real> CrossBlock<T> {
    /// Panics unless both sides are nonempty, with no stored zero: a lone vertex is no bipartite component.
    fn new(c: Rows<T>) -> Self {
        assert!(c.n_rows() > 0 && c.n_cols() > 0, "both sides hold a vertex");
        assert!(!c.values.contains(&T::zero()), "no stored zero");
        Self {
            c,
            ct: OnceLock::new(),
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.n_eliminated() + self.n_kept()
    }

    pub(crate) fn n_eliminated(&self) -> usize {
        self.c.n_rows()
    }

    pub(crate) fn n_kept(&self) -> usize {
        self.c.n_cols()
    }

    pub(crate) fn c(&self) -> &Rows<T> {
        &self.c
    }

    pub(crate) fn ct(&self) -> &Rows<T> {
        self.ct.get_or_init(|| self.c.transpose())
    }

    /// `Σ|c|` per row of `[0 C; Cᵀ 0]`: eliminated rows, then kept ones.
    pub(crate) fn abs_row_sums(&self) -> Vec<T> {
        todo!()
    }

    pub(crate) fn nonpositive(&self) -> bool {
        todo!()
    }

    /// `F C F` over the block's local order, a fresh block so no stale `Cᵀ` survives; `None` out of range.
    pub(crate) fn scaled(self, f: &[T]) -> Option<Self> {
        let (eliminated, kept) = f.split_at(self.n_eliminated());
        Some(Self::new(self.c.scaled(eliminated, kept)?))
    }
}
