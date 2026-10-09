//! Sparse storage shapes every stage reads.

use std::sync::OnceLock;

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

impl<T> Rows<T> {
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

    fn transpose(&self) -> Self
    where
        T: Copy,
    {
        todo!()
    }

    /// `x_ij ← r_i x_ij c_j`.
    fn scaled(self, r: &[T], c: &[T]) -> Self
    where
        T: num_traits::Float,
    {
        todo!()
    }
}

/// A symmetric off-diagonal's strict upper triangle by rows: each edge once.
#[derive(Clone, Debug)]
pub(crate) struct UpperAdjacency<T> {
    rows: Rows<T>,
}

impl<T> UpperAdjacency<T> {
    /// Panics unless square with every column above its row.
    pub(crate) fn new(rows: Rows<T>) -> Self {
        todo!()
    }

    pub(crate) fn n(&self) -> usize {
        self.rows.n_rows()
    }

    /// Row `i`'s neighbors above it.
    pub(crate) fn upper(&self, i: usize) -> impl Iterator<Item = (usize, &T)> {
        self.rows.row(i)
    }

    /// `F X F`.
    pub(crate) fn scaled(self, f: &[T]) -> Self
    where
        T: num_traits::Float,
    {
        Self {
            rows: self.rows.scaled(f, f),
        }
    }
}

/// The biadjacency `C` of a factor-width-2 view as assembled: rows are `A`'s left vertices.
#[derive(Clone, Debug)]
pub(crate) struct Biadjacency<T> {
    c: Rows<T>,
}

impl<T: Copy> Biadjacency<T> {
    pub(crate) fn new(c: Rows<T>) -> Self {
        Self { c }
    }

    pub(crate) fn n_left(&self) -> usize {
        self.c.n_rows()
    }

    pub(crate) fn n_right(&self) -> usize {
        self.c.n_cols()
    }

    /// One component's block, its larger side as rows; returns the vertices in that local order.
    pub(crate) fn block(&self, vertices: &[u32]) -> (Box<[u32]>, CrossBlock<T>) {
        todo!()
    }
}

/// One component's `C` of `[D₁ C; Cᵀ D₂]`, rows the eliminated side, `Cᵀ` built on first read.
#[derive(Clone, Debug)]
pub(crate) struct CrossBlock<T> {
    c: Rows<T>,
    ct: OnceLock<Rows<T>>,
}

impl<T: Copy> CrossBlock<T> {
    /// Panics unless both sides are nonempty: a lone vertex is no bipartite component.
    fn new(c: Rows<T>) -> Self {
        assert!(c.n_rows() > 0 && c.n_cols() > 0, "both sides hold a vertex");
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

    /// `F C F` over the block's local order; a fresh block, so no stale `Cᵀ` survives.
    pub(crate) fn scaled(self, f: &[T]) -> Self
    where
        T: num_traits::Float,
    {
        let (eliminated, kept) = f.split_at(self.n_eliminated());
        Self::new(self.c.scaled(eliminated, kept))
    }
}
