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

    pub(crate) fn transpose(&self) -> Self
    where
        T: Copy,
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
    pub(crate) fn n(&self) -> usize {
        self.rows.n_rows()
    }
}

/// `C` of a factor-width-2 view as assembled: rows are `A`'s left vertices, columns its right ones.
#[derive(Clone, Debug)]
pub(crate) struct CrossTab<T> {
    c: Rows<T>,
}

impl<T> CrossTab<T> {
    pub(crate) fn n_left(&self) -> usize {
        self.c.n_rows()
    }

    pub(crate) fn n_right(&self) -> usize {
        self.c.n_cols()
    }
}

/// One component's `C` of `[D₁ C; Cᵀ D₂]`, oriented so rows are the larger side, `Cᵀ` built on first read.
#[derive(Clone, Debug)]
pub(crate) struct CrossBlock<T> {
    c: Rows<T>,
    ct: OnceLock<Rows<T>>,
}

impl<T: Copy> CrossBlock<T> {
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
}
