//! Sparse storage shapes every stage reads.

use std::sync::OnceLock;

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

/// `C` of `[D₁ C; Cᵀ D₂]`, rows the larger side, and `Cᵀ` built on first read.
#[derive(Clone, Debug)]
pub(crate) struct CrossBlock<T> {
    c: Rows<T>,
    ct: OnceLock<Rows<T>>,
}

impl<T: Copy> CrossBlock<T> {
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
