//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, so `A` numbers left rows first, then `n_left +` right ones.

use std::sync::OnceLock;

use num_traits::PrimInt;

use super::{Column, IntoColumns, ViewError};
use crate::Real;

/// The diagonal `W`, one nonnegative entry per column of `V`.
#[derive(Clone, Copy, Debug)]
pub enum Weights<'a, T> {
    /// `W = I`.
    Unit,
    /// `W = diag(w)`.
    Of(&'a [T]),
    /// `W = diag(s²)`, for a caller that keeps square roots.
    SqrtOf(&'a [T]),
}

impl<T: Real> Weights<'_, T> {
    /// One entry per column, each `w` finite and nonnegative or `s` finite; both views check it in `new`.
    fn check(self, columns: usize) -> Result<(), ViewError> {
        todo!()
    }

    /// The same slice read the same way: contents are never compared, so a NaN cannot split two sides of one Gram.
    fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::Unit, Self::Unit) => true,
            (Self::Of(a), Self::Of(b)) | (Self::SqrtOf(a), Self::SqrtOf(b)) => core::ptr::eq(a, b),
            _ => false,
        }
    }
}

/// Which row `rows[k]` each column of `V` hits, under `W`, with one value vector per side `V_c = P diag(values_c)`; each side's diagonal is summed once.
#[derive(Debug)]
pub struct Incidence<'a, T, I = u32> {
    n: usize,
    rows: &'a [I],
    values: Vec<Option<&'a [T]>>,
    weights: Weights<'a, T>,
    diagonals: OnceLock<Box<[T]>>,
}

impl<'a, T: Real, I: PrimInt> Incidence<'a, T, I> {
    /// `n` rows; side `c` puts `values[c][k]` in column `k`, `1` for `None`; rows, weights and each nonzero `w·v²` in range are checked.
    pub fn new(
        n: usize,
        rows: &'a [I],
        values: Vec<Option<&'a [T]>>,
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        todo!()
    }

    /// Side `c`, borrowing this incidence; panics unless it has one.
    pub fn side(&self, c: usize) -> Side<'_, T, I> {
        todo!()
    }
}

/// One side `V_c` of a bipartite Gram, borrowed from its incidence: one nonzero per column.
#[derive(Debug)]
pub struct Side<'a, T, I = u32> {
    incidence: &'a Incidence<'a, T, I>,
    index: usize,
}

impl<T, I> Clone for Side<'_, T, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, I> Copy for Side<'_, T, I> {}

/// `A = V W Vᵀ` with `V = [V_l; V_r]`; `W` stays apart because every pair of an incidence's sides shares it.
#[derive(Debug)]
pub struct BipartiteGram<'a, T, I = u32> {
    left: Side<'a, T, I>,
    right: Side<'a, T, I>,
}

impl<T, I> Clone for BipartiteGram<'_, T, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, I> Copy for BipartiteGram<'_, T, I> {}

impl<'a, T: Real, I: PrimInt> BipartiteGram<'a, T, I> {
    /// Both sides must have the same column count and read the same `W`, and `n_left + n_right` fit a `u32` index.
    pub fn new(left: Side<'a, T, I>, right: Side<'a, T, I>) -> Result<Self, ViewError> {
        todo!()
    }

    pub(crate) fn n_left(&self) -> usize {
        todo!()
    }

    /// `A`'s diagonal, left rows then right ones; the first Gram of an incidence sums each side's, serially, as a rayon initializer deadlocks when Grams reduce in parallel.
    pub(crate) fn diagonals(&self) -> (&'a [T], &'a [T]) {
        todo!()
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for BipartiteGram<'_, T, I> {
    fn n(&self) -> usize {
        todo!()
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        todo!()
    }
}

/// Weighted bipartite multigraph; its matrix is the signless Laplacian `|B| W |B|ᵀ`, singular by shape.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteGraph<'a, T, I = u32> {
    n_left: usize,
    n_right: usize,
    left: &'a [I],
    right: &'a [I],
    weights: Weights<'a, T>,
}

impl<'a, T: Real, I: PrimInt> BipartiteGraph<'a, T, I> {
    /// Column `k` joins `left[k] < n_left` and `right[k] < n_right` under `weights`; indices, weights and each nonzero `w` in range are checked, so `C ≥ 0` folds.
    pub fn new(
        n_left: usize,
        n_right: usize,
        left: &'a [I],
        right: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        todo!()
    }

    pub(crate) fn n_left(&self) -> usize {
        self.n_left
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for BipartiteGraph<'_, T, I> {
    fn n(&self) -> usize {
        self.n_left + self.n_right
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        todo!()
    }
}
