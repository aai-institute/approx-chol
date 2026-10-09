//! Factor-width-2 views `A = V W Vᵀ`, at most two nonzeros per column of `V`, on two disjoint sides.

use std::sync::OnceLock;

use num_traits::PrimInt;

use crate::storage::CrossBlock;
use crate::Real;

/// Why a view's `new` rejected its slices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewError {
    /// A slice's length differs from the column count.
    LengthMismatch {
        /// The column count the first slice fixed.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// A Gram's two sides were built under different weights.
    WeightsDiffer,
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid factor-width-2 view: {self:?}")
    }
}

impl std::error::Error for ViewError {}

/// The diagonal `W`, one nonnegative entry per column of `V`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Weights<'a, T> {
    /// `W = I`.
    Unit,
    /// `W = diag(w)`.
    Of(&'a [T]),
    /// `W = diag(s²)`, for a caller that keeps square roots.
    SqrtOf(&'a [T]),
}

/// Weighted bipartite multigraph; its matrix is the signless Laplacian `B W Bᵀ`, left rows first.
#[derive(Clone, Copy, Debug)]
pub struct Graph<'a, T, I = u32> {
    n_left: u32,
    n_right: u32,
    left: &'a [I],
    right: &'a [I],
    weights: Weights<'a, T>,
}

impl<'a, T: Real, I: PrimInt> Graph<'a, T, I> {
    /// Column `k` joins `left[k] < n_left` and `right[k] < n_right` under `weights`, unchecked.
    pub fn new(
        n_left: u32,
        n_right: u32,
        left: &'a [I],
        right: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        todo!()
    }

    pub(crate) fn sizes(&self) -> (usize, usize) {
        (self.n_left as usize, self.n_right as usize)
    }

    pub(crate) fn columns(&self) -> Columns<'a, T, I> {
        todo!()
    }
}

/// Sides of `V` sharing one row vector, one per loading; their diagonals are summed together, once.
#[derive(Debug)]
pub struct Sides<'a, T, I = u32> {
    n: u32,
    rows: &'a [I],
    loadings: Vec<Option<&'a [T]>>,
    weights: Weights<'a, T>,
    diagonals: OnceLock<Box<[T]>>,
}

impl<'a, T: Real, I: PrimInt> Sides<'a, T, I> {
    /// `n` rows; side `c` loads column `k` with `loadings[c][k]`, `1` for `None`; rows unchecked.
    pub fn new(
        n: u32,
        rows: &'a [I],
        loadings: Vec<Option<&'a [T]>>,
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        todo!()
    }

    /// Side `c`, borrowing this family.
    pub fn side(&self, c: usize) -> Side<'_, 'a, T, I> {
        todo!()
    }
}

/// One side of `V` under `W`, borrowed from its family.
#[derive(Debug)]
pub struct Side<'s, 'a, T, I = u32> {
    sides: &'s Sides<'a, T, I>,
    index: usize,
}

/// `A = V W Vᵀ` over two borrowed sides, so one family serves every pair it is in.
#[derive(Debug)]
pub struct Gram<'s, 'a, T, I = u32> {
    left: Side<'s, 'a, T, I>,
    right: Side<'s, 'a, T, I>,
}

impl<'s, 'a, T: Real, I: PrimInt> Gram<'s, 'a, T, I> {
    /// Both sides must have the same column count and the same `W`.
    pub fn new(left: Side<'s, 'a, T, I>, right: Side<'s, 'a, T, I>) -> Result<Self, ViewError> {
        todo!()
    }

    /// `A`'s diagonal, left rows then right ones; the first Gram of a family sums it.
    pub(crate) fn diagonals(&self) -> (&'s [T], &'s [T]) {
        todo!()
    }

    pub(crate) fn columns(&self) -> Columns<'a, T, I> {
        todo!()
    }
}

/// What every pass over `V` reads: each column's two rows, optional loadings and its weight.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Columns<'a, T, I> {
    left: &'a [I],
    right: &'a [I],
    left_load: Option<&'a [T]>,
    right_load: Option<&'a [T]>,
    weights: Weights<'a, T>,
}

impl<T: Real, I: PrimInt> Columns<'_, T, I> {
    /// `C`, accumulated in a dense table when it is small: within measured +54% without it.
    pub(crate) fn assemble(&self, n_left: usize, n_right: usize) -> CrossBlock<T> {
        todo!()
    }
}
