//! Factor-width-2 views `A = V W Vᵀ`, at most two nonzeros per column of `V`, on two disjoint sides.

use std::sync::OnceLock;

use num_traits::PrimInt;

use crate::storage::Biadjacency;
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
    /// More vertices than a `u32` index reaches.
    TooLarge {
        /// The vertex count that overflows.
        count: usize,
    },
    /// A Gram's two sides read different `W` slices.
    WeightsDiffer,
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "view slice has length {found}, expected {expected}")
            }
            Self::TooLarge { count } => write!(f, "{count} vertices exceed a u32 index"),
            Self::WeightsDiffer => write!(f, "a Gram's sides read different weights"),
        }
    }
}

impl std::error::Error for ViewError {}

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

impl<T> Weights<'_, T> {
    /// The same slice read the same way: contents are never compared, so a NaN cannot split a family.
    fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::Unit, Self::Unit) => true,
            (Self::Of(a), Self::Of(b)) | (Self::SqrtOf(a), Self::SqrtOf(b)) => core::ptr::eq(a, b),
            _ => false,
        }
    }
}

/// Weighted bipartite multigraph; its matrix is the signless Laplacian `B W Bᵀ`, left rows first.
#[derive(Clone, Copy, Debug)]
pub struct Graph<'a, T, I = u32> {
    columns: Columns<'a, T, I>,
}

impl<'a, T: Real, I: PrimInt> Graph<'a, T, I> {
    /// Column `k` joins `left[k] < n_left` and `right[k] < n_right` under `weights`, unchecked.
    pub fn new(
        n_left: usize,
        n_right: usize,
        left: &'a [I],
        right: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        todo!()
    }

    pub(crate) fn columns(&self) -> Columns<'a, T, I> {
        self.columns
    }
}

/// Sides of `V` sharing one row vector, one per loading; their diagonals are summed together, once.
#[derive(Debug)]
pub struct Sides<'a, T, I = u32> {
    n: usize,
    rows: &'a [I],
    loadings: Vec<Option<&'a [T]>>,
    weights: Weights<'a, T>,
    diagonals: OnceLock<Box<[T]>>,
}

impl<'a, T: Real, I: PrimInt> Sides<'a, T, I> {
    /// `n` rows; side `c` loads column `k` with `loadings[c][k]`, `1` for `None`; rows unchecked.
    pub fn new(
        n: usize,
        rows: &'a [I],
        loadings: Vec<Option<&'a [T]>>,
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        todo!()
    }

    /// Side `c`, borrowing this family; panics unless the family has it.
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

impl<T, I> Clone for Side<'_, '_, T, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, I> Copy for Side<'_, '_, T, I> {}

/// `A = V W Vᵀ` over two borrowed sides, so one family serves every pair it is in.
#[derive(Debug)]
pub struct Gram<'s, 'a, T, I = u32> {
    left: Side<'s, 'a, T, I>,
    right: Side<'s, 'a, T, I>,
}

impl<T, I> Clone for Gram<'_, '_, T, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, I> Copy for Gram<'_, '_, T, I> {}

impl<'s, 'a, T: Real, I: PrimInt> Gram<'s, 'a, T, I> {
    /// Both sides must have the same column count and read the same `W`.
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

/// Every pass over `V`: its columns, each joining one left row to one right row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Columns<'a, T, I> {
    n_left: usize,
    n_right: usize,
    left: &'a [I],
    right: &'a [I],
    left_load: Option<&'a [T]>,
    right_load: Option<&'a [T]>,
    weights: Weights<'a, T>,
}

/// Column `k` of `V`: `v_l e_left + v_r e_right` under weight `w`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Observation<T> {
    pub(crate) left: u32,
    pub(crate) right: u32,
    pub(crate) v_left: T,
    pub(crate) v_right: T,
    pub(crate) weight: T,
}

impl<T: Real, I: PrimInt> Columns<'_, T, I> {
    pub(crate) fn n_left(&self) -> usize {
        self.n_left
    }

    pub(crate) fn observations(&self) -> impl Iterator<Item = Observation<T>> + '_ {
        (0..self.left.len()).map(|k| -> Observation<T> { todo!() })
    }

    /// `C`, accumulated in a dense table when it is small: within measured +54% without it.
    pub(crate) fn assemble(&self) -> Biadjacency<T> {
        todo!()
    }
}
