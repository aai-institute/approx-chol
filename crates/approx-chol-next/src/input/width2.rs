//! Factor-width-2 views, at most two nonzeros per column of `V`: `A = V Vᵀ` in general, or `V W Vᵀ` on two disjoint sides.

use std::sync::OnceLock;

use num_traits::PrimInt;

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
    /// A column names a row its side does not have.
    IndexOutOfRange {
        /// The offending column of `V`.
        column: usize,
    },
    /// A column names one vertex at both ends.
    RepeatedVertex {
        /// The offending column of `V`.
        column: usize,
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
            Self::IndexOutOfRange { column } => write!(f, "column {column} names a missing row"),
            Self::RepeatedVertex { column } => write!(f, "column {column} names one vertex twice"),
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
    /// The same slice read the same way: contents are never compared, so a NaN cannot split two sides of one Gram.
    fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::Unit, Self::Unit) => true,
            (Self::Of(a), Self::Of(b)) | (Self::SqrtOf(a), Self::SqrtOf(b)) => core::ptr::eq(a, b),
            _ => false,
        }
    }
}

/// `V` with at most two nonzeros per column, by construction: `values[0][k]` at vertex `a[k]` and `values[1][k]` at `b[k] ≠ a[k]`; fold any weights into `V`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth2<'a, T, I = u32> {
    columns: Columns<'a, T, I>,
}

impl<'a, T: Real, I: PrimInt> FactorWidth2<'a, T, I> {
    /// `n` vertices; lengths, `a[k], b[k] < n` and `a[k] ≠ b[k]` are checked, so a zero value is how a column holds one nonzero.
    pub fn new(n: usize, a: &'a [I], b: &'a [I], values: [&'a [T]; 2]) -> Result<Self, ViewError> {
        todo!()
    }

    pub(crate) fn columns(&self) -> Columns<'a, T, I> {
        self.columns
    }
}

/// Weighted bipartite multigraph; its matrix is the signless Laplacian `|B| W |B|ᵀ`, left rows first, singular by shape.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteGraph<'a, T, I = u32> {
    columns: Columns<'a, T, I>,
}

impl<'a, T: Real, I: PrimInt> BipartiteGraph<'a, T, I> {
    /// Column `k` joins `left[k] < n_left` and `right[k] < n_right` under `weights`; indices are checked.
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
    /// `n` rows; side `c` puts `values[c][k]` in column `k`, `1` for `None`; rows are checked.
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

/// `A = V W Vᵀ`, `V = [V_l; V_r]` on disjoint sides; `W` stays apart because every pair of an incidence's sides shares it, and `A` numbers left rows first, then `n_left +` right ones.
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

    /// `A`'s diagonal, left rows then right ones; the first Gram of an incidence sums each side's.
    pub(crate) fn diagonals(&self) -> (&'a [T], &'a [T]) {
        todo!()
    }

    pub(crate) fn columns(&self) -> Columns<'a, T, I> {
        todo!()
    }
}

/// Every pass over a factor-width-2 `V`: each column's two ends in `A`'s numbering, a bipartite view's `b` already offset by `n_left`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Columns<'a, T, I> {
    n: usize,
    /// `Some(n_left)` when every column joins a left row to a right one.
    n_left: Option<usize>,
    a: &'a [I],
    b: &'a [I],
    values: [Option<&'a [T]>; 2],
    weights: Weights<'a, T>,
}

/// Column `k` of `V`: `v_a e_a + v_b e_b` under its weight, unfolded, so the floating forest reads `V`'s values exactly as given.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Column<T> {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) v_a: T,
    pub(crate) v_b: T,
    pub(crate) weight: Weight<T>,
}

/// One column's entry of `W`, as the caller gave it; read only through [`bilinear`](Self::bilinear), so no `s²` is ever formed.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Weight<T> {
    Of(T),
    SqrtOf(T),
}

impl<T: Real> Weight<T> {
    /// `w·x·y` in `U`, as `(s·x)·(s·y)` under `SqrtOf`: every diagonal, cell and residue goes through it.
    pub(crate) fn bilinear<U: Real>(self, x: U, y: U) -> U {
        let to = |w: T| U::from(w).expect("a finite weight casts to a float");
        match self {
            Self::Of(w) => to(w) * x * y,
            Self::SqrtOf(s) => (to(s) * x) * (to(s) * y),
        }
    }

    fn is_zero(self) -> bool {
        match self {
            Self::Of(w) | Self::SqrtOf(w) => w == T::zero(),
        }
    }
}

impl<T: Real> Column<T> {
    /// Nonzero at both ends, so the floating forest may grow along it; tested factor by factor so no product underflows.
    pub(crate) fn couples(&self) -> bool {
        !self.weight.is_zero() && self.v_a != T::zero() && self.v_b != T::zero()
    }
}

impl<T: Real, I: PrimInt> Columns<'_, T, I> {
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    pub(crate) fn n_left(&self) -> Option<usize> {
        self.n_left
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = Column<T>> + '_ {
        (0..self.a.len()).map(|k| -> Column<T> { todo!() })
    }
}
