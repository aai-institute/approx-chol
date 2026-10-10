//! Factor-width-2 views, at most two nonzeros per column of `V`, and the one reading of them `sgdd` needs: column by column.

pub(super) mod bipartite;

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
    /// A column names a row its view does not have.
    IndexOutOfRange {
        /// The offending column of `V`.
        column: usize,
    },
    /// A column names one vertex at both ends.
    RepeatedVertex {
        /// The offending column of `V`.
        column: usize,
    },
    /// A weight is negative or not finite.
    InvalidWeight {
        /// The offending column of `V`.
        column: usize,
    },
    /// A bipartite Gram's two sides read different `W` slices.
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
            Self::InvalidWeight { column } => {
                write!(f, "column {column} has a negative or non-finite weight")
            }
            Self::WeightsDiffer => write!(f, "a Gram's sides read different weights"),
        }
    }
}

impl std::error::Error for ViewError {}

/// `A = Σ_k w_k x_k x_kᵀ`, each `x_k` a column of `V` with at most two nonzeros; a cheap `Copy` view, consumed by visiting its columns.
pub(crate) trait IntoColumns<T: Real>: Copy {
    /// Dimension of `A`, not the column count.
    fn n(&self) -> usize;

    /// In column order, each column's ends in `A`'s numbering; a view matches its values and weights once, then loops per case, and a caller writes `visit` inline as `#[inline(always)] |column| …`, or an outlined visitor pays the branch per column (1.6-1.9x).
    fn for_each_column(self, visit: impl FnMut(Column<T>));
}

/// Column `k` of `V`: `v_a e_a + v_b e_b` under its weight, unfolded, so the floating test reads `V`'s values exactly as given.
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
    /// A view without `W`.
    Unit,
    Of(T),
    SqrtOf(T),
}

impl<T: Real> Weight<T> {
    /// `w·x·y` in `U`, as `(s·x)·(s·y)` under `SqrtOf`: every diagonal, cell and residue goes through it.
    pub(crate) fn bilinear<U: Real>(self, x: U, y: U) -> U {
        let to = |w: T| U::from(w).expect("a finite weight casts to a float");
        match self {
            Self::Unit => x * y,
            Self::Of(w) => to(w) * x * y,
            Self::SqrtOf(s) => (to(s) * x) * (to(s) * y),
        }
    }

    pub(crate) fn is_zero(self) -> bool {
        match self {
            Self::Unit => false,
            Self::Of(w) | Self::SqrtOf(w) => w == T::zero(),
        }
    }
}

/// `V` with at most two nonzeros per column, by construction: `values[0][k]` at vertex `a[k]` and `values[1][k]` at `b[k] ≠ a[k]`; fold any weights into `V`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth2<'a, T, I = u32> {
    n: usize,
    a: &'a [I],
    b: &'a [I],
    values: [&'a [T]; 2],
}

impl<'a, T: Real, I: PrimInt> FactorWidth2<'a, T, I> {
    /// `n` vertices; lengths, `a[k], b[k] < n` and `a[k] ≠ b[k]` are checked, so a zero value is how a column holds one nonzero.
    pub fn new(n: usize, a: &'a [I], b: &'a [I], values: [&'a [T]; 2]) -> Result<Self, ViewError> {
        todo!()
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for FactorWidth2<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        todo!()
    }
}
