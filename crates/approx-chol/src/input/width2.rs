//! Factor-width-2 views: at most two nonzeros per column of `V`, read column by column.

pub(super) mod bipartite;
pub(super) mod edges;
pub(super) mod gram;
pub(super) mod laplacian;
pub(super) mod signed;

use num_traits::PrimInt;

use super::index;
use crate::widen;
use crate::{Index, Real};
use edges::Weights;
use gram::{Gram, Shape};

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
    /// A diagonal entry `σ_i`, a ground or a ridge, is negative or not finite.
    InvalidDiagonal {
        /// The offending vertex.
        vertex: usize,
    },
    /// A bipartite shape's left part has more rows than `A`.
    LeftPartTooLarge {
        /// The left part's size.
        n_left: usize,
        /// `A`'s dimension.
        n: usize,
    },
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "view slice has length {found}, expected {expected}")
            }
            Self::IndexOutOfRange { column } => write!(f, "column {column} names a missing row"),
            Self::RepeatedVertex { column } => write!(f, "column {column} names one vertex twice"),
            Self::InvalidWeight { column } => {
                write!(f, "column {column} has a negative or non-finite weight")
            }
            Self::InvalidDiagonal { vertex } => {
                write!(
                    f,
                    "vertex {vertex} has a negative or non-finite diagonal entry"
                )
            }
            Self::LeftPartTooLarge { n_left, n } => {
                write!(f, "left part of {n_left} rows exceeds dimension {n}")
            }
        }
    }
}

impl std::error::Error for ViewError {}

/// `V W Vᵀ = Σ_k w_k x_k x_kᵀ`, each `x_k` a column of `V` with at most two nonzeros; `σ` is never a column.
pub(crate) trait IntoColumns<T: Real>: Copy {
    /// Dimension of `A`, not the column count.
    fn n(&self) -> usize;

    /// Each column in order; write `visit` as `#[inline(always)] |column| …` (outlined: 1.6-1.9x).
    fn for_each_column<I: Index>(self, visit: impl FnMut(Column<T, I>));
}

/// Column `k` of `V` under its weight, unfolded, so the floating test reads `V` exactly as given.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Column<T, I> {
    pub(crate) a: I,
    pub(crate) b: I,
    pub(crate) v_a: T,
    pub(crate) v_b: T,
    pub(crate) weight: Weighting<T>,
}

impl<T: Real, I> Column<T, I> {
    /// `w·v_a·v_b` in `f64`: every assembly sums its cells there and casts once.
    pub(crate) fn cell(self) -> f64 {
        (self.weight).bilinear_unflushed(widen(self.v_a), widen(self.v_b))
    }
}

/// The diagonal `W`: per column of `V` as a slice, or one column's entry.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Weighting<S> {
    /// `W = I`.
    Unit,
    /// `W = diag(w)`.
    Of(S),
    /// `W = diag(s²)`, for a caller that keeps square roots.
    SqrtOf(S),
}

impl<T: Real> Weighting<T> {
    /// `w·x·y`, as `(s·x)·(s·y)` under `SqrtOf`.
    pub(crate) fn bilinear(self, x: f64, y: f64) -> f64 {
        match self {
            Self::Unit => x * y,
            Self::Of(w) => widen(w) * x * y,
            Self::SqrtOf(s) => (widen(s) * x) * (widen(s) * y),
        }
    }

    /// As `bilinear`, a nonzero product that underflows kept as the least subnormal, so the range screen rejects it (#163).
    pub(crate) fn bilinear_unflushed(self, x: f64, y: f64) -> f64 {
        let product = self.bilinear(x, y);
        match product == 0.0 && x != 0.0 && y != 0.0 && !self.is_zero() {
            true => f64::from_bits(1).copysign(product),
            false => product,
        }
    }

    pub(crate) fn is_zero(self) -> bool {
        match self {
            Self::Unit => false,
            Self::Of(w) | Self::SqrtOf(w) => w == T::zero(),
        }
    }
}

/// `A = V W Vᵀ`, `V` with at most two nonzeros per column: `x[k]` at `a[k]`, `y[k]` at `b[k] ≠ a[k]`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth2<'a, T, I = u32>(Gram<'a, T, I>);

impl<'a, T: Real, I: PrimInt> FactorWidth2<'a, T, I> {
    /// `n` vertices; lengths and `a[k] ≠ b[k] < n` are checked, so a zero value leaves one nonzero.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        x: &'a [T],
        y: &'a [T],
    ) -> Result<Self, ViewError> {
        Gram::values(n, Shape::General, a, b, Some(x), Some(y)).map(Self)
    }

    /// `V W Vᵀ` under `weights`, one per column; `W = I` without it.
    pub fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        self.0.with_weights(weights).map(Self)
    }

    /// `V W Vᵀ + diag(d)`, one finite nonnegative `d_i` per vertex.
    pub fn plus_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        self.0.plus_diagonal(diagonal).map(Self)
    }
}

impl<'a, T, I> From<FactorWidth2<'a, T, I>> for Gram<'a, T, I> {
    fn from(v: FactorWidth2<'a, T, I>) -> Self {
        v.0
    }
}

/// A slice of `found` entries where the view fixed `expected`.
fn length(expected: usize, found: usize) -> Result<(), ViewError> {
    match found == expected {
        true => Ok(()),
        false => Err(ViewError::LengthMismatch { expected, found }),
    }
}

/// The first entry `valid` rejects, reported by `error` at its index.
fn check<T: Real>(
    entries: &[T],
    valid: fn(T) -> bool,
    error: fn(usize) -> ViewError,
) -> Result<(), ViewError> {
    match entries.iter().position(|&x| !valid(x)) {
        Some(at) => Err(error(at)),
        None => Ok(()),
    }
}

/// Each column's two ends are distinct rows below `n`.
fn check_ends<I: PrimInt>(n: usize, a: &[I], b: &[I]) -> Result<(), ViewError> {
    let in_range = |end: &I| end.to_usize().is_some_and(|end| end < n);
    for (column, (end_a, end_b)) in a.iter().zip(b).enumerate() {
        if !in_range(end_a) || !in_range(end_b) {
            return Err(ViewError::IndexOutOfRange { column });
        }
        if end_a == end_b {
            return Err(ViewError::RepeatedVertex { column });
        }
    }
    Ok(())
}
