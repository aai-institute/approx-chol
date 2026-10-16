//! Factor-width-2 views: at most two nonzeros per column of `V`, read column by column.

pub(super) mod bipartite;
pub(super) mod edges;
pub(super) mod laplacian;
pub(super) mod signed;

use num_traits::PrimInt;

use crate::widen;
use crate::{Index, Real};

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
    /// A ground is negative or not finite.
    InvalidGround {
        /// The offending vertex.
        vertex: usize,
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
            Self::InvalidGround { vertex } => {
                write!(f, "vertex {vertex} has a negative or non-finite ground")
            }
        }
    }
}

impl std::error::Error for ViewError {}

/// `A = Σ_k w_k x_k x_kᵀ`, each `x_k` a column of `V` with at most two nonzeros.
pub(crate) trait IntoColumns<T: Real>: Copy {
    /// Dimension of `A`, not the column count.
    fn n(&self) -> usize;

    /// Each column in order; write `visit` as `#[inline(always)] |column| …` (outlined: 1.6-1.9x).
    fn for_each_column<I: Index>(self, visit: impl FnMut(Column<T, I>));
}

/// `V = [V_l; V_r]`: every column joins a left row to a right one, right rows numbered after the left.
pub(crate) trait BipartiteColumns<T: Real>: IntoColumns<T> {
    fn n_left(&self) -> usize;
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
        (self.weight).bilinear(widen(self.v_a), widen(self.v_b))
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

/// `V` with at most two nonzeros per column: `x[k]` at `a[k]`, `y[k]` at `b[k] ≠ a[k]`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth2<'a, T, I = u32> {
    n: usize,
    a: &'a [I],
    b: &'a [I],
    values: [&'a [T]; 2],
}

impl<'a, T: Real, I: PrimInt> FactorWidth2<'a, T, I> {
    /// `n` vertices; lengths and `a[k] ≠ b[k] < n` are checked, so a zero value leaves one nonzero.
    pub fn new(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        x: &'a [T],
        y: &'a [T],
    ) -> Result<Self, ViewError> {
        let columns = a.len();
        let lengths = [b.len(), x.len(), y.len()];
        if let Some(&found) = lengths.iter().find(|&&len| len != columns) {
            return Err(ViewError::LengthMismatch {
                expected: columns,
                found,
            });
        }
        check_ends(n, a, b)?;
        Ok(Self {
            n,
            a,
            b,
            values: [x, y],
        })
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

impl<T: Real, I: PrimInt> IntoColumns<T> for FactorWidth2<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    fn for_each_column<S: Index>(self, mut visit: impl FnMut(Column<T, S>)) {
        let [v_a, v_b] = self.values;
        for (k, (a, b)) in self.a.iter().zip(self.b).enumerate() {
            visit(Column {
                a: S::from_index(index(*a)),
                b: S::from_index(index(*b)),
                v_a: v_a[k],
                v_b: v_b[k],
                weight: Weighting::Unit,
            });
        }
    }
}

/// A row `new` checked below `n`.
fn index<I: PrimInt>(row: I) -> usize {
    row.to_usize().expect("a checked row")
}
