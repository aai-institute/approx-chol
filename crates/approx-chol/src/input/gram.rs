//! `A = V W Vᵀ + diag(d)`: column `k` of `V` joins `a[k]` and `b[k]`.

use num_traits::PrimInt;

use super::index;
use crate::{widen, Index, Real};

/// `V`'s columns and vertices, one variant per constructor.
#[derive(Clone, Copy, Debug)]
enum Columns<'a, T> {
    /// `e_a − e_b`, `a[k] ≠ b[k] < n`.
    Incidence { n: usize },
    /// `x[k] e_a + y[k] e_b`, `a[k] ≠ b[k] < n`.
    FactorWidth2 { n: usize, x: &'a [T], y: &'a [T] },
    /// `x[k] e_a + y[k] e_(n_left + b)`, a missing side `1`; with neither, read switched as `e_a − e_b`.
    Bipartite {
        n_left: usize,
        n_right: usize,
        x: Option<&'a [T]>,
        y: Option<&'a [T]>,
    },
}

/// `A = V W Vᵀ + diag(d)` on `n` vertices.
#[derive(Clone, Copy, Debug)]
pub struct Gram<'a, T, I = u32> {
    a: &'a [I],
    b: &'a [I],
    weights: Weights<'a, T>,
    columns: Columns<'a, T>,
    diagonal: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> Gram<'a, T, I> {
    /// Graph Laplacian: edge `k` is `e_a − e_b`, `a[k] ≠ b[k] < n`.
    pub fn laplacian(n: usize, a: &'a [I], b: &'a [I]) -> Result<Self, ViewError> {
        check_ends(n, a, b)?;
        Ok(Self::of(a, b, Columns::Incidence { n }))
    }

    /// Column `k` is `x[k] e_a + y[k] e_b`, `a[k] ≠ b[k] < n`.
    pub fn factor_width_2(
        n: usize,
        a: &'a [I],
        b: &'a [I],
        x: &'a [T],
        y: &'a [T],
    ) -> Result<Self, ViewError> {
        length(a.len(), x.len())?;
        length(a.len(), y.len())?;
        check_ends(n, a, b)?;
        Ok(Self::of(a, b, Columns::FactorWidth2 { n, x, y }))
    }

    /// `V = [V_l; V_r]`, both width 1 on disjoint rows, left rows first; the sides share the column count.
    pub fn bipartite(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
    ) -> Result<Self, ViewError> {
        length(left.rows.len(), right.rows.len())?;
        let columns = Columns::Bipartite {
            n_left: left.n,
            n_right: right.n,
            x: left.values,
            y: right.values,
        };
        Ok(Self::of(left.rows, right.rows, columns))
    }

    fn of(a: &'a [I], b: &'a [I], columns: Columns<'a, T>) -> Self {
        Self {
            a,
            b,
            weights: Weights(Weighting::Unit),
            columns,
            diagonal: None,
        }
    }

    /// `V W Vᵀ` under `weights`, one per column.
    pub fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        if let Weighting::Of(w) | Weighting::SqrtOf(w) = weights.0 {
            length(self.a.len(), w.len())?;
        }
        Ok(Self { weights, ..self })
    }

    /// `V W Vᵀ + diag(d)`, one finite nonnegative `d_i` per vertex, replacing any earlier `d`.
    pub fn with_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        length(self.n(), diagonal.len())?;
        let valid = |d: T| d.is_finite() && d >= T::zero();
        check(diagonal, valid, |vertex| ViewError::InvalidDiagonal {
            vertex,
        })?;
        Ok(Self {
            diagonal: Some(diagonal),
            ..self
        })
    }

    /// The left part's size, after which `b` is numbered, when `V` is bipartite.
    pub(crate) fn n_left(&self) -> Option<usize> {
        match self.columns {
            Columns::Bipartite { n_left, .. } => Some(n_left),
            Columns::Incidence { .. } | Columns::FactorWidth2 { .. } => None,
        }
    }

    /// `d`, which is no column of `V`.
    pub(crate) fn diagonal(&self) -> Option<&'a [T]> {
        self.diagonal
    }

    /// D: `V` is an incidence matrix, so `A` is dominant by shape and its diagonal derived.
    pub(crate) fn is_incidence(&self) -> bool {
        matches!(
            self.columns,
            Columns::Incidence { .. }
                | Columns::Bipartite {
                    x: None,
                    y: None,
                    ..
                }
        )
    }

    /// Dimension of `A`, not the column count; saturates, as a sum past `usize` fails allocation anyway.
    pub(crate) fn n(&self) -> usize {
        match self.columns {
            Columns::Incidence { n } | Columns::FactorWidth2 { n, .. } => n,
            Columns::Bipartite {
                n_left, n_right, ..
            } => n_left.saturating_add(n_right),
        }
    }

    /// Each column in order; write `visit` as `#[inline(always)] |column| …` (outlined: 1.6-1.9x).
    #[inline(always)]
    pub(crate) fn for_each_column<S: Index>(self, visit: impl FnMut(Column<T, S>)) {
        let offset = self.n_left().unwrap_or(0);
        let (one, minus_one) = (|_| T::one(), |_| -T::one());
        // Each kind loops straight: reading `V` through a per-column match cost within's paths 6%.
        match self.columns {
            Columns::Incidence { .. }
            | Columns::Bipartite {
                x: None, y: None, ..
            } => self.columns_of(offset, one, minus_one, visit),
            Columns::Bipartite {
                x: Some(x),
                y: None,
                ..
            } => self.columns_of(offset, |k| x[k], one, visit),
            Columns::Bipartite {
                x: None,
                y: Some(y),
                ..
            } => self.columns_of(offset, one, |k| y[k], visit),
            Columns::FactorWidth2 { x, y, .. }
            | Columns::Bipartite {
                x: Some(x),
                y: Some(y),
                ..
            } => self.columns_of(offset, |k| x[k], |k| y[k], visit),
        }
    }

    /// Column `k` is `v_a(k) e_a + v_b(k) e_(offset + b)`; matched once, then looped per case.
    #[inline(always)]
    fn columns_of<S: Index>(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        visit: impl FnMut(Column<T, S>),
    ) {
        match self.weights.0 {
            Weighting::Unit => self.weighted(offset, v_a, v_b, |_| Weighting::Unit, visit),
            Weighting::Of(w) => self.weighted(offset, v_a, v_b, |k| Weighting::Of(w[k]), visit),
            Weighting::SqrtOf(s) => {
                self.weighted(offset, v_a, v_b, |k| Weighting::SqrtOf(s[k]), visit)
            }
        }
    }

    #[inline(always)]
    fn weighted<S: Index>(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        weight: impl Fn(usize) -> Weighting<T>,
        mut visit: impl FnMut(Column<T, S>),
    ) {
        for (k, (a, b)) in self.a.iter().zip(self.b).enumerate() {
            visit(Column {
                a: S::from_index(index(*a)),
                b: S::from_index(offset + index(*b)),
                v_a: v_a(k),
                v_b: v_b(k),
                weight: weight(k),
            });
        }
    }
}

/// One side of `V`, with at most one nonzero per column: `values[k]` at `rows[k]`, `1` for `None`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth1<'a, T, I = u32> {
    n: usize,
    rows: &'a [I],
    values: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> FactorWidth1<'a, T, I> {
    /// `n` rows; each column's row is below `n`, and values, if any, are one per column.
    pub fn new(n: usize, rows: &'a [I], values: Option<&'a [T]>) -> Result<Self, ViewError> {
        values.map_or(Ok(()), |values| length(rows.len(), values.len()))?;
        check_rows(n, rows)?;
        Ok(Self { n, rows, values })
    }
}

/// Why a [`Gram`], [`FactorWidth1`] or [`Weights`] constructor rejected its slices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewError {
    /// A slice's length differs from the one the input fixed: the column count, or `n` for a diagonal.
    LengthMismatch {
        /// The length the input fixed.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// A column names a row its side, or `A`, does not have.
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
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "slice has length {found}, expected {expected}")
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
        }
    }
}

impl std::error::Error for ViewError {}

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

/// `W`, one nonnegative entry per column of `V`, checked once so every [`Gram`] over the same columns can share it; a `Gram` without it has `W = I`.
#[derive(Clone, Copy, Debug)]
pub struct Weights<'a, T>(Weighting<&'a [T]>);

impl<'a, T: Real> Weights<'a, T> {
    /// `W = diag(w)`; each `w` finite and nonnegative.
    pub fn of(w: &'a [T]) -> Result<Self, ViewError> {
        check(w, |w| w.is_finite() && w >= T::zero(), weight)?;
        Ok(Self(Weighting::Of(w)))
    }

    /// `W = diag(s²)`, for a caller that keeps square roots; each `s` finite.
    pub fn sqrt_of(s: &'a [T]) -> Result<Self, ViewError> {
        check(s, |s| s.is_finite(), weight)?;
        Ok(Self(Weighting::SqrtOf(s)))
    }
}

fn weight(column: usize) -> ViewError {
    ViewError::InvalidWeight { column }
}

/// A slice of `found` entries where the input fixed `expected`.
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

/// Each row below `n`.
fn check_rows<I: PrimInt>(n: usize, rows: &[I]) -> Result<(), ViewError> {
    let in_range = |row: &I| row.to_usize().is_some_and(|row| row < n);
    match rows.iter().position(|row| !in_range(row)) {
        Some(column) => Err(ViewError::IndexOutOfRange { column }),
        None => Ok(()),
    }
}

/// One `b` per `a`, each column's two ends distinct rows below `n`.
fn check_ends<I: PrimInt>(n: usize, a: &[I], b: &[I]) -> Result<(), ViewError> {
    length(a.len(), b.len())?;
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
