//! The factor-shaped inputs, each lowered to `A = V W Vᵀ + diag(s)`: column `k` of `V` joins `a[k]` and `b[k]`.

use num_traits::PrimInt;

use super::index;
use crate::{widen, Index, Real};

/// `V`'s columns and vertices, one variant per input type.
#[derive(Clone, Copy, Debug)]
enum Columns<'a, T> {
    /// `e_a − e_b`, `a[k] ≠ b[k] < n`, and an [`Sddm`]'s surplus.
    Incidence { n: usize, surplus: Option<&'a [T]> },
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

/// `A = V W Vᵀ + diag(s)` on `n` vertices, the form every factor-shaped input lowers to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gram<'a, T, I = u32> {
    a: &'a [I],
    b: &'a [I],
    weights: Weights<&'a [T]>,
    columns: Columns<'a, T>,
}

/// The Laplacian of an undirected graph on `n` vertices.
#[derive(Clone, Copy, Debug)]
pub struct Laplacian<'a, T, I = u32>(pub(crate) Gram<'a, T, I>);

impl<'a, T: Real, I: PrimInt> Laplacian<'a, T, I> {
    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight 1; repeated edges sum.
    pub fn new(n: usize, u: &'a [I], v: &'a [I]) -> Result<Self, ViewError> {
        check_ends(n, u, v)?;
        let columns = Columns::Incidence { n, surplus: None };
        Ok(Self(Gram::of(u, v, columns, Weights::Unit)))
    }

    /// Edge `k` joins `u[k] ≠ v[k] < n` with weight `w[k]`, finite and `≥ 0`; repeated edges sum.
    pub fn weighted(n: usize, u: &'a [I], v: &'a [I], w: &'a [T]) -> Result<Self, ViewError> {
        let weights = Weights::Of(w).checked(u.len())?;
        let Self(gram) = Self::new(n, u, v)?;
        Ok(Self(Gram { weights, ..gram }))
    }
}

/// `L + diag(s)`: a Laplacian plus each vertex's surplus over diagonal dominance.
#[derive(Clone, Copy, Debug)]
pub struct Sddm<'a, T, I = u32>(pub(crate) Gram<'a, T, I>);

impl<'a, T: Real, I: PrimInt> Sddm<'a, T, I> {
    /// `s[i]` finite and `≥ 0`, one per vertex, not all zero (that is a [`Laplacian`]).
    pub fn new(laplacian: Laplacian<'a, T, I>, surplus: &'a [T]) -> Result<Self, ViewError> {
        let gram = laplacian.0;
        let n = gram.n();
        length(n, surplus.len())?;
        check(surplus, nonnegative, |vertex| ViewError::InvalidSurplus {
            vertex,
        })?;
        match surplus.iter().all(|&s| s == T::zero()) {
            true => Err(ViewError::ZeroSurplus),
            false => Ok(Self(Gram {
                columns: Columns::Incidence {
                    n,
                    surplus: Some(surplus),
                },
                ..gram
            })),
        }
    }
}

/// `V Vᵀ` on `n` vertices, column `k` of `V` being `x[k] e_a[k] + y[k] e_b[k]`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth2<'a, T, I = u32>(pub(crate) Gram<'a, T, I>);

impl<'a, T: Real, I: PrimInt> FactorWidth2<'a, T, I> {
    /// `a[k] ≠ b[k] < n`; `a`, `x`, `b`, `y` one entry per column.
    pub fn new(
        n: usize,
        (a, x): (&'a [I], &'a [T]),
        (b, y): (&'a [I], &'a [T]),
    ) -> Result<Self, ViewError> {
        length(a.len(), x.len())?;
        length(a.len(), y.len())?;
        check_ends(n, a, b)?;
        let columns = Columns::FactorWidth2 { n, x, y };
        Ok(Self(Gram::of(a, b, columns, Weights::Unit)))
    }
}

/// `V W Vᵀ`, `V = [V_l; V_r]`: left vertices first, then right.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteFactorWidth2<'a, T, I = u32>(pub(crate) Gram<'a, T, I>);

impl<'a, T: Real, I: PrimInt> BipartiteFactorWidth2<'a, T, I> {
    /// Both sides have the same column count; `weights` has one entry per column.
    pub fn new(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        weights: Weights<&'a [T]>,
    ) -> Result<Self, ViewError> {
        length(left.rows.len(), right.rows.len())?;
        let weights = weights.checked(left.rows.len())?;
        let columns = Columns::Bipartite {
            n_left: left.n,
            n_right: right.n,
            x: left.values,
            y: right.values,
        };
        Ok(Self(Gram::of(left.rows, right.rows, columns, weights)))
    }
}

impl<'a, T: Real, I: PrimInt> Gram<'a, T, I> {
    fn of(a: &'a [I], b: &'a [I], columns: Columns<'a, T>, weights: Weights<&'a [T]>) -> Self {
        Self {
            a,
            b,
            weights,
            columns,
        }
    }

    /// The left part's size, after which `b` is numbered, when `V` is bipartite.
    pub(crate) fn n_left(&self) -> Option<usize> {
        match self.columns {
            Columns::Bipartite { n_left, .. } => Some(n_left),
            Columns::Incidence { .. } | Columns::FactorWidth2 { .. } => None,
        }
    }

    /// The surplus `s`, which only an [`Sddm`] has.
    pub(crate) fn surplus(&self) -> Option<&'a [T]> {
        match self.columns {
            Columns::Incidence { surplus, .. } => surplus,
            Columns::FactorWidth2 { .. } | Columns::Bipartite { .. } => None,
        }
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
            Columns::Incidence { n, .. } | Columns::FactorWidth2 { n, .. } => n,
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
        match self.weights {
            Weights::Unit => self.weighted(offset, v_a, v_b, |_| Weights::Unit, visit),
            Weights::Of(w) => self.weighted(offset, v_a, v_b, |k| Weights::Of(w[k]), visit),
            Weights::SqrtOf(s) => self.weighted(offset, v_a, v_b, |k| Weights::SqrtOf(s[k]), visit),
        }
    }

    #[inline(always)]
    fn weighted<S: Index>(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        weight: impl Fn(usize) -> Weights<T>,
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

/// Why an input constructor rejected its slices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewError {
    /// A slice's length differs from the one the input fixed: the edge or column count, or `n`.
    LengthMismatch {
        /// The length the input fixed.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// An edge or column names a vertex or row that does not exist.
    IndexOutOfRange {
        /// The offending edge or column.
        entry: usize,
    },
    /// An edge or column names one vertex at both ends.
    RepeatedVertex {
        /// The offending edge or column.
        entry: usize,
    },
    /// A weight is negative or not finite.
    InvalidWeight {
        /// The offending edge or column.
        entry: usize,
    },
    /// A surplus is negative or not finite.
    InvalidSurplus {
        /// The offending vertex.
        vertex: usize,
    },
    /// Every surplus is zero: the matrix is a [`Laplacian`].
    ZeroSurplus,
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "slice has length {found}, expected {expected}")
            }
            Self::IndexOutOfRange { entry } => write!(f, "entry {entry} names a missing vertex"),
            Self::RepeatedVertex { entry } => write!(f, "entry {entry} names one vertex twice"),
            Self::InvalidWeight { entry } => {
                write!(f, "entry {entry} has a negative or non-finite weight")
            }
            Self::InvalidSurplus { vertex } => {
                write!(f, "vertex {vertex} has a negative or non-finite surplus")
            }
            Self::ZeroSurplus => write!(f, "every surplus is zero, so the matrix is a Laplacian"),
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
    pub(crate) weight: Weights<T>,
}

impl<T: Real, I> Column<T, I> {
    /// `w·v_a·v_b` in `f64`: every assembly sums its cells there and casts once.
    pub(crate) fn cell(self) -> f64 {
        (self.weight).bilinear_unflushed(widen(self.v_a), widen(self.v_b))
    }
}

/// `W`: one entry per column as a slice, checked by the constructor that takes it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum Weights<S> {
    /// `W = I`.
    Unit,
    /// `W = diag(w)`, each `w` finite and `≥ 0`.
    Of(S),
    /// `W = diag(s²)`, each `s` finite, for a caller that keeps square roots.
    SqrtOf(S),
}

impl<T: Real> Weights<T> {
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

impl<T: Real> Weights<&[T]> {
    /// `W` over `columns` columns, each entry checked.
    fn checked(self, columns: usize) -> Result<Self, ViewError> {
        if let Self::Of(w) | Self::SqrtOf(w) = self {
            length(columns, w.len())?;
        }
        match self {
            Self::Unit => {}
            Self::Of(w) => check(w, nonnegative, weight)?,
            Self::SqrtOf(s) => check(s, |s| s.is_finite(), weight)?,
        }
        Ok(self)
    }
}

fn nonnegative<T: Real>(x: T) -> bool {
    x.is_finite() && x >= T::zero()
}

fn weight(entry: usize) -> ViewError {
    ViewError::InvalidWeight { entry }
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
        Some(entry) => Err(ViewError::IndexOutOfRange { entry }),
        None => Ok(()),
    }
}

/// One `b` per `a`, each column's two ends distinct rows below `n`.
fn check_ends<I: PrimInt>(n: usize, a: &[I], b: &[I]) -> Result<(), ViewError> {
    length(a.len(), b.len())?;
    let in_range = |end: &I| end.to_usize().is_some_and(|end| end < n);
    for (entry, (end_a, end_b)) in a.iter().zip(b).enumerate() {
        if !in_range(end_a) || !in_range(end_b) {
            return Err(ViewError::IndexOutOfRange { entry });
        }
        if end_a == end_b {
            return Err(ViewError::RepeatedVertex { entry });
        }
    }
    Ok(())
}
