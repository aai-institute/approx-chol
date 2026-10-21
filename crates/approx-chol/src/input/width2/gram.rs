//! `A = V W Vᵀ + diag(d)`, the product every factor view is: column `k` of `V` joins `a[k]` and `b[k]`.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::{check, check_ends, length, Column, IntoColumns, ViewError};
use crate::{Index, Real};

/// The incidence matrix `V` is, by its columns' signs.
#[derive(Clone, Copy, Debug)]
pub enum Incidence<'a> {
    /// `e_a − e_b`: `V W Vᵀ` is a graph Laplacian.
    Oriented,
    /// `e_a + e_b`: a signless Laplacian.
    Unoriented,
    /// `e_a + e_b` where `positive[k]`, else `e_a − e_b`: a signed Laplacian.
    Signed(&'a [bool]),
}

/// Where `V`'s two rows per column lie in `A`.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// `a[k] ≠ b[k]`, both in `0..n`.
    General,
    /// `a[k]` in `0..n_left`, `b[k]` in `0..n − n_left`, numbered after it.
    Bipartite {
        /// The left part's size.
        n_left: usize,
    },
}

/// `V`'s columns: an incidence matrix's, or `x[k] e_a + y[k] e_b` with `None` reading as `1`.
#[derive(Clone, Copy, Debug)]
enum Columns<'a, T> {
    Incidence(Incidence<'a>),
    Values([Option<&'a [T]>; 2]),
}

impl<'a, T> Columns<'a, T> {
    /// Ones on both sides are the unoriented incidence, stored as it.
    fn values(x: Option<&'a [T]>, y: Option<&'a [T]>) -> Self {
        match (x, y) {
            (None, None) => Self::Incidence(Incidence::Unoriented),
            (x, y) => Self::Values([x, y]),
        }
    }
}

/// The signs folding `A` into a Z-matrix when `V`'s shape fixes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fold {
    /// Every cell is already nonpositive.
    Identity,
    /// Every cell is positive and crosses the parts: reading the right part negated folds it.
    NegatedRight,
}

/// `A = V W Vᵀ + diag(d)` on `n` vertices.
#[derive(Clone, Copy, Debug)]
pub struct Gram<'a, T, I = u32> {
    n: usize,
    shape: Shape,
    edges: Edges<'a, T, I>,
    columns: Columns<'a, T>,
    diagonal: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> Gram<'a, T, I> {
    /// `V` an incidence matrix; every length and index is checked, `W = I`.
    pub fn incidence(
        n: usize,
        shape: Shape,
        a: &'a [I],
        b: &'a [I],
        incidence: Incidence<'a>,
    ) -> Result<Self, ViewError> {
        if let Incidence::Signed(positive) = incidence {
            length(a.len(), positive.len())?;
        }
        Self::checked(n, shape, a, b, Columns::Incidence(incidence))
    }

    /// Column `k` is `x[k] e_a + y[k] e_b`, `None` reading as `1`; every length and index is checked, `W = I`.
    pub fn values(
        n: usize,
        shape: Shape,
        a: &'a [I],
        b: &'a [I],
        x: Option<&'a [T]>,
        y: Option<&'a [T]>,
    ) -> Result<Self, ViewError> {
        for values in [x, y].into_iter().flatten() {
            length(a.len(), values.len())?;
        }
        Self::checked(n, shape, a, b, Columns::values(x, y))
    }

    fn checked(
        n: usize,
        shape: Shape,
        a: &'a [I],
        b: &'a [I],
        columns: Columns<'a, T>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(a, b)?;
        match shape {
            Shape::General => check_ends(n, a, b)?,
            Shape::Bipartite { n_left } => {
                let n_right =
                    (n.checked_sub(n_left)).ok_or(ViewError::LeftPartTooLarge { n_left, n })?;
                check_rows(n_left, a)?;
                check_rows(n_right, b)?;
            }
        }
        Ok(Self::of_checked(n, shape, edges, columns))
    }

    /// Two width-1 sides, each of which checked its rows and values.
    pub(super) fn of_sides(
        (n_left, x): (usize, Option<&'a [T]>),
        (n_right, y): (usize, Option<&'a [T]>),
        edges: Edges<'a, T, I>,
    ) -> Self {
        let (n, shape) = (n_left.saturating_add(n_right), Shape::Bipartite { n_left });
        Self::of_checked(n, shape, edges, Columns::values(x, y))
    }

    fn of_checked(n: usize, shape: Shape, edges: Edges<'a, T, I>, columns: Columns<'a, T>) -> Self {
        Self {
            n,
            shape,
            edges,
            columns,
            diagonal: None,
        }
    }

    /// `V W Vᵀ` under `weights`, one per column.
    pub fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        let edges = self.edges.with_weights(weights)?;
        Ok(Self { edges, ..self })
    }

    /// `V W Vᵀ + diag(d)`, one finite nonnegative `d_i` per vertex.
    pub fn plus_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        length(self.n, diagonal.len())?;
        let valid = |d: T| d.is_finite() && d >= T::zero();
        check(diagonal, valid, |vertex| ViewError::InvalidDiagonal {
            vertex,
        })?;
        Ok(Self {
            diagonal: Some(diagonal),
            ..self
        })
    }

    pub(crate) fn shape(&self) -> Shape {
        self.shape
    }

    /// `d`, which is no column of `V`.
    pub(crate) fn diagonal(&self) -> Option<&'a [T]> {
        self.diagonal
    }

    /// D: `V` is an incidence matrix, so `A` is dominant by shape and its diagonal derived.
    pub(crate) fn is_incidence(&self) -> bool {
        matches!(self.columns, Columns::Incidence(_))
    }

    /// X: two parallel columns can put opposite signs on one cell.
    pub(crate) fn cancels(&self) -> bool {
        matches!(self.columns, Columns::Incidence(Incidence::Signed(_)))
    }

    /// K: the fold and the reading that applies it, when every cell's sign is fixed by `V` and the shape.
    pub(crate) fn fold(self) -> Option<(Fold, Self)> {
        let oriented = Columns::Incidence(Incidence::Oriented);
        match (self.columns, self.shape) {
            (Columns::Incidence(Incidence::Oriented), _) => Some((Fold::Identity, self)),
            (Columns::Incidence(Incidence::Unoriented), Shape::Bipartite { .. }) => Some((
                Fold::NegatedRight,
                Self {
                    columns: oriented,
                    ..self
                },
            )),
            _ => None,
        }
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

impl<T: Real, I: PrimInt> IntoColumns<T> for Gram<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    #[inline(always)]
    fn for_each_column<S: Index>(self, visit: impl FnMut(Column<T, S>)) {
        let offset = match self.shape {
            Shape::General => 0,
            Shape::Bipartite { n_left } => n_left,
        };
        let (edges, one, minus_one) = (self.edges, |_| T::one(), |_| -T::one());
        // Each kind loops straight: reading `V` through a per-column match cost within's paths 6%.
        match self.columns {
            Columns::Incidence(Incidence::Oriented) => {
                edges.for_each_column(offset, one, minus_one, visit)
            }
            Columns::Incidence(Incidence::Unoriented) | Columns::Values([None, None]) => {
                edges.for_each_column(offset, one, one, visit)
            }
            Columns::Incidence(Incidence::Signed(positive)) => {
                let sign = |k: usize| match positive[k] {
                    true => T::one(),
                    false => -T::one(),
                };
                edges.for_each_column(offset, one, sign, visit)
            }
            Columns::Values([Some(x), None]) => edges.for_each_column(offset, |k| x[k], one, visit),
            Columns::Values([None, Some(y)]) => edges.for_each_column(offset, one, |k| y[k], visit),
            Columns::Values([Some(x), Some(y)]) => {
                edges.for_each_column(offset, |k| x[k], |k| y[k], visit)
            }
        }
    }
}
