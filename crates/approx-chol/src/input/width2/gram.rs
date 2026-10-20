//! `A = V W Vᵀ + diag(d)`, the product every factor view is: column `k` is `tail(k) e_a + head(k) e_b`.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::{check, check_ends, length, Column, IntoColumns, ViewError};
use crate::{Index, Real};

/// One end's entries of `V`, per column.
#[derive(Clone, Copy, Debug)]
pub enum End<'a, T> {
    /// `+1` in every column.
    One,
    /// `−1` in every column.
    MinusOne,
    /// `+1` where `true`, else `−1`.
    Signs(&'a [bool]),
    /// Any value, per column.
    Values(&'a [T]),
}

impl<T: Real> End<'_, T> {
    #[inline(always)]
    fn at(self, k: usize) -> T {
        match self {
            Self::One => T::one(),
            Self::MinusOne => -T::one(),
            Self::Signs(positive) => match positive[k] {
                true => T::one(),
                false => -T::one(),
            },
            Self::Values(values) => values[k],
        }
    }

    fn len(self) -> Option<usize> {
        match self {
            Self::One | Self::MinusOne => None,
            Self::Signs(positive) => Some(positive.len()),
            Self::Values(values) => Some(values.len()),
        }
    }
}

/// Where `V`'s two ends index `A`.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// Both ends anywhere in `0..n`, never the same row.
    General,
    /// The tail in `0..n_left`, the head in `0..n − n_left`, numbered after it.
    Bipartite {
        /// The left part's size.
        n_left: usize,
    },
}

/// The signs folding `A` into a Z-matrix when `V`'s shape fixes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fold {
    /// Every cell is already nonpositive.
    Identity,
    /// Every cell is positive and crosses the parts: reading the head negated folds it.
    NegatedHead,
}

/// `A = V W Vᵀ + diag(d)` on `n` vertices.
#[derive(Clone, Copy, Debug)]
pub struct Gram<'a, T, I = u32> {
    n: usize,
    shape: Shape,
    edges: Edges<'a, T, I>,
    ends: [End<'a, T>; 2],
    diagonal: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> Gram<'a, T, I> {
    /// Column `k` joins `a[k]` and `b[k]`, read by `shape`; every length and index is checked, `W = I`.
    pub fn new(
        n: usize,
        shape: Shape,
        a: &'a [I],
        b: &'a [I],
        tail: End<'a, T>,
        head: End<'a, T>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(a, b)?;
        for found in [tail.len(), head.len()].into_iter().flatten() {
            length(a.len(), found)?;
        }
        match shape {
            Shape::General => check_ends(n, a, b)?,
            Shape::Bipartite { n_left } => {
                let n_right =
                    (n.checked_sub(n_left)).ok_or(ViewError::LeftPartTooLarge { n_left, n })?;
                check_rows(n_left, a)?;
                check_rows(n_right, b)?;
            }
        }
        Ok(Self::of_checked(n, shape, edges, [tail, head]))
    }

    /// Columns whose lengths and rows the caller has checked against `shape`.
    pub(super) fn of_checked(
        n: usize,
        shape: Shape,
        edges: Edges<'a, T, I>,
        ends: [End<'a, T>; 2],
    ) -> Self {
        Self {
            n,
            shape,
            edges,
            ends,
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

    /// D: every entry of `V` is `±1`, so `A` is dominant by shape and its diagonal derived.
    pub(crate) fn is_incidence(&self) -> bool {
        (self.ends).iter().all(|end| !matches!(end, End::Values(_)))
    }

    /// X: two parallel columns can put opposite signs on one cell.
    pub(crate) fn cancels(&self) -> bool {
        self.is_incidence() && (self.ends).iter().any(|end| matches!(end, End::Signs(_)))
    }

    /// K: the fold and the reading that applies it, when every cell's sign is fixed by the ends and the shape.
    pub(crate) fn fold(self) -> Option<(Fold, Self)> {
        use End::{MinusOne, One};
        let negated = |head| Self {
            ends: [self.ends[0], head],
            ..self
        };
        match (self.shape, self.ends) {
            (_, [One, MinusOne] | [MinusOne, One]) => Some((Fold::Identity, self)),
            (Shape::Bipartite { .. }, [One, One]) => Some((Fold::NegatedHead, negated(MinusOne))),
            (Shape::Bipartite { .. }, [MinusOne, MinusOne]) => {
                Some((Fold::NegatedHead, negated(One)))
            }
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
        use End::{MinusOne, One, Signs, Values};
        let (edges, offset) = match self.shape {
            Shape::General => (self.edges, 0),
            Shape::Bipartite { n_left } => (self.edges, n_left),
        };
        let (one, minus_one) = (|_| T::one(), |_| -T::one());
        // Each named form's ends loop straight (per-column reads cost within's paths 6%); any other pair reads per column.
        match self.ends {
            [One, MinusOne] => edges.for_each_column(offset, one, minus_one, visit),
            [One, One] => edges.for_each_column(offset, one, one, visit),
            [One, Signs(_)] => edges.for_each_column(offset, one, |k| self.ends[1].at(k), visit),
            [Values(x), One] => edges.for_each_column(offset, |k| x[k], one, visit),
            [One, Values(y)] => edges.for_each_column(offset, one, |k| y[k], visit),
            [Values(x), Values(y)] => edges.for_each_column(offset, |k| x[k], |k| y[k], visit),
            [tail, head] => edges.for_each_column(offset, |k| tail.at(k), |k| head.at(k), visit),
        }
    }
}
