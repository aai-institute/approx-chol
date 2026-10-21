//! `A = V W Vᵀ + diag(d)`, the product every factor view is: column `k` of `V` joins `a[k]` and `b[k]`.

use num_traits::PrimInt;

use super::bipartite::FactorWidth1;
use super::edges::{Edges, Weights};
use super::{check, check_ends, check_rows, length, Column, IntoColumns, ViewError};
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

/// How `A`'s vertices split between `V`'s two rows per column.
#[derive(Clone, Copy, Debug)]
pub enum VertexPartition {
    /// `a[k] ≠ b[k]`, both in `0..n`.
    Unpartitioned,
    /// `a[k]` in `0..n_left`, `b[k]` in `0..n − n_left`, numbered after it.
    Bipartite {
        /// The left part's size.
        n_left: usize,
    },
}

/// `V`'s columns: an incidence matrix's, or `x[k] e_a + y[k] e_b` with a missing side reading as `1`.
#[derive(Clone, Copy, Debug)]
enum Columns<'a, T> {
    Incidence(Incidence<'a>),
    /// `x[k] e_a + e_b`.
    ValuesX(&'a [T]),
    /// `e_a + y[k] e_b`.
    ValuesY(&'a [T]),
    /// `x[k] e_a + y[k] e_b`.
    ValuesXY(&'a [T], &'a [T]),
}

impl<'a, T> Columns<'a, T> {
    /// Ones on both sides are the unoriented incidence, stored as it.
    fn values(x: Option<&'a [T]>, y: Option<&'a [T]>) -> Self {
        match (x, y) {
            (None, None) => Self::Incidence(Incidence::Unoriented),
            (Some(x), None) => Self::ValuesX(x),
            (None, Some(y)) => Self::ValuesY(y),
            (Some(x), Some(y)) => Self::ValuesXY(x, y),
        }
    }
}

/// The switching, a `±1` diagonal congruence, taking `A` to a Z-matrix when `V` and the partition fix it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Switching {
    /// Every cell is already nonpositive.
    Identity,
    /// Every cell is positive and crosses the parts: negating the right part switches it.
    NegateRight,
}

/// `A = V W Vᵀ + diag(d)` on `n` vertices.
#[derive(Clone, Copy, Debug)]
pub struct Gram<'a, T, I = u32> {
    n: usize,
    partition: VertexPartition,
    edges: Edges<'a, T, I>,
    columns: Columns<'a, T>,
    diagonal: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> Gram<'a, T, I> {
    /// `V` an incidence matrix; every length and index is checked, `W = I`.
    pub fn incidence(
        n: usize,
        partition: VertexPartition,
        a: &'a [I],
        b: &'a [I],
        incidence: Incidence<'a>,
    ) -> Result<Self, ViewError> {
        if let Incidence::Signed(positive) = incidence {
            length(a.len(), positive.len())?;
        }
        Self::checked(n, partition, a, b, Columns::Incidence(incidence))
    }

    /// Column `k` is `x[k] e_a + y[k] e_b`, `None` reading as `1`; every length and index is checked, `W = I`.
    pub fn values(
        n: usize,
        partition: VertexPartition,
        a: &'a [I],
        b: &'a [I],
        x: Option<&'a [T]>,
        y: Option<&'a [T]>,
    ) -> Result<Self, ViewError> {
        for values in [x, y].into_iter().flatten() {
            length(a.len(), values.len())?;
        }
        Self::checked(n, partition, a, b, Columns::values(x, y))
    }

    fn checked(
        n: usize,
        partition: VertexPartition,
        a: &'a [I],
        b: &'a [I],
        columns: Columns<'a, T>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(a, b)?;
        match partition {
            VertexPartition::Unpartitioned => check_ends(n, a, b)?,
            VertexPartition::Bipartite { n_left } => {
                let n_right =
                    (n.checked_sub(n_left)).ok_or(ViewError::LeftPartTooLarge { n_left, n })?;
                check_rows(n_left, a)?;
                check_rows(n_right, b)?;
            }
        }
        Ok(Self::of_checked(n, partition, edges, columns))
    }

    /// `V = [V_l; V_r]` from two sides that each checked their rows and values.
    pub(super) fn of_sides(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(left.rows, right.rows)?;
        let (n_left, n_right) = (left.n, right.n);
        let n = (n_left.checked_add(n_right))
            .ok_or(ViewError::DimensionOverflow { n_left, n_right })?;
        let partition = VertexPartition::Bipartite { n_left };
        let columns = Columns::values(left.values, right.values);
        Ok(Self::of_checked(n, partition, edges, columns))
    }

    fn of_checked(
        n: usize,
        partition: VertexPartition,
        edges: Edges<'a, T, I>,
        columns: Columns<'a, T>,
    ) -> Self {
        Self {
            n,
            partition,
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

    /// `V W Vᵀ + diag(d)`, one finite nonnegative `d_i` per vertex, replacing any earlier `d`.
    pub fn with_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
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

    pub(crate) fn partition(&self) -> VertexPartition {
        self.partition
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

    /// K: the switching and the reading that applies it, when `V` and the partition fix every cell's sign.
    pub(crate) fn switched(self) -> Option<(Switching, Self)> {
        let oriented = Columns::Incidence(Incidence::Oriented);
        match (self.columns, self.partition) {
            (Columns::Incidence(Incidence::Oriented), _) => Some((Switching::Identity, self)),
            (Columns::Incidence(Incidence::Unoriented), VertexPartition::Bipartite { .. }) => {
                let read = Self {
                    columns: oriented,
                    ..self
                };
                Some((Switching::NegateRight, read))
            }
            _ => None,
        }
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for Gram<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    #[inline(always)]
    fn for_each_column<S: Index>(self, visit: impl FnMut(Column<T, S>)) {
        let offset = match self.partition {
            VertexPartition::Unpartitioned => 0,
            VertexPartition::Bipartite { n_left } => n_left,
        };
        let (edges, one, minus_one) = (self.edges, |_| T::one(), |_| -T::one());
        // Each kind loops straight: reading `V` through a per-column match cost within's paths 6%.
        match self.columns {
            Columns::Incidence(Incidence::Oriented) => {
                edges.for_each_column(offset, one, minus_one, visit)
            }
            Columns::Incidence(Incidence::Unoriented) => {
                edges.for_each_column(offset, one, one, visit)
            }
            Columns::Incidence(Incidence::Signed(positive)) => {
                let sign = |k: usize| match positive[k] {
                    true => T::one(),
                    false => -T::one(),
                };
                edges.for_each_column(offset, one, sign, visit)
            }
            Columns::ValuesX(x) => edges.for_each_column(offset, |k| x[k], one, visit),
            Columns::ValuesY(y) => edges.for_each_column(offset, one, |k| y[k], visit),
            Columns::ValuesXY(x, y) => edges.for_each_column(offset, |k| x[k], |k| y[k], visit),
        }
    }
}
