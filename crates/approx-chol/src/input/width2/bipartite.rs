//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, left rows numbered first.

use std::ops::Range;

use num_traits::PrimInt;

use super::error::{below, check, length, ViewError};
use super::{Columns, FactorWidth2};
use crate::Real;

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
        check(
            rows,
            |row| below(n, row),
            |entry| ViewError::IndexOutOfRange { entry },
        )?;
        Ok(Self { n, rows, values })
    }
}

/// `V W Vᵀ`, `V = [V_l; V_r]`: left vertices first, then right.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteFactorWidth2<'a, T, I = u32> {
    v: FactorWidth2<'a, T, I>,
    n_left: usize,
}

impl<'a, T: Real, I: PrimInt> BipartiteFactorWidth2<'a, T, I> {
    /// Both sides have the same column count.
    pub fn new(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
    ) -> Result<Self, ViewError> {
        length(left.rows.len(), right.rows.len())?;
        Ok(Self::of(left, right, None))
    }

    /// `(V S)(V S)ᵀ`, `S = diag(s)`: column `k` scaled by `s[k]`, finite, on both sides.
    pub fn scaled(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        s: &'a [T],
    ) -> Result<Self, ViewError> {
        length(left.rows.len(), right.rows.len())?;
        length(left.rows.len(), s.len())?;
        check(
            s,
            |s: T| s.is_finite(),
            |entry| ViewError::InvalidScale { entry },
        )?;
        Ok(Self::of(left, right, Some(s)))
    }

    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        self.v.n()
    }

    /// The left part's size, after which the right's vertices are numbered.
    pub(crate) fn n_left(&self) -> usize {
        self.n_left
    }

    /// The run of `A`'s vertices on `side`.
    pub(crate) fn vertices(&self, side: Side) -> Range<usize> {
        match side {
            Side::Left => 0..self.n_left,
            Side::Right => self.n_left..self.n(),
        }
    }

    pub(crate) fn lower(self) -> FactorWidth2<'a, T, I> {
        self.v
    }

    fn of(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        scale: Option<&'a [T]>,
    ) -> Self {
        let columns = Columns::Bipartite {
            n_left: left.n,
            n_right: right.n,
            x: left.values,
            y: right.values,
            scale,
        };
        let v = FactorWidth2 {
            a: left.rows,
            b: right.rows,
            columns,
        };
        Self { v, n_left: left.n }
    }
}

/// One side of a [`BipartiteFactorWidth2`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// `V_l`'s rows, `A`'s vertices `0..n_left`.
    Left,
    /// `V_r`'s rows, `A`'s vertices `n_left..n`.
    Right,
}
