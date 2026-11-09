//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, left rows numbered first.

use std::ops::Range;

use num_traits::PrimInt;

use super::{BipartiteColumns, Columns, FactorWidth2};
use crate::input::error::{finite, indices, length, InputError};
use crate::Real;

/// One side of `V`, with at most one nonzero per column: `values[k]` at `rows[k]`, `1` for `None`.
#[derive(Clone, Copy, Debug)]
pub struct FactorWidth1<'a, T, I = u32> {
    n: usize,
    rows: &'a [I],
    values: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> FactorWidth1<'a, T, I> {
    /// `n` rows, each column's `rows[k]` below `n`, its value `1`.
    pub fn new(n: usize, rows: &'a [I]) -> Result<Self, InputError> {
        Self::checked(n, rows, None)
    }

    /// `n` rows, each column's `rows[k]` below `n`, its value `values[k]`.
    pub fn valued(n: usize, rows: &'a [I], values: &'a [T]) -> Result<Self, InputError> {
        length("values", rows.len(), values.len())?;
        Self::checked(n, rows, Some(values))
    }

    fn checked(n: usize, rows: &'a [I], values: Option<&'a [T]>) -> Result<Self, InputError> {
        indices("rows", n, rows)?;
        Ok(Self { n, rows, values })
    }
}

/// `V W Vᵀ`, `V = [V_l; V_r]`: left vertices first, then right.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteFactorWidth2<'a, T, I = u32> {
    a: &'a [I],
    b: &'a [I],
    columns: BipartiteColumns<'a, T>,
}

impl<'a, T: Real, I: PrimInt> BipartiteFactorWidth2<'a, T, I> {
    /// Both sides have the same column count.
    pub fn new(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
    ) -> Result<Self, InputError> {
        length("right", left.rows.len(), right.rows.len())?;
        Ok(Self::of(left, right, None))
    }

    /// `(V S)(V S)ᵀ`, `S = diag(s)`: column `k` scaled by `s[k]`, finite, on both sides.
    pub fn scaled(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        s: &'a [T],
    ) -> Result<Self, InputError> {
        length("right", left.rows.len(), right.rows.len())?;
        length("s", left.rows.len(), s.len())?;
        finite("s", s)?;
        Ok(Self::of(left, right, Some(s)))
    }

    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        self.columns.n()
    }

    /// The run of `A`'s vertices on `side`.
    pub(crate) fn vertices(&self, side: Side) -> Range<usize> {
        match side {
            Side::Left => 0..self.columns.n_left,
            Side::Right => self.columns.n_left..self.n(),
        }
    }

    pub(crate) fn lower(self) -> FactorWidth2<'a, T, I> {
        let columns = Columns::Bipartite(self.columns);
        FactorWidth2 {
            a: self.a,
            b: self.b,
            columns,
        }
    }

    fn of(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        scale: Option<&'a [T]>,
    ) -> Self {
        let columns = BipartiteColumns {
            n_left: left.n,
            n_right: right.n,
            x: left.values,
            y: right.values,
            scale,
        };
        Self {
            a: left.rows,
            b: right.rows,
            columns,
        }
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
