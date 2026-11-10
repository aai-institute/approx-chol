//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, left rows numbered first.

use std::ops::Range;

use num_traits::PrimInt;

use super::{BipartiteColumns, Columns, FactorWidth2};
use crate::input::error::{indices, length, values_in_range, InputError};
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

    /// `n` rows, each column's `rows[k]` below `n`, its value `values[k]`, [in range](InputError::OutOfRange).
    pub fn valued(n: usize, rows: &'a [I], values: &'a [T]) -> Result<Self, InputError> {
        length("values", rows.len(), values.len())?;
        values_in_range("values", values)?;
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
        Self::of(left, right, None)
    }

    /// `(V S)(V S)ᵀ`, `S = diag(s)`: column `k` scaled by `s[k]`, [in range](InputError::OutOfRange), on both sides.
    pub fn scaled(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        s: &'a [T],
    ) -> Result<Self, InputError> {
        length("right", left.rows.len(), right.rows.len())?;
        length("s", left.rows.len(), s.len())?;
        values_in_range("s", s)?;
        Self::of(left, right, Some(s))
    }

    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        self.columns.n
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
    ) -> Result<Self, InputError> {
        let n = left
            .n
            .checked_add(right.n)
            .ok_or(InputError::VertexCountOverflow {
                n_left: left.n,
                n_right: right.n,
            })?;
        let columns = BipartiteColumns {
            n,
            n_left: left.n,
            x: left.values,
            y: right.values,
            scale,
        };
        Ok(Self {
            a: left.rows,
            b: right.rows,
            columns,
        })
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
