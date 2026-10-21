//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, left rows numbered first.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::gram::Gram;
use super::{length, ViewError};
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
        let in_range = |row: &I| row.to_usize().is_some_and(|row| row < n);
        if let Some(column) = rows.iter().position(|row| !in_range(row)) {
            return Err(ViewError::IndexOutOfRange { column });
        }
        Ok(Self { n, rows, values })
    }
}

/// `A = V W Vᵀ + diag(d)` with `V = [V_l; V_r]`, both width 1 on disjoint rows, left rows numbered first.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteGram<'a, T, I = u32>(Gram<'a, T, I>);

impl<'a, T: Real, I: PrimInt> BipartiteGram<'a, T, I> {
    /// Both factors share the column count; `W = I`.
    pub fn new(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
    ) -> Result<Self, ViewError> {
        // Each side checked its rows and values, so only the column counts are left to agree.
        let edges = Edges::new(left.rows, right.rows)?;
        let (left, right) = ((left.n, left.values), (right.n, right.values));
        Ok(Self(Gram::of_sides(left, right, edges)))
    }

    /// `V W Vᵀ` under `weights`, one per column.
    pub fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        self.0.with_weights(weights).map(Self)
    }

    /// `V W Vᵀ + diag(d)`, one finite nonnegative `d_i` per vertex, left rows first.
    pub fn plus_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        self.0.plus_diagonal(diagonal).map(Self)
    }
}

impl<'a, T, I> From<BipartiteGram<'a, T, I>> for Gram<'a, T, I> {
    fn from(gram: BipartiteGram<'a, T, I>) -> Self {
        gram.0
    }
}
