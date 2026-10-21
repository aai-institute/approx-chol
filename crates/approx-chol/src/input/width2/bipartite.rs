//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, left rows numbered first.

use num_traits::PrimInt;

use super::error::{below, check, length, ViewError};
use super::weights::Weights;
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
            |column| ViewError::IndexOutOfRange { column },
        )?;
        Ok(Self { n, rows, values })
    }
}

/// `A = V W Vᵀ` with `V = [V_l; V_r]`, both width 1 on disjoint rows, left rows numbered first.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteGram<'a, T, I = u32>(pub(crate) FactorWidth2<'a, T, I>);

impl<'a, T: Real, I: PrimInt> BipartiteGram<'a, T, I> {
    /// Both factors share the column count and the weights are checked.
    pub fn new(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        length(left.rows.len(), right.rows.len())?;
        let weights = weights.of_columns(left.rows.len())?;
        let columns = Columns::Bipartite {
            n_left: left.n,
            n_right: right.n,
            x: left.values,
            y: right.values,
        };
        Ok(Self(FactorWidth2 {
            a: left.rows,
            b: right.rows,
            weights,
            columns,
        }))
    }
}
