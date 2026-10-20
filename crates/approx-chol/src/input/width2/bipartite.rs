//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, left rows numbered first.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::gram::{Bipartite, BipartiteEnds, Gram, One, Values};
use super::{length, IntoColumns, ViewError};
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
pub struct BipartiteGram<'a, T, I = u32> {
    /// The same columns with both sides' values `1`.
    incidence: Gram<'a, T, I, Bipartite, One, One>,
    values: [Option<&'a [T]>; 2],
}

impl<'a, T: Real, I: PrimInt> BipartiteGram<'a, T, I> {
    /// Both factors share the column count; `W = I`.
    pub fn new(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(left.rows, right.rows)?;
        let (n, graph) = (left.n.saturating_add(right.n), Bipartite { n_left: left.n });
        Ok(Self {
            incidence: Gram::new(n, graph, edges, One, One, None),
            values: [left.values, right.values],
        })
    }

    /// `V W Vᵀ` under `weights`, one per column.
    pub fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        let incidence = self.incidence.with_weights(weights)?;
        Ok(Self { incidence, ..self })
    }

    /// `V W Vᵀ + diag(d)`, one finite nonnegative `d_i` per vertex, left rows first.
    pub fn plus_diagonal(self, diagonal: &'a [T]) -> Result<Self, ViewError> {
        let incidence = self.incidence.plus_diagonal(diagonal)?;
        Ok(Self { incidence, ..self })
    }

    pub(crate) fn n(&self) -> usize {
        self.incidence.n()
    }

    pub(crate) fn gram(self) -> BipartiteEnds<'a, T, I> {
        let v = self.incidence;
        match self.values {
            [None, None] => BipartiteEnds::Incidence(v),
            [Some(x), None] => BipartiteEnds::LeftValues(v.with_ends(Values(x), One)),
            [None, Some(y)] => BipartiteEnds::RightValues(v.with_ends(One, Values(y))),
            [Some(x), Some(y)] => BipartiteEnds::Values(v.with_ends(Values(x), Values(y))),
        }
    }
}
