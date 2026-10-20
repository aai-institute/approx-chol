//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, left rows numbered first.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::gram::{Bipartite, Gram, OnBipartite, One, Values};
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

/// `A = V W Vᵀ` with `V = [V_l; V_r]`, both width 1 on disjoint rows, left rows numbered first.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteGram<'a, T, I = u32> {
    n_left: usize,
    edges: Edges<'a, T, I>,
    values: [Option<&'a [T]>; 2],
    n: usize,
}

impl<'a, T: Real, I: PrimInt> BipartiteGram<'a, T, I> {
    /// Both factors share the column count and the weights are checked.
    pub fn new(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(left.rows, right.rows, weights)?;
        let count = left.n.saturating_add(right.n);
        Ok(Self {
            n_left: left.n,
            edges,
            values: [left.values, right.values],
            n: count,
        })
    }

    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// `on` run on this Gram, its sides' ends read as given.
    pub(crate) fn lower<R: OnBipartite<'a, T, I>>(self, on: R) -> R::Output {
        let (n, graph, edges) = (
            self.n,
            Bipartite {
                n_left: self.n_left,
            },
            self.edges,
        );
        match self.values {
            [None, None] => on.run(Gram::new(n, graph, edges, One, One, None)),
            [Some(x), None] => on.run(Gram::new(n, graph, edges, Values(x), One, None)),
            [None, Some(y)] => on.run(Gram::new(n, graph, edges, One, Values(y), None)),
            [Some(x), Some(y)] => on.run(Gram::new(n, graph, edges, Values(x), Values(y), None)),
        }
    }
}
