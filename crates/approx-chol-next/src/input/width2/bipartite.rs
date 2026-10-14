//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, left rows numbered first.

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::{BipartiteColumns, Column, IntoColumns, ViewError};
use crate::Real;

/// One side of `V`, with at most one nonzero per column: `values[k]` at `rows[k]`, `1` for `None`.
#[derive(Debug)]
pub struct FactorWidth1<'a, T, I = u32> {
    n: usize,
    rows: &'a [I],
    values: Option<&'a [T]>,
}

impl<'a, T: Real, I: PrimInt> FactorWidth1<'a, T, I> {
    /// `n` rows; each column's row is below `n`, and values, if any, are one per column.
    pub fn new(n: usize, rows: &'a [I], values: Option<&'a [T]>) -> Result<Self, ViewError> {
        let expected = rows.len();
        if let Some(found) = values.map(<[T]>::len).filter(|&len| len != expected) {
            return Err(ViewError::LengthMismatch { expected, found });
        }
        let in_range = |row: &I| row.to_usize().is_some_and(|row| row < n);
        if let Some(column) = rows.iter().position(|row| !in_range(row)) {
            return Err(ViewError::IndexOutOfRange { column });
        }
        Ok(Self { n, rows, values })
    }
}

impl<T, I> Clone for FactorWidth1<'_, T, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, I> Copy for FactorWidth1<'_, T, I> {}

/// `A = V W Vᵀ` with `V = [V_l; V_r]`, both width 1 on disjoint rows, left rows numbered first.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteGram<'a, T, I = u32> {
    n_left: usize,
    edges: Edges<'a, T, I>,
    values: [Option<&'a [T]>; 2],
    n: usize,
}

impl<'a, T: Real, I: PrimInt> BipartiteGram<'a, T, I> {
    /// Both factors share the column count, the weights are checked, and `n_left + n_right` fits a `u32` index.
    pub fn new(
        left: FactorWidth1<'a, T, I>,
        right: FactorWidth1<'a, T, I>,
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(left.rows, right.rows, weights)?;
        let count = left.n.saturating_add(right.n);
        if u32::try_from(count).is_err() {
            return Err(ViewError::TooLarge { count });
        }
        Ok(Self {
            n_left: left.n,
            edges,
            values: [left.values, right.values],
            n: count,
        })
    }

    /// Both factors incidence: `A = |B| W |B|ᵀ`, read as the Laplacian it folds from.
    pub(crate) fn laplacian(self) -> Option<BipartiteLaplacian<'a, T, I>> {
        let [None, None] = self.values else {
            return None;
        };
        Some(BipartiteLaplacian {
            n_left: self.n_left,
            n_right: self.n - self.n_left,
            edges: self.edges,
        })
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for BipartiteGram<'_, T, I> {
    fn n(&self) -> usize {
        self.n
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        let (edges, n_left) = (self.edges, self.n_left);
        let one = |_| T::one();
        match self.values {
            [None, None] => edges.for_each_column(n_left, one, one, visit),
            [Some(a), None] => edges.for_each_column(n_left, |k| a[k], one, visit),
            [None, Some(b)] => edges.for_each_column(n_left, one, |k| b[k], visit),
            [Some(a), Some(b)] => edges.for_each_column(n_left, |k| a[k], |k| b[k], visit),
        }
    }
}

/// The same columns' Laplacian `B W Bᵀ`, which negating one side folds back into `|B| W |B|ᵀ`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BipartiteLaplacian<'a, T, I = u32> {
    n_left: usize,
    n_right: usize,
    edges: Edges<'a, T, I>,
}

impl<T: Real, I: PrimInt> IntoColumns<T> for BipartiteLaplacian<'_, T, I> {
    fn n(&self) -> usize {
        self.n_left + self.n_right
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        (self.edges).for_each_column(self.n_left, |_| T::one(), |_| -T::one(), visit)
    }
}

impl<T: Real, I: PrimInt> BipartiteColumns<T> for BipartiteGram<'_, T, I> {
    fn n_left(&self) -> usize {
        self.n_left
    }
}

impl<T: Real, I: PrimInt> BipartiteColumns<T> for BipartiteLaplacian<'_, T, I> {
    fn n_left(&self) -> usize {
        self.n_left
    }
}
