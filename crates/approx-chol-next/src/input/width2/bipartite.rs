//! `V = [V_l; V_r]`: two width-1 sides on disjoint rows, so `A` numbers left rows first, then `n_left +` right ones.

use std::sync::OnceLock;

use num_traits::PrimInt;

use super::edges::{Edges, Weights};
use super::{index, Column, IntoColumns, ViewError, Weight};
use crate::storage::{narrow, widen};
use crate::Real;

/// Which row `rows[k]` each column of `V` hits, under `W`, with one value vector per side `V_c = P diag(values_c)`; each side's diagonal is summed once.
#[derive(Debug)]
pub struct Incidence<'a, T, I = u32> {
    n: usize,
    rows: &'a [I],
    values: Vec<Option<&'a [T]>>,
    weights: Weights<'a, T>,
    diagonals: OnceLock<Box<[T]>>,
}

impl<'a, T: Real, I: PrimInt> Incidence<'a, T, I> {
    /// `n` rows; side `c` puts `values[c][k]` in column `k`, `1` for `None`; rows and weights are checked.
    pub fn new(
        n: usize,
        rows: &'a [I],
        values: Vec<Option<&'a [T]>>,
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        let columns = rows.len();
        if let Some(found) = (values.iter().flatten())
            .map(|v| v.len())
            .find(|&len| len != columns)
        {
            return Err(ViewError::LengthMismatch {
                expected: columns,
                found,
            });
        }
        weights.check(columns)?;
        let in_range = |row: &I| row.to_usize().is_some_and(|row| row < n);
        if let Some(column) = rows.iter().position(|row| !in_range(row)) {
            return Err(ViewError::IndexOutOfRange { column });
        }
        Ok(Self {
            n,
            rows,
            values,
            weights,
            diagonals: OnceLock::new(),
        })
    }

    /// Side `c`, borrowing this incidence; panics unless it has one.
    pub fn side(&self, c: usize) -> Side<'_, T, I> {
        assert!(c < self.values.len(), "the incidence has no side {c}");
        Side {
            incidence: self,
            index: c,
        }
    }

    /// Every side's diagonal, side `c` at `c·n..(c+1)·n`, summed in `f64` on first read like `abs_row_sums`.
    fn diagonals(&self) -> &[T] {
        self.diagonals.get_or_init(|| {
            let n = self.n;
            let mut sums = vec![0.0_f64; self.values.len() * n];
            for (values, sums) in self.values.iter().zip(sums.chunks_exact_mut(n)) {
                for (k, row) in self.rows.iter().enumerate() {
                    let v = values.map_or(1.0, |values| widen(values[k]));
                    sums[index(*row)] += self.weights.at(k).bilinear(v, v);
                }
            }
            sums.into_iter().map(narrow).collect()
        })
    }
}

/// One side `V_c` of a bipartite Gram, borrowed from its incidence: one nonzero per column.
#[derive(Debug)]
pub struct Side<'a, T, I = u32> {
    incidence: &'a Incidence<'a, T, I>,
    index: usize,
}

impl<T, I> Clone for Side<'_, T, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, I> Copy for Side<'_, T, I> {}

impl<'a, T: Real, I: PrimInt> Side<'a, T, I> {
    fn values(self) -> Option<&'a [T]> {
        self.incidence.values[self.index]
    }

    fn diagonal(self) -> &'a [T] {
        let n = self.incidence.n;
        &self.incidence.diagonals()[self.index * n..][..n]
    }
}

/// `A = V W Vᵀ` with `V = [V_l; V_r]`; `W` stays apart because every pair of an incidence's sides shares it.
#[derive(Debug)]
pub struct BipartiteGram<'a, T, I = u32> {
    left: Side<'a, T, I>,
    right: Side<'a, T, I>,
}

impl<T, I> Clone for BipartiteGram<'_, T, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, I> Copy for BipartiteGram<'_, T, I> {}

impl<'a, T: Real, I: PrimInt> BipartiteGram<'a, T, I> {
    /// Both sides must have the same column count and read the same `W`, and `n_left + n_right` fit a `u32` index.
    pub fn new(left: Side<'a, T, I>, right: Side<'a, T, I>) -> Result<Self, ViewError> {
        let (l, r) = (left.incidence, right.incidence);
        if l.rows.len() != r.rows.len() {
            return Err(ViewError::LengthMismatch {
                expected: l.rows.len(),
                found: r.rows.len(),
            });
        }
        if !l.weights.same(r.weights) {
            return Err(ViewError::WeightsDiffer);
        }
        let count = l.n.saturating_add(r.n);
        if u32::try_from(count).is_err() {
            return Err(ViewError::TooLarge { count });
        }
        Ok(Self { left, right })
    }

    pub(crate) fn n_left(&self) -> usize {
        self.left.incidence.n
    }

    /// `A`'s diagonal, left rows then right ones; the first Gram of an incidence sums each side's, serially, as a rayon initializer deadlocks when Grams reduce in parallel.
    pub(crate) fn diagonals(&self) -> (&'a [T], &'a [T]) {
        (self.left.diagonal(), self.right.diagonal())
    }

    #[inline(always)]
    fn with_values(self, weight: impl Fn(usize) -> Weight<T>, visit: impl FnMut(Column<T>)) {
        let one = |_| T::one();
        match (self.left.values(), self.right.values()) {
            (None, None) => self.columns(one, one, weight, visit),
            (Some(a), None) => self.columns(|k| a[k], one, weight, visit),
            (None, Some(b)) => self.columns(one, |k| b[k], weight, visit),
            (Some(a), Some(b)) => self.columns(|k| a[k], |k| b[k], weight, visit),
        }
    }

    #[inline(always)]
    fn columns(
        self,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        weight: impl Fn(usize) -> Weight<T>,
        mut visit: impl FnMut(Column<T>),
    ) {
        let n_left = self.n_left();
        let (left, right) = (self.left.incidence.rows, self.right.incidence.rows);
        for (k, (a, b)) in left.iter().zip(right).enumerate() {
            visit(Column {
                a: index(*a) as u32,
                b: (n_left + index(*b)) as u32,
                v_a: v_a(k),
                v_b: v_b(k),
                weight: weight(k),
            });
        }
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for BipartiteGram<'_, T, I> {
    fn n(&self) -> usize {
        self.left.incidence.n + self.right.incidence.n
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        match self.left.incidence.weights {
            Weights::Unit => self.with_values(|_| Weight::Unit, visit),
            Weights::Of(w) => self.with_values(|k| Weight::Of(w[k]), visit),
            Weights::SqrtOf(s) => self.with_values(|k| Weight::SqrtOf(s[k]), visit),
        }
    }
}

/// Weighted bipartite multigraph; its matrix is the signless Laplacian `|B| W |B|ᵀ`, singular by shape.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteGraph<'a, T, I = u32> {
    n_left: usize,
    n_right: usize,
    edges: Edges<'a, T, I>,
}

impl<'a, T: Real, I: PrimInt> BipartiteGraph<'a, T, I> {
    /// Column `k` joins `left[k] < n_left` and `right[k] < n_right` under `weights`; indices and weights are checked, so `C ≥ 0` folds.
    pub fn new(
        n_left: usize,
        n_right: usize,
        left: &'a [I],
        right: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        let edges = Edges::new(left, right, weights)?;
        let count = n_left.saturating_add(n_right);
        if u32::try_from(count).is_err() {
            return Err(ViewError::TooLarge { count });
        }
        let in_range = |row: &I, n: usize| row.to_usize().is_some_and(|row| row < n);
        let mut ends = left.iter().zip(right);
        if let Some(column) = ends.position(|(l, r)| !in_range(l, n_left) || !in_range(r, n_right))
        {
            return Err(ViewError::IndexOutOfRange { column });
        }
        Ok(Self {
            n_left,
            n_right,
            edges,
        })
    }

    pub(crate) fn n_left(&self) -> usize {
        self.n_left
    }
}

/// Weighted bipartite multigraph; its matrix is the Laplacian `B W Bᵀ`, a Z-matrix and singular by shape.
#[derive(Clone, Copy, Debug)]
pub struct BipartiteLaplacian<'a, T, I = u32>(BipartiteGraph<'a, T, I>);

impl<'a, T: Real, I: PrimInt> BipartiteLaplacian<'a, T, I> {
    /// The checks of [`BipartiteGraph::new`].
    pub fn new(
        n_left: usize,
        n_right: usize,
        left: &'a [I],
        right: &'a [I],
        weights: Weights<'a, T>,
    ) -> Result<Self, ViewError> {
        BipartiteGraph::new(n_left, n_right, left, right, weights).map(Self)
    }

    pub(crate) fn n_left(&self) -> usize {
        self.0.n_left
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for BipartiteLaplacian<'_, T, I> {
    fn n(&self) -> usize {
        self.0.n()
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        self.0
            .edges
            .for_each_column(self.0.n_left, |_| -T::one(), visit)
    }
}

impl<T: Real, I: PrimInt> IntoColumns<T> for BipartiteGraph<'_, T, I> {
    fn n(&self) -> usize {
        self.n_left + self.n_right
    }

    fn for_each_column(self, visit: impl FnMut(Column<T>)) {
        self.edges.for_each_column(self.n_left, |_| T::one(), visit)
    }
}
