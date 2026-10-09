//! Factor-width-2 views whose shape is bipartite: what the caller knows, stated by the type.

use std::sync::OnceLock;

use num_traits::PrimInt;

use super::cross_tab::PairColumns;

/// What every accumulation pass reads: levels, optional loadings, optional weights.
pub(crate) type Columns<'a, I> = PairColumns<'a, I, Option<&'a [f64]>, Option<&'a [f64]>>;

/// Why a view's `new` rejected its slices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewError {
    /// A slice's length differs from the column count.
    LengthMismatch {
        /// The column count the first slice fixed.
        expected: usize,
        /// The length found.
        found: usize,
    },
    /// A Gram's two sides were built under different weights.
    WeightsDiffer,
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid bipartite view: {self:?}")
    }
}

impl std::error::Error for ViewError {}

fn check_length(found: usize, expected: usize) -> Result<(), ViewError> {
    match found == expected {
        true => Ok(()),
        false => Err(ViewError::LengthMismatch { expected, found }),
    }
}

/// The diagonal `W`, one nonnegative entry per column of `V`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Weights<'a> {
    /// `W = I`.
    Unit,
    /// `W = diag(w)`.
    Of(&'a [f64]),
    /// `W = diag(s²)`, for a caller that keeps square roots.
    SqrtOf(&'a [f64]),
}

impl Weights<'_> {
    #[inline]
    pub(crate) fn at(self, k: usize) -> f64 {
        match self {
            Weights::Unit => 1.0,
            Weights::Of(w) => w[k],
            Weights::SqrtOf(s) => s[k] * s[k],
        }
    }

    fn check(self, m: usize) -> Result<(), ViewError> {
        match self {
            Weights::Unit => Ok(()),
            Weights::Of(w) | Weights::SqrtOf(w) => check_length(w.len(), m),
        }
    }

    /// The same slice read the same way, so two sides agree on `W` by construction.
    fn same(self, other: Self) -> bool {
        match (self, other) {
            (Weights::Unit, Weights::Unit) => true,
            (Weights::Of(a), Weights::Of(b)) | (Weights::SqrtOf(a), Weights::SqrtOf(b)) => {
                core::ptr::eq(a, b)
            }
            _ => false,
        }
    }
}

/// Weighted bipartite multigraph; its matrix is `B W Bᵀ`, the signless Laplacian, left rows first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Graph<'a, I = u32> {
    n_left: u32,
    n_right: u32,
    left: &'a [I],
    right: &'a [I],
    weights: Weights<'a>,
}

impl<'a, I: PrimInt> Graph<'a, I> {
    /// Column `k` joins `left[k] < n_left` and `right[k] < n_right` under `weights`, unchecked.
    pub fn new(
        n_left: u32,
        n_right: u32,
        left: &'a [I],
        right: &'a [I],
        weights: Weights<'a>,
    ) -> Result<Self, ViewError> {
        check_length(right.len(), left.len())?;
        weights.check(left.len())?;
        Ok(Self {
            n_left,
            n_right,
            left,
            right,
            weights,
        })
    }

    pub(crate) fn sizes(&self) -> (usize, usize) {
        (self.n_left as usize, self.n_right as usize)
    }

    pub(crate) fn columns(&self) -> Columns<'a, I> {
        PairColumns {
            row_levels: self.left,
            col_levels: self.right,
            row_load: None,
            col_load: None,
            weights: self.weights,
        }
    }
}

/// Sides of `V` sharing one row vector, one per loading; their diagonals are summed together, once.
#[derive(Debug)]
pub struct Sides<'a, I = u32> {
    n: u32,
    rows: &'a [I],
    loadings: Vec<Option<&'a [f64]>>,
    weights: Weights<'a>,
    /// Column-major, `n` per loading; summed on first use, so an unused family costs nothing.
    diagonals: OnceLock<Box<[f64]>>,
}

impl<'a, I: PrimInt> Sides<'a, I> {
    /// `n` rows; side `c` loads column `k` with `loadings[c][k]`, `1` for `None`; rows unchecked.
    pub fn new(
        n: u32,
        rows: &'a [I],
        loadings: Vec<Option<&'a [f64]>>,
        weights: Weights<'a>,
    ) -> Result<Self, ViewError> {
        for values in loadings.iter().flatten() {
            check_length(values.len(), rows.len())?;
        }
        weights.check(rows.len())?;
        Ok(Self {
            n,
            rows,
            loadings,
            weights,
            diagonals: OnceLock::new(),
        })
    }

    /// Side `c`, borrowing this family.
    pub fn side(&self, c: usize) -> Side<'_, 'a, I> {
        assert!(
            c < self.loadings.len(),
            "side {c} of {}",
            self.loadings.len()
        );
        Side {
            sides: self,
            index: c,
        }
    }

    fn diagonal(&self, c: usize) -> &[f64] {
        let n = self.n as usize;
        &self.diagonals.get_or_init(|| self.sum_diagonals())[c * n..(c + 1) * n]
    }

    /// One pass over the columns for every loading, in within's order and arithmetic.
    fn sum_diagonals(&self) -> Box<[f64]> {
        let n = self.n as usize;
        let mut diagonals = vec![0.0; n * self.loadings.len()];
        let row = |k: usize| self.rows[k].to_usize().expect("row fits usize");
        match (&self.loadings[..], self.weights) {
            // A sum of ones is exact, so a run of equal rows adds its length.
            ([None], Weights::Unit) => {
                let mut start = 0;
                for run in self.rows.chunk_by(|a, b| a == b) {
                    diagonals[row(start)] += run.len() as f64;
                    start += run.len();
                }
            }
            ([None], weights) => {
                for k in 0..self.rows.len() {
                    diagonals[row(k)] += weights.at(k);
                }
            }
            (loadings, weights) => {
                for k in 0..self.rows.len() {
                    let (r, w) = (row(k), weights.at(k));
                    for (c, z) in loadings.iter().enumerate() {
                        // Keep `w * z * z` left-to-right: a zero weight kills a huge loading first.
                        diagonals[c * n + r] += z.map_or(w, |z| w * z[k] * z[k]);
                    }
                }
            }
        }
        diagonals.into_boxed_slice()
    }
}

/// One side of `V` under `W`: column `k` is `values[k]·e_{rows[k]}`, borrowed from its family.
#[derive(Debug)]
pub struct Side<'s, 'a, I = u32> {
    sides: &'s Sides<'a, I>,
    index: usize,
}

impl<I> Clone for Side<'_, '_, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for Side<'_, '_, I> {}

impl<'s, 'a, I: PrimInt> Side<'s, 'a, I> {
    fn values(&self) -> Option<&'a [f64]> {
        self.sides.loadings[self.index]
    }

    fn diagonal(&self) -> &'s [f64] {
        self.sides.diagonal(self.index)
    }
}

/// `A = V W Vᵀ` over two borrowed sides, so one family serves every pair it is in.
#[derive(Debug)]
pub struct Gram<'s, 'a, I = u32> {
    left: Side<'s, 'a, I>,
    right: Side<'s, 'a, I>,
}

impl<'s, 'a, I: PrimInt> Gram<'s, 'a, I> {
    /// Both sides must have the same column count and the same `W`.
    pub fn new(left: Side<'s, 'a, I>, right: Side<'s, 'a, I>) -> Result<Self, ViewError> {
        check_length(right.sides.rows.len(), left.sides.rows.len())?;
        if !left.sides.weights.same(right.sides.weights) {
            return Err(ViewError::WeightsDiffer);
        }
        Ok(Self { left, right })
    }

    pub(crate) fn sizes(&self) -> (usize, usize) {
        (self.left.sides.n as usize, self.right.sides.n as usize)
    }

    /// `A`'s diagonal, left rows then right ones; the first Gram of a family sums it.
    pub(crate) fn diagonals(&self) -> (&'s [f64], &'s [f64]) {
        (self.left.diagonal(), self.right.diagonal())
    }

    pub(crate) fn columns(&self) -> Columns<'a, I> {
        PairColumns {
            row_levels: self.left.sides.rows,
            col_levels: self.right.sides.rows,
            row_load: self.left.values(),
            col_load: self.right.values(),
            weights: self.left.sides.weights,
        }
    }
}
