//! Factor-width-2 views whose shape is bipartite: what the caller knows, stated by the type.

use num_traits::PrimInt;

/// Why a view's `new` rejected its slices; `column` is the offending column of `V`.
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
    /// A row index is negative or not below its side's size.
    RowOutOfRange {
        /// The offending column.
        column: usize,
    },
    /// A loading is NaN or infinite.
    NonFiniteValue {
        /// The offending column.
        column: usize,
    },
    /// A weight is negative, NaN or infinite.
    InvalidWeight {
        /// The offending column.
        column: usize,
    },
}

impl core::fmt::Display for ViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid bipartite view: {self:?}")
    }
}

impl std::error::Error for ViewError {}

fn row_in_range<I: PrimInt>(row: I, n: u32) -> bool {
    row.to_u32().is_some_and(|row| row < n)
}

fn check_weights(weights: Option<&[f64]>, m: usize) -> Result<(), ViewError> {
    let Some(weights) = weights else {
        return Ok(());
    };
    if weights.len() != m {
        return Err(ViewError::LengthMismatch {
            expected: m,
            found: weights.len(),
        });
    }
    match weights.iter().position(|w| !(w.is_finite() && *w >= 0.0)) {
        Some(column) => Err(ViewError::InvalidWeight { column }),
        None => Ok(()),
    }
}

/// Weighted bipartite multigraph; its matrix is `B W Bᵀ`, the signless Laplacian, left rows first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Graph<'a, I = u32> {
    n_left: u32,
    n_right: u32,
    left: &'a [I],
    right: &'a [I],
    weights: Option<&'a [f64]>,
}

impl<'a, I: PrimInt> Graph<'a, I> {
    /// Column `k` joins `left[k]` and `right[k]` with weight `weights[k]`; `None` is `W = I`.
    pub fn new(
        n_left: u32,
        n_right: u32,
        left: &'a [I],
        right: &'a [I],
        weights: Option<&'a [f64]>,
    ) -> Result<Self, ViewError> {
        let m = left.len();
        if right.len() != m {
            return Err(ViewError::LengthMismatch {
                expected: m,
                found: right.len(),
            });
        }
        if let Some(column) =
            (0..m).find(|&k| !row_in_range(left[k], n_left) || !row_in_range(right[k], n_right))
        {
            return Err(ViewError::RowOutOfRange { column });
        }
        check_weights(weights, m)?;
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

    /// `(left, right, weight)` per column, in column order.
    pub(crate) fn columns(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.left.len()).map(|k| {
            (
                self.left[k].to_usize().expect("validated row"),
                self.right[k].to_usize().expect("validated row"),
                self.weights.map_or(1.0, |w| w[k]),
            )
        })
    }
}

/// One side of `V`: at most one nonzero per column, column `k` is `values[k]·e_{rows[k]}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Side<'a, I = u32> {
    n: u32,
    rows: &'a [I],
    values: &'a [f64],
}

impl<'a, I: PrimInt> Side<'a, I> {
    /// `n` rows; `rows` and `values` give each column's one entry.
    pub fn new(n: u32, rows: &'a [I], values: &'a [f64]) -> Result<Self, ViewError> {
        if values.len() != rows.len() {
            return Err(ViewError::LengthMismatch {
                expected: rows.len(),
                found: values.len(),
            });
        }
        if let Some(column) = rows.iter().position(|&row| !row_in_range(row, n)) {
            return Err(ViewError::RowOutOfRange { column });
        }
        if let Some(column) = values.iter().position(|v| !v.is_finite()) {
            return Err(ViewError::NonFiniteValue { column });
        }
        Ok(Self { n, rows, values })
    }

    fn row(&self, k: usize) -> usize {
        self.rows[k].to_usize().expect("validated row")
    }
}

/// `A = V W Vᵀ` with `V = [left; right]`, each side at most one nonzero per column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gram<'a, I = u32> {
    left: Side<'a, I>,
    right: Side<'a, I>,
    weights: Option<&'a [f64]>,
}

/// One column of `V` with its weight; `x` loads the left row, `y` the right.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Observation {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) w: f64,
}

impl Observation {
    /// One expression for every reader: the forest's cancellation test needs bitwise-equal sums.
    #[inline]
    pub(crate) fn cell(&self) -> f64 {
        self.w * self.x * self.y
    }
}

impl<'a, I: PrimInt> Gram<'a, I> {
    /// Both sides must have the same column count; `None` weights are `W = I`.
    pub fn new(
        left: Side<'a, I>,
        right: Side<'a, I>,
        weights: Option<&'a [f64]>,
    ) -> Result<Self, ViewError> {
        let m = left.rows.len();
        if right.rows.len() != m {
            return Err(ViewError::LengthMismatch {
                expected: m,
                found: right.rows.len(),
            });
        }
        check_weights(weights, m)?;
        Ok(Self {
            left,
            right,
            weights,
        })
    }

    pub(crate) fn sizes(&self) -> (usize, usize) {
        (self.left.n as usize, self.right.n as usize)
    }

    pub(crate) fn n_columns(&self) -> usize {
        self.left.rows.len()
    }

    #[inline]
    pub(crate) fn observation(&self, k: usize) -> Observation {
        Observation {
            row: self.left.row(k),
            col: self.right.row(k),
            x: self.left.values[k],
            y: self.right.values[k],
            w: self.weights.map_or(1.0, |w| w[k]),
        }
    }
}
