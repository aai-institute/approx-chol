use super::block::BlockDim;
use super::factor::Fallback;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::graph::{BlockVertices, Ingestion};
use crate::types::Real;
use crate::{DenseFailure, UnusablePivot};

/// Why the dense backend declined a block, in that block's own numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NotFactorable {
    InvalidPivot { pivot: usize, failure: DenseFailure },
    WillNotFit { dim: usize },
}

impl NotFactorable {
    pub(crate) fn at(self, block: &BlockVertices<'_>) -> Fallback {
        match self {
            Self::InvalidPivot { pivot, failure } => Fallback::InvalidPivot(UnusablePivot {
                vertex: block.global(pivot),
                failure,
            }),
            Self::WillNotFit { dim } => Fallback::WillNotFit { dim },
        }
    }
}

pub(crate) fn factor<T: Real>(
    ingestion: &Ingestion<'_, T>,
    block: &BlockVertices<'_>,
    dim: BlockDim,
) -> Result<LowerTriangular<T>, NotFactorable> {
    assemble(ingestion, block, dim.solved())?.factor_in_place()
}

const fn row_start(row: usize) -> usize {
    row * (row + 1) / 2
}

/// `None` when the scalar count overflows.
const fn packed_len(m: usize) -> Option<usize> {
    match m.checked_add(1) {
        Some(rows) => match m.checked_mul(rows) {
            Some(scalars) => Some(scalars / 2),
            None => None,
        },
        None => None,
    }
}

/// Lower triangle only: a stored upper triangle would double the persisted factor and
/// embed the input matrix in it.
///
/// Read from the ingested arrays rather than from an elimination graph, because a block
/// that reaches here is never eliminated on and so never needs one built.
fn assemble<T: Real>(
    ingestion: &Ingestion<'_, T>,
    block: &BlockVertices<'_>,
    m: usize,
) -> Result<LowerTriangular<T>, NotFactorable> {
    let mut matrix = LowerTriangular::zeros(m)?;
    for row in 0..m {
        matrix.row_mut(row)[row] = ingestion.block_diagonal(block, row);
    }
    // Scattered from the upper triangle rather than gathered from the lower one: the
    // upper entry is the mirror the whole crate treats as authoritative, and reaching it
    // by row means each block row is still read exactly once.
    for row in 0..m {
        ingestion.upper_row(block, row, |col, value| {
            // Past the triangle is the block's pinned last vertex, whose row and column
            // the dense factor does not carry.
            if col < m {
                let slot = &mut matrix.row_mut(col)[row];
                *slot = *slot + value;
            }
        });
    }
    Ok(matrix)
}

/// Packed: row `r` is its own `r + 1` scalars, so no consumer restates where one
/// starts.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(crate) struct LowerTriangular<T> {
    pub(super) values: Vec<T>,
}

impl<T> LowerTriangular<T> {
    /// [`packed_len`] inverted, so the row count is the triangle's own fact rather
    /// than one every consumer is handed alongside it.
    pub(super) fn rows(&self) -> usize {
        ((8 * self.values.len() + 1).isqrt() - 1) / 2
    }

    #[inline]
    fn row(&self, row: usize) -> &[T] {
        let start = row_start(row);
        &self.values[start..=start + row]
    }

    #[inline]
    fn row_mut(&mut self, row: usize) -> &mut [T] {
        let start = row_start(row);
        &mut self.values[start..=start + row]
    }
}

impl<T: Real> LowerTriangular<T> {
    fn zeros(m: usize) -> Result<Self, NotFactorable> {
        let will_not_fit = NotFactorable::WillNotFit { dim: m };
        let scalars = packed_len(m).ok_or(will_not_fit)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(scalars)
            .map_err(|_| will_not_fit)?;
        values.resize(scalars, T::zero());
        Ok(Self { values })
    }

    /// Indexes `values` directly: the split borrow [`row`](Self::row) would need
    /// measured 1.2–2.1% slower across `n = 128..384` on a complete graph.
    fn factor_in_place(mut self) -> Result<Self, NotFactorable> {
        let m = self.rows();
        let matrix = &mut self.values;
        for col in 0..m {
            let pivot_row = row_start(col);
            let mut diagonal = matrix[pivot_row + col];
            for k in 0..col {
                let value = matrix[pivot_row + k];
                diagonal = diagonal - value * value;
            }
            if let Some(failure) = DenseFailure::of(diagonal) {
                return Err(NotFactorable::InvalidPivot {
                    pivot: col,
                    failure,
                });
            }
            let pivot = diagonal.sqrt();
            matrix[pivot_row + col] = pivot;
            let inverse = T::one() / pivot;
            for row in col + 1..m {
                let start = row_start(row);
                let mut value = matrix[start + col];
                for k in 0..col {
                    value = value - matrix[start + k] * matrix[pivot_row + k];
                }
                matrix[start + col] = value * inverse;
            }
        }
        Ok(self)
    }

    pub(super) fn substitute(&self, values: &mut [T]) {
        let Some((pinned, solved)) = values.split_last_mut() else {
            return;
        };
        let m = solved.len();
        for row in 0..m {
            let entries = self.row(row);
            let mut value = solved[row];
            for (&entry, &solution) in entries[..row].iter().zip(&solved[..row]) {
                value = value - entry * solution;
            }
            solved[row] = value / entries[row];
        }
        for row in (0..m).rev() {
            let mut value = solved[row];
            // Down column `row`, so each factor entry is in a different row: strided,
            // unlike the forward pass, and the reason this one keeps an index.
            for (offset, &solution) in solved[row + 1..].iter().enumerate() {
                value = value - self.row(row + 1 + offset)[row] * solution;
            }
            solved[row] = value / self.row(row)[row];
        }
        *pinned = T::zero();
    }
}

#[cfg(any(feature = "serde", test))]
impl<T: num_traits::Float> LowerTriangular<T> {
    /// The triangle's own row count plus the variable it leaves pinned.
    pub(super) fn pinned_dim(&self) -> Result<BlockDim, FactorError> {
        let rows = self.rows();
        if packed_len(rows) != Some(self.values.len()) {
            return Err(FactorError::ExactFactorLengthInvalid {
                len: self.values.len(),
            });
        }
        Ok(BlockDim::of(rows + 1).expect("a row count plus the pinned variable is non-zero"))
    }

    pub(super) fn validate_values(&self) -> Result<(), FactorError> {
        for row in 0..self.rows() {
            let entries = self.row(row);
            // `substitute` divides by each diagonal entry twice per row, so a
            // pivot whose reciprocal overflows cannot be divided by either.
            let pivot = entries[row];
            if DenseFailure::of(pivot).is_some() || !(T::one() / pivot).is_finite() {
                return Err(FactorError::ExactPivotInvalid { index: row });
            }
            // A real Cholesky factor's row norm is a diagonal of the matrix it
            // factors, so a row that squares to infinity factors nothing.
            let norm = entries
                .iter()
                .fold(T::zero(), |sum, &value| sum + value * value);
            if !norm.is_finite() {
                return Err(FactorError::ExactRowNotRepresentable { row });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unusable_pivot_names_its_cause() {
        let cases = [
            (f64::INFINITY, DenseFailure::NonFinitePivot),
            (f64::NAN, DenseFailure::NonFinitePivot),
            (0.0, DenseFailure::NonPositivePivot),
            (-1.0, DenseFailure::NonPositivePivot),
        ];
        for (diagonal, failure) in cases {
            assert_eq!(
                LowerTriangular {
                    values: vec![diagonal],
                }
                .factor_in_place()
                .expect_err("pivot is unusable"),
                NotFactorable::InvalidPivot { pivot: 0, failure },
                "diagonal {diagonal}"
            );
        }
    }

    #[test]
    fn no_finite_positive_pivot_has_a_non_finite_reciprocal() {
        let extremes = [f64::MIN_POSITIVE, f64::from_bits(1), f64::MAX, 1.0];
        for diagonal in extremes {
            let pivot = diagonal.sqrt();
            assert!(pivot.is_finite() && pivot > 0.0, "sqrt({diagonal:e})");
            assert!((1.0 / pivot).is_finite(), "1/sqrt({diagonal:e})");
        }
    }
}
