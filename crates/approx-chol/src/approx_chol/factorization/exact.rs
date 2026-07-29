use super::block::BlockDim;
use super::factor::Fallback;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::graph::{AdjListGraph, EdgeCount, Neighbor};
use crate::types::Real;
use crate::{DenseFailure, UnusablePivot};

/// Why the dense backend declined a block, in that block's own numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NotFactorable {
    InvalidPivot { pivot: usize, failure: DenseFailure },
    WillNotFit { dim: usize },
}

impl NotFactorable {
    /// `vertices` is `None` when the block is the whole graph, where a local vertex
    /// is already a global one.
    pub(crate) fn at(self, vertices: Option<&[u32]>) -> Fallback {
        match self {
            Self::InvalidPivot { pivot, failure } => Fallback::InvalidPivot(UnusablePivot {
                vertex: vertices.map_or(pivot, |vertices| vertices[pivot] as usize),
                failure,
            }),
            Self::WillNotFit { dim } => Fallback::WillNotFit { dim },
        }
    }
}

pub(crate) fn factor<T: Real, C: EdgeCount>(
    graph: &AdjListGraph<C, T>,
    diagonal: &[T],
    dim: BlockDim,
) -> Result<LowerTriangular<T>, NotFactorable> {
    assemble(graph, diagonal, dim.solved())?.factor_in_place()
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
fn assemble<T: Real, C: EdgeCount>(
    graph: &AdjListGraph<C, T>,
    diagonal: &[T],
    m: usize,
) -> Result<LowerTriangular<T>, NotFactorable> {
    let mut matrix = LowerTriangular::zeros(m)?;
    let mut neighbors: Vec<Neighbor<T, C>> = Vec::with_capacity(m);
    for row in 0..m {
        graph.live_neighbors(row, &mut neighbors);
        let target = matrix.row_mut(row);
        target[row] = diagonal[row];
        for neighbor in &neighbors {
            let col = neighbor.to as usize;
            // The graph carries no self-loop, so `col < row` drops only the mirror
            // above the diagonal and the pinned columns at or past `m`.
            if col < row {
                target[col] = target[col] - neighbor.fill_weight;
            }
        }
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
    pub(super) fn validate_for_dim(&self, dim: BlockDim) -> Result<(), FactorError> {
        let m = dim.solved();
        if packed_len(m) != Some(self.values.len()) {
            return Err(FactorError::ExactFactorLengthInvalid {
                n: dim.total(),
                len: self.values.len(),
            });
        }
        for row in 0..m {
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
