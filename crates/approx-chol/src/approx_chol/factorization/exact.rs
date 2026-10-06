use super::factor::Fallback;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::graph::Component;
use crate::types::Real;
use crate::{DenseFailure, UnusablePivot};

/// Pivots are named in input numbering, so a failure needs no translation downstream.
pub(crate) fn factor<T: Real>(
    component: &Component<'_, T>,
) -> Result<LowerTriangular<T>, Fallback> {
    let view = component.view();
    assemble(component, component.eliminated())?
        .factor_in_place(|pivot| view.global(pivot))
        .map_err(Fallback::InvalidPivot)
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

/// From the input, not an elimination graph: an exactly factored block never needs one built.
fn assemble<T: Real>(
    component: &Component<'_, T>,
    m: usize,
) -> Result<LowerTriangular<T>, Fallback> {
    let mut matrix = LowerTriangular::zeros(m)?;
    let view = component.view();
    if let Component::Grounded { surplus, .. } = component {
        for row in 0..m {
            matrix.row_mut(row)[row] = surplus[view.global(row)];
        }
    }
    // Scattered, because the input stores only the upper triangle.
    for row in 0..m {
        view.upper_row(row, |col, weight| {
            let diagonal = &mut matrix.row_mut(row)[row];
            *diagonal = *diagonal + weight;
            // An edge to the pinned vertex still counts toward this row's diagonal.
            if col < m {
                let diagonal = &mut matrix.row_mut(col)[col];
                *diagonal = *diagonal + weight;
                let slot = &mut matrix.row_mut(col)[row];
                *slot = *slot - weight;
            }
        });
    }
    Ok(matrix)
}

/// Packed lower triangle: an upper one would double the persisted factor and embed the input.
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"),
        try_from = "Vec<T>"
    )
)]
#[derive(Clone, Debug)]
pub(crate) struct LowerTriangular<T> {
    pub(super) values: Vec<T>,
}

impl<T> LowerTriangular<T> {
    /// [`packed_len`] inverted, so the row count is the triangle's own fact, not passed alongside.
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
    fn zeros(m: usize) -> Result<Self, Fallback> {
        let will_not_fit = Fallback::WillNotFit { dim: m };
        let scalars = packed_len(m).ok_or(will_not_fit)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(scalars)
            .map_err(|_| will_not_fit)?;
        values.resize(scalars, T::zero());
        Ok(Self { values })
    }

    /// Indexes `values` directly: borrowing via [`row`](Self::row) measured 1.2–2.1% slower.
    fn factor_in_place(mut self, name: impl Fn(usize) -> usize) -> Result<Self, UnusablePivot> {
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
                return Err(UnusablePivot {
                    vertex: name(col),
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
            // Strided down column `row`, unlike the forward pass, which is why this keeps an index.
            for (offset, &solution) in solved[row + 1..].iter().enumerate() {
                value = value - self.row(row + 1 + offset)[row] * solution;
            }
            solved[row] = value / self.row(row)[row];
        }
        *pinned = T::zero();
    }
}

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for LowerTriangular<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.values.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<T: num_traits::Float> TryFrom<Vec<T>> for LowerTriangular<T> {
    type Error = FactorError;

    fn try_from(values: Vec<T>) -> Result<Self, Self::Error> {
        let lower = Self { values };
        lower.validate_values()?;
        Ok(lower)
    }
}

#[cfg(any(feature = "serde", test))]
impl<T: num_traits::Float> LowerTriangular<T> {
    pub(super) fn validate_values(&self) -> Result<(), FactorError> {
        let rows = self.rows();
        // First, so no length leaves trailing entries unread.
        if packed_len(rows) != Some(self.values.len()) {
            return Err(FactorError::ExactFactorLengthInvalid {
                len: self.values.len(),
            });
        }
        for row in 0..rows {
            let entries = self.row(row);
            // `substitute` divides by each pivot, so one whose reciprocal overflows is unusable.
            let pivot = entries[row];
            if DenseFailure::of(pivot).is_some() || !(T::one() / pivot).is_finite() {
                return Err(FactorError::ExactPivotInvalid { index: row });
            }
            // A row's norm is a diagonal of the factored matrix; an infinite one factors nothing.
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
                .factor_in_place(|pivot| pivot + 7)
                .expect_err("pivot is unusable"),
                UnusablePivot { vertex: 7, failure },
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
