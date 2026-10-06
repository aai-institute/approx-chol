use super::canonical::{column, row_ptr};
use crate::sddm::{Checked, DegreeOverflow, SumOverflow, UpperRows};
use crate::types::{count_as_scalar, Real};
use crate::{NotSddm, Sddm};
use num_traits::PrimInt;

/// A merge-join only because canonical rows guarantee each entry is claimed once.
struct Mirrors<'a, J, T> {
    row_ptrs: &'a [J],
    col_indices: &'a [J],
    values: &'a [T],
    /// `nnz` fits `u32`, which the caller checks.
    cursors: Vec<u32>,
}

impl<'a, J: PrimInt, T: Real> Mirrors<'a, J, T> {
    fn new(row_ptrs: &'a [J], col_indices: &'a [J], values: &'a [T]) -> Self {
        let cursors = row_ptrs[..row_ptrs.len() - 1]
            .iter()
            .map(|&ptr| row_ptr(ptr) as u32)
            .collect();
        Self {
            row_ptrs,
            col_indices,
            values,
            cursors,
        }
    }

    /// Stored zeros count as absent.
    fn claim(&mut self, row: usize, col: usize) -> Result<T, NotSddm> {
        let row_end = row_ptr(self.row_ptrs[row + 1]) as u32;
        let mut cursor = self.cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            // Converted where it is read, so canonical input is never copied.
            let at = column(self.col_indices[cursor as usize]);
            if at > col {
                break;
            }
            let value = self.values[cursor as usize];
            if !value.is_finite() {
                return Err(NotSddm::NonFiniteValue {
                    position: cursor as usize,
                });
            }
            cursor += 1;
            if at == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                self.cursors[row] = cursor;
                return Err(NotSddm::Asymmetric { edge: (at, row) });
            }
        }
        self.cursors[row] = cursor;
        Ok(found)
    }
}

fn approximately_equal<T: Real>(left: T, right: T) -> bool {
    if left == right {
        return true;
    }
    let ulps = T::from(8.0).unwrap_or_else(T::one);
    let scale = left.abs().max(right.abs());
    (left - right).abs() <= ulps * T::epsilon() * scale
}

/// Reads each entry once, so canonical input needs no finiteness scan; the upper mirror is kept.
pub(super) fn sddm_of<J: PrimInt, T: Real>(
    row_ptrs: &[J],
    col_indices: &[J],
    values: &[T],
    terms: impl Iterator<Item = u32>,
) -> Result<Sddm<T>, NotSddm> {
    let n = row_ptrs.len() - 1;
    let mut mirrors = Mirrors::new(row_ptrs, col_indices, values);

    let mut diagonal = vec![T::zero(); n];
    // Off-diagonal only; the diagonal joins in `surplus`.
    let mut row_sums = vec![T::zero(); n];
    let upper_estimate = col_indices.len().saturating_sub(n) / 2;
    let mut rows = UpperRows::with_capacity(n, upper_estimate);

    for row in 0..n {
        let row_end = row_ptr(row_ptrs[row + 1]) as u32;
        // Claimed like any mirror: claiming diagonals up front would skip those below.
        diagonal[row] = mirrors.claim(row, row)?;
        let mut cursor = mirrors.cursors[row];

        while cursor < row_end {
            let col = column(col_indices[cursor as usize]);
            let upper = values[cursor as usize];
            if !upper.is_finite() {
                return Err(NotSddm::NonFiniteValue {
                    position: cursor as usize,
                });
            }
            cursor += 1;
            // Duplicates can coalesce to exactly zero, which contributes no edge.
            if upper == T::zero() {
                continue;
            }
            let lower = mirrors.claim(col, row)?;
            if !approximately_equal(upper, lower) {
                return Err(NotSddm::Asymmetric { edge: (row, col) });
            }
            if upper > T::zero() {
                return Err(NotSddm::PositiveOffDiagonal { edge: (row, col) });
            }
            // Each row sums its own value; charging `upper` to both grounds `col` on mirror noise.
            row_sums[row] = row_sums[row] + upper;
            row_sums[col] = row_sums[col] + lower;
            rows.push(col as u32, -upper);
        }
        rows.end_row();
    }
    let checked = rows
        .finish()
        .map_err(|DegreeOverflow { vertex }| NotSddm::NonFiniteRow { row: vertex })?;
    with_surplus(checked, &diagonal, row_sums, terms)
}

/// `row_sums` arrives off-diagonal-only and leaves as each row's surplus.
fn with_surplus<T: Real>(
    checked: Checked<T>,
    diagonal: &[T],
    mut row_sums: Vec<T>,
    terms: impl Iterator<Item = u32>,
) -> Result<Sddm<T>, NotSddm> {
    for (row, ((sum, &d), additions)) in row_sums
        .iter_mut()
        .zip(diagonal.iter())
        .zip(terms)
        .enumerate()
    {
        let excess = d + *sum;
        // Every off-diagonal was negative; subtracting first survives `d > MAX / 2`.
        let scale = (d.abs() - excess) + d;
        // A non-finite sum forces a non-finite scale, so scale alone decides.
        if !scale.is_finite() {
            return Err(NotSddm::NonFiniteRow { row });
        }
        // One floor for both signs, else one sign grounds drift the other dismisses as noise.
        let accumulated = T::epsilon() * scale * count_as_scalar::<T, _>(additions);
        if excess < -accumulated {
            return Err(NotSddm::NotDiagonallyDominant { row });
        }
        *sum = if excess <= accumulated {
            T::zero()
        } else {
            excess
        };
    }
    checked
        .with_surplus(row_sums)
        .map_err(|overflow| match overflow {
            SumOverflow::Diagonal { vertex } => NotSddm::NonFiniteRow { row: vertex },
            SumOverflow::Total => NotSddm::SurplusOverflow,
        })
}
