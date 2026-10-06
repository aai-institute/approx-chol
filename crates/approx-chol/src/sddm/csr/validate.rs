use super::canonical::Canonical;
use super::index;
use crate::sddm::{floor, Sddm, UpperRows};
use crate::types::{count_as_scalar, Real};
use crate::Error;
use num_traits::PrimInt;

/// A merge-join only because [`Canonical`] guarantees each entry is claimed once.
struct Mirrors<'a, T, I> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
    cursors: Vec<u32>,
}

impl<'a, T: Real, I: PrimInt> Mirrors<'a, T, I> {
    fn new(row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T]) -> Self {
        let cursors = row_ptrs[..row_ptrs.len() - 1]
            .iter()
            .map(|&ptr| index(ptr) as u32)
            .collect();
        Self {
            row_ptrs,
            col_indices,
            values,
            cursors,
        }
    }

    /// Stored zeros count as absent.
    fn claim(&mut self, row: usize, col: usize) -> Result<T, Error> {
        let row_end = index(self.row_ptrs[row + 1]) as u32;
        let mut cursor = self.cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            let at = index(self.col_indices[cursor as usize]);
            if at > col {
                break;
            }
            let value = self.values[cursor as usize];
            if !value.is_finite() {
                return Err(Error::NonFiniteValue {
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
                return Err(Error::Asymmetric { edge: (at, row) });
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

/// Reads every stored entry once, so [`Canonical::of`] leaves finiteness here; the upper mirror is kept.
pub(super) fn sddm_of<T: Real, I: PrimInt>(
    canonical: &Canonical<'_, T, I>,
) -> Result<Sddm<T>, Error> {
    let (row_ptrs, col_indices, values) = canonical.arrays();
    let n = row_ptrs.len() - 1;
    let mut mirrors = Mirrors::new(row_ptrs, col_indices, values);

    let mut diagonal = vec![T::zero(); n];
    // Off-diagonal only; the diagonal joins in the balance verdict.
    let mut row_sums = vec![T::zero(); n];
    let mut rows = UpperRows::with_capacity(n, col_indices.len().saturating_sub(n) / 2);

    for row in 0..n {
        let row_end = index(row_ptrs[row + 1]) as u32;
        // Claimed like any mirror: claiming diagonals up front would skip those below.
        diagonal[row] = mirrors.claim(row, row)?;
        let mut cursor = mirrors.cursors[row];

        while cursor < row_end {
            let col = index(col_indices[cursor as usize]);
            let upper = values[cursor as usize];
            if !upper.is_finite() {
                return Err(Error::NonFiniteValue {
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
                return Err(Error::Asymmetric { edge: (row, col) });
            }
            if upper > T::zero() {
                return Err(Error::PositiveOffDiagonal { edge: (row, col) });
            }
            if -upper < floor() {
                return Err(Error::MagnitudeTooSmall { entry: (row, col) });
            }
            // Each row sums its own value; charging `upper` to both grounds `col` on mirror noise.
            row_sums[row] = row_sums[row] + upper;
            row_sums[col] = row_sums[col] + lower;
            rows.push(col as u32, -upper);
        }
        rows.end_row();
    }
    rows.finish(row_sums, |row, sum| {
        match RowBalance::of(diagonal[row], sum, canonical.terms(row)) {
            RowBalance::NonFinite => Err(Error::NonFiniteRow { row }),
            RowBalance::Deficit => Err(Error::NotDiagonallyDominant { row }),
            RowBalance::Negligible => Ok(T::zero()),
            RowBalance::Surplus(excess) => Ok(excess),
        }
    })
}

/// A row's diagonal surplus, judged against the noise its own scale and term count can carry.
enum RowBalance<T> {
    NonFinite,
    Deficit,
    Negligible,
    /// Worth closing with a ground edge.
    Surplus(T),
}

impl<T: Real> RowBalance<T> {
    /// `terms` is how many additions produced `excess`, not the row's degree.
    fn of(diagonal: T, off_diagonal_sum: T, terms: u32) -> Self {
        let excess = diagonal + off_diagonal_sum;
        // Every off-diagonal was negative; subtracting first survives `d > MAX / 2`.
        let scale = (diagonal.abs() - excess) + diagonal;
        // A non-finite sum forces a non-finite scale, so scale alone decides.
        if !scale.is_finite() {
            return Self::NonFinite;
        }
        // One floor for both signs, or a row grounds on drift the opposite sign dismisses as noise.
        let accumulated = T::epsilon() * scale * count_as_scalar::<T, _>(terms);
        if excess < -accumulated {
            return Self::Deficit;
        }
        if excess <= accumulated {
            return Self::Negligible;
        }
        Self::Surplus(excess)
    }
}
