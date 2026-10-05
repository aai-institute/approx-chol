use super::canonical::{column, row_ptr};
use crate::sddm::{surplus_sums_finitely, Diagonal};
use crate::types::{count_as_scalar, Real};
use crate::{Laplacian, NotSddm, Sddm};
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

/// Reads every stored entry of canonical arrays exactly once, which is what lets the
/// canonical path skip a finiteness scan of its own. The upper mirror is the one kept:
/// the whole crate treats it as authoritative.
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
    let mut degrees = Diagonal::starting_at(vec![T::zero(); n]);
    let mut upper_ptrs = Vec::with_capacity(n + 1);
    let mut neighbors = Vec::with_capacity(upper_estimate);
    let mut weights = Vec::with_capacity(upper_estimate);
    upper_ptrs.push(0u32);

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
            // Each row sums the value it stores: charging `upper` to both would read
            // the tolerated mirror difference as `col`'s own surplus and ground it.
            row_sums[row] = row_sums[row] + upper;
            row_sums[col] = row_sums[col] + lower;
            neighbors.push(col as u32);
            weights.push(-upper);
            degrees.add(row, col, -upper);
        }
        upper_ptrs.push(neighbors.len() as u32);
    }
    if let Some(row) = degrees.first_non_finite() {
        return Err(NotSddm::NonFiniteRow { row });
    }
    let laplacian = Laplacian::trusted(upper_ptrs, neighbors, weights);
    with_surplus(laplacian, &diagonal, &degrees, row_sums, terms)
}

/// How far one row's diagonal exceeds its off-diagonal mass, judged against the noise
/// the row's own scale and term count can carry.
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
        // One floor for both signs: forgiving more in one direction grounds a row for
        // drift that the opposite sign would dismiss as noise.
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

/// `row_sums` arrives off-diagonal-only and becomes each row's surplus; a Laplacian when
/// every row balances.
fn with_surplus<T: Real>(
    laplacian: Laplacian<T>,
    diagonal: &[T],
    degrees: &Diagonal<T>,
    mut row_sums: Vec<T>,
    terms: impl Iterator<Item = u32>,
) -> Result<Sddm<T>, NotSddm> {
    for (row, ((sum, &d), count)) in row_sums
        .iter_mut()
        .zip(diagonal.iter())
        .zip(terms)
        .enumerate()
    {
        *sum = match RowBalance::of(d, *sum, count) {
            RowBalance::NonFinite => return Err(NotSddm::NonFiniteRow { row }),
            RowBalance::Deficit => return Err(NotSddm::NotDiagonallyDominant { row }),
            RowBalance::Negligible => T::zero(),
            RowBalance::Surplus(excess) => excess,
        };
    }
    if let Some(row) = degrees.first_non_finite_with(&row_sums) {
        return Err(NotSddm::NonFiniteRow { row });
    }
    if !surplus_sums_finitely(&row_sums) {
        return Err(NotSddm::SurplusOverflow);
    }
    Ok(Sddm::trusted(laplacian, row_sums))
}
