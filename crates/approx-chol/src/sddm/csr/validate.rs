use super::canonical::Canonical;
use super::index;
use crate::sddm::{add_edge, Laplacian};
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

    /// The row's diagonal; `entry` sees each nonzero above it as `(col, upper, lower)`, its mirror within tolerance.
    #[inline]
    fn row(
        &mut self,
        row: usize,
        mut entry: impl FnMut(usize, T, T) -> Result<(), Error>,
    ) -> Result<T, Error> {
        // Claimed like any mirror: claiming diagonals up front would skip those below.
        let diagonal = self.claim(row, row)?;
        let row_end = index(self.row_ptrs[row + 1]) as u32;
        let mut cursor = self.cursors[row];
        while cursor < row_end {
            let col = index(self.col_indices[cursor as usize]);
            let upper = self.values[cursor as usize];
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
            let lower = self.claim(col, row)?;
            if !approximately_equal(upper, lower) {
                return Err(Error::Asymmetric { edge: (row, col) });
            }
            entry(col, upper, lower)?;
        }
        Ok(diagonal)
    }
}

/// An admission threshold, measured to keep uniformly scaled solves at unit-scale quality (#163).
fn floor<T: Real>() -> T {
    T::min_positive_value() / T::epsilon()
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
pub(super) fn edges<T: Real, I: PrimInt>(
    canonical: &Canonical<'_, T, I>,
) -> Result<(Laplacian<T>, RowSums<T>), Error> {
    let (row_ptrs, col_indices, values) = canonical.arrays();
    let n = row_ptrs.len() - 1;
    let mut mirrors = Mirrors::new(row_ptrs, col_indices, values);

    let mut sums = RowSums::zeros(n);
    let mut upper_ptrs = Vec::with_capacity(n + 1);
    upper_ptrs.push(0);
    let mut neighbors = Vec::with_capacity(col_indices.len() / 2);
    let mut weights = Vec::with_capacity(col_indices.len() / 2);

    for row in 0..n {
        sums.diagonal[row] = mirrors.row(row, |col, upper, lower| {
            if upper > T::zero() {
                return Err(Error::PositiveOffDiagonal { edge: (row, col) });
            }
            if -upper < floor() {
                return Err(Error::MagnitudeTooSmall { entry: (row, col) });
            }
            sums.add(row, col, upper, lower);
            neighbors.push(col as u32);
            weights.push(-upper);
            Ok(())
        })?;
        upper_ptrs.push(neighbors.len() as u32);
    }
    let laplacian = Laplacian {
        row_ptrs: upper_ptrs,
        neighbors,
        weights,
    };
    Ok((laplacian, sums))
}

/// What the balance verdict reads per row, summed as the walk claims each edge.
pub(super) struct RowSums<T> {
    diagonal: Vec<T>,
    /// Off-diagonal only; the diagonal joins in the balance verdict.
    off_diagonal: Vec<T>,
    /// Summed as ingestion sums them, so the diagonal checked here is the one it reads.
    degrees: Vec<T>,
}

impl<T: Real> RowSums<T> {
    fn zeros(n: usize) -> Self {
        Self {
            diagonal: vec![T::zero(); n],
            off_diagonal: vec![T::zero(); n],
            degrees: vec![T::zero(); n],
        }
    }

    #[inline]
    fn add(&mut self, row: usize, col: usize, upper: T, lower: T) {
        // Each row sums its own value; charging `upper` to both grounds `col` on mirror noise.
        self.off_diagonal[row] = self.off_diagonal[row] + upper;
        self.off_diagonal[col] = self.off_diagonal[col] + lower;
        add_edge(&mut self.degrees, row, col, -upper);
    }

    /// Judged in row order, so the first bad row is the one reported.
    pub(super) fn surplus(self, terms: impl Iterator<Item = u32>) -> Result<Vec<T>, Error> {
        let mut surplus = self.off_diagonal;
        for (row, ((sum, &d), (&degree, terms))) in surplus
            .iter_mut()
            .zip(&self.diagonal)
            .zip(self.degrees.iter().zip(terms))
            .enumerate()
        {
            *sum = match RowBalance::of(d, *sum, terms) {
                RowBalance::NonFinite => return Err(Error::NonFiniteRow { row }),
                RowBalance::Deficit => return Err(Error::NotDiagonallyDominant { row }),
                RowBalance::Negligible => T::zero(),
                // The ground edge's weight; after the verdict, so a row not dominant at any scale says so.
                RowBalance::Surplus(excess) if excess < floor() => {
                    return Err(Error::MagnitudeTooSmall { entry: (row, row) })
                }
                RowBalance::Surplus(excess) => excess,
            };
            // Ingestion's diagonal: the degree it sums, not the stored entry.
            if !(degree + *sum).is_finite() {
                return Err(Error::NonFiniteRow { row });
            }
        }
        Ok(surplus)
    }
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
