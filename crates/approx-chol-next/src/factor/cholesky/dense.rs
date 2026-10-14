//! Exact Cholesky of a small SDDM by GTH elimination: every pivot a sum of nonnegative terms, so none cancels or fails.

use crate::factor::sddm::Sddm;
use crate::Real;

/// A Laplacian's packed strict lower triangle, free vertex last, then `p⁻¹`; reserved before its matrix is built, not `Clone`, which would drop the capacity.
#[derive(Debug)]
pub(super) struct PackedLaplacian<T> {
    values: Vec<T>,
    dimension: usize,
}

impl<T: Real> PackedLaplacian<T> {
    /// A packed `dimension + 1` rows, room for either gauge's free row while GTH works, then `dimension` inverse pivots, contiguous as every row's fill reads them (strided in the diagonal measured +12-18% at n 16-24); `None` when it will not fit.
    pub(super) fn reserve(dimension: usize) -> Option<Self> {
        let rows = dimension.checked_add(1)?;
        let scalars = (rows.checked_mul(rows.checked_add(1)?)? / 2).checked_add(dimension)?;
        let mut values = Vec::new();
        values.try_reserve_exact(scalars).ok()?;
        Some(Self { values, dimension })
    }

    /// Left-looking GTH on the packed lower `−w` with a free last row `m` never pivoted (`m = n` holding the ground's `−s` grounded, the anchor `m = n − 1` anchored): for each column `j < m` in order, every stored `(r, j)`, `r > j`, subtracts `Σ_{k<j} min·(max·p_k⁻¹)` of the magnitudes of the already updated `(r, k)` and `(j, k)`, then `p_j = −Σ_{r>j} (r, j)` and `p_j⁻¹` goes to the tail; a last pass scales each column `j` by `√p_j⁻¹` and writes `√p_j` on its diagonal, and truncating row `m` and the tail leaves `m` rows; nothing cancels, under- or overflows (`d − Σl²` cancelled to 7e-2 on within's grounded dumps, GTH 3e-15, for 11-80% more factor time).
    pub(super) fn factor(self, sddm: &Sddm<'_, T>) -> DenseCholesky<T> {
        let Self {
            mut values,
            dimension,
        } = self;
        assert_eq!(sddm.n(), dimension, "reserved for this matrix");
        let m = dimension - usize::from(!sddm.grounded());
        let packed = row_start(m + 1);
        values.resize(packed + m, T::zero());
        for (k, j, w) in sddm.grounded_edges() {
            let (k, j) = (k as usize, j as usize);
            values[row_start(j) + k] = -w;
        }
        let (matrix, inverse_pivots) = values.split_at_mut(packed);
        for col in 0..m {
            let pivot_row = row_start(col);
            let mut diagonal = T::zero();
            for row in col + 1..=m {
                let start = row_start(row);
                let mut value = matrix[start + col];
                for k in 0..col {
                    let (a, b) = (matrix[start + k], matrix[pivot_row + k]);
                    // Both are `−w`: max is the smaller weight, min the larger.
                    value = value - a.max(b) * (a.min(b) * inverse_pivots[k]);
                }
                matrix[start + col] = value;
                diagonal = diagonal - value;
            }
            inverse_pivots[col] = T::one() / diagonal;
        }
        for share in inverse_pivots.iter_mut() {
            *share = share.sqrt();
        }
        for row in 0..m {
            let start = row_start(row);
            for (value, &scale) in matrix[start..start + row].iter_mut().zip(&*inverse_pivots) {
                *value = *value * scale;
            }
            matrix[start + row] = T::one() / inverse_pivots[row];
        }
        values.truncate(row_start(m));
        values.shrink_to_fit();
        DenseCholesky { values }
    }
}

/// Where packed row `row` starts; in range once [`PackedLaplacian::reserve`] fit `row`.
fn row_start(row: usize) -> usize {
    row * (row + 1) / 2
}

/// Packed lower triangle of `L` in `LLᵀ`, the free slot excluded: an upper one would double the factor.
#[derive(Clone, Debug)]
pub(super) struct DenseCholesky<T> {
    values: Vec<T>,
}

impl<T: Real> DenseCholesky<T> {
    fn rows(&self) -> usize {
        ((8 * self.values.len() + 1).isqrt() - 1) / 2
    }

    /// Forward, backward, and the free last slot pinned to `0`.
    pub(super) fn substitute(&self, values: &mut [T]) {
        let (pinned, solved) = values.split_last_mut().expect("a free slot");
        let m = solved.len();
        debug_assert_eq!(m, self.rows(), "one value per row and the free slot");
        for row in 0..m {
            let entries = &self.values[row_start(row)..=row_start(row) + row];
            let mut value = solved[row];
            for (&entry, &solution) in entries[..row].iter().zip(&solved[..row]) {
                value = value - entry * solution;
            }
            solved[row] = value / entries[row];
        }
        for row in (0..m).rev() {
            let mut value = solved[row];
            // Strided down column `row`, unlike the forward pass, which is why this keeps an index.
            for (below, &solution) in (row + 1..m).zip(&solved[row + 1..]) {
                value = value - self.values[row_start(below) + row] * solution;
            }
            solved[row] = value / self.values[row_start(row) + row];
        }
        *pinned = T::zero();
    }
}
