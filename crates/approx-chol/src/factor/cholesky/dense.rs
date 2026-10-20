//! Exact Cholesky of a small SDDM by GTH elimination: every pivot a sum of nonnegative terms, so none cancels or fails.

use crate::factor::sddm::Sddm;
use crate::factor::within_one;
use crate::{Index, Real};

/// Left-looking GTH on the packed lower `−w` with a free last row `m` never pivoted (`m = n` holding the ground's `−s` grounded, the anchor `m = n − 1` anchored): for each column `j < m` in order, every stored `(r, j)`, `r > j`, subtracts `Σ_{k<j} min·(max·p_k⁻¹)` of the magnitudes of the already updated `(r, k)` and `(j, k)`, then `p_j = −Σ_{r>j} (r, j)` and `p_j⁻¹` goes to the tail; a last pass scales each column `j` by `p_j⁻¹`, so each `|l| ≤ 1` and no product with a solution overflows, and writes `p_j⁻¹` on its diagonal, and truncating row `m` and the tail leaves `m` rows; nothing cancels, under- or overflows (`d − Σl²` cancelled to 7e-2 on within's grounded dumps, GTH 3e-15, for 11-80% more factor time).
/// Packed unit lower `L` of `L D Lᵀ` with `D⁻¹` on its diagonal, the free slot excluded: an upper one would double the factor; the inverse pivots trail the rows, contiguous as every row's fill reads them (strided in the diagonal measured +12-18% at n 16-24).
pub(super) fn factor<T: Real, I: Index>(sddm: &Sddm<T, I>, free: usize) -> Vec<T> {
    let m = sddm.n() - usize::from(!sddm.grounded());
    let packed = row_start(m + 1);
    let mut values = Vec::with_capacity(packed + m);
    values.resize(packed + m, T::zero());
    let slot = |i: usize| match i {
        i if i == free => m,
        i if i == m => free,
        i => i,
    };
    sddm.grounded_edges(|k, j, w| {
        let (k, j) = (slot(k.index()), slot(j.index()));
        values[row_start(k.max(j)) + k.min(j)] = -w;
    });
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
    for row in 0..m {
        let start = row_start(row);
        for (value, &scale) in matrix[start..start + row].iter_mut().zip(&*inverse_pivots) {
            *value = *value * scale;
        }
        matrix[start + row] = inverse_pivots[row];
    }
    values.truncate(row_start(m));
    values.shrink_to_fit();
    debug_assert!(
        columns_within_one(&values, m),
        "an |L| column sums past one"
    );
    values
}

/// Each `|L|` column sums to at most one, so neither sweep overflows where the solution fits.
fn columns_within_one<T: Real>(factor: &[T], m: usize) -> bool {
    (0..m).all(|j| within_one((j + 1..m).map(|row| factor[row_start(row) + j])))
}

/// Where packed row `row` starts.
fn row_start(row: usize) -> usize {
    row * (row + 1) / 2
}

/// Forward, backward on a packed factor, and the free last slot pinned to `0`.
pub(super) fn substitute<T: Real>(factor: &[T], free: usize, values: &mut [T]) {
    let last = values.len() - 1;
    values.swap(free, last);
    let (pinned, solved) = values.split_last_mut().expect("a free slot");
    let m = solved.len();
    debug_assert_eq!(
        row_start(m),
        factor.len(),
        "one value per row and the free slot"
    );
    for row in 0..m {
        let entries = &factor[row_start(row)..row_start(row) + row];
        let mut value = solved[row];
        for (&entry, &solution) in entries.iter().zip(&solved[..row]) {
            value = value - entry * solution;
        }
        solved[row] = value;
    }
    for row in (0..m).rev() {
        let mut value = solved[row] * factor[row_start(row) + row];
        // Strided down column `row`, unlike the forward pass, which is why this keeps an index.
        for (below, &solution) in (row + 1..m).zip(&solved[row + 1..]) {
            value = value - factor[row_start(below) + row] * solution;
        }
        solved[row] = value;
    }
    *pinned = T::zero();
    values.swap(free, last);
}

#[cfg(feature = "serde")]
/// A packed factor decoded from bytes: one row per slot but the free one, finite, positive pivots.
pub(super) fn fits<T: Real>(factor: &[T], slots: usize) -> bool {
    let Some(m) = slots.checked_sub(1) else {
        return false;
    };
    m.checked_mul(slots).map(|p| p / 2) == Some(factor.len())
        && factor.iter().all(|v| v.is_finite())
        && (0..m).all(|row| factor[row_start(row) + row] > T::zero())
        && columns_within_one(factor, m)
}
