//! Exact Cholesky of a small `M` by GTH elimination: every pivot a sum of nonnegative terms, so none cancels or fails.

use crate::factor::sddm::{Layout, Sdd};
use crate::factor::sign::cancel;
use crate::factor::within_one;
use crate::sgdd::FactorTail;
use crate::{Batch, Index, Real};

/// GTH on the packed lower `M`, the surplus as a last row `−s`, a floating `M`'s gauge moved last, both dropped (`d − Σl²` reached 7e-2 on within's grounded dumps, GTH 3e-15, for 11-80% more time); returns unit lower `L` with `D⁻¹` on its diagonal, and the gauge.
pub(super) fn factor<T: Real, I: Index>(
    matrix: &Sdd<T, I>,
    tail: Option<FactorTail<'_, T>>,
    values: &mut Vec<T>,
) -> (Vec<T>, Option<usize>) {
    let (n, signed) = (matrix.n(), matrix.layout() == Layout::Signed);
    let pin = (matrix.floats()).then(|| tail.map_or(n - 1, FactorTail::largest));
    let m = n - usize::from(pin.is_some());
    let packed = row_start(m + 1);
    // `D⁻¹` trails the rows, contiguous as every row's fill reads it: strided in the diagonal measured +12-18% at n 16-24.
    values.clear();
    values.resize(packed + m, T::zero());
    let slot = |i: usize| match pin {
        Some(pin) if i == pin => m,
        Some(pin) if i == m => pin,
        _ => i,
    };
    let mut entry = |k: usize, j: usize, a: T| {
        let (k, j) = (slot(k), slot(j));
        values[row_start(k.max(j)) + k.min(j)] = a;
    };
    matrix.grounded_edges(|i, j, w| entry(i.index(), j.index(), -w));
    let (matrix, inverse_pivots) = values.split_at_mut(packed);
    match signed {
        false => eliminate::<T, false>(matrix, inverse_pivots, m),
        true => eliminate::<T, true>(matrix, inverse_pivots, m),
    }
    for row in 0..m {
        let start = row_start(row);
        for (value, &scale) in matrix[start..start + row].iter_mut().zip(&*inverse_pivots) {
            *value = *value * scale;
        }
        matrix[start + row] = inverse_pivots[row];
    }
    let factor = values[..row_start(m)].to_vec();
    debug_assert!(
        columns_within_one(&factor, m),
        "an |L| column sums past one"
    );
    (factor, pin)
}

/// Left-looking GTH over `m` columns into the inverse pivots, the surplus row `m` updated as any other; `SIGNED` entries may be positive, each update adding what it [`cancel`]s to both its rows' surplus.
fn eliminate<T: Real, const SIGNED: bool>(matrix: &mut [T], inverse_pivots: &mut [T], m: usize) {
    let mut gains = vec![T::zero(); if SIGNED { m } else { 0 }];
    for col in 0..m {
        let pivot_row = row_start(col);
        let mut diagonal = T::zero();
        for row in col + 1..=m {
            let start = row_start(row);
            let mut value = matrix[start + col];
            for k in 0..col {
                let (a, b) = (matrix[start + k], matrix[pivot_row + k]);
                if !SIGNED {
                    // Both are `−w`: max is the smaller weight, min the larger.
                    value = value - a.max(b) * (a.min(b) * inverse_pivots[k]);
                    continue;
                }
                let (a_abs, b_abs) = (a.abs(), b.abs());
                let t = a_abs.min(b_abs) * (a_abs.max(b_abs) * inverse_pivots[k]);
                // The surplus row's `−s` only ever grows in `s`.
                let t = match row < m && (a < T::zero()) != (b < T::zero()) {
                    true => -t,
                    false => t,
                };
                // Gated on `cancel`'s own sign test, which then folds away: sparse rows skip a `+0` store per step.
                if row < m && (value < T::zero()) != (-t < T::zero()) && t != T::zero() {
                    let (sum, gain) = cancel(value, -t);
                    gains[row] = gains[row] + gain;
                    gains[col] = gains[col] + gain;
                    value = sum;
                } else {
                    value = value - t;
                }
            }
            if SIGNED && row == m {
                value = value - gains[col];
            }
            matrix[start + col] = value;
            diagonal = diagonal + value.abs();
        }
        inverse_pivots[col] = T::one() / diagonal;
    }
}

/// Each `|L|` column sums to at most one, so neither sweep overflows where the solution fits.
fn columns_within_one<T: Real>(factor: &[T], m: usize) -> bool {
    (0..m).all(|j| within_one((j + 1..m).map(|row| factor[row_start(row) + j])))
}

/// Where packed row `row` starts.
fn row_start(row: usize) -> usize {
    row * (row + 1) / 2
}

/// Forward, backward on a packed factor; a floating factor's `pin`, moved last, zeroed.
pub(super) fn substitute<T: Real, V: Batch<T>>(factor: &[T], pin: Option<usize>, values: &mut [V]) {
    let Some(pin) = pin else {
        return sweeps(factor, values);
    };
    let last = values.len() - 1;
    values.swap(pin, last);
    let (pinned, solved) = values.split_last_mut().expect("the pinned slot");
    sweeps(factor, solved);
    *pinned = V::zero();
    values.swap(pin, last);
}

fn sweeps<T: Real, V: Batch<T>>(factor: &[T], solved: &mut [V]) {
    let m = solved.len();
    debug_assert_eq!(row_start(m), factor.len(), "one value per row");
    for row in 0..m {
        let entries = &factor[row_start(row)..row_start(row) + row];
        let mut value = solved[row];
        for (&entry, &solution) in entries.iter().zip(&solved[..row]) {
            value = value.sub(solution.mul(entry));
        }
        solved[row] = value;
    }
    for row in (0..m).rev() {
        let mut value = solved[row].mul(factor[row_start(row) + row]);
        // Strided down column `row`, unlike the forward pass, which is why this keeps an index.
        for (below, &solution) in (row + 1..m).zip(&solved[row + 1..]) {
            value = value.sub(solution.mul(factor[row_start(below) + row]));
        }
        solved[row] = value;
    }
}

#[cfg(feature = "serde")]
/// A packed factor decoded from bytes: `m` rows, finite, positive pivots.
pub(super) fn fits<T: Real>(factor: &[T], m: usize) -> bool {
    (m.checked_add(1))
        .and_then(|slots| m.checked_mul(slots))
        .map(|p| p / 2)
        == Some(factor.len())
        && factor.iter().all(|v| v.is_finite())
        && (0..m).all(|row| factor[row_start(row) + row] > T::zero())
        && columns_within_one(factor, m)
}
