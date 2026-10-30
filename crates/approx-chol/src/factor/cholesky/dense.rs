//! Exact Cholesky of a small `M` by GTH elimination: every pivot a sum of nonnegative terms, so none cancels or fails.

use crate::factor::within_one;
use crate::sgdd::{Grounding, Sheets};
use crate::storage::UpperAdjacency;
use crate::{Index, Real};

/// `M` as it is, its off-diagonal signed and its surplus closing it: dense Cholesky needs no cover, fold or dominance.
pub(in crate::factor) struct Signed<T, I> {
    adjacency: UpperAdjacency<T, I>,
    grounding: Grounding<T>,
    /// Off-diagonals of either sign, so GTH tracks the surplus their cancellation leaves; else a Z-matrix.
    frustrated: bool,
}

impl<T: Real, I: Index> Signed<T, I> {
    /// Panics unless nonempty, with one surplus per row.
    pub(in crate::factor) fn new(
        adjacency: UpperAdjacency<T, I>,
        sheets: Sheets,
        grounding: Grounding<T>,
    ) -> Self {
        assert!(adjacency.n() > 0, "M has a vertex");
        assert!(grounding.fits(adjacency.n()), "one surplus per row");
        Self {
            adjacency,
            grounding,
            frustrated: sheets == Sheets::Two,
        }
    }

    pub(super) fn n(&self) -> usize {
        self.adjacency.n()
    }

    /// Singular exactly when balanced and ungrounded: its null vector is `1`, the fold's `F·1` in `A`.
    pub(super) fn floats(&self) -> bool {
        !self.frustrated && !self.grounding.is_grounded()
    }
}

/// GTH on the packed lower `M`, the surplus as a last row `−s`, pinned slot last and dropped (`d − Σl²` reached 7e-2 on within's grounded dumps, GTH 3e-15, for 11-80% more time); returns unit lower `L` with `D⁻¹` on its diagonal.
pub(super) fn factor<T: Real, I: Index>(signed: &Signed<T, I>, pin: usize) -> Vec<T> {
    let n = signed.n();
    let m = n - usize::from(signed.floats());
    let packed = row_start(m + 1);
    // `D⁻¹` trails the rows, contiguous as every row's fill reads it: strided in the diagonal measured +12-18% at n 16-24.
    let mut values = vec![T::zero(); packed + m];
    let slot = |i: usize| match i {
        i if i == pin => m,
        i if i == m => pin,
        i => i,
    };
    let mut entry = |k: usize, j: usize, a: T| {
        let (k, j) = (slot(k), slot(j));
        values[row_start(k.max(j)) + k.min(j)] = a;
    };
    for i in 0..n {
        signed.adjacency.upper(i).for_each(|(j, &a)| entry(i, j, a));
    }
    let ground = |i| signed.grounding.edge(i).map(|s| (i, s));
    (0..n).filter_map(ground).for_each(|(i, s)| entry(i, n, -s));
    let (matrix, inverse_pivots) = values.split_at_mut(packed);
    match signed.frustrated {
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
    values.truncate(row_start(m));
    values.shrink_to_fit();
    debug_assert!(
        columns_within_one(&values, m),
        "an |L| column sums past one"
    );
    values
}

/// Left-looking GTH over `m` columns into the inverse pivots, the surplus row `m` updated as any other; `SIGNED` entries may be positive, each update that shrinks one adding `2 min(|a|, |t|)` to both its rows' surplus, as `|a| + |t| − |a − t|` would by cancellation.
fn eliminate<T: Real, const SIGNED: bool>(matrix: &mut [T], inverse_pivots: &mut [T], m: usize) {
    let mut gains = vec![T::zero(); if SIGNED { m } else { 0 }];
    let two = T::one() + T::one();
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
                // A zero `t` gains nothing; testing it spares sparse rows a `+0` store per step.
                if row < m && (value > T::zero()) == (t > T::zero()) && t != T::zero() {
                    let gain = two * value.abs().min(t.abs());
                    gains[row] = gains[row] + gain;
                    gains[col] = gains[col] + gain;
                }
                value = value - t;
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

/// Forward, backward on a packed factor, and the pinned slot, moved last, zeroed.
pub(super) fn substitute<T: Real>(factor: &[T], pin: usize, values: &mut [T]) {
    let last = values.len() - 1;
    values.swap(pin, last);
    let (pinned, solved) = values.split_last_mut().expect("the pinned slot");
    let m = solved.len();
    debug_assert_eq!(
        row_start(m),
        factor.len(),
        "one value per row and the pinned slot"
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
    values.swap(pin, last);
}

#[cfg(feature = "serde")]
/// A packed factor decoded from bytes: one row per slot but the pinned one, finite, positive pivots.
pub(super) fn fits<T: Real>(factor: &[T], slots: usize) -> bool {
    let Some(m) = slots.checked_sub(1) else {
        return false;
    };
    m.checked_mul(slots).map(|p| p / 2) == Some(factor.len())
        && factor.iter().all(|v| v.is_finite())
        && (0..m).all(|row| factor[row_start(row) + row] > T::zero())
        && columns_within_one(factor, m)
}
