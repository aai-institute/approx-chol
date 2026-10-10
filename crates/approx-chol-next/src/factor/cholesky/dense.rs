//! Exact Cholesky of a small SDDM by GTH elimination: every pivot a sum of nonnegative terms, so none cancels or fails.

use crate::factor::sddm::Sddm;
use crate::Real;

/// Capacity for a packed triangle, taken before its matrix is built; not `Clone`, which would drop the capacity.
#[derive(Debug)]
pub(super) struct Room<T>(Vec<T>);

impl<T: Real> Room<T> {
    /// A packed `dimension + 1` rows, the free slot's row holding the ground or anchor while GTH works; `None` when it will not fit.
    pub(super) fn reserve(dimension: usize) -> Option<Self> {
        todo!()
    }

    /// Pivot `p = Σ` remaining weights `+ s`, fill `w_jk` and surplus `s_j` each gain `lo·(hi/p)` of their two factors (`w_ji, w_ki` or `w_ji, s_i`) on unscaled weights, as any other order under- or overflows, `√` taken in one last pass: `d − Σl²` cancelled to 7e-2 on within's grounded dumps, GTH 3e-15, for 11-80% more factor time; inverse pivots wait in the untouched diagonal, and the free row is truncated, leaving `n` rows grounded or `n − 1` anchored.
    pub(super) fn factor(self, sddm: &Sddm<'_, T>) -> DenseCholesky<T> {
        todo!()
    }
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
        todo!()
    }
}
