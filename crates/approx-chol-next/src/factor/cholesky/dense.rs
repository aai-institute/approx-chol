//! Exact Cholesky of a small SDDM by GTH elimination: every pivot a sum of nonnegative terms, so none cancels or fails.

use crate::factor::sddm::Sddm;
use crate::Real;

/// Capacity for a packed triangle, taken before its matrix is built; not `Clone`, which would drop the capacity.
#[derive(Debug)]
pub(super) struct Room<T>(Vec<T>);

impl<T: Real> Room<T> {
    /// A packed `dimension + 1` rows, the free slot's row holding the ground or anchor while GTH works, then `dimension` inverse pivots, contiguous as every row's fill reads them (strided in the diagonal measured +12-18% at n 16-24); `None` when it will not fit.
    pub(super) fn reserve(dimension: usize) -> Option<Self> {
        todo!()
    }

    /// Left-looking GTH on `−w`, the ground's `−s` or the anchor's row as free row `n`: stored entry `(r, j)` subtracts `Σ_{k<j} min·(max·p_k⁻¹)` of the magnitudes `|a_rk|, |a_jk|`, so its weight grows, `p_j = −Σ` of column `j` below the diagonal, `p_j⁻¹` kept in the tail, and one last pass scales by `√p⁻¹`; so nothing cancels, under- or overflows (`d − Σl²` cancelled to 7e-2 on within's grounded dumps, GTH 3e-15, for 11-80% more factor time), and the free row and tail are truncated, leaving `n` rows grounded or `n − 1` anchored.
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
