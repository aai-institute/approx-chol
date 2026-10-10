//! Exact Cholesky of a small SDDM: the packed factor and the code that builds and applies it.

use crate::factor::sddm::Sddm;
use crate::Real;

/// What makes a dense pivot unusable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PivotDefect {
    /// Infinite or NaN.
    NonFinite,
    /// Zero or negative.
    NonPositive,
}

/// An unusable dense pivot, named in `A`'s numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnusablePivot {
    /// The pivot's vertex in `A`.
    pub vertex: u32,
    /// What was wrong with it.
    pub defect: PivotDefect,
}

/// Capacity for a packed triangle, taken before its matrix is built; not `Clone`, which would drop the capacity.
#[derive(Debug)]
pub(super) struct Room<T>(Vec<T>);

impl<T: Real> Room<T> {
    /// Enough for a `dimension`-row matrix, grounded or not; `None` when it will not fit.
    pub(super) fn reserve(dimension: usize) -> Option<Self> {
        todo!()
    }

    /// Takes its rows from the matrix, `n` grounded or `n − 1` anchored, and panics past the room; `vertices` names each base vertex in `A`.
    pub(super) fn factor(
        self,
        sddm: &Sddm<'_, T>,
        vertices: &[u32],
    ) -> Result<DenseCholesky<T>, UnusablePivot> {
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
