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

/// A Cholesky routed dense and factored approximately instead: less accurate than asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The dense Cholesky reached an unusable pivot.
    UnusablePivot(UnusablePivot),
    /// The packed triangle does not fit in memory.
    WillNotFit {
        /// Dimension of the declined matrix, a cover's two sheets counted.
        dimension: usize,
    },
}

/// Packed lower triangle of `L` in `LLᵀ`, the free slot excluded: an upper one would double the factor.
#[derive(Clone, Debug)]
pub(super) struct DenseCholesky<T> {
    values: Vec<T>,
}

impl<T: Real> DenseCholesky<T> {
    /// Assembles and factors in place; `vertices` names each base vertex in `A`.
    pub(super) fn factor(sddm: &Sddm<'_, T>, vertices: &[u32]) -> Result<Self, Fallback> {
        todo!()
    }

    fn rows(&self) -> usize {
        ((8 * self.values.len() + 1).isqrt() - 1) / 2
    }

    /// Forward, backward, and the free last slot pinned to `0`.
    pub(super) fn substitute(&self, values: &mut [T]) {
        todo!()
    }
}
