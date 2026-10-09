//! Exact Cholesky of a small SDDM: the packed factor and the code that builds and applies it.

use crate::laplacian::Sddm;
use crate::Real;

/// Why a dense pivot could not be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseFailure {
    /// Infinite or NaN.
    NonFinitePivot,
    /// Zero or negative.
    NonPositivePivot,
}

/// An unusable dense pivot, named in `A`'s numbering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnusablePivot {
    /// The pivot's vertex in `A`.
    pub vertex: u32,
    /// What was wrong with it.
    pub failure: DenseFailure,
}

/// A remainder routed dense and factored approximately instead: less accurate than asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The dense Cholesky reached an unusable pivot.
    InvalidPivot(UnusablePivot),
    /// The packed triangle does not fit in memory.
    WillNotFit {
        /// Rows of the declined remainder.
        dim: usize,
    },
}

/// Packed lower triangle, the free slot excluded: an upper one would double the factor.
#[derive(Clone, Debug)]
pub(crate) struct LowerTriangular<T> {
    values: Vec<T>,
}

impl<T: Real> LowerTriangular<T> {
    /// Assembles and factors in place; `vertices` names the rows' copied vertices in `A`.
    pub(crate) fn factor(sddm: &Sddm<T>, vertices: &[u32]) -> Result<Self, Fallback> {
        todo!()
    }

    fn rows(&self) -> usize {
        ((8 * self.values.len() + 1).isqrt() - 1) / 2
    }

    /// Forward, backward, and the free last slot pinned to `0`.
    pub(crate) fn substitute(&self, values: &mut [T]) {
        todo!()
    }
}
