//! A [`block`] is grounded or floating around either [`cholesky`] arm; all four pairs occur.

#[cfg(any(feature = "serde", test))]
use core::fmt;

pub(super) mod approximate;
mod block;
mod cholesky;
pub(super) mod exact;
mod factor;
mod permutation;

pub use approximate::CliqueTreeSampler;
pub(super) use block::Block;
pub(super) use cholesky::Cholesky;
#[cfg(feature = "serde")]
pub use factor::FACTOR_FORMAT_VERSION;
pub use factor::{Factor, SolveError};
pub(super) use permutation::Permutation;

/// Raised at the serde boundary, before a corrupted persisted factor can reach the solve.
#[cfg(any(feature = "serde", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FactorError {
    // Only the deserialize path can construct this; `test` alone leaves it dead.
    #[cfg(feature = "serde")]
    NonzeroCountExceedsU32 {
        nnz: usize,
    },
    VertexOutOfBounds {
        step: usize,
        vertex: u32,
        n: usize,
    },
    NeighborOutOfBounds {
        step: usize,
        neighbor: u32,
        n: usize,
    },
    ExactFactorLengthInvalid {
        len: usize,
    },
    ExactPivotInvalid {
        index: usize,
    },
    ExactRowNotRepresentable {
        row: usize,
    },
    StepValueInvalid {
        step: usize,
    },
    UneliminatedVertexInvalid {
        vertex: u32,
        n: usize,
    },
    VertexEliminatedTwice {
        step: usize,
        vertex: u32,
    },
    PermutationInvalid {
        position: usize,
    },
}

#[cfg(any(feature = "serde", test))]
impl fmt::Display for FactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "corrupted persisted factor: {self:?}")
    }
}
