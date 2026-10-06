//! [`block`] gauge and [`cholesky`] arm are chosen independently; all four combinations occur.

#[cfg(any(feature = "serde", test))]
use core::fmt;

pub(crate) mod approximate;
mod block;
mod cholesky;
pub(crate) mod exact;
mod factor;
mod gauge;
mod permutation;

pub use approximate::CliqueTreeSampler;
pub(crate) use block::Block;
pub(crate) use cholesky::Cholesky;
#[cfg(feature = "serde")]
pub use factor::FACTOR_FORMAT_VERSION;
pub use factor::{Factor, Fallback, SolveError};
pub(crate) use permutation::Permutation;

/// Raised at the serde boundary, before a corrupted persisted factor can reach the solve.
#[cfg(any(feature = "serde", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorError {
    // Payloads are only written at the current version, so without serde this is unreachable.
    #[cfg(feature = "serde")]
    UnsupportedFormatVersion {
        found: u32,
        supported: u32,
    },
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
        // Not corruption: intact bytes written by a release this build does not read.
        #[cfg(feature = "serde")]
        if let Self::UnsupportedFormatVersion { found, supported } = self {
            return write!(
                f,
                "persisted factor declares format version {found:#010x}, but this build reads {supported:#010x}"
            );
        }
        write!(f, "corrupted persisted factor: {self:?}")
    }
}
