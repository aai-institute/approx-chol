//! A [`block`] is an [`anchor`] paired with a [`cholesky`], the two chosen
//! independently — by augmentation and by policy — so all four combinations occur.

#[cfg(any(feature = "serde", test))]
use core::fmt;

mod anchor;
pub(crate) mod approximate;
mod block;
mod cholesky;
pub(crate) mod exact;
mod factor;
mod permutation;

pub(crate) use anchor::Anchor;
pub use approximate::CliqueTreeSampler;
pub(crate) use block::{Block, BlockDim};
pub(crate) use cholesky::Cholesky;
#[cfg(feature = "serde")]
pub use factor::FACTOR_FORMAT_VERSION;
pub use factor::{Factor, Fallback, SolveError};
pub(crate) use permutation::Permutation;

/// Raised at the serde boundary, before a corrupted persisted factor can reach the
/// solve path.
#[cfg(any(feature = "serde", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorError {
    // A payload is only ever written with the current version, so `test` alone leaves
    // the mismatch arm unreachable without the serde boundary.
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
    BlockDimsDoNotCoverFactor {
        covered: usize,
        n: usize,
    },
    ExactFactorLengthInvalid {
        n: usize,
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
    MultipleGroundBlocks {
        grounded: usize,
    },
    PermutationInvalid {
        position: usize,
    },
}

#[cfg(any(feature = "serde", test))]
impl fmt::Display for FactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The version mismatch is the one arm that is not corruption: the bytes are
        // intact and were written by a release this one does not read.
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
