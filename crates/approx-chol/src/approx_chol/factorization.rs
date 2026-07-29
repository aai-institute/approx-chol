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
pub use approximate::StarSampler;
pub(crate) use block::{Block, BlockDim};
pub(crate) use cholesky::Cholesky;
pub use factor::{Factor, Fallback, SolveError};
pub(crate) use permutation::Permutation;

/// Raised at the serde boundary, before a corrupted persisted factor can reach the
/// solve path.
#[cfg(any(feature = "serde", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorError {
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
        write!(f, "corrupted persisted factor: {self:?}")
    }
}
