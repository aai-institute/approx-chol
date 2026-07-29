//! Cholesky factor. A [`block`] is an [`anchor`] (how its singular system is made
//! solvable) paired with a [`cholesky`] — [`approximate`] or [`exact`], each owning
//! its own storage and solve; [`permutation`] maps blocks back to input coordinates,
//! and [`factor`] is the solve API over all of them.

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
pub use approximate::clique_tree_sample;
pub(crate) use block::{Block, BlockDim};
pub(crate) use cholesky::Cholesky;
pub use factor::{Factor, Fallback, SolveError};
pub(crate) use permutation::Permutation;

/// Structural validation errors for a deserialized [`Factor`], raised at the
/// serde boundary before a corrupted persisted factor can reach the solve path.
/// Internal: surfaces only as a serde error string, via the `Debug`-based
/// [`Display`] below (variant + offending values), so — unlike the public error
/// enums in `error.rs` — it carries no hand-written per-variant prose.
#[cfg(any(feature = "serde", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorError {
    /// More factor nonzeros than a `u32` step offset can address.
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
    /// Block dimensions do not sum to the factor dimension.
    BlockDimsDoNotCoverFactor {
        covered: usize,
        n: usize,
    },
    /// Exact lower-triangular storage is inconsistent with its block dimension.
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
    /// More than one block is anchored on the single ground vertex.
    MultipleGroundBlocks {
        grounded: usize,
    },
    /// A permutation position is out of bounds, repeated, or a bare fixed point.
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
