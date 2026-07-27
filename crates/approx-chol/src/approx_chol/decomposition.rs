//! Approximate Cholesky factor: elimination-sequence storage ([`sequence`])
//! and the LDLᵀ [`Factor`] solve API ([`factor`]).

#[cfg(any(feature = "serde", test))]
use core::fmt;

mod factor;
mod sequence;

pub(crate) use factor::{BlockFactor, Permutation, Pin};
pub use factor::{Factor, SolveError};
pub(crate) use sequence::EliminationSequence;

/// Structural validation errors for a deserialized [`Factor`], raised at the
/// serde boundary before a corrupted persisted factor can reach the solve path.
/// Internal: surfaces only as a serde error string, via the `Debug`-based
/// [`Display`] below (variant + offending values), so — unlike the public error
/// enums in `error.rs` — it carries no hand-written per-variant prose.
#[cfg(any(feature = "serde", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorError {
    OriginalDimExceedsInternal {
        original_n: usize,
        n: usize,
    },
    /// More factor nonzeros than a `u32` step offset can address.
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
    /// A block's pinned variable is not a local index of that block.
    BlockPinInvalid {
        pin: usize,
        n: usize,
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
