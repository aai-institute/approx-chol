//! Approximate Cholesky factor: elimination-sequence storage ([`sequence`])
//! and the LDLᵀ [`Factor`] solve API ([`factor`]).

use core::fmt;

mod factor;
mod sequence;

pub use factor::{Factor, SolveError};
pub(crate) use sequence::EliminationSequence;

/// Structural validation errors for a deserialized [`Factor`], raised at the
/// serde boundary before a corrupted persisted factor can reach the solve path.
/// Internal: surfaces only as a serde error string, via the `Debug`-based
/// [`Display`] below (variant + offending values), so — unlike the public error
/// enums in `error.rs` — it carries no hand-written per-variant prose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FactorError {
    /// `original_n` exceeds the internal factor dimension `n`.
    OriginalDimExceedsInternal { original_n: usize, n: usize },
    /// `offsets.len()` must equal `n_steps + 1`.
    OffsetsLengthMismatch { expected: usize, got: usize },
    /// `inv_diagonal.len()` must equal `n_steps`.
    InvDiagonalLengthMismatch { expected: usize, got: usize },
    /// `neighbor_indices` and `elimination_fractions` differ in length.
    NeighborFractionLengthMismatch {
        neighbor_len: usize,
        fraction_len: usize,
    },
    /// `offsets[0]` must be zero.
    OffsetsMustStartAtZero { got: u32 },
    /// A step's offset range `[start, end)` is invalid (`start > end` or `end > nnz`).
    OffsetRangeInvalid {
        step: usize,
        start: usize,
        end: usize,
        nnz: usize,
    },
    /// A pivot vertex index is out of bounds for the factor dimension `n`.
    VertexOutOfBounds { step: usize, vertex: u32, n: usize },
    /// A neighbor index is out of bounds for the factor dimension `n`.
    NeighborOutOfBounds {
        step: usize,
        neighbor: u32,
        n: usize,
    },
    /// The final offset must equal the neighbor storage length (`nnz`).
    FinalOffsetMismatch { last: usize, nnz: usize },
}

impl fmt::Display for FactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "corrupted persisted factor: {self:?}")
    }
}
