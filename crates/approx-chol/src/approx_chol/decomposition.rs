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
    OriginalDimExceedsInternal {
        original_n: usize,
        n: usize,
    },
    /// `start > end` or `end > nnz`.
    NeighborRangeInvalid {
        step: usize,
        start: usize,
        end: usize,
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
    /// Trailing neighbor storage that no step references.
    TrailingNeighborStorage {
        covered: usize,
        nnz: usize,
    },
}

impl fmt::Display for FactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "corrupted persisted factor: {self:?}")
    }
}
