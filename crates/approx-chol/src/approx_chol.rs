mod builder;
pub(crate) mod clique_tree;
pub(crate) mod decomposition;
mod star;

pub use builder::Builder;
pub use decomposition::{ExactFallback, Factor, SolveError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// What to do when a block's exact Cholesky hits an invalid pivot, which means
/// the input is not positive definite within the tolerance it was accepted under.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExactFailure {
    /// Factor that block with approximate elimination instead, and record it in
    /// [`Factor::exact_fallbacks`](crate::Factor::exact_fallbacks).
    #[default]
    FallBackToApproximate,
    /// Fail the factorization with [`Error::DenseFactorizationFailed`](crate::Error::DenseFactorizationFailed).
    Error,
}

/// Which factorization to run on each connected block.
///
/// With the default [`ExactFailure`], selecting a backend does not change which
/// inputs are accepted: a block whose exact factorization hits an invalid pivot
/// falls back to approximate elimination.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Approximate elimination for every block.
    Approximate,
    /// Exact Cholesky for blocks of at most `max_dim` vertices, approximate
    /// elimination above.
    ///
    /// Exact factorization of a block costs `O(max_dim²)` memory and
    /// `O(max_dim³)` time; `max_dim` is neither capped nor validated, so a
    /// large value trades a fast approximate factor for a slow exact one. A block
    /// too large to assemble densely falls back rather than failing.
    ExactBelow {
        /// Inclusive upper bound on the block dimension factored exactly.
        max_dim: usize,
        /// How to handle a block whose exact factorization fails.
        on_failure: ExactFailure,
    },
}

impl Default for Backend {
    fn default() -> Self {
        Self::ExactBelow {
            max_dim: 24,
            on_failure: ExactFailure::FallBackToApproximate,
        }
    }
}

impl Backend {
    pub(crate) fn uses_exact(self, block_n: usize) -> bool {
        match self {
            Self::Approximate => false,
            Self::ExactBelow { max_dim, .. } => block_n <= max_dim,
        }
    }

    pub(crate) fn on_failure(self) -> ExactFailure {
        match self {
            Self::Approximate => ExactFailure::default(),
            Self::ExactBelow { on_failure, .. } => on_failure,
        }
    }
}

/// Configuration for approximate Cholesky factorization.
///
/// [`Default`] is exact Cholesky below 24 vertices, standard AC above.
/// Set [`split_merge`](Self::split_merge) to enable AC2.
///
/// # Examples
///
/// ```
/// use approx_chol::Config;
///
/// // AC2 variant with multi-edge multiplicity
/// let config = Config {
///     split_merge: Some(2),
///     seed: 42,
///     ..Default::default()
/// };
/// assert!(config.split_merge.is_some());
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    /// Random seed for the edge-weight sampler. Use different values to get
    /// reproducible but varied approximate factors.
    pub seed: u64,
    /// AC2 multi-edge multiplicity parameter (`k`).
    ///
    /// `None` = standard AC (default), `Some(k)` = AC2 with `k` edge copies
    /// and `k` merge cap per neighbor pair.
    pub split_merge: Option<u32>,
    /// Per-block factorization backend.
    #[cfg_attr(feature = "serde", serde(default))]
    pub backend: Backend,
}
