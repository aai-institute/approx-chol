use super::factorization::{BlockDim, Fallback};
use crate::graph::SplitFactor;
use crate::Error;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default)]
/// Configuration for approximate Cholesky factorization.
pub struct Config {
    /// Random seed for the edge-weight sampler.
    pub seed: u64,
    /// AC2 multi-edge multiplicity `k`. Splitting an edge fewer than twice is
    /// standard AC, so `None`, `Some(0)` and `Some(1)` all select AC.
    pub split_merge: Option<u32>,
    #[cfg_attr(feature = "serde", serde(default))]
    /// Which factorization each block gets.
    pub backend: Backend,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// What to do about a block whose exact Cholesky reaches an unusable pivot.
pub enum ExactFailure {
    #[default]
    /// Factor that block approximately and record it in [`Factor::fallbacks`](crate::Factor::fallbacks).
    FallBackToApproximate,
    /// Fail with [`Error::DenseFactorizationFailed`](crate::Error::DenseFactorizationFailed).
    Error,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Which factorization each connected block gets.
pub enum Backend {
    /// Approximate elimination for every block.
    Approximate,
    /// Exact dense Cholesky at or below `max_dim` solved variables, approximate
    /// elimination above.
    ExactBelow {
        /// Inclusive bound on the variables a block solves for; costs `O(max_dim³)`
        /// time. A bound of `0` claims no block, so it selects [`Approximate`].
        ///
        /// [`Approximate`]: Backend::Approximate
        max_dim: usize,
        /// What to do about a claimed block that reaches an unusable pivot.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Route {
    Exact { on_failure: ExactFailure },
    Approximate,
}

impl ExactFailure {
    /// Apply the policy to a block that declined exact elimination. Only an
    /// unusable pivot can be fatal — a block that will not fit is factored
    /// approximately whatever the policy says, so there is one arm, not two.
    pub(super) fn accept(self, fallback: Fallback) -> Result<Fallback, Error> {
        match (self, fallback) {
            (Self::Error, Fallback::InvalidPivot(pivot)) => {
                Err(Error::DenseFactorizationFailed(pivot))
            }
            _ => Ok(fallback),
        }
    }
}

impl Config {
    /// The AC2 split factor this configuration selects, or `None` for standard AC.
    /// The one place the total [`split_merge`](Self::split_merge) field becomes the
    /// algorithm, so nothing downstream restates which values mean AC.
    pub(super) fn split_factor(self) -> Option<SplitFactor> {
        self.split_merge.and_then(SplitFactor::new)
    }
}

impl Backend {
    /// Routing depends on the block, so this stays a function of its dimension
    /// rather than a value resolved once. It reads no other field, which is what
    /// lets the pipeline carry a backend instead of the whole [`Config`].
    pub(super) fn route(self, dim: BlockDim) -> Route {
        match self {
            // The range starts at one because a block solving for no variable has
            // no dense factor to build, whatever `max_dim` claims.
            Backend::ExactBelow {
                max_dim,
                on_failure,
            } if (1..=max_dim).contains(&dim.solved()) => Route::Exact { on_failure },
            _ => Route::Approximate,
        }
    }
}
