mod builder;
pub(crate) mod clique_tree;
pub(crate) mod decomposition;
mod star;

pub use builder::Builder;
pub(crate) use clique_tree::SampledColumn;
pub use decomposition::{Factor, SolveError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for approximate Cholesky factorization.
///
/// Use [`Default`] for standard AC (recommended for most inputs).
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
}
