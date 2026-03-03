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

/// Multi-edge split/merge parameters for the AC2 variant (Algorithm 6, GKS 2023).
///
/// AC2 replaces each graph edge with `split` copies of weight `1/split`
/// before elimination, then after compressing the star neighbourhood merges
/// back to at most `merge` multi-edges per neighbor pair. The result is a
/// denser but higher-quality approximate factor: fewer PCG iterations at
/// the cost of higher factorization time.
///
/// Typical values: `split = 2, merge = 2`.
///
/// See: Gao, Kyng & Spielman (2023), Algorithm 6.
#[derive(Clone, Copy, Debug)]
pub struct SplitMerge {
    /// Number of edge copies (`k`). Each original edge (u, v, w) becomes
    /// `split` edges each of weight `w / split`. Must be >= 1; `split = 1`
    /// is a no-op equivalent to standard AC. `split = 0` is rejected with
    /// [`crate::Error::InvalidConfig`].
    pub split: u32,
    /// Maximum multi-edges kept per neighbor pair after compression (`l`).
    /// Limits memory growth from the edge splitting step. Must be >= 1;
    /// `merge = 0` is rejected with [`crate::Error::InvalidConfig`].
    pub merge: u32,
}

impl Default for SplitMerge {
    fn default() -> Self {
        Self { split: 2, merge: 2 }
    }
}

/// Configuration for approximate Cholesky factorization.
///
/// Use [`Default`] for standard AC (recommended for most inputs).
/// Set [`split_merge`](Self::split_merge) to enable AC2.
///
/// # Examples
///
/// ```
/// use approx_chol::{Config, SplitMerge};
///
/// // AC2 variant with multi-edge split/merge
/// let config = Config {
///     split_merge: Some(SplitMerge { split: 2, merge: 2 }),
///     seed: 42,
///     ..Default::default()
/// };
/// assert!(config.split_merge.is_some());
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    /// Random seed for the edge-weight sampler. Use different values to get
    /// reproducible but varied approximate factors.
    pub seed: u64,
    /// AC2 multi-edge parameters. `None` = standard AC (default), `Some(_)` = AC2 variant.
    pub split_merge: Option<SplitMerge>,
}
