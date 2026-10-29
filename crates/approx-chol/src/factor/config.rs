use super::clique_tree::Multi;

/// Build settings for [`factorize_with`](crate::factorize_with).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Every sampler stream derives from this, keyed by block and star.
    pub seed: u64,
    /// AC2 multi-edge multiplicity `k`, costing `O(k)` sampler draws per edge; `None`, `Some(0)` and `Some(1)` select standard AC.
    pub split_merge: Option<u32>,
    /// Largest block dimension, a cover's two sheets counted, factored by exact dense Cholesky at `O(exact_max_dim³)`; `0` factors every block approximately.
    pub exact_max_dim: usize,
    /// How a bipartite block's eliminated side becomes the kept side's Schur complement.
    pub complement: SchurMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            split_merge: None,
            exact_max_dim: 24,
            complement: SchurMode::default(),
        }
    }
}

impl Config {
    /// Whether a Cholesky of `dimension`, a cover's two sheets counted, is dense: it then takes `S` exact, spending nothing on sparsity.
    pub(super) fn dense(&self, dimension: usize) -> bool {
        dimension <= self.exact_max_dim
    }

    /// The split the factor's own sampler draws with.
    pub(super) fn split(&self) -> Option<Multi> {
        self.split_merge.and_then(Multi::split)
    }
}

/// How a bipartite block's eliminated stars become the kept side's Schur complement; a dense kept side always takes it exact.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchurMode {
    /// Each star's whole clique: the exact Schur complement.
    Exact,
    /// A clique tree sampled from each star.
    Approximate {
        /// Multiplicity `k`, as [`Config::split_merge`]: `None`, `Some(0)` and `Some(1)` select AC.
        split_merge: Option<u32>,
    },
    /// Sampled with [`Config::split_merge`]'s multiplicity, as the kept side's own factor is.
    #[default]
    Same,
}

/// Whether a solve may split its own work across threads; an outer parallel loop says no.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerParallelism {
    /// Large block eliminations split their rows.
    Allow,
    /// Everything runs on the calling thread.
    Forbid,
}
