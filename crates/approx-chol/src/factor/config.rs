use crate::ScalingPolicy;

/// Build settings for [`factorize_with`](crate::factorize_with).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Config {
    /// Every sampler stream derives from this, keyed by block and star.
    pub seed: u64,
    /// AC2 multi-edge multiplicity `k`. Splitting an edge fewer than twice is
    /// standard AC, so `None`, `Some(0)` and `Some(1)` all select AC.
    pub split_merge: Option<u32>,
    /// Which factorization each block gets.
    #[cfg_attr(feature = "serde", serde(default))]
    pub backend: Backend,
    /// How a bipartite block's eliminated side becomes the kept side's Schur complement.
    #[cfg_attr(feature = "serde", serde(default))]
    pub complement: SchurMode,
    /// How a block not diagonally dominant is scaled to dominance.
    #[cfg_attr(feature = "serde", serde(default))]
    pub scaling: ScalingPolicy,
}

impl Config {
    pub(super) fn approximate(&self) -> Multiplicity {
        multiplicity(self.split_merge)
    }

    pub(super) fn complement(&self) -> CliqueRule {
        match self.complement {
            SchurMode::Exact => CliqueRule::Exact,
            SchurMode::Approximate { split_merge } => {
                CliqueRule::Sampled(multiplicity(split_merge))
            }
            SchurMode::Same => CliqueRule::Sampled(self.approximate()),
        }
    }

    /// Whether a Cholesky of `dimension`, a cover's two sheets counted, is dense: it then takes `S` exact, spending nothing on sparsity.
    pub(super) fn dense(&self, dimension: usize) -> bool {
        match self.backend {
            Backend::Approximate => false,
            Backend::ExactBelow { max_dim } => dimension <= max_dim,
        }
    }
}

/// The one place a `split_merge` becomes the algorithm: below two copies is AC.
fn multiplicity(split_merge: Option<u32>) -> Multiplicity {
    match split_merge {
        Some(k) if k >= 2 => Multiplicity::Split(k),
        _ => Multiplicity::Single,
    }
}

/// Which factorization each block gets.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Approximate elimination for every block.
    Approximate,
    /// Exact dense Cholesky at or below `max_dim`, a cover's two sheets counted, approximate elimination above.
    ExactBelow {
        /// Inclusive bound on the dimension, costing `O(max_dim³)` time; `0` selects [`Approximate`](Backend::Approximate).
        max_dim: usize,
    },
}

impl Default for Backend {
    fn default() -> Self {
        Self::ExactBelow { max_dim: 24 }
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

/// How a pivot's star becomes fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CliqueRule {
    /// The star's whole clique: the exact Schur complement.
    Exact,
    /// A clique tree sampled from the star.
    Sampled(Multiplicity),
}

/// How many virtual copies each edge splits into while sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Multiplicity {
    /// AC: one copy.
    Single,
    /// AC2: `k ≥ 2` copies, merged back under a cap.
    Split(u32),
}

/// Whether a solve may split its own work across threads; an outer parallel loop says no.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerParallelism {
    /// Large block eliminations split their rows.
    Allow,
    /// Everything runs on the calling thread.
    Forbid,
}
