use std::num::NonZeroU32;

use super::clique_tree::Multi;

/// Build settings for [`factorize_with`](crate::factorize_with).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    /// Every sampler stream derives from this, keyed by block and star.
    pub seed: u64,
    /// How the approximate Cholesky samples each pivot's fill.
    pub sampling: Sampling,
    /// How a frustrated block, one no diagonal `±1` congruence folds into an SDDM, is factored approximately; the dense arm factors it as itself.
    pub frustrated: FrustratedRoute,
    /// Largest block dimension, a bipartite block's by its kept side, factored by exact dense Cholesky at `O(exact_max_dim³)`; `0` factors every block approximately.
    pub exact_max_dim: usize,
    /// How a bipartite block's eliminated side becomes the kept side's Schur complement.
    pub complement: SchurMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            sampling: Sampling::CliqueTree {
                copies: NonZeroU32::MIN,
            },
            frustrated: FrustratedRoute::Cover(CoverSurplus::default()),
            exact_max_dim: 24,
            complement: SchurMode::default(),
        }
    }
}

impl Config {
    /// Whether a Cholesky of a block's `dimension` is dense: it then factors `M` itself, and takes `S` exact, spending nothing on sparsity.
    pub(super) fn dense(&self, dimension: usize) -> bool {
        dimension <= self.exact_max_dim
    }
}

/// How a frustrated block is factored approximately.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrustratedRoute {
    /// Through its Gremban cover, an SDDM on two sheets that solves `[x; −x]`, carrying its surplus as the [`CoverSurplus`] says.
    Cover(CoverSurplus),
    /// Eliminated as it is, on one sheet: fill sampled on magnitudes and signed by its ends, `σ_a σ_b`; opposite parallel entries cancel, and what they cancel grounds both ends, so it never floats. A bipartite block's complement is still covered, as [`CoverSurplus::Auto`] carries it.
    Signed,
}

/// How a frustrated block's Gremban cover carries the surplus each copy of a vertex holds.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CoverSurplus {
    /// To a ground vertex both sheets share; inside `Auto`'s twist band its common mode `1/s` swamps `z`, and PCG stalls where `Auto` converges.
    GroundVertex,
    /// As an edge `k⁺ — k⁻` per vertex, a quarter of its copies' surplus, which acts on `[z; −z]` as the grounds do: the cover stays singular, its common mode never in `z`. On a sampled complement the sheets draw their ground shares from different streams, so the edge takes their mean, unbiased; a whole block's cover and an exact complement mirror their rows, so the mean is exact.
    SurplusEdges,
    /// Surplus edges inside the twist band, `Σs ≤ 5ε (Σs + Σ|a_ij|)` over the cover's surplus `s` and off-diagonal `a`, where a ground vertex's common mode would swamp `z`; else the ground vertex.
    #[default]
    Auto,
}

/// How a pivot's star is sampled into fill.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sampling {
    /// A clique tree over `copies` of each edge (GKS 2023), costing `O(copies)` sampler draws per edge: `1` is standard AC, more is AC2.
    CliqueTree {
        /// Copies each edge splits into.
        copies: NonZeroU32,
    },
}

impl Sampling {
    /// The copies a sampler draws with: AC2's from two on, else AC's one.
    pub(super) fn copies(self) -> Option<Multi> {
        match self {
            Self::CliqueTree { copies } => Multi::split(copies.get()),
        }
    }
}

/// How a bipartite component's eliminated stars become the kept side's Schur complement, in a whole build of a bipartite input or a Schur build; a dense kept side always takes it exact.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SchurMode {
    /// Each star's whole clique: the exact Schur complement.
    Exact,
    /// Each star sampled under its own `sampling`.
    Approximate {
        /// How each star is sampled.
        sampling: Sampling,
    },
    /// Sampled under [`Config::sampling`], as the kept side's own factor is.
    #[default]
    Same,
}
