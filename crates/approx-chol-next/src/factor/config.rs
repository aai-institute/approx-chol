/// Build settings for [`factorize`](crate::factorize); the reduction's policy is fixed by [`Sgdd`](crate::Sgdd).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Every sampler stream derives from this, keyed by part and star.
    pub seed: u64,
    /// How a bipartite part's eliminated stars become the kept block's Schur complement.
    pub complement: CliqueRule,
    /// How approximate Cholesky samples each star.
    pub approximate: Multiplicity,
    /// Largest dimension, a cover's two sheets counted, that the Cholesky takes dense; `0` never does.
    pub dense_up_to: usize,
    /// What a dense Cholesky's unusable pivot does.
    pub on_dense_failure: OnDenseFailure,
}

impl Config {
    /// The one routing rule: a Cholesky of this dimension goes dense.
    pub(super) fn routes_dense(&self, dimension: usize) -> bool {
        dimension <= self.dense_up_to
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            complement: CliqueRule::Sampled(Multiplicity::Single),
            approximate: Multiplicity::Single,
            dense_up_to: 24,
            on_dense_failure: OnDenseFailure::FallBackToApproximate,
        }
    }
}

/// How a pivot's star becomes fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliqueRule {
    /// The star's whole clique: the exact Schur complement.
    Exact,
    /// A clique tree sampled from the star.
    Sampled(Multiplicity),
}

/// How many virtual copies each edge splits into while sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Multiplicity {
    /// AC: one copy.
    Single,
    /// AC2: `k` copies, merged back under a cap.
    Split(SplitCount),
}

/// AC2's copy count; only `k ≥ 2` exists, so no split is a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitCount(u32);

impl SplitCount {
    /// `None` below two.
    pub fn new(k: u32) -> Option<Self> {
        (k >= 2).then_some(Self(k))
    }

    /// The copy count.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// What a dense Cholesky's unusable pivot does.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnDenseFailure {
    /// Factor that matrix approximately and record a [`Fallback`](crate::Fallback).
    #[default]
    FallBackToApproximate,
    /// Fail with [`FactorError::UnusablePivot`](crate::FactorError::UnusablePivot).
    Error,
}

/// Whether a solve may split its own work across threads; an outer parallel loop says no.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerParallelism {
    /// Large block eliminations split their rows.
    Allow,
    /// Everything runs on the calling thread.
    Forbid,
}
