/// Build settings for [`factorize`](crate::factorize); the reduction's policy is fixed by [`Sgdd`](crate::Sgdd).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Every sampler stream derives from this, keyed by component and pivot.
    pub seed: u64,
    /// How an independent leading block's stars become fill.
    pub round: CliqueRule,
    /// How minimum-degree elimination samples each star.
    pub kernel: Multiplicity,
    /// Inclusive size at or below which the remainder takes a dense Cholesky; `0` never does.
    pub dense_below: usize,
    /// What a dense Cholesky's unusable pivot does.
    pub on_dense_failure: ExactFailure,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            round: CliqueRule::Sampled(Multiplicity::Single),
            kernel: Multiplicity::Single,
            dense_below: 24,
            on_dense_failure: ExactFailure::FallBackToApproximate,
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
    Split(SplitFactor),
}

/// AC2's copy count; only `k ≥ 2` exists, so no split is a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitFactor(u32);

impl SplitFactor {
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
pub enum ExactFailure {
    /// Factor that remainder approximately and record a [`Fallback`](crate::Fallback).
    #[default]
    FallBackToApproximate,
    /// Fail with [`FactorError::DenseFactorizationFailed`](crate::FactorError::DenseFactorizationFailed).
    Error,
}
