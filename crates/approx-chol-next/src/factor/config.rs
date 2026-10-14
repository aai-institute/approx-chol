/// Build settings for [`factorize`](crate::factorize); the reduction's policy is fixed by [`Sgdd`](crate::Sgdd).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Every sampler stream derives from this, keyed by component and star.
    pub seed: u64,
    /// How a bipartite component's eliminated stars become the kept block's Schur complement.
    pub complement: CliqueRule,
    /// How approximate Cholesky samples each star.
    pub approximate: Multiplicity,
    /// Largest dimension, a cover's two sheets counted, that the Cholesky takes dense; `0` never does.
    pub dense_up_to: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            complement: CliqueRule::Sampled(Multiplicity::Single),
            approximate: Multiplicity::Single,
            dense_up_to: 24,
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

/// Whether a solve may split its own work across threads; an outer parallel loop says no.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InnerParallelism {
    /// Large block eliminations split their rows.
    Allow,
    /// Everything runs on the calling thread.
    Forbid,
}
