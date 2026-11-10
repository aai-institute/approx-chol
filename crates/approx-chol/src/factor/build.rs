//! The build: one call on every thread, or a reduction and its factoring as two stages.

use num_traits::PrimInt;

use super::{ApproxInverse, Summand};
use crate::input::{BipartiteFactorWidth2, Input, Side};
use crate::sgdd::{Reduced, ReducedComplement, Summands};
use crate::{BuildError, Config, Real, ScalingPolicy, Threads};

/// Factorizes any [`Input`] with [`Config::default`] and [`ScalingPolicy::default`].
pub fn factorize<'a, T, I, M>(input: M) -> Result<ApproxInverse<T>, BuildError>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: Into<Input<'a, T, I>>,
{
    factorize_with(input, Config::default(), ScalingPolicy::default())
}

/// Factorizes any [`Input`]: [`Sgdd::reduce`] under `scaling`, then [`Sgdd::factor`] under `config`.
pub fn factorize_with<'a, T, I, M>(
    input: M,
    config: Config,
    scaling: ScalingPolicy,
) -> Result<ApproxInverse<T>, BuildError>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: Into<Input<'a, T, I>>,
{
    Ok(Sgdd::reduce(input, scaling)?.factor(&config))
}

/// Factors a bipartite `A`'s Schur complement onto its `kept` side, formed under [`Config::complement`] and factored as [`factorize_with`] would; the factor's vertex `k` is the kept side's `k`-th.
pub fn factorize_complement<T: Real, I: PrimInt + Sync>(
    input: BipartiteFactorWidth2<'_, T, I>,
    kept: Side,
    config: Config,
    scaling: ScalingPolicy,
) -> Result<ApproxInverse<T>, BuildError> {
    let narrow = || ReducedComplement::reduce_onto(input, kept, scaling, Threads::All);
    let wide = || ReducedComplement::reduce_onto(input, kept, scaling, Threads::All);
    Ok(Indexed::of(input.n(), narrow, wide)?.factor(&config, Threads::All))
}

/// Built only by [`Sgdd::reduce`]: `A` in symmetric generalized diagonally dominant form, each irreducible block scaled to dominance.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64>(Indexed<Reduced<T, u32>, Reduced<T, u64>>);

/// A value in the narrowest index a cover's `2n + 1` slots fit.
#[derive(Clone, Copy, Debug)]
enum Indexed<N, W> {
    U32(N),
    U64(W),
}

impl<N, W> Indexed<N, W> {
    /// `narrow` while `u32` indexes a cover's two sheets and its ground, the most slots any block of `n` vertices takes, else `wide`.
    fn of(
        n: usize,
        narrow: impl FnOnce() -> Result<N, BuildError>,
        wide: impl FnOnce() -> Result<W, BuildError>,
    ) -> Result<Self, BuildError> {
        Ok(match u32::try_from(n.saturating_mul(2).saturating_add(1)) {
            Ok(_) => Self::U32(narrow()?),
            Err(_) => Self::U64(wide()?),
        })
    }
}

impl<T: Real, B: Summand<T, u32>, C: Summand<T, u64>>
    Indexed<Summands<B, u32, T>, Summands<C, u64, T>>
{
    fn factor(self, config: &Config, threads: Threads) -> ApproxInverse<T> {
        match self {
            Self::U32(summands) => ApproxInverse::build(summands, config, threads),
            Self::U64(summands) => ApproxInverse::build(summands, config, threads),
        }
    }
}

impl<T: Real> Sgdd<T> {
    /// Reduces any [`Input`], scaling each block short of dominance under `scaling`; nothing after this reads `input`.
    pub fn reduce<'a, I, M>(input: M, scaling: ScalingPolicy) -> Result<Self, BuildError>
    where
        I: PrimInt + Sync + 'a,
        M: Into<Input<'a, T, I>>,
    {
        let input = input.into();
        let narrow = || Reduced::reduce(input, scaling, Threads::All);
        let wide = || Reduced::reduce(input, scaling, Threads::All);
        Indexed::of(input.n(), narrow, wide).map(Self)
    }

    /// Factors each block in parallel; clone first to factor again under another `config`.
    pub fn factor(self, config: &Config) -> ApproxInverse<T> {
        self.0.factor(config, Threads::All)
    }
}
