//! The build: one call on every thread, or a reduction and its factoring as two stages under a named budget.

use num_traits::PrimInt;
use rayon::iter::Either;

use super::{ApproxInverse, Summand};
use crate::input::{BipartiteFactorWidth2, Input, Side};
use crate::order::Slots;
use crate::sgdd::{
    BlockComplement, CoupledBlock, IrreducibleBlock, Packed, Reduced, ReducedComplement, Summands,
};
use crate::{BuildError, Clamp, Config, Real, ScalingPolicy, Threads};

/// Factorizes any [`Input`] with [`Config::default`] and [`ScalingPolicy::default`].
pub fn factorize<'a, T, I, M>(input: M) -> Result<ApproxInverse<T>, BuildError>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: Into<Input<'a, T, I>>,
{
    factorize_with(input, Config::default(), ScalingPolicy::default())
}

/// Factorizes any [`Input`] on every thread: [`Sgdd::reduce`] under `scaling`, then [`Sgdd::factor`] under `config`.
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
    Ok(Sgdd::reduce(input, scaling, Threads::All)?.factor(config, Threads::All))
}

/// Factors a bipartite `A`'s Schur complement onto its `kept` side on every thread: [`Complement::reduce_onto`], then [`Complement::factor`].
pub fn factorize_complement<T: Real, I: PrimInt + Sync>(
    input: BipartiteFactorWidth2<'_, T, I>,
    kept: Side,
    config: Config,
    scaling: ScalingPolicy,
) -> Result<ApproxInverse<T>, BuildError> {
    let complement = Complement::reduce_onto(input, kept, scaling, Threads::All)?;
    Ok(complement.factor(config, Threads::All))
}

/// Built only by [`Sgdd::reduce`]: `A` in symmetric generalized diagonally dominant form, each irreducible block scaled to dominance.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64>(Indexed<PackedSgdd<T, u32>, PackedSgdd<T, u64>>);

/// Built only by [`Complement::reduce_onto`]: a bipartite `A` reduced as [`Sgdd`] reduces it, each block's complement onto the kept side formed when it is factored.
#[derive(Clone, Debug)]
pub struct Complement<T = f64>(Indexed<PackedComplement<T, u32>, PackedComplement<T, u64>>);

/// A reduction packed for its factor, its indices in `I`.
type PackedSgdd<T, I> = Packed<IrreducibleBlock<T, I>, T>;

/// A reduced complement packed for its factor, its indices in `I`.
type PackedComplement<T, I> = Packed<BlockComplement<T, I>, T>;

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

impl<T: Real, B: Summand<T, u32>, C: Summand<T, u64>> Indexed<Packed<B, T>, Packed<C, T>> {
    fn clamps(&self) -> impl Iterator<Item = Clamped<'_>> {
        match self {
            Self::U32(packed) => Either::Left(Clamped::of(packed)),
            Self::U64(packed) => Either::Right(Clamped::of(packed)),
        }
    }

    fn factor(self, config: &Config, threads: Threads) -> ApproxInverse<T> {
        match self {
            Self::U32(packed) => ApproxInverse::build(packed, config, threads),
            Self::U64(packed) => ApproxInverse::build(packed, config, threads),
        }
    }
}

impl<T: Real> Sgdd<T> {
    /// Reduces any [`Input`] under `threads`, scaling each block short of dominance under `scaling`; nothing after this reads `input`.
    pub fn reduce<'a, I, M>(
        input: M,
        scaling: ScalingPolicy,
        threads: Threads,
    ) -> Result<Self, BuildError>
    where
        I: PrimInt + Sync + 'a,
        M: Into<Input<'a, T, I>>,
    {
        let input = input.into();
        let narrow = || Reduced::reduce(input, scaling, threads).map(Summands::pack);
        let wide = || Reduced::reduce(input, scaling, threads).map(Summands::pack);
        Indexed::of(input.n(), narrow, wide).map(Self)
    }

    /// One [`Clamped`] per clamped component of `A`, in the order [`ApproxInverse::components`] will list them.
    pub fn clamps(&self) -> impl Iterator<Item = Clamped<'_>> {
        self.0.clamps()
    }

    /// Factors each block under `threads`; clone first to factor again under another `config`.
    pub fn factor(self, config: Config, threads: Threads) -> ApproxInverse<T> {
        self.0.factor(&config, threads)
    }
}

impl<T: Real> Complement<T> {
    /// Reduces a bipartite `A` under `threads`, as [`Sgdd::reduce`] does, to factor its Schur complement onto its `kept` side.
    pub fn reduce_onto<I: PrimInt + Sync>(
        input: BipartiteFactorWidth2<'_, T, I>,
        kept: Side,
        scaling: ScalingPolicy,
        threads: Threads,
    ) -> Result<Self, BuildError> {
        let narrow =
            || ReducedComplement::reduce_onto(input, kept, scaling, threads).map(Summands::pack);
        let wide =
            || ReducedComplement::reduce_onto(input, kept, scaling, threads).map(Summands::pack);
        Indexed::of(input.n(), narrow, wide).map(Self)
    }

    /// One [`Clamped`] per clamped component of `A`, on its kept vertices, in the order [`ApproxInverse::components`] will list them.
    pub fn clamps(&self) -> impl Iterator<Item = Clamped<'_>> {
        self.0.clamps()
    }

    /// Forms each block's complement under [`Config::complement`] and factors it under `threads`; the factor's vertex `k` is the kept side's `k`-th.
    pub fn factor(self, config: Config, threads: Threads) -> ApproxInverse<T> {
        self.0.factor(&config, threads)
    }
}

/// A clamped component of `A`, borrowed from its reduction before factoring.
#[derive(Clone, Copy, Debug)]
pub struct Clamped<'s> {
    slots: Slots<'s>,
    clamp: &'s Clamp,
}

impl<'s> Clamped<'s> {
    /// Each clamped block of `packed` on its segment of the order.
    fn of<B: CoupledBlock, T>(packed: &'s Packed<B, T>) -> impl Iterator<Item = Self> {
        (packed.coupled.iter().zip(&packed.segments)).filter_map(|(block, &segment)| {
            Some(Self {
                slots: packed.order.slots(segment),
                clamp: block.clamp()?,
            })
        })
    }

    /// Its vertices in the factored matrix's numbering, in the order its factor will number them, as [`Component::vertices`](crate::Component::vertices) lists them.
    pub fn vertices(&self) -> impl ExactSizeIterator<Item = usize> + 's {
        self.slots.vertices()
    }

    /// How the reduction clamped the component's rows to make it dominant.
    pub fn clamp(&self) -> &'s Clamp {
        self.clamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::CsrRef;

    /// Two `D L D` paths, `D` over two decades, apart by a lone vertex: neither dominant, so each clamps when no scaling step may run.
    fn clamped() -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let d = [1.0, 30.0, 0.5, 100.0, 2.0, 0.0, 1.0, 50.0, 0.2, 10.0];
        let (mut ptrs, mut columns, mut values) = (vec![0u32], vec![], vec![]);
        for i in 0..d.len() {
            let path = |j: usize| j / 5 == i / 5 && j != 5 && i != 5;
            let neighbors = [i.checked_sub(1), Some(i + 1)].into_iter().flatten();
            let neighbors: Vec<usize> = neighbors.filter(|&j| j < d.len() && path(j)).collect();
            let mut row: Vec<(usize, f64)> = neighbors.iter().map(|&j| (j, -d[i] * d[j])).collect();
            let diagonal = match i {
                5 => 1.0,
                _ => d[i] * d[i] * neighbors.len() as f64,
            };
            row.push((i, diagonal));
            row.sort_by_key(|&(j, _)| j);
            for (j, value) in row {
                columns.push(j as u32);
                values.push(value);
            }
            ptrs.push(columns.len() as u32);
        }
        (ptrs, columns, values)
    }

    fn read<'s>(clamps: impl Iterator<Item = Clamped<'s>>) -> Vec<(Vec<usize>, Clamp)> {
        clamps
            .map(|c| (c.vertices().collect(), c.clamp().clone()))
            .collect()
    }

    /// The `u64` arm reads the clamps the `u32` arm reads, on the same vertices in the same order.
    #[test]
    fn the_wide_index_reads_the_clamps_the_narrow_one_reads() {
        let (ptrs, columns, values) = clamped();
        let csr = CsrRef::new(ptrs.len() - 1, &ptrs, &columns, &values).expect("valid csr");
        let policy = ScalingPolicy::new(0.0, 0).expect("valid policy");
        let narrow = Reduced::<f64, u32>::reduce(csr.into(), policy, Threads::All);
        let wide = Reduced::<f64, u64>::reduce(csr.into(), policy, Threads::All);
        let narrow: Indexed<_, PackedSgdd<f64, u64>> =
            Indexed::U32(narrow.expect("reduces").pack());
        let wide: Indexed<PackedSgdd<f64, u32>, _> = Indexed::U64(wide.expect("reduces").pack());
        let clamps = read(narrow.clamps());
        assert_eq!(clamps.len(), 2, "{clamps:?}");
        assert_eq!(read(wide.clamps()), clamps);
    }
}
