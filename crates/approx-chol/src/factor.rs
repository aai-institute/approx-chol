//! Each irreducible block as an optional block elimination `L` and a Cholesky of what it leaves: `A_c⁺ ≈ P F L⁻ᵀ diag(D_e⁻¹, S⁺) L⁻¹ F P`, `P` projecting out `F·1` when the block floats, else `I`.

mod apply;
mod block_elimination;
mod cholesky;
mod clique_tree;
mod config;
mod schur;
mod sddm;
#[cfg(feature = "serde")]
mod wire;

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;
use rayon::prelude::*;

pub use apply::{ApplyError, MatrixMut};
pub use config::{Config, SchurMode};
#[cfg(feature = "serde")]
pub use wire::FORMAT_VERSION;

use crate::order::{Listed, Order, Slots};
use crate::sgdd::{BlockComplement, Components, Congruence, IrreducibleBlock, Reduction, Summands};
use crate::storage::OffDiagonal;
use crate::Clamp;
use crate::{Batch, Index, MemBuffer, MemStack, Real, StackReq, Threads};
use apply::Kernel;
use block_elimination::BlockElimination;
use cholesky::{Arm, Cholesky, Signed, Workspace};
use schur::BipartiteBlock;
use sddm::Sddm;

/// `log₂` of the multiply-adds a block elimination's task carries, below which a split costs more than it saves.
const MULTIPLY_ADD_GRAIN_BITS: u32 = 17;

/// `count` slots of `V` on a stack; none asks no alignment either, so any stack serves a solve that takes none.
fn slots<V>(count: usize) -> StackReq {
    match count {
        0 => StackReq::EMPTY,
        count => StackReq::new::<V>(count),
    }
}

/// A block of `rows` must match the operator and `stack` hold its scratch.
fn check_buffers(
    rows: usize,
    n: usize,
    stack: &MemStack,
    needed: StackReq,
) -> Result<(), ApplyError> {
    // An overflowing requirement's alignment is 0, on which can_hold panics.
    match rows == n && needed.align_bytes() != 0 && stack.can_hold(needed) {
        true => Ok(()),
        false => Err(buffer_error(rows, n)),
    }
}

// Built inline, the error cost within's solves 5–9%.
#[cold]
#[inline(never)]
fn buffer_error(found: usize, n: usize) -> ApplyError {
    match found == n {
        false => ApplyError::Rows { expected: n, found },
        true => ApplyError::Scratch,
    }
}

/// Block-diagonal approximate inverse of the factored matrix, `A` or its [complement](crate::factorize_complement) numbered by the kept side: one [`Component`] per irreducible block of `A`.
#[derive(Clone, Debug)]
pub struct ApproxInverse<T = f64> {
    order: Order,
    blocks: Blocks<T>,
    /// Each lone vertex's whole factor `1/σ`, in `order`'s lone order.
    inverse_pivots: Box<[T]>,
    clamps: Box<[Clamp]>,
    scratch_slots: usize,
}

/// Every coupled block's factor, block `k` on `order`'s segment `k`, all in `u32` while a cover's `2n + 1` slots fit it, else in `u64`.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(deserialize = "T: Real + serde::Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub(crate) enum Blocks<T> {
    U32(Box<[Block<T, u32>]>),
    U64(Box<[Block<T, u64>]>),
}

impl<T> From<Box<[Block<T, u32>]>> for Blocks<T> {
    fn from(blocks: Box<[Block<T, u32>]>) -> Self {
        Self::U32(blocks)
    }
}

impl<T> From<Box<[Block<T, u64>]>> for Blocks<T> {
    fn from(blocks: Box<[Block<T, u64>]>) -> Self {
        Self::U64(blocks)
    }
}

/// Runs `$body` on the blocks, whichever index type stores them.
macro_rules! on {
    ($blocks:expr, $each:ident => $body:expr) => {
        match $blocks {
            Blocks::U32($each) => $body,
            Blocks::U64($each) => $body,
        }
    };
}

impl<T: Real> ApproxInverse<T> {
    /// Factors every coupled summand in parallel, each keyed by its vertices once they are `order`'s.
    pub(crate) fn build<I: Index, B: Summand<T, I>>(
        reduction: Reduction<B, I, T>,
        config: &Config,
    ) -> Self
    where
        Blocks<T>: From<Box<[Block<T, I>]>>,
        Listed: From<Box<[I]>>,
    {
        let Reduction {
            summands: Summands { components, blocks },
            clamps,
        } = reduction;
        let Components {
            vertices,
            coupled,
            lone,
            dead,
            numbering,
        } = components;
        let inverse_pivots = lone.iter().map(|&(_, pivot)| T::one() / pivot).collect();
        let lone = lone.into_iter().map(|(v, _)| v);
        let order = Order::pack(vertices, &coupled, lone, dead.into_iter(), numbering);
        let workspace = || Workspace::new(config.split());
        // Exactly sized: collect's reserve rounds a few blocks up to 4 and into_boxed_slice reallocates.
        let mut factored = Vec::with_capacity(blocks.len());
        (blocks.into_par_iter().enumerate())
            .map_init(workspace, |reused, (k, block)| {
                block.factor(order.block(k), config, reused)
            })
            .collect_into_vec(&mut factored);
        let blocks = Blocks::from(factored.into_boxed_slice());
        Self::new(order, blocks, inverse_pivots, clamps.into())
    }

    fn new(
        order: Order,
        blocks: Blocks<T>,
        inverse_pivots: Box<[T]>,
        clamps: Box<[Clamp]>,
    ) -> Self {
        let scratch_slots = on!(&blocks, blocks => (blocks.iter().enumerate())
            .map(|(k, block)| block.global_scratch_slots(order.block(k)))
            .max())
        .unwrap_or(0);
        Self {
            order,
            blocks,
            inverse_pivots,
            clamps,
            scratch_slots,
        }
    }

    /// Dimension of the factored matrix.
    pub fn n(&self) -> usize {
        self.order.n()
    }

    /// Total elimination steps: every slot but a floating block's gauge, whichever arm factored it.
    pub fn n_steps(&self) -> usize {
        let coupled: usize = on!(&self.blocks, blocks => blocks.iter().map(Block::n_steps).sum());
        coupled + self.inverse_pivots.len()
    }

    /// Every component of `A` the reduction clamped, ordered by lowest vertex, in the input's numbering also for a Schur build.
    pub fn clamps(&self) -> &[Clamp] {
        &self.clamps
    }

    /// One per irreducible block of `A` the factored matrix meets, coupled blocks by their component's lowest vertex in the input's numbering, then isolated vertices; dead vertices are in none.
    pub fn components(&self) -> impl ExactSizeIterator<Item = Component<'_, T>> {
        (0..self.order.len()).map(|k| Component { factor: self, k })
    }

    /// What [`apply_with`](Self::apply_with) takes from its stack for a block of `columns` columns: the most any one block of `A` takes, its Cholesky's slots plus a gather unless its vertices are consecutive, per column solved in place, or one column copied through. It grows with `columns` and serves every block of fewer.
    pub fn scratch(&self, columns: usize, _threads: Threads) -> StackReq {
        if columns == 0 {
            return StackReq::EMPTY;
        }
        let in_place = slots::<T>(self.scratch_slots).array(columns);
        in_place.or(slots::<T>(self.n() + self.scratch_slots))
    }

    /// Applies the approximate inverse to every column of `block` at once, each bit for bit as it would be alone, through scratch allocated for this call; [`ApplyError::Scratch`] if it cannot be.
    pub fn apply<M: MatrixMut<T> + ?Sized>(
        &self,
        block: &mut M,
        threads: Threads,
    ) -> Result<(), ApplyError> {
        let fits = block.rows() == self.n();
        let scratch = fits.then(|| MemBuffer::try_new(self.scratch(block.columns(), threads)));
        let Some(Ok(mut scratch)) = scratch else {
            return Err(buffer_error(block.rows(), self.n()));
        };
        self.apply_with(block, threads, MemStack::new(&mut scratch))
    }

    /// As [`apply`](Self::apply), allocating nothing: `block` holds the right-hand sides on entry and `A⁺b` on floating blocks on return, `0` at vertices in no block; `stack` holds [`scratch`](Self::scratch) for its columns. Either error leaves `block` as it was, and a block without columns is left as it is.
    pub fn apply_with<M: MatrixMut<T> + ?Sized>(
        &self,
        block: &mut M,
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), ApplyError> {
        let columns = block.columns();
        check_buffers(
            block.rows(),
            self.n(),
            stack,
            self.scratch(columns, threads),
        )?;
        if columns > 0 {
            block.solve_by(Kernel {
                inverse: self,
                threads,
                stack,
            });
        }
        Ok(())
    }

    /// Solves `x`, of `n` rows, in place through `stack`, which holds every block's scratch for `V`.
    fn solve_unchecked<V: Batch<T>>(&self, x: &mut [V], threads: Threads, stack: &mut MemStack) {
        on!(&self.blocks, blocks => for (k, block) in blocks.iter().enumerate() {
            block.solve_global(self.order.block(k), x, threads, stack);
        });
        for (v, &inverse_pivot) in self.order.lone().vertices().zip(&self.inverse_pivots) {
            x[v] = x[v].mul(inverse_pivot);
        }
        self.order.dead().vertices().for_each(|v| x[v] = V::zero());
    }
}

/// A coupled summand of the factored matrix: a block of `A`, or a block's complement onto its kept side.
pub(crate) trait Summand<T, I>: Send {
    /// Its factor; `vertices` are its own in the factored matrix's numbering.
    fn factor(
        self,
        vertices: Slots<'_>,
        config: &Config,
        reused: &mut Workspace<T, I>,
    ) -> Block<T, I>;
}

impl<T: Real, I: Index> Summand<T, I> for IrreducibleBlock<T, I> {
    fn factor(
        self,
        vertices: Slots<'_>,
        config: &Config,
        reused: &mut Workspace<T, I>,
    ) -> Block<T, I> {
        let (congruence, off_diagonal, sheets, grounding, grounding_unproven) = self.into_parts();
        let streams = Streams::of(vertices.vertex(0), config);
        let tail = |from: usize| congruence.factor_tail(from);
        let draws = streams.cholesky();
        let (elimination, cholesky) = match off_diagonal {
            OffDiagonal::General(adjacency) => {
                let arm = match config.dense(adjacency.n()) {
                    true => Arm::Exact(Signed::new(adjacency, sheets, grounding)),
                    false => Arm::Approximate(Sddm::of_block(adjacency, sheets, grounding)),
                };
                (None, Cholesky::factor(arm, draws, tail(0), reused))
            }
            OffDiagonal::Bipartite(cross) => {
                let kept = tail(cross.n_rows());
                let factor = |s| Cholesky::factor(s, draws, kept, reused);
                let block = BipartiteBlock::of(cross, sheets, grounding);
                let (elimination, factored) = block.eliminate(config, streams, factor);
                (Some(Box::new(elimination)), factored)
            }
        };
        Block {
            congruence,
            elimination,
            cholesky,
            grounding_unproven,
        }
    }
}

impl<T: Real, I: Index> Summand<T, I> for BlockComplement<T, I> {
    /// Keyed as a whole block keys its draws.
    fn factor(self, _: Slots<'_>, config: &Config, reused: &mut Workspace<T, I>) -> Block<T, I> {
        let (first, congruence, cross, sheets, grounding, grounding_unproven) = self.into_parts();
        let streams = Streams::of(first, config);
        let s = BipartiteBlock::of(cross, sheets, grounding).complement(config, streams);
        let tail = congruence.factor_tail(0);
        let cholesky = Cholesky::factor(s, streams.cholesky(), tail, reused);
        Block {
            congruence,
            elimination: None,
            cholesky,
            grounding_unproven,
        }
    }
}

/// One irreducible block of `A` the factored matrix meets, borrowed from its [`ApproxInverse`].
#[derive(Clone, Copy)]
pub struct Component<'f, T = f64> {
    factor: &'f ApproxInverse<T>,
    /// Coupled while below the factor's coupled blocks, else lone.
    k: usize,
}

impl<T> core::fmt::Debug for Component<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let vertices = self.factor.order.block(self.k);
        f.debug_struct("Component")
            .field("vertices", &vertices)
            .finish()
    }
}

impl<'f, T: Real> Component<'f, T> {
    /// Its vertices in the factored matrix's numbering, in the order its factor numbers them.
    pub fn vertices(self) -> impl ExactSizeIterator<Item = usize> + 'f {
        self.factor.order.block(self.k).vertices()
    }

    /// Whether it was grounded or floated on a scaling that did not settle, so a singular component may have been grounded.
    pub fn grounding_unproven(self) -> bool {
        on!(&self.factor.blocks, blocks => blocks.get(self.k).is_some_and(|b| b.grounding_unproven))
    }

    /// Whether it is folded and solved as singular, `A_c⁺b` or its complement's `S_c⁺b`.
    pub fn floats(self) -> bool {
        on!(&self.factor.blocks, blocks => blocks.get(self.k).is_some_and(|b| b.cholesky.singular()))
    }
}

/// One irreducible block's factor, its local indices stored in `I`; its vertices are its [`Order`]'s.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize, I: Index + serde::Serialize",
        deserialize = "T: Real + serde::Deserialize<'de>, I: Index + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) struct Block<T, I> {
    congruence: Box<Congruence<T>>,
    elimination: Option<Box<BlockElimination<T, I>>>,
    cholesky: Cholesky<T, I>,
    grounding_unproven: bool,
}

impl<T: Real, I: Index> Block<T, I> {
    /// Every slot but a floating block's gauge, eliminated ones and the Cholesky's alike.
    fn n_steps(&self) -> usize {
        let eliminated = (self.elimination.as_deref()).map_or(0, BlockElimination::n_eliminated);
        eliminated + self.cholesky.n_steps()
    }

    /// Slots [`solve_global`](Self::solve_global) takes: room to gather unless the vertices are consecutive, plus the Cholesky's.
    fn global_scratch_slots(&self, slots: Slots<'_>) -> usize {
        slots.gather_slots() + self.cholesky.scratch_slots()
    }

    /// Solves the block's vertices `slots` of `x` in place, through a stack of [`global_scratch_slots`](Self::global_scratch_slots) slots.
    fn solve_global<V: Batch<T>>(
        &self,
        slots: Slots<'_>,
        x: &mut [V],
        threads: Threads,
        stack: &mut MemStack,
    ) {
        slots.on_local(
            x,
            stack,
            // on_local's two arms would otherwise each call it: ~19 instructions per block.
            #[inline(always)]
            |local: &mut [V], rest: &mut MemStack| self.solve_unchecked(local, rest, threads),
        );
    }

    // Both of on_local's arms call it, and rustc's cost model alone keeps those calls.
    #[inline(always)]
    fn solve_unchecked<V: Batch<T>>(&self, x: &mut [V], stack: &mut MemStack, threads: Threads) {
        let singular = self.cholesky.singular();
        self.congruence.restrict(x, singular);
        match &self.elimination {
            None => self.cholesky.solve(x, stack),
            Some(elimination) => {
                elimination.solve(x, threads, |kept| self.cholesky.solve(kept, stack));
                if singular {
                    self.congruence.regauge(x);
                }
            }
        }
        self.congruence.prolongate(x, singular);
    }
}

/// `1/d` for a `d` that is `sgdd::diagonal_in_range`: normal, and at most `ε/MIN_POSITIVE`.
#[cfg(feature = "serde")]
fn inverse_pivot_in_range<T: Real>(inverse: T) -> bool {
    inverse >= T::min_positive_value() && inverse <= T::epsilon() / T::min_positive_value()
}

/// `Σ|t| ≤ 1` up to the terms' rounding: a substitution through such multipliers cannot overflow where its solution fits.
fn within_one<T: Real>(terms: impl Iterator<Item = T>) -> bool {
    let (sum, len) = terms.fold((0.0, 0), |(sum, len), t| {
        (sum + crate::widen(t.abs()), len + 1)
    });
    sum <= 1.0 + 4.0 * (len + 2) as f64 * crate::widen(T::epsilon())
}

/// One block's key, so no draw depends on thread scheduling or on how other blocks routed.
#[derive(Clone, Copy, Debug)]
struct Streams {
    /// The seed and the block mixed once: every star would otherwise repeat them on its first draw's path.
    key: u64,
}

impl Streams {
    /// Keyed by the block's first vertex in `A`, so a complement eliminating the side its whole factor does draws as that factor does.
    fn of(first: usize, config: &Config) -> Self {
        Self {
            key: splitmix(splitmix(config.seed) ^ first as u64),
        }
    }

    /// The approximate Cholesky's one sequence: pivot `t`'s sample is conditioned on every earlier one.
    fn cholesky(self) -> Xoshiro256PlusPlus {
        rng(self, 0)
    }

    /// Star `eliminated`'s own sequence, by its index in the block, never by task.
    fn star(self, eliminated: usize) -> Xoshiro256PlusPlus {
        rng(self, eliminated as u64 + 1)
    }
}

/// The only place a key becomes draws; stage `0` is the Cholesky's, `e + 1` star `e`'s.
fn rng(streams: Streams, stage: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(splitmix(streams.key ^ stage))
}

/// SplitMix64's finalizer: each part changes every bit of the key, so no two keys collide by addition.
fn splitmix(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// `(FORMAT_VERSION, (order, blocks, inverse_pivots, clamps))`: the vertices once, in `order`, and each block's factor without them.
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for ApproxInverse<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let body = (
            &self.order,
            &self.blocks,
            &self.inverse_pivots,
            &self.clamps,
        );
        wire::serialize(serializer, body)
    }
}

/// `(order, blocks, inverse_pivots, clamps)` as decoded.
#[cfg(feature = "serde")]
type Body<T> = (crate::order::Encoded, Blocks<T>, Box<[T]>, Box<[Clamp]>);

/// Checks the order before anything sized by its `n`: offsets and vertices in bounds, each block's parts agreeing on its segment's length, so a run's length is one its block's bytes back; only then whether each vertex is in one place.
#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for ApproxInverse<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (order, blocks, inverse_pivots, clamps): Body<T> = wire::deserialize(deserializer)?;
        let fits = |order: &Order| {
            on!(&blocks, blocks => blocks.len() == order.coupled()
                && (blocks.iter().enumerate()).all(|(k, block)| block.holds(order.block(k).len())))
        };
        let pivots = inverse_pivots.iter().all(|&p| inverse_pivot_in_range(p));
        match Order::decoded(order, inverse_pivots.len()) {
            Some(order) if pivots && fits(&order) && order.once() => {
                Ok(Self::new(order, blocks, inverse_pivots, clamps))
            }
            _ => Err(serde::de::Error::custom("corrupt ApproxInverse")),
        }
    }
}

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Block<T, I> {
    /// Whether its parts agree on `n` vertices.
    fn holds(&self, n: usize) -> bool {
        let kept = match &self.elimination {
            None => Some(n),
            Some(elimination) => n
                .checked_sub(elimination.n_eliminated())
                .filter(|&kept| kept == elimination.n_kept()),
        };
        kept.is_some_and(|kept| kept > 0 && self.congruence.holds(n) && self.cholesky.solves(kept))
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1};
    use crate::sgdd::Reduced;
    use crate::ScalingPolicy;

    /// The factor's one block, which a coupled `u32` input stores in `u32`.
    fn only_block_mut(factor: &mut ApproxInverse) -> &mut Block<f64, u32> {
        match &mut factor.blocks {
            Blocks::U32(blocks) if blocks.len() == 1 => &mut blocks[0],
            _ => unreachable!("one coupled u32 block"),
        }
    }

    fn csr_factor(row_ptrs: &[u32], columns: &[u32], values: &[f64]) -> ApproxInverse {
        let n = (row_ptrs.len() - 1) as u32;
        let csr = CsrRef::new(row_ptrs, columns, values, n).expect("valid csr");
        let summands =
            Reduced::<f64, u32>::reduce(csr.into(), ScalingPolicy::default()).expect("reduces");
        ApproxInverse::build(summands, &Config::default())
    }

    fn grounded() -> ApproxInverse {
        let values = [2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0];
        csr_factor(&[0, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2], &values)
    }

    /// A 2-vertex floating block over a 3-vertex grounded block's Cholesky.
    fn mismatched_cholesky() -> ApproxInverse {
        let mut factor = csr_factor(&[0, 2, 4], &[0, 1, 0, 1], &[1.0, -1.0, -1.0, 1.0]);
        only_block_mut(&mut factor).cholesky = only_block_mut(&mut grounded()).cholesky.clone();
        factor
    }

    /// A bipartite block, so it carries a block elimination.
    fn bipartite() -> ApproxInverse {
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(2, &[0u32, 0, 1, 1], None).expect("valid rows"),
            FactorWidth1::new(2, &[0u32, 1, 0, 1], None).expect("valid columns"),
        )
        .expect("valid gram");
        let summands =
            Reduced::<f64, u32>::reduce(gram.into(), ScalingPolicy::default()).expect("reduces");
        let mut factor = ApproxInverse::build(summands, &Config::default());
        assert!(only_block_mut(&mut factor).elimination.is_some());
        factor
    }

    /// A bipartite block stripped of the elimination its Cholesky follows.
    fn without_elimination() -> ApproxInverse {
        let mut factor = bipartite();
        only_block_mut(&mut factor).elimination = None;
        factor
    }

    /// An elimination of 2 keeping 2, between a 5-vertex block's `F` and a 3-vertex Cholesky.
    fn elimination_keeping_another_count() -> ApproxInverse {
        let elimination = only_block_mut(&mut bipartite()).elimination.clone();
        let counts = elimination.as_ref().map(|e| (e.n_eliminated(), e.n_kept()));
        assert_eq!(counts, Some((2, 2)));
        let mut factor = csr_factor(
            &[0, 2, 5, 8, 11, 13],
            &[0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4],
            &[
                3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0,
            ],
        );
        let block = only_block_mut(&mut factor);
        block.elimination = elimination;
        block.cholesky = only_block_mut(&mut grounded()).cholesky.clone();
        factor
    }

    #[rstest]
    #[case::mismatched_cholesky(mismatched_cholesky())]
    #[case::without_elimination(without_elimination())]
    #[case::elimination_keeping_another_count(elimination_keeping_another_count())]
    fn a_block_whose_parts_disagree_is_rejected_on_decode(#[case] spliced: ApproxInverse) {
        let bytes = postcard::to_stdvec(&spliced).expect("serialize");
        assert_eq!(
            postcard::from_bytes::<ApproxInverse>(&bytes).expect_err("parts that disagree"),
            postcard::Error::SerdeDeCustom
        );
    }
}

#[cfg(test)]
mod storage {
    use super::*;
    use crate::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1, Input};
    use crate::sgdd::Reduced;
    use crate::{ScalingPolicy, SchurMode};

    fn bits(factor: &ApproxInverse, b: &[f64]) -> Vec<u64> {
        let mut x = b.to_vec();
        factor.apply(&mut x, Threads::All).expect("b has length n");
        x.iter().map(|x| x.to_bits()).collect()
    }

    fn rhs(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.37).sin()).collect()
    }

    fn configs() -> [Config; 3] {
        [
            Config::default(),
            Config {
                exact_max_dim: 0,
                split_merge: Some(2),
                ..Config::default()
            },
            Config {
                exact_max_dim: 0,
                complement: SchurMode::Exact,
                ..Config::default()
            },
        ]
    }

    fn assert_storage_agrees(input: Input<'_, f64, u32>) {
        let policy = ScalingPolicy::default();
        let b = rhs(input.n());
        for config in configs() {
            let narrow = ApproxInverse::build(
                Reduced::<f64, u32>::reduce(input, policy).expect("reduces"),
                &config,
            );
            let wide = ApproxInverse::build(
                Reduced::<f64, u64>::reduce(input, policy).expect("reduces"),
                &config,
            );
            assert!(matches!(wide.blocks, Blocks::U64(_)));
            assert_eq!(bits(&narrow, &b), bits(&wide, &b), "{config:?}");
        }
    }

    /// Each block on a stack of exactly its own slots solves alike NaN-filled or zeroed: its slots suffice, and none is read before it is written.
    fn assert_solves_on_its_own_slots(input: Input<'_, f64, u32>) {
        let b = rhs(input.n());
        for config in configs() {
            let reduced =
                Reduced::<f64, u32>::reduce(input, ScalingPolicy::default()).expect("reduces");
            let factor = ApproxInverse::build(reduced, &config);
            let Blocks::U32(blocks) = &factor.blocks else {
                unreachable!("built in u32")
            };
            for (k, block) in blocks.iter().enumerate() {
                let slots = factor.order.block(k);
                let len = block.global_scratch_slots(slots);
                let solve = |fill: f64| {
                    let mut x = b.clone();
                    let mut cells = vec![std::mem::MaybeUninit::new(fill); len];
                    block.solve_global(slots, &mut x, Threads::All, MemStack::new_any(&mut cells));
                    x.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                };
                assert_eq!(solve(0.0), solve(f64::NAN), "{config:?}");
            }
        }
    }

    /// `rows × cols` grid, `-1` off the diagonal but `sign` on row 0's edges, each diagonal `diagonal` times its degree.
    fn grid(rows: u32, cols: u32, sign: f64, diagonal: f64) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        let (mut row_ptrs, mut columns, mut values) = (vec![0], vec![], vec![]);
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                let mut neighbours = vec![];
                (c > 0).then(|| neighbours.push(v - 1));
                (c + 1 < cols).then(|| neighbours.push(v + 1));
                (r > 0).then(|| neighbours.push(v - cols));
                (r + 1 < rows).then(|| neighbours.push(v + cols));
                let degree = neighbours.len() as f64;
                neighbours.push(v);
                neighbours.sort_unstable();
                for &u in &neighbours {
                    columns.push(u);
                    let row_zero = u < cols && v < cols;
                    let off = if row_zero { sign } else { -1.0 };
                    values.push(if u == v { diagonal * degree } else { off });
                }
                row_ptrs.push(columns.len() as u32);
            }
        }
        (row_ptrs, columns, values)
    }

    /// A floating Laplacian, signs no ±1 folds (a cover), a block scaled to dominance, interleaved paths that gather, and a scaled gram.
    fn each_input(mut check: impl FnMut(Input<'_, f64, u32>)) {
        for (sign, diagonal) in [(-1.0, 1.0), (1.0, 1.0), (-1.0, 0.8)] {
            let (row_ptrs, columns, values) = grid(7, 6, sign, diagonal);
            check(
                CsrRef::new(&row_ptrs, &columns, &values, 42)
                    .expect("valid CSR")
                    .into(),
            );
        }
        let (row_ptrs, columns) = (
            [0u32, 2, 4, 7, 10, 12, 14],
            [0u32, 2, 1, 3, 0, 2, 4, 1, 3, 5, 2, 4, 3, 5],
        );
        let values = [
            1.0, -1.0, 1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0, -1.0, 1.0,
        ];
        check(
            CsrRef::new(&row_ptrs, &columns, &values, 6)
                .expect("valid CSR")
                .into(),
        );
        let rows: Vec<u32> = (0..60).map(|k| k % 12).collect();
        let cols: Vec<u32> = (0..60).map(|k| (k * 7 + k / 12) % 9).collect();
        let scales: Vec<f64> = (0..60).map(|k| 1.0 + (k % 5) as f64).collect();
        let gram = BipartiteFactorWidth2::scaled(
            FactorWidth1::new(12, &rows, None).expect("valid rows"),
            FactorWidth1::new(9, &cols, None).expect("valid columns"),
            &scales,
        )
        .expect("valid gram");
        check(gram.into());
    }

    #[test]
    fn u64_storage_solves_bit_identically_to_u32() {
        each_input(assert_storage_agrees);
    }

    #[test]
    fn each_block_solves_on_a_stack_of_its_own_slots() {
        each_input(assert_solves_on_its_own_slots);
    }
}
