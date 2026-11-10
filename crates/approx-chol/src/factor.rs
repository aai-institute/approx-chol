//! Each irreducible block as an optional block elimination `L` and a Cholesky of what it leaves: `A_c⁺ ≈ P F L⁻ᵀ diag(D_e⁻¹, S⁺) L⁻¹ F P`, `P` projecting out `F·1` when the block floats, else `I`.

mod apply;
mod block_elimination;
mod cholesky;
mod clique_tree;
mod config;
mod schur;
mod sddm;
mod sign;
#[cfg(feature = "serde")]
mod wire;

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;
use rayon::prelude::*;

pub use apply::{ApplyError, ColumnMajor, Columns, LayoutError, MatrixMut};
pub use config::{Config, CoverSurplus, FrustratedRoute, Sampling, SchurMode};
#[cfg(feature = "serde")]
pub use wire::FORMAT_VERSION;

use crate::order::{Listed, Order, Segment, Slots};
use crate::sgdd::{ranges, BlockComplement, Congruence, IrreducibleBlock, KeptVertices, Summands};
use crate::storage::OffDiagonal;
use crate::threads::task;
use crate::Clamp;
use crate::{Batch, Index, MemStack, Real, StackReq, Threads};
pub(crate) use apply::Kernel;
#[cfg(feature = "faer")]
pub(crate) use apply::Slices;
use apply::{rows_error, Room, Stack};
use block_elimination::BlockElimination;
use cholesky::{Cholesky, Workspace};
use schur::BipartiteBlock;
use sddm::Factored;

/// `log₂` of the multiply-adds a block elimination's task carries, below which a split costs more than it saves.
const MULTIPLY_ADD_GRAIN_BITS: u32 = 17;

/// `count` slots of `V` on a stack; none asks no alignment either, so any stack serves a solve that takes none.
fn slots<V>(count: usize) -> StackReq {
    match count {
        0 => StackReq::EMPTY,
        count => StackReq::new::<V>(count),
    }
}

/// Block-diagonal approximate inverse of the factored matrix, `A` or its [complement](crate::factorize_complement) numbered by the kept side: one [`Component`] per irreducible block of `A`.
#[derive(Clone, Debug)]
pub struct ApproxInverse<T = f64> {
    order: Order,
    blocks: Blocks<T>,
    /// Each lone vertex's whole factor `1/σ`, in the order `order` lists them ahead of the dead ones.
    inverse_pivots: Box<[T]>,
    /// The coupled blocks' elimination steps.
    coupled_steps: usize,
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

mod spread;

impl<T: Real> ApproxInverse<T> {
    /// Factors every coupled summand in parallel, each keyed by its vertices once they are `order`'s.
    pub(crate) fn build<I: Index, B: Summand<T, I>>(
        summands: Summands<B, I, T>,
        config: &Config,
        threads: Threads,
    ) -> Self
    where
        Blocks<T>: From<Box<[Block<T, I>]>>,
        Listed: From<Box<[I]>>,
    {
        let Summands {
            vertices,
            coupled,
            pivots,
            numbering,
        } = summands;
        let inverse_pivots = pivots.iter().map(|&pivot| T::one() / pivot).collect();
        let (order, segments) = Order::pack(vertices, ranges(&coupled), numbering);
        let workspace = || Workspace::new(config.sampling.copies());
        // Exactly sized: collect's reserve rounds a few blocks up to 4 and into_boxed_slice reallocates.
        let mut factored = Vec::with_capacity(coupled.len());
        match threads.per_task(coupled.len(), 1) {
            Some(per) => {
                let nested = threads.within_task(coupled.len().div_ceil(per));
                (coupled.into_par_iter().zip(segments))
                    .with_min_len(per)
                    .map_init(workspace, |reused, (block, segment)| {
                        task(|| block.factor(segment, &order, config, nested, reused))
                    })
                    .collect_into_vec(&mut factored);
            }
            None => {
                let mut reused = workspace();
                factored.extend(coupled.into_iter().zip(segments).map(|(block, segment)| {
                    block.factor(segment, &order, config, threads, &mut reused)
                }));
            }
        }
        let blocks = Blocks::from(factored.into_boxed_slice());
        Self::new(order, blocks, inverse_pivots)
    }

    fn new(order: Order, blocks: Blocks<T>, inverse_pivots: Box<[T]>) -> Self {
        let coupled_steps = on!(&blocks, blocks => blocks.iter().map(Block::n_steps).sum());
        Self {
            order,
            blocks,
            inverse_pivots,
            coupled_steps,
        }
    }

    /// Dimension of the factored matrix.
    pub fn n(&self) -> usize {
        self.order.n()
    }

    /// Total elimination steps: every slot but a floating block's gauge, whichever arm factored it.
    pub fn n_steps(&self) -> usize {
        self.coupled_steps + self.inverse_pivots.len()
    }

    /// One per irreducible block of `A` the factored matrix meets, coupled blocks by their component's lowest vertex in the input's numbering, then isolated vertices; dead vertices are in none.
    pub fn components(&self) -> impl ExactSizeIterator<Item = Component<'_, T>> {
        let coupled = on!(&self.blocks, blocks => blocks.len());
        (0..coupled + self.inverse_pivots.len()).map(|k| Component { factor: self, k })
    }

    /// What [`apply_with`](Self::apply_with) takes from its stack for any block of `columns` columns at `threads`, whichever its layout; a copied block takes one pack's scratch whatever its columns. It serves every block of fewer columns; one thread's serves every budget, solving fewer blocks at once to the same bits.
    pub fn scratch(&self, columns: usize, threads: Threads) -> StackReq {
        if columns == 0 {
            return StackReq::EMPTY;
        }
        let room = Room::of(self, threads);
        room.in_place(columns).or(room.staged(columns))
    }

    /// Applies the approximate inverse to every column of `block` at once, each bit for bit as it would be alone, through scratch allocated for this call, what `block`'s layout takes at `threads`; [`ApplyError::Scratch`] if it cannot be. Its rows and errors are [`apply_with`](Self::apply_with)'s.
    pub fn apply<M: MatrixMut<T> + ?Sized>(
        &self,
        block: &mut M,
        threads: Threads,
    ) -> Result<(), ApplyError> {
        self.apply_on(block, threads, Stack::Allocate)
    }

    /// As [`apply`](Self::apply), allocating nothing at `Count(1)`; a budget above one runs on rayon's pool, which builds itself on first use: `block` holds the right-hand sides on entry and `A⁺b` on floating blocks on return, `0` at vertices in no block. A block without columns is left as it is, whatever its rows; one with columns is [`ApplyError::Rows`] unless it has `n` rows, then [`ApplyError::Scratch`] unless `stack` holds what its layout takes at one thread, at most [`scratch`](Self::scratch) for its columns at one thread; either error leaves `block` as it was. The stack solves as many blocks at once as it holds regions, up to `threads`. A floating block's null vector is projected out of `b`, so a right-hand side along it returns `0` on that block: a preconditioned CG must stop at `rᵀz = 0` rather than divide by it.
    pub fn apply_with<M: MatrixMut<T> + ?Sized>(
        &self,
        block: &mut M,
        threads: Threads,
        stack: &mut MemStack,
    ) -> Result<(), ApplyError> {
        self.apply_on(block, threads, Stack::Holds(stack))
    }

    fn apply_on<M: MatrixMut<T> + ?Sized>(
        &self,
        block: &mut M,
        threads: Threads,
        stack: Stack<'_>,
    ) -> Result<(), ApplyError> {
        if block.columns() == 0 {
            return Ok(());
        }
        if block.rows() != self.n() {
            return Err(rows_error(block.rows(), self.n()));
        }
        // An empty system has nothing to solve, and a stack that holds nothing may not be aligned to carve from.
        if self.n() == 0 {
            return Ok(());
        }
        block.solve_by(Kernel {
            inverse: self,
            threads,
            stack,
        })
    }

    /// Solves `x`, of `n` rows, in place through `stack`, which holds every block's scratch for `V`.
    fn solve_unchecked<V: Batch<T>>(&self, x: &mut [V], threads: Threads, stack: &mut MemStack) {
        spread::solve_blocks(self, x, threads, stack);
        let (lone, dead) = self.order.lone_and_dead(self.inverse_pivots.len());
        for (v, &inverse_pivot) in lone.vertices().zip(&self.inverse_pivots) {
            x[v] = x[v].mul(inverse_pivot);
        }
        dead.vertices().for_each(|v| x[v] = V::zero());
    }
}

/// A coupled summand of the factored matrix: a block of `A`, or a block's complement onto its kept side.
pub(crate) trait Summand<T, I>: Send + KeptVertices {
    /// Its factor on `segment` of `order`, its vertices in the factored matrix's numbering.
    fn factor(
        self,
        segment: Segment,
        order: &Order,
        config: &Config,
        threads: Threads,
        reused: &mut Workspace<T, I>,
    ) -> Block<T, I>;
}

impl<T: Real, I: Index> Summand<T, I> for IrreducibleBlock<T, I> {
    fn factor(
        self,
        segment: Segment,
        order: &Order,
        config: &Config,
        threads: Threads,
        reused: &mut Workspace<T, I>,
    ) -> Block<T, I> {
        let (congruence, off_diagonal, balance, grounding, grounding_unproven, clamp) =
            self.into_parts();
        let streams = Streams::of(order.slots(segment).vertex(0), config);
        let tail = |from: usize| congruence.factor_tail(from);
        let draws = streams.cholesky();
        let (elimination, cholesky) = match off_diagonal {
            OffDiagonal::General(adjacency) => {
                let arm = Factored::of(config, balance, adjacency.n()).arm(adjacency, grounding);
                (None, Cholesky::factor(arm, draws, tail(0), reused))
            }
            OffDiagonal::Bipartite(cross) => {
                let kept = tail(cross.n_rows());
                let factor = |s| Cholesky::factor(s, draws, kept, reused);
                let block = BipartiteBlock::of(cross, balance, grounding);
                let (elimination, factored) = block.eliminate(config, streams, threads, factor);
                (Some(Box::new(elimination)), factored)
            }
        };
        Block {
            segment,
            congruence,
            elimination,
            cholesky,
            grounding_unproven,
            clamp,
        }
    }
}

impl<T: Real, I: Index> Summand<T, I> for BlockComplement<T, I> {
    /// Keyed as a whole block keys its draws.
    fn factor(
        self,
        segment: Segment,
        _: &Order,
        config: &Config,
        threads: Threads,
        reused: &mut Workspace<T, I>,
    ) -> Block<T, I> {
        let (first, congruence, cross, balance, grounding, grounding_unproven, clamp) =
            self.into_parts();
        let streams = Streams::of(first, config);
        let s = BipartiteBlock::of(cross, balance, grounding).complement(config, streams, threads);
        let tail = congruence.factor_tail(0);
        let cholesky = Cholesky::factor(s, streams.cholesky(), tail, reused);
        Block {
            segment,
            congruence,
            elimination: None,
            cholesky,
            grounding_unproven,
            clamp,
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

impl<'f, T> Component<'f, T> {
    fn slots(&self) -> Slots<'f> {
        let factor = self.factor;
        on!(&factor.blocks, blocks => match blocks.get(self.k) {
            Some(block) => factor.order.slots(block.segment),
            None => factor.order.lone(self.k - blocks.len()),
        })
    }
}

impl<T> core::fmt::Debug for Component<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let vertices = self.slots();
        f.debug_struct("Component")
            .field("vertices", &vertices)
            .finish()
    }
}

impl<'f, T: Real> Component<'f, T> {
    /// Its vertices in the factored matrix's numbering, in the order its factor numbers them.
    pub fn vertices(self) -> impl ExactSizeIterator<Item = usize> + 'f {
        self.slots().vertices()
    }

    /// Whether it was grounded or floated on a scaling that did not settle, so a singular component may have been grounded.
    pub fn grounding_unproven(self) -> bool {
        on!(&self.factor.blocks, blocks => blocks.get(self.k).is_some_and(|b| b.grounding_unproven))
    }

    /// How the reduction clamped its rows to make it dominant, if it did; a complement's is its whole block's.
    pub fn clamp(self) -> Option<&'f Clamp> {
        on!(&self.factor.blocks, blocks => blocks.get(self.k).and_then(|b| b.clamp.as_deref()))
    }

    /// Whether it is folded and solved as singular, `A_c⁺b` or its complement's `S_c⁺b`.
    pub fn floats(self) -> bool {
        on!(&self.factor.blocks, blocks => blocks.get(self.k).is_some_and(|b| b.cholesky.singular()))
    }
}

/// One irreducible block's factor, its local indices stored in `I`, and its vertices as a segment of its [`Order`].
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
    segment: Segment,
    congruence: Box<Congruence<T>>,
    elimination: Option<Box<BlockElimination<T, I>>>,
    cholesky: Cholesky<T, I>,
    grounding_unproven: bool,
    clamp: Option<Box<Clamp>>,
}

impl<T: Real, I: Index> Block<T, I> {
    /// Every slot but a floating block's gauge, eliminated ones and the Cholesky's alike.
    fn n_steps(&self) -> usize {
        let eliminated = (self.elimination.as_deref()).map_or(0, BlockElimination::n_eliminated);
        eliminated + self.cholesky.n_steps()
    }

    /// Slots [`solve_global`](Self::solve_global) takes: room to gather unless the vertices are consecutive, plus the Cholesky's.
    fn global_scratch_slots(&self) -> usize {
        self.segment.gather_slots() + self.cholesky.scratch_slots()
    }

    /// Solves the block's vertices of `x`, its segment of `order`, in place through a stack of [`global_scratch_slots`](Self::global_scratch_slots) slots.
    fn solve_global<V: Batch<T>>(
        &self,
        order: &Order,
        x: &mut [V],
        threads: Threads,
        stack: &mut MemStack,
    ) {
        order.slots(self.segment).on_local(
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

/// `1/d` for a `d` that is `crate::diagonal_in_range`: normal, and at most `ε/MIN_POSITIVE`.
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

/// `(FORMAT_VERSION, (listed, blocks, inverse_pivots))`: the vertices once, each block's segment a run or a length of the order's `listed`.
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for ApproxInverse<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let body = (self.order.listed(), &self.blocks, &self.inverse_pivots);
        wire::serialize(serializer, body)
    }
}

/// `(listed, blocks, inverse_pivots)` as decoded.
#[cfg(feature = "serde")]
type Body<T> = (Listed, Blocks<T>, Box<[T]>);

/// Checks the order and the blocks' segments before anything sized by their `n`: offsets and vertices in bounds, each block's parts agreeing on its segment's length, so a run's length is one its block's bytes back; only then whether each vertex is in one place.
#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for ApproxInverse<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (order, mut blocks, inverse_pivots): Body<T> = wire::deserialize(deserializer)?;
        let order = on!(&mut blocks, blocks => Order::decoded(order, blocks.iter_mut().map(|b| &mut b.segment)));
        let fits = || on!(&blocks, blocks => blocks.iter().all(Block::holds));
        let once =
            |order: &Order| on!(&blocks, blocks => order.once(blocks.iter().map(|b| b.segment)));
        let pivots = inverse_pivots.iter().all(|&p| inverse_pivot_in_range(p));
        let lone = |order: &Order| inverse_pivots.len() <= order.isolated().len();
        match order {
            Some(order) if pivots && lone(&order) && fits() && once(&order) => {
                Ok(Self::new(order, blocks, inverse_pivots))
            }
            _ => Err(serde::de::Error::custom("corrupt ApproxInverse")),
        }
    }
}

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Block<T, I> {
    /// Whether its parts agree on `n` vertices.
    fn holds(&self) -> bool {
        let n = self.segment.len();
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
        let n = row_ptrs.len() - 1;
        let csr = CsrRef::new(n, row_ptrs, columns, values).expect("valid csr");
        let summands =
            Reduced::<f64, u32>::reduce(csr.into(), ScalingPolicy::default(), Threads::All)
                .expect("reduces");
        ApproxInverse::build(summands, &Config::default(), Threads::All)
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
            FactorWidth1::new(2, &[0u32, 0, 1, 1]).expect("valid rows"),
            FactorWidth1::new(2, &[0u32, 1, 0, 1]).expect("valid columns"),
        )
        .expect("valid gram");
        let summands =
            Reduced::<f64, u32>::reduce(gram.into(), ScalingPolicy::default(), Threads::All)
                .expect("reduces");
        let mut factor = ApproxInverse::build(summands, &Config::default(), Threads::All);
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
    use crate::{Sampling, ScalingPolicy, SchurMode};

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
                sampling: Sampling::CliqueTree {
                    copies: std::num::NonZeroU32::new(2).expect("two"),
                },
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
                Reduced::<f64, u32>::reduce(input, policy, Threads::All).expect("reduces"),
                &config,
                Threads::All,
            );
            let wide = ApproxInverse::build(
                Reduced::<f64, u64>::reduce(input, policy, Threads::All).expect("reduces"),
                &config,
                Threads::All,
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
                Reduced::<f64, u32>::reduce(input, ScalingPolicy::default(), Threads::All)
                    .expect("reduces");
            let factor = ApproxInverse::build(reduced, &config, Threads::All);
            let Blocks::U32(blocks) = &factor.blocks else {
                unreachable!("built in u32")
            };
            for block in blocks.iter() {
                let len = block.global_scratch_slots();
                let solve = |fill: f64| {
                    let mut x = b.clone();
                    let mut cells = vec![std::mem::MaybeUninit::new(fill); len];
                    let stack = MemStack::new_any(&mut cells);
                    block.solve_global(&factor.order, &mut x, Threads::All, stack);
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
                CsrRef::new(42, &row_ptrs, &columns, &values)
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
            CsrRef::new(6, &row_ptrs, &columns, &values)
                .expect("valid CSR")
                .into(),
        );
        let rows: Vec<u32> = (0..60).map(|k| k % 12).collect();
        let cols: Vec<u32> = (0..60).map(|k| (k * 7 + k / 12) % 9).collect();
        let scales: Vec<f64> = (0..60).map(|k| 1.0 + (k % 5) as f64).collect();
        let gram = BipartiteFactorWidth2::scaled(
            FactorWidth1::new(12, &rows).expect("valid rows"),
            FactorWidth1::new(9, &cols).expect("valid columns"),
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

#[cfg(test)]
mod budget {
    use std::num::{NonZeroU32, NonZeroUsize};

    use rstest::rstest;

    use super::*;
    use crate::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1, Side};
    use crate::sgdd::{Complement, Reduced};
    use crate::threads::{probe, ONE};
    use crate::{widen, CoverSurplus, FrustratedRoute, Sampling, ScalingPolicy, SchurMode};

    const THREE: Threads = Threads::Count(NonZeroUsize::new(3).unwrap());

    /// The factor as shown, every stored field, and its apply's bits.
    fn fingerprint<T: Real>(factor: ApproxInverse<T>) -> (String, Vec<u64>) {
        let mut x: Vec<_> = (0..factor.n())
            .map(|i| T::from((i % 101) as f64 - 50.0).expect("small"))
            .collect();
        factor.apply(&mut x, ONE).expect("b has length n");
        let bits = x.into_iter().map(|x| widen(x).to_bits()).collect();
        (format!("{factor:?}"), bits)
    }

    #[derive(Clone, Copy, Debug)]
    enum Case {
        /// `groups` components of 400 left and 200 right vertices, 48 entries a left vertex, the second frustrated: one block splits its star fold and exact rows, three split the blocks.
        Bipartite(u32),
        /// `groups` rings of 2000 vertices, each joined to its six nearest on either side and scaled off dominance, so the Perron product gathers; relaxed `0`, `8` and `100` iterations.
        Perron64(u32),
        Perron32(u32),
    }

    const CASES: [Case; 6] = [
        Case::Bipartite(1),
        Case::Bipartite(3),
        Case::Perron64(1),
        Case::Perron64(8),
        Case::Perron32(1),
        Case::Perron32(8),
    ];

    type Build<'a> = &'a (dyn Fn(Threads) -> (String, Vec<u64>) + Sync);

    impl Case {
        /// Each build of the case, run by `check`.
        fn each_build(self, mut check: impl FnMut(Build<'_>)) {
            match self {
                Case::Bipartite(groups) => bipartite(groups, &mut check),
                Case::Perron64(groups) => perron::<f64>(groups, &mut check),
                Case::Perron32(groups) => perron::<f32>(groups, &mut check),
            }
        }
    }

    fn bipartite(groups: u32, check: &mut impl FnMut(Build<'_>)) {
        let (mut left, mut right, mut x, mut y) = (vec![], vec![], vec![], vec![]);
        for group in 0..groups {
            for i in 0..400 {
                for d in 0..48 {
                    left.push(group * 400 + i);
                    right.push(group * 200 + (i * 7 + d * 13) % 200);
                    x.push(1.0 + (d % 7) as f64 / 8.0);
                    y.push(if group == 1 && d % 5 == 0 { 0.5 } else { -0.5 });
                }
            }
        }
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::valued(groups as usize * 400, &left, &x).expect("valid rows"),
            FactorWidth1::valued(groups as usize * 200, &right, &y).expect("valid columns"),
        )
        .expect("valid gram");
        let policy = ScalingPolicy::default();
        let configs = [
            Config::default(),
            Config {
                sampling: Sampling::CliqueTree {
                    copies: NonZeroU32::new(3).unwrap(),
                },
                ..Config::default()
            },
            Config {
                complement: SchurMode::Exact,
                ..Config::default()
            },
            Config {
                frustrated: FrustratedRoute::Cover(CoverSurplus::SurplusEdges),
                ..Config::default()
            },
            Config {
                frustrated: FrustratedRoute::Signed,
                ..Config::default()
            },
        ];
        for config in configs {
            for kept in [None, Some(Side::Left), Some(Side::Right)] {
                check(&|threads| {
                    fingerprint(match kept {
                        None => ApproxInverse::build(
                            Reduced::<_, u32>::reduce(gram.into(), policy, threads)
                                .expect("reduces"),
                            &config,
                            threads,
                        ),
                        Some(side) => ApproxInverse::build(
                            Complement::<_, u32>::reduce_onto(gram, side, policy, threads)
                                .expect("reduces"),
                            &config,
                            threads,
                        ),
                    })
                });
            }
        }
    }

    fn perron<T: Real>(groups: u32, check: &mut impl FnMut(Build<'_>)) {
        let size = 2000u32;
        let n = size * groups;
        let (mut ptrs, mut columns, mut values) = (vec![0u32], vec![], vec![]);
        let scale = |i: u32| 1.0 + (i % 17) as f64;
        for i in 0..n {
            let mut row = vec![(i, 12.0 * scale(i).powi(2))];
            for offset in 1..=6 {
                for j in [(i + offset) % size, (i + size - offset) % size] {
                    let j = i / size * size + j;
                    row.push((j, -scale(i) * scale(j)));
                }
            }
            row.sort_by_key(|&(j, _)| j);
            for (j, value) in row {
                columns.push(j);
                values.push(T::from(value).expect("small"));
            }
            ptrs.push(columns.len() as u32);
        }
        let csr = CsrRef::new(n as usize, &ptrs, &columns, &values).expect("valid csr");
        for iterations in [0, 8, 100] {
            let policy = ScalingPolicy::new(1e-9, iterations).expect("valid policy");
            check(&|threads| {
                fingerprint(ApproxInverse::build(
                    Reduced::<_, u32>::reduce(csr.into(), policy, threads).expect("reduces"),
                    &Config::default(),
                    threads,
                ))
            });
        }
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("{}-{i}", probe::POOL))
            .build()
            .expect("pool")
    }

    /// Under `All`, `Count(1)` and `Count(3)`, in pools of one, two and seven threads, each build equals it under `All` outside any.
    #[rstest]
    fn a_build_is_bit_identical_across_budgets(
        #[values(CASES[0], CASES[1], CASES[2], CASES[3], CASES[4], CASES[5])] case: Case,
    ) {
        let pools = [1, 2, 7].map(|size| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(size)
                .build()
                .expect("pool")
        });
        case.each_build(|build| {
            let reference = build(Threads::All);
            for pool in &pools {
                let size = pool.current_num_threads();
                for budget in [Threads::All, ONE, THREE] {
                    let built = pool.install(|| build(budget));
                    assert!(built == reference, "{case:?}, pool {size}, {budget:?}");
                }
            }
        });
    }

    /// In a pool of seven, a count's build never has more of its threads in a loop's items at once; the pool's whole budget is seen to.
    #[test]
    fn a_build_runs_at_most_its_count_of_threads() {
        let pool = pool(7);
        let mut widest = 0;
        for case in CASES {
            case.each_build(|build| {
                for (budget, n) in [(ONE, 1), (THREE, 3)] {
                    let peak = probe::peak(|| drop(pool.install(|| build(budget))));
                    assert!(peak <= n, "{case:?}: {peak} threads under {budget:?}");
                }
                widest = widest.max(probe::peak(|| drop(pool.install(|| build(Threads::All)))));
            });
        }
        assert!(
            widest > 3,
            "the probe saw at most {widest} threads under All"
        );
    }
}
