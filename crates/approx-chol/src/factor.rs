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

use crate::sgdd::{BlockComplement, IrreducibleBlock, Reduction, Restriction, Selection, Summands};
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

/// Block-diagonal approximate inverse of the factored matrix, `A` or its [complement](crate::factorize_complement) numbered by the kept side: one [`BlockFactor`] per irreducible block of `A`.
#[derive(Clone, Debug)]
pub struct ApproxInverse<T = f64> {
    n: usize,
    blocks: Vec<BlockFactor<T>>,
    /// Vertices in no block, which solve to `0`.
    dead: Box<[usize]>,
    clamps: Box<[Clamp]>,
    scratch_slots: usize,
}

impl<T: Real> ApproxInverse<T> {
    /// Factors every coupled summand in parallel.
    pub(crate) fn build<I: Index, B: Summand<T, I>>(
        reduction: Reduction<B, I, T>,
        config: &Config,
        store: fn(Block<T, I>) -> Storage<T>,
    ) -> Self {
        let Reduction {
            summands:
                Summands {
                    blocks: coupled,
                    lone,
                    dead,
                },
            clamps,
        } = reduction;
        let mut blocks = Vec::with_capacity(coupled.len() + lone.len());
        let build = |reused: &mut _, block: B| BlockFactor(store(block.factor(config, reused)));
        let workspace = || Workspace::new(config.split());
        blocks.par_extend(coupled.into_par_iter().map_init(workspace, build));
        let isolated = |(v, pivot)| BlockFactor(Storage::Isolated(Isolated::new(v, pivot)));
        blocks.extend(lone.into_iter().map(isolated));
        let dead = dead.into_iter().map(I::index).collect();
        Self::new(blocks, dead, clamps.into())
    }

    fn new(blocks: Vec<BlockFactor<T>>, dead: Box<[usize]>, clamps: Box<[Clamp]>) -> Self {
        let n = blocks.iter().map(|b| b.vertices().len()).sum::<usize>() + dead.len();
        let scratch_slots = (blocks.iter())
            .map(|block| on!(block, b => b.global_scratch_slots(), _ => 0))
            .max()
            .unwrap_or(0);
        Self {
            n,
            blocks,
            dead,
            clamps,
            scratch_slots,
        }
    }

    /// Dimension of the factored matrix.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Total elimination steps: every slot but a floating block's gauge, whichever arm factored it.
    pub fn n_steps(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| on!(block, b => b.n_steps(), _ => 1))
            .sum()
    }

    /// Every component of `A` the reduction clamped, ordered by first vertex, in the input's numbering also for a Schur build.
    pub fn clamps(&self) -> &[Clamp] {
        &self.clamps
    }

    /// The blocks' factors, one per irreducible block of `A` the factored matrix meets, coupled blocks before isolated vertices; dead vertices have none.
    pub fn blocks(&self) -> &[BlockFactor<T>] {
        &self.blocks
    }

    /// The blocks' factors, owned, for a caller that keeps one solver per block; dead vertices need none.
    pub fn into_blocks(self) -> Vec<BlockFactor<T>> {
        self.blocks
    }

    /// What [`apply_with`](Self::apply_with) takes from its stack for a block of `columns` columns: the most any one block of `A` takes, its Cholesky's slots plus a gather unless its vertices are consecutive, per column solved in place, or one column copied through. It grows with `columns` and serves every block of fewer.
    pub fn scratch(&self, columns: usize, _threads: Threads) -> StackReq {
        if columns == 0 {
            return StackReq::EMPTY;
        }
        let in_place = slots::<T>(self.scratch_slots).array(columns);
        in_place.or(slots::<T>(self.n + self.scratch_slots))
    }

    /// Applies the approximate inverse to every column of `block` at once, each bit for bit as it would be alone, through scratch allocated for this call; [`ApplyError::Scratch`] if it cannot be.
    pub fn apply<M: MatrixMut<T> + ?Sized>(
        &self,
        block: &mut M,
        threads: Threads,
    ) -> Result<(), ApplyError> {
        let fits = block.rows() == self.n;
        let scratch = fits.then(|| MemBuffer::try_new(self.scratch(block.columns(), threads)));
        let Some(Ok(mut scratch)) = scratch else {
            return Err(buffer_error(block.rows(), self.n));
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
        check_buffers(block.rows(), self.n, stack, self.scratch(columns, threads))?;
        if columns > 0 {
            block.solve_by(Kernel {
                inverse: self,
                threads,
                stack,
            });
        }
        Ok(())
    }

    /// `A⁺b` on floating blocks; vertices in no block solve to `0`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, ApplyError> {
        let mut x = b.to_vec();
        self.apply(x.as_mut_slice(), Threads::All)?;
        Ok(x)
    }

    /// As [`solve`](Self::solve), into `x` of length [`n`](Self::n).
    pub fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), ApplyError> {
        if let Some(found) = [b.len(), x.len()].into_iter().find(|&len| len != self.n) {
            return Err(ApplyError::Rows {
                expected: self.n,
                found,
            });
        }
        x.copy_from_slice(b);
        self.apply(x, Threads::All)
    }

    /// As [`solve`](Self::solve), allocating nothing: `x` holds `b` on entry and the solution on return; `stack` holds [`scratch`](Self::scratch) for one column.
    pub fn solve_in_place(&self, x: &mut [T], stack: &mut MemStack) -> Result<(), ApplyError> {
        self.apply_with(x, Threads::All, stack)
    }

    /// Solves `x`, of `n` rows, in place through `stack`, which holds every block's scratch for `V`.
    fn solve_unchecked<V: Batch<T>>(&self, x: &mut [V], threads: Threads, stack: &mut MemStack) {
        for block in &self.blocks {
            on!(block, b => b.solve_global(x, threads, stack), i => i.solve_global(x));
        }
        self.dead.iter().for_each(|&v| x[v] = V::zero());
    }
}

/// A coupled summand of the factored matrix: a block of `A`, or a block's complement onto its kept side.
pub(crate) trait Summand<T, I>: Send {
    fn factor(self, config: &Config, reused: &mut Workspace<T, I>) -> Block<T, I>;
}

impl<T: Real, I: Index> Summand<T, I> for IrreducibleBlock<T, I> {
    fn factor(self, config: &Config, reused: &mut Workspace<T, I>) -> Block<T, I> {
        let (restriction, off_diagonal, sheets, grounding, grounding_unproven) = self.into_parts();
        let streams = Streams::of(restriction.vertices().vertex(0), config);
        let tail = |from: usize| restriction.factor_tail(from);
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
            restriction,
            elimination,
            cholesky,
            grounding_unproven,
        }
    }
}

impl<T: Real, I: Index> Summand<T, I> for BlockComplement<T, I> {
    /// Keyed as a whole block keys its draws.
    fn factor(self, config: &Config, reused: &mut Workspace<T, I>) -> Block<T, I> {
        let (first, restriction, cross, sheets, grounding, grounding_unproven) = self.into_parts();
        let streams = Streams::of(first, config);
        let s = BipartiteBlock::of(cross, sheets, grounding).complement(config, streams);
        let tail = restriction.factor_tail(0);
        let cholesky = Cholesky::factor(s, streams.cholesky(), tail, reused);
        Block {
            restriction,
            elimination: None,
            cholesky,
            grounding_unproven,
        }
    }
}

/// Runs `$body` on a [`BlockFactor`]'s block, whichever index type stores it, or `$alone` on an isolated vertex.
macro_rules! on {
    ($factor:expr, $block:ident => $body:expr, $isolated:pat => $alone:expr) => {
        match &$factor.0 {
            Storage::U32($block) => $body,
            Storage::U64($block) => $body,
            Storage::Isolated($isolated) => $alone,
        }
    };
}
use on;

/// The factor of one irreducible block of `A`, over its vertices of the factored matrix.
#[derive(Clone, Debug)]
pub struct BlockFactor<T = f64>(Storage<T>);

/// A block's factor with its vertex indices in `u32` when they fit, else in `u64`; an isolated vertex stores only its inverse pivot.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: Real + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) enum Storage<T> {
    U32(Block<T, u32>),
    U64(Block<T, u64>),
    Isolated(Isolated<T>),
}

impl<T: Real> BlockFactor<T> {
    /// The block's vertices in the factored matrix's numbering, in local order.
    pub fn vertices(&self) -> Vertices<'_> {
        match &self.0 {
            Storage::U32(block) => Vertices::of(block.vertices(), Indices::U32),
            Storage::U64(block) => Vertices::of(block.vertices(), Indices::U64),
            Storage::Isolated(block) => {
                Vertices(Indices::U64(std::slice::from_ref(&block.vertex).iter()))
            }
        }
    }

    /// Whether the block was grounded or floated on a scaling that did not settle, so a singular block may have been grounded.
    pub fn grounding_unproven(&self) -> bool {
        on!(self, block => block.grounding_unproven, _ => false)
    }

    /// Whether the block is folded and solved as singular, `A_c⁺b` or its complement's `S_c⁺b`.
    pub fn floats(&self) -> bool {
        on!(self, block => block.cholesky.singular(), _ => false)
    }

    /// Work a solve may split across its [`Threads`], `0` when nothing does: an outer scheduler weighs it against its own parallelism.
    pub fn splittable_work(&self) -> usize {
        on!(self, block => (block.elimination.as_deref()).map_or(0, BlockElimination::splittable_work), _ => 0)
    }

    /// What [`BlockFactor::solve_local`] takes from its stack, fixed at factorization.
    pub fn scratch(&self) -> StackReq {
        slots::<T>(on!(self, block => block.cholesky.scratch_slots(), _ => 0))
    }

    /// Solves in local order: `x` holds `b` on entry and the solution on return; `stack` holds [`scratch`](Self::scratch).
    pub fn solve_local(
        &self,
        x: &mut [T],
        stack: &mut MemStack,
        threads: Threads,
    ) -> Result<(), ApplyError> {
        check_buffers(x.len(), self.vertices().len(), stack, self.scratch())?;
        on!(self, block => block.solve_unchecked(x, stack, threads), i => i.solve_local(x));
        Ok(())
    }
}

/// A [`BlockFactor`]'s vertices in the factored matrix's numbering; named, so a caller may move the block after its last use.
#[derive(Clone, Debug)]
pub struct Vertices<'a>(Indices<'a>);

#[derive(Clone, Debug)]
enum Indices<'a> {
    Range(std::ops::Range<usize>),
    U32(std::slice::Iter<'a, u32>),
    U64(std::slice::Iter<'a, u64>),
}

impl<'a> Vertices<'a> {
    fn of<I>(
        selection: &'a Selection<I>,
        listed: fn(std::slice::Iter<'a, I>) -> Indices<'a>,
    ) -> Self {
        match selection {
            Selection::Range(range) => Self(Indices::Range(range.clone())),
            Selection::Listed(vertices) => Self(listed(vertices.iter())),
        }
    }
}

impl Iterator for Vertices<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match &mut self.0 {
            Indices::Range(vertices) => vertices.next(),
            Indices::U32(vertices) => vertices.next().map(|&v| v.index()),
            Indices::U64(vertices) => vertices.next().map(|&v| v.index()),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.0 {
            Indices::Range(vertices) => vertices.size_hint(),
            Indices::U32(vertices) => vertices.size_hint(),
            Indices::U64(vertices) => vertices.size_hint(),
        }
    }
}

impl ExactSizeIterator for Vertices<'_> {}

/// One irreducible block's factor, its vertices stored in `I`.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self"),
    serde(bound(
        serialize = "T: serde::Serialize, I: Index + serde::Serialize",
        deserialize = "T: Real + serde::Deserialize<'de>, I: Index + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) struct Block<T, I> {
    restriction: Box<Restriction<T, I>>,
    elimination: Option<Box<BlockElimination<T, I>>>,
    cholesky: Cholesky<T, I>,
    grounding_unproven: bool,
}

impl<T: Real, I: Index> Block<T, I> {
    fn vertices(&self) -> &Selection<I> {
        self.restriction.vertices()
    }

    /// Every slot but a floating block's gauge, eliminated ones and the Cholesky's alike.
    fn n_steps(&self) -> usize {
        let eliminated = (self.elimination.as_deref()).map_or(0, BlockElimination::n_eliminated);
        eliminated + self.cholesky.n_steps()
    }

    /// Slots [`solve_global`](Self::solve_global) takes: room to gather unless the vertices are consecutive, plus the Cholesky's.
    fn global_scratch_slots(&self) -> usize {
        self.vertices().gather_slots() + self.cholesky.scratch_slots()
    }

    /// Solves the block's restriction of `x` in place, through a stack of [`global_scratch_slots`](Self::global_scratch_slots) slots.
    fn solve_global<V: Batch<T>>(&self, x: &mut [V], threads: Threads, stack: &mut MemStack) {
        self.vertices().on_local(
            x,
            stack,
            // on_local's two arms would otherwise each call it: ~19 instructions per block.
            #[inline(always)]
            |local: &mut [V], rest: &mut MemStack| self.solve_unchecked(local, rest, threads),
        );
    }

    // Two call sites through the closure plus solve_local's: by cost alone it stays a call.
    #[inline(always)]
    fn solve_unchecked<V: Batch<T>>(&self, x: &mut [V], stack: &mut MemStack, threads: Threads) {
        let singular = self.cholesky.singular();
        self.restriction.restrict(x, singular);
        match &self.elimination {
            None => self.cholesky.solve(x, stack),
            Some(elimination) => {
                elimination.solve(x, threads, |kept| self.cholesky.solve(kept, stack));
                if singular {
                    self.restriction.regauge(x);
                }
            }
        }
        self.restriction.prolongate(x, singular);
    }
}

/// A vertex no entry couples: its whole factor is `1/σ`, so it holds no heap part.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(
        bound(deserialize = "T: Real + serde::Deserialize<'de>"),
        deny_unknown_fields
    )
)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Isolated<T> {
    vertex: u64,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "inverse_of_a_pivot"))]
    inverse_pivot: T,
}

impl<T: Real> Isolated<T> {
    fn new<I: Index>(vertex: I, pivot: T) -> Self {
        Self {
            vertex: vertex.index() as u64,
            inverse_pivot: T::one() / pivot,
        }
    }

    fn solve_global<V: Batch<T>>(&self, x: &mut [V]) {
        let v = &mut x[self.vertex as usize];
        *v = v.mul(self.inverse_pivot);
    }

    fn solve_local<V: Batch<T>>(&self, x: &mut [V]) {
        x[0] = x[0].mul(self.inverse_pivot);
    }
}

#[cfg(feature = "serde")]
fn inverse_of_a_pivot<'de, D: serde::Deserializer<'de>, T: Real + serde::Deserialize<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    let inverse = T::deserialize(deserializer)?;
    inverse_pivot_in_range(inverse)
        .then_some(inverse)
        .ok_or_else(|| serde::de::Error::custom("corrupt Isolated"))
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

/// `(FORMAT_VERSION, (n, dead, blocks, clamps))`: every vertex of the factored matrix is in the bytes, so decoding allocates nothing they do not hold.
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for ApproxInverse<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Bodies<'a, T>(&'a [BlockFactor<T>]);
        impl<T: serde::Serialize> serde::Serialize for Bodies<'_, T> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.collect_seq(self.0.iter().map(|block| &block.0))
            }
        }
        let body = (self.n, &self.dead, Bodies(&self.blocks), &self.clamps);
        wire::serialize(serializer, body)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for ApproxInverse<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (n, dead, blocks, clamps) = wire::deserialize(deserializer)?;
        let blocks = Vec::into_iter(blocks).map(BlockFactor).collect();
        let value = Self::new(blocks, dead, clamps);
        let corrupt = || serde::de::Error::custom("corrupt ApproxInverse");
        value.checked(n).ok_or_else(corrupt)
    }
}

#[cfg(feature = "serde")]
impl<T: Real> ApproxInverse<T> {
    /// The blocks' vertices and the dead ones cover `0..n` once each.
    fn checked(self, n: usize) -> Option<Self> {
        let mut vertices = self.dead.to_vec();
        vertices.extend(self.blocks.iter().flat_map(BlockFactor::vertices));
        if vertices.len() != n {
            return None;
        }
        vertices.sort_unstable();
        (vertices.iter().enumerate())
            .all(|(i, &v)| i == v)
            .then_some(self)
    }
}

/// `(FORMAT_VERSION, storage)`: the version leads, so no future format has to keep the storage tag first.
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for BlockFactor<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        wire::serialize(serializer, &self.0)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for BlockFactor<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        wire::deserialize(deserializer).map(Self)
    }
}

#[cfg(feature = "serde")]
checked_serde!(Block);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Block<T, I> {
    fn checked(self) -> Option<Self> {
        let n = self.vertices().len();
        let kept = match &self.elimination {
            None => n,
            Some(elimination) => n
                .checked_sub(elimination.n_eliminated())
                .filter(|&kept| kept == elimination.n_kept())?,
        };
        (kept > 0 && self.cholesky.solves(kept)).then_some(self)
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1};
    use crate::sgdd::Reduced;
    use crate::ScalingPolicy;

    fn only_block(summands: Reduced<f64, u32>, config: &Config) -> Block<f64, u32> {
        let mut blocks = ApproxInverse::build(summands, config, Storage::U32).into_blocks();
        assert_eq!(blocks.len(), 1);
        match blocks.pop().expect("one block").0 {
            Storage::U32(block) => block,
            Storage::U64(_) | Storage::Isolated(_) => {
                unreachable!("a coupled u32 block stores u32")
            }
        }
    }

    fn csr_block(row_ptrs: &[u32], columns: &[u32], values: &[f64]) -> Block<f64, u32> {
        let n = (row_ptrs.len() - 1) as u32;
        let csr = CsrRef::new(row_ptrs, columns, values, n).expect("valid csr");
        let summands = Reduced::reduce(csr.into(), ScalingPolicy::default()).expect("reduces");
        only_block(summands, &Config::default())
    }

    fn grounded_block() -> Block<f64, u32> {
        let values = [2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0];
        csr_block(&[0, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2], &values)
    }

    /// A 2-vertex floating block's restriction over a 3-vertex grounded block's Cholesky.
    fn mismatched_cholesky() -> Block<f64, u32> {
        let floating = csr_block(&[0, 2, 4], &[0, 1, 0, 1], &[1.0, -1.0, -1.0, 1.0]);
        Block {
            cholesky: grounded_block().cholesky,
            ..floating
        }
    }

    /// A bipartite block, so it carries a block elimination.
    fn bipartite_block() -> Block<f64, u32> {
        let gram = BipartiteFactorWidth2::new(
            FactorWidth1::new(2, &[0u32, 0, 1, 1], None).expect("valid rows"),
            FactorWidth1::new(2, &[0u32, 1, 0, 1], None).expect("valid columns"),
        )
        .expect("valid gram");
        let summands = Reduced::reduce(gram.into(), ScalingPolicy::default()).expect("reduces");
        let block = only_block(summands, &Config::default());
        assert!(block.elimination.is_some());
        block
    }

    /// A bipartite block stripped of the elimination its Cholesky follows.
    fn without_elimination() -> Block<f64, u32> {
        Block {
            elimination: None,
            ..bipartite_block()
        }
    }

    /// An elimination of 2 keeping 2, between a 5-vertex restriction and a 3-vertex Cholesky.
    fn elimination_keeping_another_count() -> Block<f64, u32> {
        let elimination = bipartite_block().elimination;
        let counts = elimination.as_ref().map(|e| (e.n_eliminated(), e.n_kept()));
        assert_eq!(counts, Some((2, 2)));
        let path = csr_block(
            &[0, 2, 5, 8, 11, 13],
            &[0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4],
            &[
                3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0,
            ],
        );
        Block {
            restriction: path.restriction,
            elimination,
            cholesky: grounded_block().cholesky,
            grounding_unproven: false,
        }
    }

    #[rstest]
    #[case::mismatched_cholesky(mismatched_cholesky())]
    #[case::without_elimination(without_elimination())]
    #[case::elimination_keeping_another_count(elimination_keeping_another_count())]
    fn a_block_whose_parts_disagree_is_rejected_on_decode(#[case] spliced: Block<f64, u32>) {
        let bytes = postcard::to_stdvec(&BlockFactor(Storage::U32(spliced))).expect("serialize");
        assert_eq!(
            postcard::from_bytes::<BlockFactor>(&bytes).expect_err("parts that disagree"),
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
        let x = factor.solve(b).expect("b has length n");
        x.iter().map(|x| x.to_bits()).collect()
    }

    fn assert_storage_agrees(input: Input<'_, f64, u32>) {
        let policy = ScalingPolicy::default();
        let n = input.n();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let configs = [
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
        ];
        for config in configs {
            let narrow = ApproxInverse::build(
                Reduced::reduce(input, policy).expect("reduces"),
                &config,
                Storage::U32,
            );
            let wide = ApproxInverse::build(
                Reduced::reduce(input, policy).expect("reduces"),
                &config,
                Storage::U64,
            );
            assert!(matches!(wide.blocks[0].0, Storage::U64(_)));
            assert_eq!(bits(&narrow, &b), bits(&wide, &b), "{config:?}");
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

    #[test]
    fn u64_storage_solves_bit_identically_to_u32() {
        // Floating Laplacian, signs no ±1 folds (a cover), and a block scaled to dominance.
        for (sign, diagonal) in [(-1.0, 1.0), (1.0, 1.0), (-1.0, 0.8)] {
            let (row_ptrs, columns, values) = grid(7, 6, sign, diagonal);
            let csr = CsrRef::new(&row_ptrs, &columns, &values, 42).expect("valid CSR");
            assert_storage_agrees(csr.into());
        }
        let rows: Vec<u32> = (0..60).map(|k| k % 12).collect();
        let cols: Vec<u32> = (0..60).map(|k| (k * 7 + k / 12) % 9).collect();
        let scales: Vec<f64> = (0..60).map(|k| 1.0 + (k % 5) as f64).collect();
        let gram = BipartiteFactorWidth2::scaled(
            FactorWidth1::new(12, &rows, None).expect("valid rows"),
            FactorWidth1::new(9, &cols, None).expect("valid columns"),
            &scales,
        )
        .expect("valid gram");
        assert_storage_agrees(gram.into());
    }
}
