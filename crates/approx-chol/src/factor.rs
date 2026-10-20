//! Each irreducible block as an optional block elimination `L` and a Cholesky of what it leaves: `A_c⁺ ≈ P F L⁻ᵀ diag(D_e⁻¹, S⁺) L⁻¹ F P`, `P` projecting out `F·1` when the block floats, else `I`.

mod cholesky;
mod clique_tree;
mod config;
mod schur;
mod sddm;

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;
use rayon::iter::Either;
use rayon::prelude::*;

use clique_tree::{Copies, Single};
use config::multiplicity;
pub use config::{Backend, Config, InnerParallelism, SchurMode};

use crate::sgdd::{Balance, IrreducibleBlock, Restriction, Sgdd};
use crate::storage::OffDiagonal;
use crate::ScalingReport;
use crate::{Index, Real};
use cholesky::{Cholesky, CliqueTreeElimination};
use schur::BlockElimination;
use sddm::{Sddm, Sheets};

/// Errors returned by fallible solves.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveError {
    /// A right-hand side of a length other than the operator's dimension.
    LengthMismatch {
        /// The operator's dimension.
        expected: usize,
        /// The length provided.
        found: usize,
    },
    /// A workspace shorter than the factor's `scratch_len`.
    ScratchTooShort {
        /// The factor's `scratch_len`.
        needed: usize,
        /// The length provided.
        found: usize,
    },
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "right-hand side has length {found}, expected {expected}")
            }
            Self::ScratchTooShort { needed, found } => {
                write!(f, "scratch has length {found}, needs at least {needed}")
            }
        }
    }
}

impl std::error::Error for SolveError {}

/// `x` must match the operator and `scratch` cover its workspace.
fn check_buffers<T>(x: &[T], n: usize, scratch: &[T], needed: usize) -> Result<(), SolveError> {
    match x.len() == n && scratch.len() >= needed {
        true => Ok(()),
        false => Err(buffer_error(x.len(), n, scratch.len(), needed)),
    }
}

// Built inline, the error cost within's solves 5–9%.
#[cold]
#[inline(never)]
fn buffer_error(found: usize, n: usize, scratch: usize, needed: usize) -> SolveError {
    match found == n {
        false => SolveError::LengthMismatch { expected: n, found },
        true => SolveError::ScratchTooShort {
            needed,
            found: scratch,
        },
    }
}

/// Block-diagonal approximate inverse of `A`: one [`BlockFactor`] per irreducible block.
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    n: usize,
    blocks: Vec<BlockFactor<T>>,
    /// Vertices in no block, which solve to `0`.
    dead: Box<[usize]>,
    scratch_len: usize,
}

impl<T: Real> Factor<T> {
    /// Factors every irreducible block of `sgdd` in parallel.
    pub(crate) fn build<I: Index>(
        sgdd: Sgdd<T, I>,
        config: &Config,
        store: fn(Block<T, I>) -> Storage<T>,
    ) -> Self {
        match multiplicity(config.split_merge) {
            None => Self::build_with(sgdd, config, store, Single),
            Some(split) => Self::build_with(sgdd, config, store, split),
        }
    }

    fn build_with<I: Index, C: Copies>(
        sgdd: Sgdd<T, I>,
        config: &Config,
        store: fn(Block<T, I>) -> Storage<T>,
        split: C,
    ) -> Self {
        let n = sgdd.n();
        let (coupled, lone) = sgdd.into_blocks();
        let mut blocks = Vec::with_capacity(coupled.len() + lone.len());
        let build = |reused: &mut _, block| BlockFactor(store(Block::build(block, config, reused)));
        let elimination = || CliqueTreeElimination::new(split);
        blocks.par_extend(coupled.into_par_iter().map_init(elimination, build));
        let isolated = |(v, pivot)| BlockFactor(Storage::Isolated(Isolated::new(v, pivot)));
        blocks.extend(lone.into_iter().map(isolated));
        let mut covered = vec![false; n];
        (blocks.iter().flat_map(BlockFactor::vertices)).for_each(|v| covered[v] = true);
        let dead = (0..n).filter(|&v| !covered[v]).collect();
        Self::new(n, blocks, dead)
    }

    fn new(n: usize, blocks: Vec<BlockFactor<T>>, dead: Box<[usize]>) -> Self {
        let scratch_len = (blocks.iter())
            .map(|block| on!(block, b => b.vertices().len() + b.cholesky.scratch_len(), _ => 0))
            .max()
            .unwrap_or(0);
        Self {
            n,
            blocks,
            dead,
            scratch_len,
        }
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Total elimination steps: every slot but one per block, whichever arm factored it.
    pub fn n_steps(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| on!(block, b => b.n_steps(), _ => 1))
            .sum()
    }

    /// The blocks' factors, one per irreducible block of `A`, coupled blocks before isolated vertices; dead vertices have none.
    pub fn blocks(&self) -> &[BlockFactor<T>] {
        &self.blocks
    }

    /// The blocks' factors, owned, for a caller that keeps one solver per block; dead vertices need none.
    pub fn into_blocks(self) -> Vec<BlockFactor<T>> {
        self.blocks
    }

    /// Workspace [`solve_in_place`](Self::solve_in_place) needs: the most any one block's vertices plus its own scratch take.
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// `A⁺b` on floating blocks; vertices in no block solve to `0`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut x = b.to_vec();
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.solve_in_place(&mut x, &mut scratch)?;
        Ok(x)
    }

    /// As [`solve`](Self::solve), allocating nothing: `x` holds `b` on entry and the solution on return.
    pub fn solve_in_place(&self, x: &mut [T], scratch: &mut [T]) -> Result<(), SolveError> {
        check_buffers(x, self.n, scratch, self.scratch_len())?;
        for block in &self.blocks {
            on!(block, b => b.solve_global(x, scratch), i => i.solve_global(x));
        }
        self.dead.iter().for_each(|&v| x[v] = T::zero());
        Ok(())
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

/// One irreducible block's factor over its vertices of `A`.
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
    /// The block's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> Vertices<'_> {
        match &self.0 {
            Storage::U32(block) => Vertices(Either::Left(block.vertices().iter())),
            Storage::U64(block) => Vertices(Either::Right(block.vertices().iter())),
            Storage::Isolated(block) => {
                Vertices(Either::Right(std::slice::from_ref(&block.vertex).iter()))
            }
        }
    }

    /// Present iff a deficit past rounding was clamped, the block's scaling fell short beyond the policy's tolerance, or did not settle.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        on!(self, block => block.report.as_deref().copied(), _ => None)
    }

    /// Whether the block is folded and solved as singular, `A_c⁺b`.
    pub fn floats(&self) -> bool {
        on!(self, block => block.cholesky.singular(), _ => false)
    }

    /// Work [`InnerParallelism::Allow`] may split per solve, `0` when nothing does: an outer scheduler weighs it against its own parallelism.
    pub fn splittable_work(&self) -> usize {
        on!(self, block => (block.elimination.as_deref()).map_or(0, BlockElimination::splittable_work), _ => 0)
    }

    /// Workspace [`BlockFactor::solve_local`] needs, fixed at factorization.
    pub fn scratch_len(&self) -> usize {
        on!(self, block => block.cholesky.scratch_len(), _ => 0)
    }

    /// Solves in local order: `x` holds `b` on entry and the solution on return.
    pub fn solve_local(
        &self,
        x: &mut [T],
        scratch: &mut [T],
        inner: InnerParallelism,
    ) -> Result<(), SolveError> {
        check_buffers(x, self.vertices().len(), scratch, self.scratch_len())?;
        on!(self, block => block.solve_unchecked(x, scratch, inner), i => i.solve_local(x));
        Ok(())
    }
}

/// A [`BlockFactor`]'s vertices in `A`'s numbering; named, so a caller may move the block after its last use.
#[derive(Clone, Debug)]
pub struct Vertices<'a>(Either<std::slice::Iter<'a, u32>, std::slice::Iter<'a, u64>>);

impl Iterator for Vertices<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        match &mut self.0 {
            Either::Left(vertices) => vertices.next().map(|&v| v.index()),
            Either::Right(vertices) => vertices.next().map(|&v| v.index()),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.0 {
            Either::Left(vertices) => vertices.size_hint(),
            Either::Right(vertices) => vertices.size_hint(),
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
        deserialize = "T: Real + serde::Deserialize<'de>, I: Index + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) struct Block<T, I> {
    restriction: Box<Restriction<T, I>>,
    elimination: Option<Box<BlockElimination<T, I>>>,
    cholesky: Cholesky<T, I>,
    report: Option<Box<ScalingReport>>,
}

impl<T: Real, I: Index> Block<T, I> {
    fn build<C: Copies>(
        block: IrreducibleBlock<T, I>,
        config: &Config,
        reused: &mut CliqueTreeElimination<T, I, C>,
    ) -> Self {
        let (restriction, off_diagonal, balance, report) = block.into_parts();
        let streams = Streams {
            seed: config.seed,
            block: restriction.vertices()[0].index() as u64,
        };
        let (sheets, grounding) = match balance {
            Balance::Balanced(grounding) => (Sheets::One, grounding),
            Balance::Frustrated(grounding) => (Sheets::Two, grounding),
        };
        let factors = |from: usize| restriction.scaled_factors(from);
        let (elimination, cholesky) = match off_diagonal {
            OffDiagonal::General(adjacency) => {
                let sddm = match sheets {
                    Sheets::One => Sddm::new(adjacency, grounding, sheets),
                    Sheets::Two => Sddm::cover(adjacency, grounding),
                };
                let draws = streams.cholesky();
                let factored = Cholesky::factor(sddm, draws, config, factors(0), reused);
                (None, factored)
            }
            OffDiagonal::Bipartite(cross) => {
                let kept = factors(cross.n_rows());
                let dense = config.dense(sheets.count() * cross.n_cols());
                let inv_pivot = BlockElimination::inverse_pivots(&cross, &grounding);
                let s = match (dense, config.complement) {
                    (true, _) | (false, SchurMode::Exact) => {
                        schur::exact(&cross, &inv_pivot, sheets, grounding)
                    }
                    (false, sampled) => {
                        let split = multiplicity(match sampled {
                            SchurMode::Approximate { split_merge } => split_merge,
                            _ => config.split_merge,
                        });
                        schur::sampled(&cross, split, sheets, grounding, streams)
                    }
                };
                // A kept `Cᵀ` is built once `S` is factored, so it never shares that peak.
                let factored = Cholesky::factor(s, streams.cholesky(), config, kept, reused);
                let elimination = BlockElimination::with_inverse_pivots(cross, inv_pivot);
                (Some(Box::new(elimination)), factored)
            }
        };
        Self {
            restriction,
            elimination,
            cholesky,
            report,
        }
    }

    fn vertices(&self) -> &[I] {
        self.restriction.vertices()
    }

    /// Every slot but one, eliminated ones and the Cholesky's alike.
    fn n_steps(&self) -> usize {
        let eliminated = (self.elimination.as_deref()).map_or(0, BlockElimination::n_eliminated);
        eliminated + self.cholesky.n_steps()
    }

    /// Solves the block's restriction of `x` in place, through `scratch` of its vertices plus its Cholesky's.
    fn solve_global(&self, x: &mut [T], scratch: &mut [T]) {
        let (local, rest) = scratch.split_at_mut(self.vertices().len());
        self.restriction.gather(x, local);
        self.solve_unchecked(local, rest, InnerParallelism::Allow);
        self.restriction.scatter(local, x);
    }

    fn solve_unchecked(&self, x: &mut [T], scratch: &mut [T], inner: InnerParallelism) {
        let singular = self.cholesky.singular();
        self.restriction.restrict(x, singular);
        match &self.elimination {
            None => self.cholesky.solve(x, scratch),
            Some(elimination) => {
                elimination.solve(x, inner, |kept| self.cholesky.solve(kept, scratch));
                if singular {
                    self.restriction.regauge(x);
                }
            }
        }
        self.restriction.prolongate(x, singular);
    }
}

/// A vertex no entry couples: its whole factor is `1/σ`, so it holds no heap part.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Isolated<T> {
    vertex: u64,
    inverse_pivot: T,
}

impl<T: Real> Isolated<T> {
    fn new<I: Index>(vertex: I, pivot: T) -> Self {
        Self {
            vertex: vertex.index() as u64,
            inverse_pivot: T::one() / pivot,
        }
    }

    fn solve_global(&self, x: &mut [T]) {
        let v = &mut x[self.vertex as usize];
        *v = *v * self.inverse_pivot;
    }

    fn solve_local(&self, x: &mut [T]) {
        x[0] = x[0] * self.inverse_pivot;
    }
}

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for Isolated<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.vertex, &self.inverse_pivot).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for Isolated<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (vertex, inverse_pivot) = <(u64, T)>::deserialize(deserializer)?;
        match inverse_pivot.is_finite() && inverse_pivot > T::zero() {
            true => Ok(Self {
                vertex,
                inverse_pivot,
            }),
            false => Err(serde::de::Error::custom("corrupt Isolated")),
        }
    }
}

/// `Σ|t| ≤ 1` up to the terms' rounding: a substitution through such multipliers cannot overflow where its solution fits.
fn within_one<T: Real>(terms: impl Iterator<Item = T>) -> bool {
    let (sum, len) = terms.fold((0.0, 0), |(sum, len), t| {
        (sum + crate::widen(t.abs()), len + 1)
    });
    sum <= 1.0 + 4.0 * (len + 2) as f64 * crate::widen(T::epsilon())
}

/// One block's keys, so no draw depends on thread scheduling or on how other blocks routed.
#[derive(Clone, Copy, Debug)]
struct Streams {
    seed: u64,
    block: u64,
}

impl Streams {
    /// The approximate Cholesky's one sequence: pivot `t`'s sample is conditioned on every earlier one.
    fn cholesky(self) -> Xoshiro256PlusPlus {
        rng(self, 0)
    }

    /// Star `eliminated`'s own sequence, by its index in the block, never by task.
    fn star(self, eliminated: usize) -> Xoshiro256PlusPlus {
        rng(self, eliminated as u64 + 1)
    }
}

/// The whole key mixed once, the only place a key becomes draws; stage `0` is the Cholesky's, `e + 1` star `e`'s.
fn rng(streams: Streams, stage: u64) -> Xoshiro256PlusPlus {
    let key = [streams.block, stage]
        .into_iter()
        .fold(splitmix(streams.seed), |key, part| splitmix(key ^ part));
    Xoshiro256PlusPlus::seed_from_u64(key)
}

/// SplitMix64's finalizer: each part changes every bit of the key, so no two keys collide by addition.
fn splitmix(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// `(FORMAT_VERSION, (n, dead, blocks))`: every vertex of `A` is in the bytes, so decoding allocates nothing they do not hold.
#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for Factor<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        struct Bodies<'a, T>(&'a [BlockFactor<T>]);
        impl<T: serde::Serialize> serde::Serialize for Bodies<'_, T> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.collect_seq(self.0.iter().map(|block| &block.0))
            }
        }
        let body = (self.n, &self.dead, Bodies(&self.blocks));
        crate::wire::serialize(serializer, body)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for Factor<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (n, dead, blocks): (usize, Box<[usize]>, Vec<Storage<T>>) =
            crate::wire::deserialize(deserializer)?;
        let blocks = blocks.into_iter().map(BlockFactor).collect();
        let value = Self::new(n, blocks, dead);
        let corrupt = || serde::de::Error::custom("corrupt Factor");
        value.checked().ok_or_else(corrupt)
    }
}

#[cfg(feature = "serde")]
impl<T: Real> Factor<T> {
    /// The blocks' vertices and the dead ones cover `0..n` once each.
    fn checked(self) -> Option<Self> {
        let mut vertices = self.dead.to_vec();
        vertices.extend(self.blocks.iter().flat_map(BlockFactor::vertices));
        if vertices.len() != self.n {
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
        crate::wire::serialize(serializer, &self.0)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for BlockFactor<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        crate::wire::deserialize(deserializer).map(Self)
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
    use crate::input::{BipartiteGram, CsrRef, FactorWidth1, Weights};
    use crate::ScalingPolicy;

    fn only_block(sgdd: Sgdd, config: &Config) -> Block<f64, u32> {
        let mut blocks = Factor::build(sgdd, config, Storage::U32).into_blocks();
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
        let sgdd = Sgdd::reduce(csr.into(), ScalingPolicy::default()).expect("reduces");
        only_block(sgdd, &Config::default())
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
        let gram = BipartiteGram::new(
            FactorWidth1::new(2, &[0u32, 0, 1, 1], None).expect("valid rows"),
            FactorWidth1::new(2, &[0u32, 1, 0, 1], None).expect("valid columns"),
            Weights::unit(),
        )
        .expect("valid gram");
        let sgdd = Sgdd::from_bipartite_gram(gram, ScalingPolicy::default()).expect("reduces");
        let block = only_block(sgdd, &Config::default());
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
            report: None,
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
    use crate::input::{BipartiteGram, CsrRef, FactorWidth1, Input, Weights};
    use crate::{ScalingPolicy, SchurMode};

    fn bits(factor: &Factor, b: &[f64]) -> Vec<u64> {
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
                backend: Backend::Approximate,
                split_merge: Some(2),
                ..Config::default()
            },
            Config {
                backend: Backend::Approximate,
                complement: SchurMode::Exact,
                ..Config::default()
            },
        ];
        for config in configs {
            let narrow = Factor::build(
                Sgdd::reduce(input, policy).expect("reduces"),
                &config,
                Storage::U32,
            );
            let wide = Factor::build(
                Sgdd::reduce(input, policy).expect("reduces"),
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
        let weights: Vec<f64> = (0..60).map(|k| 1.0 + (k % 5) as f64).collect();
        let gram = BipartiteGram::new(
            FactorWidth1::new(12, &rows, None).expect("valid rows"),
            FactorWidth1::new(9, &cols, None).expect("valid columns"),
            Weights::of(&weights).expect("nonnegative"),
        )
        .expect("valid gram");
        assert_storage_agrees(gram.into());
    }
}
