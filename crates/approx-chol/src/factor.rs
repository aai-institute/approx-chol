//! Each irreducible block as an optional block elimination `L` and a Cholesky of what it leaves: `A_c⁺ ≈ P F L⁻ᵀ diag(D_e⁻¹, S⁺) L⁻¹ F P`, `P` projecting out `F·1` when the block floats, else `I`.

mod cholesky;
mod clique_tree;
mod config;
mod schur;
mod sddm;
#[cfg(feature = "serde")]
mod wire;

use rayon::prelude::*;

pub use config::{Config, InnerParallelism, SchurMode};
#[cfg(feature = "serde")]
pub use wire::FORMAT_VERSION;

use crate::sgdd::{
    Lift, Lone, Null, Reduced, ReducedComplement, ReducedComponent, Selection, Summands,
};
use crate::storage::{Index, Real};
use crate::ScalingReport;
pub(crate) use cholesky::fits_u32;
use cholesky::{Factorization, Workspace};

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

/// Block-diagonal approximate inverse of the factored matrix, `A` or its [complement](crate::factorize_complement) numbered by the kept side: one [`BlockFactor`] per irreducible block of `A`.
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    n: usize,
    blocks: Vec<BlockFactor<T>>,
    /// Vertices in no block, which solve to `0`.
    dead: Box<[usize]>,
    scratch_len: usize,
}

impl<T: Real> Factor<T> {
    /// Factors every coupled summand in parallel.
    pub(crate) fn build<I: Index, B: Summand<T, I>>(
        summands: Summands<B, I, T>,
        config: &Config,
    ) -> Self
    where
        Storage<T>: From<Block<T, I>>,
    {
        let Summands {
            blocks: coupled,
            lone,
            dead,
        } = summands;
        let mut blocks = Vec::with_capacity(coupled.len() + lone.len());
        let build = |reused: &mut _, block: B| BlockFactor(block.factor(config, reused).into());
        let workspace = || Workspace::new(config.split());
        blocks.par_extend(coupled.into_par_iter().map_init(workspace, build));
        let isolated =
            |Lone { vertex, pivot }| BlockFactor(Storage::Isolated(Isolated::new(vertex, pivot)));
        blocks.extend(lone.into_iter().map(isolated));
        Self::new(blocks, dead.into_iter().map(I::index).collect())
    }

    fn new(blocks: Vec<BlockFactor<T>>, dead: Box<[usize]>) -> Self {
        let n = blocks.iter().map(|b| b.vertices().len()).sum::<usize>() + dead.len();
        let scratch_len = (blocks.iter())
            .map(|block| on!(block, b => b.global_scratch_len(), _ => 0))
            .max()
            .unwrap_or(0);
        Self {
            n,
            blocks,
            dead,
            scratch_len,
        }
    }

    /// Dimension of the factored matrix.
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

    /// The blocks' factors, one per irreducible block of `A` the factored matrix meets, coupled blocks before isolated vertices; dead vertices have none.
    pub fn blocks(&self) -> &[BlockFactor<T>] {
        &self.blocks
    }

    /// The blocks' factors, owned, for a caller that keeps one solver per block; dead vertices need none.
    pub fn into_blocks(self) -> Vec<BlockFactor<T>> {
        self.blocks
    }

    /// Workspace [`solve_in_place`](Self::solve_in_place) needs: the most any one block takes, its own scratch plus its vertices unless they are consecutive.
    pub fn scratch_len(&self) -> usize {
        self.scratch_len
    }

    /// `A⁺b` on floating blocks; vertices in no block solve to `0`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut x = vec![T::zero(); self.n];
        self.solve_into(b, &mut x)?;
        Ok(x)
    }

    /// As [`solve`](Self::solve), into `x` of length [`n`](Self::n).
    pub fn solve_into(&self, b: &[T], x: &mut [T]) -> Result<(), SolveError> {
        if let Some(found) = [b.len(), x.len()].into_iter().find(|&len| len != self.n) {
            return Err(SolveError::LengthMismatch {
                expected: self.n,
                found,
            });
        }
        x.copy_from_slice(b);
        self.solve_in_place(x, &mut vec![T::zero(); self.scratch_len()])
    }

    /// As [`solve`](Self::solve), allocating nothing: `x` holds `b` on entry and the solution on return; `scratch`'s contents on entry are ignored.
    pub fn solve_in_place(&self, x: &mut [T], scratch: &mut [T]) -> Result<(), SolveError> {
        check_buffers(x, self.n, scratch, self.scratch_len())?;
        for block in &self.blocks {
            on!(block, b => b.solve_global(x, scratch), i => i.solve_global(x));
        }
        self.dead.iter().for_each(|&v| x[v] = T::zero());
        Ok(())
    }
}

/// A coupled summand of the factored matrix: a block of `A`, or a block's complement onto its kept side.
pub(crate) trait Summand<T, I>: Send {
    fn factor(self, config: &Config, reused: &mut Workspace<T, I>) -> Block<T, I>;
}

impl<T: Real, I: Index> Summand<T, I> for ReducedComponent<T, I> {
    fn factor(self, config: &Config, reused: &mut Workspace<T, I>) -> Block<T, I> {
        let Reduced {
            key,
            lift,
            matrix,
            balance,
            gauge,
            report,
        } = self;
        let factorization = Factorization::component(matrix, balance, gauge, key, config, reused);
        Block {
            lift,
            factorization,
            report,
        }
    }
}

impl<T: Real, I: Index> Summand<T, I> for ReducedComplement<T, I> {
    fn factor(self, config: &Config, reused: &mut Workspace<T, I>) -> Block<T, I> {
        let Reduced {
            key,
            lift,
            matrix,
            balance,
            gauge,
            report,
        } = self.into_reduced();
        let factorization = Factorization::complement(matrix, balance, gauge, key, config, reused);
        Block {
            lift,
            factorization,
            report,
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

impl<T> From<Block<T, u32>> for Storage<T> {
    fn from(block: Block<T, u32>) -> Self {
        Self::U32(block)
    }
}

impl<T> From<Block<T, u64>> for Storage<T> {
    fn from(block: Block<T, u64>) -> Self {
        Self::U64(block)
    }
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

    /// Present iff a deficit past rounding was clamped, the block's scaling fell short beyond the policy's tolerance, or did not settle.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        on!(self, block => block.report.as_deref().copied(), _ => None)
    }

    /// Whether the block is folded and solved as singular, `A_c⁺b` or its complement's `S_c⁺b`.
    pub fn floats(&self) -> bool {
        on!(self, block => block.lift.null() != Null::None, _ => false)
    }

    /// Work [`InnerParallelism::Allow`] may split per solve, `0` when nothing does: an outer scheduler weighs it against its own parallelism.
    pub fn splittable_work(&self) -> usize {
        on!(self, block => block.factorization.splittable_work(), _ => 0)
    }

    /// Workspace [`BlockFactor::solve_local`] needs, fixed at factorization.
    pub fn scratch_len(&self) -> usize {
        on!(self, block => block.factorization.scratch_len(), _ => 0)
    }

    /// Solves in local order: `x` holds `b` on entry and the solution on return; `scratch`'s contents on entry are ignored.
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
    lift: Box<Lift<T, I>>,
    factorization: Factorization<T, I>,
    report: Option<Box<ScalingReport>>,
}

impl<T: Real, I: Index> Block<T, I> {
    fn vertices(&self) -> &Selection<I> {
        self.lift.vertices()
    }

    /// Every slot but one, eliminated ones and the Cholesky's alike.
    fn n_steps(&self) -> usize {
        self.factorization.n_steps()
    }

    /// Workspace [`solve_global`](Self::solve_global) needs: room to gather unless the vertices are consecutive, plus the Cholesky's.
    fn global_scratch_len(&self) -> usize {
        self.vertices().gather_len() + self.factorization.scratch_len()
    }

    /// Solves the block's part of `x` in place, through `scratch` of [`global_scratch_len`](Self::global_scratch_len).
    fn solve_global(&self, x: &mut [T], scratch: &mut [T]) {
        self.vertices().on_local(
            x,
            scratch,
            // on_local's two arms would otherwise each call it: ~19 instructions per block.
            #[inline(always)]
            |local: &mut [T], rest: &mut [T]| {
                self.solve_unchecked(local, rest, InnerParallelism::Allow)
            },
        );
    }

    // Two call sites through the closure plus solve_local's: by cost alone it stays a call.
    #[inline(always)]
    fn solve_unchecked(&self, x: &mut [T], scratch: &mut [T], inner: InnerParallelism) {
        self.lift.restrict(x);
        self.factorization.solve(x, scratch, inner);
        self.lift.regauge(x);
        self.lift.prolongate(x);
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

    fn solve_global(&self, x: &mut [T]) {
        let v = &mut x[self.vertex as usize];
        *v = *v * self.inverse_pivot;
    }

    fn solve_local(&self, x: &mut [T]) {
        x[0] = x[0] * self.inverse_pivot;
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

/// `(FORMAT_VERSION, (n, dead, blocks))`: every vertex of the factored matrix is in the bytes, so decoding allocates nothing they do not hold.
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
        wire::serialize(serializer, body)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for Factor<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (n, dead, blocks): (usize, Box<[usize]>, Vec<Storage<T>>) =
            wire::deserialize(deserializer)?;
        let blocks = blocks.into_iter().map(BlockFactor).collect();
        let value = Self::new(blocks, dead);
        let corrupt = || serde::de::Error::custom("corrupt Factor");
        value.checked(n).ok_or_else(corrupt)
    }
}

#[cfg(feature = "serde")]
impl<T: Real> Factor<T> {
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
        let null = match self.factorization.singular() {
            (false, _) => Null::None,
            (true, false) => Null::Projected,
            (true, true) => Null::Regauged,
        };
        let n = self.vertices().len();
        let factorization = self.factorization.checked(n)?;
        (self.lift.null() == null).then_some(Self {
            factorization,
            ..self
        })
    }
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::input::{BipartiteFactorWidth2, CsrRef, FactorWidth1};
    use crate::sgdd::Reduction;
    use crate::ScalingPolicy;

    fn only_block(summands: Reduction<f64, u32>, config: &Config) -> Block<f64, u32> {
        let mut blocks = Factor::build(summands, config).into_blocks();
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
        let summands = Reduction::reduce(csr.into(), ScalingPolicy::default()).expect("reduces");
        only_block(summands, &Config::default())
    }

    fn grounded_block() -> Block<f64, u32> {
        let values = [2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0];
        csr_block(&[0, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2], &values)
    }

    /// A 2-vertex floating block's lift over a 3-vertex grounded block's Cholesky.
    fn mismatched_cholesky() -> Block<f64, u32> {
        let floating = csr_block(&[0, 2, 4], &[0, 1, 0, 1], &[1.0, -1.0, -1.0, 1.0]);
        Block {
            factorization: grounded_block().factorization,
            ..floating
        }
    }

    /// A floating 3-vertex path's lift, which projects, over a grounded 3-vertex block's Cholesky.
    fn floating_lift_over_a_grounded_cholesky() -> Block<f64, u32> {
        let values = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
        let floating = csr_block(&[0, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2], &values);
        assert_eq!(floating.lift.null(), Null::Projected);
        Block {
            factorization: grounded_block().factorization,
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
        let summands = Reduction::reduce(gram.into(), ScalingPolicy::default()).expect("reduces");
        let block = only_block(summands, &Config::default());
        assert!(block.factorization.elimination.is_some());
        block
    }

    /// A bipartite block stripped of the elimination its Cholesky follows.
    fn without_elimination() -> Block<f64, u32> {
        let block = bipartite_block();
        let factorization = Factorization {
            elimination: None,
            ..block.factorization
        };
        Block {
            factorization,
            ..block
        }
    }

    /// An elimination of 2 keeping 2, between a 5-vertex lift and a 3-vertex Cholesky.
    fn elimination_keeping_another_count() -> Block<f64, u32> {
        let elimination = bipartite_block().factorization.elimination;
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
            lift: path.lift,
            factorization: Factorization {
                elimination,
                ..grounded_block().factorization
            },
            report: None,
        }
    }

    #[rstest]
    #[case::mismatched_cholesky(mismatched_cholesky())]
    #[case::floating_lift_over_a_grounded_cholesky(floating_lift_over_a_grounded_cholesky())]
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
    use crate::sgdd::Reduction;
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
            let narrow: Reduction<f64, u32> = Reduction::reduce(input, policy).expect("reduces");
            let narrow = Factor::build(narrow, &config);
            let wide: Reduction<f64, u64> = Reduction::reduce(input, policy).expect("reduces");
            let wide = Factor::build(wide, &config);
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
