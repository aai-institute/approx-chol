//! Each irreducible block as an optional block elimination `L` and a Cholesky of what it leaves: `A_c⁺ ≈ P F L⁻ᵀ diag(D_e⁻¹, S⁺) L⁻¹ F P`, `P` projecting out `F·1` when the block floats, else `I`.

mod cholesky;
mod clique_tree;
mod config;
mod schur;
mod sddm;

use std::borrow::Cow;
use std::sync::Arc;

use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;
use rayon::prelude::*;

pub use config::{CliqueRule, Config, InnerParallelism, Multiplicity, SplitCount};

use crate::sgdd::{Balance, IrreducibleBlock, Restriction, Sgdd};
use crate::storage::OffDiagonal;
use crate::{Index, Real};
use cholesky::Cholesky;
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

/// Block-diagonal approximate inverse of `A`: one [`BlockFactor`] per [`IrreducibleBlock`].
#[derive(Clone, Debug)]
pub struct Factor<T = f64, I = u32> {
    n: usize,
    blocks: Vec<BlockFactor<T, I>>,
}

/// Factors every irreducible block of `sgdd` in parallel; `sgdd` survives for another `config`.
pub fn factorize<T: Real, I: Index>(sgdd: &Sgdd<T, I>, config: &Config) -> Factor<T, I> {
    let blocks = (sgdd.blocks().par_iter())
        .map(|block| BlockFactor::build(block, config))
        .collect();
    Factor {
        n: sgdd.n(),
        blocks,
    }
}

impl<T: Real, I: Index> Factor<T, I> {
    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The blocks' factors, owned, for a caller that keeps one solver per block; dead vertices need none.
    pub fn into_blocks(self) -> Vec<BlockFactor<T, I>> {
        self.blocks
    }

    /// `A⁺b` on floating blocks; vertices in no block solve to `0`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        check_buffers(b, self.n, &[], 0)?;
        let scratch_len = (self.blocks.iter())
            .map(|block| block.vertices().len() + block.scratch_len())
            .max();
        let mut scratch = vec![T::zero(); scratch_len.unwrap_or(0)];
        let mut x = vec![T::zero(); self.n];
        for block in &self.blocks {
            let restriction = &block.restriction;
            let (local, rest) = scratch.split_at_mut(block.vertices().len());
            restriction.gather(b, local);
            block.solve_unchecked(local, rest, InnerParallelism::Allow);
            restriction.scatter(local, &mut x);
        }
        Ok(x)
    }
}

/// One irreducible block's factor over its vertices of `A`.
#[derive(Clone, Debug)]
pub struct BlockFactor<T = f64, I = u32> {
    restriction: Arc<Restriction<T, I>>,
    elimination: Option<Box<BlockElimination<T, I>>>,
    cholesky: Cholesky<T, I>,
}

impl<T: Real, I: Index> BlockFactor<T, I> {
    fn build(block: &IrreducibleBlock<T, I>, config: &Config) -> Self {
        let streams = Streams {
            seed: config.seed,
            block: block.vertices()[0].index() as u64,
        };
        let (sheets, grounding) = match block.balance() {
            Balance::Balanced(grounding) => (Sheets::One, grounding),
            Balance::Frustrated(grounding) => (Sheets::Two, grounding),
        };
        let scales = |from: usize| block.restriction().scales(from);
        let (elimination, cholesky) = match block.off_diagonal() {
            OffDiagonal::General(adjacency) => {
                let sddm = match sheets {
                    Sheets::One => {
                        Sddm::new(Cow::Borrowed(adjacency), Cow::Borrowed(grounding), sheets)
                    }
                    Sheets::Two => Sddm::cover(adjacency, grounding),
                };
                let factored = Cholesky::factor(sddm, streams.cholesky(), config, scales(0));
                (None, factored)
            }
            OffDiagonal::Bipartite(cross) => {
                let factor =
                    |s| Cholesky::factor(s, streams.cholesky(), config, scales(cross.n_rows()));
                let rule = match config.dense(sheets.count() * cross.n_cols()) {
                    true => CliqueRule::Exact,
                    false => config.complement,
                };
                let (elimination, factored) = match rule {
                    CliqueRule::Exact => {
                        let elimination = BlockElimination::new(cross, grounding, scales(0));
                        let factored = factor(elimination.complement(sheets, grounding));
                        (elimination, factored)
                    }
                    // A kept `Cᵀ` is built once `S` is factored, so it never shares that peak.
                    CliqueRule::Sampled(copies) if BlockElimination::transposes(cross) => {
                        let s = schur::sampled(cross, copies, sheets, grounding, streams);
                        let factored = factor(s);
                        (BlockElimination::new(cross, grounding, scales(0)), factored)
                    }
                    // Otherwise the pivot pass reads `C` while it is still warm.
                    CliqueRule::Sampled(copies) => {
                        let elimination = BlockElimination::new(cross, grounding, scales(0));
                        let s = schur::sampled(cross, copies, sheets, grounding, streams);
                        (elimination, factor(s))
                    }
                };
                (Some(Box::new(elimination)), factored)
            }
        };
        Self {
            restriction: Arc::clone(block.restriction()),
            elimination,
            cholesky,
        }
    }

    /// The block's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[I] {
        self.restriction.vertices()
    }

    /// Work [`InnerParallelism::Allow`] may split per solve, `0` when nothing does: an outer scheduler weighs it against its own parallelism.
    pub fn splittable_work(&self) -> usize {
        (self.elimination.as_deref()).map_or(0, BlockElimination::splittable_work)
    }

    /// Workspace [`BlockFactor::solve_local`] needs, fixed at factorization.
    pub fn scratch_len(&self) -> usize {
        self.cholesky.scratch_len()
    }

    /// Solves in local order: `x` holds `b` on entry and the solution on return.
    pub fn solve_local(
        &self,
        x: &mut [T],
        scratch: &mut [T],
        inner: InnerParallelism,
    ) -> Result<(), SolveError> {
        check_buffers(x, self.vertices().len(), scratch, self.scratch_len())?;
        self.solve_unchecked(x, scratch, inner);
        Ok(())
    }

    fn solve_unchecked(&self, x: &mut [T], scratch: &mut [T], inner: InnerParallelism) {
        let singular = self.cholesky.singular();
        self.restriction.restrict(x, singular);
        match &self.elimination {
            None => self.cholesky.solve(x, scratch),
            Some(elimination) => {
                elimination.forward(x, inner);
                let (eliminated, kept) = x.split_at_mut(elimination.n_eliminated());
                self.cholesky.solve(kept, scratch);
                elimination.backward(eliminated, kept, inner);
                if singular {
                    elimination.regauge(x);
                }
            }
        }
        self.restriction.prolongate(x, singular);
    }
}

/// An index of the largest `|f|`, where pinning the gauge leaves its rounding unamplified by `1/f`.
fn largest<T: Real>(f: &[T]) -> usize {
    let last = f.len() - 1;
    (0..last).fold(last, |best, i| match f[i].abs() > f[best].abs() {
        true => i,
        false => best,
    })
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

#[cfg(feature = "serde")]
impl<T: serde::Serialize, I: serde::Serialize> serde::Serialize for BlockFactor<T, I> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let body = (&self.restriction, &self.elimination, &self.cholesky);
        crate::wire::serialize(serializer, body)
    }
}

// Not `remote = "Self"`: on a `pub` type it exports an unchecked, unversioned inherent `deserialize`.
#[cfg(feature = "serde")]
impl<'de, T, I> serde::Deserialize<'de> for BlockFactor<T, I>
where
    T: Real + serde::Deserialize<'de>,
    I: Index + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (restriction, elimination, cholesky) = crate::wire::deserialize(deserializer)?;
        let value = Self {
            restriction,
            elimination,
            cholesky,
        };
        let corrupt = || serde::de::Error::custom("corrupt BlockFactor");
        value.checked().ok_or_else(corrupt)
    }
}

#[cfg(feature = "serde")]
impl<T: Real, I: Index> BlockFactor<T, I> {
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
    use crate::input::{BipartiteGram, Csr, FactorWidth1, Weights};
    use crate::ScalingPolicy;

    fn only_block(sgdd: Sgdd, config: &Config) -> BlockFactor {
        let mut blocks = factorize(&sgdd, config).into_blocks();
        assert_eq!(blocks.len(), 1);
        blocks.pop().expect("one block")
    }

    fn csr_block(row_ptrs: &[u32], columns: &[u32], values: &[f64]) -> BlockFactor {
        let csr = Csr::new(row_ptrs.len() - 1, row_ptrs, columns, values).expect("valid csr");
        let sgdd = Sgdd::from_csr(csr, ScalingPolicy::default()).expect("reduces");
        only_block(sgdd, &Config::default())
    }

    fn grounded_block() -> BlockFactor {
        let values = [2.0, -1.0, -1.0, 3.0, -1.0, -1.0, 2.0];
        csr_block(&[0, 2, 5, 7], &[0, 1, 0, 1, 2, 1, 2], &values)
    }

    /// A 2-vertex floating block's restriction over a 3-vertex grounded block's Cholesky.
    fn mismatched_cholesky() -> BlockFactor {
        let floating = csr_block(&[0, 2, 4], &[0, 1, 0, 1], &[1.0, -1.0, -1.0, 1.0]);
        BlockFactor {
            cholesky: grounded_block().cholesky,
            ..floating
        }
    }

    /// A bipartite block, so it carries a block elimination.
    fn bipartite_block() -> BlockFactor {
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
    fn without_elimination() -> BlockFactor {
        BlockFactor {
            elimination: None,
            ..bipartite_block()
        }
    }

    /// An elimination of 2 keeping 2, between a 5-vertex restriction and a 3-vertex Cholesky.
    fn elimination_keeping_another_count() -> BlockFactor {
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
        BlockFactor {
            restriction: path.restriction,
            elimination,
            cholesky: grounded_block().cholesky,
        }
    }

    #[rstest]
    #[case::mismatched_cholesky(mismatched_cholesky())]
    #[case::without_elimination(without_elimination())]
    #[case::elimination_keeping_another_count(elimination_keeping_another_count())]
    fn a_block_whose_parts_disagree_is_rejected_on_decode(#[case] spliced: BlockFactor) {
        let bytes = postcard::to_stdvec(&spliced).expect("serialize");
        assert_eq!(
            postcard::from_bytes::<BlockFactor>(&bytes).expect_err("parts that disagree"),
            postcard::Error::SerdeDeCustom
        );
    }
}
