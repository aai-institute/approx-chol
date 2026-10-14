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

pub use cholesky::Fallback;
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
    dead: Box<[I]>,
}

/// Factors every irreducible block of `sgdd` in parallel; `sgdd` survives for another `config`.
pub fn factorize<T: Real, I: Index>(sgdd: &Sgdd<T, I>, config: &Config) -> Factor<T, I> {
    let blocks = (sgdd.blocks().par_iter())
        .map(|block| BlockFactor::build(block, config))
        .collect();
    Factor {
        n: sgdd.n(),
        blocks,
        dead: sgdd.dead().into(),
    }
}

impl<T: Real, I: Index> Factor<T, I> {
    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Each block's factor, in [`Sgdd::blocks`]' order, for a caller that solves them on its own.
    pub fn blocks(&self) -> &[BlockFactor<T, I>] {
        &self.blocks
    }

    /// The blocks' factors, owned, for a caller that keeps one solver per block; dead vertices need none.
    pub fn into_blocks(self) -> Vec<BlockFactor<T, I>> {
        self.blocks
    }

    /// Cholesky factors routed dense and factored approximately instead.
    pub fn fallbacks(&self) -> impl Iterator<Item = Fallback> + '_ {
        self.blocks.iter().filter_map(BlockFactor::fallback)
    }

    /// Workspace [`Factor::solve_in_place`] needs, fixed at factorization.
    pub fn scratch_len(&self) -> usize {
        (self.blocks.iter())
            .map(|block| block.vertices().len() + block.scratch_len())
            .max()
            .unwrap_or(0)
    }

    /// `A⁺b` on floating blocks; vertices in no block solve to `0`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut x = b.to_vec();
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.solve_in_place(&mut x, &mut scratch, InnerParallelism::Allow)?;
        Ok(x)
    }

    /// `x` holds `b` on entry and the solution on return; blocks solve in turn, each under `inner`.
    pub fn solve_in_place(
        &self,
        x: &mut [T],
        scratch: &mut [T],
        inner: InnerParallelism,
    ) -> Result<(), SolveError> {
        check_buffers(x, self.n, scratch, self.scratch_len())?;
        for block in &self.blocks {
            let restriction = &block.restriction;
            let (local, rest) = scratch.split_at_mut(block.vertices().len());
            restriction.gather(x, local);
            block.solve_unchecked(local, rest, inner);
            restriction.scatter(local, x);
        }
        for &i in &self.dead {
            x[i.index()] = T::zero();
        }
        Ok(())
    }
}

/// One irreducible block's factor over its vertices of `A`.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self"),
    serde(bound(
        deserialize = "T: crate::Real + serde::Deserialize<'de>, I: crate::Index + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct BlockFactor<T = f64, I = u32> {
    restriction: Arc<Restriction<T, I>>,
    elimination: Option<BlockElimination<T, I>>,
    cholesky: Cholesky<T, I>,
    fallback: Option<Fallback>,
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
        let anchor = |from: usize| block.floats().then(|| block.restriction().anchor(from));
        let (elimination, (cholesky, fallback)) = match block.off_diagonal() {
            OffDiagonal::General(adjacency) => {
                let sddm = match sheets {
                    Sheets::One => {
                        Sddm::new(Cow::Borrowed(adjacency), Cow::Borrowed(grounding), sheets)
                    }
                    Sheets::Two => Sddm::cover(adjacency, grounding),
                };
                let wide = block.restriction().spans_wide(0);
                let factored = Cholesky::factor(sddm, streams.cholesky(), config, anchor(0), wide);
                (None, factored)
            }
            OffDiagonal::Bipartite(cross) => {
                let elimination = BlockElimination::new(cross, grounding);
                let schur = elimination.complement(sheets, grounding, config.complement, streams);
                let from = elimination.n_eliminated();
                let wide = block.restriction().spans_wide(from);
                let factored =
                    Cholesky::factor(schur, streams.cholesky(), config, anchor(from), wide);
                (Some(elimination), factored)
            }
        };
        Self {
            restriction: Arc::clone(block.restriction()),
            elimination,
            cholesky,
            fallback,
        }
    }

    /// The block's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[I] {
        self.restriction.vertices()
    }

    /// Why this block is less accurate than its config asked, if it is.
    pub fn fallback(&self) -> Option<Fallback> {
        self.fallback
    }

    /// Work [`InnerParallelism::Allow`] may split per solve, `0` when nothing does: an outer scheduler weighs it against its own parallelism.
    pub fn splittable_work(&self) -> usize {
        (self.elimination.as_ref()).map_or(0, BlockElimination::splittable_work)
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
            }
        }
        self.restriction.prolongate(x, singular);
    }
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
checked_serde!(BlockFactor);

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
