//! Routes each part to its eliminations, and applies `A_c⁺ ≈ P F · Round · Rem⁺ · Round' · F P`, `P` projecting out `F·1`.

mod clique_tree;
mod config;
mod remainder;
mod round;
mod sddm;

use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;

pub use config::{CliqueRule, Config, InnerParallelism, Multiplicity, OnDenseFailure, SplitCount};
pub use remainder::{Fallback, PivotDefect, UnusablePivot};

use crate::sgdd::{Balance, Frame, OffDiagonal, Part, Sgdd};
use crate::Real;
use clique_tree::Streams;
use remainder::SddmFactor;
use round::{Round, RoundBuilder};
use sddm::{Sddm, Sheets};

/// Why [`factorize`](crate::factorize) failed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorError {
    /// Under [`OnDenseFailure::Error`], a dense Cholesky reached an unusable pivot.
    UnusablePivot(UnusablePivot),
}

impl core::fmt::Display for FactorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnusablePivot(UnusablePivot { vertex, defect }) => {
                let defect = match defect {
                    PivotDefect::NonFinite => "non-finite",
                    PivotDefect::NonPositive => "non-positive",
                };
                write!(
                    f,
                    "dense Cholesky reached a {defect} pivot at vertex {vertex}"
                )
            }
        }
    }
}

impl std::error::Error for FactorError {}

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
    /// Scratch shorter than the operator's `scratch_len`.
    ScratchTooSmall {
        /// The scratch the operator needs.
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
            Self::ScratchTooSmall { needed, found } => {
                write!(f, "scratch has length {found}, needs {needed}")
            }
        }
    }
}

impl std::error::Error for SolveError {}

/// Block-diagonal approximate inverse of `A`: one [`PartFactor`] per [`Part`].
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    n: usize,
    parts: Vec<PartFactor<T>>,
    dead: Box<[u32]>,
}

impl<T: Real> Factor<T> {
    pub(crate) fn build(sgdd: &Sgdd<T>, config: &Config) -> Result<Self, FactorError> {
        // Collected whole first: the first failing part in order wins, not the first a worker saw.
        let built: Vec<_> = (sgdd.parts().par_iter())
            .map(|part| PartFactor::build(part, config))
            .collect();
        let parts = built.into_iter().collect::<Result<_, _>>()?;
        Ok(Self {
            n: sgdd.n(),
            parts,
            dead: sgdd.dead().into(),
        })
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Each part's factor, in [`Sgdd::parts`]' order, for a caller that solves them on its own.
    pub fn parts(&self) -> &[PartFactor<T>] {
        &self.parts
    }

    /// The parts, owned, for a caller that keeps one solver per part; dead vertices need none.
    pub fn into_parts(self) -> Vec<PartFactor<T>> {
        self.parts
    }

    /// Remainders routed dense and factored approximately instead.
    pub fn fallbacks(&self) -> impl Iterator<Item = Fallback> + '_ {
        self.parts.iter().filter_map(PartFactor::fallback)
    }

    /// Scratch [`solve_in_place`](Self::solve_in_place) needs: a part's gathered values and its own scratch.
    pub fn scratch_len(&self) -> usize {
        (self.parts.iter())
            .map(|part| part.vertices().len() + part.scratch_len())
            .max()
            .unwrap_or(0)
    }

    /// `A⁺b` on singular parts; vertices in no part solve to `0`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut x = b.to_vec();
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.solve_in_place(&mut x, &mut scratch, InnerParallelism::Allow)?;
        Ok(x)
    }

    /// `x` holds `b` on entry and the solution on return; parts solve in turn, each under `inner`.
    pub fn solve_in_place(
        &self,
        x: &mut [T],
        scratch: &mut [T],
        inner: InnerParallelism,
    ) -> Result<(), SolveError> {
        todo!()
    }
}

/// One part's factor over its vertices of `A`.
#[derive(Clone, Debug)]
pub struct PartFactor<T = f64> {
    frame: Arc<Frame<T>>,
    /// The part's own fact, not its remainder's: an underflowed complement surplus never makes it singular.
    floats: bool,
    round: Option<Round<T>>,
    remainder: SddmFactor<T>,
    fallback: Option<Fallback>,
}

impl<T: Real> PartFactor<T> {
    fn build(part: &Part<T>, config: &Config) -> Result<Self, FactorError> {
        let frame = part.frame();
        let vertices = frame.vertices();
        let streams = Streams::new(config.seed, vertices[0]);
        let (round, (remainder, fallback)) = match part.m() {
            (OffDiagonal::General(adjacency), balance) => {
                let sddm = match balance {
                    Balance::Balanced(grounding) => Sddm::new(
                        Cow::Borrowed(adjacency),
                        Cow::Borrowed(grounding),
                        Sheets::One,
                    ),
                    Balance::Frustrated(grounding) => Sddm::cover(adjacency, grounding),
                };
                let dense = |sddm: &_| SddmFactor::dense(sddm, vertices);
                let sampled = |sddm| SddmFactor::sampled(sddm, config.remainder, streams);
                (None, route(sddm.n(), sddm, dense, sampled, config)?)
            }
            (OffDiagonal::Bipartite(cross), balance) => {
                let builder = RoundBuilder::new(cross, balance);
                let kept = &vertices[builder.n_eliminated()..];
                // A dense Cholesky spends nothing on sparsity, so only the exact complement goes dense.
                let dense = |builder: &&RoundBuilder<'_, T>| {
                    SddmFactor::dense(&builder.complement(CliqueRule::Exact, streams), kept)
                };
                let sampled = |builder: &RoundBuilder<'_, T>| {
                    let complement = builder.complement(config.complement, streams);
                    SddmFactor::sampled(complement, config.remainder, streams)
                };
                let dimension = builder.complement_dimension();
                let routed = route(dimension, &builder, dense, sampled, config)?;
                (Some(builder.finish()), routed)
            }
        };
        Ok(Self {
            frame: Arc::clone(frame),
            floats: part.floats(),
            round,
            remainder,
            fallback,
        })
    }

    /// The part's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        self.frame.vertices()
    }

    /// Why this part is less accurate than its config asked, if it is.
    pub fn fallback(&self) -> Option<Fallback> {
        self.fallback
    }

    /// Scratch [`solve_local`](Self::solve_local) needs.
    pub fn scratch_len(&self) -> usize {
        let eliminated = self.round.as_ref().map_or(0, Round::n_eliminated);
        self.remainder
            .scratch_len(self.vertices().len() - eliminated)
    }

    /// Solves in local order: `x` holds `b` on entry and the solution on return.
    pub fn solve_local(
        &self,
        x: &mut [T],
        scratch: &mut [T],
        inner: InnerParallelism,
    ) -> Result<(), SolveError> {
        let expected = self.vertices().len();
        if x.len() != expected {
            return Err(SolveError::LengthMismatch {
                expected,
                found: x.len(),
            });
        }
        let needed = self.scratch_len();
        if scratch.len() < needed {
            return Err(SolveError::ScratchTooSmall {
                needed,
                found: scratch.len(),
            });
        }
        let congruence = self.frame.congruence();
        if self.floats {
            congruence.project_out_null(x);
        }
        congruence.apply(x);
        match &self.round {
            None => self.remainder.solve(x, scratch),
            Some(round) => {
                round.forward(x, inner);
                let (eliminated, kept) = x.split_at_mut(round.n_eliminated());
                self.remainder.solve(kept, scratch);
                round.backward(eliminated, kept, inner);
            }
        }
        congruence.apply(x);
        if self.floats {
            congruence.project_out_null(x);
        }
        Ok(())
    }
}

/// Dense when the rule says so; a declined pivot falls to the approximate arm, never to a second dense one.
fn route<T: Real, M>(
    dimension: usize,
    remainder: M,
    dense: impl FnOnce(&M) -> Result<SddmFactor<T>, Fallback>,
    sampled: impl FnOnce(M) -> SddmFactor<T>,
    config: &Config,
) -> Result<(SddmFactor<T>, Option<Fallback>), FactorError> {
    if !config.routes_dense(dimension) {
        return Ok((sampled(remainder), None));
    }
    match dense(&remainder) {
        Ok(factor) => Ok((factor, None)),
        // Only a pivot can be fatal: a remainder that will not fit falls back whatever the policy.
        Err(Fallback::UnusablePivot(pivot)) if config.on_dense_failure == OnDenseFailure::Error => {
            Err(FactorError::UnusablePivot(pivot))
        }
        Err(declined) => Ok((sampled(remainder), Some(declined))),
    }
}
