//! Routes each part to its eliminations, and stores and applies `A_c⁺ ≈ F · Round · K⁺ · Round' · F`.

mod kernel;

use std::sync::Arc;

use rayon::prelude::*;

use crate::clique_tree::Streams;
use crate::config::{CliqueRule, Config, InnerParallelism, OnDenseFailure};
use crate::dense::{Fallback, UnusablePivot};
use crate::round::Round;
use crate::sgdd::{Elimination, Frame, Part, Sgdd};
use crate::Real;
use kernel::SddmFactor;

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
            Self::UnusablePivot(pivot) => write!(
                f,
                "dense Cholesky reached a {:?} pivot at vertex {}",
                pivot.defect, pivot.vertex
            ),
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

/// Block-diagonal approximate inverse of `A`: one [`ComponentFactor`] per part.
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    n: usize,
    components: Vec<ComponentFactor<T>>,
}

impl<T: Real> Factor<T> {
    pub(crate) fn build(sgdd: &Sgdd<T>, config: &Config) -> Result<Self, FactorError> {
        let components = (sgdd.parts().par_iter())
            .map(|part| ComponentFactor::build(part, config))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            n: sgdd.n(),
            components,
        })
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Each part's factor, for a caller that solves them on its own, as Schwarz does.
    pub fn components(&self) -> &[ComponentFactor<T>] {
        &self.components
    }

    /// Remainders routed dense and factored approximately instead.
    pub fn fallbacks(&self) -> impl Iterator<Item = Fallback> + '_ {
        self.components.iter().filter_map(ComponentFactor::fallback)
    }

    /// Scratch [`solve_in_place`](Self::solve_in_place) needs.
    pub fn scratch_len(&self) -> usize {
        todo!()
    }

    /// `A⁺b` on singular parts; vertices in no part solve to `0`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut x = b.to_vec();
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.solve_in_place(&mut x, &mut scratch)?;
        Ok(x)
    }

    /// `x` holds `b` on entry and the solution on return; scratch is overwritten, never read.
    pub fn solve_in_place(&self, x: &mut [T], scratch: &mut [T]) -> Result<(), SolveError> {
        todo!()
    }
}

/// One part's factor over its vertices of `A`.
#[derive(Clone, Debug)]
pub struct ComponentFactor<T = f64> {
    frame: Arc<Frame<T>>,
    round: Option<Round<T>>,
    kernel: SddmFactor<T>,
    fallback: Option<Fallback>,
}

impl<T: Real> ComponentFactor<T> {
    fn build(part: &Part<T>, config: &Config) -> Result<Self, FactorError> {
        let frame = part.frame();
        let vertices = frame.vertices();
        let streams = Streams::new(config.seed, vertices[0]);
        let (round, (kernel, fallback)) = match part.elimination() {
            Elimination::Whole(sddm) => {
                let dense = || SddmFactor::dense(&sddm, vertices);
                let sampled = || SddmFactor::sampled(&sddm, config.min_degree, streams);
                (None, route(sddm.rank(), dense, sampled, config)?)
            }
            Elimination::Round(builder) => {
                let kept = &vertices[builder.n_eliminated()..];
                // A dense Cholesky spends nothing on sparsity, so only the exact complement goes dense.
                let dense = || SddmFactor::dense(&builder.schur(CliqueRule::Exact, streams), kept);
                let sampled = || {
                    let complement = builder.schur(config.schur, streams);
                    SddmFactor::sampled(&complement, config.min_degree, streams)
                };
                let routed = route(builder.complement_rank(), dense, sampled, config)?;
                (Some(builder.finish()), routed)
            }
        };
        Ok(Self {
            frame: Arc::clone(frame),
            round,
            kernel,
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

    /// Scratch [`solve_local`](Self::solve_local) needs: the kernel's ground slot, a cover's sheets.
    pub fn scratch_len(&self) -> usize {
        todo!()
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
        let singular = self.kernel.singular();
        if singular {
            congruence.project_out_null(x);
        }
        congruence.apply(x);
        match &self.round {
            None => self.kernel.solve(x, scratch),
            Some(round) => {
                round.forward(x, inner);
                let (eliminated, kept) = x.split_at_mut(round.n_eliminated());
                self.kernel.solve(kept, scratch);
                round.backward(eliminated, kept, inner);
            }
        }
        congruence.apply(x);
        if singular {
            congruence.project_out_null(x);
        }
        Ok(())
    }
}

/// Dense when the rule says so; a declined pivot falls to the approximate arm, never to a second dense one.
fn route<T: Real>(
    rank: usize,
    dense: impl FnOnce() -> Result<SddmFactor<T>, Fallback>,
    sampled: impl FnOnce() -> SddmFactor<T>,
    config: &Config,
) -> Result<(SddmFactor<T>, Option<Fallback>), FactorError> {
    if !config.routes_dense(rank) {
        return Ok((sampled(), None));
    }
    match dense() {
        Ok(kernel) => Ok((kernel, None)),
        Err(declined) => {
            let fallback = accept(config.on_dense_failure, declined)?;
            Ok((sampled(), Some(fallback)))
        }
    }
}

/// Only a pivot can be fatal: a remainder that will not fit falls back whatever the policy.
fn accept(on_failure: OnDenseFailure, fallback: Fallback) -> Result<Fallback, FactorError> {
    match (on_failure, fallback) {
        (OnDenseFailure::Error, Fallback::UnusablePivot(pivot)) => {
            Err(FactorError::UnusablePivot(pivot))
        }
        _ => Ok(fallback),
    }
}
