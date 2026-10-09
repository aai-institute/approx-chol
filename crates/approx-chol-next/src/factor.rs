//! Routes each part to its eliminations, and stores and applies `A_c⁺ ≈ F · Round · Kept⁺ · Round' · F`.

mod kernel;

use std::sync::Arc;

use rayon::prelude::*;

use crate::config::{CliqueRule, Config, ExactFailure, InnerParallelism};
use crate::dense::Fallback;
use crate::round::{Round, RoundBuilder};
use crate::sampling::Streams;
use crate::sgdd::{Body, Frame, Part, Sgdd};
use crate::{Real, UnusablePivot};
use kernel::Kept;

/// Why [`factorize`](crate::factorize) failed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactorError {
    /// Under [`ExactFailure::Error`], a dense Cholesky reached an unusable pivot.
    DenseFactorizationFailed(UnusablePivot),
}

impl core::fmt::Display for FactorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "factorization failed: {self:?}")
    }
}

impl std::error::Error for FactorError {}

/// Errors returned by fallible solves.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveError {
    /// A right-hand side of a length other than the operator's.
    LengthMismatch {
        /// Provided length.
        len: usize,
        /// The operator's dimension.
        dim: usize,
    },
    /// Scratch shorter than the operator's `scratch_len`.
    ScratchTooSmall {
        /// Provided scratch length.
        len: usize,
        /// Required scratch length.
        needed: usize,
    },
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cannot solve: {self:?}")
    }
}

impl std::error::Error for SolveError {}

/// Block-diagonal approximate inverse of `A`: one [`ComponentFactor`] per part.
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    n: usize,
    components: Vec<ComponentFactor<T>>,
    fallbacks: Vec<Fallback>,
}

impl<T: Real> Factor<T> {
    pub(crate) fn build(sgdd: &Sgdd<T>, config: &Config) -> Result<Self, FactorError> {
        let built = (sgdd.parts().par_iter())
            .map(|part| ComponentFactor::build(part, config))
            .collect::<Result<Vec<_>, _>>()?;
        let mut fallbacks = Vec::new();
        let components = (built.into_iter())
            .map(|(component, fallback)| {
                fallbacks.extend(fallback);
                component
            })
            .collect();
        Ok(Self {
            n: sgdd.n(),
            components,
            fallbacks,
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
    pub fn fallbacks(&self) -> &[Fallback] {
        &self.fallbacks
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
    kept: Kept<T>,
}

impl<T: Real> ComponentFactor<T> {
    fn build(part: &Part<T>, config: &Config) -> Result<(Self, Option<Fallback>), FactorError> {
        let frame = part.frame();
        let streams = Streams::new(config.seed, frame.vertices()[0]);
        let vertices = frame.vertices();
        let (round, (kept, fallback)) = match part.body() {
            Body::Laplacian(sddm) => {
                let dense = || Kept::dense(sddm, vertices);
                let sampled = || Kept::sampled(sddm, config.kernel, streams);
                (None, route(sddm.eliminated(), dense, sampled, config)?)
            }
            Body::Signed(signed) => {
                let cover = signed.cover();
                let dense = || Kept::dense(&cover, vertices);
                let sampled = || Kept::sampled(&cover, config.kernel, streams);
                (None, route(cover.eliminated(), dense, sampled, config)?)
            }
            Body::Bipartite(bipartite) => {
                let builder = RoundBuilder::new(bipartite);
                let kept = &vertices[bipartite.n_eliminated()..];
                // A dense Cholesky spends nothing on sparsity, so only the exact complement goes dense.
                let dense = || Kept::dense(&builder.schur(CliqueRule::Exact, streams), kept);
                let sampled = || {
                    Kept::sampled(
                        &builder.schur(config.round, streams),
                        config.kernel,
                        streams,
                    )
                };
                let routed = route(builder.complement_eliminated(), dense, sampled, config)?;
                (Some(builder.finish()), routed)
            }
        };
        let factor = Self {
            frame: Arc::clone(frame),
            round,
            kept,
        };
        Ok((factor, fallback))
    }

    /// The part's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        self.frame.vertices()
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
        let dim = self.vertices().len();
        if x.len() != dim {
            return Err(SolveError::LengthMismatch { len: x.len(), dim });
        }
        let needed = self.scratch_len();
        if scratch.len() < needed {
            return Err(SolveError::ScratchTooSmall {
                len: scratch.len(),
                needed,
            });
        }
        let congruence = self.frame.congruence();
        let singular = self.kept.singular();
        if singular {
            congruence.project_null(x);
        }
        congruence.apply(x);
        match &self.round {
            None => self.kept.solve(x, scratch),
            Some(round) => {
                round.forward(x, inner);
                let (eliminated, kept) = x.split_at_mut(round.n_eliminated());
                self.kept.solve(kept, scratch);
                round.backward(eliminated, kept, inner);
            }
        }
        congruence.apply(x);
        if singular {
            congruence.project_null(x);
        }
        Ok(())
    }
}

/// Dense when the rule says so; a declined pivot falls to the approximate arm, never to a second dense one.
fn route<T: Real>(
    eliminated: usize,
    dense: impl FnOnce() -> Result<Kept<T>, Fallback>,
    sampled: impl FnOnce() -> Kept<T>,
    config: &Config,
) -> Result<(Kept<T>, Option<Fallback>), FactorError> {
    if !config.routes_dense(eliminated) {
        return Ok((sampled(), None));
    }
    match dense() {
        Ok(kept) => Ok((kept, None)),
        Err(declined) => {
            let fallback = accept(config.on_dense_failure, declined)?;
            Ok((sampled(), Some(fallback)))
        }
    }
}

/// Only a pivot can be fatal: a remainder that will not fit falls back whatever the policy.
fn accept(on_failure: ExactFailure, fallback: Fallback) -> Result<Fallback, FactorError> {
    match (on_failure, fallback) {
        (ExactFailure::Error, Fallback::InvalidPivot(pivot)) => {
            Err(FactorError::DenseFactorizationFailed(pivot))
        }
        _ => Ok(fallback),
    }
}
