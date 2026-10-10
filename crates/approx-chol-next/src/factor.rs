//! Each component as an optional block elimination `L` and a Cholesky of what it leaves: `A_c⁺ ≈ P F L⁻ᵀ diag(D_e⁻¹, S⁺) L⁻¹ F P`, `P` projecting out `F·1` when the component floats, else `I`.

mod cholesky;
mod clique_tree;
mod config;
mod schur;
mod sddm;

use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;

pub use cholesky::{Fallback, PivotDefect, UnusablePivot};
pub use config::{CliqueRule, Config, InnerParallelism, Multiplicity, OnDenseFailure, SplitCount};

use crate::sgdd::{Balance, Component, Frame, OffDiagonal, Sgdd};
use crate::Real;
use cholesky::{Cholesky, DenseCholesky};
use clique_tree::Streams;
use schur::{BlockElimination, SchurComplement};
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

/// Block-diagonal approximate inverse of `A`: one [`ComponentFactor`] per [`Component`].
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    n: usize,
    components: Vec<ComponentFactor<T>>,
    dead: Box<[u32]>,
}

impl<T: Real> Factor<T> {
    pub(crate) fn build(sgdd: &Sgdd<T>, config: &Config) -> Result<Self, FactorError> {
        // Collected whole first: the first failing component in order wins, not the first a worker saw.
        let built: Vec<_> = (sgdd.components().par_iter())
            .map(|component| ComponentFactor::build(component, config))
            .collect();
        let components = built.into_iter().collect::<Result<_, _>>()?;
        Ok(Self {
            n: sgdd.n(),
            components,
            dead: sgdd.dead().into(),
        })
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Each component's factor, in [`Sgdd::components`]' order, for a caller that solves them on its own.
    pub fn components(&self) -> &[ComponentFactor<T>] {
        &self.components
    }

    /// The components, owned, for a caller that keeps one solver per component; dead vertices need none.
    pub fn into_components(self) -> Vec<ComponentFactor<T>> {
        self.components
    }

    /// Cholesky factors routed dense and factored approximately instead.
    pub fn fallbacks(&self) -> impl Iterator<Item = Fallback> + '_ {
        self.components.iter().filter_map(ComponentFactor::fallback)
    }

    /// Scratch [`solve_in_place`](Self::solve_in_place) needs: a component's gathered values and its own scratch.
    pub fn scratch_len(&self) -> usize {
        (self.components.iter())
            .map(|component| component.vertices().len() + component.scratch_len())
            .max()
            .unwrap_or(0)
    }

    /// `A⁺b` on singular components; vertices in no component solve to `0`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut x = b.to_vec();
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.solve_in_place(&mut x, &mut scratch, InnerParallelism::Allow)?;
        Ok(x)
    }

    /// `x` holds `b` on entry and the solution on return; components solve in turn, each under `inner`.
    pub fn solve_in_place(
        &self,
        x: &mut [T],
        scratch: &mut [T],
        inner: InnerParallelism,
    ) -> Result<(), SolveError> {
        todo!()
    }
}

/// One component's factor over its vertices of `A`.
#[derive(Clone, Debug)]
pub struct ComponentFactor<T = f64> {
    frame: Arc<Frame<T>>,
    /// The component's own fact, not its Cholesky's: an underflowed complement surplus never makes it singular.
    floats: bool,
    elimination: Option<BlockElimination<T>>,
    cholesky: Cholesky<T>,
    fallback: Option<Fallback>,
}

impl<T: Real> ComponentFactor<T> {
    fn build(component: &Component<T>, config: &Config) -> Result<Self, FactorError> {
        let frame = component.frame();
        let vertices = frame.vertices();
        let streams = Streams::new(config.seed, vertices[0]);
        let (elimination, (cholesky, fallback)) = match component.m() {
            (OffDiagonal::General(adjacency), balance) => {
                let sddm = match balance {
                    Balance::Balanced(grounding) => Sddm::new(
                        Cow::Borrowed(adjacency),
                        Cow::Borrowed(grounding),
                        Sheets::One,
                    ),
                    Balance::Frustrated(grounding) => Sddm::cover(adjacency, grounding),
                };
                let dense = |sddm: &_, room| Cholesky::dense(room, sddm, vertices);
                let approximate = |sddm| Cholesky::approximate(sddm, config.approximate, streams);
                (None, route(sddm.n(), sddm, dense, approximate, config)?)
            }
            (OffDiagonal::Bipartite(cross), balance) => {
                let schur = SchurComplement::new(cross, balance);
                let kept = &vertices[schur.n_eliminated()..];
                // A dense Cholesky spends nothing on sparsity, so only the exact complement goes dense.
                let dense = |schur: &&SchurComplement<'_, T>, room| {
                    Cholesky::dense(room, &schur.build(CliqueRule::Exact, streams), kept)
                };
                let approximate = |schur: &SchurComplement<'_, T>| {
                    let s = schur.build(config.complement, streams);
                    Cholesky::approximate(s, config.approximate, streams)
                };
                let routed = route(schur.dimension(), &schur, dense, approximate, config)?;
                (Some(schur.finish()), routed)
            }
        };
        Ok(Self {
            frame: Arc::clone(frame),
            floats: component.floats(),
            elimination,
            cholesky,
            fallback,
        })
    }

    /// The component's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        self.frame.vertices()
    }

    /// Why this component is less accurate than its config asked, if it is.
    pub fn fallback(&self) -> Option<Fallback> {
        self.fallback
    }

    /// Scratch [`solve_local`](Self::solve_local) needs.
    pub fn scratch_len(&self) -> usize {
        let eliminated = (self.elimination.as_ref()).map_or(0, BlockElimination::n_eliminated);
        self.cholesky
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
        match &self.elimination {
            None => self.cholesky.solve(x, scratch),
            Some(elimination) => {
                elimination.forward(x, inner);
                let (eliminated, kept) = x.split_at_mut(elimination.n_eliminated());
                self.cholesky.solve(kept, scratch);
                elimination.backward(eliminated, kept, inner);
            }
        }
        congruence.apply(x);
        if self.floats {
            congruence.project_out_null(x);
        }
        Ok(())
    }
}

/// Dense when the rule says so, its room reserved before the matrix is built; a declined dense arm falls to the approximate one.
fn route<T: Real, M>(
    dimension: usize,
    matrix: M,
    dense: impl FnOnce(&M, DenseCholesky<T>) -> Result<Cholesky<T>, UnusablePivot>,
    approximate: impl FnOnce(M) -> Cholesky<T>,
    config: &Config,
) -> Result<(Cholesky<T>, Option<Fallback>), FactorError> {
    if !config.routes_dense(dimension) {
        return Ok((approximate(matrix), None));
    }
    let Some(room) = DenseCholesky::reserve(dimension) else {
        return Ok((
            approximate(matrix),
            Some(Fallback::WillNotFit { dimension }),
        ));
    };
    match dense(&matrix, room) {
        Ok(factor) => Ok((factor, None)),
        Err(pivot) if config.on_dense_failure == OnDenseFailure::Error => {
            Err(FactorError::UnusablePivot(pivot))
        }
        Err(pivot) => Ok((approximate(matrix), Some(Fallback::UnusablePivot(pivot)))),
    }
}
