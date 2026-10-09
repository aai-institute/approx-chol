//! Routes each part to its eliminations, and stores and applies `A_c⁺ ≈ D · Round · Kept⁺ · Round' · D`.

mod cover;
mod kernel;

use rayon::prelude::*;

pub use crate::round::InnerParallelism;

use crate::config::{CliqueRule, Config, ExactFailure};
use crate::dense::{LowerTriangular, PivotFailure, UnusablePivot};
use crate::min_degree;
use crate::round::{Round, RoundBuilder};
use crate::sampling::Streams;
use crate::sgdd::{Diagonal, Edges, NullVector, Part, Sddm, Sgdd, Sheets};
use crate::Real;
use kernel::{Cholesky, Kept, Kernel};

/// A remainder routed dense and factored approximately instead: less accurate than asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// The dense Cholesky reached an unusable pivot.
    InvalidPivot(UnusablePivot),
    /// The packed triangle does not fit in memory.
    WillNotFit {
        /// Rows of the declined remainder.
        dim: usize,
    },
}

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
    vertices: Box<[u32]>,
    diagonal: Diagonal<T>,
    round: Option<Round<T>>,
    kept: Kept<T>,
}

impl<T: Real> ComponentFactor<T> {
    /// The round runs on the signed matrix, so a frustrated part covers only what the kernel sees.
    fn build(part: &Part<T>, config: &Config) -> Result<(Self, Option<Fallback>), FactorError> {
        let streams = Streams::new(config.seed, part.vertices()[0]);
        let (matrix, sheets) = part.matrix().split();
        let (round, (kernel, fallback)) = match matrix.edges() {
            Edges::General(adjacency) => {
                let sddm = match sheets {
                    Sheets::One => Sddm::borrowed(adjacency, matrix.ground()),
                    Sheets::Two => cover::cover(adjacency, matrix.ground()),
                };
                let name = |row| kernel_vertex(part.vertices(), 0, sheets, row);
                let kernel = factor_kernel(sddm, config, streams, config.on_dense_failure, name)?;
                (None, kernel)
            }
            Edges::Bipartite(cross) => {
                let builder = RoundBuilder::new(cross, matrix.ground());
                let name = |row| kernel_vertex(part.vertices(), cross.n_eliminated(), sheets, row);
                let kernel = factor_complement(&builder, sheets, config, streams, name)?;
                (Some(builder.finish()), kernel)
            }
        };
        let factor = Self {
            vertices: part.vertices().into(),
            diagonal: part.diagonal().clone(),
            round,
            kept: Kept { sheets, kernel },
        };
        Ok((factor, fallback))
    }

    /// The part's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        &self.vertices
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
        if x.len() != self.vertices.len() {
            return Err(SolveError::LengthMismatch {
                len: x.len(),
                dim: self.vertices.len(),
            });
        }
        let needed = self.scratch_len();
        if scratch.len() < needed {
            return Err(SolveError::ScratchTooSmall {
                len: scratch.len(),
                needed,
            });
        }
        let gauge = self.gauge();
        gauge.project(x);
        self.diagonal.apply(x);
        match &self.round {
            None => self.kept.solve(x, scratch),
            Some(round) => {
                round.forward(x, inner);
                let (eliminated, kept) = x.split_at_mut(round.n_eliminated());
                self.kept.solve(kept, scratch);
                round.backward(eliminated, kept, inner);
            }
        }
        self.diagonal.apply(x);
        gauge.project(x);
        Ok(())
    }

    /// Derived, never stored: only a folded floating `M` leaves `A_c` singular.
    fn gauge(&self) -> Gauge<'_, T> {
        match &self.kept {
            Kept {
                sheets: Sheets::One,
                kernel: Kernel::Floating(_),
            } => Gauge::Singular(self.diagonal.null_vector()),
            _ => Gauge::Definite,
        }
    }
}

/// How the solve normalizes `A_c`'s null space so it returns `A⁺b`.
enum Gauge<'a, T> {
    Definite,
    Singular(NullVector<'a, T>),
}

impl<T: Real> Gauge<'_, T> {
    fn project(&self, x: &mut [T]) {
        if let Self::Singular(null) = self {
            null.project(x);
        }
    }
}

/// A kernel row's vertex in `A`: past the round's block, folded across the cover's sheets.
fn kernel_vertex(vertices: &[u32], n_eliminated: usize, sheets: Sheets, row: usize) -> u32 {
    todo!()
}

/// A dense kernel spends nothing on sparsity, so it is handed the exact complement first.
fn factor_complement<T: Real>(
    builder: &RoundBuilder<'_, T>,
    sheets: Sheets,
    config: &Config,
    streams: Streams,
    name: impl Fn(usize) -> u32,
) -> Result<(Kernel<T>, Option<Fallback>), FactorError> {
    if (1..=config.dense_below).contains(&(builder.n_kept() * sheets.count())) {
        let exact = builder.schur(sheets, CliqueRule::Exact, streams);
        if let Ok(done) = factor_kernel(exact, config, streams, ExactFailure::Error, &name) {
            return Ok(done);
        }
    }
    let reduced = builder.schur(sheets, config.round, streams);
    factor_kernel(reduced, config, streams, config.on_dense_failure, name)
}

/// Routes first, so a remainder the dense arm claims never builds an elimination graph.
fn factor_kernel<T: Real>(
    sddm: Sddm<'_, T>,
    config: &Config,
    streams: Streams,
    on_failure: ExactFailure,
    name: impl Fn(usize) -> u32,
) -> Result<(Kernel<T>, Option<Fallback>), FactorError> {
    let mut fallback = None;
    if (1..=config.dense_below).contains(&sddm.eliminated()) {
        match LowerTriangular::factor(&sddm) {
            Ok(lower) => return Ok((Kernel::new(sddm.ground(), Cholesky::Dense(lower)), None)),
            Err(failure) => fallback = Some(accept(on_failure, named(failure, name))?),
        }
    }
    let sequence = min_degree::eliminate(&sddm, config.kernel, streams);
    Ok((
        Kernel::new(sddm.ground(), Cholesky::Sequence(sequence)),
        fallback,
    ))
}

fn named(failure: PivotFailure, vertex: impl Fn(usize) -> u32) -> Fallback {
    match failure {
        PivotFailure::Invalid { row, failure } => Fallback::InvalidPivot(UnusablePivot {
            vertex: vertex(row),
            failure,
        }),
        PivotFailure::WillNotFit { dim } => Fallback::WillNotFit { dim },
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
