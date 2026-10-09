//! `A` as one reduced part `M = F · A_c · F` per connected component, decided once per input.

mod components;
mod congruence;
mod family;
mod scaling;
mod signature;

use std::borrow::Cow;
use std::sync::Arc;

use num_traits::PrimInt;

pub(crate) use congruence::Congruence;
use family::Family;
use scaling::NotScalable;
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::input::{Csr, CsrError, Gram, Graph};
use crate::round::RoundBuilder;
use crate::sddm::{Grounding, Sddm, Sheets};
use crate::storage::{CrossBlock, UpperAdjacency};
use crate::Real;

/// One component of `A`, reduced: `M = F · A_c · F` over its vertices, eliminated block first.
#[derive(Clone, Debug)]
pub struct Part<T = f64> {
    frame: Arc<Frame<T>>,
    m: Reduced<T>,
    report: Option<ScalingReport>,
}

impl<T: Real> Part<T> {
    /// Panics unless `M` has one row per vertex of the frame.
    fn new(frame: Frame<T>, m: Reduced<T>, report: Option<ScalingReport>) -> Self {
        assert_eq!(m.n(), frame.vertices.len(), "one row per vertex");
        Self {
            frame: Arc::new(frame),
            m,
            report,
        }
    }

    /// The part's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        &self.frame.vertices
    }

    /// Present iff this part's scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        self.report
    }

    /// Shared with every factor of this part.
    pub(crate) fn frame(&self) -> &Arc<Frame<T>> {
        &self.frame
    }

    /// What the kernel factors: `M` itself, its cover, or a round's complement.
    pub(crate) fn elimination(&self) -> Elimination<'_, T> {
        let Reduced {
            off_diagonal,
            grounding,
            sheets,
        } = &self.m;
        match off_diagonal {
            OffDiagonal::General(adjacency) => Elimination::Whole(match sheets {
                Sheets::One => Sddm::new(
                    Cow::Borrowed(adjacency),
                    Cow::Borrowed(grounding),
                    Sheets::One,
                ),
                Sheets::Two => Sddm::cover(adjacency, grounding),
            }),
            OffDiagonal::Bipartite(cross) => {
                Elimination::Round(RoundBuilder::new(cross, grounding, *sheets))
            }
        }
    }
}

/// How a part's factorization starts.
pub(crate) enum Elimination<'a, T: Real> {
    /// The kernel factors this matrix whole.
    Whole(Sddm<'a, T>),
    /// A round eliminates the leading block, and the kernel its complement.
    Round(RoundBuilder<'a, T>),
}

/// Where a part sits in `A`: its vertices, and the congruence between `A_c` and `M`.
#[derive(Debug)]
pub(crate) struct Frame<T> {
    vertices: Box<[u32]>,
    congruence: Congruence<T>,
}

impl<T: Real> Frame<T> {
    /// Panics unless the congruence scales exactly these vertices.
    fn new(vertices: Box<[u32]>, congruence: Congruence<T>) -> Self {
        assert!(congruence.fits(vertices.len()), "one factor per vertex");
        Self {
            vertices,
            congruence,
        }
    }
}

impl<T> Frame<T> {
    pub(crate) fn vertices(&self) -> &[u32] {
        &self.vertices
    }

    pub(crate) fn congruence(&self) -> &Congruence<T> {
        &self.congruence
    }
}

/// `M`: its off-diagonal, its surplus, and the sheets its kernel factors it on.
#[derive(Clone, Debug)]
struct Reduced<T> {
    off_diagonal: OffDiagonal<T>,
    grounding: Grounding<T>,
    /// Two when no signature folds `M`, so its kernel sees a cover.
    sheets: Sheets,
}

impl<T: Real> Reduced<T> {
    /// Panics unless the grounding has one surplus per row.
    fn new(off_diagonal: OffDiagonal<T>, grounding: Grounding<T>, sheets: Sheets) -> Self {
        assert!(grounding.fits(off_diagonal.n()), "one surplus per row");
        Self {
            off_diagonal,
            grounding,
            sheets,
        }
    }

    fn n(&self) -> usize {
        self.off_diagonal.n()
    }

    /// A proven alias anchors one vertex instead of the surplus rounding left.
    fn float(&mut self) {
        self.grounding = Grounding::Floating;
    }
}

/// A component's off-diagonal by shape: `A_c`'s signed one, or `M`'s once congruent.
#[derive(Clone, Debug)]
enum OffDiagonal<T> {
    General(UpperAdjacency<T>),
    /// Both sides nonempty, the larger eliminated first.
    Bipartite(Arc<CrossBlock<T>>),
}

impl<T: Real> OffDiagonal<T> {
    fn n(&self) -> usize {
        match self {
            Self::General(adjacency) => adjacency.n(),
            Self::Bipartite(cross) => cross.n(),
        }
    }
}

/// `A` as one reduced [`Part`] per connected component; vertices in no part solve to `0`.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64> {
    n: usize,
    parts: Vec<Part<T>>,
}

impl<T: Real> Sgdd<T> {
    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The parts, each factored and solved on its own.
    pub fn parts(&self) -> &[Part<T>] {
        &self.parts
    }

    /// The worst part's report: present iff some scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        (self.parts.iter())
            .filter_map(Part::scaling_report)
            .max_by(|a, b| a.violation.total_cmp(&b.violation))
    }

    /// Checks symmetry, then scales each component to dominance unless it already is.
    pub fn from_csr<I: PrimInt>(
        csr: Csr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(Self::reduce(Family::csr(csr)?, &policy)?)
    }

    /// Scales, folds or covers each component of `V W Vᵀ`, deciding floating from `V`.
    pub fn from_gram<I: PrimInt>(
        gram: Gram<'_, '_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        Self::reduce(Family::gram(gram), &policy)
    }

    /// Dead components produce no part; the floating test runs once, over the whole family.
    fn reduce<I: PrimInt>(
        family: Family<'_, T, I>,
        policy: &ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let mut parts = Vec::new();
        for a in family.components().filter(|a| !a.is_dead()) {
            let vertex = a.vertices[0] as usize;
            parts.push(a.reduce(policy).map_err(|e| match e {
                NotScalable::NotGdd => ReduceError::NotGdd { vertex },
                NotScalable::Overflow => ReduceError::Unscalable { vertex },
            })?);
        }
        family.float(&mut parts);
        Ok(Self {
            n: family.n(),
            parts,
        })
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<Graph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A signless Laplacian has zero surplus by shape: every part floats untested.
    fn try_from(graph: Graph<'a, T, I>) -> Result<Self, Self::Error> {
        Self::reduce(Family::signless(graph), &ScalingPolicy::default())
    }
}

/// One component of `A` as the input gave it: signed, unscaled, in the input's shape.
struct Unreduced<T> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[u32]>,
    off_diagonal: OffDiagonal<T>,
    diagonal: Vec<T>,
}

impl<T: Real> Unreduced<T> {
    /// Components join only nonzeros, so a vertex with nothing to solve is alone with a zero diagonal.
    fn is_dead(&self) -> bool {
        self.diagonal == [T::zero()]
    }

    fn reduce(self, policy: &ScalingPolicy) -> Result<Part<T>, NotScalable> {
        let scaling = scaling::dominance_scaling(&self, policy)?;
        let signs = signature::signature(&self.off_diagonal);
        self.congruence(scaling, signs)
    }

    /// `F = S R` folds and scales `A`, or only scales a frustrated one; deficits are clamped to zero surplus.
    fn congruence(
        self,
        scaling: scaling::Scaling<T>,
        signs: Option<Congruence<T>>,
    ) -> Result<Part<T>, NotScalable> {
        todo!()
    }
}

/// Why a component has no SDDM form; `vertex` is its first in local order, larger side first.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceError {
    /// No positive diagonal scaling makes the component weakly dominant.
    NotGdd {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: usize,
    },
    /// A scale or a scaled entry left the finite range.
    Unscalable {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: usize,
    },
}

impl core::fmt::Display for ReduceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotGdd { vertex } => write!(
                f,
                "the component of vertex {vertex} is not generalized diagonally dominant"
            ),
            Self::Unscalable { vertex } => write!(
                f,
                "scaling the component of vertex {vertex} leaves the finite range"
            ),
        }
    }
}

impl std::error::Error for ReduceError {}

/// Why a CSR did not become an [`Sgdd`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FromCsrError {
    /// The arrays are not a symmetric matrix.
    Csr(CsrError),
    /// A component has no SDDM form.
    Reduce(ReduceError),
}

impl From<CsrError> for FromCsrError {
    fn from(error: CsrError) -> Self {
        Self::Csr(error)
    }
}

impl From<ReduceError> for FromCsrError {
    fn from(error: ReduceError) -> Self {
        Self::Reduce(error)
    }
}

impl core::fmt::Display for FromCsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("cannot convert CSR to Sgdd")
    }
}

impl std::error::Error for FromCsrError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Csr(error) => Some(error),
            Self::Reduce(error) => Some(error),
        }
    }
}
