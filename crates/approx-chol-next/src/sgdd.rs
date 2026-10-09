//! `A` as one reduced part `M = F · A_c · F` per connected component, decided once per input.

mod components;
mod congruence;
mod decomposition;
mod scaling;
mod signature;

use std::borrow::Cow;
use std::sync::Arc;

use num_traits::PrimInt;

use congruence::Congruence;
use decomposition::Decomposition;
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
    m: Sdd<T>,
    report: Option<ScalingReport>,
}

impl<T: Real> Part<T> {
    /// Panics unless `M` has one row per vertex of the frame.
    fn new(frame: Frame<T>, m: Sdd<T>, report: Option<ScalingReport>) -> Self {
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

    /// Whether `A_c` is singular, so a solve returns `A_c⁺b`.
    pub(crate) fn floats(&self) -> bool {
        matches!(&self.m.balance, Balance::Balanced(grounding) if !grounding.is_grounded())
    }

    /// What the remainder factors: `M` itself, its cover, or a round's complement.
    pub(crate) fn elimination(&self) -> Elimination<'_, T> {
        match (&self.m.off_diagonal, &self.m.balance) {
            (OffDiagonal::General(adjacency), Balance::Balanced(grounding)) => {
                let sddm = Sddm::new(
                    Cow::Borrowed(adjacency),
                    Cow::Borrowed(grounding),
                    Sheets::One,
                );
                Elimination::Whole(sddm)
            }
            (OffDiagonal::General(adjacency), Balance::Frustrated(grounding)) => {
                Elimination::Whole(Sddm::cover(adjacency, grounding))
            }
            (OffDiagonal::Bipartite(cross), balance) => {
                Elimination::Round(RoundBuilder::new(cross, balance))
            }
        }
    }
}

/// How a part's factorization starts.
pub(crate) enum Elimination<'a, T: Real> {
    /// The remainder is this matrix whole.
    Whole(Sddm<'a, T>),
    /// A round eliminates the leading block, and the remainder is its complement.
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

/// `M`, weakly dominant once scaled and clamped: its off-diagonal, and whether a signature folded it.
#[derive(Clone, Debug)]
struct Sdd<T> {
    off_diagonal: OffDiagonal<T>,
    balance: Balance<T>,
}

impl<T: Real> Sdd<T> {
    /// Panics unless one surplus per row, and a balanced `M` folded to a Z-matrix.
    fn new(off_diagonal: OffDiagonal<T>, balance: Balance<T>) -> Self {
        assert!(
            balance.grounding().fits(off_diagonal.n()),
            "one surplus per row"
        );
        if let Balance::Balanced(_) = balance {
            assert!(off_diagonal.nonpositive(), "a folded M is a Z-matrix");
        }
        Self {
            off_diagonal,
            balance,
        }
    }

    fn n(&self) -> usize {
        self.off_diagonal.n()
    }

    /// A proven alias anchors one vertex instead of the surplus rounding left; a frustrated `M` is definite.
    fn float(&mut self) {
        if let Balance::Balanced(grounding) = &mut self.balance {
            *grounding = Grounding::floating();
        }
    }
}

/// Whether a signature folds `M` (Harary), and the grounding of the matrix its remainder factors.
#[derive(Clone, Debug)]
pub(crate) enum Balance<T> {
    /// Folded to a Z-matrix: `M` itself, singular when it floats.
    Balanced(Grounding<T>),
    /// No signature exists: a two-sheet cover, definite on `[z; −z]` even when it floats.
    Frustrated(Grounding<T>),
}

impl<T: Real> Balance<T> {
    pub(crate) fn sheets(&self) -> Sheets {
        match self {
            Self::Balanced(_) => Sheets::One,
            Self::Frustrated(_) => Sheets::Two,
        }
    }

    pub(crate) fn grounding(&self) -> &Grounding<T> {
        match self {
            Self::Balanced(grounding) | Self::Frustrated(grounding) => grounding,
        }
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

    /// `Σ_j |a_ij|` per row in local order: dominance, surplus and pivots all start here.
    fn abs_row_sums(&self) -> Vec<T> {
        match self {
            Self::General(adjacency) => adjacency.abs_row_sums(),
            Self::Bipartite(cross) => cross.abs_row_sums(),
        }
    }

    fn nonpositive(&self) -> bool {
        match self {
            Self::General(adjacency) => adjacency.nonpositive(),
            Self::Bipartite(cross) => cross.nonpositive(),
        }
    }
}

/// `A` as one reduced [`Part`] per connected component; vertices in no part solve to `0`.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64> {
    n: usize,
    parts: Vec<Part<T>>,
    /// Lone vertices with a zero diagonal, recorded so a solve zeroes them without a pass over `n`.
    dead: Box<[u32]>,
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

    pub(crate) fn dead(&self) -> &[u32] {
        &self.dead
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
        Ok(Self::reduce(Decomposition::csr(csr)?, &policy)?)
    }

    /// Scales, folds or covers each component of `V W Vᵀ`, deciding floating from `V`.
    pub fn from_gram<I: PrimInt>(
        gram: Gram<'_, '_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        Self::reduce(Decomposition::gram(gram)?, &policy)
    }

    /// Dead components produce no part; the floating test runs once, over the whole input.
    fn reduce<I: PrimInt>(
        mut decomposition: Decomposition<'_, T, I>,
        policy: &ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let (mut parts, mut dead) = (Vec::new(), Vec::new());
        for a in decomposition.components() {
            let vertex = a.vertices[0];
            if a.is_dead() {
                dead.push(vertex);
                continue;
            }
            parts.push(a.reduce(policy).map_err(|e| match e {
                NotScalable::NotGdd => ReduceError::NotGdd { vertex },
                NotScalable::ScaleOutOfRange => ReduceError::Unscalable { vertex },
                NotScalable::DiagonalOutOfRange => ReduceError::OutOfRange { vertex },
            })?);
        }
        decomposition.float(&mut parts);
        Ok(Self {
            n: decomposition.n(),
            parts,
            dead: dead.into_boxed_slice(),
        })
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<Graph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A signless Laplacian has zero surplus by shape: every part floats untested.
    fn try_from(graph: Graph<'a, T, I>) -> Result<Self, Self::Error> {
        Self::reduce(Decomposition::signless(graph)?, &ScalingPolicy::default())
    }
}

/// One component `A_c` as the input gave it: signed, unscaled, in the input's shape.
struct Component<T> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[u32]>,
    off_diagonal: OffDiagonal<T>,
    diagonal: Vec<T>,
}

impl<T: Real> Component<T> {
    /// Components join only nonzeros, so a vertex with nothing to solve is alone with a zero diagonal.
    fn is_dead(&self) -> bool {
        self.diagonal == [T::zero()]
    }

    fn reduce(self, policy: &ScalingPolicy) -> Result<Part<T>, NotScalable> {
        let scaling = scaling::dominance_scaling(&self, policy)?;
        let signature = signature::signature(&self.off_diagonal);
        self.into_part(scaling, signature)
    }

    /// `F = S R` folds and scales `A_c`, or only scales a frustrated one; deficits clamp to zero surplus.
    fn into_part(
        self,
        scaling: scaling::Scaling<T>,
        signature: Option<signature::Signature>,
    ) -> Result<Part<T>, NotScalable> {
        todo!()
    }
}

/// Why a component has no SDDM form; each `vertex` is in `A`'s numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceError {
    /// No positive diagonal scaling makes the component weakly dominant.
    NotGdd {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: u32,
    },
    /// A scale or a scaled entry overflowed, or fell nonzero below `MIN_POSITIVE/EPSILON`.
    Unscalable {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: u32,
    },
    /// An assembled entry or diagonal is non-finite, or nonzero below `MIN_POSITIVE/EPSILON`.
    OutOfRange {
        /// A vertex of the entry, in `A`'s numbering.
        vertex: u32,
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
                "scaling the component of vertex {vertex} leaves the representable range"
            ),
            Self::OutOfRange { vertex } => {
                write!(
                    f,
                    "an entry at vertex {vertex} is outside the representable range"
                )
            }
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
