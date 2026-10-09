//! `A` as one reduced part `A_c = D · M · D` per connected component, decided once per input.

mod components;
mod diagonal;
mod floating;
mod ground;
mod scaling;
mod signature;

use std::borrow::Cow;
use std::sync::Arc;

use num_traits::PrimInt;

pub(crate) use diagonal::{Diagonal, NullVector};
use floating::FloatingTest;
pub(crate) use ground::Ground;
use scaling::NotScalable;
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::input::{Csr, CsrError, Gram, Graph};
use crate::storage::{CrossBlock, UpperAdjacency};
use crate::Real;

/// `M`'s off-diagonal by shape; a bipartite one names its leading block independent.
#[derive(Clone, Debug)]
pub(crate) enum Edges<T> {
    General(UpperAdjacency<T>),
    Bipartite(Arc<CrossBlock<T>>),
}

/// `M = L + diag(s)`: its off-diagonal and its surplus.
#[derive(Clone, Debug)]
pub(crate) struct Matrix<T> {
    edges: Edges<T>,
    ground: Ground<T>,
}

impl<T> Matrix<T> {
    pub(crate) fn edges(&self) -> &Edges<T> {
        &self.edges
    }

    pub(crate) fn ground(&self) -> &Ground<T> {
        &self.ground
    }
}

/// A part's `M`; only this module builds one, so a `Laplacian` has Laplacian weights by construction.
#[derive(Clone, Debug)]
pub(crate) enum Sdd<T> {
    /// Folded to a Z-matrix.
    Laplacian(Matrix<T>),
    /// Frustrated: no signature folds it, so the kernel factors its Gremban cover.
    Signed(Matrix<T>),
}

impl<T> Sdd<T> {
    pub(crate) fn split(&self) -> (&Matrix<T>, Sheets) {
        match self {
            Self::Laplacian(matrix) => (matrix, Sheets::One),
            Self::Signed(matrix) => (matrix, Sheets::Two),
        }
    }

    /// A proven alias anchors one vertex instead, so its surplus is dropped.
    fn float(&mut self) {
        let (Self::Laplacian(matrix) | Self::Signed(matrix)) = self;
        matrix.ground = Ground::Floating;
    }
}

/// How many copies of the kept vertices the kernel factors: two for a Gremban cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Sheets {
    One,
    Two,
}

impl Sheets {
    pub(crate) fn count(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
        }
    }
}

/// The kernel's only input: Laplacian weights in general storage, borrowed when nothing reduced it.
#[derive(Clone, Debug)]
pub(crate) struct Sddm<'a, T: Clone> {
    adjacency: Cow<'a, UpperAdjacency<T>>,
    ground: Cow<'a, Ground<T>>,
}

impl<'a, T: Real> Sddm<'a, T> {
    pub(crate) fn borrowed(adjacency: &'a UpperAdjacency<T>, ground: &'a Ground<T>) -> Self {
        Self {
            adjacency: Cow::Borrowed(adjacency),
            ground: Cow::Borrowed(ground),
        }
    }

    pub(crate) fn owned(adjacency: UpperAdjacency<T>, ground: Ground<T>) -> Self {
        Self {
            adjacency: Cow::Owned(adjacency),
            ground: Cow::Owned(ground),
        }
    }

    pub(crate) fn adjacency(&self) -> &UpperAdjacency<T> {
        &self.adjacency
    }

    pub(crate) fn ground(&self) -> &Ground<T> {
        &self.ground
    }

    /// Vertices a Cholesky solves for: all of a grounded one, all but one of a floating one.
    pub(crate) fn eliminated(&self) -> usize {
        match *self.ground {
            Ground::Grounded(_) => self.adjacency.n(),
            Ground::Floating => self.adjacency.n() - 1,
        }
    }
}

/// One component of `A`, reduced: `A_c = D · M · D` over its vertices, eliminated block first.
#[derive(Clone, Debug)]
pub struct Part<T = f64> {
    vertices: Box<[u32]>,
    diagonal: Diagonal<T>,
    matrix: Sdd<T>,
}

impl<T> Part<T> {
    /// The part's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        &self.vertices
    }

    pub(crate) fn diagonal(&self) -> &Diagonal<T> {
        &self.diagonal
    }

    pub(crate) fn matrix(&self) -> &Sdd<T> {
        &self.matrix
    }
}

/// `A` as one reduced [`Part`] per connected component; vertices in no part solve to `0`.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64> {
    n: usize,
    parts: Vec<Part<T>>,
    report: Option<ScalingReport>,
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

    /// Present iff some part's scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        self.report
    }

    /// Checks symmetry, then scales each component to dominance unless it already is.
    pub fn from_csr<I: PrimInt>(
        csr: Csr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        let partition = components::of_csr(&csr)?;
        let components = (partition.components()).map(|vertices| Unreduced::of_csr(&csr, vertices));
        Ok(Self::reduce(
            csr.n(),
            components,
            &policy,
            Some(FloatingTest::RowFloor(csr)),
        )?)
    }

    /// Scales, folds or covers each component of `V W Vᵀ`, deciding floating from `V`.
    pub fn from_gram<I: PrimInt>(
        gram: Gram<'_, '_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        let (left, right) = gram.diagonals();
        let columns = gram.columns();
        let cross = columns.assemble(left.len(), right.len());
        let partition = components::of_cross(&cross);
        let components = (partition.components()).map(|vertices| {
            Unreduced::of_cross(
                &cross,
                left.len(),
                vertices,
                CrossDiagonal::Given(left, right),
            )
        });
        Self::reduce(
            left.len() + right.len(),
            components,
            &policy,
            Some(FloatingTest::Observations(columns)),
        )
    }

    /// Dead components produce no part; the floating test runs once, over every part.
    fn reduce<I: PrimInt>(
        n: usize,
        components: impl Iterator<Item = Unreduced<T>>,
        policy: &ScalingPolicy,
        test: Option<FloatingTest<'_, T, I>>,
    ) -> Result<Self, ReduceError> {
        let mut parts = Vec::new();
        let mut report: Option<ScalingReport> = None;
        for a in components.filter(|a| !a.is_dead()) {
            let vertex = a.vertices[0];
            let (part, uncertified) = a.reduce(policy).map_err(|e| match e {
                NotScalable::NotGdd => ReduceError::NotGdd { vertex },
                NotScalable::Overflow => ReduceError::Unscalable { vertex },
            })?;
            if let Some(found) = uncertified {
                if report.is_none_or(|worst| found.violation > worst.violation) {
                    report = Some(found);
                }
            }
            parts.push(part);
        }
        if let Some(test) = test {
            let floating = test.floating(&parts);
            for (part, _) in parts.iter_mut().zip(floating).filter(|(_, f)| *f) {
                part.matrix.float();
            }
        }
        Ok(Self { n, parts, report })
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<Graph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A signless Laplacian has zero surplus by shape: every part floats untested.
    fn try_from(graph: Graph<'a, T, I>) -> Result<Self, Self::Error> {
        let columns = graph.columns();
        let (n_left, n_right) = graph.sizes();
        let cross = columns.assemble(n_left, n_right);
        let partition = components::of_cross(&cross);
        let components = (partition.components()).map(|vertices| {
            Unreduced::of_cross(&cross, n_left, vertices, CrossDiagonal::Adjacency)
        });
        Self::reduce::<I>(
            n_left + n_right,
            components,
            &ScalingPolicy::default(),
            None,
        )
    }
}

/// One component of `A` as the input gave it: signed, unscaled, in the input's shape.
struct Unreduced<T> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[u32]>,
    edges: Edges<T>,
    diagonal: Vec<T>,
}

impl<T: Real> Unreduced<T> {
    /// Copies the component's rows: in-place reading measured 2-10% cheaper, so revisit with a view.
    fn of_csr<I: PrimInt>(csr: &Csr<'_, T, I>, vertices: &[u32]) -> Self {
        todo!()
    }

    /// `vertices` are left rows then right ones; the larger side is oriented first.
    fn of_cross(
        cross: &CrossBlock<T>,
        n_left: usize,
        vertices: &[u32],
        diagonal: CrossDiagonal<'_, T>,
    ) -> Self {
        todo!()
    }

    /// An all-zero diagonal leaves nothing to solve: its vertices solve to `0`.
    fn is_dead(&self) -> bool {
        todo!()
    }

    /// Grounded wherever surplus remains; the family's floating test drops what it proves aliased.
    fn reduce(
        self,
        policy: &ScalingPolicy,
    ) -> Result<(Part<T>, Option<ScalingReport>), NotScalable> {
        let certificate = scaling::dominance_scaling(&self, policy)?;
        let signature = signature::signature(&self.edges);
        self.congruence(certificate, signature, policy)
    }

    /// `D = signs · scales` folds and scales `A`; surplus is what dominance leaves, deficits clamped.
    fn congruence(
        self,
        certificate: scaling::Certificate<T>,
        signature: signature::Signature,
        policy: &ScalingPolicy,
    ) -> Result<(Part<T>, Option<ScalingReport>), NotScalable> {
        todo!()
    }
}

/// Where a bipartite component's diagonal comes from.
enum CrossDiagonal<'a, T> {
    /// A signless Laplacian's: its adjacency sums.
    Adjacency,
    /// A Gram's, left rows then right ones.
    Given(&'a [T], &'a [T]),
}

/// Why a component has no SDDM form; `vertex` is its first vertex in `A`'s numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceError {
    /// No positive diagonal scaling makes the component weakly dominant.
    NotGdd {
        /// First vertex of the component.
        vertex: u32,
    },
    /// A scale or a scaled entry left the finite range.
    Unscalable {
        /// First vertex of the component.
        vertex: u32,
    },
}

impl core::fmt::Display for ReduceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cannot reduce to SDDM form: {self:?}")
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
        write!(f, "cannot convert CSR to Sgdd: {self:?}")
    }
}

impl std::error::Error for FromCsrError {}
