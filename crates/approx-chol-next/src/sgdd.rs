//! `A` as one reduced part `A_c = D · M · D` per connected component, decided once per input.

mod components;
mod congruence;
mod cover;
mod floating;
mod scaling;
mod signature;

use std::sync::Arc;

use num_traits::PrimInt;

pub(crate) use congruence::Congruence;
use floating::FloatingTest;
use scaling::NotScalable;
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::input::{Csr, CsrError, Gram, Graph};
use crate::laplacian::{Ground, Sddm, Sheets};
use crate::storage::{CrossBlock, UpperAdjacency};
use crate::Real;

/// One component of `A`, reduced: `A_c = D · M · D` over its vertices, eliminated block first.
#[derive(Clone, Debug)]
pub struct Part<T = f64> {
    frame: Arc<Frame<T>>,
    body: Body<T>,
}

impl<T> Part<T> {
    /// The part's vertices in `A`'s numbering, in local order.
    pub fn vertices(&self) -> &[u32] {
        &self.frame.vertices
    }

    /// Shared with every factor of this part.
    pub(crate) fn frame(&self) -> &Arc<Frame<T>> {
        &self.frame
    }

    pub(crate) fn body(&self) -> &Body<T> {
        &self.body
    }
}

/// Where a part sits in `A`: its vertices, and the congruence between `A_c` and `M`.
#[derive(Debug)]
pub(crate) struct Frame<T> {
    vertices: Box<[u32]>,
    congruence: Congruence<T>,
}

impl<T> Frame<T> {
    pub(crate) fn vertices(&self) -> &[u32] {
        &self.vertices
    }

    pub(crate) fn congruence(&self) -> &Congruence<T> {
        &self.congruence
    }
}

/// `M`, by what its factorization does with it; each payload states its own invariant.
#[derive(Clone, Debug)]
pub(crate) enum Body<T> {
    /// Folded to a Z-matrix: the kernel's input as it stands.
    Laplacian(Sddm<T>),
    /// Frustrated, with no independent block: the kernel factors its cover.
    Signed(SignedGeneral<T>),
    /// A leading independent block, eliminated as one round before the kernel.
    Bipartite(Bipartite<T>),
}

/// A general SDD matrix no signature folds.
#[derive(Clone, Debug)]
pub(crate) struct SignedGeneral<T> {
    adjacency: UpperAdjacency<T>,
    ground: Ground<T>,
}

/// `[D₁ C; Cᵀ D₂]`, signed when frustrated, which its kernel then sees on two sheets.
#[derive(Clone, Debug)]
pub(crate) struct Bipartite<T> {
    cross: Arc<CrossBlock<T>>,
    ground: Ground<T>,
    sheets: Sheets,
}

impl<T: Real> Bipartite<T> {
    /// Panics unless the surplus covers both blocks.
    fn new(cross: Arc<CrossBlock<T>>, ground: Ground<T>, sheets: Sheets) -> Self {
        let n = cross.n_eliminated() + cross.n_kept();
        assert!(
            ground.surplus().is_none_or(|s| s.len() == n),
            "one surplus per vertex"
        );
        Self {
            cross,
            ground,
            sheets,
        }
    }

    pub(crate) fn cross(&self) -> &Arc<CrossBlock<T>> {
        &self.cross
    }

    pub(crate) fn ground(&self) -> &Ground<T> {
        &self.ground
    }

    pub(crate) fn sheets(&self) -> Sheets {
        self.sheets
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

    /// Dead components produce no part; the floating test runs once, over the whole family.
    fn reduce<I: PrimInt>(
        n: usize,
        components: impl Iterator<Item = Unreduced<T>>,
        policy: &ScalingPolicy,
        test: Option<FloatingTest<'_, T, I>>,
    ) -> Result<Self, ReduceError> {
        let mut reduced = Vec::new();
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
            reduced.push(part);
        }
        let floating = match test {
            Some(test) => test.floating(&reduced),
            None => vec![false; reduced.len()],
        };
        let parts = (reduced.into_iter().zip(floating))
            .map(|(part, floats)| part.into_part(floats))
            .collect();
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

/// `A`'s off-diagonal by shape, before the body decides what it means.
enum Edges<T> {
    General(UpperAdjacency<T>),
    Bipartite(Arc<CrossBlock<T>>),
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

    fn reduce(
        self,
        policy: &ScalingPolicy,
    ) -> Result<(Reduced<T>, Option<ScalingReport>), NotScalable> {
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
    ) -> Result<(Reduced<T>, Option<ScalingReport>), NotScalable> {
        todo!()
    }
}

/// A reduced component whose ground waits for the family's floating test.
struct Reduced<T> {
    frame: Frame<T>,
    edges: Edges<T>,
    surplus: Vec<T>,
    /// Two when no signature folded it.
    sheets: Sheets,
}

impl<T: Real> Reduced<T> {
    fn into_part(self, floats: bool) -> Part<T> {
        let ground = match floats {
            true => Ground::floating(),
            false => Ground::of(self.surplus),
        };
        let body = match (self.edges, self.sheets) {
            (Edges::General(adjacency), Sheets::One) => {
                Body::Laplacian(Sddm::new(adjacency, ground, Sheets::One))
            }
            (Edges::General(adjacency), Sheets::Two) => {
                Body::Signed(SignedGeneral { adjacency, ground })
            }
            (Edges::Bipartite(cross), sheets) => {
                Body::Bipartite(Bipartite::new(cross, ground, sheets))
            }
        };
        Part {
            frame: Arc::new(self.frame),
            body,
        }
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
