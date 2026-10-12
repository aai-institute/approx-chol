//! `A` as its connected components, each reduced to `M = F · A_c · F`, decided once per input.

mod component;
mod congruence;
mod decomposition;
mod null_vectors;
mod reduction;
mod scaling;
mod signature;

use std::sync::Arc;

use num_traits::PrimInt;

pub use component::Component;
use component::Sdd;
use congruence::Congruence;
use reduction::DirectSum;
pub use scaling::{Existence, ScalingPolicy, ScalingReport};

use crate::input::{
    BipartiteGram, BipartiteGraph, BipartiteLaplacian, Csr, CsrError, FactorWidth2, Laplacian,
    SignedGraph, UpperCsr,
};
use crate::storage::{diagonal_in_range, CrossBlock, UpperAdjacency};
use crate::Real;

/// Where a component sits in `A`: its vertices, and the congruence between `A_c` and `M`.
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

/// Whether a signature folds `M` (Harary), and `M`'s grounding.
#[derive(Clone, Debug)]
pub(crate) enum Balance<T> {
    /// Folded to a Z-matrix: `M` itself, singular exactly when ungrounded, which reduction allows only on proof.
    Balanced(Grounding<T>),
    /// No signature exists: a two-sheet cover, definite on `[z; −z]` even ungrounded.
    Frustrated(Grounding<T>),
}

impl<T: Real> Balance<T> {
    pub(crate) fn grounding(&self) -> &Grounding<T> {
        match self {
            Self::Balanced(grounding) | Self::Frustrated(grounding) => grounding,
        }
    }

    /// The ground is a vertex too: its diagonal `Σs`, once per sheet, clears the same bound as any other.
    fn ground_in_range(&self) -> bool {
        let sheets = match self {
            Self::Balanced(_) => T::one(),
            Self::Frustrated(_) => T::one() + T::one(),
        };
        let total = |surplus: &[T]| surplus.iter().fold(T::zero(), |sum, &s| sum + s);
        (self.grounding().surplus())
            .is_none_or(|surplus| diagonal_in_range(sheets * total(surplus)))
    }
}

/// What closes `L` into an SDDM: a surplus `s = diag(M) − Σ|offdiag|` with a positive total, or none, leaving it singular.
#[derive(Clone, Debug)]
pub(crate) struct Grounding<T>(Option<Box<[T]>>);

impl<T: Real> Grounding<T> {
    /// Panics on a negative or non-finite entry; ungrounded exactly when every entry is zero.
    pub(crate) fn of(surplus: Vec<T>) -> Self {
        let valid = |&s: &T| s.is_finite() && s >= T::zero();
        assert!(
            surplus.iter().all(valid),
            "a surplus is finite and nonnegative"
        );
        match surplus.iter().all(|&s| s == T::zero()) {
            true => Self(None),
            false => Self(Some(surplus.into())),
        }
    }

    fn ungrounded() -> Self {
        Self(None)
    }

    pub(crate) fn surplus(&self) -> Option<&[T]> {
        self.0.as_deref()
    }

    pub(crate) fn is_grounded(&self) -> bool {
        self.0.is_some()
    }

    /// Whether it grounds exactly `n` rows.
    pub(crate) fn fits(&self, n: usize) -> bool {
        self.0.as_ref().is_none_or(|surplus| surplus.len() == n)
    }
}

/// A component's off-diagonal by shape: `A_c`'s signed one, or `M`'s once congruent.
#[derive(Clone, Debug)]
pub(crate) enum OffDiagonal<T> {
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

    /// Entries per row in local order: the rounding a row sum can carry.
    fn degrees(&self) -> Vec<usize> {
        match self {
            Self::General(adjacency) => adjacency.degrees(),
            Self::Bipartite(cross) => cross.degrees(),
        }
    }

    /// `F X F`; `None` if an entry leaves [`in_range`](crate::storage::in_range).
    fn scaled(self, f: &[T]) -> Option<Self> {
        Some(match self {
            Self::General(adjacency) => Self::General(adjacency.scaled(f)?),
            Self::Bipartite(cross) => {
                Self::Bipartite(Arc::new(Arc::unwrap_or_clone(cross).scaled(f)?))
            }
        })
    }

    /// As [`scaled`](Self::scaled), with the result's [`abs_row_sums`](Self::abs_row_sums) from the same pass.
    fn scaled_abs_row_sums(self, f: &[T]) -> Option<(Self, Vec<T>)> {
        Some(match self {
            Self::General(adjacency) => {
                let (adjacency, sums) = adjacency.scaled_abs_row_sums(f)?;
                (Self::General(adjacency), sums)
            }
            Self::Bipartite(cross) => {
                let (cross, sums) = Arc::unwrap_or_clone(cross).scaled_abs_row_sums(f)?;
                (Self::Bipartite(Arc::new(cross)), sums)
            }
        })
    }

    fn nonpositive(&self) -> bool {
        match self {
            Self::General(adjacency) => adjacency.nonpositive(),
            Self::Bipartite(cross) => cross.nonpositive(),
        }
    }
}

/// `A` as its connected [`Component`]s, each reduced; vertices in none solve to `0`.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64> {
    n: usize,
    components: Vec<Component<T>>,
    /// Lone vertices with a zero diagonal, recorded so a solve zeroes them without a pass over `n`.
    dead: Box<[u32]>,
}

impl<T: Real> Sgdd<T> {
    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The components, each factored and solved on its own.
    pub fn components(&self) -> &[Component<T>] {
        &self.components
    }

    pub(crate) fn dead(&self) -> &[u32] {
        &self.dead
    }

    /// The worst component's report: present iff some scaling was clamped beyond the policy's tolerance.
    pub fn scaling_report(&self) -> Option<ScalingReport> {
        (self.components.iter())
            .filter_map(Component::scaling_report)
            .max_by(|a, b| a.violation.total_cmp(&b.violation))
    }

    /// Checks symmetry, then scales each component to dominance unless it already is; an unscaled row's surplus up to about `2tε·a_ii` (`t` its entries) counts as rounding, and a scaled component floats only if its clamped surplus vanishes.
    pub fn from_csr<I: PrimInt>(
        csr: Csr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(DirectSum::from_csr(csr)?.by_rows(&policy)?)
    }

    /// As [`from_csr`](Self::from_csr), the triangle symmetric by storage: no mirror is checked.
    pub fn from_upper_csr<I: PrimInt>(
        csr: UpperCsr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(DirectSum::from_upper_csr(csr)?.by_rows(&policy)?)
    }

    /// `A = V Vᵀ`: scales, folds or covers each component as general, deciding floating from `V`.
    pub fn from_gram<I: PrimInt>(
        v: FactorWidth2<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        DirectSum::from_general(v)?.by_columns(&policy, v)
    }

    /// `A = V W Vᵀ` on two disjoint sides: as [`from_gram`](Self::from_gram), a bipartite component then taking the block elimination.
    pub fn from_bipartite_gram<I: PrimInt>(
        gram: BipartiteGram<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        DirectSum::from_bipartite(gram)?.by_columns(&policy, gram)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<Laplacian<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A graph Laplacian is dominant with surplus `g` by shape: nothing is searched, and a component floats exactly when its ground is zero.
    fn try_from(graph: Laplacian<'a, T, I>) -> Result<Self, Self::Error> {
        Self::from_laplacian(graph)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<SignedGraph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A signed graph is dominant by shape, each end's surplus its pairs' cancelled mass: nothing is scaled, and its signs are searched.
    fn try_from(graph: SignedGraph<'a, T, I>) -> Result<Self, Self::Error> {
        Self::from_signed_graph(graph)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<BipartiteGraph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A signless bipartite Laplacian has zero surplus by shape: every component floats untested.
    fn try_from(graph: BipartiteGraph<'a, T, I>) -> Result<Self, Self::Error> {
        Self::from_bipartite_graph(graph)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<BipartiteLaplacian<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A bipartite Laplacian is a Z-matrix with zero surplus by shape: every component floats untested.
    fn try_from(graph: BipartiteLaplacian<'a, T, I>) -> Result<Self, Self::Error> {
        Self::from_bipartite_laplacian(graph)
    }
}

/// Past `1/ε` vertices, floor weights composed in series fall below `MIN_POSITIVE` (f32: 2^23).
fn within_series_floor<T: Real>(vertices: &[u32]) -> Result<(), ReduceError> {
    match T::from(vertices.len()).is_none_or(|n| n * T::epsilon() >= T::one()) {
        true => Err(ReduceError::Unscalable {
            vertex: vertices[0],
        }),
        false => Ok(()),
    }
}

/// Why a component has no SDDM form; each `vertex` is in `A`'s numbering.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceError {
    /// No positive diagonal scaling makes the component weakly dominant: `diag − |offdiag|` is not PSD.
    NotGdd {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: u32,
    },
    /// A scale or a scaled entry overflowed or fell nonzero below `MIN_POSITIVE/EPSILON`, or the component passes `1/ε` vertices.
    Unscalable {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: u32,
    },
    /// An assembled entry is non-finite or nonzero below `MIN_POSITIVE/EPSILON`, a diagonal, a row's `Σ|a_ij|` or the ground's `Σs` exceeds `MAX/4`, or a view's diagonal underflowed to zero.
    OutOfRange {
        /// The entry's vertex, in `A`'s numbering.
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
