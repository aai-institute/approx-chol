//! `A` as its connected components, each reduced to `M = F · A_c · F`, decided once per input.

mod component;
mod congruence;
mod decomposition;
mod scaling;
mod signature;

use std::sync::Arc;

use num_traits::PrimInt;

pub use component::Component;
use congruence::Congruence;
use decomposition::Decomposition;
pub use scaling::{ScalingPolicy, ScalingReport};

use crate::input::{BipartiteGram, BipartiteGraph, Csr, CsrError, FactorWidth2};
use crate::storage::{CrossBlock, UpperAdjacency};
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
}

/// What closes `L` into an SDDM: a surplus `s = diag(M) − Σ|offdiag|` with a positive total, or none, leaving it singular.
#[derive(Clone, Debug)]
pub(crate) struct Grounding<T>(Option<Box<[T]>>);

impl<T: Real> Grounding<T> {
    /// Panics on a negative or non-finite entry; ungrounded exactly when every entry is zero.
    pub(crate) fn of(surplus: Vec<T>) -> Self {
        todo!()
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
        Ok(Decomposition::csr(csr)?.reduce(&policy)?)
    }

    /// `A = V Vᵀ`: scales, folds or covers each component as general, deciding floating from `V`.
    pub fn from_gram<I: PrimInt>(
        v: FactorWidth2<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        Decomposition::gram(v)?.reduce(&policy)
    }

    /// `A = V W Vᵀ` on two disjoint sides: as [`from_gram`](Self::from_gram), a bipartite component then taking the block elimination.
    pub fn from_bipartite_gram<I: PrimInt>(
        gram: BipartiteGram<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        Decomposition::bipartite_gram(gram)?.reduce(&policy)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<BipartiteGraph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A signless bipartite Laplacian has zero surplus by shape: every component floats untested.
    fn try_from(graph: BipartiteGraph<'a, T, I>) -> Result<Self, Self::Error> {
        Decomposition::signless(graph)?.reduce(&ScalingPolicy::default())
    }
}

/// One component `A_c` as the input gave it: signed, unscaled, in the input's shape.
struct Unreduced<T> {
    /// In `A`'s numbering, in local order.
    vertices: Box<[u32]>,
    off_diagonal: OffDiagonal<T>,
    diagonal: Vec<T>,
    /// `Σ_j |a_ij|`, summed while its rows are extracted, so dominance and surplus reread no value.
    abs_row_sums: Vec<T>,
    /// No positive off-diagonal: already a Z-matrix, so no signature is searched.
    nonpositive: bool,
}

/// Who proves a component singular: its own rows, or its input once every component is reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Proof {
    Rows,
    Input,
}

impl<T: Real> Unreduced<T> {
    /// Components join only nonzeros, so a vertex with nothing to solve is alone with a zero diagonal.
    fn is_dead(&self) -> bool {
        self.diagonal == [T::zero()]
    }

    fn reduce(self, policy: &ScalingPolicy, proof: Proof) -> Result<Component<T>, ReduceError> {
        // A view's zero diagonal beside a cell is its terms underflowing, not a non-GDD input.
        let lost = self.diagonal.iter().position(|&d| d == T::zero());
        if let (Proof::Input, Some(i)) = (proof, lost) {
            let vertex = self.vertices[i];
            return Err(ReduceError::OutOfRange { vertex });
        }
        let scaling = scaling::dominance_scaling(&self, policy)?;
        let signs = match self.nonpositive {
            true => Some(Congruence::identity()),
            false => signature::signature(&self.off_diagonal),
        };
        // A frustrated component is definite, so only a folded one asks its rows.
        let singular = proof == Proof::Rows
            && signs.is_some()
            && scaling::rows_prove_singular(&self, &scaling);
        let mut component = self.into_component(scaling, signs)?;
        if singular {
            component.float();
        }
        Ok(component)
    }

    /// `F = S R` folds and scales `A_c`, or only scales a frustrated one; deficits clamp to zero surplus.
    fn into_component(
        self,
        scaling: scaling::Scaling<T>,
        signs: Option<Congruence<T>>,
    ) -> Result<Component<T>, ReduceError> {
        todo!()
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
    /// A scale or a scaled entry left `[MIN_POSITIVE/EPSILON, MAX·EPSILON]`.
    Unscalable {
        /// The component's first vertex in local order, in `A`'s numbering.
        vertex: u32,
    },
    /// An assembled entry or diagonal is non-finite, nonzero outside `[MIN_POSITIVE/EPSILON, MAX·EPSILON]`, or a view's diagonal underflowed to zero.
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
