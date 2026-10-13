//! `A` as its connected components, each reduced to `M = F · A_c · F`, decided once per input.

mod component;
mod congruence;
mod decomposition;
mod null_vectors;
mod reduction;
mod scaling;
mod signature;

use num_traits::PrimInt;

pub(crate) use component::{Balance, Frame, Grounding, OffDiagonal};
pub use component::{Component, ReduceError, Sgdd};
use decomposition::{of_csr, of_upper};
use reduction::{reduce_bipartite_gram, reduce_bipartite_graph, reduce_bipartite_laplacian};
use reduction::{reduce_gram, reduce_laplacian, reduce_signed};
pub use scaling::{Existence, ScalingPolicy, ScalingReport};

use crate::input::{
    BipartiteGram, BipartiteGraph, BipartiteLaplacian, Canonical, Csr, CsrError, FactorWidth2,
    Laplacian, SignedGraph, UpperCsr,
};
use crate::Real;

impl<T: Real> Sgdd<T> {
    /// Checks symmetry, then scales each component to dominance unless it already is.
    pub fn from_csr<I: PrimInt>(
        csr: Csr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(of_csr(Canonical::of(csr)?)?.reduce(&policy)?)
    }

    /// As [`from_csr`](Self::from_csr), the triangle symmetric by storage: no mirror is checked.
    pub fn from_upper_csr<I: PrimInt>(
        csr: UpperCsr<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, FromCsrError> {
        Ok(of_upper(Canonical::upper(csr)?)?.reduce(&policy)?)
    }

    /// `A = V Vᵀ`: scales, folds or covers each component as general, deciding floating from `V`.
    pub fn from_gram<I: PrimInt>(
        v: FactorWidth2<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        reduce_gram(v, &policy)
    }

    /// `A = V W Vᵀ` on two disjoint sides `[D_l C; Cᵀ D_r]`, as [`from_gram`](Self::from_gram).
    pub fn from_bipartite_gram<I: PrimInt>(
        gram: BipartiteGram<'_, T, I>,
        policy: ScalingPolicy,
    ) -> Result<Self, ReduceError> {
        reduce_bipartite_gram(gram, &policy)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<Laplacian<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// Dominant with surplus `g` by shape: a component floats exactly when its ground is zero.
    fn try_from(graph: Laplacian<'a, T, I>) -> Result<Self, Self::Error> {
        reduce_laplacian(graph)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<BipartiteGraph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// Negating one side folds it into a bipartite Laplacian: every component floats untested.
    fn try_from(graph: BipartiteGraph<'a, T, I>) -> Result<Self, Self::Error> {
        reduce_bipartite_graph(graph)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<BipartiteLaplacian<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// A Z-matrix with zero surplus by shape: every component floats untested.
    fn try_from(graph: BipartiteLaplacian<'a, T, I>) -> Result<Self, Self::Error> {
        reduce_bipartite_laplacian(graph)
    }
}

impl<'a, T: Real, I: PrimInt> TryFrom<SignedGraph<'a, T, I>> for Sgdd<T> {
    type Error = ReduceError;

    /// Dominant by shape, each end's surplus its pairs' cancelled mass; its signs are searched.
    fn try_from(graph: SignedGraph<'a, T, I>) -> Result<Self, Self::Error> {
        reduce_signed(graph)
    }
}

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
