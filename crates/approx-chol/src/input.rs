//! What the caller states about `A`: a CSR matrix, a Laplacian, an SDDM or a factor-width-2 matrix.

mod csr;
#[cfg(any(feature = "sprs", feature = "faer"))]
mod interop;
mod width2;

use num_traits::PrimInt;

pub(crate) use csr::Canonical;
pub use csr::{CsrError, CsrRef, IndexKind, OwnedCsr, UpperCsrRef};
pub use width2::bipartite::{BipartiteFactorWidth2, FactorWidth1};
pub use width2::error::ViewError;
pub use width2::laplacian::{Laplacian, Sddm};
pub(crate) use width2::Column;
pub use width2::FactorWidth2;

use crate::Real;

/// A shape-checked index is non-negative and fits `usize`.
fn index<I: PrimInt>(k: I) -> usize {
    k.to_usize().expect("a shape-checked index fits usize")
}

/// Any input [`factorize`](crate::factorize) takes, each converted by `From`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum Input<'a, T, I = u32> {
    /// A symmetric matrix stored whole.
    Csr(CsrRef<'a, T, I>),
    /// A symmetric matrix stored by its upper triangle.
    UpperCsr(UpperCsrRef<'a, T, I>),
    /// A graph Laplacian `L`.
    Laplacian(Laplacian<'a, T, I>),
    /// `L + diag(s)`, `s ≥ 0` and not all zero.
    Sddm(Sddm<'a, T, I>),
    /// `V Vᵀ`, each column of `V` with two nonzeros.
    FactorWidth2(FactorWidth2<'a, T, I>),
    /// `V W Vᵀ`, `V = [V_l; V_r]` with two width-1 sides on disjoint vertex sets.
    BipartiteFactorWidth2(BipartiteFactorWidth2<'a, T, I>),
}

impl<T: Real, J: PrimInt + Sync> Input<'_, T, J> {
    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        match self {
            Self::Csr(csr) => csr.n(),
            Self::UpperCsr(csr) => csr.0.n(),
            Self::Laplacian(laplacian) => laplacian.lower().n(),
            Self::Sddm(sddm) => sddm.lower().n(),
            Self::FactorWidth2(v) | Self::BipartiteFactorWidth2(BipartiteFactorWidth2(v)) => v.n(),
        }
    }
}

macro_rules! input_from {
    ($($variant:ident($view:ident),)+) => {$(
        impl<'a, T, I> From<$view<'a, T, I>> for Input<'a, T, I> {
            fn from(view: $view<'a, T, I>) -> Self {
                Self::$variant(view)
            }
        }
    )+};
}

input_from! {
    Csr(CsrRef),
    UpperCsr(UpperCsrRef),
    Laplacian(Laplacian),
    Sddm(Sddm),
    FactorWidth2(FactorWidth2),
    BipartiteFactorWidth2(BipartiteFactorWidth2),
}

impl<'a, T, I: PrimInt> From<&'a OwnedCsr<T, I>> for Input<'a, T, I> {
    fn from(owned: &'a OwnedCsr<T, I>) -> Self {
        Self::Csr(owned.as_csr_ref())
    }
}
