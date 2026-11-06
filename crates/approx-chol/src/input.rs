//! What the caller states about `A`: a CSR matrix, a Laplacian, an SDDM or a factor-width-2 matrix.

mod csr;
#[cfg(any(feature = "sprs", feature = "faer"))]
mod interop;
mod width2;

use num_traits::PrimInt;

pub(crate) use csr::Canonical;
pub use csr::{CsrError, CsrRef, IndexKind, OwnedCsr, UpperCsrRef};
pub use width2::bipartite::{BipartiteFactorWidth2, FactorWidth1, Side};
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

impl<'a, T: Real, J: PrimInt> Input<'a, T, J> {
    /// The one of three forms the core reads this input as.
    pub(crate) fn lower(self) -> Lowered<'a, T, J> {
        match self {
            Self::Csr(csr) => Lowered::Csr(csr),
            Self::UpperCsr(csr) => Lowered::UpperCsr(csr.lower()),
            Self::Laplacian(laplacian) => Lowered::FactorWidth2(laplacian.lower()),
            Self::Sddm(sddm) => Lowered::FactorWidth2(sddm.lower()),
            Self::FactorWidth2(v) => Lowered::FactorWidth2(v),
            Self::BipartiteFactorWidth2(v) => Lowered::FactorWidth2(v.lower()),
        }
    }

    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        match self.lower() {
            Lowered::Csr(csr) | Lowered::UpperCsr(csr) => csr.n(),
            Lowered::FactorWidth2(v) => v.n(),
        }
    }
}

/// An [`Input`] as the core reads it: a CSR matrix stored whole or by its upper triangle, or `V W Vᵀ`.
pub(crate) enum Lowered<'a, T, I> {
    Csr(CsrRef<'a, T, I>),
    UpperCsr(CsrRef<'a, T, I>),
    FactorWidth2(FactorWidth2<'a, T, I>),
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
