//! What the caller states about `A`, each view by its type.

mod csr;
#[cfg(any(feature = "sprs", feature = "faer"))]
mod interop;
mod width2;

use num_traits::PrimInt;

pub(crate) use csr::Canonical;
pub use csr::{CsrError, CsrRef, IndexKind, OwnedCsr, UpperCsrRef};
pub use width2::bipartite::{BipartiteGram, FactorWidth1};
pub use width2::error::ViewError;
pub use width2::laplacian::Laplacian;
pub use width2::weights::Weights;
pub(crate) use width2::Column;
pub use width2::FactorWidth2;

use crate::Real;

/// A shape-checked index is non-negative and fits `usize`.
fn index<I: PrimInt>(k: I) -> usize {
    k.to_usize().expect("a shape-checked index fits usize")
}

/// Any view [`factorize`](crate::factorize) takes, each converted by `From`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum Input<'a, T, I = u32> {
    /// A symmetric matrix stored whole.
    Csr(CsrRef<'a, T, I>),
    /// A symmetric matrix stored by its upper triangle.
    UpperCsr(UpperCsrRef<'a, T, I>),
    /// A graph Laplacian by its edges.
    Laplacian(Laplacian<'a, T, I>),
    /// `V Vᵀ`, `V` of factor width 2.
    Gram(FactorWidth2<'a, T, I>),
    /// `V W Vᵀ` on two disjoint sides.
    BipartiteGram(BipartiteGram<'a, T, I>),
}

impl<T: Real, J: PrimInt + Sync> Input<'_, T, J> {
    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        match self {
            Self::Csr(csr) => csr.n(),
            Self::UpperCsr(csr) => csr.0.n(),
            Self::Laplacian(Laplacian(v))
            | Self::Gram(v)
            | Self::BipartiteGram(BipartiteGram(v)) => v.n(),
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
    Gram(FactorWidth2),
    BipartiteGram(BipartiteGram),
}

impl<'a, T, I: PrimInt> From<&'a OwnedCsr<T, I>> for Input<'a, T, I> {
    fn from(owned: &'a OwnedCsr<T, I>) -> Self {
        Self::Csr(owned.as_csr_ref())
    }
}
