//! What the caller states about `A`, each view by its type.

mod csr;
#[cfg(any(feature = "sprs", feature = "faer"))]
mod interop;
mod width2;

use num_traits::PrimInt;

pub(crate) use csr::Canonical;
pub use csr::{CsrError, CsrRef, IndexKind, OwnedCsr, UpperCsrRef};
pub use width2::bipartite::{BipartiteGram, FactorWidth1};
pub use width2::edges::Weights;
pub(crate) use width2::gram::Switching;
pub use width2::gram::{Gram, Incidence, VertexPartition};
pub use width2::laplacian::Laplacian;
pub use width2::signed::SignedGraph;
pub(crate) use width2::{Column, IntoColumns};
pub use width2::{FactorWidth2, ViewError};

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
    /// `V W Vᵀ + diag(d)`, `V` of factor width 2: every factor view.
    Gram(Gram<'a, T, I>),
}

impl<T: Real, J: PrimInt + Sync> Input<'_, T, J> {
    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        match self {
            Self::Csr(csr) => csr.n(),
            Self::UpperCsr(csr) => csr.0.n(),
            Self::Gram(v) => v.n(),
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
    Gram(Gram),
}

/// Each factor view is its [`Gram`].
macro_rules! input_from_gram {
    ($($view:ident),+) => {$(
        impl<'a, T, I> From<$view<'a, T, I>> for Input<'a, T, I> {
            fn from(view: $view<'a, T, I>) -> Self {
                Self::Gram(view.into())
            }
        }
    )+};
}

input_from_gram!(Laplacian, SignedGraph, FactorWidth2, BipartiteGram);

impl<'a, T, I: PrimInt> From<&'a OwnedCsr<T, I>> for Input<'a, T, I> {
    fn from(owned: &'a OwnedCsr<T, I>) -> Self {
        Self::Csr(owned.as_csr_ref())
    }
}
