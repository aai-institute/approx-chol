//! What the caller states about `A`: a CSR matrix or a [`Gram`].

mod csr;
mod gram;
#[cfg(any(feature = "sprs", feature = "faer"))]
mod interop;

use num_traits::PrimInt;

pub(crate) use csr::Canonical;
pub use csr::{CsrError, CsrRef, IndexKind, OwnedCsr, UpperCsrRef};
pub(crate) use gram::Column;
pub use gram::{FactorWidth1, Gram, ViewError, Weights};

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
    /// `V W Vᵀ + diag(d)`, `V` of factor width 2.
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

impl<'a, T, I: PrimInt> From<&'a OwnedCsr<T, I>> for Input<'a, T, I> {
    fn from(owned: &'a OwnedCsr<T, I>) -> Self {
        Self::Csr(owned.as_csr_ref())
    }
}
