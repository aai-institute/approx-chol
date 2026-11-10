//! What the caller states about `A`: a CSR matrix, a Laplacian, an SDDM or a factor-width-2 matrix.

mod csr;
mod error;
#[cfg(any(feature = "sprs", feature = "faer", feature = "nalgebra"))]
mod interop;
mod width2;

use num_traits::PrimInt;

pub(crate) use csr::{Canonical, CsrArrays};
pub use csr::{CsrRef, TriangleCsrRef};
pub use error::InputError;
pub use width2::bipartite::{BipartiteFactorWidth2, FactorWidth1, Side};
pub use width2::laplacian::{Laplacian, Sddm};
pub(crate) use width2::Column;
pub use width2::FactorWidth2;

use crate::Real;

/// A shape-checked index is non-negative and fits `usize`.
fn index<I: PrimInt>(k: I) -> usize {
    k.to_usize().expect("a shape-checked index fits usize")
}

/// Any input [`factorize`](crate::factorize) takes, each converted by `From`.
#[derive(Clone, Copy, Debug)]
pub struct Input<'a, T, I = u32>(pub(crate) Lowered<'a, T, I>);

impl<T: Real, J: PrimInt> Input<'_, T, J> {
    /// Dimension of `A`.
    pub(crate) fn n(&self) -> usize {
        match self.0 {
            Lowered::Csr(csr) => csr.n(),
            Lowered::FactorWidth2(v) => v.n(),
        }
    }
}

/// An [`Input`] as the core reads it: a CSR matrix stored whole or by its upper triangle, or `V W Vᵀ`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Lowered<'a, T, I> {
    Csr(CsrArrays<'a, T, I>),
    FactorWidth2(FactorWidth2<'a, T, I>),
}

macro_rules! input_from {
    ($($view:ident => |$v:ident| $lowered:expr,)+) => {$(
        impl<'a, T: Real, I: PrimInt> From<$view<'a, T, I>> for Input<'a, T, I> {
            fn from($v: $view<'a, T, I>) -> Self {
                Self($lowered)
            }
        }
    )+};
}

input_from! {
    CsrRef => |csr| Lowered::Csr(csr.0),
    TriangleCsrRef => |triangle| Lowered::Csr(triangle.0),
    Laplacian => |laplacian| Lowered::FactorWidth2(laplacian.lower()),
    Sddm => |sddm| Lowered::FactorWidth2(sddm.lower()),
    FactorWidth2 => |v| Lowered::FactorWidth2(v),
    BipartiteFactorWidth2 => |v| Lowered::FactorWidth2(v.lower()),
}
