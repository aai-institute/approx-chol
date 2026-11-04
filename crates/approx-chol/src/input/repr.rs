//! The representation an [`Input`](super::Input) holds, apart from the public types that build it.

pub(crate) mod gram;

use super::CsrRef;
use gram::Gram;

/// A CSR form or a Gram form.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Repr<'a, T, I> {
    Csr(CsrRef<'a, T, I>, Stored),
    Gram(Gram<'a, T, I>),
}

/// How much of a symmetric CSR matrix is stored.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Stored {
    Whole,
    UpperTriangle,
}
