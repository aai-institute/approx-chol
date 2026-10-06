//! The CSR path into [`Sddm`]: canonicalize, check mirrors, judge which surplus is noise.

mod canonical;
mod validate;

use super::Sddm;
use crate::types::Real;
use crate::{CsrError, CsrRef, Error};
use canonical::Canonical;
use num_traits::PrimInt;

/// A validated [`CsrRef`] index is non-negative and fits `usize`.
#[inline(always)]
fn index<I: PrimInt>(value: I) -> usize {
    value.to_usize().expect("a validated CSR index is a usize")
}

impl<'a, T: Real, I: PrimInt> TryFrom<CsrRef<'a, T, I>> for Sddm<T> {
    type Error = Error;

    fn try_from(csr: CsrRef<'a, T, I>) -> Result<Self, Error> {
        // A ground vertex needs an index of its own.
        if csr.n() == u32::MAX as usize {
            return Err(Error::InvalidCsr(
                CsrError::MatrixDimensionExceedsIndexType {
                    n: csr.n().saturating_add(1),
                },
            ));
        }
        validate::sddm_of(&Canonical::of(csr)?)
    }
}
