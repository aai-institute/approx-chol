//! The CSR path into [`Sddm`]: canonicalize, check mirrors, judge which surplus is noise.

mod canonical;
mod validate;

use crate::{CsrRef, NotSddm, Sddm};
use canonical::Canonical;
use num_traits::{Float, PrimInt};

/// A validated [`CsrRef`] index is non-negative and fits `usize`.
#[inline(always)]
fn index<I: PrimInt>(value: I) -> usize {
    value.to_usize().expect("a validated CSR index is a usize")
}

/// The only path that judges mirrors and surplus noise; everything downstream trusts the result.
impl<'a, T: Float, I: PrimInt> TryFrom<CsrRef<'a, T, I>> for Sddm<T> {
    type Error = NotSddm;

    fn try_from(csr: CsrRef<'a, T, I>) -> Result<Self, NotSddm> {
        if csr.n() == u32::MAX as usize {
            return Err(NotSddm::DimensionTooLarge { n: csr.n() });
        }
        validate::sddm_of(&Canonical::of(csr)?)
    }
}
