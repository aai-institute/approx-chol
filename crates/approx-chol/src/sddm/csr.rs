mod canonical;
mod validate;

use super::Sddm;
use crate::types::Real;
use crate::{CsrRef, Error};
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
        let canonical = Canonical::of(csr)?;
        let (summed, sums) = validate::edges(&canonical)?;
        let surplus = sums.surplus(canonical.terms())?;
        Sddm::with_surplus(summed, surplus)
    }
}
