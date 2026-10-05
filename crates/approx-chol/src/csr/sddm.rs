//! The CSR path into [`Sddm`]: canonicalize, check mirrors, judge which surplus is noise.

mod canonical;
mod validate;

use crate::types::Real;
use crate::{CsrRef, NotSddm, Sddm};
use num_traits::PrimInt;

/// Canonical input is read in place, in the caller's own index type.
pub(super) fn from_csr<T: Real, I: PrimInt>(csr: CsrRef<'_, T, I>) -> Result<Sddm<T>, NotSddm> {
    let nnz = csr.col_indices().len();
    if nnz > u32::MAX as usize {
        return Err(NotSddm::TooManyNonzeros { nnz });
    }
    if csr.n() == u32::MAX as usize {
        return Err(NotSddm::DimensionTooLarge { n: csr.n() });
    }
    let terms = canonical::terms(csr.row_ptrs());
    if canonical::is_canonical(csr.row_ptrs(), csr.col_indices()) {
        return validate::sddm_of(csr.row_ptrs(), csr.col_indices(), csr.values(), terms);
    }
    // Before rewriting, so the position stays the caller's own.
    if let Some(position) = csr.values().iter().position(|value| !value.is_finite()) {
        return Err(NotSddm::NonFiniteValue { position });
    }
    let rewritten = canonical::rewrite(csr)?;
    validate::sddm_of(
        &rewritten.row_ptrs,
        &rewritten.col_indices,
        &rewritten.values,
        terms,
    )
}
