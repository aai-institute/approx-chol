//! A sprs view stored by rows as a [`CsrRef`].

use super::square;
use crate::input::{CsrNames, CsrRef, InputError};
use crate::Real;

/// sprs's names of a compressed matrix's arrays.
const NAMES: CsrNames = CsrNames {
    row_ptrs: "indptr",
    col_indices: "indices",
    values: "data",
};

impl<'a, T: Real, I: sprs::SpIndex> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    type Error = InputError;

    fn try_from(matrix: sprs::CsMatViewI<'a, T, I>) -> Result<Self, InputError> {
        let n = square(matrix.rows(), matrix.cols())?;
        if matrix.is_csc() {
            return Err(InputError::StoredByColumns);
        }
        let (indptr, indices, data) = matrix.into_raw_storage();
        CsrRef::named(n, indptr, indices, data, NAMES)
    }
}
