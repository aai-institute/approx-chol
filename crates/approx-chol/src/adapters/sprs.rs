//! A sprs view as a [`CsrRef`].

use super::square;
use crate::input::{CsrRef, InputError};
use crate::Real;

impl<'a, T: Real, I: sprs::SpIndex> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    type Error = InputError;

    fn try_from(matrix: sprs::CsMatViewI<'a, T, I>) -> Result<Self, InputError> {
        let n = square(matrix.rows(), matrix.cols())?;
        let (row_ptrs, col_indices, values) = matrix.into_raw_storage();
        CsrRef::new(n, row_ptrs, col_indices, values)
    }
}
