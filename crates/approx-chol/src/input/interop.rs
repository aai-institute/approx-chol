//! [`CsrRef`] views of `sprs`, `faer` and `nalgebra-sparse` matrices, checked square before [`CsrRef::new`]; a CSC matrix is read as its transpose, which a symmetric one equals, so a storage defect names its column as the row.

#[cfg(feature = "faer")]
use num_traits::PrimInt;

use super::{CsrRef, InputError};
use crate::Real;

fn square(rows: usize, cols: usize) -> Result<usize, InputError> {
    match rows == cols {
        true => Ok(rows),
        false => Err(InputError::NotSquare { rows, cols }),
    }
}

#[cfg(feature = "sprs")]
impl<'a, T: Real, I: sprs::SpIndex> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    type Error = InputError;

    fn try_from(matrix: sprs::CsMatViewI<'a, T, I>) -> Result<Self, InputError> {
        let n = square(matrix.rows(), matrix.cols())?;
        let (row_ptrs, col_indices, values) = matrix.into_raw_storage();
        CsrRef::new(n, row_ptrs, col_indices, values)
    }
}

#[cfg(feature = "faer")]
impl<'a, T: Real, I: faer::Index + PrimInt> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = InputError;

    fn try_from(matrix: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, InputError> {
        let n = square(matrix.nrows(), matrix.ncols())?;
        let symbolic = matrix.symbolic();
        let row_ptrs = symbolic.row_ptr();
        // Row counts filling every span are compact storage, so the arrays still borrow as CSR.
        if let Some(row) = (0..n).find(|&i| symbolic.row_range(i).end != row_ptrs[i + 1].zx()) {
            return Err(InputError::RowGap { row });
        }
        // faer may keep spare capacity past the last row.
        let end = row_ptrs[n].zx();
        let (col_indices, values) = (&symbolic.col_idx()[..end], &matrix.val()[..end]);
        CsrRef::new(n, row_ptrs, col_indices, values)
    }
}

#[cfg(feature = "nalgebra")]
impl<'a, T: Real> TryFrom<&'a nalgebra_sparse::CsrMatrix<T>> for CsrRef<'a, T, usize> {
    type Error = InputError;

    fn try_from(matrix: &'a nalgebra_sparse::CsrMatrix<T>) -> Result<Self, InputError> {
        let n = square(matrix.nrows(), matrix.ncols())?;
        let (row_ptrs, col_indices, values) = matrix.csr_data();
        CsrRef::new(n, row_ptrs, col_indices, values)
    }
}

#[cfg(feature = "nalgebra")]
impl<'a, T: Real> TryFrom<&'a nalgebra_sparse::CscMatrix<T>> for CsrRef<'a, T, usize> {
    type Error = InputError;

    fn try_from(matrix: &'a nalgebra_sparse::CscMatrix<T>) -> Result<Self, InputError> {
        let n = square(matrix.nrows(), matrix.ncols())?;
        let (col_ptrs, row_indices, values) = matrix.csc_data();
        CsrRef::new(n, col_ptrs, row_indices, values)
    }
}
