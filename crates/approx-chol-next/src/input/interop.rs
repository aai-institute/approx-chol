//! [`Csr`] views of `sprs` and `faer` matrices, checked square and row-major before [`Csr::new`].

use super::{Csr, CsrError};

fn square(rows: usize, cols: usize) -> Result<usize, CsrError> {
    match rows == cols {
        true => Ok(rows),
        false => Err(CsrError::NotSquare { rows, cols }),
    }
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex> TryFrom<sprs::CsMatViewI<'a, T, I>> for Csr<'a, T, I> {
    type Error = CsrError;

    fn try_from(matrix: sprs::CsMatViewI<'a, T, I>) -> Result<Self, CsrError> {
        if !matrix.is_csr() {
            return Err(CsrError::Csc);
        }
        let n = square(matrix.rows(), matrix.cols())?;
        let (row_ptrs, col_indices, values) = matrix.into_raw_storage();
        Csr::new(n, row_ptrs, col_indices, values)
    }
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex> TryFrom<&'a sprs::CsMatI<T, I>> for Csr<'a, T, I> {
    type Error = CsrError;

    fn try_from(matrix: &'a sprs::CsMatI<T, I>) -> Result<Self, CsrError> {
        Self::try_from(matrix.view())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + num_traits::PrimInt> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
    for Csr<'a, T, I>
{
    type Error = CsrError;

    fn try_from(matrix: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, CsrError> {
        let n = square(matrix.nrows(), matrix.ncols())?;
        let symbolic = matrix.symbolic();
        let row_ptrs = symbolic.row_ptr();
        // Row counts filling every span are compact storage, so the arrays still borrow as CSR.
        if let Some(row_nnz) = symbolic.row_nnz() {
            if let Some(row) = (0..n).find(|&i| row_ptrs[i] + row_nnz[i] != row_ptrs[i + 1]) {
                return Err(CsrError::RowGap { row });
            }
        }
        Csr::new(n, row_ptrs, symbolic.col_idx(), matrix.val())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + num_traits::PrimInt> TryFrom<&'a faer::sparse::SparseRowMat<I, T>>
    for Csr<'a, T, I>
{
    type Error = CsrError;

    fn try_from(matrix: &'a faer::sparse::SparseRowMat<I, T>) -> Result<Self, CsrError> {
        Self::try_from(matrix.as_ref())
    }
}
