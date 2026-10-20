//! [`CsrRef`] views of `sprs` and `faer` matrices, checked square before [`CsrRef::new`]; a CSC matrix is read as its transpose, which a symmetric one equals, so a storage defect names its column as the row.

#[cfg(feature = "faer")]
use num_traits::PrimInt;

use super::{CsrError, CsrRef, Input};
use crate::Error;

fn square(rows: usize, cols: usize) -> Result<u32, CsrError> {
    if rows != cols {
        return Err(CsrError::ExpectedSquareMatrix { rows, cols });
    }
    u32::try_from(rows).map_err(|_| CsrError::MatrixDimensionExceedsIndexType { n: rows })
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    type Error = Error;

    fn try_from(matrix: sprs::CsMatViewI<'a, T, I>) -> Result<Self, Error> {
        let n = square(matrix.rows(), matrix.cols())?;
        let (row_ptrs, col_indices, values) = matrix.into_raw_storage();
        CsrRef::new(row_ptrs, col_indices, values, n)
    }
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex> TryFrom<&'a sprs::CsMatI<T, I>> for CsrRef<'a, T, I> {
    type Error = Error;

    fn try_from(matrix: &'a sprs::CsMatI<T, I>) -> Result<Self, Error> {
        Self::try_from(matrix.view())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(matrix: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, Error> {
        let n = square(matrix.nrows(), matrix.ncols())?;
        let symbolic = matrix.symbolic();
        let row_ptrs = symbolic.row_ptr();
        // Row counts filling every span are compact storage, so the arrays still borrow as CSR.
        let rows = n as usize;
        if let Some(row) = (0..rows).find(|&i| symbolic.row_range(i).end != row_ptrs[i + 1].zx()) {
            return Err(CsrError::RowGap { row }.into());
        }
        // faer may keep spare capacity past the last row.
        let end = row_ptrs[rows].zx();
        let (col_indices, values) = (&symbolic.col_idx()[..end], &matrix.val()[..end]);
        CsrRef::new(row_ptrs, col_indices, values, n)
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<&'a faer::sparse::SparseRowMat<I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(matrix: &'a faer::sparse::SparseRowMat<I, T>) -> Result<Self, Error> {
        Self::try_from(matrix.as_ref())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<faer::sparse::SparseColMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(matrix: faer::sparse::SparseColMatRef<'a, I, T>) -> Result<Self, Error> {
        square(matrix.nrows(), matrix.ncols())?;
        Self::try_from(matrix.transpose())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<&'a faer::sparse::SparseColMat<I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(matrix: &'a faer::sparse::SparseColMat<I, T>) -> Result<Self, Error> {
        Self::try_from(matrix.as_ref())
    }
}

/// Each conversion into [`Input`] goes through its [`CsrRef`], so `factorize(&matrix)` reads as before.
macro_rules! input_via_csr {
    ($($matrix:ty where $($bound:path),+;)+) => {$(
        impl<'a, T, I: $($bound +)+> TryFrom<$matrix> for Input<'a, T, I> {
            type Error = Error;

            fn try_from(matrix: $matrix) -> Result<Self, Error> {
                CsrRef::try_from(matrix).map(Input::Csr)
            }
        }
    )+};
}

#[cfg(feature = "sprs")]
input_via_csr! {
    sprs::CsMatViewI<'a, T, I> where sprs::SpIndex;
    &'a sprs::CsMatI<T, I> where sprs::SpIndex;
}

#[cfg(feature = "faer")]
input_via_csr! {
    faer::sparse::SparseRowMatRef<'a, I, T> where faer::Index, PrimInt;
    &'a faer::sparse::SparseRowMat<I, T> where faer::Index, PrimInt;
    faer::sparse::SparseColMatRef<'a, I, T> where faer::Index, PrimInt;
    &'a faer::sparse::SparseColMat<I, T> where faer::Index, PrimInt;
}
