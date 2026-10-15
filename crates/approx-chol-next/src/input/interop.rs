//! [`Csr`] views of `sprs` and `faer` matrices, checked square and row-major before [`Csr::new`].

use num_traits::PrimInt;

use super::{Csr, CsrError};

/// Why a `sprs` or `faer` matrix has no [`Csr`] view.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvertError {
    /// The matrix is not square.
    NotSquare {
        /// Its row count.
        rows: usize,
        /// Its column count.
        cols: usize,
    },
    /// The matrix is stored by columns (CSC), not rows.
    #[cfg(feature = "sprs")]
    Csc,
    /// Row pointers start past `0`, as in a row slice; an owned copy converts.
    OffsetRows,
    /// A row leaves unused slots after its entries.
    #[cfg(feature = "faer")]
    RowGap {
        /// The first row with unused slots.
        row: usize,
    },
    /// The storage is not a CSR, which only an unchecked constructor builds.
    Csr(CsrError),
}

impl From<CsrError> for ConvertError {
    fn from(error: CsrError) -> Self {
        Self::Csr(error)
    }
}

impl core::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare { rows, cols } => write!(f, "matrix is {rows}x{cols}, not square"),
            #[cfg(feature = "sprs")]
            Self::Csc => f.write_str("matrix is stored by columns (CSC), not rows"),
            Self::OffsetRows => f.write_str("row pointers start past 0; convert an owned copy"),
            #[cfg(feature = "faer")]
            Self::RowGap { row } => write!(f, "row {row} leaves unused slots after its entries"),
            Self::Csr(_) => f.write_str("matrix storage is not a CSR"),
        }
    }
}

impl std::error::Error for ConvertError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Csr(error) => Some(error),
            _ => None,
        }
    }
}

fn square(rows: usize, cols: usize) -> Result<usize, ConvertError> {
    match rows == cols {
        true => Ok(rows),
        false => Err(ConvertError::NotSquare { rows, cols }),
    }
}

fn starts_at_zero<I: PrimInt>(row_ptrs: &[I]) -> Result<(), ConvertError> {
    match row_ptrs.first().is_none_or(|&start| start == I::zero()) {
        true => Ok(()),
        false => Err(ConvertError::OffsetRows),
    }
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex> TryFrom<sprs::CsMatViewI<'a, T, I>> for Csr<'a, T, I> {
    type Error = ConvertError;

    fn try_from(matrix: sprs::CsMatViewI<'a, T, I>) -> Result<Self, ConvertError> {
        if !matrix.is_csr() {
            return Err(ConvertError::Csc);
        }
        let n = square(matrix.rows(), matrix.cols())?;
        let (row_ptrs, col_indices, values) = matrix.into_raw_storage();
        starts_at_zero(row_ptrs)?;
        Ok(Csr::new(n, row_ptrs, col_indices, values)?)
    }
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex> TryFrom<&'a sprs::CsMatI<T, I>> for Csr<'a, T, I> {
    type Error = ConvertError;

    fn try_from(matrix: &'a sprs::CsMatI<T, I>) -> Result<Self, ConvertError> {
        Self::try_from(matrix.view())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
    for Csr<'a, T, I>
{
    type Error = ConvertError;

    fn try_from(matrix: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, ConvertError> {
        let n = square(matrix.nrows(), matrix.ncols())?;
        let symbolic = matrix.symbolic();
        let row_ptrs = symbolic.row_ptr();
        starts_at_zero(row_ptrs)?;
        // Row counts filling every span are compact storage, so the arrays still borrow as CSR.
        if let Some(row_nnz) = symbolic.row_nnz() {
            let gap = |i: usize| {
                let span = row_ptrs[i + 1].checked_sub(&row_ptrs[i]);
                span.is_some_and(|span| row_nnz[i] != span)
            };
            if let Some(row) = (0..n).find(|&i| gap(i)) {
                return Err(ConvertError::RowGap { row });
            }
        }
        // faer may keep spare capacity past the last row; a short array is left to `Csr::new`.
        let end = (row_ptrs.last()).map_or(0, |end| end.to_usize().unwrap_or(usize::MAX));
        let (col_indices, values) = (symbolic.col_idx(), matrix.val());
        Ok(Csr::new(
            n,
            row_ptrs,
            &col_indices[..end.min(col_indices.len())],
            &values[..end.min(values.len())],
        )?)
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<&'a faer::sparse::SparseRowMat<I, T>>
    for Csr<'a, T, I>
{
    type Error = ConvertError;

    fn try_from(matrix: &'a faer::sparse::SparseRowMat<I, T>) -> Result<Self, ConvertError> {
        Self::try_from(matrix.as_ref())
    }
}
