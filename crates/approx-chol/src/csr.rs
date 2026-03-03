use crate::{CsrError, Error};
use num_traits::{cast, PrimInt};

/// Borrowed CSR matrix view. Zero-copy from any CSR source.
///
/// This is the primary input type for
/// [`Builder::build`](crate::Builder::build).
/// Construct from raw arrays owned by any sparse matrix library
/// (`sprs`, `faer`, or plain `Vec`s).
#[derive(Debug, Clone, Copy)]
pub struct CsrRef<'a, T = f64, I = u32> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
    n: u32,
}

// ---------------------------------------------------------------------------
// Structural methods — only need I: PrimInt, no T bounds
// ---------------------------------------------------------------------------

impl<'a, T, I: PrimInt> CsrRef<'a, T, I> {
    /// Construct a `CsrRef` with full validation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if:
    /// - `row_ptrs.len() != n + 1`
    /// - `col_indices.len() != values.len()`
    /// - `row_ptrs[0] != 0`
    /// - `row_ptrs[n] != col_indices.len()`
    /// - `row_ptrs` is not non-decreasing
    /// - any column index is out of bounds (>= n)
    /// - a pointer/index cannot be represented as `usize`
    pub fn new(
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
        n: u32,
    ) -> Result<Self, Error> {
        let csr = Self {
            row_ptrs,
            col_indices,
            values,
            n,
        };
        csr.validate()?;
        Ok(csr)
    }

    /// Validate structural invariants of the CSR matrix.
    pub(crate) fn validate(&self) -> Result<(), Error> {
        let n = self.n as usize;
        if self.row_ptrs.len() != n + 1 {
            return Err(Error::InvalidCsr(CsrError::RowPtrsLenMismatch {
                expected: n + 1,
                got: self.row_ptrs.len(),
            }));
        }
        if self.col_indices.len() != self.values.len() {
            return Err(Error::InvalidCsr(CsrError::ColIndicesValuesLenMismatch {
                col_indices_len: self.col_indices.len(),
                values_len: self.values.len(),
            }));
        }

        let row_ptr_last = self.row_ptrs[n].to_usize().ok_or(Error::InvalidCsr(
            CsrError::RowPtrNotRepresentableAsUsize { position: n },
        ))?;
        let row_ptr_first = self.row_ptrs[0].to_usize().ok_or(Error::InvalidCsr(
            CsrError::RowPtrNotRepresentableAsUsize { position: 0 },
        ))?;
        if row_ptr_first != 0 {
            return Err(Error::InvalidCsr(CsrError::RowPtrsMustStartAtZero {
                got: row_ptr_first,
            }));
        }
        if row_ptr_last != self.col_indices.len() {
            return Err(Error::InvalidCsr(CsrError::RowPtrsEndMismatchNnz {
                row_ptr_end: row_ptr_last,
                nnz: self.col_indices.len(),
            }));
        }

        for i in 0..n {
            let a = self.row_ptrs[i].to_usize().ok_or(Error::InvalidCsr(
                CsrError::RowPtrNotRepresentableAsUsize { position: i },
            ))?;
            let b = self.row_ptrs[i + 1].to_usize().ok_or(Error::InvalidCsr(
                CsrError::RowPtrNotRepresentableAsUsize { position: i + 1 },
            ))?;
            if a > b {
                return Err(Error::InvalidCsr(CsrError::RowPtrsNotNonDecreasing {
                    row: i,
                    prev: a,
                    next: b,
                }));
            }
        }

        for (position, &col) in self.col_indices.iter().enumerate() {
            let col_usize = col.to_usize().ok_or(Error::InvalidCsr(
                CsrError::ColIndexNotRepresentableAsUsize { position },
            ))?;
            if col_usize >= self.n as usize {
                return Err(Error::InvalidCsr(CsrError::ColumnIndexOutOfBounds {
                    position,
                    col: col_usize,
                    n: self.n as usize,
                }));
            }
        }
        Ok(())
    }

    /// Row pointer array (length `n + 1`).
    #[inline]
    pub fn row_ptrs(&self) -> &[I] {
        self.row_ptrs
    }

    /// Column index array (length `nnz`).
    #[inline]
    pub fn col_indices(&self) -> &[I] {
        self.col_indices
    }

    /// Value array (length `nnz`).
    #[inline]
    pub fn values(&self) -> &[T] {
        self.values
    }

    /// Returns (col_indices, values) for row `i`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if `i >= n` or if row pointers are not
    /// representable as `usize`.
    #[inline]
    pub fn try_row(&self, i: usize) -> Result<(&'a [I], &'a [T]), Error> {
        if i >= self.n as usize {
            return Err(Error::InvalidCsr(CsrError::RowIndexOutOfBounds {
                row: i,
                n: self.n as usize,
            }));
        }
        let start = self.row_ptrs[i].to_usize().ok_or(Error::InvalidCsr(
            CsrError::RowPtrNotRepresentableAsUsize { position: i },
        ))?;
        let end = self.row_ptrs[i + 1].to_usize().ok_or(Error::InvalidCsr(
            CsrError::RowPtrNotRepresentableAsUsize { position: i + 1 },
        ))?;
        Ok((&self.col_indices[start..end], &self.values[start..end]))
    }

    #[inline]
    pub(crate) fn row_unchecked(&self, i: usize) -> (&'a [I], &'a [T]) {
        self.try_row(i)
            .expect("row index must be < n and row pointers validated")
    }

    /// Number of rows (and columns — the matrix is square).
    #[inline]
    pub fn n(&self) -> usize {
        self.n as usize
    }

    /// Validate structural invariants (debug builds only).
    pub fn debug_validate(&self) {
        let n = self.n as usize;
        debug_assert_eq!(self.row_ptrs.len(), n + 1);
        debug_assert_eq!(self.col_indices.len(), self.values.len());
        debug_assert_eq!(
            self.row_ptrs[n]
                .to_usize()
                .expect("row_ptr value cannot be represented as usize"),
            self.col_indices.len()
        );
    }
}

// ---------------------------------------------------------------------------
// Conversion methods — need T: Clone for value copies
// ---------------------------------------------------------------------------

impl<'a, T: Clone, I: PrimInt> CsrRef<'a, T, I> {
    /// Convert to an owned CSR with `u32` indices.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if any index does not fit in `u32`.
    pub fn to_owned_u32(&self) -> Result<OwnedCsr<T, u32>, Error> {
        let row_ptrs = self
            .row_ptrs
            .iter()
            .map(|&v| cast::<I, u32>(v))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidCsr(CsrError::RowPtrExceedsU32))?;
        let col_indices = self
            .col_indices
            .iter()
            .map(|&v| cast::<I, u32>(v))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidCsr(CsrError::ColIndexExceedsU32))?;
        Ok(OwnedCsr {
            row_ptrs,
            col_indices,
            values: self.values.to_vec(),
            n: self.n,
        })
    }
}

/// Owned CSR matrix. Convenience for sources that use `usize`.
#[derive(Debug, Clone)]
pub struct OwnedCsr<T = f64, I = u32> {
    row_ptrs: Vec<I>,
    col_indices: Vec<I>,
    values: Vec<T>,
    n: u32,
}

impl<T: Clone, I: PrimInt> OwnedCsr<T, I> {
    /// Convert `usize`-indexed CSR arrays to an owned representation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if any value exceeds index type
    /// capacity.
    pub fn try_from_usize(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[T],
        n: usize,
    ) -> Result<Self, Error> {
        let _ = cast::<usize, I>(n)
            .ok_or(Error::InvalidCsr(CsrError::NExceedsTargetIndexType { n }))?;
        let n = u32::try_from(n).map_err(|_| Error::InvalidCsr(CsrError::NExceedsU32 { n }))?;

        let row_ptrs = row_ptrs
            .iter()
            .map(|&v| cast::<usize, I>(v))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidCsr(CsrError::RowPtrExceedsTargetIndexType))?;

        let col_indices = col_indices
            .iter()
            .map(|&v| cast::<usize, I>(v))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidCsr(CsrError::ColIndexExceedsTargetIndexType))?;

        CsrRef::new(&row_ptrs, &col_indices, values, n)?;

        Ok(Self {
            row_ptrs,
            col_indices,
            values: values.to_vec(),
            n,
        })
    }
}

impl<T, I: PrimInt> OwnedCsr<T, I> {
    /// Borrow as a [`CsrRef`] for use with
    /// [`Builder::build`](crate::Builder::build).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if internal CSR invariants are violated.
    pub fn try_as_ref(&self) -> Result<CsrRef<'_, T, I>, Error> {
        CsrRef::new(&self.row_ptrs, &self.col_indices, &self.values, self.n)
    }
}

#[cfg(feature = "sprs")]
fn try_from_sprs_view_impl<'a, T, I: sprs::SpIndex + PrimInt>(
    mat: sprs::CsMatViewI<'a, T, I>,
) -> Result<CsrRef<'a, T, I>, Error> {
    if !mat.is_csr() {
        return Err(Error::InvalidCsr(CsrError::ExpectedCsrMatrixGotCsc));
    }
    let rows = mat.rows();
    let cols = mat.cols();
    if rows != cols {
        return Err(Error::InvalidCsr(CsrError::ExpectedSquareMatrix {
            rows,
            cols,
        }));
    }
    let n = u32::try_from(rows)
        .map_err(|_| Error::InvalidCsr(CsrError::MatrixDimensionExceedsU32 { n: rows }))?;
    let (indptr, indices, data) = mat.into_raw_storage();
    CsrRef::new(indptr, indices, data, n)
}

#[cfg(feature = "faer")]
fn try_from_faer_view_impl<'a, T, I: faer::Index + PrimInt>(
    mat: faer::sparse::SparseRowMatRef<'a, I, T>,
) -> Result<CsrRef<'a, T, I>, Error> {
    let rows = mat.nrows();
    let cols = mat.ncols();
    if rows != cols {
        return Err(Error::InvalidCsr(CsrError::ExpectedSquareMatrix {
            rows,
            cols,
        }));
    }
    let n = u32::try_from(rows)
        .map_err(|_| Error::InvalidCsr(CsrError::MatrixDimensionExceedsU32 { n: rows }))?;
    let symbolic = mat.symbolic();
    CsrRef::new(symbolic.row_ptr(), symbolic.col_idx(), mat.val(), n)
}

#[cfg(feature = "sprs")]
impl<'a, T, I> CsrRef<'a, T, I>
where
    I: sprs::SpIndex + PrimInt,
{
    /// Fallible zero-copy conversion from an `sprs` CSR matrix view.
    ///
    /// Returns [`Error::InvalidCsr`] if the matrix is not CSR, not
    /// square, or has dimension larger than `u32::MAX`.
    pub fn try_from_sprs_view(mat: sprs::CsMatViewI<'a, T, I>) -> Result<Self, Error> {
        try_from_sprs_view_impl(mat)
    }

    /// Fallible zero-copy conversion from a borrowed `sprs` CSR matrix.
    ///
    /// Returns [`Error::InvalidCsr`] with the same conditions as
    /// [`Self::try_from_sprs_view`].
    pub fn try_from_sprs(mat: &'a sprs::CsMatI<T, I>) -> Result<Self, Error> {
        Self::try_from_sprs_view(mat.view())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I> CsrRef<'a, T, I>
where
    I: faer::Index + PrimInt,
{
    /// Fallible zero-copy conversion from a `faer` CSR matrix view.
    ///
    /// Returns [`Error::InvalidCsr`] if the matrix is not square or
    /// has dimension larger than `u32::MAX`.
    pub fn try_from_faer_view(mat: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, Error> {
        try_from_faer_view_impl(mat)
    }

    /// Fallible zero-copy conversion from a borrowed `faer` sparse row matrix.
    ///
    /// Returns [`Error::InvalidCsr`] with the same conditions as
    /// [`Self::try_from_faer_view`].
    pub fn try_from_faer(mat: &'a faer::sparse::SparseRowMat<I, T>) -> Result<Self, Error> {
        Self::try_from_faer_view(mat.as_ref())
    }
}

/// Fallible zero-copy conversion from an `sprs` CSR matrix view.
#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    type Error = Error;

    fn try_from(mat: sprs::CsMatViewI<'a, T, I>) -> Result<Self, Self::Error> {
        try_from_sprs_view_impl(mat)
    }
}

/// Fallible zero-copy conversion from a borrowed `sprs` CSR matrix.
#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> TryFrom<&'a sprs::CsMatI<T, I>> for CsrRef<'a, T, I> {
    type Error = Error;

    fn try_from(mat: &'a sprs::CsMatI<T, I>) -> Result<Self, Self::Error> {
        try_from_sprs_view_impl(mat.view())
    }
}

/// Fallible zero-copy conversion from a `faer` sparse row matrix view.
#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(mat: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, Self::Error> {
        try_from_faer_view_impl(mat)
    }
}

/// Fallible zero-copy conversion from a borrowed `faer` sparse row matrix.
#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<&'a faer::sparse::SparseRowMat<I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(mat: &'a faer::sparse::SparseRowMat<I, T>) -> Result<Self, Self::Error> {
        try_from_faer_view_impl(mat.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_owned_u32_preserves_indices_for_u32_input() {
        let row_ptrs = [0u32, 1];
        let col_indices = [0u32];
        let values = [1.0f64];
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 1).expect("valid csr");

        let converted = csr.to_owned_u32().expect("conversion");
        let converted_ref = converted.try_as_ref().expect("must stay valid");
        assert_eq!(converted_ref.row_ptrs(), &row_ptrs);
        assert_eq!(converted_ref.col_indices(), &col_indices);
        assert_eq!(converted_ref.values(), &values);
    }

    #[test]
    fn to_owned_u32_converts_non_u32_indices() {
        let row_ptrs = [0usize, 1];
        let col_indices = [0usize];
        let values = [1.0f64];
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 1).expect("valid csr");

        let converted = csr.to_owned_u32().expect("conversion");
        let converted_ref = converted.try_as_ref().expect("must stay valid");
        assert_eq!(converted_ref.row_ptrs(), &[0u32, 1]);
        assert_eq!(converted_ref.col_indices(), &[0u32]);
        assert_eq!(converted_ref.values(), &values);
    }
}
