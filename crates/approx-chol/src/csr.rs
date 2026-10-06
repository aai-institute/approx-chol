use crate::{CsrError, Error, IndexKind};
use num_traits::{cast, PrimInt};

/// Reserves up front: collecting into `Option<Vec<_>>` drops the size hint and cost 3.5x the traffic.
fn cast_slice<S: PrimInt, D: PrimInt>(src: &[S], kind: IndexKind) -> Result<Vec<D>, Error> {
    let mut out = Vec::with_capacity(src.len());
    for &value in src {
        out.push(
            cast::<S, D>(value)
                .ok_or(Error::InvalidCsr(CsrError::IndexExceedsIndexType { kind }))?,
        );
    }
    Ok(out)
}

fn as_usize<I: PrimInt>(value: I, kind: IndexKind, position: usize) -> Result<usize, CsrError> {
    value
        .to_usize()
        .ok_or(CsrError::IndexNotRepresentableAsUsize { kind, position })
}

/// Zero-copy, validated CSR view over any library's arrays: the factorization input.
#[derive(Debug, Clone, Copy)]
pub struct CsrRef<'a, T = f64, I = u32> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
    n: u32,
}

impl<'a, T, I: PrimInt> CsrRef<'a, T, I> {
    /// The only constructor, so every `CsrRef` is valid; [`Error::InvalidCsr`] names a violation.
    pub fn new(
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
        n: u32,
    ) -> Result<Self, Error> {
        Self::validated(row_ptrs, col_indices, values, n).map_err(Error::InvalidCsr)
    }

    pub(crate) fn validated(
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
        n: u32,
    ) -> Result<Self, CsrError> {
        let csr = Self {
            row_ptrs,
            col_indices,
            values,
            n,
        };
        csr.validate()?;
        Ok(csr)
    }

    fn validate(&self) -> Result<(), CsrError> {
        let n = self.n as usize;
        if self.row_ptrs.len() != n + 1 {
            return Err(CsrError::RowPtrsLenMismatch {
                expected: n + 1,
                got: self.row_ptrs.len(),
            });
        }
        if self.col_indices.len() != self.values.len() {
            return Err(CsrError::ColIndicesValuesLenMismatch {
                col_indices_len: self.col_indices.len(),
                values_len: self.values.len(),
            });
        }

        let row_ptr_last = as_usize(self.row_ptrs[n], IndexKind::RowPtr, n)?;
        if self.row_ptrs[0] != I::zero() {
            return Err(CsrError::RowPtrsMustStartAtZero {
                got: as_usize(self.row_ptrs[0], IndexKind::RowPtr, 0)?,
            });
        }
        if row_ptr_last != self.col_indices.len() {
            return Err(CsrError::RowPtrsEndMismatchNnz {
                row_ptr_end: row_ptr_last,
                nnz: self.col_indices.len(),
            });
        }

        // Both scans compare in `I`, so only the error arms convert an index to `usize`.
        for i in 0..n {
            if self.row_ptrs[i] > self.row_ptrs[i + 1] {
                return Err(CsrError::RowPtrsNotNonDecreasing {
                    row: i,
                    prev: as_usize(self.row_ptrs[i], IndexKind::RowPtr, i)?,
                    next: as_usize(self.row_ptrs[i + 1], IndexKind::RowPtr, i + 1)?,
                });
            }
        }

        // `None` means `I` cannot represent `n`, so every `I` value is below it.
        let limit = cast::<u32, I>(self.n);
        for (position, &col) in self.col_indices.iter().enumerate() {
            // Downstream reads columns in place as `usize`, which a negative one is not.
            if col < I::zero() {
                return Err(CsrError::IndexNotRepresentableAsUsize {
                    kind: IndexKind::ColIndex,
                    position,
                });
            }
            if limit.is_some_and(|limit| col >= limit) {
                return Err(CsrError::ColumnIndexOutOfBounds {
                    position,
                    col: as_usize(col, IndexKind::ColIndex, position)?,
                    n,
                });
            }
        }
        Ok(())
    }

    /// Row pointer array (length `n + 1`).
    #[inline]
    pub fn row_ptrs(&self) -> &'a [I] {
        self.row_ptrs
    }

    /// Column index array (length `nnz`).
    #[inline]
    pub fn col_indices(&self) -> &'a [I] {
        self.col_indices
    }

    /// Value array (length `nnz`).
    #[inline]
    pub fn values(&self) -> &'a [T] {
        self.values
    }

    /// Number of rows (and columns — the matrix is square).
    #[inline]
    pub fn n(&self) -> usize {
        self.n as usize
    }
}

impl<'a, T: Clone, I: PrimInt> CsrRef<'a, T, I> {
    /// Owned copy with `u32` indices; [`Error::InvalidCsr`] if an index does not fit.
    pub fn to_owned_u32(&self) -> Result<OwnedCsr<T, u32>, Error> {
        Ok(OwnedCsr {
            row_ptrs: cast_slice(self.row_ptrs, IndexKind::RowPtr)?,
            col_indices: cast_slice(self.col_indices, IndexKind::ColIndex)?,
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
    /// Owned CSR from `usize` arrays; [`Error::InvalidCsr`] if a value exceeds the index type.
    pub fn try_from_usize(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[T],
        n: usize,
    ) -> Result<Self, Error> {
        // `n` must fit `u32` to be stored, and `I` for `validate` to bounds-check columns against it.
        let n = u32::try_from(n)
            .ok()
            .filter(|&fits| cast::<u32, I>(fits).is_some())
            .ok_or(Error::InvalidCsr(
                CsrError::MatrixDimensionExceedsIndexType { n },
            ))?;

        let row_ptrs = cast_slice(row_ptrs, IndexKind::RowPtr)?;
        let col_indices = cast_slice(col_indices, IndexKind::ColIndex)?;

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
    /// Infallible: both constructors validate and the fields are private.
    pub fn as_csr_ref(&self) -> CsrRef<'_, T, I> {
        CsrRef {
            row_ptrs: &self.row_ptrs,
            col_indices: &self.col_indices,
            values: &self.values,
            n: self.n,
        }
    }
}

/// Lets `factorize(&owned)` work through the induced `TryFrom<Error = Infallible>`.
impl<'a, T, I: PrimInt> From<&'a OwnedCsr<T, I>> for CsrRef<'a, T, I> {
    fn from(owned: &'a OwnedCsr<T, I>) -> Self {
        owned.as_csr_ref()
    }
}

#[cfg(any(feature = "sprs", feature = "faer"))]
fn validate_square_dims(rows: usize, cols: usize) -> Result<u32, Error> {
    if rows != cols {
        return Err(Error::InvalidCsr(CsrError::ExpectedSquareMatrix {
            rows,
            cols,
        }));
    }
    u32::try_from(rows)
        .map_err(|_| Error::InvalidCsr(CsrError::MatrixDimensionExceedsIndexType { n: rows }))
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    type Error = Error;

    fn try_from(mat: sprs::CsMatViewI<'a, T, I>) -> Result<Self, Self::Error> {
        if !mat.is_csr() {
            return Err(Error::InvalidCsr(CsrError::ExpectedCsrMatrixGotCsc));
        }
        let n = validate_square_dims(mat.rows(), mat.cols())?;
        let (indptr, indices, data) = mat.into_raw_storage();
        CsrRef::new(indptr, indices, data, n)
    }
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> TryFrom<&'a sprs::CsMatI<T, I>> for CsrRef<'a, T, I> {
    type Error = Error;

    fn try_from(mat: &'a sprs::CsMatI<T, I>) -> Result<Self, Self::Error> {
        Self::try_from(mat.view())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(mat: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, Self::Error> {
        let n = validate_square_dims(mat.nrows(), mat.ncols())?;
        let symbolic = mat.symbolic();
        CsrRef::new(symbolic.row_ptr(), symbolic.col_idx(), mat.val(), n)
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<&'a faer::sparse::SparseRowMat<I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(mat: &'a faer::sparse::SparseRowMat<I, T>) -> Result<Self, Self::Error> {
        Self::try_from(mat.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_owned_u32_narrows_any_index_type_and_keeps_values() {
        let values = [1.0f64];
        let (wide_row_ptrs, wide_col_indices) = ([0usize, 1], [0usize]);
        let narrow = CsrRef::new(&[0u32, 1], &[0u32], &values, 1).expect("valid csr");
        let wide = CsrRef::new(&wide_row_ptrs, &wide_col_indices, &values, 1).expect("valid csr");

        for owned in [
            narrow.to_owned_u32().expect("u32 conversion"),
            wide.to_owned_u32().expect("usize conversion"),
        ] {
            let converted = owned.as_csr_ref();
            assert_eq!(converted.row_ptrs(), &[0u32, 1]);
            assert_eq!(converted.col_indices(), &[0u32]);
            assert_eq!(converted.values(), &values);
        }
    }

    #[test]
    fn owned_csr_borrows_into_csr_ref() {
        let (row_ptrs, col_indices, values, n) = crate::test_utils::path_laplacian_4();
        let owned = CsrRef::new(&row_ptrs, &col_indices, &values, n)
            .expect("valid csr")
            .to_owned_u32()
            .expect("to owned");

        let as_ref: CsrRef<'_, f64, u32> = (&owned).into();
        assert_eq!(as_ref.n(), 4);

        let factor = crate::factorize(&owned).expect("factorize &OwnedCsr");
        assert_eq!(factor.n(), 4);
    }
}
