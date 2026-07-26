use crate::{CsrError, Error, IndexKind};
use num_traits::{cast, PrimInt};

fn cast_slice<S: PrimInt, D: PrimInt>(src: &[S], kind: IndexKind) -> Result<Vec<D>, Error> {
    src.iter()
        .map(|&v| cast::<S, D>(v))
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::InvalidCsr(CsrError::IndexExceedsIndexType { kind }))
}

fn as_usize<I: PrimInt>(value: I, kind: IndexKind, position: usize) -> Result<usize, Error> {
    value
        .to_usize()
        .ok_or(Error::InvalidCsr(CsrError::IndexNotRepresentableAsUsize {
            kind,
            position,
        }))
}

/// Borrowed CSR matrix view. Zero-copy from any CSR source.
///
/// This is the primary input type for
/// [`Builder::build`](crate::low_level::Builder::build).
/// Construct from raw arrays owned by any sparse matrix library
/// (`sprs`, `faer`, or plain `Vec`s).
#[derive(Debug, Clone, Copy)]
pub struct CsrRef<'a, T = f64, I = u32> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
    n: u32,
}

/// `u32`-narrowed index arrays still carrying the source view's validated
/// invariants, so re-pairing them with values skips the `nnz` re-walk.
pub(crate) struct NarrowedCsr {
    row_ptrs: Vec<u32>,
    col_indices: Vec<u32>,
    n: u32,
}

impl NarrowedCsr {
    pub(crate) fn with_values<'a, T>(&'a self, values: &'a [T]) -> CsrRef<'a, T, u32> {
        CsrRef {
            row_ptrs: &self.row_ptrs,
            col_indices: &self.col_indices,
            values,
            n: self.n,
        }
    }
}

impl<'a, T, I: PrimInt> CsrRef<'a, T, I> {
    /// Construct a `CsrRef` with full validation. The only constructor, so every
    /// `CsrRef` that exists is structurally valid.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if the arrays are not a structurally valid
    /// CSR of dimension `n`; the [`CsrError`] variant names the violation.
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

    fn validate(&self) -> Result<(), Error> {
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

        let row_ptr_last = as_usize(self.row_ptrs[n], IndexKind::RowPtr, n)?;
        if self.row_ptrs[0] != I::zero() {
            return Err(Error::InvalidCsr(CsrError::RowPtrsMustStartAtZero {
                got: as_usize(self.row_ptrs[0], IndexKind::RowPtr, 0)?,
            }));
        }
        if row_ptr_last != self.col_indices.len() {
            return Err(Error::InvalidCsr(CsrError::RowPtrsEndMismatchNnz {
                row_ptr_end: row_ptr_last,
                nnz: self.col_indices.len(),
            }));
        }

        // Both scans compare in `I`, so the happy path converts no index; only the
        // error arms need a `usize` for the payload.
        for i in 0..n {
            if self.row_ptrs[i] > self.row_ptrs[i + 1] {
                return Err(Error::InvalidCsr(CsrError::RowPtrsNotNonDecreasing {
                    row: i,
                    prev: as_usize(self.row_ptrs[i], IndexKind::RowPtr, i)?,
                    next: as_usize(self.row_ptrs[i + 1], IndexKind::RowPtr, i + 1)?,
                }));
            }
        }

        // `None` means `I` cannot represent `n`, so every `I` value is below it.
        if let Some(limit) = cast::<u32, I>(self.n) {
            for (position, &col) in self.col_indices.iter().enumerate() {
                if col >= limit {
                    return Err(Error::InvalidCsr(CsrError::ColumnIndexOutOfBounds {
                        position,
                        col: as_usize(col, IndexKind::ColIndex, position)?,
                        n,
                    }));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn narrow_indices(&self) -> Result<NarrowedCsr, Error> {
        Ok(NarrowedCsr {
            row_ptrs: cast_slice(self.row_ptrs, IndexKind::RowPtr)?,
            col_indices: cast_slice(self.col_indices, IndexKind::ColIndex)?,
            n: self.n,
        })
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

impl<'a, T> CsrRef<'a, T, u32> {
    /// Each row's `(col_indices, values)`, in order. Infallible: pinning `I` to
    /// `u32` discharges the index conversion that [`validate`](Self::validate)
    /// has to check for a general `I`.
    pub(crate) fn rows(self) -> impl Iterator<Item = (&'a [u32], &'a [T])> {
        (0..self.n as usize).map(move |i| {
            let start = self.row_ptrs[i] as usize;
            let end = self.row_ptrs[i + 1] as usize;
            (&self.col_indices[start..end], &self.values[start..end])
        })
    }
}

impl<'a, T: Clone, I: PrimInt> CsrRef<'a, T, I> {
    /// Convert to an owned CSR with `u32` indices.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidCsr`] if any index does not fit in `u32`.
    pub fn to_owned_u32(&self) -> Result<OwnedCsr<T, u32>, Error> {
        let NarrowedCsr {
            row_ptrs,
            col_indices,
            n,
        } = self.narrow_indices()?;
        Ok(OwnedCsr {
            row_ptrs,
            col_indices,
            values: self.values.to_vec(),
            n,
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
        // `n` must fit `u32` to be stored and `I` for `validate` to bounds-check
        // columns against it at all.
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
    /// Borrow as a [`CsrRef`] for use with
    /// [`Builder::build`](crate::low_level::Builder::build).
    ///
    /// Infallible: both constructors validate and the fields are private, so
    /// there is nothing left to check.
    pub fn as_csr_ref(&self) -> CsrRef<'_, T, I> {
        CsrRef {
            row_ptrs: &self.row_ptrs,
            col_indices: &self.col_indices,
            values: &self.values,
            n: self.n,
        }
    }
}

/// Lets `&OwnedCsr` be used directly at the `TryInto<CsrRef>` entry point
/// (e.g. `factorize(&owned)`), like the zero-copy sparse conversions. The
/// blanket `TryFrom` this induces has `Error = Infallible`, which
/// [`Builder::build`](crate::low_level::Builder::build) already accepts.
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
fn try_from_sprs_view_impl<'a, T, I: sprs::SpIndex + PrimInt>(
    mat: sprs::CsMatViewI<'a, T, I>,
) -> Result<CsrRef<'a, T, I>, Error> {
    if !mat.is_csr() {
        return Err(Error::InvalidCsr(CsrError::ExpectedCsrMatrixGotCsc));
    }
    let n = validate_square_dims(mat.rows(), mat.cols())?;
    let (indptr, indices, data) = mat.into_raw_storage();
    CsrRef::new(indptr, indices, data, n)
}

#[cfg(feature = "faer")]
fn try_from_faer_view_impl<'a, T, I: faer::Index + PrimInt>(
    mat: faer::sparse::SparseRowMatRef<'a, I, T>,
) -> Result<CsrRef<'a, T, I>, Error> {
    let n = validate_square_dims(mat.nrows(), mat.ncols())?;
    let symbolic = mat.symbolic();
    CsrRef::new(symbolic.row_ptr(), symbolic.col_idx(), mat.val(), n)
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    type Error = Error;

    fn try_from(mat: sprs::CsMatViewI<'a, T, I>) -> Result<Self, Self::Error> {
        try_from_sprs_view_impl(mat)
    }
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> TryFrom<&'a sprs::CsMatI<T, I>> for CsrRef<'a, T, I> {
    type Error = Error;

    fn try_from(mat: &'a sprs::CsMatI<T, I>) -> Result<Self, Self::Error> {
        try_from_sprs_view_impl(mat.view())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = Error;

    fn try_from(mat: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, Self::Error> {
        try_from_faer_view_impl(mat)
    }
}

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
    use crate::test_utils::OrPanic;

    #[test]
    fn to_owned_u32_narrows_any_index_type_and_keeps_values() {
        let values = [1.0f64];
        let (wide_row_ptrs, wide_col_indices) = ([0usize, 1], [0usize]);
        let narrow = CsrRef::new(&[0u32, 1], &[0u32], &values, 1).or_panic("valid csr");
        let wide = CsrRef::new(&wide_row_ptrs, &wide_col_indices, &values, 1).or_panic("valid csr");

        for owned in [
            narrow.to_owned_u32().or_panic("u32 conversion"),
            wide.to_owned_u32().or_panic("usize conversion"),
        ] {
            let converted = owned.as_csr_ref();
            assert_eq!(converted.row_ptrs(), &[0u32, 1]);
            assert_eq!(converted.col_indices(), &[0u32]);
            assert_eq!(converted.values(), &values);
        }
    }

    #[test]
    fn owned_csr_borrows_into_csr_ref() {
        let (row_ptrs, col_indices, values) = crate::test_utils::path_laplacian_4();
        let owned = CsrRef::new(&row_ptrs, &col_indices, &values, 4)
            .or_panic("valid csr")
            .to_owned_u32()
            .or_panic("to owned");

        let as_ref: CsrRef<'_, f64, u32> = (&owned).into();
        assert_eq!(as_ref.n(), 4);

        // The `TryInto` bound at the `factorize` entry point still accepts it,
        // now through `Error = Infallible`.
        let factor = crate::factorize(&owned).or_panic("factorize &OwnedCsr");
        assert_eq!(factor.n(), 4);
    }
}
