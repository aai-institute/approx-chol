mod sddm;

use crate::{CsrError, IndexKind, NotSddm, Sddm};
use num_traits::{cast, PrimInt};

fn as_usize<I: PrimInt>(value: I, kind: IndexKind, position: usize) -> Result<usize, CsrError> {
    value
        .to_usize()
        .ok_or(CsrError::IndexNotRepresentableAsUsize { kind, position })
}

/// Borrowed CSR matrix view. Zero-copy from any CSR source (`sprs`, `faer`, or plain
/// `Vec`s); converts into an [`Sddm`].
#[derive(Debug, Clone, Copy)]
pub struct CsrRef<'a, T = f64, I = u32> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
    n: u32,
}

impl<'a, T, I: PrimInt> CsrRef<'a, T, I> {
    /// Construct a `CsrRef` with full validation. The only constructor, so every
    /// `CsrRef` that exists is structurally valid.
    ///
    /// # Errors
    ///
    /// The [`CsrError`] naming the violation when the arrays are not a structurally
    /// valid CSR of dimension `n`.
    pub fn new(
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

        // Both scans compare in `I`, so the happy path converts no index; only the
        // error arms need a `usize` for the payload.
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
            // A signed index type's negative column is no column at all.
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

#[cfg(any(feature = "sprs", feature = "faer"))]
fn validate_square_dims(rows: usize, cols: usize) -> Result<u32, CsrError> {
    if rows != cols {
        return Err(CsrError::ExpectedSquareMatrix { rows, cols });
    }
    u32::try_from(rows).map_err(|_| CsrError::MatrixDimensionExceedsIndexType { n: rows })
}

#[cfg(feature = "sprs")]
fn try_from_sprs_view_impl<'a, T, I: sprs::SpIndex + PrimInt>(
    mat: sprs::CsMatViewI<'a, T, I>,
) -> Result<CsrRef<'a, T, I>, CsrError> {
    if !mat.is_csr() {
        return Err(CsrError::ExpectedCsrMatrixGotCsc);
    }
    let n = validate_square_dims(mat.rows(), mat.cols())?;
    let (indptr, indices, data) = mat.into_raw_storage();
    CsrRef::new(indptr, indices, data, n)
}

#[cfg(feature = "faer")]
fn try_from_faer_view_impl<'a, T, I: faer::Index + PrimInt>(
    mat: faer::sparse::SparseRowMatRef<'a, I, T>,
) -> Result<CsrRef<'a, T, I>, CsrError> {
    let n = validate_square_dims(mat.nrows(), mat.ncols())?;
    let symbolic = mat.symbolic();
    CsrRef::new(symbolic.row_ptr(), symbolic.col_idx(), mat.val(), n)
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> TryFrom<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    type Error = CsrError;

    fn try_from(mat: sprs::CsMatViewI<'a, T, I>) -> Result<Self, Self::Error> {
        try_from_sprs_view_impl(mat)
    }
}

#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> TryFrom<&'a sprs::CsMatI<T, I>> for CsrRef<'a, T, I> {
    type Error = CsrError;

    fn try_from(mat: &'a sprs::CsMatI<T, I>) -> Result<Self, Self::Error> {
        try_from_sprs_view_impl(mat.view())
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<faer::sparse::SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    type Error = CsrError;

    fn try_from(mat: faer::sparse::SparseRowMatRef<'a, I, T>) -> Result<Self, Self::Error> {
        try_from_faer_view_impl(mat)
    }
}

#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> TryFrom<&'a faer::sparse::SparseRowMat<I, T>>
    for CsrRef<'a, T, I>
{
    type Error = CsrError;

    fn try_from(mat: &'a faer::sparse::SparseRowMat<I, T>) -> Result<Self, Self::Error> {
        try_from_faer_view_impl(mat.as_ref())
    }
}

/// The CSR path: the mirror check and the surplus-noise judgement live only here.
impl<'a, T, I> TryFrom<CsrRef<'a, T, I>> for Sddm<T>
where
    T: num_traits::Float + Send + Sync + 'static,
    I: PrimInt,
{
    type Error = NotSddm;

    fn try_from(csr: CsrRef<'a, T, I>) -> Result<Self, NotSddm> {
        sddm::from_csr(csr)
    }
}
