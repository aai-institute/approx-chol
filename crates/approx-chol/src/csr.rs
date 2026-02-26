use crate::Error;
use core::any::TypeId;
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

pub(crate) enum U32Csr<'a, T> {
    Borrowed(CsrRef<'a, T, u32>),
    Owned(OwnedCsr<T, u32>),
}

impl<T> U32Csr<'_, T> {
    #[inline]
    pub(crate) fn as_ref(&self) -> CsrRef<'_, T, u32> {
        match self {
            Self::Borrowed(csr) => {
                CsrRef::new_unchecked(csr.row_ptrs(), csr.col_indices(), csr.values(), csr.n)
            }
            Self::Owned(csr) => csr.as_ref(),
        }
    }
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
            return Err(Error::InvalidCsr("row_ptrs length != n + 1"));
        }
        if self.col_indices.len() != self.values.len() {
            return Err(Error::InvalidCsr(
                "col_indices and values have different lengths",
            ));
        }

        let row_ptr_last = self.row_ptrs[n].to_usize().ok_or(Error::InvalidCsr(
            "row_ptr value cannot be represented as usize",
        ))?;
        if row_ptr_last != self.col_indices.len() {
            return Err(Error::InvalidCsr("row_ptrs[n] != col_indices.len()"));
        }

        for i in 0..n {
            let a = self.row_ptrs[i].to_usize().ok_or(Error::InvalidCsr(
                "row_ptr value cannot be represented as usize",
            ))?;
            let b = self.row_ptrs[i + 1].to_usize().ok_or(Error::InvalidCsr(
                "row_ptr value cannot be represented as usize",
            ))?;
            if a > b {
                return Err(Error::InvalidCsr("row_ptrs is not non-decreasing"));
            }
        }

        for &col in self.col_indices {
            let col_usize = col.to_usize().ok_or(Error::InvalidCsr(
                "column index cannot be represented as usize",
            ))?;
            if col_usize >= self.n as usize {
                return Err(Error::InvalidCsr("column index out of bounds"));
            }
        }
        Ok(())
    }

    /// Construct a `CsrRef` without validation.
    pub fn new_unchecked(row_ptrs: &'a [I], col_indices: &'a [I], values: &'a [T], n: u32) -> Self {
        Self {
            row_ptrs,
            col_indices,
            values,
            n,
        }
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
    #[inline]
    pub fn row(&self, i: usize) -> (&'a [I], &'a [T]) {
        let start = self.row_ptrs[i]
            .to_usize()
            .expect("row_ptr value cannot be represented as usize");
        let end = self.row_ptrs[i + 1]
            .to_usize()
            .expect("row_ptr value cannot be represented as usize");
        (&self.col_indices[start..end], &self.values[start..end])
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
            .ok_or(Error::InvalidCsr("row_ptr exceeds u32::MAX"))?;
        let col_indices = self
            .col_indices
            .iter()
            .map(|&v| cast::<I, u32>(v))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidCsr("col_index exceeds u32::MAX"))?;
        Ok(OwnedCsr {
            row_ptrs,
            col_indices,
            values: self.values.to_vec(),
            n: self.n,
        })
    }
}

impl<'a, T: Clone, I: PrimInt + 'static> CsrRef<'a, T, I> {
    pub(crate) fn to_u32_fast_or_owned(&self) -> Result<U32Csr<'a, T>, Error> {
        if TypeId::of::<I>() == TypeId::of::<u32>() {
            // SAFETY: `TypeId` equality proves `I` is exactly `u32`, so the
            // slice metadata and element layout are identical.
            let row_ptrs = unsafe { &*(self.row_ptrs as *const [I] as *const [u32]) };
            // SAFETY: same argument as above.
            let col_indices = unsafe { &*(self.col_indices as *const [I] as *const [u32]) };
            return Ok(U32Csr::Borrowed(CsrRef::new_unchecked(
                row_ptrs,
                col_indices,
                self.values,
                self.n,
            )));
        }
        self.to_owned_u32().map(U32Csr::Owned)
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
        let _ = cast::<usize, I>(n).ok_or(Error::InvalidCsr("n exceeds target index type"))?;
        let n = u32::try_from(n).map_err(|_| Error::InvalidCsr("n exceeds u32::MAX"))?;

        let row_ptrs = row_ptrs
            .iter()
            .map(|&v| cast::<usize, I>(v))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidCsr("row_ptr exceeds target index type"))?;

        let col_indices = col_indices
            .iter()
            .map(|&v| cast::<usize, I>(v))
            .collect::<Option<Vec<_>>>()
            .ok_or(Error::InvalidCsr("col_index exceeds target index type"))?;

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
    pub fn as_ref(&self) -> CsrRef<'_, T, I> {
        CsrRef::new_unchecked(&self.row_ptrs, &self.col_indices, &self.values, self.n)
    }
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
        if !mat.is_csr() {
            return Err(Error::InvalidCsr("expected CSR matrix, got CSC"));
        }
        let n = mat.rows();
        if n != mat.cols() {
            return Err(Error::InvalidCsr("expected square matrix"));
        }
        let n =
            u32::try_from(n).map_err(|_| Error::InvalidCsr("matrix dimension exceeds u32::MAX"))?;
        let (indptr, indices, data) = mat.into_raw_storage();
        Ok(CsrRef::new_unchecked(indptr, indices, data, n))
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
        if mat.nrows() != mat.ncols() {
            return Err(Error::InvalidCsr("expected square matrix"));
        }
        let n = u32::try_from(mat.nrows())
            .map_err(|_| Error::InvalidCsr("matrix dimension exceeds u32::MAX"))?;
        let symbolic = mat.symbolic();
        Ok(CsrRef::new_unchecked(
            symbolic.row_ptr(),
            symbolic.col_idx(),
            mat.val(),
            n,
        ))
    }

    /// Fallible zero-copy conversion from a borrowed `faer` sparse row matrix.
    ///
    /// Returns [`Error::InvalidCsr`] with the same conditions as
    /// [`Self::try_from_faer_view`].
    pub fn try_from_faer(mat: &'a faer::sparse::SparseRowMat<I, T>) -> Result<Self, Error> {
        Self::try_from_faer_view(mat.as_ref())
    }
}

/// Zero-copy conversion from an `sprs` CSR matrix view.
///
/// Prefer [`CsrRef::try_from_sprs_view`] for panic-free conversion.
///
/// # Panics
///
/// Panics if the matrix is not in CSR format or is not square.
#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> From<sprs::CsMatViewI<'a, T, I>> for CsrRef<'a, T, I> {
    fn from(mat: sprs::CsMatViewI<'a, T, I>) -> Self {
        assert!(mat.is_csr(), "expected CSR matrix, got CSC");
        let n = mat.rows();
        assert_eq!(n, mat.cols(), "expected square matrix");
        let (indptr, indices, data) = mat.into_raw_storage();
        CsrRef::new_unchecked(
            indptr,
            indices,
            data,
            u32::try_from(n).expect("matrix dimension exceeds u32::MAX"),
        )
    }
}

/// Zero-copy conversion from a borrowed `sprs` CSR matrix.
///
/// Prefer [`CsrRef::try_from_sprs`] for panic-free conversion.
///
/// # Panics
///
/// Panics if the matrix is not in CSR format or is not square.
#[cfg(feature = "sprs")]
impl<'a, T, I: sprs::SpIndex + PrimInt> From<&'a sprs::CsMatI<T, I>> for CsrRef<'a, T, I> {
    fn from(mat: &'a sprs::CsMatI<T, I>) -> Self {
        mat.view().into()
    }
}

/// Zero-copy conversion from a `faer` CSR matrix view.
///
/// Prefer [`CsrRef::try_from_faer_view`] for panic-free conversion.
///
/// # Panics
///
/// Panics if the matrix is not square.
#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> From<faer::sparse::SparseRowMatRef<'a, I, T>>
    for CsrRef<'a, T, I>
{
    fn from(mat: faer::sparse::SparseRowMatRef<'a, I, T>) -> Self {
        assert_eq!(mat.nrows(), mat.ncols(), "expected square matrix");
        let symbolic = mat.symbolic();
        CsrRef::new_unchecked(
            symbolic.row_ptr(),
            symbolic.col_idx(),
            mat.val(),
            u32::try_from(mat.nrows()).expect("matrix dimension exceeds u32::MAX"),
        )
    }
}

/// Zero-copy conversion from a borrowed `faer` sparse row matrix.
///
/// Prefer [`CsrRef::try_from_faer`] for panic-free conversion.
///
/// # Panics
///
/// Panics if the matrix is not square.
#[cfg(feature = "faer")]
impl<'a, T, I: faer::Index + PrimInt> From<&'a faer::sparse::SparseRowMat<I, T>>
    for CsrRef<'a, T, I>
{
    fn from(mat: &'a faer::sparse::SparseRowMat<I, T>) -> Self {
        mat.as_ref().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_u32_fast_or_owned_borrows_for_u32() {
        let row_ptrs = [0u32, 1];
        let col_indices = [0u32];
        let values = [1.0f64];
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 1).expect("valid csr");

        let converted = csr.to_u32_fast_or_owned().expect("conversion");
        let U32Csr::Borrowed(csr_u32) = converted else {
            panic!("expected borrowed u32 fast path");
        };

        assert!(core::ptr::eq(
            csr_u32.row_ptrs().as_ptr(),
            row_ptrs.as_ptr()
        ));
        assert!(core::ptr::eq(
            csr_u32.col_indices().as_ptr(),
            col_indices.as_ptr()
        ));
    }

    #[test]
    fn to_u32_fast_or_owned_allocates_for_non_u32_indices() {
        let row_ptrs = [0usize, 1];
        let col_indices = [0usize];
        let values = [1.0f64];
        let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 1).expect("valid csr");

        let converted = csr.to_u32_fast_or_owned().expect("conversion");
        assert!(matches!(converted, U32Csr::Owned(_)));
    }
}
