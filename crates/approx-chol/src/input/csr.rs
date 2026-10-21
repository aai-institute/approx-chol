use std::borrow::Cow;

use num_traits::{cast, PrimInt};

use super::index;
use crate::in_range;
use crate::Real;

/// Borrowed CSR claiming a symmetric `A`; shape is checked here, the claim when it factors.
#[derive(Clone, Copy, Debug)]
pub struct CsrRef<'a, T = f64, I = u32> {
    row_ptrs: &'a [I],
    col_indices: &'a [I],
    values: &'a [T],
}

impl<'a, T, I: PrimInt> CsrRef<'a, T, I> {
    /// Checks the arrays' shape in 0.6.0's order, so no row slice can fail and each defect reports as it did.
    pub fn new(
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
        n: u32,
    ) -> Result<Self, CsrError> {
        let n = n as usize;
        if row_ptrs.len() != n + 1 {
            return Err(CsrError::RowPtrsLenMismatch {
                expected: n + 1,
                got: row_ptrs.len(),
            });
        }
        if col_indices.len() != values.len() {
            return Err(CsrError::ColIndicesValuesLenMismatch {
                col_indices_len: col_indices.len(),
                values_len: values.len(),
            });
        }
        let row_ptr_last = as_usize(row_ptrs[n], IndexKind::RowPtr, n)?;
        if row_ptrs[0] != I::zero() {
            return Err(CsrError::RowPtrsMustStartAtZero {
                got: as_usize(row_ptrs[0], IndexKind::RowPtr, 0)?,
            });
        }
        if row_ptr_last != col_indices.len() {
            return Err(CsrError::RowPtrsEndMismatchNnz {
                row_ptr_end: row_ptr_last,
                nnz: col_indices.len(),
            });
        }
        // Compared in `I`, so only the error arm converts a pointer to `usize`.
        if let Some(row) = (0..n).find(|&i| row_ptrs[i] > row_ptrs[i + 1]) {
            return Err(CsrError::RowPtrsNotNonDecreasing {
                row,
                prev: as_usize(row_ptrs[row], IndexKind::RowPtr, row)?,
                next: as_usize(row_ptrs[row + 1], IndexKind::RowPtr, row + 1)?,
            });
        }
        Ok(Self {
            row_ptrs,
            col_indices,
            values,
        })
    }

    /// Row pointer array (length `n + 1`).
    pub fn row_ptrs(&self) -> &'a [I] {
        self.row_ptrs
    }

    /// Column index array (length `nnz`).
    pub fn col_indices(&self) -> &'a [I] {
        self.col_indices
    }

    /// Value array (length `nnz`).
    pub fn values(&self) -> &'a [T] {
        self.values
    }

    /// Dimension of `A`.
    pub fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }
}

impl<T: Clone, I: PrimInt> CsrRef<'_, T, I> {
    /// Owned copy with `u32` indices; [`CsrError::IndexExceedsIndexType`] if an index does not fit.
    pub fn to_owned_u32(&self) -> Result<OwnedCsr<T, u32>, CsrError> {
        Ok(OwnedCsr {
            row_ptrs: cast_slice(self.row_ptrs, IndexKind::RowPtr)?,
            col_indices: cast_slice(self.col_indices, IndexKind::ColIndex)?,
            values: self.values.to_vec(),
        })
    }
}

/// Borrowed CSR of the upper triangle of a symmetric `A`, diagonal included.
#[derive(Clone, Copy, Debug)]
pub struct UpperCsrRef<'a, T = f64, I = u32>(pub(crate) CsrRef<'a, T, I>);

impl<'a, T, I: PrimInt> UpperCsrRef<'a, T, I> {
    /// The shape checks of [`CsrRef::new`]; a nonzero below the diagonal is caught when it factors.
    pub fn new(
        row_ptrs: &'a [I],
        col_indices: &'a [I],
        values: &'a [T],
        n: u32,
    ) -> Result<Self, CsrError> {
        CsrRef::new(row_ptrs, col_indices, values, n).map(Self)
    }
}

/// Owned CSR matrix, for sources that index in `usize`.
#[derive(Clone, Debug)]
pub struct OwnedCsr<T = f64, I = u32> {
    row_ptrs: Vec<I>,
    col_indices: Vec<I>,
    values: Vec<T>,
}

impl<T: Clone, I: PrimInt> OwnedCsr<T, I> {
    /// Owned CSR from `usize` arrays, shape-checked as [`CsrRef::new`]; an index past `I` is an error.
    pub fn try_from_usize(
        row_ptrs: &[usize],
        col_indices: &[usize],
        values: &[T],
        n: usize,
    ) -> Result<Self, CsrError> {
        let n = u32::try_from(n).map_err(|_| CsrError::MatrixDimensionExceedsIndexType { n })?;
        let row_ptrs = cast_slice(row_ptrs, IndexKind::RowPtr)?;
        let col_indices = cast_slice(col_indices, IndexKind::ColIndex)?;
        CsrRef::new(&row_ptrs, &col_indices, values, n)?;
        Ok(Self {
            row_ptrs,
            col_indices,
            values: values.to_vec(),
        })
    }
}

impl<T, I: PrimInt> OwnedCsr<T, I> {
    /// Infallible: the constructors check shape and the fields are private.
    pub fn as_csr_ref(&self) -> CsrRef<'_, T, I> {
        CsrRef {
            row_ptrs: &self.row_ptrs,
            col_indices: &self.col_indices,
            values: &self.values,
        }
    }
}

impl<'a, T, I: PrimInt> From<&'a OwnedCsr<T, I>> for CsrRef<'a, T, I> {
    fn from(owned: &'a OwnedCsr<T, I>) -> Self {
        owned.as_csr_ref()
    }
}

fn as_usize<I: PrimInt>(value: I, kind: IndexKind, position: usize) -> Result<usize, CsrError> {
    (value.to_usize()).ok_or(CsrError::IndexNotRepresentableAsUsize { kind, position })
}

/// Reserves up front: collecting into `Option<Vec<_>>` drops the size hint and cost 3.5x the traffic.
fn cast_slice<S: PrimInt, D: PrimInt>(src: &[S], kind: IndexKind) -> Result<Vec<D>, CsrError> {
    let mut out = Vec::with_capacity(src.len());
    for &value in src {
        out.push(cast::<S, D>(value).ok_or(CsrError::IndexExceedsIndexType { kind })?);
    }
    Ok(out)
}

/// Which CSR array an index belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexKind {
    /// `row_ptrs` array.
    RowPtr,
    /// `col_indices` array.
    ColIndex,
}

impl core::fmt::Display for IndexKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RowPtr => f.write_str("row_ptr"),
            Self::ColIndex => f.write_str("col_index"),
        }
    }
}

/// Strictly ascending columns per row; scipy already emits them, so only rare input pays for a copy.
pub(crate) struct Canonical<'a, T: Clone, I: Clone> {
    row_ptrs: Cow<'a, [I]>,
    col_indices: Cow<'a, [I]>,
    values: Cow<'a, [T]>,
}

impl<'a, T: Real, I: PrimInt> Canonical<'a, T, I> {
    /// Reads no value, sparing a stream: [`walk`](Self::walk) checks each as it reads it.
    pub(crate) fn of(csr: CsrRef<'a, T, I>) -> Result<Self, CsrError> {
        match is_canonical(&csr)? {
            true => Ok(Self {
                row_ptrs: Cow::Borrowed(csr.row_ptrs),
                col_indices: Cow::Borrowed(csr.col_indices),
                values: Cow::Borrowed(csr.values),
            }),
            false => rewrite(&csr),
        }
    }

    pub(crate) fn n(&self) -> usize {
        self.row_ptrs.len() - 1
    }

    fn arrays(&self) -> (&[I], &[I], &[T]) {
        (&self.row_ptrs, &self.col_indices, &self.values)
    }

    /// Row `i`'s canonical entries, stored zeros included.
    pub(crate) fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        let (row_ptrs, col_indices, values) = self.arrays();
        let span = index(row_ptrs[i])..index(row_ptrs[i + 1]);
        (col_indices[span.clone()].iter().map(|&j| index(j))).zip(values[span].iter().copied())
    }

    /// Row `i`'s stored entries above its diagonal, stored zeros included.
    pub(crate) fn upper_len(&self, i: usize) -> usize {
        let (row_ptrs, col_indices, _) = self.arrays();
        let row = &col_indices[index(row_ptrs[i])..index(row_ptrs[i + 1])];
        row.len() - row.partition_point(|&j| index(j) <= i)
    }

    /// Each nonzero `(i, j)`, `i < j`, mirrors checked by a per-row cursor (binary search: +6-15% build).
    pub(crate) fn walk(&self, mut edge: impl FnMut(usize, usize, T)) -> Result<(), CsrError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        let mut cursors = row_ptrs[..self.n()].to_vec();
        for row in 0..self.n() {
            let row_end = row_ptrs[row + 1];
            // Claimed like any mirror: claiming diagonals up front would skip those below.
            let diagonal = self.claim(&mut cursors, row, row)?;
            if !in_range(diagonal) {
                return Err(CsrError::OutOfRange { row });
            }
            let mut cursor = cursors[row];
            while cursor < row_end {
                let col = index(col_indices[index(cursor)]);
                let upper = values[index(cursor)];
                cursor = cursor + I::one();
                if !in_range(upper) {
                    return Err(CsrError::OutOfRange { row });
                }
                // Duplicates can coalesce to exactly zero, which contributes no edge.
                if upper == T::zero() {
                    continue;
                }
                let lower = self.claim(&mut cursors, col, row)?;
                if !approximately_equal(upper, lower) {
                    return Err(CsrError::NotSymmetric { row });
                }
                edge(row, col, upper);
            }
        }
        Ok(())
    }

    /// Advances row `row`'s cursor to `col`; a merge-join only because each entry is claimed once. Stored zeros count as absent.
    fn claim(&self, cursors: &mut [I], row: usize, col: usize) -> Result<T, CsrError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        let row_end = row_ptrs[row + 1];
        let mut cursor = cursors[row];
        let mut found = T::zero();
        while cursor < row_end {
            let at = index(col_indices[index(cursor)]);
            if at > col {
                break;
            }
            let value = values[index(cursor)];
            if !value.is_finite() {
                return Err(CsrError::OutOfRange { row });
            }
            cursor = cursor + I::one();
            if at == col {
                found = value;
                break;
            }
            // Skipped a stored entry whose own mirror above the diagonal is missing.
            if value != T::zero() {
                cursors[row] = cursor;
                return Err(CsrError::NotSymmetric { row });
            }
        }
        cursors[row] = cursor;
        Ok(found)
    }

    /// Each nonzero `(i, j)`, `i < j`, of upper storage, symmetric by shape: no mirror is found, only stored zeros admitted below the diagonal.
    pub(crate) fn walk_upper(&self, mut edge: impl FnMut(usize, usize, T)) -> Result<(), CsrError> {
        let (row_ptrs, col_indices, values) = self.arrays();
        for row in 0..self.n() {
            let span = index(row_ptrs[row])..index(row_ptrs[row + 1]);
            let columns = &col_indices[span.clone()];
            let values = &values[span];
            // Columns ascend once canonical, so only a row whose first lies below the diagonal pays the scan.
            if columns.first().is_some_and(|&j| index(j) < row)
                && stores_below(columns, values, row)
            {
                return Err(CsrError::BelowDiagonal { row });
            }
            for (&col, &value) in columns.iter().zip(values) {
                if !in_range(value) {
                    return Err(CsrError::OutOfRange { row });
                }
                let col = index(col);
                if col > row && value != T::zero() {
                    edge(row, col, value);
                }
            }
        }
        Ok(())
    }
}

#[cold]
fn stores_below<T: Real, I: PrimInt>(columns: &[I], values: &[T], row: usize) -> bool {
    (columns.iter().zip(values))
        .take_while(|&(&j, _)| index(j) < row)
        .any(|(_, &value)| value != T::zero())
}

/// Mirrors within 8 ulps of the larger are equal: scipy's assembly rounds the two triangles apart.
fn approximately_equal<T: Real>(left: T, right: T) -> bool {
    if left == right {
        return true;
    }
    let ulps = T::from(8.0).unwrap_or_else(T::one);
    let scale = left.abs().max(right.abs());
    (left - right).abs() <= ulps * T::epsilon() * scale
}

/// Accumulated, not short-circuited: the `all` form measured 0.4-2% slower.
fn is_canonical<T, I: PrimInt>(csr: &CsrRef<'_, T, I>) -> Result<bool, CsrError> {
    // `None` means `I` cannot represent `n`, so every `I` value is below it.
    let limit = I::from(csr.n());
    let mut canonical = true;
    let outside = |j: I| j < I::zero() || limit.is_some_and(|limit| j >= limit);
    for bounds in csr.row_ptrs.windows(2) {
        let start = index(bounds[0]);
        let columns = &csr.col_indices[start..index(bounds[1])];
        if let Some(at) = columns.iter().position(|&j| outside(j)) {
            return Err(column_error(start + at, columns[at], csr.n()));
        }
        for pair in columns.windows(2) {
            canonical &= pair[0] < pair[1];
        }
    }
    Ok(canonical)
}

/// 0.6.0's report of a column outside `0..n`; built only on the error path.
#[cold]
fn column_error<I: PrimInt>(position: usize, col: I, n: usize) -> CsrError {
    match (col >= I::zero()).then(|| col.to_usize()).flatten() {
        Some(col) => CsrError::ColumnIndexOutOfBounds { position, col, n },
        None => CsrError::IndexNotRepresentableAsUsize {
            kind: IndexKind::ColIndex,
            position,
        },
    }
}

/// Only non-canonical input pays this copy, in its own index type.
fn rewrite<'a, T: Real, I: PrimInt>(
    csr: &CsrRef<'_, T, I>,
) -> Result<Canonical<'a, T, I>, CsrError> {
    let nnz = csr.col_indices.len();
    let mut row_ptrs = Vec::with_capacity(csr.row_ptrs.len());
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut entries: Vec<(I, T)> = Vec::new();
    row_ptrs.push(I::zero());
    for (row, bounds) in csr.row_ptrs.windows(2).enumerate() {
        let (from, to) = (index(bounds[0]), index(bounds[1]));
        entries.clear();
        entries.extend(
            csr.col_indices[from..to]
                .iter()
                .copied()
                .zip(csr.values[from..to].iter().copied()),
        );
        // One row's degree, not nnz. Stable, so duplicates sum in stored order.
        entries.sort_by_key(|&(col, _)| col);
        for group in entries.chunk_by(|left, right| left.0 == right.0) {
            let folded = group[1..].iter().fold(group[0].1, |sum, &(_, v)| sum + v);
            // Only this fold can overflow; caught here so downstream stays all-finite.
            if !folded.is_finite() {
                return Err(CsrError::OutOfRange { row });
            }
            col_indices.push(group[0].0);
            values.push(folded);
        }
        // At most the caller's own `nnz`, which its last row pointer already holds in `I`.
        row_ptrs.push(I::from(col_indices.len()).expect("a coalesced count fits the input's type"));
    }
    Ok(Canonical {
        row_ptrs: Cow::Owned(row_ptrs),
        col_indices: Cow::Owned(col_indices),
        values: Cow::Owned(values),
    })
}

/// Why a CSR is not a symmetric matrix; the variants 0.6.0 had come first, in its order.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsrError {
    /// `row_ptrs.len()` does not equal `n + 1`.
    RowPtrsLenMismatch {
        /// Expected `row_ptrs` length (`n + 1`).
        expected: usize,
        /// Actual `row_ptrs` length.
        got: usize,
    },
    /// `col_indices.len()` does not equal `values.len()`.
    ColIndicesValuesLenMismatch {
        /// Length of the column-index array.
        col_indices_len: usize,
        /// Length of the values array.
        values_len: usize,
    },
    /// An index value cannot be represented as `usize`.
    IndexNotRepresentableAsUsize {
        /// Which CSR array the bad value came from.
        kind: IndexKind,
        /// Position in the source array.
        position: usize,
    },
    /// `row_ptrs[0]` must be zero.
    RowPtrsMustStartAtZero {
        /// The observed non-zero start pointer.
        got: usize,
    },
    /// `row_ptrs[n]` must match nnz.
    RowPtrsEndMismatchNnz {
        /// Value of `row_ptrs[n]`.
        row_ptr_end: usize,
        /// Number of non-zeros (`col_indices.len()`).
        nnz: usize,
    },
    /// `row_ptrs` must be non-decreasing.
    RowPtrsNotNonDecreasing {
        /// Row index `i` where `row_ptrs[i] > row_ptrs[i + 1]`.
        row: usize,
        /// Value of `row_ptrs[i]`.
        prev: usize,
        /// Value of `row_ptrs[i + 1]`.
        next: usize,
    },
    /// Column index is out of bounds; found when the input factors.
    ColumnIndexOutOfBounds {
        /// Position in `col_indices`.
        position: usize,
        /// Out-of-bounds column value.
        col: usize,
        /// Matrix dimension.
        n: usize,
    },
    /// An index does not fit the target index type.
    IndexExceedsIndexType {
        /// Which array the index came from.
        kind: IndexKind,
    },
    /// The dimension does not fit `u32`.
    MatrixDimensionExceedsIndexType {
        /// The dimension.
        n: usize,
    },
    /// A converted matrix is not square.
    ExpectedSquareMatrix {
        /// Its row count.
        rows: usize,
        /// Its column count.
        cols: usize,
    },
    /// An entry has no mirror within 8 ulps.
    NotSymmetric {
        /// The offending row.
        row: usize,
    },
    /// An upper triangle stores a nonzero entry, or a repeated column's nonzero sum, below its diagonal.
    BelowDiagonal {
        /// The offending row.
        row: usize,
    },
    /// An entry or a repeated column's sum is not finite, or nonzero below `MIN_POSITIVE/EPSILON`.
    OutOfRange {
        /// The offending row.
        row: usize,
    },
    /// A converted matrix's row leaves unused slots after its entries.
    RowGap {
        /// The first row with unused slots.
        row: usize,
    },
}

impl core::fmt::Display for CsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::RowPtrsLenMismatch { expected, got } => write!(
                f,
                "row_ptrs length != n + 1 (expected {expected}, got {got})"
            ),
            Self::ColIndicesValuesLenMismatch {
                col_indices_len,
                values_len,
            } => write!(
                f,
                "col_indices and values have different lengths ({col_indices_len} != {values_len})"
            ),
            Self::IndexNotRepresentableAsUsize { kind, position } => write!(
                f,
                "{kind} value at position {position} cannot be represented as usize"
            ),
            Self::RowPtrsMustStartAtZero { got } => write!(f, "row_ptrs[0] must be 0 (got {got})"),
            Self::RowPtrsEndMismatchNnz { row_ptr_end, nnz } => {
                write!(f, "row_ptrs[n] must equal nnz ({row_ptr_end} != {nnz})")
            }
            Self::RowPtrsNotNonDecreasing { row, prev, next } => write!(
                f,
                "row_ptrs is not non-decreasing at row {row}: {prev} > {next}"
            ),
            Self::ColumnIndexOutOfBounds { position, col, n } => write!(
                f,
                "column index out of bounds at position {position}: {col} >= {n}"
            ),
            Self::IndexExceedsIndexType { kind } => {
                write!(f, "{kind} exceeds target index type capacity")
            }
            Self::MatrixDimensionExceedsIndexType { n } => {
                write!(f, "matrix dimension exceeds index type capacity (n={n})")
            }
            Self::ExpectedSquareMatrix { rows, cols } => {
                write!(f, "expected square matrix (got {rows}x{cols})")
            }
            Self::NotSymmetric { row } => write!(f, "row {row} has an entry with no equal mirror"),
            Self::BelowDiagonal { row } => write!(
                f,
                "row {row} of an upper triangle has an entry below the diagonal"
            ),
            Self::OutOfRange { row } => {
                write!(f, "row {row} has an entry outside the representable range")
            }
            Self::RowGap { row } => write!(f, "row {row} leaves unused slots after its entries"),
        }
    }
}

impl std::error::Error for CsrError {}
