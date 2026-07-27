//! Error vocabulary shared across the crate.
//!
//! These types form the public error surface (re-exported at the crate root)
//! and the shared failure vocabulary that the low-level modules (`csr`,
//! `graph`, `ordering`) construct. Keeping them in one leaf module lets every
//! layer depend *down* to it rather than reaching up to the crate root.

use std::fmt;

/// Errors that can occur during approximate Cholesky factorization.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The input CSR matrix has inconsistent dimensions or invalid structure.
    InvalidCsr(CsrError),

    /// A coalesced off-diagonal entry is strictly positive, so the matrix is
    /// outside the SDDM/Laplacian class.
    PositiveOffDiagonal {
        /// `(row, column)` of the offending strictly-positive off-diagonal.
        edge: (usize, usize),
    },

    /// The factorization configuration is invalid.
    InvalidConfig(ConfigError),

    /// A matrix value is NaN or infinite.
    NonFiniteValue {
        /// Position in the CSR value array.
        position: usize,
    },

    /// Coalesced transpose entries are missing or unequal.
    Asymmetric {
        /// Canonical off-diagonal coordinate with `row < column`.
        edge: (usize, usize),
    },

    /// A row has negative diagonal surplus beyond the rounding tolerance.
    NotDiagonallyDominant {
        /// Row whose diagonal is smaller than its off-diagonal magnitude sum.
        row: usize,
    },

    /// A row's accumulated diagonal or magnitude sum overflowed, though every
    /// stored value is finite.
    NonFiniteRow {
        /// Row that overflowed.
        row: usize,
    },
}

/// Structured configuration errors returned by factorization setup.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// `split_merge` must be at least 1 when provided.
    SplitMergeMustBePositive {
        /// The invalid `split_merge` value provided by the caller.
        split_merge: u32,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SplitMergeMustBePositive { split_merge } => {
                write!(f, "split_merge must be >= 1 (got {split_merge})")
            }
        }
    }
}

/// Which CSR array an index belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// `row_ptrs` array.
    RowPtr,
    /// `col_indices` array.
    ColIndex,
}

impl fmt::Display for IndexKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowPtr => write!(f, "row_ptr"),
            Self::ColIndex => write!(f, "col_index"),
        }
    }
}

/// Structured CSR conversion and validation errors.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Column index is out of bounds.
    ColumnIndexOutOfBounds {
        /// Position in `col_indices`.
        position: usize,
        /// Out-of-bounds column value.
        col: usize,
        /// Matrix dimension.
        n: usize,
    },
    /// An index value (row pointer or column index) cannot be represented in
    /// the target integer type.
    IndexExceedsIndexType {
        /// Which CSR array the bad value came from.
        kind: IndexKind,
    },
    /// Matrix dimension `n` cannot be represented in the target integer type
    /// (internally `u32`).
    MatrixDimensionExceedsIndexType {
        /// Matrix dimension that does not fit.
        n: usize,
    },
    /// Expected CSR layout but received CSC.
    ExpectedCsrMatrixGotCsc,
    /// Expected square matrix.
    ExpectedSquareMatrix {
        /// Observed row count.
        rows: usize,
        /// Observed column count.
        cols: usize,
    },
    /// Input conversion via `TryFrom` panicked.
    InputConversionPanicked,
}

impl fmt::Display for CsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            Self::ExpectedCsrMatrixGotCsc => write!(f, "expected CSR matrix, got CSC"),
            Self::ExpectedSquareMatrix { rows, cols } => {
                write!(f, "expected square matrix (got {rows}x{cols})")
            }
            Self::InputConversionPanicked => write!(f, "input conversion panicked"),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidCsr(err) => write!(f, "invalid CSR matrix: {err}"),
            Error::PositiveOffDiagonal { edge: (row, col) } => write!(
                f,
                "off-diagonal ({row}, {col}) is positive; approx-chol requires SDDM/Laplacian input (off-diagonals must be <= 0)"
            ),
            Error::InvalidConfig(err) => write!(f, "invalid factorization config: {err}"),
            Error::NonFiniteValue { position } => {
                write!(f, "matrix value at CSR position {position} is not finite")
            }
            Error::Asymmetric { edge: (row, col) } => write!(
                f,
                "matrix is not symmetric at ({row}, {col}) and ({col}, {row})"
            ),
            Error::NotDiagonallyDominant { row } => write!(
                f,
                "row {row} is not diagonally dominant; approx-chol requires SDDM/Laplacian input"
            ),
            Error::NonFiniteRow { row } => write!(
                f,
                "row {row} sums to a non-finite diagonal or off-diagonal magnitude; approx-chol requires SDDM/Laplacian input"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<core::convert::Infallible> for Error {
    fn from(value: core::convert::Infallible) -> Self {
        match value {}
    }
}
