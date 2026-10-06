use std::fmt;

/// Why a [`CsrRef`](crate::CsrRef) is not an [`Sddm`](crate::Sddm).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotSddm {
    /// `n` is `u32::MAX` or more, leaving no index for a ground vertex.
    DimensionTooLarge {
        /// Matrix dimension.
        n: usize,
    },
    /// More stored entries than `u32` positions.
    TooManyNonzeros {
        /// Stored entries.
        nnz: usize,
    },
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
    /// A coalesced off-diagonal entry is strictly positive.
    PositiveOffDiagonal {
        /// `(row, column)` of the offending entry.
        edge: (usize, usize),
    },
    /// A row's diagonal falls short of its off-diagonal magnitude beyond rounding.
    NotDiagonallyDominant {
        /// The deficient row.
        row: usize,
    },
    /// A row's diagonal or off-diagonal magnitude sums to a non-finite value.
    NonFiniteRow {
        /// The row.
        row: usize,
    },
    /// The diagonal surplus total is not finite.
    SurplusOverflow,
}

impl fmt::Display for NotSddm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionTooLarge { n } => {
                write!(f, "matrix dimension {n} leaves no u32 index for a ground vertex")
            }
            Self::TooManyNonzeros { nnz } => write!(f, "{nnz} stored entries exceed u32"),
            Self::NonFiniteValue { position } => {
                write!(f, "matrix value at CSR position {position} is not finite")
            }
            Self::Asymmetric { edge: (row, col) } => write!(
                f,
                "matrix is not symmetric at ({row}, {col}) and ({col}, {row})"
            ),
            Self::PositiveOffDiagonal { edge: (row, col) } => write!(
                f,
                "off-diagonal ({row}, {col}) is positive; approx-chol requires SDDM/Laplacian input (off-diagonals must be <= 0)"
            ),
            Self::NotDiagonallyDominant { row } => write!(
                f,
                "row {row} is not diagonally dominant; approx-chol requires SDDM/Laplacian input"
            ),
            Self::NonFiniteRow { row } => write!(
                f,
                "row {row} sums to a non-finite diagonal or off-diagonal magnitude; approx-chol requires SDDM/Laplacian input"
            ),
            Self::SurplusOverflow => write!(f, "diagonal surplus total is not finite"),
        }
    }
}

impl std::error::Error for NotSddm {}

/// Why arrays are not a [`Laplacian`](crate::Laplacian).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaplacianError {
    /// The arrays are not a CSR matrix.
    Structure(CsrError),
    /// `u32::MAX` or more vertices, leaving no index for a ground vertex.
    TooManyVertices {
        /// Vertex count.
        n: usize,
    },
    /// A row lists a neighbor at or below its own index.
    NotStrictlyUpper {
        /// `(row, neighbor)` with `neighbor <= row`.
        edge: (usize, usize),
    },
    /// A row's neighbors are not strictly ascending.
    UnsortedNeighbors {
        /// Row with a repeated or out-of-order neighbor.
        row: usize,
    },
    /// An edge weight is not finite and positive.
    InvalidWeight {
        /// `(row, neighbor)` of the offending edge.
        edge: (usize, usize),
    },
    /// A weighted degree that is not finite.
    DegreeOverflow {
        /// The vertex.
        vertex: usize,
    },
}

impl fmt::Display for LaplacianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structure(err) => write!(f, "invalid Laplacian adjacency: {err}"),
            Self::TooManyVertices { n } => {
                write!(f, "{n} vertices leave no u32 index for a ground vertex")
            }
            Self::NotStrictlyUpper { edge: (row, col) } => {
                write!(
                    f,
                    "Laplacian row {row} lists neighbor {col}, which is not above it"
                )
            }
            Self::UnsortedNeighbors { row } => {
                write!(
                    f,
                    "Laplacian row {row} neighbors are not strictly ascending"
                )
            }
            Self::InvalidWeight { edge: (row, col) } => write!(
                f,
                "Laplacian edge ({row}, {col}) has a weight that is not finite and positive"
            ),
            Self::DegreeOverflow { vertex } => {
                write!(f, "vertex {vertex}'s weighted degree is not finite")
            }
        }
    }
}

impl std::error::Error for LaplacianError {}

/// Why a surplus does not ground a [`Laplacian`](crate::Laplacian).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroundedError {
    /// A surplus count other than the vertex count.
    LengthMismatch {
        /// The Laplacian's vertex count.
        expected: usize,
        /// The surplus count given.
        got: usize,
    },
    /// A surplus that is negative or not finite.
    InvalidSurplus {
        /// Vertex carrying it.
        vertex: usize,
    },
    /// Zero everywhere: that is a [`Laplacian`](crate::Laplacian).
    NoSurplus,
    /// The surplus total is not finite.
    SurplusOverflow,
    /// A diagonal entry, weighted degree plus surplus, that is not finite.
    DiagonalOverflow {
        /// The vertex.
        vertex: usize,
    },
}

impl fmt::Display for GroundedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, got } => {
                write!(f, "expected {expected} surplus entries, got {got}")
            }
            Self::InvalidSurplus { vertex } => {
                write!(f, "surplus at vertex {vertex} is negative or not finite")
            }
            Self::NoSurplus => write!(f, "surplus is zero everywhere, which is a Laplacian"),
            Self::SurplusOverflow => write!(f, "surplus total is not finite"),
            Self::DiagonalOverflow { vertex } => {
                write!(f, "diagonal at vertex {vertex} is not finite")
            }
        }
    }
}

impl std::error::Error for GroundedError {}

/// An unusable exact dense Cholesky pivot, reported as a [`Fallback`](crate::Fallback) or an error.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnusablePivot {
    /// The failing pivot's vertex, in the numbering of the factorized input.
    pub vertex: usize,
    /// Why the pivot was unusable.
    pub failure: DenseFailure,
}

impl fmt::Display for UnusablePivot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vertex {}: {}", self.vertex, self.failure)
    }
}

impl std::error::Error for UnusablePivot {}

/// Why an exact dense Cholesky pivot was unusable.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseFailure {
    /// The updated diagonal was zero or negative.
    NonPositivePivot,
    /// The updated diagonal was NaN or infinite.
    NonFinitePivot,
}

impl DenseFailure {
    /// The one definition, so build and deserialize-validate cannot disagree.
    pub(crate) fn of<T: num_traits::Float>(pivot: T) -> Option<Self> {
        if !pivot.is_finite() {
            Some(Self::NonFinitePivot)
        } else if pivot <= T::zero() {
            Some(Self::NonPositivePivot)
        } else {
            None
        }
    }
}

impl fmt::Display for DenseFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositivePivot => write!(f, "pivot is zero or negative"),
            Self::NonFinitePivot => write!(f, "pivot is not finite"),
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
    /// Matrix dimension `n` does not fit the internal `u32` index type.
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
            Self::MatrixDimensionExceedsIndexType { n } => {
                write!(f, "matrix dimension exceeds index type capacity (n={n})")
            }
            Self::ExpectedCsrMatrixGotCsc => write!(f, "expected CSR matrix, got CSC"),
            Self::ExpectedSquareMatrix { rows, cols } => {
                write!(f, "expected square matrix (got {rows}x{cols})")
            }
        }
    }
}

impl std::error::Error for CsrError {}
