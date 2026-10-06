use std::fmt;

/// Why a [`CsrRef`](crate::CsrRef) is not an [`Sddm`](crate::Sddm).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotSddm {
    /// `n` is `u32::MAX`, leaving no index for a ground vertex.
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
    /// A nonzero entry's magnitude is below `MIN_POSITIVE / EPSILON`, the measured floor of accurate solves.
    MagnitudeTooSmall {
        /// `(row, column)` of the entry, the column canonical with `row <= column`.
        entry: (usize, usize),
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
            Self::MagnitudeTooSmall { entry: (row, col) } => write!(
                f,
                "entry ({row}, {col}) is below MIN_POSITIVE / EPSILON of the scalar type; scale the matrix up"
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
    /// An edge weight is below `MIN_POSITIVE / EPSILON`, the measured floor of accurate solves.
    WeightTooSmall {
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
            Self::NotStrictlyUpper { edge: (row, col) } => write!(
                f,
                "Laplacian row {row} lists neighbor {col}, which is not above it"
            ),
            Self::UnsortedNeighbors { row } => {
                write!(f, "Laplacian row {row} neighbors are not strictly ascending")
            }
            Self::InvalidWeight { edge: (row, col) } => write!(
                f,
                "Laplacian edge ({row}, {col}) has a weight that is not finite and positive"
            ),
            Self::WeightTooSmall { edge: (row, col) } => write!(
                f,
                "Laplacian edge ({row}, {col}) is below MIN_POSITIVE / EPSILON of the scalar type; scale the weights up"
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
    /// A surplus that is negative, subnormal or not finite.
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
    /// A positive diagonal entry below `MIN_POSITIVE / EPSILON`, the measured floor of accurate solves.
    DiagonalTooSmall {
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
            Self::InvalidSurplus { vertex } => write!(
                f,
                "surplus at vertex {vertex} is negative, subnormal or not finite"
            ),
            Self::NoSurplus => write!(f, "surplus is zero everywhere, which is a Laplacian"),
            Self::SurplusOverflow => write!(f, "surplus total is not finite"),
            Self::DiagonalOverflow { vertex } => {
                write!(f, "diagonal at vertex {vertex} is not finite")
            }
            Self::DiagonalTooSmall { vertex } => write!(
                f,
                "diagonal at vertex {vertex} is below MIN_POSITIVE / EPSILON of the scalar type; scale the matrix up"
            ),
        }
    }
}

impl std::error::Error for GroundedError {}

/// Errors that can occur during approximate Cholesky factorization.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The input CSR matrix has inconsistent dimensions or invalid structure.
    InvalidCsr(CsrError),

    /// A coalesced off-diagonal entry is strictly positive: the matrix is not SDDM.
    PositiveOffDiagonal {
        /// `(row, column)` of the offending strictly-positive off-diagonal.
        edge: (usize, usize),
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

    /// A row has negative diagonal surplus beyond the rounding tolerance.
    NotDiagonallyDominant {
        /// Row whose diagonal is smaller than its off-diagonal magnitude sum.
        row: usize,
    },

    /// A row's accumulated diagonal or magnitude sum overflowed.
    NonFiniteRow {
        /// Row that overflowed.
        row: usize,
    },

    /// A nonzero entry's magnitude is below `MIN_POSITIVE / EPSILON`, the measured floor of accurate solves.
    MagnitudeTooSmall {
        /// `(row, column)` of the entry, the column canonical with `row <= column`.
        entry: (usize, usize),
    },

    /// The diagonal surplus total is not finite.
    SurplusOverflow,

    /// Exact dense Cholesky hit an unusable pivot and [`ExactFailure::Error`](crate::ExactFailure::Error) asked for that to fail.
    DenseFactorizationFailed(UnusablePivot),
}

impl From<NotSddm> for Error {
    fn from(err: NotSddm) -> Self {
        match err {
            NotSddm::DimensionTooLarge { n } => {
                Self::InvalidCsr(CsrError::MatrixDimensionExceedsIndexType {
                    n: n.saturating_add(1),
                })
            }
            NotSddm::TooManyNonzeros { .. } => Self::InvalidCsr(CsrError::IndexExceedsIndexType {
                kind: IndexKind::RowPtr,
            }),
            NotSddm::NonFiniteValue { position } => Self::NonFiniteValue { position },
            NotSddm::Asymmetric { edge } => Self::Asymmetric { edge },
            NotSddm::PositiveOffDiagonal { edge } => Self::PositiveOffDiagonal { edge },
            NotSddm::NotDiagonallyDominant { row } => Self::NotDiagonallyDominant { row },
            NotSddm::NonFiniteRow { row } => Self::NonFiniteRow { row },
            NotSddm::MagnitudeTooSmall { entry } => Self::MagnitudeTooSmall { entry },
            NotSddm::SurplusOverflow => Self::SurplusOverflow,
        }
    }
}

/// An unusable exact pivot, reported as a [`Fallback`] or raised as [`Error::DenseFactorizationFailed`].
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why a block [`Backend::ExactBelow`](crate::Backend::ExactBelow) claimed was factored approximately.
pub enum Fallback {
    /// Dense elimination reached a pivot it could not use.
    InvalidPivot(UnusablePivot),
    /// The dense copy would not fit in memory; never fatal, whatever [`ExactFailure`](crate::ExactFailure) says.
    WillNotFit {
        /// Variables the block solves for, so the copy is `dim * dim` scalars.
        dim: usize,
    },
}

impl fmt::Display for Fallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPivot(pivot) => write!(f, "{pivot}"),
            Self::WillNotFit { dim } => write!(f, "{dim} variables do not fit in memory"),
        }
    }
}

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
    /// A row pointer or column index does not fit the target integer type.
    IndexExceedsIndexType {
        /// Which CSR array the bad value came from.
        kind: IndexKind,
    },
    /// Matrix dimension `n` does not fit the target integer type (internally `u32`).
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
            Error::MagnitudeTooSmall { entry: (row, col) } => write!(
                f,
                "entry ({row}, {col}) is below MIN_POSITIVE / EPSILON of the scalar type; scale the matrix up"
            ),
            Error::SurplusOverflow => write!(f, "diagonal surplus total is not finite"),
            Error::DenseFactorizationFailed(pivot) => {
                write!(f, "exact dense Cholesky failed at {pivot}")
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<core::convert::Infallible> for Error {
    fn from(value: core::convert::Infallible) -> Self {
        match value {}
    }
}
