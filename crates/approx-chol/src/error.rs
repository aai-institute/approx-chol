use std::fmt;

/// Why a [`CsrRef`](crate::CsrRef) is not an SDDM matrix, from [`Sddm::try_from`](crate::Sddm).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
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

    /// A nonzero entry's magnitude, or a diagonal's surplus, is below `MIN_POSITIVE / EPSILON`, the measured floor of accurate solves.
    MagnitudeTooSmall {
        /// `(row, column)` of the entry, the column canonical with `row <= column`.
        entry: (usize, usize),
    },

    /// A component's ground, the sum of its diagonal surplus, is not finite.
    GroundOverflow {
        /// The component's lowest vertex.
        vertex: usize,
    },
}

/// Why an edge weight cannot be stored.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDefect {
    /// NaN or infinite.
    NonFinite,
    /// Zero or negative.
    NotPositive,
    /// Below `MIN_POSITIVE / EPSILON`, the measured floor of accurate solves.
    BelowFloor,
}

/// Why the arrays given to [`Laplacian::new`](crate::Laplacian::new) are not a strict upper adjacency.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdjacencyError {
    /// `row_ptrs` is empty, so it does not even name zero rows.
    RowPtrsEmpty,
    /// `row_ptrs[0]` is not zero.
    RowPtrsMustStartAtZero {
        /// The observed start.
        got: u32,
    },
    /// `neighbors` and `weights` differ in length.
    NeighborsWeightsLenMismatch {
        /// Length of `neighbors`.
        neighbors: usize,
        /// Length of `weights`.
        weights: usize,
    },
    /// The last row pointer is not the length of `neighbors`.
    RowPtrsEndMismatch {
        /// The last row pointer.
        end: u32,
        /// Length of `neighbors`.
        len: usize,
    },
    /// `row_ptrs[row] > row_ptrs[row + 1]`.
    RowPtrsDecrease {
        /// The row whose end precedes its start.
        row: usize,
    },
    /// A ground slot after the vertices would not fit `u32`.
    TooManyVertices {
        /// The number of vertices.
        n: usize,
    },
    /// A neighbor is not above the diagonal.
    NotStrictlyUpper {
        /// `(row, neighbor)` with `neighbor <= row`.
        edge: (usize, usize),
    },
    /// A row's neighbors are not strictly ascending.
    Unsorted {
        /// `(row, neighbor)` at or before its predecessor.
        edge: (usize, usize),
    },
    /// A neighbor is not a vertex.
    NeighborOutOfBounds {
        /// `(row, neighbor)` with `neighbor >= n`.
        edge: (usize, usize),
        /// The number of vertices.
        n: usize,
    },
    /// An edge weight cannot be stored.
    Weight {
        /// `(row, neighbor)` of the edge.
        edge: (usize, usize),
        /// What is wrong with its weight.
        defect: WeightDefect,
    },
}

/// Why [`Laplacian::new`](crate::Laplacian::new) rejected its arrays.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaplacianError {
    /// The arrays are not a strict upper adjacency.
    Adjacency(AdjacencyError),
    /// A vertex's weighted degree, its diagonal, is not finite.
    DegreeNotFinite {
        /// The vertex.
        vertex: usize,
    },
}

/// Why a diagonal surplus cannot be stored.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurplusDefect {
    /// Negative.
    Negative,
    /// Positive but below `MIN_POSITIVE / EPSILON`, the measured floor of accurate solves.
    BelowFloor,
}

/// Why [`Sddm::new`](crate::Sddm::new) rejected its arrays.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SddmError {
    /// The arrays are not a strict upper adjacency.
    Adjacency(AdjacencyError),
    /// The surplus does not have one entry per vertex.
    SurplusLength {
        /// Length of the surplus.
        len: usize,
        /// The number of vertices.
        n: usize,
    },
    /// A vertex's surplus cannot be stored.
    Surplus {
        /// The vertex.
        vertex: usize,
        /// What is wrong with its surplus.
        defect: SurplusDefect,
    },
    /// A vertex's diagonal, its weighted degree plus its surplus, is not finite.
    DiagonalNotFinite {
        /// The vertex.
        vertex: usize,
    },
    /// A component's ground, the sum of its surplus, is not finite.
    GroundOverflow {
        /// The component's lowest vertex.
        vertex: usize,
    },
}

impl From<AdjacencyError> for LaplacianError {
    fn from(error: AdjacencyError) -> Self {
        Self::Adjacency(error)
    }
}

impl From<AdjacencyError> for SddmError {
    fn from(error: AdjacencyError) -> Self {
        Self::Adjacency(error)
    }
}

impl fmt::Display for WeightDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "is not finite"),
            Self::NotPositive => write!(f, "is not positive"),
            Self::BelowFloor => write!(f, "is below MIN_POSITIVE / EPSILON of the scalar type"),
        }
    }
}

impl fmt::Display for SurplusDefect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Negative => write!(f, "is negative"),
            Self::BelowFloor => write!(f, "is below MIN_POSITIVE / EPSILON of the scalar type"),
        }
    }
}

impl fmt::Display for AdjacencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowPtrsEmpty => write!(f, "row_ptrs is empty"),
            Self::RowPtrsMustStartAtZero { got } => write!(f, "row_ptrs[0] must be 0 (got {got})"),
            Self::NeighborsWeightsLenMismatch { neighbors, weights } => write!(
                f,
                "neighbors and weights have different lengths ({neighbors} != {weights})"
            ),
            Self::RowPtrsEndMismatch { end, len } => {
                write!(
                    f,
                    "the last row pointer must equal the neighbor count ({end} != {len})"
                )
            }
            Self::RowPtrsDecrease { row } => write!(f, "row_ptrs decreases after row {row}"),
            Self::TooManyVertices { n } => write!(f, "{n} vertices leave no u32 for a ground slot"),
            Self::NotStrictlyUpper { edge: (row, col) } => {
                write!(f, "neighbor {col} of row {row} is not above the diagonal")
            }
            Self::Unsorted { edge: (row, col) } => {
                write!(f, "neighbor {col} of row {row} is not strictly ascending")
            }
            Self::NeighborOutOfBounds {
                edge: (row, col),
                n,
            } => {
                write!(
                    f,
                    "neighbor {col} of row {row} is not one of the {n} vertices"
                )
            }
            Self::Weight {
                edge: (row, col),
                defect,
            } => write!(f, "the weight of edge ({row}, {col}) {defect}"),
        }
    }
}

impl fmt::Display for LaplacianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adjacency(error) => write!(f, "invalid adjacency: {error}"),
            Self::DegreeNotFinite { vertex } => {
                write!(f, "vertex {vertex}'s weighted degree is not finite")
            }
        }
    }
}

impl fmt::Display for SddmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adjacency(error) => write!(f, "invalid adjacency: {error}"),
            Self::SurplusLength { len, n } => {
                write!(f, "surplus has {len} entries for {n} vertices")
            }
            Self::Surplus { vertex, defect } => write!(f, "vertex {vertex}'s surplus {defect}"),
            Self::DiagonalNotFinite { vertex } => {
                write!(f, "vertex {vertex}'s diagonal is not finite")
            }
            Self::GroundOverflow { vertex } => write!(
                f,
                "the ground of the component holding vertex {vertex} sums to a non-finite value"
            ),
        }
    }
}

impl std::error::Error for AdjacencyError {}
impl std::error::Error for LaplacianError {}
impl std::error::Error for SddmError {}

/// An unusable exact pivot, reported as a [`Fallback`] or returned by [`factorize_with`](crate::factorize_with).
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
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Error::MagnitudeTooSmall { entry: (row, col) } if row == col => write!(
                f,
                "row {row}'s diagonal surplus is below MIN_POSITIVE / EPSILON of the scalar type; scale the matrix up"
            ),
            Error::MagnitudeTooSmall { entry: (row, col) } => write!(
                f,
                "entry ({row}, {col}) is below MIN_POSITIVE / EPSILON of the scalar type; scale the matrix up"
            ),
            Error::GroundOverflow { vertex } => write!(
                f,
                "the ground of the component holding vertex {vertex} sums to a non-finite value"
            ),
        }
    }
}

impl std::error::Error for Error {}
impl std::error::Error for CsrError {}
impl std::error::Error for UnusablePivot {}
