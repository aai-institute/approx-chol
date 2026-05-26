//! Approximate Cholesky factorization for SDDM and graph Laplacian systems.
//!
//! This crate provides a robust Rust implementation of approximate Cholesky
//! (AC) factorization, suitable as a preconditioner for iterative solvers on
//! symmetric diagonally dominant (SDDM) linear systems. SDDM matrices arise
//! naturally in graph Laplacians, finite-element discretizations, and
//! fixed-effects normal equations. Every graph Laplacian is SDDM; every SDDM
//! matrix can be converted to a Laplacian via Gremban's reduction (1996).
//!
//! # Quick start
//!
//! ```
//! use approx_chol::{factorize, CsrRef};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//! // 4-node path graph Laplacian (0-1-2-3)
//! let row_ptrs    = [0u32, 2, 5, 8, 10];
//! let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//!
//! let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
//! let decomp = factorize(csr)?;
//!
//! // RHS must lie in the range of the Laplacian (sum to zero)
//! let b = [1.0, -1.0, 1.0, -1.0];
//! let x = decomp.solve(&b)?;
//! assert!(x.iter().all(|v| f64::is_finite(*v)));
//! # Ok(())
//! # }
//! ```
//!
//! # Algorithm variants
//!
//! ## AC — standard approximate Cholesky
//!
//! The default variant (Algorithm 5 / 8 in Gao-Kyng-Spielman 2023) performs
//! one random edge sampling per elimination step. It is fast, memory-efficient,
//! and sufficient for most applications. Use [`Config::default()`].
//!
//! ## AC2 — multi-edge multiplicity
//!
//! AC2 (Algorithm 6 in Gao-Kyng-Spielman 2023) splits each edge into `k`
//! copies before elimination, then keeps at most `k` multi-edges per neighbor
//! pair after compression. This produces a denser but higher-quality
//! approximate factor that converges in fewer PCG iterations, at the cost of
//! higher factorization time. Enable via [`Config::split_merge`].
//!
//! # Ordering strategies
//!
//! The high-level API ([`factorize`], [`factorize_with`]) always uses
//! **DynamicPQ** — a bucket priority queue that tracks vertex degrees
//! throughout elimination. This is the best default for nearly all inputs.
//!
//! # Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `sprs`  | Enables zero-copy [`CsrRef`] conversion from `sprs` matrices via `TryFrom`. |
//! | `faer`  | Enables zero-copy [`CsrRef`] conversion from `faer` matrices via `TryFrom`. |
//!
//! # References
//!
//! - Gao, J., Kyng, R., & Spielman, D. A. (2023). *Robust and Practical
//!   Solution of Laplacian Equations by Approximate Elimination.*
//!   <https://arxiv.org/abs/2303.00709>
//! - Kyng, R., & Sachdeva, S. (2016). *Approximate Gaussian Elimination for
//!   Laplacians — Fast, Sparse, and Simple.*
//!   <https://arxiv.org/abs/1605.02353>
//! - Gremban, K. D. (1996). *Combinatorial Preconditioners for Sparse,
//!   Symmetric, Diagonally Dominant Linear Systems.* Ph.D. thesis, CMU.
//! - Laplacians.jl — Julia reference implementation by Spielman et al.
//!   <https://github.com/danspielman/Laplacians.jl>

#![deny(missing_docs)]
#![warn(clippy::all, clippy::undocumented_unsafe_blocks)]

mod approx_chol;
mod csr;
pub(crate) mod graph;
pub(crate) mod ordering;
pub(crate) mod sampling;
#[cfg(test)]
pub(crate) mod test_utils;
mod types;

/// Low-level API for advanced use cases (custom pipelines, graph inspection, research).
pub mod low_level;

pub use approx_chol::{Config, Factor, SolveError};
pub use csr::{CsrRef, OwnedCsr};
pub(crate) use types::Real;

use std::fmt;

/// Errors that can occur during approximate Cholesky factorization.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The factorization configuration is invalid.
    InvalidConfig(ConfigError),

    /// The input CSR matrix has inconsistent dimensions or invalid structure.
    InvalidCsr(CsrError),
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

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SplitMergeMustBePositive { split_merge } => {
                write!(f, "split_merge must be >= 1 (got {split_merge})")
            }
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
            Error::InvalidConfig(err) => write!(f, "invalid factorization config: {err}"),
            Error::InvalidCsr(err) => write!(f, "invalid CSR matrix: {err}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<core::convert::Infallible> for Error {
    fn from(value: core::convert::Infallible) -> Self {
        match value {}
    }
}

/// Factorize an SDDM matrix with default configuration.
///
/// Uses standard AC with dynamic ordering (DynamicPQ). For AC2 or
/// custom seeds, use [`factorize_with`].
///
/// Accepts any input fallibly convertible into [`CsrRef`], including `CsrRef`
/// directly and, with feature flags, borrowed matrices from `sprs`/`faer`.
///
/// # Errors
///
/// Returns [`Error::InvalidCsr`] if conversion or validation fails, if index
/// conversion to `u32` fails, or if input conversion panics.
///
/// # Examples
///
/// ```
/// use approx_chol::{factorize, CsrRef};
///
/// let row_ptrs    = [0u32, 2, 5, 8, 10];
/// let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
/// let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
///
/// let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
/// let decomp = factorize(csr)?;
/// assert_eq!(decomp.n(), 4);
/// # Ok::<(), approx_chol::Error>(())
/// ```
pub fn factorize<'a, T, I, M>(sddm: M) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
    I: num_traits::PrimInt + 'a + 'static,
    M: TryInto<CsrRef<'a, T, I>>,
    <M as TryInto<CsrRef<'a, T, I>>>::Error: Into<Error>,
{
    factorize_with(sddm, Config::default())
}

/// Factorize an SDDM matrix with custom configuration.
///
/// Uses [`Config`] to control the random seed and AC2 split multiplicity
/// parameters. Always uses DynamicPQ ordering.
///
/// # Errors
///
/// Returns [`Error::InvalidCsr`] if conversion or validation fails, if index
/// conversion to `u32` fails, or if input conversion panics.
/// Returns [`Error::InvalidConfig`] if configuration values are inconsistent
/// (e.g. `split_merge == Some(0)`).
///
/// # Examples
///
/// ```
/// use approx_chol::{factorize_with, Config, CsrRef};
///
/// let row_ptrs    = [0u32, 2, 5, 8, 10];
/// let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
/// let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
///
/// let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
/// let factor = factorize_with(csr, Config {
///     seed: 42,
///     split_merge: Some(2),
///     ..Default::default()
/// })?;
/// assert_eq!(factor.n(), 4);
/// # Ok::<(), approx_chol::Error>(())
/// ```
pub fn factorize_with<'a, T, I, M>(sddm: M, config: Config) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
    I: num_traits::PrimInt + 'a + 'static,
    M: TryInto<CsrRef<'a, T, I>>,
    <M as TryInto<CsrRef<'a, T, I>>>::Error: Into<Error>,
{
    approx_chol::Builder::<T>::new(config).build(sddm)
}
