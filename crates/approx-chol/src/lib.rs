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
//! let x = decomp.solve(&b);
//! assert!(x.iter().all(|v| f64::is_finite(*v)));
//! # Ok::<(), approx_chol::Error>(())
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
//! ## AC2 — multi-edge split/merge
//!
//! AC2 (Algorithm 6 in Gao-Kyng-Spielman 2023) splits each edge into `k`
//! copies before elimination, then merges back to at most `l` multi-edges per
//! neighbor pair after compression. This produces a denser but higher-quality
//! approximate factor that converges in fewer PCG iterations, at the cost of
//! higher factorization time. Enable via [`SplitMerge`] in
//! [`Config::split_merge`].
//!
//! # Ordering strategies
//!
//! The high-level API ([`factorize`], [`factorize_with`]) always uses
//! **DynamicPQ** — a bucket priority queue that tracks vertex degrees
//! throughout elimination. This is the best default for nearly all inputs.
//!
//! For explicit ordering control, use [`low_level::Builder::ordering`]:
//!
//! ```
//! use approx_chol::{Config, CsrRef};
//! use approx_chol::low_level::{Builder, Ordering};
//!
//! # let row_ptrs    = [0u32, 2, 5, 8, 10];
//! # let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! # let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//! # let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
//! let factor = Builder::new(Config::default())
//!     .ordering(Ordering::StaticAMD)
//!     .build(csr)?;
//! # Ok::<(), approx_chol::Error>(())
//! ```
//!
//! # Feature flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `sprs`  | Enables zero-copy [`CsrRef`] conversion from `sprs` matrices (`From` and fallible `try_from_sprs*`). |
//! | `faer`  | Enables zero-copy [`CsrRef`] conversion from `faer` matrices (`From` and fallible `try_from_faer*`). |
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
#![warn(clippy::all)]

mod approx_chol;
mod csr;
pub(crate) mod graph;
pub(crate) mod ordering;
pub(crate) mod sampling;
mod star_clique;
mod types;

/// Low-level API for advanced use cases (custom pipelines, graph inspection, research).
pub mod low_level;

pub use approx_chol::Factor;
pub use approx_chol::{Config, SplitMerge};
pub use csr::{CsrRef, OwnedCsr};
pub(crate) use types::Real;

// Re-export Builder, Ordering, and sample_star_clique at the crate root
// for backward compatibility with current consumers. New code should prefer
// `approx_chol::low_level::{Builder, Ordering, sample_star_clique}`.
#[doc(hidden)]
pub use approx_chol::{Builder, Ordering};
#[doc(hidden)]
pub use star_clique::sample_star_clique;

use std::fmt;

/// Errors that can occur during approximate Cholesky factorization.
#[derive(Debug, Clone)]
pub enum Error {
    /// The factorization configuration is invalid.
    ///
    /// The inner string describes the specific failure (e.g.
    /// `"split_merge.merge must be >= 1"`).
    InvalidConfig(&'static str),

    /// The input CSR matrix has inconsistent dimensions or invalid structure.
    ///
    /// The inner string describes the specific validation failure (e.g.
    /// `"row_ptrs length != n + 1"` or `"column index out of bounds"`).
    InvalidCsr(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidConfig(msg) => write!(f, "invalid factorization config: {msg}"),
            Error::InvalidCsr(msg) => write!(f, "invalid CSR matrix: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Factorize an SDDM matrix with default configuration.
///
/// Uses standard AC with dynamic ordering (DynamicPQ). For AC2 or
/// custom seeds, use [`factorize_with`].
///
/// # Errors
///
/// Returns [`Error::InvalidCsr`] if the input matrix structure is
/// invalid (see [`CsrRef::new`] for the full list of checks).
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
pub fn factorize<T>(sddm: CsrRef<'_, T, u32>) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    Builder::<T>::new(Config::default()).build(sddm)
}

/// Factorize an SDDM matrix with custom configuration.
///
/// Uses [`Config`] to control the random seed and AC2 split/merge
/// parameters. Always uses DynamicPQ ordering. For ordering control,
/// use [`low_level::Builder`].
///
/// # Errors
///
/// Returns [`Error::InvalidCsr`] if the input matrix structure is
/// invalid, or [`Error::InvalidConfig`] if configuration values are
/// inconsistent (e.g. `split_merge.split == 0`).
///
/// # Examples
///
/// ```
/// use approx_chol::{factorize_with, Config, SplitMerge, CsrRef};
///
/// let row_ptrs    = [0u32, 2, 5, 8, 10];
/// let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
/// let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
///
/// let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
/// let factor = factorize_with(csr, Config {
///     seed: 42,
///     split_merge: Some(SplitMerge { split: 2, merge: 2 }),
///     ..Default::default()
/// })?;
/// assert_eq!(factor.n(), 4);
/// # Ok::<(), approx_chol::Error>(())
/// ```
pub fn factorize_with<T>(sddm: CsrRef<'_, T, u32>, config: Config) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    Builder::<T>::new(config).build(sddm)
}

/// Factorize an SDDM matrix with default configuration from any index type
/// that can be converted to `u32`.
///
/// Uses a zero-copy fast path when the input index type is `u32`; otherwise
/// performs a checked conversion of row pointers and column indices.
pub fn factorize_generic<T, I>(sddm: CsrRef<'_, T, I>) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
    I: num_traits::PrimInt + 'static,
{
    Builder::<T>::new(Config::default()).build_generic(sddm)
}

/// Factorize an SDDM matrix from any input convertible into [`CsrRef`].
///
/// This is the highest-level entry point for ergonomic integration with sparse
/// matrix libraries. Any type with a `From`/`Into` conversion to `CsrRef`
/// can be passed directly.
///
/// For panic-free matrix-adapter paths, prefer the fallible constructors on
/// [`CsrRef`] (`try_from_sprs_view`, `try_from_sprs`, `try_from_faer_view`,
/// `try_from_faer`) and then call [`factorize_generic`].
///
/// # Errors
///
/// Returns [`Error::InvalidCsr`] if validation fails, index
/// conversion to `u32` fails, or conversion into [`CsrRef`] panics.
/// Returns [`Error::InvalidConfig`] if configuration values are
/// inconsistent.
pub fn factorize_from<'a, T, I, M>(sddm: M) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
    I: num_traits::PrimInt + 'a + 'static,
    M: Into<CsrRef<'a, T, I>>,
{
    Builder::<T>::new(Config::default()).build_from(sddm)
}
