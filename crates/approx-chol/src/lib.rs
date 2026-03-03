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
//! let x = decomp.solve(&b).expect("rhs length must be <= factor dimension");
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
//! | `sprs`  | Enables zero-copy [`CsrRef`] conversion from `sprs` matrices (`TryFrom` and `try_from_sprs*`). |
//! | `faer`  | Enables zero-copy [`CsrRef`] conversion from `faer` matrices (`TryFrom` and `try_from_faer*`). |
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
mod types;

/// Low-level API for advanced use cases (custom pipelines, graph inspection, research).
pub mod low_level;

pub use approx_chol::{Config, Factor, SolveError};
pub use csr::{CsrRef, OwnedCsr};
pub(crate) use types::Real;

// Re-export Builder and star-clique sampling helpers at the crate root.
// New code should prefer
// `approx_chol::low_level::{Builder, clique_tree_sample, clique_tree_sample_multi}`.
#[doc(hidden)]
pub use approx_chol::clique_tree::{clique_tree_sample, clique_tree_sample_multi};
#[doc(hidden)]
pub use approx_chol::Builder;

use std::fmt;

/// Errors that can occur during approximate Cholesky factorization.
#[derive(Debug, Clone)]
pub enum Error {
    /// The factorization configuration is invalid.
    ///
    /// The inner string describes the specific failure (e.g.
    /// `"split_merge must be >= 1"`).
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
    Builder::<T>::new(Config::default()).build(sddm)
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
    Builder::<T>::new(config).build(sddm)
}
