//! Approximate Cholesky factorization for SDDM and graph Laplacian systems.
//!
//! An [`Sddm`] is a [`Laplacian`], given as its strict upper adjacency, alone or
//! [`Grounded`] by a diagonal surplus:
//!
//! ```
//! use approx_chol::{factorize, Grounded, Laplacian};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Path 0-1-2-3 with unit weights.
//! let path: Laplacian = Laplacian::new(vec![0, 1, 2, 3, 3], vec![1, 2, 3], vec![1.0, 1.0, 1.0])?;
//! let x = factorize(path.clone())?.solve(&[1.0, -1.0, 1.0, -1.0])?;
//! assert!(x.iter().all(|v| v.is_finite()));
//!
//! // The same path with surplus on vertex 0, so any right-hand side is consistent.
//! let grounded = Grounded::new(path, vec![1.0, 0.0, 0.0, 0.0])?;
//! let x = factorize(grounded)?.solve(&[1.0, 2.0, 3.0, 4.0])?;
//! assert!(x.iter().all(|v| v.is_finite()));
//! # Ok(())
//! # }
//! ```
//!
//! A CSR matrix converts into an [`Sddm`], which checks symmetry and dominance:
//!
//! ```
//! use approx_chol::{factorize, CsrRef};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let row_ptrs    = [0u32, 2, 5, 8, 10];
//! let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//!
//! let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
//! let x = factorize(csr)?.solve(&[1.0, -1.0, 1.0, -1.0])?;
//! assert!(x.iter().all(|v| f64::is_finite(*v)));
//! # Ok(())
//! # }
//! ```
//!
//! [`Config::backend`] picks a factorization per connected block: exact dense
//! Cholesky at or below `max_dim` solved variables, approximate elimination above.
//!
//! ```
//! use approx_chol::{factorize_with, Backend, Config, CsrRef, ExactFailure};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let row_ptrs    = [0u32, 2, 5, 8, 10];
//! let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//! let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
//!
//! let config = Config {
//!     backend: Backend::ExactBelow {
//!         max_dim: 64,
//!         on_failure: ExactFailure::FallBackToApproximate,
//!     },
//!     ..Config::default()
//! };
//! let factor = factorize_with(csr, config)?;
//!
//! // A block whose exact pivot is unusable is factored approximately and listed
//! // here, so a non-empty slice means the factor is less accurate than asked for.
//! assert!(factor.fallbacks().is_empty());
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

mod approx_chol;
mod csr;
mod error;
pub(crate) mod graph;
pub(crate) mod sampling;
mod sddm;
#[cfg(test)]
pub(crate) mod test_utils;
mod types;

pub mod low_level;

#[cfg(feature = "serde")]
pub use approx_chol::FACTOR_FORMAT_VERSION;
pub use approx_chol::{Backend, Config, ExactFailure, Factor, Fallback, SolveError};
pub use csr::{CsrRef, OwnedCsr};
pub use error::{CsrError, DenseFailure, Error, IndexKind, UnusablePivot};
pub use sddm::{Grounded, Laplacian, Sddm};

/// Factorize an SDDM matrix with [`Config::default`].
pub fn factorize<T, M>(sddm: M) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
    M: TryInto<Sddm<T>>,
    <M as TryInto<Sddm<T>>>::Error: Into<Error>,
{
    factorize_with(sddm, Config::default())
}

/// Factorize an SDDM matrix with a custom [`Config`].
///
/// # Errors
///
/// Beyond the input rejections [`factorize`] shares, returns
/// [`Error::DenseFactorizationFailed`] when a block's exact pivot is unusable and
/// [`ExactFailure::Error`] asked for that to fail rather than fall back.
pub fn factorize_with<T, M>(sddm: M, config: Config) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
    M: TryInto<Sddm<T>>,
    <M as TryInto<Sddm<T>>>::Error: Into<Error>,
{
    approx_chol::Builder::<T>::new(config).build(sddm)
}
