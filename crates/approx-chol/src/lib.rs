//! Approximate Cholesky factorization for SDDM and graph Laplacian systems.
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
//! [`Config::backend`] picks exact dense Cholesky or approximate elimination per connected block.
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
//! // Lists blocks factored approximately after an unusable exact pivot: less accurate than asked.
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
pub use approx_chol::{Backend, Config, ExactFailure, Factor, SolveError};
pub use csr::{CsrRef, OwnedCsr};
pub use error::{CsrError, DenseFailure, Error, Fallback, IndexKind, UnusablePivot};

/// Factorize an SDDM matrix with [`Config::default`].
pub fn factorize<'a, T, I, M>(sddm: M) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
    I: num_traits::PrimInt + 'a + 'static,
    M: TryInto<CsrRef<'a, T, I>>,
    <M as TryInto<CsrRef<'a, T, I>>>::Error: Into<Error>,
{
    factorize_with(sddm, Config::default())
}

/// [`factorize`] with a custom [`Config`], whose [`ExactFailure::Error`] can raise a pivot error.
pub fn factorize_with<'a, T, I, M>(sddm: M, config: Config) -> Result<Factor<T>, Error>
where
    T: num_traits::Float + Send + Sync + 'static,
    I: num_traits::PrimInt + 'a + 'static,
    M: TryInto<CsrRef<'a, T, I>>,
    <M as TryInto<CsrRef<'a, T, I>>>::Error: Into<Error>,
{
    approx_chol::Builder::<T>::new(config).build(sddm)
}
