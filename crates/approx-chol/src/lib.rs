//! Approximate Cholesky factorization for SDDM and graph Laplacian systems.
//!
//! ```
//! use approx_chol::{factorize, CsrRef, Laplacian, Sddm};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // The path 0-1-2-3 as a strict upper adjacency.
//! let laplacian = Laplacian::new(vec![0, 1, 2, 3, 3], vec![1, 2, 3], vec![1.0, 1.0, 1.0])?;
//! let x = factorize(laplacian).solve(&[1.0, -1.0, 1.0, -1.0])?;
//! assert!(x.iter().all(|v| f64::is_finite(*v)));
//!
//! // The same matrix as a symmetric CSR.
//! let row_ptrs    = [0u32, 2, 5, 8, 10];
//! let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//!
//! let csr = CsrRef::new(&row_ptrs, &col_indices, &values, 4)?;
//! let x = factorize(Sddm::try_from(csr)?).solve(&[1.0, -1.0, 1.0, -1.0])?;
//! assert!(x.iter().all(|v| f64::is_finite(*v)));
//! # Ok(())
//! # }
//! ```
//!
//! [`Config::backend`] picks exact dense Cholesky or approximate elimination per connected block.
//!
//! ```
//! use approx_chol::{factorize_with, Backend, Config, CsrRef, ExactFailure, Sddm};
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
//! let factor = factorize_with(Sddm::try_from(csr)?, config)?;
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
pub use error::{
    AdjacencyError, CsrError, DenseFailure, Error, Fallback, IndexKind, LaplacianError, SddmError,
    SurplusDefect, UnusablePivot, WeightDefect,
};
pub use sddm::{Laplacian, Sddm};

/// Factorize an SDDM matrix with [`Config::default`], which falls back rather than fail.
pub fn factorize<T>(sddm: impl Into<Sddm<T>>) -> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    factorize_with(sddm, Config::default())
        .expect("the default backend falls back on an unusable pivot")
}

/// [`factorize`] with a custom [`Config`], whose [`ExactFailure::Error`] can raise a pivot error.
pub fn factorize_with<T>(
    sddm: impl Into<Sddm<T>>,
    config: Config,
) -> Result<Factor<T>, UnusablePivot>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    approx_chol::factor(sddm.into(), config)
}
