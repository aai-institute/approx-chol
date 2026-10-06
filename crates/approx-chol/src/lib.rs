//! Approximate Cholesky factorization for SDDM and graph Laplacian systems.
//!
//! An [`Sddm`] is a [`Laplacian`] (strict upper adjacency), alone or [`Grounded`] by a surplus:
//!
//! ```
//! use approx_chol::{factorize, Grounded, Laplacian};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Path 0-1-2-3 with unit weights.
//! let path: Laplacian = Laplacian::new(vec![0, 1, 2, 3, 3], vec![1, 2, 3], vec![1.0, 1.0, 1.0])?;
//! let x = factorize(path.clone()).solve(&[1.0, -1.0, 1.0, -1.0])?;
//! assert!(x.iter().all(|v| v.is_finite()));
//!
//! // The same path with surplus on vertex 0, so any right-hand side is consistent.
//! let grounded = Grounded::new(path, vec![1.0, 0.0, 0.0, 0.0])?;
//! let x = factorize(grounded).solve(&[1.0, 2.0, 3.0, 4.0])?;
//! assert!(x.iter().all(|v| v.is_finite()));
//! # Ok(())
//! # }
//! ```
//!
//! A CSR becomes a checked [`Sddm`]; [`Config::backend`] picks exact or approximate per block:
//!
//! ```
//! use approx_chol::{factorize_with, Backend, Config, CsrRef, ExactFailure, Sddm};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let row_ptrs    = [0u32, 2, 5, 8, 10];
//! let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//! let sddm = Sddm::try_from(CsrRef::new(&row_ptrs, &col_indices, &values, 4)?)?;
//!
//! let config = Config {
//!     backend: Backend::ExactBelow {
//!         max_dim: 64,
//!         on_failure: ExactFailure::FallBackToApproximate,
//!     },
//!     ..Config::default()
//! };
//! let factor = factorize_with(sddm, config)?;
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
mod types;

pub mod low_level;

#[cfg(feature = "serde")]
pub use approx_chol::FACTOR_FORMAT_VERSION;
pub use approx_chol::{Backend, Config, ExactFailure, Factor, Fallback, SolveError};
pub use csr::CsrRef;
pub use error::{
    CsrError, DenseFailure, GroundedError, IndexKind, LaplacianError, NotSddm, UnusablePivot,
};
pub use sddm::{Grounded, Laplacian, Sddm};

/// Factorize with [`Config::default`], whose policy falls back rather than failing.
pub fn factorize<T>(sddm: impl Into<Sddm<T>>) -> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    factorize_with(sddm, Config::default())
        .expect("the default policy falls back on an unusable pivot rather than failing")
}

/// Factorize with a custom [`Config`]; [`UnusablePivot`] only under [`ExactFailure::Error`].
pub fn factorize_with<T>(
    sddm: impl Into<Sddm<T>>,
    config: Config,
) -> Result<Factor<T>, UnusablePivot>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    approx_chol::factorize(&sddm.into(), config)
}
