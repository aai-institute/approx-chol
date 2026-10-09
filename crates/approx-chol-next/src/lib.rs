//! Approximate Cholesky of symmetric generalized diagonally dominant `A`, one component at a time.

#![deny(missing_docs)]
#![expect(
    dead_code,
    unused_variables,
    reason = "types-only skeleton; unfulfilled once the bodies land"
)]

mod config;
mod dense;
mod factor;
pub mod input;
mod laplacian;
mod min_degree;
mod round;
mod sampling;
mod sgdd;
mod storage;

use num_traits::Float;

pub use config::{CliqueRule, Config, ExactFailure, InnerParallelism, Multiplicity, SplitFactor};
pub use dense::{DenseFailure, Fallback, UnusablePivot};
pub use factor::{ComponentFactor, Factor, FactorError, SolveError};
pub use sgdd::{FromCsrError, Part, ReduceError, ScalingPolicy, ScalingReport, Sgdd};

/// The scalar every stage computes in.
pub trait Real: Float + Send + Sync + 'static {}

impl<T: Float + Send + Sync + 'static> Real for T {}

/// Factors every part of `sgdd` in parallel; `sgdd` survives for another `config`.
pub fn factorize<T: Real>(sgdd: &Sgdd<T>, config: &Config) -> Result<Factor<T>, FactorError> {
    Factor::build(sgdd, config)
}
