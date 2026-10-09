//! Approximate Cholesky of symmetric generalized diagonally dominant `A`, one component at a time.

#![deny(missing_docs)]
#![expect(
    dead_code,
    unused_variables,
    reason = "types-only skeleton; unfulfilled once the bodies land"
)]

mod factor;
pub mod input;
mod sgdd;
mod storage;

use num_traits::Float;

pub use factor::{CliqueRule, Config, InnerParallelism, Multiplicity, OnDenseFailure, SplitFactor};
pub use factor::{Factor, FactorError, PartFactor, SolveError};
pub use factor::{Fallback, PivotDefect, UnusablePivot};
pub use sgdd::{FromCsrError, Part, ReduceError, ScalingPolicy, ScalingReport, Sgdd};

/// The scalar every stage computes in.
pub trait Real: Float + Send + Sync + 'static {}

impl<T: Float + Send + Sync + 'static> Real for T {}

/// Factors every part of `sgdd` in parallel; `sgdd` survives for another `config`.
pub fn factorize<T: Real>(sgdd: &Sgdd<T>, config: &Config) -> Result<Factor<T>, FactorError> {
    Factor::build(sgdd, config)
}
