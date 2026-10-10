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

use std::fmt::Debug;

pub use factor::{CliqueRule, Config, InnerParallelism, Multiplicity, OnDenseFailure, SplitCount};
pub use factor::{ComponentFactor, Factor, FactorError, SolveError};
pub use factor::{Fallback, PivotDefect, UnusablePivot};
pub use sgdd::{Component, FromCsrError, ReduceError, ScalingPolicy, ScalingReport, Sgdd};

/// The scalar every stage computes in: any `Float` a factor can share across threads and print.
pub trait Real: num_traits::Float + Debug + Send + Sync + 'static {}

impl<T: num_traits::Float + Debug + Send + Sync + 'static> Real for T {}

/// Factors every component of `sgdd` in parallel; `sgdd` survives for another `config`.
pub fn factorize<T: Real>(sgdd: &Sgdd<T>, config: &Config) -> Result<Factor<T>, FactorError> {
    Factor::build(sgdd, config)
}
