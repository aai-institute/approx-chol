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

pub use factor::{CliqueRule, Config, InnerParallelism, Multiplicity, OnDenseFailure, SplitCount};
pub use factor::{Factor, FactorError, PartFactor, SolveError};
pub use factor::{Fallback, PivotDefect, UnusablePivot};
pub use sgdd::{FromCsrError, Part, ReduceError, ScalingPolicy, ScalingReport, Sgdd};

/// The scalar every stage computes in: `f32` or `f64`.
pub trait Real: sealed::Sealed {}

impl Real for f32 {}
impl Real for f64 {}

/// Unnameable outside, so `num_traits` stays out of the public API and `Real` can grow.
mod sealed {
    pub trait Sealed: num_traits::Float + Send + Sync + 'static {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Factors every part of `sgdd` in parallel; `sgdd` survives for another `config`.
pub fn factorize<T: Real>(sgdd: &Sgdd<T>, config: &Config) -> Result<Factor<T>, FactorError> {
    Factor::build(sgdd, config)
}
