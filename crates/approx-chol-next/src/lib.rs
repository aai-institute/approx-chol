//! Approximate Cholesky of symmetric generalized diagonally dominant `A`, one component at a time.

#![deny(missing_docs)]

mod factor;
pub mod input;
mod sgdd;
mod storage;

use std::fmt::Debug;

pub use factor::{factorize, ComponentFactor, Factor, Fallback, SolveError};
pub use factor::{CliqueRule, Config, InnerParallelism, Multiplicity, SplitCount};
pub use sgdd::{
    Component, Existence, FromCsrError, ReduceError, ScalingPolicy, ScalingReport, Sgdd,
};

/// The scalar every stage computes in: any `Float` a factor can share across threads and print.
pub trait Real: num_traits::Float + Debug + Send + Sync + 'static {}

impl<T: num_traits::Float + Debug + Send + Sync + 'static> Real for T {}

/// Finite, and zero or at least `MIN_POSITIVE/EPSILON`, so no product turns subnormal (#163).
pub(crate) fn in_range<T: Real>(x: T) -> bool {
    x.is_finite() && (x == T::zero() || x.abs() >= T::min_positive_value() / T::epsilon())
}

/// Every `f64` sum starts here: a `Float` always casts up.
pub(crate) fn widen<T: Real>(x: T) -> f64 {
    x.to_f64().expect("a float casts to f64")
}

/// An `f64` sum in `T`: past `T::MAX` it reads infinite, which every range screen rejects.
pub(crate) fn narrow<T: Real>(sum: f64) -> T {
    T::from(sum).unwrap_or_else(T::infinity)
}

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
pub(crate) fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}
