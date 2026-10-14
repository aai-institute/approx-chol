//! Approximate Cholesky of symmetric generalized diagonally dominant `A`, one irreducible block at a time.

#![deny(missing_docs)]

/// Serde through `remote = "Self"`, every decoded value passed through its type's `checked`.
#[cfg(feature = "serde")]
macro_rules! checked_serde {
    ($ty:ident) => {
        impl<T: serde::Serialize> serde::Serialize for $ty<T> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                $ty::serialize(self, serializer)
            }
        }
        impl<'de, T: crate::Real + serde::Deserialize<'de>> serde::Deserialize<'de> for $ty<T> {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let corrupt = || serde::de::Error::custom(concat!("corrupt ", stringify!($ty)));
                let value = $ty::deserialize(deserializer)?;
                value.checked().ok_or_else(corrupt)
            }
        }
    };
}

mod factor;
pub mod input;
mod sgdd;
mod storage;

use std::fmt::Debug;

pub use factor::{factorize, BlockFactor, Factor, Fallback, SolveError};
pub use factor::{CliqueRule, Config, InnerParallelism, Multiplicity, SplitCount};
pub use sgdd::{
    Existence, FromCsrError, IrreducibleBlock, ReduceError, ScalingPolicy, ScalingReport, Sgdd,
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

/// `Σ x` with each addition's rounding carried (Knuth's TwoSum): within `ε` of exact for nonnegative terms.
pub(crate) fn compensated_sum(terms: impl Iterator<Item = f64>) -> f64 {
    let (sum, carry) = terms.fold((0.0f64, 0.0f64), |(sum, carry), x| {
        let next = sum + x;
        let x_part = next - sum;
        (next, carry + ((sum - (next - x_part)) + (x - x_part)))
    });
    sum + carry
}
