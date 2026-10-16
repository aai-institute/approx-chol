//! Approximate Cholesky of symmetric generalized diagonally dominant `A`, one irreducible block at a time.

#![deny(missing_docs)]

/// Serde through `remote = "Self"`, every decoded value passed through its type's `checked`.
#[cfg(feature = "serde")]
macro_rules! checked_serde {
    ($ty:ident) => {
        impl<T: serde::Serialize, I: serde::Serialize> serde::Serialize for $ty<T, I> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                $ty::serialize(self, serializer)
            }
        }
        impl<'de, T, I> serde::Deserialize<'de> for $ty<T, I>
        where
            T: crate::Real + serde::Deserialize<'de>,
            I: crate::Index + serde::Deserialize<'de>,
        {
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
#[cfg(feature = "serde")]
mod wire;

use std::fmt::Debug;

use num_traits::{PrimInt, Unsigned};

pub use factor::{factorize, BlockFactor, Factor, SolveError};
pub use factor::{CliqueRule, Config, InnerParallelism, Multiplicity, SplitCount};
pub use sgdd::{
    Existence, FromCsrError, IrreducibleBlock, ReduceError, ScalingPolicy, ScalingReport, Sgdd,
};

/// The scalar every stage computes in: `f32` or `f64`, each summed in `f64`, which holds it exactly.
pub trait Real: num_traits::Float + Debug + Send + Sync + 'static + seal::Real {}

impl Real for f32 {}
impl Real for f64 {}

/// The unsigned integer a reduction and its factor store each vertex index in: `u32` by default, `u64` past `u32::MAX` vertices.
pub trait Index: PrimInt + Unsigned + Send + Sync + 'static + seal::Index {
    /// `self` as `usize`, so `x[v.index()]` reads vertex `v` of any `x` over `A`'s vertices.
    fn index(self) -> usize;
}

mod seal {
    /// Only the crate implements [`Real`](super::Real), so every `T` widens to `f64` exactly.
    pub trait Real {}

    impl Real for f32 {}
    impl Real for f64 {}

    /// Only the crate implements [`Index`](super::Index), so its conversions are plain casts.
    pub trait Index: Copy {
        /// `k` as `Self`, truncating: every caller passes a value a checked count bounds.
        fn from_index(k: usize) -> Self;
    }
}

macro_rules! index {
    ($int:ty) => {
        impl Index for $int {
            #[inline(always)]
            fn index(self) -> usize {
                self as usize
            }
        }

        impl seal::Index for $int {
            #[inline(always)]
            fn from_index(k: usize) -> Self {
                debug_assert!(<$int>::try_from(k).is_ok(), "{k} fits the index");
                k as $int
            }
        }
    };
}

index!(u32);
index!(u64);

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
