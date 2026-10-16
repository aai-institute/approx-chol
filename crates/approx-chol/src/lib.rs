//! Approximate Cholesky of symmetric generalized diagonally dominant `A`, one irreducible block at a time.
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

mod error;
mod factor;
pub mod input;
mod sgdd;
mod storage;
#[cfg(feature = "serde")]
mod wire;

use std::fmt::Debug;

use num_traits::{PrimInt, Unsigned};

pub use error::Error;
use factor::Storage;
pub use factor::{Backend, BlockFactor, Config, Factor, InnerParallelism, SchurMode, SolveError};
pub use input::{CsrError, CsrRef, IndexKind, Input, OwnedCsr};
pub use sgdd::{ReduceError, ScalingPolicy, ScalingReport};
#[cfg(feature = "serde")]
pub use wire::FORMAT_VERSION as FACTOR_FORMAT_VERSION;

/// Factorizes any [`Input`] view with [`Config::default`].
pub fn factorize<'a, T, I, M>(input: M) -> Result<Factor<T>, Error>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: TryInto<Input<'a, T, I>>,
    <M as TryInto<Input<'a, T, I>>>::Error: Into<Error>,
{
    factorize_with(input, Config::default())
}

/// Factorizes any [`Input`] view: reduces it to its irreducible blocks, then factors each in parallel.
pub fn factorize_with<'a, T, I, M>(input: M, config: Config) -> Result<Factor<T>, Error>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: TryInto<Input<'a, T, I>>,
    <M as TryInto<Input<'a, T, I>>>::Error: Into<Error>,
{
    let input = input.try_into().map_err(Into::into)?;
    // A cover's two sheets and its ground are the most slots any block indexes.
    let slots = input.n().saturating_mul(2).saturating_add(1);
    Ok(match u32::try_from(slots).is_ok() {
        true => Factor::build(&input.reduce(config.scaling)?, &config, Storage::U32),
        false => Factor::build(&input.reduce(config.scaling)?, &config, Storage::U64),
    })
}

/// The scalar every stage computes in: `f32` or `f64`, each summed in `f64`, which holds it exactly.
pub trait Real: num_traits::Float + Debug + Send + Sync + 'static + seal::Real {}

impl Real for f32 {}
impl Real for f64 {}

/// The unsigned integer a reduction and its factor store each vertex index in: `u32` by default, `u64` past `u32::MAX` vertices.
pub(crate) trait Index: PrimInt + Unsigned + Send + Sync + 'static + seal::Index {
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

#[cfg(test)]
mod tests {
    use super::compensated_sum;

    /// The dropped `1.0` is the smaller operand as the addend in one order and as the running sum in the other.
    #[test]
    fn compensated_sum_recovers_the_term_a_plain_fold_drops() {
        for values in [[1e16, 1.0, -1e16], [1.0, 1e16, -1e16]] {
            assert_eq!(values.iter().sum::<f64>(), 0.0, "plain fold keeps the term");
            assert_eq!(compensated_sum(values.into_iter()), 1.0);
        }
    }
}
