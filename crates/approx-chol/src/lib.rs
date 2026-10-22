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

#![deny(missing_docs, unreachable_pub)]

/// Serde through `remote = "Self"`, every decoded value passed through its type's `checked`.
#[cfg(feature = "serde")]
macro_rules! checked_serde {
    ($ty:ident) => {
        impl<T: serde::Serialize, I: crate::Index + serde::Serialize> serde::Serialize
            for $ty<T, I>
        {
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

use std::array;
use std::fmt::Debug;

use num_traits::{float::TotalOrder, Float, PrimInt, Unsigned};

pub use error::Error;
use factor::Storage;
#[cfg(feature = "serde")]
pub use factor::FORMAT_VERSION as FACTOR_FORMAT_VERSION;
pub use factor::{
    Backend, BlockFactor, Config, Factor, InnerParallelism, SchurMode, SolveError, Vertices,
};
pub use input::{CsrError, CsrRef, IndexKind, Input, OwnedCsr};
use sgdd::Sgdd;
pub use sgdd::{ReduceError, ScalingPolicy, ScalingReport};

/// Factorizes any [`Input`] with [`Config::default`].
pub fn factorize<'a, T, I, M>(input: M) -> Result<Factor<T>, Error>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: TryInto<Input<'a, T, I>>,
    <M as TryInto<Input<'a, T, I>>>::Error: Into<Error>,
{
    factorize_with(input, Config::default())
}

/// Factorizes any [`Input`]: reduces it to its irreducible blocks, then factors each in parallel.
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
        true => Factor::build(Sgdd::reduce(input, config.scaling)?, &config, Storage::U32),
        false => Factor::build(Sgdd::reduce(input, config.scaling)?, &config, Storage::U64),
    })
}

/// The scalar every stage computes in: `f32` or `f64`, each summed in `f64`, which holds it exactly.
pub trait Real: Float + TotalOrder + Debug + Send + Sync + 'static + seal::Real {}

impl Real for f32 {}
impl Real for f64 {}

/// The unsigned integer a reduction and its factor store each vertex index in: `u32` while a cover's `2n + 1` slots fit it, else `u64`.
trait Index: PrimInt + Unsigned + Send + Sync + 'static {
    /// `self` as `usize`, so `x[v.index()]` reads vertex `v` of any `x` over `A`'s vertices.
    fn index(self) -> usize;

    /// `k` as `Self`, truncating: every caller passes a value a checked count bounds.
    fn from_index(k: usize) -> Self;
}

mod seal {
    /// Only the crate implements [`Real`](super::Real), so every `T` widens to `f64` exactly.
    pub trait Real {}

    impl Real for f32 {}
    impl Real for f64 {}
}

macro_rules! index {
    ($int:ty) => {
        impl Index for $int {
            #[inline(always)]
            fn index(self) -> usize {
                self as usize
            }

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

/// `units` per task so each carries `2^bits` of `work`, in `u64` as a 32-bit `usize` wraps past `2^(32-bits)` units.
fn per_task(units: usize, work: u64, bits: u32) -> usize {
    let per = ((units as u64) << bits).div_ceil(work.max(1));
    usize::try_from(per).unwrap_or(usize::MAX).max(1)
}

/// Finite, and zero or at least `MIN_POSITIVE/EPSILON`, so no product turns subnormal (#163).
fn in_range<T: Real>(x: T) -> bool {
    x.is_finite() && (x == T::zero() || x.abs() >= T::min_positive_value() / T::epsilon())
}

/// Every `f64` sum starts here: a `Float` always casts up.
fn widen<T: Real>(x: T) -> f64 {
    x.to_f64().expect("a float casts to f64")
}

/// An `f64` sum in `T`: past `T::MAX` it reads infinite, which every range screen rejects.
fn narrow<T: Real>(sum: f64) -> T {
    T::from(sum).unwrap_or_else(T::infinity)
}

/// `scale · Σ|x|` in eight fixed lanes, uncompensated: it vectorizes and sums alike at any thread count, for nonnegative sums nothing cancels in.
fn lane_sum<T: Real>(x: &[T], scale: f64) -> f64 {
    let mut lanes = [0.0f64; 8];
    let chunks = x.chunks_exact(8);
    let rest = chunks.remainder();
    for chunk in chunks {
        (lanes.iter_mut().zip(chunk)).for_each(|(lane, &x)| *lane += scale * widen(x.abs()));
    }
    (rest.iter()).fold(lanes.iter().sum::<f64>(), |sum, &x| {
        sum + scale * widen(x.abs())
    })
}

/// `Σ term(i, x_i)` as [`Sum2`] gives it; serial below 20 terms, where the measured lane setup and merge cost more than they save.
fn compensated_sum<T: Copy>(x: &[T], term: impl Fn(usize, T) -> f64) -> f64 {
    // Outlined, so a short sum's caller spills no lane state.
    #[inline(never)]
    fn lanes<T: Copy>(x: &[T], term: impl Fn(usize, T) -> f64) -> f64 {
        Sum2::of(x, term).total()
    }
    if x.len() >= 20 {
        return lanes(x, term);
    }
    let terms = x.iter().enumerate().map(|(i, &x)| term(i, x));
    let (sum, carry) = terms.fold((0.0, 0.0), |(sum, carry), x| {
        let (sum, error) = two_sum(sum, x);
        (sum, carry + error)
    });
    sum + carry
}

/// Ogita–Rump–Oishi Sum2 in eight fixed TwoSum lanes, as AccurateArithmetic.jl vectorizes it: off by at most `ε|Σx| + (nε)² Σ|x|`.
#[derive(Clone, Copy, Default)]
struct Sum2 {
    sums: [f64; 8],
    carries: [f64; 8],
}

impl Sum2 {
    fn of<T: Copy>(x: &[T], term: impl Fn(usize, T) -> f64) -> Self {
        Self::default().plus(x, term)
    }

    // Inlined into a large caller, SLP leaves the lanes scalar.
    #[inline(never)]
    fn plus<T: Copy>(mut self, x: &[T], term: impl Fn(usize, T) -> f64) -> Self {
        let chunks = x.chunks_exact(8);
        let rest = chunks.remainder();
        for (k, chunk) in chunks.enumerate() {
            self.add(array::from_fn(|lane| term(8 * k + lane, chunk[lane])));
        }
        if !rest.is_empty() {
            let start = x.len() - rest.len();
            // Zero-padded rather than a shorter pass, so every lane stays in one vector register.
            self.add(array::from_fn(|lane| {
                rest.get(lane).map_or(0.0, |&x| term(start + lane, x))
            }));
        }
        self
    }

    // Outlined, every chunk round-trips the lanes through memory.
    #[inline(always)]
    fn add(&mut self, terms: [f64; 8]) {
        let lanes = (self.sums.iter_mut().zip(&mut self.carries)).zip(terms);
        for ((sum, carry), x) in lanes {
            let error;
            (*sum, error) = two_sum(*sum, x);
            *carry += error;
        }
    }

    fn total(&self) -> f64 {
        let first = (self.sums[0], 0.0);
        let (sum, carry) = self.sums[1..].iter().fold(first, |(sum, carry), &x| {
            let (sum, error) = two_sum(sum, x);
            (sum, carry + error)
        });
        sum + (carry + self.carries.iter().sum::<f64>())
    }
}

/// `a + b` and its rounding error, exactly (Knuth's TwoSum).
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_part = sum - a;
    (sum, (a - (sum - b_part)) + (b - b_part))
}

#[cfg(test)]
mod tests {
    use super::{compensated_sum, Sum2};

    /// The dropped `1.0` is the smaller operand as the addend in one order and as the running sum in the other.
    #[test]
    fn compensated_sum_recovers_the_term_a_plain_fold_drops() {
        for values in [[1e16, 1.0, -1e16], [1.0, 1e16, -1e16]] {
            assert_eq!(values.iter().sum::<f64>(), 0.0, "plain fold keeps the term");
            assert_eq!(compensated_sum(&values, |_, x| x), 1.0, "serial");
            assert_eq!(Sum2::of(&values, |_, x| x).total(), 1.0, "lanes");
        }
    }

    /// Strided so each `±1e16·k` pair meets the `1.0`s across lanes and chunks: only every lane's carry keeps them.
    #[test]
    fn compensated_sum_holds_its_bound_across_lanes() {
        let (triples, stride) = (1001, 5);
        let n = 3 * triples;
        let term = |j: usize| match j % 3 {
            0 => 1e16 * (j % 7 + 1) as f64,
            1 => 1.0,
            _ => -1e16 * ((j - 2) % 7 + 1) as f64,
        };
        let values: Vec<f64> = (0..n).map(|i| term(i * stride % n)).collect();
        let exact = triples as f64;
        let epsilon = f64::EPSILON;
        let magnitude: f64 = values.iter().map(|x| x.abs()).sum();
        let bound = epsilon * exact + (n as f64 * epsilon).powi(2) * magnitude;
        assert!(
            (values.iter().sum::<f64>() - exact).abs() > bound,
            "plain sum is within the bound"
        );
        let sum = Sum2::of(&values, |_, x| x).total();
        assert!((sum - exact).abs() <= bound, "{sum} vs {exact}");
    }
}
