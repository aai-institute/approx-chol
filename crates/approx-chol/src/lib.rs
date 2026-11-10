//! Approximate Cholesky of symmetric generalized diagonally dominant `A`, one irreducible block at a time.
//!
//! ```
//! use approx_chol::input::CsrRef;
//! use approx_chol::{factorize, Threads};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let row_ptrs    = [0u32, 2, 5, 8, 10];
//! let col_indices = [0u32, 1, 0, 1, 2, 1, 2, 3, 2, 3];
//! let values      = [1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0];
//!
//! let csr = CsrRef::new(4, &row_ptrs, &col_indices, &values)?;
//! let mut x = vec![1.0, -1.0, 1.0, -1.0];
//! factorize(csr)?.apply(&mut x, Threads::All)?;
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

mod adapters;
mod batch;
mod error;
mod factor;
pub mod input;
mod order;
mod sgdd;
mod storage;
mod threads;

use std::array;
use std::fmt::Debug;

use num_traits::{float::TotalOrder, Float, PrimInt, Unsigned};

use batch::Batch;
pub use dyn_stack::{MemBuffer, MemStack, StackReq};
pub use error::BuildError;
pub use factor::{
    ApplyError, ApproxInverse, ColumnMajor, Columns, Component, Config, CoverSurplus,
    FrustratedRoute, Header, HeaderMismatch, LayoutError, MatrixMut, Sampling, Scalar, SchurMode,
};
use input::{Input, Side};
pub use sgdd::{Clamp, ScalingOutcome, ScalingPolicy};
use sgdd::{Complement, Reduced};
pub use threads::Threads;

/// Factorizes any [`Input`] with [`Config::default`] and [`ScalingPolicy::default`].
pub fn factorize<'a, T, I, M>(input: M) -> Result<ApproxInverse<T>, BuildError>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: Into<Input<'a, T, I>>,
{
    factorize_with(input, Config::default(), ScalingPolicy::default())
}

/// Factorizes any [`Input`]: [`Sgdd::reduce`] under `scaling`, then [`Sgdd::factor`] under `config`.
pub fn factorize_with<'a, T, I, M>(
    input: M,
    config: Config,
    scaling: ScalingPolicy,
) -> Result<ApproxInverse<T>, BuildError>
where
    T: Real,
    I: PrimInt + Sync + 'a,
    M: Into<Input<'a, T, I>>,
{
    Ok(Sgdd::reduce(input, scaling)?.factor(&config))
}

/// Factors a bipartite `A`'s Schur complement onto its `kept` side, formed under [`Config::complement`] and factored as [`factorize_with`] would; the factor's vertex `k` is the kept side's `k`-th.
pub fn factorize_complement<T: Real, I: PrimInt + Sync>(
    input: input::BipartiteFactorWidth2<'_, T, I>,
    kept: Side,
    config: Config,
    scaling: ScalingPolicy,
) -> Result<ApproxInverse<T>, BuildError> {
    Ok(match fits_u32(input.n()) {
        true => ApproxInverse::build(
            Complement::<T, u32>::reduce_onto(input, kept, scaling, Threads::All)?,
            &config,
            Threads::All,
        ),
        false => ApproxInverse::build(
            Complement::<T, u64>::reduce_onto(input, kept, scaling, Threads::All)?,
            &config,
            Threads::All,
        ),
    })
}

/// Whether `u32` indexes a cover's two sheets and its ground, the most slots any block of `n` vertices takes.
fn fits_u32(n: usize) -> bool {
    u32::try_from(n.saturating_mul(2).saturating_add(1)).is_ok()
}

/// Built only by [`Sgdd::reduce`]: `A` in symmetric generalized diagonally dominant form, each irreducible block scaled to dominance.
#[derive(Clone, Debug)]
pub struct Sgdd<T = f64>(Indexed<T>);

/// The reduction in the narrowest index its cover's `2n + 1` slots fit.
#[derive(Clone, Debug)]
enum Indexed<T> {
    U32(Reduced<T, u32>),
    U64(Reduced<T, u64>),
}

impl<T: Real> Sgdd<T> {
    /// Reduces any [`Input`], scaling each block short of dominance under `scaling`; nothing after this reads `input`.
    pub fn reduce<'a, I, M>(input: M, scaling: ScalingPolicy) -> Result<Self, BuildError>
    where
        I: PrimInt + Sync + 'a,
        M: Into<Input<'a, T, I>>,
    {
        let input = input.into();
        Ok(Self(match fits_u32(input.n()) {
            true => Indexed::U32(Reduced::reduce(input, scaling, Threads::All)?),
            false => Indexed::U64(Reduced::reduce(input, scaling, Threads::All)?),
        }))
    }

    /// Factors each block in parallel; clone first to factor again under another `config`.
    pub fn factor(self, config: &Config) -> ApproxInverse<T> {
        match self.0 {
            Indexed::U32(summands) => ApproxInverse::build(summands, config, Threads::All),
            Indexed::U64(summands) => ApproxInverse::build(summands, config, Threads::All),
        }
    }
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
    pub trait Real {
        /// What a persisted factor's [`Header`](crate::Header) records for it.
        const SCALAR: crate::Scalar;
    }

    impl Real for f32 {
        const SCALAR: crate::Scalar = crate::Scalar::F32;
    }
    impl Real for f64 {
        const SCALAR: crate::Scalar = crate::Scalar::F64;
    }
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
    let a = x.abs();
    // `&`, not `&&`: branch-free, so whole-slice checks vectorize; `a < ∞` also rejects NaN.
    (a < T::infinity()) & ((a == T::zero()) | (a >= T::min_positive_value() / T::epsilon()))
}

/// In range and at most `MAX/4`, so every row sum, pivot and surplus stays finite.
fn diagonal_in_range<T: Real>(d: T) -> bool {
    in_range(d) && d <= T::max_value() / (T::one() + T::one() + T::one() + T::one())
}

/// The index of the largest `key`: the last entry unless an earlier one strictly passes it, then the first of the largest earlier ones; a NaN wins no comparison, and an empty slice gives `0`.
fn argmax<T: Real>(values: &[T], key: impl Fn(T) -> T) -> usize {
    let last = values.len().saturating_sub(1);
    (0..last).fold(last, |best, i| match key(values[i]) > key(values[best]) {
        true => i,
        false => best,
    })
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
fn compensated_sum<X: Copy, W: Batch<f64>>(x: &[X], term: impl Fn(usize, X) -> W) -> W {
    // Outlined, so a short sum's caller spills no lane state.
    #[inline(never)]
    fn lanes<X: Copy, W: Batch<f64>>(x: &[X], term: impl Fn(usize, X) -> W) -> W {
        Sum2::of(x, term).total()
    }
    if x.len() >= 20 {
        return lanes(x, term);
    }
    let terms = x.iter().enumerate().map(|(i, &x)| term(i, x));
    let (sum, carry) = terms.fold((W::zero(), W::zero()), |(sum, carry), x| {
        let (sum, error) = two_sum(sum, x);
        (sum, carry.add(error))
    });
    sum.add(carry)
}

/// Ogita–Rump–Oishi Sum2 (SISC 26(6), 2005) in eight fixed TwoSum lanes: off by at most `ε|Σx| + (nε)² Σ|x|`.
#[derive(Clone, Copy)]
struct Sum2<W = f64> {
    sums: [W; 8],
    carries: [W; 8],
}

impl<W: Batch<f64>> Default for Sum2<W> {
    fn default() -> Self {
        Self {
            sums: [W::zero(); 8],
            carries: [W::zero(); 8],
        }
    }
}

impl<W: Batch<f64>> Sum2<W> {
    fn of<X: Copy>(x: &[X], term: impl Fn(usize, X) -> W) -> Self {
        Self::default().plus(x, term)
    }

    // Inlined into a large caller, SLP leaves the lanes scalar.
    #[inline(never)]
    fn plus<X: Copy>(mut self, x: &[X], term: impl Fn(usize, X) -> W) -> Self {
        let chunks = x.chunks_exact(8);
        let rest = chunks.remainder();
        for (k, chunk) in chunks.enumerate() {
            self.add(array::from_fn(|lane| term(8 * k + lane, chunk[lane])));
        }
        if !rest.is_empty() {
            let start = x.len() - rest.len();
            // Zero-padded rather than a shorter pass, so every lane stays in one vector register.
            self.add(array::from_fn(|lane| {
                rest.get(lane).map_or(W::zero(), |&x| term(start + lane, x))
            }));
        }
        self
    }

    // If outlined, every chunk round-trips the lanes through memory.
    #[inline(always)]
    fn add(&mut self, terms: [W; 8]) {
        let lanes = (self.sums.iter_mut().zip(&mut self.carries)).zip(terms);
        for ((sum, carry), x) in lanes {
            let error;
            (*sum, error) = two_sum(*sum, x);
            *carry = carry.add(error);
        }
    }

    fn total(&self) -> W {
        let first = (self.sums[0], W::zero());
        let (sum, carry) = self.sums[1..].iter().fold(first, |(sum, carry), &x| {
            let (sum, error) = two_sum(sum, x);
            (sum, carry.add(error))
        });
        // From `−0`, as `Iterator::sum` starts, so a zero keeps its sign.
        let carries = (self.carries.iter()).fold(W::zero().neg(), |total, &c| total.add(c));
        sum.add(carry.add(carries))
    }
}

/// `a + b` and its rounding error, exactly (Knuth's TwoSum).
fn two_sum<W: Batch<f64>>(a: W, b: W) -> (W, W) {
    let sum = a.add(b);
    let b_part = sum.sub(a);
    (sum, (a.sub(sum.sub(b_part))).add(b.sub(b_part)))
}

#[cfg(test)]
mod tests {
    use super::{argmax, compensated_sum, Sum2};

    /// The gauge pin's choice, which a factor's bits follow: a tie with the last entry keeps the last, else the first of the earlier maxima; a NaN never wins.
    #[test]
    fn argmax_keeps_the_last_on_a_tie_with_it_else_the_first_maximum() {
        let id = |x: f64| x;
        assert_eq!(argmax(&[3.0, 1.0, 3.0], id), 2);
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0], id), 1);
        assert_eq!(argmax(&[f64::NAN, 1.0, 2.0], id), 2);
        assert_eq!(argmax(&[1.0, 2.0, f64::NAN], id), 2);
        assert_eq!(argmax(&[-4.0, 1.0, 4.0], f64::abs), 2);
        assert_eq!(argmax::<f64>(&[], id), 0);
    }

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
