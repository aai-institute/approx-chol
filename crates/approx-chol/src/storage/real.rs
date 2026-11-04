//! The scalar and index every stage computes in, the sums they take and the range screens they pass.

use std::array;
use std::fmt::Debug;

use num_traits::{float::TotalOrder, Float, PrimInt, Unsigned};

/// The scalar every stage computes in: `f32` or `f64`, each summed in `f64`, which holds it exactly.
pub trait Real: Float + TotalOrder + Debug + Send + Sync + 'static + seal::Real {}

impl Real for f32 {}
impl Real for f64 {}

/// The unsigned integer a reduction and its factor store each vertex index in: `u32` while Cholesky's `fits_u32`, else `u64`.
pub(crate) trait Index: PrimInt + Unsigned + Send + Sync + 'static {
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

/// `scale · Σ|x|` in eight fixed lanes, uncompensated: it vectorizes and sums alike at any thread count, for nonnegative sums nothing cancels in.
pub(crate) fn lane_sum<T: Real>(x: &[T], scale: f64) -> f64 {
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
pub(crate) fn compensated_sum<T: Copy>(x: &[T], term: impl Fn(usize, T) -> f64) -> f64 {
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

/// Ogita–Rump–Oishi Sum2 (SISC 26(6), 2005) in eight fixed TwoSum lanes: off by at most `ε|Σx| + (nε)² Σ|x|`.
#[derive(Clone, Copy, Default)]
pub(crate) struct Sum2 {
    sums: [f64; 8],
    carries: [f64; 8],
}

impl Sum2 {
    pub(crate) fn of<T: Copy>(x: &[T], term: impl Fn(usize, T) -> f64) -> Self {
        Self::default().plus(x, term)
    }

    // Inlined into a large caller, SLP leaves the lanes scalar.
    #[inline(never)]
    pub(super) fn plus<T: Copy>(mut self, x: &[T], term: impl Fn(usize, T) -> f64) -> Self {
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

    // If outlined, every chunk round-trips the lanes through memory.
    #[inline(always)]
    fn add(&mut self, terms: [f64; 8]) {
        let lanes = (self.sums.iter_mut().zip(&mut self.carries)).zip(terms);
        for ((sum, carry), x) in lanes {
            let error;
            (*sum, error) = two_sum(*sum, x);
            *carry += error;
        }
    }

    pub(crate) fn total(&self) -> f64 {
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
