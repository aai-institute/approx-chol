use core::cmp::Ordering;

use num_traits::{Float, NumCast};

/// Internal scalar trait supported by approximate Cholesky kernels.
pub(crate) trait Real: Float + Send + Sync + 'static {
    /// Pick a threshold by the type's precision: `single` for `f32`-width
    /// scalars, `double` otherwise.
    fn by_precision(single: f64, double: f64) -> Self;

    /// Near-zero threshold for numeric guards.
    fn near_zero() -> Self {
        Self::by_precision(1e-6, 1e-14)
    }
}

impl<T> Real for T
where
    T: Float + Send + Sync + 'static,
{
    #[inline]
    fn by_precision(single: f64, double: f64) -> Self {
        let value = if core::mem::size_of::<T>() <= 4 {
            single
        } else {
            double
        };
        <T as NumCast>::from(value).unwrap_or_else(T::epsilon)
    }
}

/// A count (edge multiplicity, neighbor count) as a scalar, for the weight and
/// sort-key formulas that multiply or divide by it.
///
/// `f32` and `f64` represent every such count, so the cast only fails for an
/// exotic `Float`. It panics there rather than substituting a value: the call
/// sites previously disagreed about the substitute — skip the neighbor, use one,
/// skip the whole edge split, panic — and all but the last silently corrupt the
/// factor.
#[inline]
pub(crate) fn count_as_scalar<T: Float, N: num_traits::ToPrimitive>(count: N) -> T {
    <T as NumCast>::from(count).expect("count is representable in T")
}

/// Total ordering for floats. NaN sorts last.
///
/// `partial_cmp` returns `None` when NaN is involved, which violates the total
/// order required by Rust's sort algorithms (Rust 1.81+ panics on violation).
/// This function provides a proper total order by placing NaN after all non-NaN.
#[inline]
pub(crate) fn float_total_cmp<T: Float>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}
