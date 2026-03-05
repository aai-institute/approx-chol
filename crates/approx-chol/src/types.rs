use core::cmp::Ordering;

use num_traits::{Float, NumCast};

/// Internal scalar trait supported by approximate Cholesky kernels.
pub(crate) trait Real: Float + Send + Sync + 'static {
    /// Near-zero threshold for numeric guards.
    fn near_zero() -> Self;
}

impl<T> Real for T
where
    T: Float + Send + Sync + 'static,
{
    #[inline]
    fn near_zero() -> Self {
        if core::mem::size_of::<T>() <= 4 {
            <T as NumCast>::from(1e-6_f64).unwrap_or_else(T::epsilon)
        } else {
            <T as NumCast>::from(1e-14_f64).unwrap_or_else(T::epsilon)
        }
    }
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
