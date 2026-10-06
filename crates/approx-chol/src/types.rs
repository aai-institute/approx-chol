use core::cmp::Ordering;

use num_traits::{Float, NumCast};

/// Only the scalar bound: tolerances are the algorithm's policy, not the scalar's capability.
pub(crate) trait Real: Float + Send + Sync + 'static {}

impl<T> Real for T where T: Float + Send + Sync + 'static {}

/// Panics on an exotic `Float`, since any substitute would silently corrupt the factor.
#[inline]
pub(crate) fn count_as_scalar<T: Float, N: num_traits::ToPrimitive>(count: N) -> T {
    <T as NumCast>::from(count).expect("count is representable in T")
}

/// NaN last: `partial_cmp`'s `None` breaks the total order sorts require (1.81+ panics).
#[inline]
pub(crate) fn float_total_cmp<T: Float>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}
