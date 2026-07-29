use core::cmp::Ordering;

use num_traits::{Float, NumCast};

/// Nothing but the scalar bound: the tolerances below are the algorithm's policy, not
/// the scalar's capability.
pub(crate) trait Real: Float + Send + Sync + 'static {}

impl<T> Real for T where T: Float + Send + Sync + 'static {}

/// Panics on an exotic `Float` rather than substituting: every substitute a call site
/// could pick silently corrupts the factor instead. `f32` and `f64` never fail.
#[inline]
pub(crate) fn count_as_scalar<T: Float, N: num_traits::ToPrimitive>(count: N) -> T {
    <T as NumCast>::from(count).expect("count is representable in T")
}

/// NaN last: `partial_cmp` returns `None` there, violating the total order Rust's
/// sorts require (1.81+ panics on it).
#[inline]
pub(crate) fn float_total_cmp<T: Float>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b)
        .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}
