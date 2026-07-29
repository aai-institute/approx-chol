use core::cmp::Ordering;

use num_traits::{Float, NumCast};

/// Nothing but the scalar bound: the tolerances below are the algorithm's policy, not
/// the scalar's capability.
pub(crate) trait Real: Float + Send + Sync + 'static {}

impl<T> Real for T where T: Float + Send + Sync + 'static {}

/// `single` for `f32`-width scalars, `double` otherwise.
#[inline]
fn by_precision<T: Float>(single: f64, double: f64) -> T {
    let value = if core::mem::size_of::<T>() <= 4 {
        single
    } else {
        double
    };
    <T as NumCast>::from(value).unwrap_or_else(T::epsilon)
}

/// The scale below which a weight or pivot is not resolvable, so dividing by it is
/// what breaks rather than the value being wrong.
#[inline]
pub(crate) fn near_zero<T: Float>() -> T {
    by_precision(1e-6, 1e-14)
}

/// A departure from zero smaller than this fraction of the row's magnitude is not
/// real — four orders coarser than [`near_zero`], which answers a different question.
#[inline]
pub(crate) fn row_sum_slack<T: Float>() -> T {
    by_precision(1e-6, 1e-10)
}

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
