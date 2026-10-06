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
