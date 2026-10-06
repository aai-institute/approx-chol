use num_traits::{Float, NumCast};

/// Only the scalar bound: tolerances are the algorithm's policy, not the scalar's capability.
pub(crate) trait Real: Float + Send + Sync + 'static {}

impl<T> Real for T where T: Float + Send + Sync + 'static {}

/// Panics on an exotic `Float`, since any substitute would silently corrupt the factor.
#[inline]
pub(crate) fn count_as_scalar<T: Float, N: num_traits::ToPrimitive>(count: N) -> T {
    <T as NumCast>::from(count).expect("count is representable in T")
}
