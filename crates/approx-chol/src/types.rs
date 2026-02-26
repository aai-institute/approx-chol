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
            <T as NumCast>::from(1e-6_f64).expect("finite threshold")
        } else {
            <T as NumCast>::from(1e-14_f64).expect("finite threshold")
        }
    }
}
