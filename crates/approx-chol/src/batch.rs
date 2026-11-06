//! What a solve computes on: one right-hand side, or several solved side by side.

use crate::Real;

/// A value per right-hand side, every operation per value, so each rounds as it would solved alone.
pub(crate) trait Batch<T: Real>: Copy + Send + Sync {
    type Wide: Batch<f64>;

    fn zero() -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn neg(self) -> Self;
    fn mul(self, by: T) -> Self;
    fn div(self, by: T) -> Self;
    fn widen(self) -> Self::Wide;
    fn narrow(wide: Self::Wide) -> Self;
}

impl<T: Real> Batch<T> for T {
    type Wide = f64;

    #[inline(always)]
    fn zero() -> Self {
        T::zero()
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self - other
    }

    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn mul(self, by: T) -> Self {
        self * by
    }

    #[inline(always)]
    fn div(self, by: T) -> Self {
        self / by
    }

    #[inline(always)]
    fn widen(self) -> f64 {
        crate::widen(self)
    }

    #[inline(always)]
    fn narrow(wide: f64) -> Self {
        crate::narrow(wide)
    }
}
