//! What a solve computes on: one right-hand side, or several solved side by side.

use std::array;

use crate::Real;

/// A value per right-hand side, every operation per value, so each rounds as it would solved alone, a NaN's sign and payload aside: the optimizer may compile the instances apart, and a NaN's sign may differ between them (#437).
pub(crate) trait Batch<T: Real>: Copy + Send + Sync {
    type Wide: Batch<f64>;

    /// Right-hand sides side by side.
    const LANES: usize;

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

    const LANES: usize = 1;

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

/// `K` right-hand sides side by side, each lane computed as the scalar instance computes it alone.
impl<T: Real, const K: usize> Batch<T> for [T; K] {
    type Wide = [f64; K];

    const LANES: usize = K;

    #[inline(always)]
    fn zero() -> Self {
        [T::zero(); K]
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        array::from_fn(|i| self[i] + other[i])
    }

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        array::from_fn(|i| self[i] - other[i])
    }

    #[inline(always)]
    fn neg(self) -> Self {
        self.map(|a| -a)
    }

    #[inline(always)]
    fn mul(self, by: T) -> Self {
        self.map(|a| a * by)
    }

    #[inline(always)]
    fn div(self, by: T) -> Self {
        self.map(|a| a / by)
    }

    #[inline(always)]
    fn widen(self) -> [f64; K] {
        self.map(crate::widen)
    }

    #[inline(always)]
    fn narrow(wide: [f64; K]) -> Self {
        wide.map(crate::narrow)
    }
}
