//! A column's weight `w_k` in `A = Σ_k w_k v_k v_kᵀ`.

use crate::{widen, Real};

/// Column `k`'s `w_k`: none, a Laplacian's edge weight, or the square of `V`'s column scale.
#[derive(Clone, Copy, Debug)]
pub(super) enum Weight<T> {
    /// No `W`.
    Unit,
    /// `w`.
    Of(T),
    /// `w = s²`, kept as `s` so `(s·x)·(s·y)` neither overflows nor underflows through `s²`.
    Scale(T),
}

impl<T: Real> Weight<T> {
    /// `w·x·y`, as `(s·x)·(s·y)` under a scale.
    pub(super) fn bilinear(self, x: f64, y: f64) -> f64 {
        match self {
            Self::Unit => x * y,
            Self::Of(w) => widen(w) * x * y,
            Self::Scale(s) => (widen(s) * x) * (widen(s) * y),
        }
    }

    /// As `bilinear`, a nonzero product that underflows kept as the least subnormal, so the range screen rejects it (#163).
    pub(super) fn bilinear_unflushed(self, x: f64, y: f64) -> f64 {
        let product = self.bilinear(x, y);
        match product == 0.0 && x != 0.0 && y != 0.0 && !self.is_zero() {
            true => f64::from_bits(1).copysign(product),
            false => product,
        }
    }

    pub(super) fn is_zero(self) -> bool {
        match self {
            Self::Unit => false,
            Self::Of(w) | Self::Scale(w) => w == T::zero(),
        }
    }
}
