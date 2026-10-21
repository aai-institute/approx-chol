//! The diagonal `W` of `A = V W Vᵀ`.

use super::error::{check, length, nonnegative, ViewError};
use crate::{widen, Real};

/// `W`: one entry per column as a slice, checked by the constructor that takes it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum Weights<S> {
    /// `W = I`.
    Unit,
    /// `W = diag(w)`, each `w` finite and `≥ 0`.
    Of(S),
    /// `W = diag(s²)`, each `s` finite, for a caller that keeps square roots.
    SqrtOf(S),
}

impl<T: Real> Weights<T> {
    /// `w·x·y`, as `(s·x)·(s·y)` under `SqrtOf`.
    pub(crate) fn bilinear(self, x: f64, y: f64) -> f64 {
        match self {
            Self::Unit => x * y,
            Self::Of(w) => widen(w) * x * y,
            Self::SqrtOf(s) => (widen(s) * x) * (widen(s) * y),
        }
    }

    /// As `bilinear`, a nonzero product that underflows kept as the least subnormal, so the range screen rejects it (#163).
    pub(crate) fn bilinear_unflushed(self, x: f64, y: f64) -> f64 {
        let product = self.bilinear(x, y);
        match product == 0.0 && x != 0.0 && y != 0.0 && !self.is_zero() {
            true => f64::from_bits(1).copysign(product),
            false => product,
        }
    }

    pub(crate) fn is_zero(self) -> bool {
        match self {
            Self::Unit => false,
            Self::Of(w) | Self::SqrtOf(w) => w == T::zero(),
        }
    }
}

impl<T: Real> Weights<&[T]> {
    /// `W` over `columns` columns, each entry checked.
    pub(super) fn checked(self, columns: usize) -> Result<Self, ViewError> {
        if let Self::Of(w) | Self::SqrtOf(w) = self {
            length(columns, w.len())?;
        }
        match self {
            Self::Unit => {}
            Self::Of(w) => check(w, nonnegative, weight)?,
            Self::SqrtOf(s) => check(s, |s: T| s.is_finite(), weight)?,
        }
        Ok(self)
    }
}

fn weight(entry: usize) -> ViewError {
    ViewError::InvalidWeight { entry }
}
