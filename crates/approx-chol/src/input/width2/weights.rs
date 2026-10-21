//! The diagonal `W` of `A = V W Vᵀ`.

use super::error::{check, length, nonnegative, ViewError};
use crate::{widen, Real};

/// The diagonal `W`: per column of `V` as a slice, or one column's entry.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Weighting<S> {
    /// `W = I`.
    Unit,
    /// `W = diag(w)`, each `w` finite and `≥ 0`.
    Of(S),
    /// `W = diag(s²)`, each `s` finite, for a caller that keeps square roots.
    SqrtOf(S),
}

impl<T: Real> Weighting<T> {
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

/// `W`, one nonnegative entry per column of `V`, checked once so every view over the same columns can share it.
#[derive(Clone, Copy, Debug)]
pub struct Weights<'a, T>(Weighting<&'a [T]>);

impl<'a, T: Real> Weights<'a, T> {
    /// `W = I`.
    pub fn unit() -> Self {
        Self(Weighting::Unit)
    }

    /// `W = diag(w)`; each `w` finite and nonnegative.
    pub fn of(w: &'a [T]) -> Result<Self, ViewError> {
        check(w, nonnegative, weight)?;
        Ok(Self(Weighting::Of(w)))
    }

    /// `W = diag(s²)`, for a caller that keeps square roots; each `s` finite.
    pub fn sqrt_of(s: &'a [T]) -> Result<Self, ViewError> {
        check(s, |s| s.is_finite(), weight)?;
        Ok(Self(Weighting::SqrtOf(s)))
    }

    /// `W` over `columns` columns.
    pub(super) fn of_columns(self, columns: usize) -> Result<Weighting<&'a [T]>, ViewError> {
        if let Weighting::Of(w) | Weighting::SqrtOf(w) = self.0 {
            length(columns, w.len())?;
        }
        Ok(self.0)
    }
}

fn weight(entry: usize) -> ViewError {
    ViewError::InvalidWeight { entry }
}
