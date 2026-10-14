//! A weighted edge list: `W` and the two ends of each column.

use num_traits::PrimInt;

use super::{index, Column, ViewError, Weighting};
use crate::Real;

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
        check(w, |w| w.is_finite() && w >= T::zero())?;
        Ok(Self(Weighting::Of(w)))
    }

    /// `W = diag(s²)`, for a caller that keeps square roots; each `s` finite.
    pub fn sqrt_of(s: &'a [T]) -> Result<Self, ViewError> {
        check(s, |s| s.is_finite())?;
        Ok(Self(Weighting::SqrtOf(s)))
    }

    /// One entry per column.
    pub(super) fn check(self, columns: usize) -> Result<(), ViewError> {
        let found = match self.0 {
            Weighting::Unit => return Ok(()),
            Weighting::Of(entries) | Weighting::SqrtOf(entries) => entries.len(),
        };
        match found == columns {
            true => Ok(()),
            false => Err(ViewError::LengthMismatch {
                expected: columns,
                found,
            }),
        }
    }
}

impl<T> Default for Weights<'_, T> {
    /// `W = I`.
    fn default() -> Self {
        Self(Weighting::Unit)
    }
}

fn check<T: Real>(entries: &[T], valid: fn(T) -> bool) -> Result<(), ViewError> {
    match entries.iter().position(|&x| !valid(x)) {
        Some(column) => Err(ViewError::InvalidWeight { column }),
        None => Ok(()),
    }
}

/// Edge `k` joins `a[k]` and `b[k]` under `weights`; the view fixes the numbering and the signs.
#[derive(Clone, Copy, Debug)]
pub(super) struct Edges<'a, T, I> {
    a: &'a [I],
    b: &'a [I],
    weights: Weights<'a, T>,
}

impl<'a, T: Real, I: PrimInt> Edges<'a, T, I> {
    /// Lengths and weights are checked; the ends are the view's to check.
    pub(super) fn new(a: &'a [I], b: &'a [I], weights: Weights<'a, T>) -> Result<Self, ViewError> {
        if a.len() != b.len() {
            return Err(ViewError::LengthMismatch {
                expected: a.len(),
                found: b.len(),
            });
        }
        weights.check(a.len())?;
        Ok(Self { a, b, weights })
    }

    /// Column `k` is `v_a(k) e_a + v_b(k) e_(offset + b)`; matched once, then looped per case.
    #[inline(always)]
    pub(super) fn for_each_column(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        visit: impl FnMut(Column<T>),
    ) {
        match self.weights.0 {
            Weighting::Unit => self.columns(offset, v_a, v_b, |_| Weighting::Unit, visit),
            Weighting::Of(w) => self.columns(offset, v_a, v_b, |k| Weighting::Of(w[k]), visit),
            Weighting::SqrtOf(s) => {
                self.columns(offset, v_a, v_b, |k| Weighting::SqrtOf(s[k]), visit)
            }
        }
    }

    #[inline(always)]
    fn columns(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        weight: impl Fn(usize) -> Weighting<T>,
        mut visit: impl FnMut(Column<T>),
    ) {
        for (k, (a, b)) in self.a.iter().zip(self.b).enumerate() {
            visit(Column {
                a: index(*a) as u32,
                b: (offset + index(*b)) as u32,
                v_a: v_a(k),
                v_b: v_b(k),
                weight: weight(k),
            });
        }
    }
}
