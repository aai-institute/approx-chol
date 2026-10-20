//! A weighted edge list: `W` and the two ends of each column.

use num_traits::PrimInt;

use super::{check, index, length, Column, ViewError, Weighting};
use crate::{Index, Real};

/// `W`, one nonnegative entry per column of `V`, checked once so every view over the same columns can share it; a view without it has `W = I`.
#[derive(Clone, Copy, Debug)]
pub struct Weights<'a, T>(Weighting<&'a [T]>);

impl<'a, T: Real> Weights<'a, T> {
    /// `W = diag(w)`; each `w` finite and nonnegative.
    pub fn of(w: &'a [T]) -> Result<Self, ViewError> {
        check(w, |w| w.is_finite() && w >= T::zero(), weight)?;
        Ok(Self(Weighting::Of(w)))
    }

    /// `W = diag(s²)`, for a caller that keeps square roots; each `s` finite.
    pub fn sqrt_of(s: &'a [T]) -> Result<Self, ViewError> {
        check(s, |s| s.is_finite(), weight)?;
        Ok(Self(Weighting::SqrtOf(s)))
    }
}

fn weight(column: usize) -> ViewError {
    ViewError::InvalidWeight { column }
}

/// Edge `k` joins `a[k]` and `b[k]` under `weights`, `W = I` until given; the view fixes the numbering and the signs.
#[derive(Clone, Copy, Debug)]
pub(super) struct Edges<'a, T, I> {
    a: &'a [I],
    b: &'a [I],
    weights: Weights<'a, T>,
}

impl<'a, T: Real, I: PrimInt> Edges<'a, T, I> {
    /// Lengths are checked; the ends are the view's to check.
    pub(super) fn new(a: &'a [I], b: &'a [I]) -> Result<Self, ViewError> {
        length(a.len(), b.len())?;
        let weights = Weights(Weighting::Unit);
        Ok(Self { a, b, weights })
    }

    /// One weight per column.
    pub(super) fn with_weights(self, weights: Weights<'a, T>) -> Result<Self, ViewError> {
        if let Weighting::Of(w) | Weighting::SqrtOf(w) = weights.0 {
            length(self.a.len(), w.len())?;
        }
        Ok(Self { weights, ..self })
    }

    /// Column `k` is `v_a(k) e_a + v_b(k) e_(offset + b)`; matched once, then looped per case.
    #[inline(always)]
    pub(super) fn for_each_column<S: Index>(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        visit: impl FnMut(Column<T, S>),
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
    fn columns<S: Index>(
        self,
        offset: usize,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        weight: impl Fn(usize) -> Weighting<T>,
        mut visit: impl FnMut(Column<T, S>),
    ) {
        for (k, (a, b)) in self.a.iter().zip(self.b).enumerate() {
            visit(Column {
                a: S::from_index(index(*a)),
                b: S::from_index(offset + index(*b)),
                v_a: v_a(k),
                v_b: v_b(k),
                weight: weight(k),
            });
        }
    }
}
