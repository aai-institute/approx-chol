//! A weighted edge list, `W` and the two ends of each column, shared by the views whose `V` holds fixed signs.

use num_traits::PrimInt;

use super::{index, Column, ViewError, Weight};
use crate::Real;

/// The diagonal `W`, one nonnegative entry per column of `V`.
#[derive(Clone, Copy, Debug)]
pub enum Weights<'a, T> {
    /// `W = I`.
    Unit,
    /// `W = diag(w)`.
    Of(&'a [T]),
    /// `W = diag(s²)`, for a caller that keeps square roots.
    SqrtOf(&'a [T]),
}

impl<T: Real> Weights<'_, T> {
    /// One entry per column, each `w` finite and nonnegative or `s` finite; every weighted view checks it in `new`.
    pub(super) fn check(self, columns: usize) -> Result<(), ViewError> {
        let (entries, valid): (&[T], fn(T) -> bool) = match self {
            Self::Unit => return Ok(()),
            Self::Of(w) => (w, |w| w.is_finite() && w >= T::zero()),
            Self::SqrtOf(s) => (s, |s| s.is_finite()),
        };
        if entries.len() != columns {
            return Err(ViewError::LengthMismatch {
                expected: columns,
                found: entries.len(),
            });
        }
        match entries.iter().position(|&x| !valid(x)) {
            Some(column) => Err(ViewError::InvalidWeight { column }),
            None => Ok(()),
        }
    }

    /// The same slice read the same way: contents are never compared, so a NaN cannot split two sides of one Gram.
    pub(super) fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::Unit, Self::Unit) => true,
            (Self::Of(a), Self::Of(b)) | (Self::SqrtOf(a), Self::SqrtOf(b)) => core::ptr::eq(a, b),
            _ => false,
        }
    }

    /// Column `k`'s entry, matched per call: for one-time passes, not the column visit.
    pub(super) fn at(self, k: usize) -> Weight<T> {
        match self {
            Self::Unit => Weight::Unit,
            Self::Of(w) => Weight::Of(w[k]),
            Self::SqrtOf(s) => Weight::SqrtOf(s[k]),
        }
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

    /// Column `k` is `e_a + v_b(k) e_(offset + b)`; matched once, then looped per case.
    #[inline(always)]
    pub(super) fn for_each_column(
        self,
        offset: usize,
        v_b: impl Fn(usize) -> T,
        visit: impl FnMut(Column<T>),
    ) {
        match self.weights {
            Weights::Unit => self.columns(offset, v_b, |_| Weight::Unit, visit),
            Weights::Of(w) => self.columns(offset, v_b, |k| Weight::Of(w[k]), visit),
            Weights::SqrtOf(s) => self.columns(offset, v_b, |k| Weight::SqrtOf(s[k]), visit),
        }
    }

    #[inline(always)]
    fn columns(
        self,
        offset: usize,
        v_b: impl Fn(usize) -> T,
        weight: impl Fn(usize) -> Weight<T>,
        mut visit: impl FnMut(Column<T>),
    ) {
        for (k, (a, b)) in self.a.iter().zip(self.b).enumerate() {
            visit(Column {
                a: index(*a) as u32,
                b: (offset + index(*b)) as u32,
                v_a: T::one(),
                v_b: v_b(k),
                weight: weight(k),
            });
        }
    }
}
