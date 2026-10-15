use std::borrow::Cow;

use crate::narrow;
use crate::{widen, Real};

/// `F` in `M = F A_c F`, finite and nonzero; a folded `A_c`'s null vector is then `F·1`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(crate) enum Congruence<T> {
    Identity,
    /// Every vertex from this one on is negated.
    NegateFrom(usize),
    /// Signs only, exact, so a sign never moves a magnitude.
    Signs(Box<[T]>),
    /// `S R`.
    Scaled(Box<[T]>),
}

impl<T: Real> Congruence<T> {
    /// `true` where the vertex is negated.
    pub(super) fn negated(signs: &[bool]) -> Self {
        let sign = |&negated: &bool| match negated {
            true => -T::one(),
            false => T::one(),
        };
        Self::Signs(signs.iter().map(sign).collect())
    }

    /// `F = S R`'s factors, `self` the signs `S`, for positive scales `R`.
    pub(super) fn times(&self, scales: &[T]) -> Box<[T]> {
        match self.factors(scales.len()) {
            None => scales.into(),
            Some(factors) => (factors.iter().zip(scales))
                .map(|(&factor, &scale)| factor * scale)
                .collect(),
        }
    }

    /// `f_i`, vertex `i`'s entry of `F·1`.
    pub(super) fn factor(&self, i: usize) -> T {
        match self {
            Self::Identity => T::one(),
            Self::NegateFrom(first) if i >= *first => -T::one(),
            Self::NegateFrom(_) => T::one(),
            Self::Signs(factors) | Self::Scaled(factors) => factors[i],
        }
    }

    /// Whether it scales exactly `n` vertices.
    pub(super) fn fits(&self, n: usize) -> bool {
        match self {
            Self::Identity => true,
            Self::NegateFrom(first) => *first <= n,
            Self::Signs(factors) | Self::Scaled(factors) => factors.len() == n,
        }
    }

    /// `F·1` over `n` vertices, `None` for the identity.
    pub(super) fn factors(&self, n: usize) -> Option<Cow<'_, [T]>> {
        match self {
            Self::Identity => None,
            Self::NegateFrom(first) => Some(Cow::Owned(signs(*first, n))),
            Self::Signs(factors) | Self::Scaled(factors) => Some(Cow::Borrowed(&factors[..])),
        }
    }

    /// `x ← F x`: the solve applies it on both sides, `A_c⁺ = P F M⁺ F P`.
    pub(super) fn apply(&self, x: &mut [T]) {
        match self {
            Self::Identity => {}
            Self::NegateFrom(first) => x[*first..].iter_mut().for_each(|x| *x = -*x),
            Self::Signs(factors) | Self::Scaled(factors) => {
                (x.iter_mut().zip(factors)).for_each(|(x, &f)| *x = *x * f)
            }
        }
    }

    /// Removes the component along `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    pub(super) fn project_out_null(&self, x: &mut [T]) {
        match self {
            Self::Signs(factors) | Self::Scaled(factors) => project_out(x, |i| factors[i]),
            _ => project_out(x, |i| self.factor(i)),
        }
    }
}

fn project_out<T: Real>(x: &mut [T], factor: impl Fn(usize) -> T) {
    let largest = (0..x.len()).fold(T::zero(), |m, i| m.max(factor(i).abs()));
    let u = |i: usize| factor(i) / largest;
    // Compensated, or a constant added to a floating `b` moves the solution (#147).
    let ux = crate::compensated_sum(x.iter().enumerate().map(|(i, &x)| widen(u(i)) * widen(x)));
    let uu = crate::compensated_sum((0..x.len()).map(|i| widen(u(i)) * widen(u(i))));
    let along = ux / uu;
    for (i, x) in x.iter_mut().enumerate() {
        *x = narrow::<T>(widen(*x) - widen(u(i)) * along);
    }
}

/// `+1` before `first`, `−1` from it on.
fn signs<T: Real>(first: usize, n: usize) -> Vec<T> {
    let (plus, minus) = (
        std::iter::repeat_n(T::one(), first),
        std::iter::repeat_n(-T::one(), n - first),
    );
    plus.chain(minus).collect()
}
