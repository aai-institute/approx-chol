use std::borrow::Cow;

use super::{OffDiagonal, ScalingReport};
use crate::storage::{narrow, widen};
use crate::Real;

/// `S` in `F = S R`: signs only, exact, so a sign never moves a magnitude.
#[derive(Clone, Debug)]
pub(crate) struct Signature<T>(Signs<T>);

#[derive(Clone, Debug)]
enum Signs<T> {
    Identity,
    /// Every vertex from this one on is negated.
    NegateFrom(usize),
    Each(Box<[T]>),
}

impl<T: Real> Signature<T> {
    pub(super) fn identity() -> Self {
        Self(Signs::Identity)
    }

    /// A positive bipartite block's signature, unmaterialized: every vertex from `first` on is negated.
    pub(super) fn negate_from(first: usize) -> Self {
        Self(Signs::NegateFrom(first))
    }

    /// `true` where the vertex is negated.
    pub(super) fn negated(signs: &[bool]) -> Self {
        let sign = |&negated: &bool| match negated {
            true => -T::one(),
            false => T::one(),
        };
        Self(Signs::Each(signs.iter().map(sign).collect()))
    }

    /// `S R`'s factors for positive scales `R`.
    pub(super) fn times(self, scales: &[T]) -> Box<[T]> {
        match self.factors(scales.len()) {
            None => scales.into(),
            Some(signs) => (signs.iter().zip(scales))
                .map(|(&sign, &scale)| sign * scale)
                .collect(),
        }
    }

    /// `S X S`: `M`'s off-diagonal from `A_c`'s; `None` where an entry leaves [`in_range`](crate::storage::in_range).
    pub(super) fn transform(&self, off_diagonal: OffDiagonal<T>) -> Option<OffDiagonal<T>> {
        match self.factors(off_diagonal.n()) {
            None => Some(off_diagonal),
            Some(factors) => off_diagonal.scaled(&factors),
        }
    }

    fn factor(&self, i: usize) -> T {
        match &self.0 {
            Signs::Identity => T::one(),
            Signs::NegateFrom(first) if i >= *first => -T::one(),
            Signs::NegateFrom(_) => T::one(),
            Signs::Each(signs) => signs[i],
        }
    }

    fn fits(&self, n: usize) -> bool {
        match &self.0 {
            Signs::Identity => true,
            Signs::NegateFrom(first) => *first <= n,
            Signs::Each(signs) => signs.len() == n,
        }
    }

    /// `S·1` over `n` vertices, `None` for the identity.
    fn factors(&self, n: usize) -> Option<Cow<'_, [T]>> {
        match &self.0 {
            Signs::Identity => None,
            Signs::NegateFrom(first) => Some(Cow::Owned(signs(*first, n))),
            Signs::Each(signs) => Some(Cow::Borrowed(&signs[..])),
        }
    }

    fn apply(&self, x: &mut [T]) {
        match &self.0 {
            Signs::Identity => {}
            Signs::NegateFrom(first) => x[*first..].iter_mut().for_each(|x| *x = -*x),
            Signs::Each(signs) => (x.iter_mut().zip(signs)).for_each(|(x, &s)| *x = *x * s),
        }
    }
}

/// `F` in `M = F A_c F`: signs and scales, finite and nonzero; a folded `A_c`'s null vector is then `F·1`.
#[derive(Clone, Debug)]
pub(crate) enum Congruence<T> {
    Signed(Signature<T>),
    /// `S R` from [`Signature::times`], and how far `R` fell short when that exceeded the policy's tolerance.
    Scaled(Box<[T]>, Option<ScalingReport>),
}

impl<T: Real> Congruence<T> {
    pub(super) fn report(&self) -> Option<ScalingReport> {
        match self {
            Self::Signed(_) => None,
            Self::Scaled(_, report) => *report,
        }
    }

    /// `f_i`, vertex `i`'s entry of `F·1`.
    pub(super) fn factor(&self, i: usize) -> T {
        match self {
            Self::Signed(signature) => signature.factor(i),
            Self::Scaled(factors, _) => factors[i],
        }
    }

    /// Whether it scales exactly `n` vertices.
    pub(super) fn fits(&self, n: usize) -> bool {
        match self {
            Self::Signed(signature) => signature.fits(n),
            Self::Scaled(factors, _) => factors.len() == n,
        }
    }

    /// `x ← F x`; with `P` projecting out `F·1` when the component floats, else `I`, `A_c⁺ = P F M⁺ F P`, so the solve applies it on both sides.
    pub(crate) fn apply(&self, x: &mut [T]) {
        match self {
            Self::Signed(signature) => signature.apply(x),
            Self::Scaled(factors, _) => (x.iter_mut().zip(factors)).for_each(|(x, &f)| *x = *x * f),
        }
    }

    /// Removes the component along `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    pub(crate) fn project_out_null(&self, x: &mut [T]) {
        match self {
            Self::Signed(signature) => project_out(x, |i| signature.factor(i)),
            Self::Scaled(factors, _) => project_out(x, |i| factors[i]),
        }
    }
}

fn project_out<T: Real>(x: &mut [T], factor: impl Fn(usize) -> T) {
    let largest = (0..x.len()).fold(T::zero(), |m, i| m.max(factor(i).abs()));
    let u = |i: usize| factor(i) / largest;
    let (mut ux, mut uu) = (0.0, 0.0);
    for (i, &x) in x.iter().enumerate() {
        ux += widen(u(i)) * widen(x);
        uu += widen(u(i)) * widen(u(i));
    }
    let along = ux / uu;
    for (i, x) in x.iter_mut().enumerate() {
        *x = narrow::<T>(widen(*x) - widen(u(i)) * along);
    }
}

/// `+1` before `first`, `−1` from it on: exact, so a sign never moves a magnitude.
fn signs<T: Real>(first: usize, n: usize) -> Vec<T> {
    let (plus, minus) = (
        std::iter::repeat_n(T::one(), first),
        std::iter::repeat_n(-T::one(), n - first),
    );
    plus.chain(minus).collect()
}
