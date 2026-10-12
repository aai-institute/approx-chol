use std::borrow::Cow;

use super::OffDiagonal;
use crate::storage::{narrow, widen};
use crate::Real;

/// `F` in `M = F A_c F`: signs and scales, finite and nonzero; a folded `A_c`'s null vector is then `F·1`.
#[derive(Clone, Debug)]
pub(crate) struct Congruence<T>(Kind<T>);

#[derive(Clone, Debug)]
enum Kind<T> {
    Identity,
    /// Every vertex from this one on is negated.
    NegateFrom(usize),
    Factors(Box<[T]>),
}

impl<T: Real> Congruence<T> {
    pub(super) fn identity() -> Self {
        Self(Kind::Identity)
    }

    /// A positive bipartite block's signature, unmaterialized: every vertex from `first` on is negated.
    pub(super) fn negate_from(first: usize) -> Self {
        Self(Kind::NegateFrom(first))
    }

    /// `true` where the vertex is negated.
    pub(super) fn negated(signs: &[bool]) -> Self {
        let sign = |&negated: &bool| match negated {
            true => -T::one(),
            false => T::one(),
        };
        Self(Kind::Factors(signs.iter().map(sign).collect()))
    }

    /// `S R` for these signs and positive scales `R`.
    pub(super) fn scaled(self, scales: &[T]) -> Self {
        let factors = match self.0 {
            Kind::Identity => scales.into(),
            Kind::NegateFrom(first) => (signs::<T>(first, scales.len()).iter().zip(scales))
                .map(|(&sign, &scale)| sign * scale)
                .collect(),
            Kind::Factors(signs) => (signs.iter().zip(scales))
                .map(|(&sign, &scale)| sign * scale)
                .collect(),
        };
        Self(Kind::Factors(factors))
    }

    /// `f_i`, vertex `i`'s entry of `F·1`.
    pub(super) fn factor(&self, i: usize) -> T {
        match &self.0 {
            Kind::Identity => T::one(),
            Kind::NegateFrom(first) if i >= *first => -T::one(),
            Kind::NegateFrom(_) => T::one(),
            Kind::Factors(factors) => factors[i],
        }
    }

    /// Whether it scales exactly `n` vertices.
    pub(super) fn fits(&self, n: usize) -> bool {
        match &self.0 {
            Kind::Identity => true,
            Kind::NegateFrom(first) => *first <= n,
            Kind::Factors(factors) => factors.len() == n,
        }
    }

    /// `F X F`: `M`'s off-diagonal from `A_c`'s; `None` where an entry leaves [`in_range`](crate::storage::in_range).
    pub(super) fn transform(&self, off_diagonal: OffDiagonal<T>) -> Option<OffDiagonal<T>> {
        match self.factors(off_diagonal.n()) {
            None => Some(off_diagonal),
            Some(factors) => off_diagonal.scaled(&factors),
        }
    }

    /// As [`transform`](Self::transform), with `Σ_j |m_ij|` summed while `M` is formed.
    pub(super) fn transform_abs_row_sums(
        &self,
        off_diagonal: OffDiagonal<T>,
    ) -> Option<(OffDiagonal<T>, Vec<T>)> {
        match self.factors(off_diagonal.n()) {
            None => {
                let sums = off_diagonal.abs_row_sums();
                Some((off_diagonal, sums))
            }
            Some(factors) => off_diagonal.scaled_abs_row_sums(&factors),
        }
    }

    /// `F·1` over `n` vertices, `None` for the identity.
    fn factors(&self, n: usize) -> Option<Cow<'_, [T]>> {
        match &self.0 {
            Kind::Identity => None,
            Kind::NegateFrom(first) => Some(Cow::Owned(signs(*first, n))),
            Kind::Factors(factors) => Some(Cow::Borrowed(&factors[..])),
        }
    }

    /// `x ← F x`; with `P` projecting out `F·1` when the component floats, else `I`, `A_c⁺ = P F M⁺ F P`, so the solve applies it on both sides.
    pub(crate) fn apply(&self, x: &mut [T]) {
        match &self.0 {
            Kind::Identity => {}
            Kind::NegateFrom(first) => x[*first..].iter_mut().for_each(|x| *x = -*x),
            Kind::Factors(factors) => (x.iter_mut().zip(factors)).for_each(|(x, &f)| *x = *x * f),
        }
    }

    /// Removes the component along `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    pub(crate) fn project_out_null(&self, x: &mut [T]) {
        let largest = (0..x.len()).fold(T::zero(), |m, i| m.max(self.factor(i).abs()));
        let u = |i: usize| self.factor(i) / largest;
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
}

/// `+1` before `first`, `−1` from it on: exact, so a sign never moves a magnitude.
fn signs<T: Real>(first: usize, n: usize) -> Vec<T> {
    let (plus, minus) = (
        std::iter::repeat_n(T::one(), first),
        std::iter::repeat_n(-T::one(), n - first),
    );
    plus.chain(minus).collect()
}
