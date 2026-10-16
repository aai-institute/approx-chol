use std::borrow::Cow;

use crate::narrow;
use crate::{widen, Real};

/// `F` in `M = F A_c F`, finite and nonzero; a folded `A_c`'s null vector is then `F·1`.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(deserialize = "T: Real + serde::Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub(crate) enum Congruence<T> {
    Identity,
    /// Every vertex from this one on is negated.
    NegateFrom(usize),
    /// Signs only, exact, so a sign never moves a magnitude.
    Signs(Box<[T]>),
    /// `S R`.
    Scaled(Scales<T>),
}

/// `F = S R`'s factors and `F·1`'s normalization, computed once: every singular solve projects `F·1` out.
#[derive(Clone, Debug)]
pub(crate) struct Scales<T> {
    factors: Box<[T]>,
    largest: T,
    norm: f64,
}

impl<T: Real> Scales<T> {
    pub(super) fn new(factors: Box<[T]>) -> Self {
        let largest = factors.iter().fold(T::zero(), |m, f| m.max(f.abs()));
        let unit = |&f: &T| {
            let u = widen(f / largest);
            u * u
        };
        let norm = crate::compensated_sum(factors.iter().map(unit));
        Self {
            factors,
            largest,
            norm,
        }
    }

    pub(crate) fn factors(&self) -> &[T] {
        &self.factors
    }
}

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for Scales<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.factors.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for Scales<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <Box<[T]>>::deserialize(deserializer).map(Self::new)
    }
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
            Self::Signs(factors) | Self::Scaled(Scales { factors, .. }) => factors[i],
        }
    }

    /// Whether it scales exactly `n` vertices.
    pub(super) fn fits(&self, n: usize) -> bool {
        match self {
            Self::Identity => true,
            Self::NegateFrom(first) => *first <= n,
            Self::Signs(factors) | Self::Scaled(Scales { factors, .. }) => factors.len() == n,
        }
    }

    /// `F·1` over `n` vertices, `None` for the identity.
    pub(super) fn factors(&self, n: usize) -> Option<Cow<'_, [T]>> {
        match self {
            Self::Identity => None,
            Self::NegateFrom(first) => Some(Cow::Owned(signs(*first, n))),
            Self::Signs(factors) | Self::Scaled(Scales { factors, .. }) => {
                Some(Cow::Borrowed(&factors[..]))
            }
        }
    }

    /// `x ← F x`: the solve applies it on both sides, `A_c⁺ = P F M⁺ F P`.
    pub(super) fn apply(&self, x: &mut [T]) {
        match self {
            Self::Identity => {}
            Self::NegateFrom(first) => x[*first..].iter_mut().for_each(|x| *x = -*x),
            Self::Signs(factors) | Self::Scaled(Scales { factors, .. }) => {
                (x.iter_mut().zip(factors)).for_each(|(x, &f)| *x = *x * f)
            }
        }
    }

    /// Removes the component along `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    pub(super) fn project_out_null(&self, x: &mut [T]) {
        // Every unit `F·1` is its own normalization, with `uᵀu = n` exact below 2⁵³.
        let n = x.len() as f64;
        match self {
            Self::Identity => project_out(x, |_| T::one(), n),
            Self::NegateFrom(first) => {
                let first = *first;
                project_out(x, |i| if i >= first { -T::one() } else { T::one() }, n)
            }
            Self::Signs(factors) => project_out(x, |i| factors[i], n),
            Self::Scaled(scales) => {
                project_out(x, |i| scales.factors[i] / scales.largest, scales.norm)
            }
        }
    }
}

fn project_out<T: Real>(x: &mut [T], u: impl Fn(usize) -> T, uu: f64) {
    // Compensated, or a constant added to a floating `b` moves the solution (#147).
    let ux = crate::compensated_sum(x.iter().enumerate().map(|(i, &x)| widen(u(i)) * widen(x)));
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
