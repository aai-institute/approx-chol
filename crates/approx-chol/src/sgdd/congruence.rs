use std::borrow::Cow;

use crate::narrow;
use crate::{widen, Index, Real};

/// `R = F Pᵀ`, so `M = R A Rᵀ`: `Pᵀ` picks the block's vertices of `A`, `F` is the congruence.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self"),
    serde(bound(deserialize = "T: Real + serde::Deserialize<'de>, I: serde::Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub(crate) struct Restriction<T, I> {
    vertices: Box<[I]>,
    congruence: Congruence<T>,
}

impl<T: Real, I: Index> Restriction<T, I> {
    /// Panics unless one factor per vertex.
    pub(super) fn new(vertices: Box<[I]>, congruence: Congruence<T>) -> Self {
        assert!(congruence.fits(vertices.len()), "one factor per vertex");
        Self {
            vertices,
            congruence,
        }
    }

    pub(crate) fn vertices(&self) -> &[I] {
        &self.vertices
    }

    /// `local ← Pᵀ x`.
    pub(crate) fn gather(&self, x: &[T], local: &mut [T]) {
        (local.iter_mut().zip(self.vertices.iter())).for_each(|(v, &i)| *v = x[i.index()]);
    }

    /// `x ← P local` on the block's vertices, the rest untouched.
    pub(crate) fn scatter(&self, local: &[T], x: &mut [T]) {
        (local.iter().zip(self.vertices.iter())).for_each(|(&v, &i)| x[i.index()] = v);
    }

    /// `F`'s entries from `from` on, `None` when every `|f|` is `1`.
    pub(crate) fn scales(&self, from: usize) -> Option<&[T]> {
        match &self.congruence {
            Congruence::Scaled(scales) => Some(&scales.factors[from..]),
            _ => None,
        }
    }

    /// `x ← F x` on a gathered `x`, `F·1` projected out first when `M` is singular.
    pub(crate) fn restrict(&self, x: &mut [T], singular: bool) {
        if singular {
            self.congruence.project_out_null(x);
        }
        self.congruence.apply(x);
    }

    /// `x ← x − x_q·1` at the largest `|f|`, where a split block's singular solve fixes `M`'s gauge.
    pub(crate) fn regauge(&self, x: &mut [T]) {
        if let Congruence::Scaled(scales) = &self.congruence {
            let gauge = x[scales.peak];
            x.iter_mut().for_each(|x| *x = *x - gauge);
        }
    }

    /// `x ← F x` back in `A_c`'s scale, `F·1` projected out last when `M` is singular.
    pub(crate) fn prolongate(&self, x: &mut [T], singular: bool) {
        self.congruence.apply(x);
        if singular {
            self.congruence.project_out_null(x);
        }
    }
}

#[cfg(feature = "serde")]
checked_serde!(Restriction);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> Restriction<T, I> {
    fn checked(self) -> Option<Self> {
        let factors = match &self.congruence {
            Congruence::Signs(f) => f.iter().all(|f| f.abs() == T::one()),
            Congruence::Scaled(s) => s.factors.iter().all(|f| f.is_finite() && *f != T::zero()),
            Congruence::Identity | Congruence::NegateFrom(_) => true,
        };
        let fits = !self.vertices.is_empty() && self.congruence.fits(self.vertices.len());
        (fits && factors).then_some(self)
    }
}

/// `F` in `M = F A_c F`, finite and nonzero; a folded `A_c`'s null vector is then `F·1`.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(deserialize = "T: Real + serde::Deserialize<'de>"))
)]
#[derive(Clone, Debug)]
pub(super) enum Congruence<T> {
    Identity,
    /// Every vertex from this one on is negated.
    NegateFrom(usize),
    /// Signs only, exact, so a sign never moves a magnitude.
    Signs(Box<[T]>),
    /// `S R`.
    Scaled(Scales<T>),
}

/// `F = S R`'s factors, the largest `|f|`'s index and `F·1`'s normalization, computed once: every singular solve projects `F·1` out.
#[derive(Clone, Debug)]
pub(super) struct Scales<T> {
    factors: Box<[T]>,
    peak: usize,
    norm: f64,
}

impl<T: Real> Scales<T> {
    pub(super) fn new(factors: Box<[T]>) -> Self {
        let peak = crate::largest(&factors);
        let largest = factors.iter().fold(T::zero(), |m, f| m.max(f.abs()));
        let unit = |&f: &T| {
            let u = widen(f / largest);
            u * u
        };
        let norm = crate::compensated_sum(factors.iter().map(unit));
        Self {
            factors,
            peak,
            norm,
        }
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
    fn fits(&self, n: usize) -> bool {
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
    fn apply(&self, x: &mut [T]) {
        match self {
            Self::Identity => {}
            Self::NegateFrom(first) => x[*first..].iter_mut().for_each(|x| *x = -*x),
            Self::Signs(factors) | Self::Scaled(Scales { factors, .. }) => {
                (x.iter_mut().zip(factors)).for_each(|(x, &f)| *x = *x * f)
            }
        }
    }

    /// Removes the component along `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    fn project_out_null(&self, x: &mut [T]) {
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
                let largest = scales.factors[scales.peak].abs();
                project_out(x, |i| scales.factors[i] / largest, scales.norm)
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
