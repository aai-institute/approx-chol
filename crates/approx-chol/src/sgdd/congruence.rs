use std::borrow::Cow;

use crate::{compensated_sum, widen, Batch, Real};

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
    Scaled(ScaledFactors<T>),
}

/// `F = S R`'s factors, the largest `|f|`'s index and `F·1`'s normalization, computed once: every singular solve projects `F·1` out.
#[derive(Clone, Debug)]
pub(crate) struct ScaledFactors<T> {
    factors: Box<[T]>,
    peak: usize,
    norm: f64,
}

/// `F` on a block's vertices from an offset on, asked only where its gauge goes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FactorTail<'f, T>(&'f [T]);

impl<T: Real> FactorTail<'_, T> {
    /// The offset of the largest `|f|`.
    pub(crate) fn largest(self) -> usize {
        largest(self.0)
    }

    /// Whether `|f|` spans past `1/√ε`, where a solve pinned at a small `|f|` loses more than half its digits.
    pub(crate) fn spans_wide(self) -> bool {
        let (least, most) = (self.0.iter()).fold((T::infinity(), T::zero()), |(least, most), f| {
            (least.min(f.abs()), most.max(f.abs()))
        });
        least < most * T::epsilon().sqrt()
    }
}

/// An index of the largest `|f|`, where pinning the gauge leaves its rounding unamplified by `1/f`.
fn largest<T: Real>(f: &[T]) -> usize {
    let last = f.len().saturating_sub(1);
    (0..last).fold(last, |best, i| match f[i].abs() > f[best].abs() {
        true => i,
        false => best,
    })
}

impl<T: Real> ScaledFactors<T> {
    pub(super) fn new(factors: Box<[T]>) -> Self {
        let peak = largest(&factors);
        let largest = factors.get(peak).map_or(T::zero(), |f| f.abs());
        let unit = |&f: &T| {
            let u = widen(f / largest);
            u * u
        };
        let norm = compensated_sum(&factors, |_, f| unit(&f));
        Self {
            factors,
            peak,
            norm,
        }
    }
}

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for ScaledFactors<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.factors.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Real + serde::Deserialize<'de>> serde::Deserialize<'de> for ScaledFactors<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <Box<[T]>>::deserialize(deserializer).map(Self::new)
    }
}

impl<T: Real> Congruence<T> {
    /// `x ← F x` on a gathered `x`, `F·1` projected out first when `M` is singular.
    pub(crate) fn restrict<V: Batch<T>>(&self, x: &mut [V], singular: bool) {
        if singular {
            self.project_out_null(x);
        }
        self.apply(x);
    }

    /// `x ← x − x_q·1` at the largest `|f|`, where a split block's singular solve fixes `M`'s gauge.
    pub(crate) fn regauge<V: Batch<T>>(&self, x: &mut [V]) {
        if let Congruence::Scaled(scales) = self {
            let gauge = x[scales.peak];
            x.iter_mut().for_each(|x| *x = x.sub(gauge));
        }
    }

    /// `x ← F x` back in `A_c`'s scale, `F·1` projected out last when `M` is singular.
    pub(crate) fn prolongate<V: Batch<T>>(&self, x: &mut [V], singular: bool) {
        self.apply(x);
        if singular {
            self.project_out_null(x);
        }
    }

    /// `f_i`, vertex `i`'s entry of `F·1`.
    pub(super) fn factor(&self, i: usize) -> T {
        match self {
            Self::Identity => T::one(),
            Self::NegateFrom(first) if i >= *first => -T::one(),
            Self::NegateFrom(_) => T::one(),
            Self::Signs(factors) | Self::Scaled(ScaledFactors { factors, .. }) => factors[i],
        }
    }

    /// `F` on vertices `from..`, renumbered from `0`: a complement's, `F` commuting with the elimination.
    pub(super) fn tail(&self, from: usize) -> Self {
        match self {
            Self::Identity => Self::Identity,
            // `(−I) S (−I) = S`: a tail negated whole needs no congruence.
            Self::NegateFrom(first) if *first <= from => Self::Identity,
            Self::NegateFrom(first) => Self::NegateFrom(first - from),
            Self::Signs(factors) => Self::Signs(factors[from..].into()),
            Self::Scaled(scales) => Self::Scaled(ScaledFactors::new(scales.factors[from..].into())),
        }
    }

    /// `F` from vertex `from` on, `None` unless scaled, so every `|f|` is `1`.
    pub(crate) fn factor_tail(&self, from: usize) -> Option<FactorTail<'_, T>> {
        match self {
            Self::Scaled(scales) => Some(FactorTail(&scales.factors[from..])),
            _ => None,
        }
    }

    /// Whether it scales exactly `n` vertices.
    pub(super) fn fits(&self, n: usize) -> bool {
        match self {
            Self::Identity => true,
            Self::NegateFrom(first) => *first <= n,
            Self::Signs(factors) | Self::Scaled(ScaledFactors { factors, .. }) => {
                factors.len() == n
            }
        }
    }

    /// Whether it scales exactly `n` vertices, each by a sign or a finite nonzero factor.
    #[cfg(feature = "serde")]
    pub(crate) fn holds(&self, n: usize) -> bool {
        let factors = match self {
            Self::Signs(f) => f.iter().all(|f| f.abs() == T::one()),
            Self::Scaled(s) => s.factors.iter().all(|f| f.is_finite() && *f != T::zero()),
            Self::Identity | Self::NegateFrom(_) => true,
        };
        factors && self.fits(n)
    }

    /// `F·1` over `n` vertices, `None` for the identity.
    pub(super) fn factors(&self, n: usize) -> Option<Cow<'_, [T]>> {
        match self {
            Self::Identity => None,
            Self::NegateFrom(_) => Some(Cow::Owned((0..n).map(|i| self.factor(i)).collect())),
            Self::Signs(factors) | Self::Scaled(ScaledFactors { factors, .. }) => {
                Some(Cow::Borrowed(&factors[..]))
            }
        }
    }

    /// `x ← F x`: the solve applies it on both sides, `A_c⁺ = P F M⁺ F P`.
    fn apply<V: Batch<T>>(&self, x: &mut [V]) {
        match self {
            Self::Identity => {}
            Self::NegateFrom(first) => x[*first..].iter_mut().for_each(|x| *x = x.neg()),
            Self::Signs(factors) | Self::Scaled(ScaledFactors { factors, .. }) => {
                (x.iter_mut().zip(factors)).for_each(|(x, &f)| *x = x.mul(f))
            }
        }
    }

    /// Removes the component along `F·1`, max-normalized: the raw norm overflows at a 1e200 scale range.
    fn project_out_null<V: Batch<T>>(&self, x: &mut [V]) {
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

fn project_out<T: Real, V: Batch<T>>(x: &mut [V], u: impl Fn(usize) -> T, uu: f64) {
    // Compensated, or a constant added to a floating `b` moves the solution (#147).
    let ux = compensated_sum(x, |i, x: V| x.widen().mul(widen(u(i))));
    let along = ux.div(uu);
    for (i, x) in x.iter_mut().enumerate() {
        *x = V::narrow(x.widen().sub(along.mul(widen(u(i)))));
    }
}
