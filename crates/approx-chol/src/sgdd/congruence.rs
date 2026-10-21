use std::borrow::Cow;
use std::ops::Range;

use crate::narrow;
use crate::{widen, Index, Real};

/// `R = F Pᵀ`, so `M = R A Rᵀ`: `Pᵀ` picks the block's vertices of `A`, `F` is the congruence.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self"),
    serde(bound(
        serialize = "T: serde::Serialize, I: Index + serde::Serialize",
        deserialize = "T: Real + serde::Deserialize<'de>, I: Index + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) struct Restriction<T, I> {
    vertices: Selection<I>,
    congruence: Congruence<T>,
}

impl<T: Real, I: Index> Restriction<T, I> {
    /// Panics unless one factor per vertex.
    pub(super) fn new(vertices: Box<[I]>, congruence: Congruence<T>) -> Self {
        assert!(congruence.fits(vertices.len()), "one factor per vertex");
        Self {
            vertices: Selection::of(vertices),
            congruence,
        }
    }

    pub(crate) fn vertices(&self) -> &Selection<I> {
        &self.vertices
    }

    /// `F` from vertex `from` on, `None` unless scaled, so every `|f|` is `1`.
    pub(crate) fn factor_tail(&self, from: usize) -> Option<FactorTail<'_, T>> {
        match &self.congruence {
            Congruence::Scaled(scales) => Some(FactorTail(&scales.factors[from..])),
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
        let fits = self.vertices.len() > 0 && self.congruence.fits(self.vertices.len());
        (fits && factors).then_some(self)
    }
}

/// `Pᵀ`: the block's vertices of `A` in local order, a run of consecutive ones kept as its range, where `Pᵀ x` is a slice of `x`.
#[derive(Clone, Debug)]
pub(crate) enum Selection<I> {
    Range(Range<usize>),
    Listed(Box<[I]>),
}

impl<I: Index> Selection<I> {
    fn of(vertices: Box<[I]>) -> Self {
        let start = vertices.first().map_or(0, |v| v.index());
        match start.checked_add(vertices.len()) {
            Some(end) if (vertices.iter().enumerate()).all(|(k, v)| v.index() == start + k) => {
                Self::Range(start..end)
            }
            _ => Self::Listed(vertices),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Range(range) => range.len(),
            Self::Listed(vertices) => vertices.len(),
        }
    }

    /// Local vertex `k` in `A`'s numbering.
    pub(crate) fn vertex(&self, k: usize) -> usize {
        match self {
            Self::Range(range) => range.start + k,
            Self::Listed(vertices) => vertices[k].index(),
        }
    }

    /// Workspace [`on_local`](Self::on_local) keeps ahead of what it passes on: room to gather, none for a range.
    pub(crate) fn gather_len(&self) -> usize {
        match self {
            Self::Range(_) => 0,
            Self::Listed(vertices) => vertices.len(),
        }
    }

    /// Runs `solve` on `Pᵀ x` and the rest of `scratch`, then sets the block's vertices of `x` to the result.
    pub(crate) fn on_local<T: Copy>(
        &self,
        x: &mut [T],
        scratch: &mut [T],
        solve: impl FnOnce(&mut [T], &mut [T]),
    ) {
        match self {
            Self::Range(range) => solve(&mut x[range.clone()], scratch),
            Self::Listed(vertices) => {
                let (local, rest) = scratch.split_at_mut(self.gather_len());
                (local.iter_mut().zip(vertices.iter())).for_each(|(v, &i)| *v = x[i.index()]);
                solve(local, rest);
                (local.iter().zip(vertices.iter())).for_each(|(&v, &i)| x[i.index()] = v);
            }
        }
    }
}

/// The vertex list either way, so the wire never shows how a block stores it.
#[cfg(feature = "serde")]
impl<I: Index + serde::Serialize> serde::Serialize for Selection<I> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Range(range) => serializer.collect_seq(range.clone().map(I::from_index)),
            Self::Listed(vertices) => vertices.serialize(serializer),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, I: Index + serde::Deserialize<'de>> serde::Deserialize<'de> for Selection<I> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <Box<[I]>>::deserialize(deserializer).map(Self::of)
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
    Scaled(ScaledFactors<T>),
}

/// `F = S R`'s factors, the largest `|f|`'s index and `F·1`'s normalization, computed once: every singular solve projects `F·1` out.
#[derive(Clone, Debug)]
pub(super) struct ScaledFactors<T> {
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
        let norm = crate::compensated_sum(factors.iter().map(unit));
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
    /// `f_i`, vertex `i`'s entry of `F·1`.
    pub(super) fn factor(&self, i: usize) -> T {
        match self {
            Self::Identity => T::one(),
            Self::NegateFrom(first) if i >= *first => -T::one(),
            Self::NegateFrom(_) => T::one(),
            Self::Signs(factors) | Self::Scaled(ScaledFactors { factors, .. }) => factors[i],
        }
    }

    /// Whether it scales exactly `n` vertices.
    fn fits(&self, n: usize) -> bool {
        match self {
            Self::Identity => true,
            Self::NegateFrom(first) => *first <= n,
            Self::Signs(factors) | Self::Scaled(ScaledFactors { factors, .. }) => {
                factors.len() == n
            }
        }
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
    fn apply(&self, x: &mut [T]) {
        match self {
            Self::Identity => {}
            Self::NegateFrom(first) => x[*first..].iter_mut().for_each(|x| *x = -*x),
            Self::Signs(factors) | Self::Scaled(ScaledFactors { factors, .. }) => {
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
