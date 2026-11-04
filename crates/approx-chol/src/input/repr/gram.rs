//! `A = V W Vᵀ (+ diag(s))`, column `k` of `V` joining `a[k]` and `b[k]`, as every Gram input is read.

use num_traits::PrimInt;

use super::super::index;
use crate::storage::{widen, Index, Real};

/// A Gram input by its layout, values and weights.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gram<'a, T, I> {
    a: &'a [I],
    b: &'a [I],
    layout: Layout,
    values: Values<'a, T>,
    weights: Weights<'a, T>,
}

/// Which vertices `a` and `b` number.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Layout {
    /// Both ends among `n` vertices.
    Whole { n: usize },
    /// `a` among the `n_left` left vertices, `b` among the `n_right` right ones, numbered after them.
    Bipartite { n_left: usize, n_right: usize },
}

/// Column `k`'s two entries.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Values<'a, T> {
    /// `e_a − e_b`: `A` dominant by shape, its diagonal derived, plus an SDDM's surplus `s`.
    Incidence { surplus: Option<&'a [T]> },
    /// `x[k] e_a + y[k] e_b`, a missing side `1`.
    Valued {
        x: Option<&'a [T]>,
        y: Option<&'a [T]>,
    },
}

/// `W`: none, a Laplacian's edge weights, or `V`'s column scales `s` with `w = s²`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Weights<'a, T> {
    Unit,
    Of(&'a [T]),
    Scale(&'a [T]),
}

impl<'a, T: Real, I: PrimInt> Gram<'a, T, I> {
    /// Each input type checks its own ends, lengths and values before building one.
    pub(in crate::input) fn new(
        (a, b): (&'a [I], &'a [I]),
        layout: Layout,
        values: Values<'a, T>,
        weights: Weights<'a, T>,
    ) -> Self {
        Self {
            a,
            b,
            layout,
            values,
            weights,
        }
    }

    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }

    pub(crate) fn values(&self) -> Values<'a, T> {
        self.values
    }

    /// The surplus `s`, which only an SDDM has.
    pub(crate) fn surplus(&self) -> Option<&'a [T]> {
        match self.values {
            Values::Incidence { surplus } => surplus,
            Values::Valued { .. } => None,
        }
    }

    /// Dimension of `A`, not the column count; saturates, as a sum past `usize` fails allocation anyway.
    pub(crate) fn n(&self) -> usize {
        match self.layout {
            Layout::Whole { n } => n,
            Layout::Bipartite { n_left, n_right } => n_left.saturating_add(n_right),
        }
    }

    /// `V`'s column count, known before any column is read.
    pub(crate) fn n_columns(&self) -> usize {
        self.a.len()
    }

    /// Each column in order; write `visit` as `#[inline(always)] |column| …` (outlined: 1.6-1.9x).
    #[inline(always)]
    pub(crate) fn for_each_column<S: Index>(self, visit: impl FnMut(Column<T, S>)) {
        // Each kind loops straight: reading `V` through a per-column match cost within's paths 6%.
        match self.weights {
            Weights::Unit => self.weighted(|_| Weight::Unit, visit),
            Weights::Of(w) => self.weighted(|k| Weight::Of(w[k]), visit),
            Weights::Scale(s) => self.weighted(|k| Weight::Scale(s[k]), visit),
        }
    }

    /// The columns under `weight(k)`; an incidence column reads `e_a − e_b`, a valued one's missing side `1`.
    #[inline(always)]
    fn weighted<S: Index>(
        self,
        weight: impl Fn(usize) -> Weight<T>,
        visit: impl FnMut(Column<T, S>),
    ) {
        let (one, minus_one) = (|_| T::one(), |_| -T::one());
        match self.values {
            Values::Incidence { .. } => self.columns_of(one, minus_one, weight, visit),
            Values::Valued { x: None, y: None } => self.columns_of(one, one, weight, visit),
            Values::Valued {
                x: Some(x),
                y: None,
            } => self.columns_of(|k| x[k], one, weight, visit),
            Values::Valued {
                x: None,
                y: Some(y),
            } => self.columns_of(one, |k| y[k], weight, visit),
            Values::Valued {
                x: Some(x),
                y: Some(y),
            } => self.columns_of(|k| x[k], |k| y[k], weight, visit),
        }
    }

    /// Column `k` is `v_a(k) e_a + v_b(k) e_(offset + b)` under `weight(k)`, `offset` the left part's size.
    #[inline(always)]
    fn columns_of<S: Index>(
        self,
        v_a: impl Fn(usize) -> T,
        v_b: impl Fn(usize) -> T,
        weight: impl Fn(usize) -> Weight<T>,
        mut visit: impl FnMut(Column<T, S>),
    ) {
        let offset = match self.layout {
            Layout::Whole { .. } => 0,
            Layout::Bipartite { n_left, .. } => n_left,
        };
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

/// Column `k` of `V` under its weight, unfolded, so the floating test reads `V` exactly as given.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Column<T, I> {
    pub(crate) a: I,
    pub(crate) b: I,
    pub(crate) v_a: T,
    pub(crate) v_b: T,
    weight: Weight<T>,
}

impl<T: Real, I> Column<T, I> {
    /// `w·v_a·v_b` in `f64`: every assembly sums its cells there and casts once.
    pub(crate) fn cell(self) -> f64 {
        self.weight
            .bilinear_unflushed(widen(self.v_a), widen(self.v_b))
    }

    /// `w·x·y` for any `x`, `y` on this column's ends.
    pub(crate) fn bilinear(self, x: f64, y: f64) -> f64 {
        self.weight.bilinear(x, y)
    }

    /// `w·v²`, an underflow kept as the least subnormal.
    pub(crate) fn square(self, v: f64) -> f64 {
        self.weight.bilinear_unflushed(v, v)
    }

    /// `w = 0`, so the column adds nothing to `A`.
    pub(crate) fn zero_weight(self) -> bool {
        self.weight.is_zero()
    }
}

/// Column `k`'s `w_k` in `A = Σ_k w_k v_k v_kᵀ`.
#[derive(Clone, Copy, Debug)]
enum Weight<T> {
    /// No `W`.
    Unit,
    /// `w`.
    Of(T),
    /// `w = s²`, kept as `s` so `(s·x)·(s·y)` neither overflows nor underflows through `s²`.
    Scale(T),
}

impl<T: Real> Weight<T> {
    /// `w·x·y`, as `(s·x)·(s·y)` under a scale.
    fn bilinear(self, x: f64, y: f64) -> f64 {
        match self {
            Self::Unit => x * y,
            Self::Of(w) => widen(w) * x * y,
            Self::Scale(s) => (widen(s) * x) * (widen(s) * y),
        }
    }

    /// As `bilinear`, a nonzero product that underflows kept as the least subnormal, so the range screen rejects it (#163).
    fn bilinear_unflushed(self, x: f64, y: f64) -> f64 {
        let product = self.bilinear(x, y);
        match product == 0.0 && x != 0.0 && y != 0.0 && !self.is_zero() {
            true => f64::from_bits(1).copysign(product),
            false => product,
        }
    }

    fn is_zero(self) -> bool {
        match self {
            Self::Unit => false,
            Self::Of(w) | Self::Scale(w) => w == T::zero(),
        }
    }
}
