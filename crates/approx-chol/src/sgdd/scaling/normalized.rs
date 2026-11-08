use std::borrow::Cow;
use std::cell::Cell;

use crate::storage::{OffDiagonal, Rows};
use crate::{narrow, widen, Index, Real};

/// `N = D^{-1/2}|offdiag(A_c)|D^{-1/2}`: scales `R` make `R|A_c|R` dominant iff `N y ≤ y` for `y = R√D`, so iff `ρ(N) ≤ 1` (Boman, Chen, Parekh, Toledo, LAA 2005).
pub(super) struct Normalized<'a, T, I> {
    off_diagonal: &'a OffDiagonal<T, I>,
    inv_sqrt: Box<[f64]>,
    /// A product's operand times its scales, formed once per product, not per entry read.
    scaled: Cell<Vec<f64>>,
}

impl<'a, T: Real, I: Index> Normalized<'a, T, I> {
    pub(super) fn new(off_diagonal: &'a OffDiagonal<T, I>, diagonal: &[T]) -> Self {
        let inv_sqrt = diagonal.iter().map(|&d| 1.0 / widen(d).sqrt()).collect();
        Self {
            off_diagonal,
            inv_sqrt,
            scaled: Cell::default(),
        }
    }

    pub(super) fn n(&self) -> usize {
        self.inv_sqrt.len()
    }

    /// `(αI − N) y = b` as CG solves it, with its right-hand side: a bipartite `N` on its kept side.
    pub(super) fn shifted<'s>(
        &'s self,
        alpha: f64,
        b: &'s [f64],
    ) -> (Shifted<'s, 'a, T, I>, Vec<f64>) {
        match self.off_diagonal {
            OffDiagonal::General(_) => (Shifted::Whole { n: self, alpha }, b.to_vec()),
            OffDiagonal::Bipartite(cross) => {
                let kept = Kept::new(self, cross, alpha, b);
                let rhs = kept.rhs(b);
                (Shifted::Kept(kept), rhs)
            }
        }
    }

    /// `out = |offdiag(A_c)| D^{-1/2} y`, returning the outer `D^{-1/2}` that makes it `N y`.
    pub(super) fn unscaled(&self, y: &[f64], out: &mut [f64]) -> &[f64] {
        self.with_scaled(y, &self.inv_sqrt, |x| self.off_diagonal.abs_times(x, out));
        &self.inv_sqrt
    }

    /// `product(s∘y)`.
    fn with_scaled(&self, y: &[f64], s: &[f64], product: impl FnOnce(&[f64])) {
        let mut x = self.scaled.take();
        x.clear();
        x.extend(y.iter().zip(s).map(|(&y, &s)| y * s));
        product(&x);
        self.scaled.set(x);
    }

    /// `R = D^{-1/2} y` with `y` peaking at `1`, so `M`'s diagonal never exceeds `1`.
    pub(super) fn scales(&self, y: &[f64]) -> Box<[T]> {
        let peak = y.iter().copied().fold(0.0f64, f64::max);
        (y.iter().zip(&self.inv_sqrt[..]))
            .map(|(&y, &s)| narrow::<T>(y / peak * s))
            .collect()
    }
}

/// The system CG iterates on for one shift `α`.
pub(super) enum Shifted<'n, 'a, T, I> {
    /// `(αI − N) y = b` over every vertex.
    Whole {
        n: &'n Normalized<'a, T, I>,
        alpha: f64,
    },
    /// A bipartite `N` with its eliminated side solved out.
    Kept(Kept<'n, 'a, T, I>),
}

impl<T: Real, I: Index> Shifted<'_, '_, T, I> {
    /// `out = S p` for the system's matrix `S`, returning `pᵀp` and `pᵀSp` from the same pass.
    pub(super) fn apply(&self, p: &[f64], out: &mut [f64]) -> (f64, f64) {
        let (shift, s) = match self {
            Self::Whole { n, alpha } => (*alpha, n.unscaled(p, out)),
            Self::Kept(kept) => kept.unscaled(p, out),
        };
        let (mut pp, mut curvature) = (0.0, 0.0);
        for ((o, &p), &s) in out.iter_mut().zip(p).zip(s) {
            *o = shift * p - s * *o;
            pp += p * p;
            curvature += p * *o;
        }
        (pp, curvature)
    }

    /// A lower bound on `ρ(N)` from `p`'s Rayleigh quotient, `N` symmetric and nonnegative.
    pub(super) fn spectral_floor(&self, pp: f64, curvature: f64) -> f64 {
        match self {
            Self::Whole { alpha, .. } => alpha - curvature / pp,
            Self::Kept(kept) => (kept.alpha * kept.alpha - curvature / pp).max(0.0).sqrt(),
        }
    }

    /// The whole `y` of an iterate.
    pub(super) fn solution<'x>(&self, x: &'x [f64]) -> Cow<'x, [f64]> {
        match self {
            Self::Whole { .. } => Cow::Borrowed(x),
            Self::Kept(kept) => Cow::Owned(kept.lift(x, Some(kept.b_e))),
        }
    }

    /// The whole vector of a direction, which carries no right-hand side.
    pub(super) fn direction<'x>(&self, p: &'x [f64]) -> Cow<'x, [f64]> {
        match self {
            Self::Whole { .. } => Cow::Borrowed(p),
            Self::Kept(kept) => Cow::Owned(kept.lift(p, None)),
        }
    }
}

/// `N = [0 B; Bᵀ 0]`: `(α² − BᵀB) y_k = α b_k + Bᵀ b_e`, then `y_e = (b_e + B y_k)/α`; half the steps of the whole system.
pub(super) struct Kept<'n, 'a, T, I> {
    n: &'n Normalized<'a, T, I>,
    cross: &'a Rows<T, I>,
    alpha: f64,
    b_e: &'n [f64],
}

impl<'n, 'a, T: Real, I: Index> Kept<'n, 'a, T, I> {
    fn new(n: &'n Normalized<'a, T, I>, cross: &'a Rows<T, I>, alpha: f64, b: &'n [f64]) -> Self {
        let b_e = &b[..cross.n_rows()];
        Self {
            n,
            cross,
            alpha,
            b_e,
        }
    }

    fn rhs(&self, b: &[f64]) -> Vec<f64> {
        let (s_e, s_k) = self.n.inv_sqrt.split_at(self.cross.n_rows());
        let b_k = &b[self.cross.n_rows()..];
        let mut rhs = vec![0.0; b_k.len()];
        (self.n).with_scaled(self.b_e, s_e, |x| {
            self.cross.abs_transpose_times(x, &mut rhs);
        });
        for ((r, &s), &b) in rhs.iter_mut().zip(s_k).zip(b_k) {
            *r = self.alpha * b + s * *r;
        }
        rhs
    }

    /// `out = |C|ᵀ D_e^{-1} |C| D_k^{-1/2} p`, with the shift `α²` and the outer `D_k^{-1/2}` left to the caller.
    fn unscaled(&self, p: &[f64], out: &mut [f64]) -> (f64, &[f64]) {
        let (s_e, s_k) = self.n.inv_sqrt.split_at(self.cross.n_rows());
        (self.n).with_scaled(p, s_k, |x| self.cross.abs_gram_times(s_e, x, out));
        (self.alpha * self.alpha, s_k)
    }

    fn lift(&self, y_k: &[f64], b_e: Option<&[f64]>) -> Vec<f64> {
        let (s_e, s_k) = self.n.inv_sqrt.split_at(self.cross.n_rows());
        let mut y = vec![0.0; self.n.n()];
        let (y_e, kept) = y.split_at_mut(self.cross.n_rows());
        (self.n).with_scaled(y_k, s_k, |x| self.cross.abs_times(x, y_e));
        for (i, (y, &s)) in y_e.iter_mut().zip(s_e).enumerate() {
            *y = (b_e.map_or(0.0, |b| b[i]) + s * *y) / self.alpha;
        }
        kept.copy_from_slice(y_k);
        y
    }
}
