use std::borrow::Cow;

use crate::storage::{OffDiagonal, Rows};
use crate::{narrow, widen, Real};

/// `N = D^{-1/2}|offdiag(A_c)|D^{-1/2}`: scales `R` make `R|A_c|R` dominant iff `N y ≤ y` for `y = R√D`, so iff `ρ(N) ≤ 1` (Boman, Chen, Parekh, Toledo, LAA 2005).
pub(super) struct Normalized<'a, T> {
    off_diagonal: &'a OffDiagonal<T>,
    inv_sqrt: Box<[f64]>,
}

impl<'a, T: Real> Normalized<'a, T> {
    pub(super) fn new(off_diagonal: &'a OffDiagonal<T>, diagonal: &[T]) -> Self {
        let inv_sqrt = diagonal.iter().map(|&d| 1.0 / widen(d).sqrt()).collect();
        Self {
            off_diagonal,
            inv_sqrt,
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
    ) -> (Shifted<'s, 'a, T>, Vec<f64>) {
        match self.off_diagonal {
            OffDiagonal::General(_) => (Shifted::Whole { n: self, alpha }, b.to_vec()),
            OffDiagonal::Bipartite(cross) => {
                let kept = Kept::new(self, cross, alpha, b);
                let rhs = kept.rhs(b);
                (Shifted::Kept(kept), rhs)
            }
        }
    }

    /// `[min_i, max_i] (Ny)_i / y_i` and `‖Ny − y‖_∞ / ‖y‖_∞`; `None` unless `y` is positive and every ratio finite.
    pub(super) fn ratio_range(
        &self,
        y: &[f64],
        work: &mut [f64],
        out: &mut [f64],
    ) -> Option<(f64, f64, f64)> {
        if !y.iter().all(|&v| v.is_finite() && v > 0.0) {
            return None;
        }
        self.unscaled(y, work, out);
        let (mut low, mut high) = (f64::INFINITY, 0.0f64);
        let (mut residual, mut peak) = (0.0f64, 0.0f64);
        for ((&o, &y), &s) in out.iter().zip(y).zip(&self.inv_sqrt[..]) {
            let t = s * o / y;
            (low, high) = (low.min(t), high.max(t));
            residual = residual.max((s * o - y).abs());
            peak = peak.max(y);
        }
        (low.is_finite() && high.is_finite()).then_some((low, high, residual / peak))
    }

    /// `R = D^{-1/2} y` with `y` peaking at `1`, so `M`'s diagonal never exceeds `1`.
    pub(super) fn scales(&self, y: &[f64]) -> Box<[T]> {
        let peak = y.iter().copied().fold(0.0f64, f64::max);
        (y.iter().zip(&self.inv_sqrt[..]))
            .map(|(&y, &s)| narrow::<T>(y / peak * s))
            .collect()
    }

    /// `out = |O| D^{-1/2} y`, `work` holding `D^{-1/2} y`: `N y` but for the outer `D^{-1/2}`.
    fn unscaled(&self, y: &[f64], work: &mut [f64], out: &mut [f64]) {
        for ((w, &y), &s) in work.iter_mut().zip(y).zip(&self.inv_sqrt[..]) {
            *w = y * s;
        }
        self.off_diagonal.abs_times(work, out);
    }
}

/// The system CG iterates on for one shift `α`.
pub(super) enum Shifted<'n, 'a, T> {
    /// `(αI − N) y = b` over every vertex.
    Whole {
        n: &'n Normalized<'a, T>,
        alpha: f64,
    },
    /// A bipartite `N` with its eliminated side solved out.
    Kept(Kept<'n, 'a, T>),
}

impl<T: Real> Shifted<'_, '_, T> {
    /// `out = S p` for the system's matrix `S`, returning `pᵀp` and `pᵀSp` from the same pass.
    pub(super) fn apply(&self, p: &[f64], work: &mut [f64], out: &mut [f64]) -> (f64, f64) {
        let (shift, s) = match self {
            Self::Whole { n, alpha } => {
                n.unscaled(p, work, out);
                (*alpha, &n.inv_sqrt[..])
            }
            Self::Kept(kept) => kept.unscaled(p, work, out),
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
pub(super) struct Kept<'n, 'a, T> {
    n: &'n Normalized<'a, T>,
    cross: &'a Rows<T>,
    alpha: f64,
    b_e: &'n [f64],
}

impl<'n, 'a, T: Real> Kept<'n, 'a, T> {
    fn new(n: &'n Normalized<'a, T>, cross: &'a Rows<T>, alpha: f64, b: &'n [f64]) -> Self {
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
        let scaled: Vec<f64> = self.b_e.iter().zip(s_e).map(|(b, s)| b * s).collect();
        let mut rhs = vec![0.0; b_k.len()];
        self.cross.abs_transpose_times(&scaled, &mut rhs);
        for ((r, &s), &b) in rhs.iter_mut().zip(s_k).zip(b_k) {
            *r = self.alpha * b + s * *r;
        }
        rhs
    }

    /// `out = |C|ᵀ D_e^{-1} |C| D_k^{-1/2} p`, with the shift `α²` and the outer `D_k^{-1/2}` left to the caller.
    fn unscaled(&self, p: &[f64], work: &mut [f64], out: &mut [f64]) -> (f64, &[f64]) {
        let s_k = &self.n.inv_sqrt[self.cross.n_rows()..];
        for ((w, &p), &s) in work.iter_mut().zip(p).zip(s_k) {
            *w = p * s;
        }
        let s_e = &self.n.inv_sqrt[..self.cross.n_rows()];
        self.cross.abs_gram_times(s_e, work, out);
        (self.alpha * self.alpha, s_k)
    }

    fn lift(&self, y_k: &[f64], b_e: Option<&[f64]>) -> Vec<f64> {
        let (s_e, s_k) = self.n.inv_sqrt.split_at(self.cross.n_rows());
        let scaled: Vec<f64> = y_k.iter().zip(s_k).map(|(y, s)| y * s).collect();
        let mut y = vec![0.0; self.n.n()];
        let (y_e, kept) = y.split_at_mut(self.cross.n_rows());
        self.cross.abs_times(&scaled, y_e);
        for (i, (y, &s)) in y_e.iter_mut().zip(s_e).enumerate() {
            *y = (b_e.map_or(0.0, |b| b[i]) + s * *y) / self.alpha;
        }
        kept.copy_from_slice(y_k);
        y
    }
}
