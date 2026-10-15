use super::normalized::Normalized;
use crate::{Index, Real};

/// `[min_i, max_i] (Ny)_i / y_i` of a positive `y`, which brackets `ρ(N)` (Collatz 1942, Wielandt 1950).
#[derive(Clone, Copy)]
struct Bracket {
    low: f64,
    high: f64,
    /// `‖Ny − y‖_∞ / ‖y‖_∞`, the eigen-residual at `ρ = 1` (Jia, Lin, Liu's outer stop).
    eigen_residual: f64,
}

impl Bracket {
    /// Of a positive `y`; `None` unless every ratio is finite.
    fn of<T: Real, I: Index>(
        n: &Normalized<'_, T, I>,
        y: &[f64],
        work: &mut [f64],
    ) -> Option<Self> {
        if !y.iter().all(|&v| v.is_finite() && v > 0.0) {
            return None;
        }
        let scales = n.unscaled(y, work);
        let (mut low, mut high) = (f64::INFINITY, 0.0f64);
        let (mut residual, mut peak) = (0.0f64, 0.0f64);
        for ((&o, &y), &s) in work.iter().zip(y).zip(scales) {
            let t = s * o / y;
            (low, high) = (low.min(t), high.max(t));
            residual = residual.max((s * o - y).abs());
            peak = peak.max(y);
        }
        (low.is_finite() && high.is_finite()).then_some(Self {
            low,
            high,
            eigen_residual: residual / peak,
        })
    }

    /// How far `y` leaves `M` short of dominance.
    fn violation(&self) -> f64 {
        (self.high - 1.0).max(0.0)
    }

    fn certifies(&self, tolerance: f64) -> bool {
        self.violation() <= tolerance
    }

    /// `ρ ≥ low` past `1 + tolerance`: no scaling certifies (Guan, Lu, Li, Shao, JCAM 302 (2016), Thm 2.1).
    fn refutes(&self, tolerance: f64) -> bool {
        self.low > 1.0 + tolerance
    }

    /// `ρ < 1` proven, or `y` a Perron vector of `ρ = 1` to rounding: no later `y` moves the float decision.
    fn settled(&self, band: f64) -> bool {
        // Not the width, which never closes once `y` has components below `ε` (Li & Li, arXiv:2111.12642 §3).
        self.high < 1.0 - band || self.eigen_residual <= band
    }

    /// Certified and near settling: a long CG run stalls at a floor that grows with `n`, past `4×` the band at `n ≈ 5000`.
    fn nearly_settled(&self, tolerance: f64, band: f64) -> bool {
        self.certifies(tolerance) && self.eigen_residual <= 64.0 * band
    }

    /// Just above `ρ`: `1` once certified, as a scalable block's `ρ ≤ 1`, else the top (Noda 1971).
    fn shift(&self, tolerance: f64) -> f64 {
        let top = if self.certifies(tolerance) {
            1.0
        } else {
            self.high
        };
        top + 1000.0 * f64::EPSILON
    }
}

/// Inverse iteration towards `N`'s Perron vector, each step `(αI − N) y' = y` solved by CG from `y' = 0` (Jia, Lin, Liu, arXiv:1309.3926).
pub(super) struct InverseIteration<'n, 'a, T, I> {
    n: &'n Normalized<'a, T, I>,
    tolerance: f64,
    /// The rounding band a settled bracket closes to, around `ρ = 1`.
    band: f64,
    max_iterations: usize,
    iterations: usize,
    /// A Rayleigh quotient passed `1 + max(tolerance, band)`: `ρ` is too, so no later `y` certifies.
    refuted: bool,
    /// The best positive `y` so far and its bracket.
    best: (Vec<f64>, Bracket),
    /// The product of every product with `N`.
    work: Vec<f64>,
    /// Steps a nearly settled `best` may go unimproved; doubles per restart, so stalls cannot chain.
    patience: usize,
}

impl<'n, 'a, T: Real, I: Index> InverseIteration<'n, 'a, T, I> {
    /// From `y = 1`; `None` if its bracket is not finite.
    pub(super) fn new(
        n: &'n Normalized<'a, T, I>,
        tolerance: f64,
        band: f64,
        max_iterations: usize,
    ) -> Option<Self> {
        let mut work = vec![0.0; n.n()];
        let unit = vec![1.0; n.n()];
        let bracket = Bracket::of(n, &unit, &mut work)?;
        Some(Self {
            n,
            tolerance,
            band,
            max_iterations,
            iterations: 0,
            refuted: false,
            best: (unit, bracket),
            work,
            patience: 64,
        })
    }

    /// `α = 1 + tolerance/2` until a step certifies or refutes (as `M⁻¹𝟙` does: Ahmadinejad et al., arXiv:1810.02348 §3.1), then each step from the bracket's [`shift`](Bracket::shift).
    pub(super) fn run(mut self) -> (Vec<f64>, usize, f64, bool) {
        let (tolerance, band) = (self.tolerance, self.band);
        let polished =
            |b: &Bracket| (b.certifies(tolerance) && b.settled(band)) || b.refutes(tolerance);
        let unit = vec![1.0; self.n.n()];
        self.solve(1.0 + 0.5 * tolerance, unit, |b| {
            b.certifies(tolerance) || b.refutes(tolerance)
        });
        while !self.refuted && !polished(&self.best.1) && self.iterations < self.max_iterations {
            let (y, bracket) = &self.best;
            let peak = y.iter().copied().fold(0.0f64, f64::max);
            let rhs = y.iter().map(|&y| y / peak).collect();
            if !self.solve(bracket.shift(tolerance), rhs, polished) {
                break;
            }
            self.patience = self.patience.saturating_mul(2);
        }
        let (y, bracket) = self.best;
        let settled = bracket.certifies(tolerance) && bracket.settled(band);
        (y, self.iterations, bracket.violation(), settled)
    }

    /// CG on `(αI − N) y = b`; each probed iterate, and a vanishing curvature's or refuting Rayleigh quotient's direction, replaces `best` when it ranks above it. Whether one did.
    fn solve(&mut self, alpha: f64, b: Vec<f64>, goal: impl Fn(&Bracket) -> bool) -> bool {
        let n = self.n;
        let (system, rhs) = n.shifted(alpha, &b);
        let mut adopted = false;
        let mut residual_norm_squared: f64 = rhs.iter().map(|x| x * x).sum();
        let mut solution: Vec<f64> = vec![0.0; rhs.len()];
        let (mut residual, mut direction) = (rhs.clone(), rhs);
        let start = self.iterations;
        let mut last_adoption = start;
        while self.iterations < self.max_iterations && !goal(&self.best.1) {
            self.iterations += 1;
            let m = direction.len();
            let image = &mut self.work[..m];
            let (direction_norm_squared, curvature) = system.apply(&direction, image);
            if !(direction_norm_squared.is_finite() && curvature.is_finite()) {
                break;
            }
            let vanishing = curvature <= 64.0 * f64::EPSILON * direction_norm_squared;
            let floor = system.spectral_floor(direction_norm_squared, curvature);
            // Past the rounding band too: an f32 block's `N` reads a few ulps over a singular `ρ = 1`.
            self.refuted = floor > 1.0 + self.tolerance.max(self.band);
            if vanishing || self.refuted {
                // Either exposes a near-Perron vector: a singular boundary's zero curvature, or the refuting direction.
                let witness: Vec<f64> = direction.iter().map(|value| value.abs()).collect();
                adopted |= self.consider(&system.direction(&witness), &goal);
                let iterate: Vec<f64> = solution.iter().map(|value| value.abs()).collect();
                adopted |= self.refuted && self.consider(&system.solution(&iterate), &goal);
                break;
            }
            let step = residual_norm_squared / curvature;
            let mut next_residual_norm_squared = 0.0;
            for ((value, residual), (&direction, &image)) in
                (solution.iter_mut().zip(&mut residual)).zip(direction.iter().zip(&self.work[..]))
            {
                *value += step * direction;
                *residual -= step * image;
                next_residual_norm_squared += *residual * *residual;
            }
            if !next_residual_norm_squared.is_finite() {
                break;
            }
            // A long run's iterate stalls a few ulps above the band; a fresh solve from `best` polishes it.
            let stalled = self.best.1.nearly_settled(self.tolerance, self.band)
                && self.iterations - last_adoption >= self.patience;
            let exhausted = next_residual_norm_squared == 0.0 || stalled;
            if !exhausted {
                let beta = next_residual_norm_squared / residual_norm_squared;
                for (direction, &residual) in direction.iter_mut().zip(&residual) {
                    *direction = residual + beta * *direction;
                }
                residual_norm_squared = next_residual_norm_squared;
            }
            // A probe costs one more pass: past the cheap early exits, a stride of `k/8` sees a goal within 12.5% more steps.
            let k = self.iterations - start;
            let last = exhausted || self.iterations == self.max_iterations;
            let on_stride = k % (k / 8).max(4) == 0;
            if (last || k <= 8 || on_stride) && self.consider(&system.solution(&solution), &goal) {
                adopted = true;
                last_adoption = self.iterations;
            }
            if exhausted {
                break;
            }
        }
        adopted
    }

    /// Adopts a positive `y` whose bracket ranks above the best's: meeting `goal` first, then certified by the smaller eigen-residual, else by the smaller violation.
    fn consider(&mut self, y: &[f64], goal: &impl Fn(&Bracket) -> bool) -> bool {
        let Some(bracket) = Bracket::of(self.n, y, &mut self.work) else {
            return false;
        };
        let tolerance = self.tolerance;
        // Certified, only the eigen-residual moves towards settling; a smaller violation alone resumes a restart from a stale best.
        let rank = |b: &Bracket| {
            let certified = b.certifies(tolerance);
            let progress = if certified {
                -b.eigen_residual
            } else {
                -b.violation()
            };
            (goal(b), certified, progress)
        };
        let better = rank(&bracket) > rank(&self.best.1);
        if better {
            self.best.0.clear();
            self.best.0.extend_from_slice(y);
            self.best.1 = bracket;
        }
        better
    }
}
