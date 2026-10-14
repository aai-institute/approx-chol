use super::normalized::Normalized;
use crate::Real;

/// `[min_i, max_i] (Ny)_i / y_i` of a positive `y`, which brackets `ρ(N)` (Collatz–Wielandt).
#[derive(Clone, Copy)]
struct Bracket {
    low: f64,
    high: f64,
}

impl Bracket {
    /// Of a positive `y`; `None` where a ratio is not finite.
    fn of<T: Real>(
        n: &Normalized<'_, T>,
        y: &[f64],
        work: &mut (Vec<f64>, Vec<f64>),
    ) -> Option<Self> {
        let (low, high) = n.ratio_range(y, &mut work.0, &mut work.1)?;
        Some(Self { low, high })
    }

    /// How far `y` leaves `M` short of dominance.
    fn violation(&self) -> f64 {
        (self.high - 1.0).max(0.0)
    }

    fn certifies(&self, tolerance: f64) -> bool {
        self.violation() <= tolerance
    }

    /// `ρ ≥ low` past `1 + tolerance`: no scaling certifies.
    fn refutes(&self, tolerance: f64) -> bool {
        self.low > 1.0 + tolerance
    }

    /// `ρ < 1` proven, or the bracket closed on `ρ`: no later `y` moves the float decision.
    fn settled(&self, band: f64) -> bool {
        self.high < 1.0 - band || self.high - self.low <= band
    }

    /// Just above `ρ`: `1` once certified, as a scalable block's `ρ ≤ 1`, else the top (Noda).
    fn shift(&self, tolerance: f64) -> f64 {
        let top = if self.certifies(tolerance) {
            1.0
        } else {
            self.high
        };
        top + 1000.0 * f64::EPSILON
    }
}

/// Inverse iteration towards `N`'s Perron vector, each step `(αI − N) y' = y` solved by CG from `y' = 0`.
pub(super) struct InverseIteration<'n, 'a, T> {
    n: &'n Normalized<'a, T>,
    tolerance: f64,
    /// How close to `1` a settled bracket closes.
    band: f64,
    max_iterations: usize,
    iterations: usize,
    /// A Rayleigh quotient passed `1 + max(tolerance, band)`: `ρ` is too, so no later `y` certifies.
    refuted: bool,
    /// The best positive `y` so far and its bracket.
    best: (Vec<f64>, Bracket),
    /// `D^{-1/2} p` and the product, for every product with `N`.
    work: (Vec<f64>, Vec<f64>),
}

impl<'n, 'a, T: Real> InverseIteration<'n, 'a, T> {
    /// From `y = 1`; `None` if its bracket is not finite.
    pub(super) fn new(
        n: &'n Normalized<'a, T>,
        tolerance: f64,
        band: f64,
        max_iterations: usize,
    ) -> Option<Self> {
        let mut work = (vec![0.0; n.n()], vec![0.0; n.n()]);
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
        })
    }

    /// `α = 1 + tolerance/2` until a step certifies or refutes, then each step from the bracket's [`shift`](Bracket::shift).
    pub(super) fn run(mut self) -> (Vec<f64>, usize, f64) {
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
        }
        let (y, bracket) = self.best;
        (y, self.iterations, bracket.violation())
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
        while self.iterations < self.max_iterations && !goal(&self.best.1) {
            self.iterations += 1;
            let m = direction.len();
            let (work, image) = (&mut self.work.0[..m], &mut self.work.1[..m]);
            let (direction_norm_squared, curvature) = system.apply(&direction, work, image);
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
                (solution.iter_mut().zip(&mut residual)).zip(direction.iter().zip(&self.work.1[..]))
            {
                *value += step * direction;
                *residual -= step * image;
                next_residual_norm_squared += *residual * *residual;
            }
            if !next_residual_norm_squared.is_finite() {
                break;
            }
            let exhausted = next_residual_norm_squared == 0.0;
            if !exhausted {
                let beta = next_residual_norm_squared / residual_norm_squared;
                for (direction, &residual) in direction.iter_mut().zip(&residual) {
                    *direction = residual + beta * *direction;
                }
                residual_norm_squared = next_residual_norm_squared;
            }
            // A probe costs one more pass; past the cheap early exits, probe on a stride.
            let k = self.iterations - start;
            let last = exhausted || self.iterations == self.max_iterations;
            if last || k <= 8 || k % 4 == 0 {
                adopted |= self.consider(&system.solution(&solution), &goal);
            }
            if exhausted {
                break;
            }
        }
        adopted
    }

    /// Adopts a positive `y` whose bracket ranks above the best's: meeting `goal` first, then the smaller violation.
    fn consider(&mut self, y: &[f64], goal: &impl Fn(&Bracket) -> bool) -> bool {
        let Some(bracket) = Bracket::of(self.n, y, &mut self.work) else {
            return false;
        };
        let rank = |b: &Bracket| (goal(b), -b.violation());
        let better = rank(&bracket) > rank(&self.best.1);
        if better {
            self.best.0.clear();
            self.best.0.extend_from_slice(y);
            self.best.1 = bracket;
        }
        better
    }
}
