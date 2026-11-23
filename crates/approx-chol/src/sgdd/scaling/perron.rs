use super::normalized::{Normalized, Shifted};
use super::{ScaleVector, ScalingOutcome};
use crate::{Index, Real};

/// `[min_i, max_i] (Ny)_i / y_i` of a positive `y`, which brackets `ρ(N)` (Collatz 1942, Wielandt 1950).
#[derive(Clone, Copy)]
struct Bracket {
    low: f64,
    high: f64,
    /// `‖Ny − y‖_∞ / ‖y‖_∞`, the eigen-residual at `ρ = 1` (Jia, Lin, Liu's outer stop).
    eigen_residual: f64,
    /// A proven `ρ ≥ floor`: the least ratio less its own row's rounding, or `y`'s Rayleigh quotient, `N` symmetric, where it can decide settling, less its rounding; an iteration raises it to every bracket it has seen.
    floor: f64,
}

impl Bracket {
    /// Of a positive `y`, row `i`'s `(Ny)_i` rounded by at most `rows[i]` relative, its floor `floor` or above, raised by the Rayleigh quotient where `rayleigh` asks for it; `Err` at the first row whose `y` is not positive or whose ratio is not finite.
    fn of<T: Real, I: Index>(
        n: &Normalized<'_, T, I>,
        y: &[f64],
        work: &mut [f64],
        rows: &[f64],
        floor: f64,
        rayleigh: impl FnOnce(&Self) -> bool,
    ) -> Result<Self, usize> {
        if let Some(i) = y.iter().position(|&v| !(v.is_finite() && v > 0.0)) {
            return Err(i);
        }
        let scales = n.unscaled(y, work);
        let ratio = |((&o, &y), &s): ((&f64, &f64), &f64)| s * o / y;
        let ratios = || work.iter().zip(y).zip(scales);
        let (mut low, mut high, mut finite) = (f64::INFINITY, 0.0f64, true);
        let (mut residual, mut peak, mut least) = (0.0f64, 0.0f64, f64::INFINITY);
        for (entry @ ((&o, &y), &s), &row) in ratios().zip(rows) {
            let t = ratio(entry);
            finite &= t.is_finite();
            (low, high) = (low.min(t), high.max(t));
            // `ρ ≥ min_i t_i` exactly (Collatz–Wielandt), each `t_i` computed within its own row's rounding.
            least = least.min(t * (1.0 - row));
            residual = residual.max((s * o - y).abs());
            peak = peak.max(y);
        }
        if !finite {
            let row = ratios().position(|entry| !ratio(entry).is_finite());
            return Err(row.expect("a ratio that is not finite has a row"));
        }
        let mut bracket = Self {
            low,
            high,
            eigen_residual: residual / peak,
            floor: least.max(floor),
        };
        if rayleigh(&bracket) {
            // `yᵀy`, `yᵀ(Ny − y)`, `Σ y_i |(Ny − y)_i|` and `yᵀNy` from the product already formed.
            let (mut norm, mut excess, mut spread, mut product) = (0.0, 0.0, 0.0, 0.0);
            for (((&o, &y), &s), &row) in ratios().zip(rows) {
                let difference = s * o - y;
                norm += y * y;
                excess += y * difference;
                spread += y * difference.abs();
                product += row * y * (s * o);
            }
            // First order, each `(Ny)_i` rounds by its own row's `(k_i + 3)ε`, charged in `product`, and each difference, product and `m`-term sum by `ε` a term; doubled, that bounds the whole error while `(m + k + 3)ε ≤ 1/2` for every `k_i`. The final division adds `ε |RQ − 1| ≤ ε`, as a certified bracket has `RQ ≤ high < 2`.
            let (m, epsilon) = (y.len() as f64, f64::EPSILON);
            let error = 2.0 * (product + (m + 2.0) * epsilon * spread);
            let quotient = (excess - error) / (norm * (1.0 + (m + 1.0) * epsilon));
            // Only a floor past `1` decides anything; below it the widened `yᵀy` is the wrong side, and the floor stays the ratios'.
            if quotient > 0.0 {
                bracket.floor = bracket.floor.max(1.0 + quotient - epsilon);
            }
        }
        Ok(bracket)
    }

    /// How far `y` leaves `M` short of dominance.
    fn violation(&self) -> f64 {
        (self.high - 1.0).max(0.0)
    }

    fn certifies(&self, tolerance: f64) -> bool {
        self.violation() <= tolerance
    }

    /// `ρ ≥ low` past `ceiling`: no scaling certifies (Guan, Lu, Li, Shao, JCAM 302 (2016), Thm 2.1).
    fn refutes(&self, ceiling: f64) -> bool {
        self.low > ceiling
    }

    /// `ρ < 1` proven, or `y` a Perron vector of `ρ = 1` to rounding, which a floor past `1 + band` rules out: no later `y` moves the float decision.
    fn settled(&self, tolerance: f64, band: f64) -> bool {
        // Not the width, which never closes once `y` has components below `ε` (Li & Li, arXiv:2111.12642 §3).
        let near_one = self.floor <= 1.0 + band;
        self.certifies(tolerance)
            && (self.high < 1.0 - band || near_one && self.eigen_residual <= band)
    }

    /// Certified with `ρ ≤ 1 + band` still open: only then can `y` settle as a Perron vector of `ρ = 1`, or its Rayleigh quotient decide that.
    fn open(&self, tolerance: f64, band: f64) -> bool {
        self.certifies(tolerance) && self.floor <= 1.0 + band
    }

    /// Certified and near settling: a CG run stalls at a floor that grows with its steps, past `4×` the band after ~5000.
    fn nearly_settled(&self, tolerance: f64, band: f64) -> bool {
        self.certifies(tolerance) && self.eigen_residual <= 64.0 * band
    }

    /// Just above `ρ ≤ high`, never under a singular scalable block's `ρ = 1` (Noda 1971).
    fn shift(&self) -> f64 {
        self.high.max(1.0) + 1000.0 * f64::EPSILON
    }
}

/// INI 1's inner tolerance: `‖f_k‖₂ ≤ γ min(x_k)` keeps `x_k + f_k`, so `y_{k+1}`, positive (Jia, Lin, Liu, Numer. Math. 130 (2015), Lemma 3).
const GAMMA: f64 = 0.8;

/// `g`, a Noda step's least shift past the bracket's top, at the curvature a CG step resolves. The exact top is `≥ ρ`, and the computed one sits at most `rounding · top` under it, first order, past `g` from degree 62 on: shifted by `max(g, rounding · top)`, `α ≥ ρ` to first order, and `αI − N` keeps every eigenvalue `≥ g − rounding · top` where that is positive.
// Departs from Jia, Lin, Liu's `α_k = high_k`: positivity holds for any `α > ρ`, as `(αI − N)⁻¹ ≥ 0`, but an inexact step only bounds `high_{k+1} < α_k`.
const GAP: f64 = 64.0 * f64::EPSILON;

/// Inverse iteration towards `N`'s Perron vector, each step `(αI − N) y' = y` solved by CG from `y' = 0`: a first solve gives `x₀`, inexact Noda steps approach `ρ` from it (Jia, Lin, Liu, Numer. Math. 130 (2015), Alg. 2, INI 1, shifted by [`GAP`]), and fixed-shift restarts polish the last.
pub(super) struct InverseIteration<'n, 'a, T, I> {
    n: &'n Normalized<'a, T, I>,
    tolerance: f64,
    /// The rounding band a settled bracket closes to, around `ρ = 1`.
    band: f64,
    /// `1 + max(tolerance, band)`: a bound on `ρ` past it refutes, as an `f32` block's `N` reads a few ulps over a singular `ρ = 1`.
    ceiling: f64,
    max_iterations: usize,
    /// How far each computed `(Ny)_i` may sit off, relative: `(k_i + 3)ε` over row `i`'s `k_i` positive terms.
    rows: Box<[f64]>,
    /// The largest of `rows`, which bounds how far, relative, a computed top may sit under `ρ`: a Noda shift clears it.
    rounding: f64,
    iterations: usize,
    /// A Rayleigh quotient passed the ceiling: `ρ` is too, so no later `y` certifies.
    refuted: bool,
    /// The best positive `y` so far and its bracket.
    best: (Vec<f64>, Bracket),
    /// The product of every product with `N`.
    work: Vec<f64>,
    /// Steps a nearly settled `best` may go unimproved; doubles per restart and per stall before an adoption, so stalls cannot chain.
    patience: usize,
}

impl<'n, 'a, T: Real, I: Index> InverseIteration<'n, 'a, T, I> {
    /// From `y = 1`; `Err` at the row whose ratio is not finite.
    pub(super) fn new(
        n: &'n Normalized<'a, T, I>,
        tolerance: f64,
        band: f64,
        rows: Box<[f64]>,
        rounding: f64,
        max_iterations: usize,
    ) -> Result<Self, usize> {
        let mut work = vec![0.0; n.n()];
        let unit = vec![1.0; n.n()];
        let open = |b: &Bracket| b.open(tolerance, band);
        let bracket = Bracket::of(n, &unit, &mut work, &rows, f64::NEG_INFINITY, open)?;
        Ok(Self {
            n,
            tolerance,
            band,
            ceiling: 1.0 + tolerance.max(band),
            max_iterations,
            rows,
            rounding,
            iterations: 0,
            refuted: false,
            best: (unit, bracket),
            work,
            patience: 64,
        })
    }

    /// `α = 1 + tolerance/2` until a step certifies or refutes (as `M⁻¹𝟙` does: Ahmadinejad et al., arXiv:1810.02348 §3.1); unless that refuted, [Noda steps](Self::noda_step) while each lowers the top and `best` is not yet near settling; then each step from the bracket's [`shift`](Bracket::shift) until it settles, refutes, or a certified `best` proves `ρ > 1 + band`, which [`settled`](Bracket::settled) excludes and no restart can refute.
    pub(super) fn run(mut self) -> (Vec<f64>, ScaleVector) {
        let (tolerance, band, ceiling) = (self.tolerance, self.band, self.ceiling);
        let polished = |b: &Bracket| b.settled(tolerance, band) || b.refutes(ceiling);
        // Within `64×` the band of `ρ = 1`, main's polish is measured to settle (`nearly_settled`); a block proven past `1 + band` keeps approaching, so its clamp falls to `ρ − 1`.
        let near = |b: &Bracket| b.open(tolerance, band) && b.high - 1.0 <= 64.0 * band;
        let unit = vec![1.0; self.n.n()];
        self.solve(1.0 + 0.5 * tolerance, unit, |b| {
            b.certifies(tolerance) || b.refutes(ceiling)
        });
        if !self.refuted {
            while !polished(&self.best.1) && !near(&self.best.1) && self.noda_step(&polished) {}
        }
        // Certified, `ρ ≤ high ≤ ceiling` cannot refute, and a floor past `1 + band` cannot settle; uncertified, a restart may still certify or refute.
        let finished =
            |b: &Bracket| polished(b) || b.certifies(tolerance) && !b.open(tolerance, band);
        while !self.refuted && !finished(&self.best.1) && self.iterations < self.max_iterations {
            let (y, bracket) = &self.best;
            let peak = y.iter().copied().fold(0.0f64, f64::max);
            let rhs = y.iter().map(|&y| y / peak).collect();
            if !self.solve(bracket.shift(), rhs, finished) {
                break;
            }
            self.patience = self.patience.saturating_mul(2);
        }
        let (y, bracket) = self.best;
        let scaling = if bracket.certifies(tolerance) {
            ScalingOutcome::Certified
        } else if self.refuted || bracket.refutes(ceiling) {
            ScalingOutcome::Refuted
        } else {
            ScalingOutcome::Unsettled
        };
        let vector = ScaleVector::PerronVector {
            band,
            violation: bracket.violation(),
            scaling,
            settled: bracket.settled(tolerance, band),
        };
        (y, vector)
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
            self.refuted = floor > self.ceiling;
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
            // A long run's iterate stalls a few ulps above the band; a fresh solve from an improved `best` polishes it.
            let stalled = self.best.1.nearly_settled(self.tolerance, self.band)
                && self.iterations - last_adoption >= self.patience;
            // From an unchanged `best`, a restart would replay this solve step for step.
            if stalled && !adopted {
                self.patience = self.patience.saturating_mul(2);
            }
            let exhausted = next_residual_norm_squared == 0.0 || (stalled && adopted);
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

    /// One INI 1 step from `x_k = best`, `α_k = high_k + max(g, rounding · high_k)`: CG until `‖f_k‖₂ ≤ γ min(x_k)`. `(α_k I − N)⁻¹` is nonnegative and commutes with `N`, so `y = (α_k I − N)⁻¹ (x_k + f_k)` is positive, which [`Bracket::of`] checks in rounding, and an exact solve never raises the top. A `y` that lowers the top is taken, the approach's own goal; whether one was. The approach ends at the first that is not, which still goes to [`consider`](Self::consider), at the budget, or at curvature `≤ 0`, whose iterate and `|direction|` are considered as [`solve`](Self::solve) considers them.
    fn noda_step(&mut self, goal: &impl Fn(&Bracket) -> bool) -> bool {
        let n = self.n;
        let (x, bracket) = &self.best;
        let top = bracket.high;
        let (peak, least) =
            (x.iter()).fold((0.0f64, f64::INFINITY), |(p, l), &x| (p.max(x), l.min(x)));
        let b: Vec<f64> = x.iter().map(|&x| x / peak).collect();
        // Division by `peak > 0` rounds monotonically: `least/peak` is `min(b)`.
        let bound = GAMMA * (least / peak);
        let (system, rhs) = n.shifted(top + GAP.max(self.rounding * top), &b);
        let mut cg = Cg::new(rhs);
        while system.whole_residual(cg.residual_norm_squared.sqrt()) > bound {
            if self.iterations == self.max_iterations {
                self.consider(&system.solution(&cg.solution), goal);
                return false;
            }
            self.iterations += 1;
            let Some(rayleigh) = cg.step(&system, &mut self.work) else {
                let witness: Vec<f64> = cg.direction.iter().map(|value| value.abs()).collect();
                self.consider(&system.direction(&witness), goal);
                self.consider(&system.solution(&cg.solution), goal);
                return false;
            };
            // A refuting Rayleigh quotient is recorded and the step still taken, so the top keeps falling towards `ρ`.
            self.refuted |= rayleigh > self.ceiling;
        }
        let y = system.solution(&cg.solution).into_owned();
        match self.bracket(&y) {
            Some(next) if next.high < top => {
                self.best = (y, next);
                true
            }
            Some(_) => {
                self.consider(&y, goal);
                false
            }
            None => false,
        }
    }

    /// `y`'s bracket if positive, its floor and best's raised to the higher: what any `y` proved of `ρ` holds for every bracket, and spares the Rayleigh pass once it is past `1 + band`.
    fn bracket(&mut self, y: &[f64]) -> Option<Bracket> {
        let (tolerance, band) = (self.tolerance, self.band);
        // The Rayleigh quotient decides settling only for a certified `y` whose floor leaves `ρ ≤ 1 + band` open.
        let rayleigh = |b: &Bracket| b.open(tolerance, band);
        let floor = self.best.1.floor;
        let bracket = Bracket::of(self.n, y, &mut self.work, &self.rows, floor, rayleigh).ok()?;
        self.best.1.floor = bracket.floor;
        Some(bracket)
    }

    /// Adopts a positive `y` whose bracket ranks above the best's: meeting `goal` first, then certified by the smaller eigen-residual, else by the smaller violation.
    fn consider(&mut self, y: &[f64], goal: &impl Fn(&Bracket) -> bool) -> bool {
        let Some(bracket) = self.bracket(y) else {
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

/// A Noda step's CG on a [`Shifted`] system from `y = 0`.
struct Cg {
    solution: Vec<f64>,
    residual: Vec<f64>,
    direction: Vec<f64>,
    residual_norm_squared: f64,
}

impl Cg {
    fn new(rhs: Vec<f64>) -> Self {
        Self {
            residual_norm_squared: rhs.iter().map(|x| x * x).sum(),
            solution: vec![0.0; rhs.len()],
            residual: rhs.clone(),
            direction: rhs,
        }
    }

    /// One step, its product in `work`: the Rayleigh quotient of the direction it moved along, which bounds `ρ ≥` it; `None` at curvature `≤ 0` or past `f64`.
    fn step<T: Real, I: Index>(
        &mut self,
        system: &Shifted<'_, '_, T, I>,
        work: &mut [f64],
    ) -> Option<f64> {
        let image = &mut work[..self.direction.len()];
        let (direction_norm_squared, curvature) = system.apply(&self.direction, image);
        if !(direction_norm_squared.is_finite() && curvature.is_finite()) || curvature <= 0.0 {
            return None;
        }
        let rayleigh = system.spectral_floor(direction_norm_squared, curvature);
        let step = self.residual_norm_squared / curvature;
        let mut next_residual_norm_squared = 0.0;
        let pairs = (self.solution.iter_mut()).zip(&mut self.residual);
        for ((value, residual), (&direction, &image)) in
            pairs.zip(self.direction.iter().zip(&*image))
        {
            *value += step * direction;
            *residual -= step * image;
            next_residual_norm_squared += *residual * *residual;
        }
        if !next_residual_norm_squared.is_finite() {
            return None;
        }
        let beta = next_residual_norm_squared / self.residual_norm_squared;
        for (direction, &residual) in self.direction.iter_mut().zip(&self.residual) {
            *direction = residual + beta * *direction;
        }
        self.residual_norm_squared = next_residual_norm_squared;
        Some(rayleigh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{OffDiagonal, Rows, UpperAdjacency};

    /// `A`'s strict upper triangle from `edges` `(i, j, w)`, `i < j`, by row then column, each `−w (1 + η)`; `A`'s diagonal, the unraised degrees; each row's `(k + 3)ε`.
    fn raised(
        n: usize,
        edges: impl IntoIterator<Item = (usize, usize, f64)>,
        eta: f64,
    ) -> (OffDiagonal<f64, u32>, Vec<f64>, Vec<f64>) {
        let (mut ptrs, mut indices, mut values) = (vec![0], Vec::new(), Vec::new());
        let mut degree = vec![0.0; n];
        for (i, j, w) in edges {
            ptrs.resize(i + 1, indices.len());
            indices.push(j as u32);
            values.push(-w * (1.0 + eta));
            degree[i] += w;
            degree[j] += w;
        }
        ptrs.resize(n + 1, indices.len());
        let off_diagonal =
            OffDiagonal::General(UpperAdjacency::new(Rows::new(ptrs, indices, values, n)));
        let rows = (off_diagonal.degrees().into_iter())
            .map(|k| (k + 3) as f64 * f64::EPSILON)
            .collect();
        (off_diagonal, degree, rows)
    }

    /// `y`'s bracket under `N = D^{-1/2}|A|D^{-1/2}`, its floor from `y` alone.
    fn bracket(
        (off_diagonal, degree, rows): &(OffDiagonal<f64, u32>, Vec<f64>, Vec<f64>),
        y: &[f64],
        (tolerance, band): (f64, f64),
    ) -> Bracket {
        let normalized = Normalized::new(off_diagonal, degree, crate::threads::ONE);
        let mut work = vec![0.0; y.len()];
        let open = |b: &Bracket| b.open(tolerance, band);
        Bracket::of(&normalized, y, &mut work, rows, f64::NEG_INFINITY, open).expect("positive")
    }

    /// A `50 × 50` grid and a hub joined to 2400 of its vertices, off-diagonals raised by `1 + η` over a diagonal of unraised degrees: `y = √degree` is `N`'s Perron vector, `ρ = 1 + η`, and `η = 1.5×` the hub's band. Charged the hub's `(k + 3)ε` on every row, the Rayleigh floor stayed under `1 + band`, so the endgame ran to the budget.
    #[test]
    fn a_hub_charges_its_rounding_to_its_own_row_so_the_floor_clears_the_band() {
        let (k, h) = (50, 2400);
        let band = (h + 3) as f64 * f64::EPSILON;
        let weight = |e: usize| 0.5 + ((7919 * e) % 1000) as f64 / 1000.0;
        // The hub, vertex `k²`, is the last column of each vertex it joins.
        let mut joins = vec![false; k * k];
        (0..h).for_each(|t| joins[(t * 7919) % (k * k)] = true);
        let edges = (0..k * k).flat_map(|v| {
            let right = (v % k + 1 < k).then_some(v + 1);
            let down = (v + k < k * k).then_some(v + k);
            let hub = joins[v].then_some(k * k);
            [right, down, hub]
                .into_iter()
                .flatten()
                .map(move |j| (v, j))
        });
        let edges = edges.enumerate().map(|(e, (i, j))| (i, j, weight(e)));
        let block = raised(k * k + 1, edges, 1.5 * band);
        let y: Vec<f64> = block.1.iter().map(|d| d.sqrt()).collect();
        let bracket = bracket(&block, &y, (1e-9, band));
        assert!(bracket.certifies(1e-9));
        assert!(
            bracket.floor > 1.0 + band,
            "floor − 1 = {:.3} band",
            (bracket.floor - 1.0) / band
        );
    }

    /// A star of 2400 unit leaves, its hub's component `1 − δ` of the Perron vector's: each leaf's ratio is the least, `1 − δ`, and the hub's `1/(1 − δ)`. Uncertified, the Rayleigh pass is skipped, so the floor is the ratios': each leaf's less its own `4ε`, where charged the hub's `2403ε` it sat that far under.
    #[test]
    fn the_least_ratio_is_charged_its_own_rows_rounding() {
        let (h, delta) = (2400, 1e-6);
        let block = raised(h + 1, (0..h).map(|j| (j, h, 1.0)), 0.0);
        let mut y = vec![1.0; h + 1];
        y[h] = (h as f64).sqrt() * (1.0 - delta);
        let band = (h + 3) as f64 * f64::EPSILON;
        let bracket = bracket(&block, &y, (1e-9, band));
        assert!(!bracket.certifies(1e-9));
        // In `ε`, how far the floor sits under the least ratio.
        let under = (1.0 - bracket.floor / bracket.low) / f64::EPSILON;
        assert!(under < 64.0, "{under:.1}ε under the least ratio");
    }
}
