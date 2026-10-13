mod cg;

use crate::storage::{narrow, widen, CrossBlock, UpperAdjacency};
use crate::Real;

/// Certification policy for a dominance scaling; a deficit left at hand-over is clamped and reported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingPolicy {
    tolerance: f64,
    max_iterations: usize,
}

impl ScalingPolicy {
    /// `None` unless `0 ≤ tolerance < 1`.
    pub fn new(tolerance: f64, max_iterations: usize) -> Option<Self> {
        (0.0..1.0).contains(&tolerance).then_some(Self {
            tolerance,
            max_iterations,
        })
    }
}

impl Default for ScalingPolicy {
    fn default() -> Self {
        Self {
            tolerance: 1e-9,
            max_iterations: 2048,
        }
    }
}

/// A component's uncertified scaling: present iff its violation exceeded the policy's tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingReport {
    /// Scaling iterations spent.
    pub iterations: usize,
    /// Largest relative dominance violation, clamped away.
    pub violation: f64,
    /// Whether a certified scaling exists: the shortfall is the budget's, or perhaps the input's.
    pub scaling: Existence,
}

/// Whether a dominance scaling is known to exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Existence {
    /// A factor-width-2 Gram, GDD by type: only the policy's budget ran out.
    Guaranteed,
    /// Rows: the input itself may not be GDD.
    Unknown,
}

impl ScalingPolicy {
    /// `Σ_j |a_ij| ≤ a_ii (1 + max(tolerance, 2 deg_i ε))`: a rounded Laplacian stays bit for bit.
    pub(super) fn dominates<T: Real>(&self, d: T, abs_row_sum: T, degree: usize) -> bool {
        let slack = (self.tolerance).max(2.0 * degree as f64 * widen(T::epsilon()));
        widen(abs_row_sum) <= widen(d) * (1.0 + slack)
    }

    /// Reported when its violation or clamped deficit, whichever is worse, passes the tolerance.
    pub(super) fn report<T: Real>(
        &self,
        relaxation: &Relaxation<T>,
        existence: Existence,
        clamped: f64,
    ) -> Option<ScalingReport> {
        let violation = widen(relaxation.violation).max(clamped);
        (violation > self.tolerance).then_some(ScalingReport {
            iterations: relaxation.iterations,
            violation,
            scaling: existence,
        })
    }
}

/// Positive scales `R` making `R|A|R` weakly dominant.
pub(super) enum Scaling<T> {
    /// Every SDDM input, so an SDDM CSR reduces bit for bit as before.
    AlreadyDominant,
    /// Uncertified until the hand-over shows its clamped deficit.
    Scaled(Relaxation<T>, Existence),
}

/// What an iteration reached: scales and how far short of dominance they leave `A`.
pub(super) struct Relaxation<T> {
    /// `f = μ/√a_ii` with `μ` peaking at `1`, so `M`'s diagonal `μ²` never exceeds `1`.
    pub(super) scales: Box<[T]>,
    iterations: usize,
    violation: T,
}

/// `max_i (Σ_j|m_ij| − m_ii)/m_ii` outside each row's rounding band: what clamping adds.
pub(super) fn clamped_deficit<T: Real>(
    diagonal: &[T],
    abs_row_sums: &[T],
    degrees: &[usize],
) -> f64 {
    (diagonal.iter().zip(abs_row_sums).zip(degrees))
        .map(|((&d, &s), &degree)| {
            let (d, s) = (widen(d), widen(s));
            match s - d > rounding_band::<T>(d, s, degree) {
                true => (s - d) / d,
                false => 0.0,
            }
        })
        .fold(0.0, f64::max)
}

/// `t ε (m_ii + Σ_j|m_ij|)`, `t = deg + 1`: how far rounding alone moves a surplus.
fn rounding_band<T: Real>(d: f64, s: f64, degree: usize) -> f64 {
    widen(T::epsilon()) * (d + s) * (degree + 1) as f64
}

/// A surplus inside its rounding band reads zero, so a balanced component floats.
pub(super) fn unscaled_surplus<T: Real>(d: T, abs_row_sum: T, degree: usize) -> T {
    let (d, s) = (widen(d), widen(abs_row_sum));
    match d - s > rounding_band::<T>(d, s, degree) {
        true => narrow(d - s),
        false => T::zero(),
    }
}

/// `|u|` of a certified null vector: no iteration runs; `None` if a scale leaves `T`.
pub(super) fn null_scaling<T: Real>(u: &[f64], diagonal: &[T]) -> Option<Relaxation<T>> {
    let peak =
        (u.iter().zip(diagonal)).fold(0.0f64, |peak, (&x, &d)| peak.max(x.abs() * widen(d).sqrt()));
    let scales: Box<[T]> = u.iter().map(|&x| narrow::<T>(x.abs() / peak)).collect();
    (scales.iter().all(|&f| f.is_finite() && f > T::zero())).then(|| Relaxation {
        scales,
        iterations: 0,
        violation: T::zero(),
    })
}

/// Self-corrective iterations (Guan, Lu, Li, Shao 2016, Algorithm 3), handing over the best iterate.
pub(super) fn sci<T: Real>(
    adjacency: &UpperAdjacency<T>,
    diagonal: &[T],
    policy: &ScalingPolicy,
) -> Option<Relaxation<T>> {
    let diagonal: Vec<f64> = diagonal.iter().map(|&d| widen(d)).collect();
    let n = diagonal.len();
    let mut scales = vec![1.0f64; n];
    let mut t = vec![0.0f64; n];
    let mut best = (f64::INFINITY, scales.clone());
    let mut iterations = 0;
    loop {
        ratios(adjacency, &diagonal, &scales, &mut t);
        let (t_p, t_q) = t.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
        if !(t_p.is_finite() && t_q.is_finite()) {
            return None;
        }
        let violation = (t_q - 1.0).max(0.0);
        if violation < best.0 {
            best = (violation, scales.clone());
        }
        // `t_p ≥ 1` everywhere off-dominant: no positive scaling certifies, so hand over the best seen.
        if violation <= policy.tolerance || t_p >= 1.0 || iterations == policy.max_iterations {
            break;
        }
        iterations += 1;
        let towards_dominant = t_p * t_q <= 1.0;
        for (f, &ti) in scales.iter_mut().zip(&t) {
            let selected = match towards_dominant {
                true => ti < 1.0,
                false => ti > 1.0,
            };
            if selected && ti > 0.0 {
                *f *= ti;
            }
        }
        let peak = scales.iter().copied().fold(0.0f64, f64::max);
        if !(peak.is_finite() && peak > 0.0) {
            return None;
        }
        scales.iter_mut().for_each(|f| *f /= peak);
    }
    let (violation, scales) = best;
    let peak = (scales.iter().zip(&diagonal)).fold(0.0f64, |peak, (&f, &d)| peak.max(f * d.sqrt()));
    let scales: Box<[T]> = scales.iter().map(|&f| narrow::<T>(f / peak)).collect();
    if !scales.iter().all(|&f| f.is_finite() && f > T::zero()) {
        return None;
    }
    Some(Relaxation {
        scales,
        iterations,
        violation: narrow(violation),
    })
}

/// `t_i = Σ_j |a_ij| f_j / (a_ii f_i)`, each upper edge read for both its ends.
fn ratios<T: Real>(adjacency: &UpperAdjacency<T>, diagonal: &[f64], scales: &[f64], t: &mut [f64]) {
    t.fill(0.0);
    for i in 0..adjacency.n() {
        for (j, &a) in adjacency.upper(i) {
            let a = widen(a.abs());
            t[i] += a * scales[j];
            t[j] += a * scales[i];
        }
    }
    for ((ti, &d), &f) in t.iter_mut().zip(diagonal).zip(scales) {
        *ti /= d * f;
    }
}

/// CG on `α² − NᵀN` over the smaller side; `None` when no scale stays a positive `T`.
pub(super) fn reduced_cg<T: Real>(
    cross: &CrossBlock<T>,
    diagonal: &[T],
    policy: &ScalingPolicy,
) -> Option<Relaxation<T>> {
    let inv_sqrt: Vec<f64> = diagonal.iter().map(|&d| 1.0 / widen(d).sqrt()).collect();
    let operator = cg::NormalizedCrossOperator::new(cross, &inv_sqrt, 1.0 + 0.5 * policy.tolerance);
    let initial = Relaxation {
        scales: vec![1.0; diagonal.len()].into(),
        iterations: 0,
        violation: operator.unit_violation(),
    };
    let relaxation = cg::reduced_cg_scaling(&operator, initial, policy)?;
    let scales: Box<[T]> = (relaxation.scales.iter().zip(&inv_sqrt))
        .map(|(&mu, &normalizer)| narrow::<T>(mu * normalizer))
        .collect();
    if !scales.iter().all(|&f| f.is_finite() && f > T::zero()) {
        return None;
    }
    Some(Relaxation {
        scales,
        iterations: relaxation.iterations,
        violation: narrow(relaxation.violation.max(0.0)),
    })
}
