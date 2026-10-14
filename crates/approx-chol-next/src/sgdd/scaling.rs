mod normalized;
mod perron;

use normalized::Normalized;
use perron::InverseIteration;

use crate::storage::OffDiagonal;
use crate::{narrow, widen, Index, Real};

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

/// A block's uncertified scaling: present iff its violation exceeded the policy's tolerance.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Existence {
    /// A factor-width-2 Gram, GDD by type: only the policy's budget ran out.
    Guaranteed,
    /// Rows: the input itself may not be GDD.
    Unknown,
}

impl ScalingPolicy {
    /// `Σ_j |a_ij| ≤ a_ii (1 + max(tolerance, RATIO_BAND ε))`: a rounded Laplacian stays bit for bit.
    pub(super) fn dominates<T: Real>(&self, d: T, abs_row_sum: T) -> bool {
        let slack = (self.tolerance).max(RATIO_BAND * widen(T::epsilon()));
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

/// Positive scales `R` making `R|A|R` weakly dominant, reducing GDD to SDDM (Daitch, Spielman, arXiv:0803.0988 §4).
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

impl<T: Real> Relaxation<T> {
    /// `None` unless every scale is a positive `T`.
    fn new(scales: Box<[T]>, iterations: usize, violation: T) -> Option<Self> {
        let positive = scales.iter().all(|&f| f.is_finite() && f > T::zero());
        positive.then_some(Self {
            scales,
            iterations,
            violation,
        })
    }
}

/// `|u|` of a certified null vector: no iteration runs; `None` if a scale leaves `T`.
pub(super) fn null_scaling<T: Real>(u: &[f64], diagonal: &[T]) -> Option<Relaxation<T>> {
    let peak =
        (u.iter().zip(diagonal)).fold(0.0f64, |peak, (&x, &d)| peak.max(x.abs() * widen(d).sqrt()));
    let scales: Box<[T]> = u.iter().map(|&x| narrow::<T>(x.abs() / peak)).collect();
    Relaxation::new(scales, 0, T::zero())
}

/// How far, in `ε`, a row ratio `Σ_j|a_ij|f_j / (a_ii f_i)` may sit from `1`: a settled residual and the dominance slack's floor.
const RATIO_BAND: f64 = 16.0;

/// Scales from `N`'s Perron vector, iterated until `ρ < 1` is proven, pinned at `1`, or refuted past the tolerance.
pub(super) fn relax<T: Real, I: Index>(
    off_diagonal: &OffDiagonal<T, I>,
    diagonal: &[T],
    policy: &ScalingPolicy,
) -> Option<Relaxation<T>> {
    let n = Normalized::new(off_diagonal, diagonal);
    // A ratio over `k` positive terms rounds by `(k + 3)ε`: no bracket closes tighter.
    let rounding = (off_diagonal.max_degree() + 3) as f64 * f64::EPSILON;
    let band = (RATIO_BAND * widen(T::epsilon())).max(rounding);
    let iteration = InverseIteration::new(&n, policy.tolerance, band, policy.max_iterations)?;
    let (y, iterations, violation) = iteration.run();
    Relaxation::new(n.scales(&y), iterations, narrow(violation))
}
