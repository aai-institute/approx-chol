mod normalized;
mod perron;

use normalized::Normalized;
use perron::InverseIteration;

use crate::storage::OffDiagonal;
use crate::{narrow, widen, Index, Real};

/// Certification policy for a dominance scaling; a deficit past rounding left at hand-over is clamped and reported.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingPolicy {
    #[cfg_attr(feature = "serde", serde(deserialize_with = "tolerance"))]
    tolerance: f64,
    max_iterations: usize,
}

fn admits(tolerance: f64) -> bool {
    (0.0..1.0).contains(&tolerance)
}

#[cfg(feature = "serde")]
fn tolerance<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let tolerance = <f64 as serde::Deserialize>::deserialize(deserializer)?;
    (admits(tolerance).then_some(tolerance))
        .ok_or_else(|| serde::de::Error::custom("tolerance outside [0, 1)"))
}

impl ScalingPolicy {
    /// `None` unless `0 ≤ tolerance < 1`.
    pub fn new(tolerance: f64, max_iterations: usize) -> Option<Self> {
        admits(tolerance).then_some(Self {
            tolerance,
            max_iterations,
        })
    }

    /// The relative dominance violation a row may show and still count as dominant; never below rounding.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// The scaling iterations a block may spend before it is reported unsettled.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
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

#[cfg(feature = "serde")]
fn violation<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let violation = <f64 as serde::Deserialize>::deserialize(deserializer)?;
    (violation.is_finite() && violation > 0.0)
        .then_some(violation)
        .ok_or_else(|| serde::de::Error::custom("clamp violation not finite and positive"))
}

/// How the reduction clamped a component's rows past rounding to make it dominant.
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Clamp {
    /// Largest relative dominance violation, clamped away: finite and positive.
    #[cfg_attr(feature = "serde", serde(deserialize_with = "violation"))]
    pub violation: f64,
    /// What the scaling proved about the violation.
    pub scaling: ScalingOutcome,
}

/// What a component's scaling proved: whether a certificate of generalized diagonal dominance exists.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalingOutcome {
    /// Proven dominant, by rows within the policy's tolerance, a null vector closing within `√ε`, or a Perron vector within the tolerance; the clamp removed that proof's rounding.
    Certified,
    /// Proven not dominant past the tolerance.
    Refuted,
    /// Neither proven: the iteration stopped first.
    Unsettled,
}

impl ScalingPolicy {
    /// `Σ_j |a_ij| ≤ a_ii (1 + max(tolerance, RATIO_BAND ε))`: a rounded Laplacian stays bit for bit.
    pub(super) fn dominates<T: Real>(&self, d: T, abs_row_sum: T) -> bool {
        widen(abs_row_sum) <= widen(d) * (1.0 + self.slack(widen(T::epsilon())))
    }

    /// The tolerance, never below what `T`'s rounding of a dominant row leaves.
    fn slack(&self, epsilon: f64) -> f64 {
        self.tolerance.max(RATIO_BAND * epsilon)
    }
}

/// Scales `R = μ/√a_ii`, positive once [`positive`] admits them, making `R|A|R` weakly dominant (Daitch, Spielman, arXiv:0803.0988 §4), and what `μ` is.
pub(super) type Relaxation<T> = (Box<[T]>, ScaleVector);

/// The vector the scales are taken from, and so how far past rounding each row of `M` may sit.
#[derive(Clone, Copy, Debug)]
pub(super) enum ScaleVector {
    /// `R = I`: a row rounds by its `k_i` entries, never tighter than a settled residual.
    Identity,
    /// `|u|` of a certified null vector, accurate relative to each entry.
    NullVector,
    /// An inverse-iteration Perron vector settled to `band` relative to its peak: a row of `M_ii = μ_i²` sits within `band/μ_i`.
    // Spreads a ground as `1 − ρ` over every row, so a block within the band floats (#308).
    PerronVector {
        band: f64,
        /// How far the best `y` leaves `M` short of dominance.
        violation: f64,
        scaling: ScalingOutcome,
        /// Settled `y` proves the float decision; else a singular block may have been grounded.
        settled: bool,
    },
}

impl ScaleVector {
    /// Each row's `excess_i / m_ii` past its band, `0` within it; `None` while every row is within the `RATIO_BAND ε` floor, which costs one pass; `degrees`, each `k_i ≤ max_k`, is read once and only for a row the bands at `0` and `max_k` leave open.
    pub(super) fn past_band(
        self,
        epsilon: f64,
        excess: impl Iterator<Item = f64> + Clone,
        diagonal: impl Iterator<Item = f64> + Clone,
        (max_k, degrees): (usize, impl FnOnce() -> Vec<usize>),
    ) -> Option<impl Iterator<Item = f64>> {
        let ratios = (excess.zip(diagonal)).map(|(e, d)| (e / d, d));
        if (ratios.clone()).all(|(ratio, _)| ratio <= RATIO_BAND * epsilon) {
            return None;
        }
        let (mut degrees, mut read) = (Some(degrees), Vec::new());
        Some(ratios.enumerate().map(move |(i, (ratio, d))| {
            let past = |k| ratio > self.row_band(k, d, epsilon);
            let past = past(0)
                && (past(max_k) || {
                    if let Some(degrees) = degrees.take() {
                        read = degrees();
                    }
                    past(read[i])
                });
            match past {
                true => ratio,
                false => 0.0,
            }
        }))
    }

    /// Row `i`'s band for its `k_i` entries and diagonal `m_ii`, in units where `T`'s rounding is `epsilon`.
    fn row_band(self, k: usize, m_ii: f64, epsilon: f64) -> f64 {
        let rounding = (k + 3) as f64 * epsilon;
        match self {
            Self::Identity => rounding.max(RATIO_BAND * epsilon),
            Self::NullVector => rounding + RATIO_BAND * epsilon,
            Self::PerronVector { band, .. } => rounding + band / m_ii.sqrt(),
        }
    }

    /// The clamp of a component whose worst `deficit` is past its band, `None` at `0`.
    pub(super) fn clamp(self, deficit: f64) -> Option<Clamp> {
        let (violation, scaling) = match self {
            Self::Identity | Self::NullVector => (0.0, ScalingOutcome::Certified),
            Self::PerronVector {
                violation, scaling, ..
            } => (violation, scaling),
        };
        (deficit > 0.0).then(|| Clamp {
            violation: violation.max(deficit),
            scaling,
        })
    }

    /// Whether the float decision rests on an unsettled Perron vector.
    pub(super) fn grounding_unproven(self) -> bool {
        matches!(self, Self::PerronVector { settled, .. } if !settled)
    }
}

/// `None` unless every scale is a positive `T`.
fn positive<T: Real>(scales: Box<[T]>, vector: ScaleVector) -> Option<Relaxation<T>> {
    let admitted = scales.iter().all(|&f| f.is_finite() && f > T::zero());
    admitted.then_some((scales, vector))
}

/// `|u|` of a certified null vector: no iteration runs; `None` if a scale leaves `T`.
pub(super) fn null_scaling<T: Real>(
    u: impl Iterator<Item = f64> + Clone,
    diagonal: &[T],
) -> Option<Relaxation<T>> {
    let (low, high) =
        (u.clone().zip(diagonal)).fold((f64::INFINITY, 0.0f64), |(low, high), (x, &d)| {
            let mu = x.abs() * widen(d).sqrt();
            (low.min(mu), high.max(mu))
        });
    // Centred by a power of two, not peaked at 1: `μ²` spans `(high/low)²`, past `f32`'s range from `2^52` on.
    let peak = high * 2f64.powi(-((high / low).log2() / 2.0).floor() as i32);
    let scales: Box<[T]> = u.map(|x| narrow::<T>(x.abs() / peak)).collect();
    positive(scales, ScaleVector::NullVector)
}

/// How far, in `ε`, a row ratio `Σ_j|a_ij|f_j / (a_ii f_i)` may sit from `1`: a settled residual and the dominance slack's floor.
pub(super) const RATIO_BAND: f64 = 16.0;

/// Scales from `N`'s Perron vector, iterated until `ρ < 1` is proven, pinned at `1`, or refuted past the tolerance.
pub(super) fn relax<T: Real, I: Index>(
    off_diagonal: &OffDiagonal<T, I>,
    diagonal: &[T],
    policy: &ScalingPolicy,
) -> Option<Relaxation<T>> {
    let n = Normalized::new(off_diagonal, diagonal);
    // A ratio over `k` positive terms rounds by `(k + 3)ε`: no bracket closes tighter.
    let rounding =
        (off_diagonal.degrees().into_iter().max().unwrap_or(0) + 3) as f64 * f64::EPSILON;
    let band = (RATIO_BAND * widen(T::epsilon())).max(rounding);
    let slack = policy.slack(widen(T::epsilon()));
    let iteration = InverseIteration::new(&n, slack, band, policy.max_iterations)?;
    let (y, vector) = iteration.run();
    positive(n.scales(&y), vector)
}
