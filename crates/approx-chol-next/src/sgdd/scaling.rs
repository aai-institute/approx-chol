use super::{Edges, Unreduced};
use crate::Real;

/// Certification policy for a dominance scaling; a deficit left at hand-over is clamped and reported.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingPolicy {
    tolerance: f64,
    max_iterations: usize,
}

impl ScalingPolicy {
    /// `None` unless `tolerance` is finite and non-negative.
    pub fn new(tolerance: f64, max_iterations: usize) -> Option<Self> {
        (tolerance.is_finite() && tolerance >= 0.0).then_some(Self {
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

/// The worst uncertified scaling across parts; present iff some violation exceeded the tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingReport {
    /// Scaling iterations spent on the worst part.
    pub iterations: usize,
    /// Largest relative dominance violation, clamped away.
    pub violation: f64,
}

/// Positive scales `r` with `R|A|R` weakly dominant up to `violation`.
pub(super) struct Certificate<T> {
    scales: Vec<T>,
    iterations: usize,
    violation: T,
}

/// Why no certificate exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NotScalable {
    /// `M(A)` is not PSD: no positive diagonal reaches dominance.
    NotGdd,
    /// A scale or a scaled entry left the finite range.
    Overflow,
}

/// SCI reads only row ratios, so it serves any pattern; the reduced CG needs the bipartite operator.
pub(super) fn dominance_scaling<T: Real>(
    a: &Unreduced<T>,
    policy: &ScalingPolicy,
) -> Result<Certificate<T>, NotScalable> {
    if let Some(identity) = already_dominant(a, policy) {
        return Ok(identity);
    }
    match &a.edges {
        Edges::General(_) => sci(a, policy),
        Edges::Bipartite(_) => reduced_cg(a, policy),
    }
}

/// Zero iterations: every SDDM input, so an SDDM CSR reduces bit for bit as before.
fn already_dominant<T: Real>(a: &Unreduced<T>, policy: &ScalingPolicy) -> Option<Certificate<T>> {
    todo!()
}

/// Self-corrective iterations (Guan, Lu, Li, Shao 2016).
fn sci<T: Real>(a: &Unreduced<T>, policy: &ScalingPolicy) -> Result<Certificate<T>, NotScalable> {
    todo!()
}

/// CG on `α² − NᵀN` over the smaller side, `α = 1 + tolerance/2` off the Perron vector.
fn reduced_cg<T: Real>(
    a: &Unreduced<T>,
    policy: &ScalingPolicy,
) -> Result<Certificate<T>, NotScalable> {
    todo!()
}
