use super::{OffDiagonal, Unreduced};
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

/// A part's uncertified scaling: present iff its violation exceeded the policy's tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalingReport {
    /// Scaling iterations spent.
    pub iterations: usize,
    /// Largest relative dominance violation, clamped away.
    pub violation: f64,
}

/// Positive scales `R` making `R|A|R` weakly dominant, with what the policy says of their violation.
pub(super) struct Scaling<T> {
    /// `None` when `A` already is: every SDDM input, so an SDDM CSR reduces bit for bit as before.
    pub(super) scales: Option<Box<[T]>>,
    pub(super) report: Option<ScalingReport>,
}

/// Positive scales with `R|A|R` weakly dominant up to `violation`.
struct Certificate<T> {
    scales: Box<[T]>,
    iterations: usize,
    violation: T,
}

/// Why no scaling exists.
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
) -> Result<Scaling<T>, NotScalable> {
    if already_dominant(a) {
        return Ok(Scaling {
            scales: None,
            report: None,
        });
    }
    let certificate = match &a.off_diagonal {
        OffDiagonal::General(_) => sci(a, policy),
        OffDiagonal::Bipartite(_) => reduced_cg(a, policy),
    }?;
    Ok(policy.judge(certificate))
}

impl ScalingPolicy {
    /// The one place the tolerance decides whether a certificate is reported.
    fn judge<T: Real>(&self, certificate: Certificate<T>) -> Scaling<T> {
        todo!()
    }
}

fn already_dominant<T: Real>(a: &Unreduced<T>) -> bool {
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
