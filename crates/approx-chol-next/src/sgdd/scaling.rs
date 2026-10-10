mod cg;

use super::{OffDiagonal, Unreduced};
use crate::storage::{diagonal_in_range, narrow, widen, CrossBlock};
use crate::{Real, ReduceError};

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
}

/// Positive scales `R` making `R|A|R` weakly dominant.
pub(super) enum Scaling<T> {
    /// Every SDDM input, so an SDDM CSR reduces bit for bit as before.
    AlreadyDominant,
    /// Uncertified until its hand-over clamps: the report weighs both.
    Scaled(Relaxation<T>),
}

/// What an iteration reached: scales and how far short of dominance they leave `A`.
pub(super) struct Relaxation<T> {
    /// `f = μ/√a_ii` with `μ` peaking at `1`, so `M`'s diagonal `μ²` never exceeds `1`.
    pub(super) scales: Box<[T]>,
    iterations: usize,
    violation: T,
}

/// Every input's diagonal is screened here, before any dominance test; a view's zero diagonal was already `OutOfRange`.
pub(super) fn dominance_scaling<T: Real>(
    a: &Unreduced<T>,
    policy: &ScalingPolicy,
) -> Result<Scaling<T>, ReduceError> {
    if let Some(i) = a.diagonal.iter().position(|&d| !diagonal_in_range(d)) {
        return Err(ReduceError::OutOfRange {
            vertex: a.vertices[i],
        });
    }
    if let Some(i) = a.abs_row_sums.iter().position(|&s| !diagonal_in_range(s)) {
        return Err(ReduceError::OutOfRange {
            vertex: a.vertices[i],
        });
    }
    if a.diagonal.iter().any(|&d| d <= T::zero()) {
        return Err(ReduceError::NotGdd {
            vertex: a.vertices[0],
        });
    }
    if already_dominant(a, policy) {
        return Ok(Scaling::AlreadyDominant);
    }
    // SCI reads only row ratios, so it serves any pattern; the reduced CG needs the bipartite operator.
    let relaxation = match &a.off_diagonal {
        OffDiagonal::General(_) => sci(a, policy),
        OffDiagonal::Bipartite(cross) => {
            reduced_cg(cross, &a.diagonal, policy).ok_or(ReduceError::Unscalable {
                vertex: a.vertices[0],
            })
        }
    }?;
    Ok(Scaling::Scaled(relaxation))
}

impl ScalingPolicy {
    /// The one place the tolerance decides whether a scaling is reported: its own violation or the deficit its hand-over clamped, whichever is worse.
    pub(super) fn certify<T: Real>(
        &self,
        relaxation: &Relaxation<T>,
        clamped: f64,
    ) -> Option<ScalingReport> {
        let violation = widen(relaxation.violation).max(clamped);
        (violation > self.tolerance).then_some(ScalingReport {
            iterations: relaxation.iterations,
            violation,
        })
    }
}

/// `max_i (Σ_j|m_ij| − m_ii)/m_ii`, at least `0`: what clamping `M`'s deficits to zero surplus adds.
pub(super) fn clamped_deficit<T: Real>(diagonal: &[T], abs_row_sums: &[T]) -> f64 {
    (diagonal.iter().zip(abs_row_sums))
        .map(|(&d, &s)| (widen(s) - widen(d)) / widen(d))
        .fold(0.0, f64::max)
}

/// Unscaled, a surplus inside its band `t_i ε a_ii + t_i ε Σ_j|a_ij|`, `t_i` the additions that formed it, or a certified deficit counts as zero, and the component floats when none remains; a scaled surplus is its scaling's residue, which no row bound tells from a real one.
pub(super) fn rows_prove_singular<T: Real>(a: &Unreduced<T>, scaling: &Scaling<T>) -> bool {
    match scaling {
        Scaling::AlreadyDominant => todo!(),
        Scaling::Scaled { .. } => false,
    }
}

/// `Σ_j |a_ij| ≤ a_ii (1 + max(tolerance, 2 deg_i ε))`: a rounded Laplacian stays bit for bit, even in `f32`.
fn already_dominant<T: Real>(a: &Unreduced<T>, policy: &ScalingPolicy) -> bool {
    let epsilon = widen(T::epsilon());
    let rows = a.diagonal.iter().zip(&a.abs_row_sums);
    rows.zip(a.off_diagonal.degrees())
        .all(|((&d, &s), degree)| {
            let slack = policy.tolerance.max(2.0 * degree as f64 * epsilon);
            widen(s) <= widen(d) * (1.0 + slack)
        })
}

/// Self-corrective iterations (Guan, Lu, Li, Shao 2016).
fn sci<T: Real>(a: &Unreduced<T>, policy: &ScalingPolicy) -> Result<Relaxation<T>, ReduceError> {
    todo!()
}

/// CG on `α² − NᵀN` over the smaller side, `α = 1 + tolerance/2` off the Perron vector; `None` when no scale stays a positive `T`.
fn reduced_cg<T: Real>(
    cross: &CrossBlock<T>,
    diagonal: &[T],
    policy: &ScalingPolicy,
) -> Option<Relaxation<T>> {
    let inv_sqrt: Vec<f64> = diagonal.iter().map(|&d| 1.0 / widen(d).sqrt()).collect();
    let operator = cg::NormalizedCrossOperator::new(cross, &inv_sqrt, 1.0 + 0.5 * policy.tolerance);
    let initial = cg::ScalingCandidate {
        mu: vec![1.0; diagonal.len()],
        violation: operator.unit_violation(),
    };
    let result = cg::reduced_cg_scaling(&operator, initial, policy).ok()?;
    let scales: Box<[T]> = (result.candidate.mu.iter().zip(&inv_sqrt))
        .map(|(&mu, &normalizer)| narrow::<T>(mu * normalizer))
        .collect();
    if !scales.iter().all(|&f| f.is_finite() && f > T::zero()) {
        return None;
    }
    Some(Relaxation {
        scales,
        iterations: result.iterations,
        violation: narrow(result.candidate.violation.max(0.0)),
    })
}
