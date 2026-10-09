//! Probe: Self-Corrective Iterations (Guan, Lu, Li, Shao 2016, Algorithms 3 and 3b) as a dominance scaling.

use super::scaling::DominanceScaling;
use super::NotScalable;
use crate::sgdd::cross_tab::CrossTab;
use crate::sgdd::ScalingPolicy;

/// Which columns a step rescales.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Selection {
    /// Algorithm 3: every row on the suppressed side of 1.
    All,
    /// Algorithm 3b: only rows past the reciprocal of the opposite extreme.
    Balanced,
}

/// `t_i = Σ_j |a_ij| f_j / (a_ii f_i)` for every level, in place.
fn ratios(cross: &CrossTab, diagonal: &[f64], scales: &[f64], t: &mut [f64]) {
    for (i, ti) in t.iter_mut().enumerate() {
        let off: f64 = cross.neighbors(i).map(|(j, v)| v.abs() * scales[j]).sum();
        *ti = off / (diagonal[i] * scales[i]);
    }
}

pub(crate) fn sci_scaling(
    cross: &CrossTab,
    diagonal: &[f64],
    policy: &ScalingPolicy,
    selection: Selection,
) -> Result<DominanceScaling, NotScalable> {
    let n = cross.n_local();
    let mut scales = vec![1.0f64; n];
    let mut t = vec![0.0f64; n];
    let mut best = (f64::INFINITY, scales.clone());
    let mut iterations = 0;
    loop {
        ratios(cross, diagonal, &scales, &mut t);
        let (t_p, t_q) = t.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
        if !(t_p.is_finite() && t_q.is_finite()) {
            return Err(NotScalable);
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
            let selected = match (selection, towards_dominant) {
                (Selection::All, true) => ti < 1.0,
                (Selection::All, false) => ti > 1.0,
                (Selection::Balanced, true) => ti * t_q <= 1.0,
                (Selection::Balanced, false) => t_p * ti > 1.0,
            };
            if selected && ti > 0.0 {
                *f *= ti;
            }
        }
        let peak = scales.iter().copied().fold(0.0f64, f64::max);
        if !(peak.is_finite() && peak > 0.0) {
            return Err(NotScalable);
        }
        scales.iter_mut().for_each(|f| *f /= peak);
    }
    let (violation, mut scales) = best;
    let peak = scales.iter().copied().fold(0.0f64, f64::max);
    scales.iter_mut().for_each(|f| *f /= peak);
    if !scales.iter().all(|f| f.is_finite() && *f > 0.0) {
        return Err(NotScalable);
    }
    Ok(DominanceScaling {
        scales,
        iterations,
        violation,
    })
}
