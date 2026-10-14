use crate::storage::{CrossBlock, Rows};
use crate::{narrow, widen, Real};

use super::{Relaxation, ScalingPolicy};

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&x, &y)| x * y).sum()
}

/// `N = D^{-1/2}|C|D^{-1/2}` between the kept (small) and eliminated (large) sides.
struct NormalizedCrossOperator<'a, T> {
    small_to_large: &'a Rows<T>,
    large_to_small: Rows<T>,
    small_inv_sqrt: &'a [f64],
    large_inv_sqrt: &'a [f64],
    /// `α > 1` keeps `α² − NᵀN` definite, so a singular component never lands on its Perron vector.
    shift: f64,
}

#[derive(Clone, Copy)]
enum CandidateKind {
    Strict,
    Boundary,
}

impl<'a, T: Real> NormalizedCrossOperator<'a, T> {
    /// `inv_sqrt` in local order, eliminated rows first.
    fn new(cross: &'a CrossBlock<T>, inv_sqrt: &'a [f64], shift: f64) -> Self {
        let (large_inv_sqrt, small_inv_sqrt) = inv_sqrt.split_at(cross.n_eliminated());
        Self {
            small_to_large: cross.c(),
            large_to_small: cross.c().transpose(),
            small_inv_sqrt,
            large_inv_sqrt,
            shift,
        }
    }

    fn small_len(&self) -> usize {
        self.small_inv_sqrt.len()
    }

    fn large_len(&self) -> usize {
        self.large_inv_sqrt.len()
    }

    fn apply_large(&self, input: &[f64], output: &mut [f64]) {
        multiply_normalized(
            self.small_to_large,
            self.large_inv_sqrt,
            self.small_inv_sqrt,
            input,
            output,
        );
    }

    fn apply_small(&self, input: &[f64], output: &mut [f64]) {
        multiply_normalized(
            &self.large_to_small,
            self.small_inv_sqrt,
            self.large_inv_sqrt,
            input,
            output,
        );
    }

    fn apply_reduced(&self, input: &[f64], large_work: &mut [f64], output: &mut [f64]) {
        // Eliminating the large side of `αI − N` gives `α² − NᵀN` on the cheaper side.
        let shift_squared = self.shift * self.shift;
        self.apply_large(input, large_work);
        self.apply_small(large_work, output);
        for (value, &input_value) in output.iter_mut().zip(input) {
            *value = shift_squared * input_value - *value;
        }
    }

    fn reduced_rhs(&self, large_work: &mut [f64]) -> Vec<f64> {
        large_work.fill(1.0);
        let mut rhs = vec![0.0; self.small_len()];
        self.apply_small(large_work, &mut rhs);
        for value in &mut rhs {
            *value += self.shift;
        }
        rhs
    }

    /// `max_i (N·1)_i − 1`, the violation of `μ = 1`.
    fn unit_violation(&self) -> f64 {
        let mut large = vec![0.0; self.large_len()];
        let mut small = vec![0.0; self.small_len()];
        self.apply_large(&vec![1.0; self.small_len()], &mut large);
        self.apply_small(&vec![1.0; self.large_len()], &mut small);
        (large.iter().chain(&small)).fold(0.0f64, |m, &target| m.max(target - 1.0))
    }

    fn candidate_violation(
        &self,
        small: &[f64],
        kind: CandidateKind,
        large: &mut [f64],
        small_image: &mut [f64],
    ) -> Option<f64> {
        if !small.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        self.apply_large(small, large);
        // A strict large row solves `αy − Nx = 1`, so its ratio is `α − 1/y`: within `α − 1` of dominant.
        let large_violation = match kind {
            CandidateKind::Strict => {
                for value in large.iter_mut() {
                    *value = (*value + 1.0) / self.shift;
                }
                large
                    .iter()
                    .map(|&value| self.shift - 1.0 / value - 1.0)
                    .fold(0.0f64, f64::max)
            }
            CandidateKind::Boundary => 0.0,
        };
        if !large.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        self.apply_small(large, small_image);
        Some(
            small_image
                .iter()
                .zip(small)
                .map(|(&target, &value)| target / value - 1.0)
                .fold(large_violation, f64::max),
        )
    }

    fn materialize_candidate(
        &self,
        small: &[f64],
        large: &[f64],
        violation: f64,
    ) -> Option<Relaxation<f64>> {
        let mut mu: Vec<f64> = large.iter().chain(small).copied().collect();
        let peak = mu.iter().copied().fold(0.0f64, f64::max);
        if !peak.is_finite() || peak <= 0.0 {
            return None;
        }
        for value in mu.iter_mut() {
            *value /= peak;
        }
        if !mu.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        Some(Relaxation {
            scales: mu.into(),
            iterations: 0,
            violation,
        })
    }
}

fn multiply_normalized<T: Real>(
    matrix: &Rows<T>,
    output_inv_sqrt: &[f64],
    input_inv_sqrt: &[f64],
    input: &[f64],
    output: &mut [f64],
) {
    debug_assert_eq!(matrix.n_rows(), output.len());
    debug_assert_eq!(matrix.n_cols(), input.len());
    for (i, value) in output.iter_mut().enumerate() {
        *value = matrix
            .row(i)
            .map(|(j, &entry)| {
                widen(entry.abs()) * output_inv_sqrt[i] * input_inv_sqrt[j] * input[j]
            })
            .sum();
    }
}

#[derive(Clone, Copy)]
enum ReducedCgStep {
    Advanced,
    ResidualExhausted,
    NonPositiveCurvature,
}

struct ReducedCg {
    solution: Vec<f64>,
    residual: Vec<f64>,
    direction: Vec<f64>,
    reduced_image: Vec<f64>,
    residual_norm_squared: f64,
}

impl ReducedCg {
    fn new(rhs: Vec<f64>) -> Option<Self> {
        let dimension = rhs.len();
        let residual_norm_squared = dot(&rhs, &rhs);
        if !residual_norm_squared.is_finite() {
            return None;
        }
        Some(Self {
            solution: vec![0.0; dimension],
            residual: rhs.clone(),
            direction: rhs,
            reduced_image: vec![0.0; dimension],
            residual_norm_squared,
        })
    }

    fn step<T: Real>(
        &mut self,
        operator: &NormalizedCrossOperator<'_, T>,
        large_work: &mut [f64],
    ) -> Option<ReducedCgStep> {
        operator.apply_reduced(&self.direction, large_work, &mut self.reduced_image);
        let direction_norm_squared = dot(&self.direction, &self.direction);
        let curvature = dot(&self.direction, &self.reduced_image);
        if !direction_norm_squared.is_finite() || !curvature.is_finite() {
            return None;
        }
        let slack = 64.0 * f64::EPSILON * direction_norm_squared;
        if curvature <= slack {
            return Some(ReducedCgStep::NonPositiveCurvature);
        }

        let alpha = self.residual_norm_squared / curvature;
        if !alpha.is_finite() {
            return None;
        }
        for ((value, residual), (&direction, &image)) in self
            .solution
            .iter_mut()
            .zip(&mut self.residual)
            .zip(self.direction.iter().zip(&self.reduced_image))
        {
            *value += alpha * direction;
            *residual -= alpha * image;
        }

        let next_residual_norm_squared = dot(&self.residual, &self.residual);
        if !next_residual_norm_squared.is_finite() {
            return None;
        }
        if next_residual_norm_squared == 0.0 {
            return Some(ReducedCgStep::ResidualExhausted);
        }
        let beta = next_residual_norm_squared / self.residual_norm_squared;
        for (direction, &residual) in self.direction.iter_mut().zip(&self.residual) {
            *direction = residual + beta * *direction;
        }
        self.residual_norm_squared = next_residual_norm_squared;
        Some(ReducedCgStep::Advanced)
    }
}

fn reduced_cg_scaling<T: Real>(
    operator: &NormalizedCrossOperator<'_, T>,
    initial: Relaxation<f64>,
    scaling: &ScalingPolicy,
) -> Option<Relaxation<f64>> {
    let mut large_work = vec![0.0; operator.large_len()];
    let rhs = operator.reduced_rhs(&mut large_work);
    let mut cg = ReducedCg::new(rhs)?;
    let mut best = initial;
    let mut small_image = vec![0.0; operator.small_len()];

    for iteration in 1..=scaling.max_iterations {
        let step = cg.step(operator, &mut large_work)?;
        if matches!(step, ReducedCgStep::NonPositiveCurvature) {
            // A singular boundary exposes its Perron vector as the zero-curvature direction.
            let boundary_small: Vec<f64> = cg.direction.iter().map(|value| value.abs()).collect();
            if let Some(violation) = operator.candidate_violation(
                &boundary_small,
                CandidateKind::Boundary,
                &mut large_work,
                &mut small_image,
            ) {
                if violation < best.violation {
                    if let Some(candidate) =
                        operator.materialize_candidate(&boundary_small, &large_work, violation)
                    {
                        best = candidate;
                    }
                }
            }
            return Some(Relaxation {
                iterations: iteration,
                ..best
            });
        }

        // A certificate check costs two extra passes; past the cheap early exits, probe on a stride.
        let last =
            matches!(step, ReducedCgStep::ResidualExhausted) || iteration == scaling.max_iterations;
        if last || iteration <= 8 || iteration % 4 == 0 {
            if let Some(violation) = operator.candidate_violation(
                &cg.solution,
                CandidateKind::Strict,
                &mut large_work,
                &mut small_image,
            ) {
                if violation < best.violation {
                    if let Some(candidate) =
                        operator.materialize_candidate(&cg.solution, &large_work, violation)
                    {
                        best = candidate;
                    }
                }
                if best.violation <= scaling.tolerance {
                    return Some(Relaxation {
                        iterations: iteration,
                        ..best
                    });
                }
            }
        }

        if matches!(step, ReducedCgStep::ResidualExhausted) {
            return Some(Relaxation {
                iterations: iteration,
                ..best
            });
        }
    }
    Some(Relaxation {
        iterations: scaling.max_iterations,
        ..best
    })
}

/// CG on `α² − NᵀN` over the smaller side; `None` when no scale stays a positive `T`.
pub(in crate::sgdd) fn reduced_cg<T: Real>(
    cross: &CrossBlock<T>,
    diagonal: &[T],
    policy: &ScalingPolicy,
) -> Option<Relaxation<T>> {
    let inv_sqrt: Vec<f64> = diagonal.iter().map(|&d| 1.0 / widen(d).sqrt()).collect();
    let operator = NormalizedCrossOperator::new(cross, &inv_sqrt, 1.0 + 0.5 * policy.tolerance);
    let initial = Relaxation {
        scales: vec![1.0; diagonal.len()].into(),
        iterations: 0,
        violation: operator.unit_violation(),
    };
    let relaxation = reduced_cg_scaling(&operator, initial, policy)?;
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
