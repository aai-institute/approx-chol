//! Bipartite block-elimination local solver, one per component.
//!
//! [`BlockElimSolver`] handles a 2x2 bipartite Gramian `[D_q, C; C^T, D_r]` by
//! eliminating the larger diagonal block, then solving the Schur complement
//! via approximate or dense Cholesky (see [`schur`] and [`factor`]).

pub(crate) mod csr_matrix;
pub(crate) mod factor;
pub(crate) mod schur;
pub(crate) mod solver;

pub(crate) use solver::BlockElimSolver;

use crate::Config;

use super::SchurMode;

/// within's local-solver knobs: the kernel's config, the Schur mode and the dense threshold.
pub(crate) struct LocalSolverConfig {
    pub(crate) approx_chol: Config,
    pub(crate) schur: SchurMode,
    pub(crate) dense_threshold: usize,
}

impl LocalSolverConfig {
    pub(crate) fn to_approx_chol(&self, on_failure: crate::ExactFailure) -> Config {
        Config {
            // `max_dim: 0` claims no block, which is what `dense_threshold: 0` asks for.
            backend: crate::Backend::ExactBelow {
                max_dim: self.dense_threshold,
                on_failure,
            },
            ..self.approx_chol
        }
    }
}

/// within's sampled-Schur knobs.
pub(crate) struct ApproxSchurConfig {
    pub(crate) seed: u64,
    pub(crate) split: u32,
}

/// Neumaier compensated sum: a flat `iter().sum()` biases the mean for large `n`.
#[inline]
pub(crate) fn compensated_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for &value in values {
        let next = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - next) + value;
        } else {
            compensation += (value - next) + sum;
        }
        sum = next;
    }
    sum + compensation
}
