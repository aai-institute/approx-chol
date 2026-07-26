//! Flat storage for the elimination sequence and its per-step row kernels.

use super::FactorError;
use crate::types::Real;

/// Zero-copy view of a single elimination step (row operation).
///
/// Borrows slices from the flat CSR storage in `EliminationSequence`.
/// Each step eliminates `vertex` by splitting its weight among neighbors
/// according to `elimination_fractions`.
pub struct EliminationStep<'a, T> {
    pub vertex: usize,
    /// Zero when the pivot diagonal was clamped to near-zero.
    pub inv_diag: T,
    pub neighbor_indices: &'a [u32],
    pub elimination_fractions: &'a [T],
}

impl<'a, T: num_traits::Float + Send + Sync + 'static> EliminationStep<'a, T> {
    #[inline(always)]
    fn debug_assert_in_bounds(&self, y_len: usize) {
        debug_assert!(
            self.vertex < y_len,
            "pivot vertex {} out of bounds for work buffer len {}",
            self.vertex,
            y_len
        );
        debug_assert_eq!(
            self.neighbor_indices.len(),
            self.elimination_fractions.len(),
            "neighbors/fractions length mismatch"
        );
        for &j in self.neighbor_indices {
            debug_assert!(
                (j as usize) < y_len,
                "neighbor index {} out of bounds for work buffer len {}",
                j,
                y_len
            );
        }
    }

    /// Forward elimination: scatter pivot weight to neighbors, then scale by D^{-1}.
    #[inline(always)]
    pub(crate) fn apply_forward(&self, y: &mut [T]) {
        self.debug_assert_in_bounds(y.len());
        let vertex = self.vertex;
        let inv_diag = self.inv_diag;
        let n = self.neighbor_indices.len();
        let zero = T::zero();
        let one = T::one();
        if n == 0 {
            if inv_diag != zero {
                y[vertex] = y[vertex] * inv_diag;
            }
            return;
        }

        let mut yi = y[vertex];

        for (&j, &f) in self.neighbor_indices[..n - 1]
            .iter()
            .zip(self.elimination_fractions.iter())
        {
            let j = j as usize;
            y[j] = y[j] + f * yi;
            yi = yi * (one - f);
        }

        let j_last = self.neighbor_indices[n - 1] as usize;
        y[j_last] = y[j_last] + yi;
        y[vertex] = if inv_diag != zero { yi * inv_diag } else { yi };
    }

    /// Backward substitution: gather neighbor contributions back to pivot.
    #[inline(always)]
    pub(crate) fn apply_backward(&self, y: &mut [T]) {
        self.debug_assert_in_bounds(y.len());
        let vertex = self.vertex;
        let n = self.neighbor_indices.len();
        let one = T::one();
        if n == 0 {
            return;
        }

        let j_last = self.neighbor_indices[n - 1] as usize;
        let mut yi = y[vertex] + y[j_last];

        for (&j, &f) in self.neighbor_indices[..n - 1]
            .iter()
            .zip(self.elimination_fractions.iter())
            .rev()
        {
            yi = (one - f) * yi + f * y[j as usize];
        }

        y[vertex] = yi;
    }
}

/// Header for one elimination step: which vertex, its reciprocal diagonal, and
/// where its neighbor range ends. The range *starts* at the previous header's
/// `end`, so there is no second array that could disagree about step count,
/// about where step 0 begins, or about which diagonal belongs to which vertex.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Step<T> {
    pub(crate) vertex: u32,
    pub(crate) end: u32,
    pub(crate) inv_diag: T,
}

/// Contiguous memory owner for a sequence of elimination steps.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Debug)]
pub struct EliminationSequence<T> {
    pub(crate) steps: Vec<Step<T>>,
    pub(crate) neighbor_indices: Vec<u32>,
    pub(crate) elimination_fractions: Vec<T>,
}

// Public read-only API (no internal trait bounds).
impl<T> EliminationSequence<T> {
    #[inline(always)]
    pub fn n_steps(&self) -> usize {
        self.steps.len()
    }

    #[inline(always)]
    pub fn step(&self, i: usize) -> EliminationStep<'_, T>
    where
        T: Copy,
    {
        let step = &self.steps[i];
        let start = if i == 0 {
            0
        } else {
            self.steps[i - 1].end as usize
        };
        let end = step.end as usize;
        EliminationStep {
            vertex: step.vertex as usize,
            inv_diag: step.inv_diag,
            neighbor_indices: &self.neighbor_indices[start..end],
            elimination_fractions: &self.elimination_fractions[start..end],
        }
    }

    /// Check every structural invariant the solve path relies on, against a
    /// factor dimension `n`. Runs in release builds (unlike the `debug_assert`
    /// on the solve path), so a deserialized (untrusted) factor is rejected
    /// before it can index storage out of bounds or silently return garbage.
    pub(crate) fn validate_for_dim(&self, n: usize) -> Result<(), FactorError> {
        if self.neighbor_indices.len() != self.elimination_fractions.len() {
            return Err(FactorError::NeighborFractionLengthMismatch {
                neighbor_len: self.neighbor_indices.len(),
                fraction_len: self.elimination_fractions.len(),
            });
        }

        // Threading `start` through the loop makes the ranges contiguous and
        // non-decreasing by construction; only `start <= end <= nnz` is left to check.
        let nnz = self.neighbor_indices.len();
        let mut start = 0usize;
        for (i, step) in self.steps.iter().enumerate() {
            let end = step.end as usize;
            if start > end || end > nnz {
                return Err(FactorError::NeighborRangeInvalid {
                    step: i,
                    start,
                    end,
                    nnz,
                });
            }

            if (step.vertex as usize) >= n {
                return Err(FactorError::VertexOutOfBounds {
                    step: i,
                    vertex: step.vertex,
                    n,
                });
            }
            for &j in &self.neighbor_indices[start..end] {
                if (j as usize) >= n {
                    return Err(FactorError::NeighborOutOfBounds {
                        step: i,
                        neighbor: j,
                        n,
                    });
                }
            }
            start = end;
        }
        if start != nnz {
            return Err(FactorError::TrailingNeighborStorage {
                covered: start,
                nnz,
            });
        }
        Ok(())
    }
}

// Internal construction methods (pub(crate) only, Real bound is internal).
#[allow(private_bounds)]
impl<T: Real> EliminationSequence<T> {
    pub(crate) fn with_capacity(n: usize, degree_sum: usize) -> Self {
        Self {
            steps: Vec::with_capacity(n),
            neighbor_indices: Vec::with_capacity(degree_sum),
            elimination_fractions: Vec::with_capacity(degree_sum),
        }
    }

    /// Close the current step at the running nonzero count. Offset overflow is
    /// unreachable for tractable inputs, so assert (in release too) rather than
    /// truncate silently.
    fn push_step(&mut self, vertex: usize, diagonal: T) {
        let nnz = self.neighbor_indices.len();
        assert!(
            nnz <= u32::MAX as usize,
            "factor nonzero count {nnz} exceeds u32 offset capacity"
        );
        self.steps.push(Step {
            vertex: vertex as u32,
            end: nnz as u32,
            inv_diag: if diagonal.abs() > T::near_zero() {
                T::one() / diagonal
            } else {
                T::zero()
            },
        });
    }

    /// Record an isolated vertex (no neighbors, clamped diagonal).
    pub(crate) fn record_isolated(&mut self, vertex: usize, diagonal: T) {
        self.push_step(vertex, diagonal);
    }

    /// Record one sampled column (diagonal value plus its neighbor/fraction pattern).
    pub(crate) fn record_column(
        &mut self,
        vertex: usize,
        diagonal: T,
        neighbors: &[u32],
        fractions: &[T],
    ) {
        self.neighbor_indices.extend_from_slice(neighbors);
        self.elimination_fractions.extend_from_slice(fractions);
        self.push_step(vertex, diagonal);
    }
}
