//! Flat storage for the elimination sequence and its per-step row kernels.

#[cfg(feature = "serde")]
use super::FactorError;
use crate::types::Real;

/// Zero-copy view of a single elimination step (row operation).
///
/// Borrows slices from the flat CSR storage in `EliminationSequence`.
/// Each step eliminates `vertex` by splitting its weight among neighbors
/// according to `elimination_fractions`.
pub struct EliminationStep<'a, T> {
    /// Index of the eliminated vertex.
    pub vertex: usize,
    /// Indices of neighbors that receive fill weight.
    pub neighbor_indices: &'a [u32],
    /// Fraction of remaining weight distributed to each neighbor.
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
    pub(crate) fn apply_forward(&self, y: &mut [T], inv_diag: T) {
        self.debug_assert_in_bounds(y.len());
        let vertex = self.vertex;
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
        let val = if inv_diag != zero { yi * inv_diag } else { yi };
        y[vertex] = val;
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
    pub(crate) vertices: Vec<u32>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) neighbor_indices: Vec<u32>,
    pub(crate) elimination_fractions: Vec<T>,
    pub(crate) inv_diagonal: Vec<T>,
}

// Public read-only API (no internal trait bounds).
impl<T> EliminationSequence<T> {
    /// Number of elimination steps recorded.
    #[inline(always)]
    pub fn n_steps(&self) -> usize {
        self.vertices.len()
    }

    /// Borrow step `i` as a zero-copy view.
    #[inline(always)]
    pub fn step(&self, i: usize) -> EliminationStep<'_, T> {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        EliminationStep {
            vertex: self.vertices[i] as usize,
            neighbor_indices: &self.neighbor_indices[start..end],
            elimination_fractions: &self.elimination_fractions[start..end],
        }
    }

    /// Check every structural invariant the solve path relies on, against a
    /// factor dimension `n`. Runs in release builds (unlike the `debug_assert`
    /// on the solve path), so a deserialized (untrusted) factor is rejected
    /// before it can index storage out of bounds or silently return garbage.
    #[cfg(feature = "serde")]
    pub(crate) fn validate_for_dim(&self, n: usize) -> Result<(), FactorError> {
        let n_steps = self.vertices.len();
        if self.offsets.len() != n_steps + 1 {
            return Err(FactorError::OffsetsLengthMismatch {
                expected: n_steps + 1,
                got: self.offsets.len(),
            });
        }
        if self.inv_diagonal.len() != n_steps {
            return Err(FactorError::InvDiagonalLengthMismatch {
                expected: n_steps,
                got: self.inv_diagonal.len(),
            });
        }
        if self.neighbor_indices.len() != self.elimination_fractions.len() {
            return Err(FactorError::NeighborFractionLengthMismatch {
                neighbor_len: self.neighbor_indices.len(),
                fraction_len: self.elimination_fractions.len(),
            });
        }

        // offsets.len() == n_steps + 1 >= 1, so offsets[0] and offsets[n_steps] exist.
        if self.offsets[0] != 0 {
            return Err(FactorError::OffsetsMustStartAtZero {
                got: self.offsets[0],
            });
        }

        // Per-step `start <= end` (with the shared boundary offsets[i+1] serving
        // as both step i's end and step i+1's start) already forces the whole
        // offsets array to be non-decreasing, so no separate monotonicity check
        // is needed.
        let nnz = self.neighbor_indices.len();
        for (i, &vertex) in self.vertices.iter().enumerate() {
            let start = self.offsets[i] as usize;
            let end = self.offsets[i + 1] as usize;
            if start > end || end > nnz {
                return Err(FactorError::OffsetRangeInvalid {
                    step: i,
                    start,
                    end,
                    nnz,
                });
            }

            if (vertex as usize) >= n {
                return Err(FactorError::VertexOutOfBounds { step: i, vertex, n });
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
        }
        let last = self.offsets[n_steps] as usize;
        if last != nnz {
            return Err(FactorError::FinalOffsetMismatch { last, nnz });
        }
        Ok(())
    }
}

// Internal construction methods (pub(crate) only, Real bound is internal).
#[allow(private_bounds)]
impl<T: Real> EliminationSequence<T> {
    /// Pre-allocate for `n` elimination steps with an estimated total neighbor count.
    pub(crate) fn with_capacity(n: usize, degree_sum: usize) -> Self {
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0u32);
        debug_assert_eq!(offsets.len(), 1);
        Self {
            vertices: Vec::with_capacity(n),
            offsets,
            neighbor_indices: Vec::with_capacity(degree_sum),
            elimination_fractions: Vec::with_capacity(degree_sum),
            inv_diagonal: Vec::with_capacity(n),
        }
    }

    /// Push the running nonzero count as the next `u32` offset. Overflow is
    /// unreachable for tractable inputs, so assert (in release too) rather than
    /// truncate silently.
    fn push_offset(&mut self) {
        let nnz = self.neighbor_indices.len();
        assert!(
            nnz <= u32::MAX as usize,
            "factor nonzero count {nnz} exceeds u32 offset capacity"
        );
        self.offsets.push(nnz as u32);
    }

    /// Record an isolated vertex (no neighbors, clamped diagonal).
    pub(crate) fn record_isolated(&mut self, vertex: usize, diagonal: T) {
        self.vertices.push(vertex as u32);
        self.inv_diagonal.push(if diagonal.abs() > T::near_zero() {
            T::one() / diagonal
        } else {
            T::zero()
        });
        self.push_offset();
        debug_assert_eq!(self.offsets.len(), self.vertices.len() + 1);
    }

    /// Record one sampled column (diagonal value plus its neighbor/fraction pattern).
    pub(crate) fn record_column(
        &mut self,
        vertex: usize,
        diagonal: T,
        neighbors: &[u32],
        fractions: &[T],
    ) {
        self.vertices.push(vertex as u32);
        self.inv_diagonal.push(if diagonal.abs() > T::near_zero() {
            T::one() / diagonal
        } else {
            T::zero()
        });
        self.neighbor_indices.extend_from_slice(neighbors);
        self.elimination_fractions.extend_from_slice(fractions);
        self.push_offset();
        debug_assert_eq!(self.offsets.len(), self.vertices.len() + 1);
    }
}
