use crate::types::Real;
use core::fmt;

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

    #[inline]
    fn debug_assert_valid_for_dim(&self, n: usize) {
        debug_assert_eq!(
            self.offsets.len(),
            self.vertices.len() + 1,
            "offsets length must be n_steps + 1"
        );
        debug_assert_eq!(
            self.inv_diagonal.len(),
            self.vertices.len(),
            "inv_diagonal length must match n_steps"
        );
        debug_assert_eq!(
            self.neighbor_indices.len(),
            self.elimination_fractions.len(),
            "neighbor and fraction storage must be aligned"
        );

        let Some(&first_offset) = self.offsets.first() else {
            return;
        };
        debug_assert_eq!(first_offset, 0, "offsets must start at zero");

        let nnz = self.neighbor_indices.len();
        let mut prev = 0usize;
        for (i, &vertex) in self.vertices.iter().enumerate() {
            let start = self.offsets[i] as usize;
            let end = self.offsets[i + 1] as usize;
            debug_assert!(
                start <= end && end <= nnz,
                "invalid offset range [{start}, {end}) for nnz={nnz}"
            );
            debug_assert!(
                start >= prev,
                "offsets must be non-decreasing (prev={prev}, start={start})"
            );
            prev = end;

            debug_assert!(
                (vertex as usize) < n,
                "vertex {} out of bounds for factor dim {}",
                vertex,
                n
            );
            for &j in &self.neighbor_indices[start..end] {
                debug_assert!(
                    (j as usize) < n,
                    "neighbor {} out of bounds for factor dim {}",
                    j,
                    n
                );
            }
        }
        debug_assert_eq!(
            self.offsets.last().copied().unwrap_or_default() as usize,
            nnz,
            "final offset must match nnz"
        );
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

    /// Record an isolated vertex (no neighbors, clamped diagonal).
    pub(crate) fn record_isolated(&mut self, vertex: usize, diagonal: T) {
        self.vertices.push(vertex as u32);
        self.inv_diagonal.push(if diagonal.abs() > T::near_zero() {
            T::one() / diagonal
        } else {
            T::zero()
        });
        self.offsets.push(self.neighbor_indices.len() as u32);
        debug_assert_eq!(self.offsets.len(), self.vertices.len() + 1);
    }

    /// Record one column from a `SampledColumn`.
    pub(crate) fn record_column(&mut self, vertex: usize, column: &super::SampledColumn<T>) {
        self.vertices.push(vertex as u32);
        self.inv_diagonal
            .push(if column.diagonal.abs() > T::near_zero() {
                T::one() / column.diagonal
            } else {
                T::zero()
            });
        self.neighbor_indices.extend_from_slice(&column.neighbors);
        self.elimination_fractions
            .extend_from_slice(&column.fractions);
        self.offsets.push(self.neighbor_indices.len() as u32);
        debug_assert_eq!(self.offsets.len(), self.vertices.len() + 1);
    }
}

/// Approximate Cholesky decomposition L D L^T of an SDDM matrix.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    /// Dimension of the internal factorization (may include Gremban augmentation vertex).
    pub(crate) n: usize,
    /// Original input matrix dimension (before possible Gremban augmentation).
    pub(crate) original_n: usize,
    pub(crate) sequence: EliminationSequence<T>,
}

/// Errors returned by fallible [`Factor`] solve methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveError {
    /// Right-hand side length exceeds factor dimension.
    RhsLengthExceedsFactor {
        /// Provided RHS length.
        rhs_len: usize,
        /// Factor dimension (`Factor::n()`).
        factor_dim: usize,
    },
    /// Work buffer is smaller than factor dimension.
    WorkBufferTooSmall {
        /// Provided work length.
        work_len: usize,
        /// Factor dimension (`Factor::n()`).
        factor_dim: usize,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RhsLengthExceedsFactor {
                rhs_len,
                factor_dim,
            } => write!(
                f,
                "rhs length {} exceeds factor dimension {}",
                rhs_len, factor_dim
            ),
            Self::WorkBufferTooSmall {
                work_len,
                factor_dim,
            } => write!(
                f,
                "work buffer too small: got {}, need at least {}",
                work_len, factor_dim
            ),
        }
    }
}

impl std::error::Error for SolveError {}

impl<T> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    #[inline]
    fn validate_rhs_and_work(&self, b: &[T], work: &[T]) -> Result<(), SolveError> {
        if b.len() > self.n {
            return Err(SolveError::RhsLengthExceedsFactor {
                rhs_len: b.len(),
                factor_dim: self.n,
            });
        }
        if work.len() < self.n {
            return Err(SolveError::WorkBufferTooSmall {
                work_len: work.len(),
                factor_dim: self.n,
            });
        }
        Ok(())
    }

    #[inline]
    fn validate_in_place_work(&self, y: &[T]) -> Result<(), SolveError> {
        if y.len() < self.n {
            return Err(SolveError::WorkBufferTooSmall {
                work_len: y.len(),
                factor_dim: self.n,
            });
        }
        Ok(())
    }

    /// Dimension of the original input matrix.
    ///
    /// This is the dimension of vectors returned by [`Self::solve`] and accepted
    /// by the preconditioner interface. For pure Laplacians this equals
    /// [`Self::n`]; for SDDM matrices with Gremban augmentation it is one less.
    #[inline]
    pub fn original_n(&self) -> usize {
        self.original_n
    }

    /// Internal factor dimension (may be larger than [`Self::original_n`] if
    /// Gremban augmentation was applied).
    ///
    /// This is the size required for work buffers in low-level methods like
    /// [`Self::solve_into`] and [`Self::solve_in_place`].
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of elimination steps in the factor.
    #[inline]
    pub fn n_steps(&self) -> usize {
        self.sequence.n_steps()
    }

    fn forward(&self, y: &mut [T]) {
        let seq = &self.sequence;
        debug_assert!(
            y.len() >= self.n,
            "work buffer too small in forward: got {}, need at least {}",
            y.len(),
            self.n
        );
        seq.debug_assert_valid_for_dim(self.n);
        for i in 0..seq.n_steps() {
            let step = seq.step(i);
            let inv_diag = seq.inv_diagonal[i];
            step.apply_forward(y, inv_diag);
        }
    }

    fn backward(&self, y: &mut [T]) {
        let seq = &self.sequence;
        debug_assert!(
            y.len() >= self.n,
            "work buffer too small in backward: got {}, need at least {}",
            y.len(),
            self.n
        );
        seq.debug_assert_valid_for_dim(self.n);
        for i in (0..seq.n_steps()).rev() {
            let step = seq.step(i);
            step.apply_backward(y);
        }
    }

    #[inline]
    fn project_zero_mean(&self, y: &mut [T]) {
        let n = self.n.min(y.len());
        if n == 0 {
            return;
        }
        let Some(n_scalar): Option<T> = <T as num_traits::NumCast>::from(n) else {
            return;
        };
        let mean = y[..n].iter().fold(T::zero(), |a, &b| a + b) / n_scalar;
        for yi in &mut y[..n] {
            *yi = *yi - mean;
        }
    }

    #[inline]
    fn solve_into_kernel(&self, b: &[T], work: &mut [T], project_zero_mean: bool) {
        work[..b.len()].copy_from_slice(b);
        work[b.len()..self.n].fill(T::zero());
        self.forward(work);
        self.backward(work);
        if project_zero_mean {
            self.project_zero_mean(work);
        }
    }

    /// Solve LDL^T x = b, returning a newly allocated solution vector.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.n()`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut work = vec![T::zero(); self.n];
        self.solve_into(b, &mut work)?;
        work.truncate(self.original_n);
        Ok(work)
    }

    /// Backwards-compatible alias of [`Self::solve`].
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.n()`.
    pub fn try_solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        self.solve(b)
    }

    /// Solve L D L^T x = b in-place, writing the result into `work`.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.n()`.
    /// Returns [`SolveError::WorkBufferTooSmall`] if `work.len() < self.n()`.
    pub fn solve_into(&self, b: &[T], work: &mut [T]) -> Result<(), SolveError> {
        self.solve_into_with_projection(b, work, true)
    }

    /// Backwards-compatible alias of [`Self::solve_into`].
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.n()`.
    /// Returns [`SolveError::WorkBufferTooSmall`] if `work.len() < self.n()`.
    pub fn try_solve_into(&self, b: &[T], work: &mut [T]) -> Result<(), SolveError> {
        self.solve_into(b, work)
    }

    /// Solve L D L^T x = b in-place with configurable zero-mean projection.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.n()`.
    /// Returns [`SolveError::WorkBufferTooSmall`] if `work.len() < self.n()`.
    pub fn solve_into_with_projection(
        &self,
        b: &[T],
        work: &mut [T],
        project_zero_mean: bool,
    ) -> Result<(), SolveError> {
        self.validate_rhs_and_work(b, work)?;
        self.solve_into_kernel(b, work, project_zero_mean);
        Ok(())
    }

    /// Backwards-compatible alias of [`Self::solve_into_with_projection`].
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.n()`.
    /// Returns [`SolveError::WorkBufferTooSmall`] if `work.len() < self.n()`.
    pub fn try_solve_into_with_projection(
        &self,
        b: &[T],
        work: &mut [T],
        project_zero_mean: bool,
    ) -> Result<(), SolveError> {
        self.solve_into_with_projection(b, work, project_zero_mean)
    }

    /// Solve L D L^T x = b in-place, assuming `y` already contains the RHS.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::WorkBufferTooSmall`] if `y.len() < self.n()`.
    pub fn solve_in_place(&self, y: &mut [T]) -> Result<(), SolveError> {
        self.validate_in_place_work(y)?;
        self.solve_in_place_unvalidated(y);
        Ok(())
    }

    /// Backwards-compatible alias of [`Self::solve_in_place`].
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::WorkBufferTooSmall`] if `y.len() < self.n()`.
    pub fn try_solve_in_place(&self, y: &mut [T]) -> Result<(), SolveError> {
        self.solve_in_place(y)
    }

    /// Solve L D L^T x = b in-place without upfront `y` length validation.
    ///
    /// This method is fully safe and keeps Rust bounds checks on slice indexing.
    /// If `y.len() < self.n()`, it can panic while indexing.
    pub(crate) fn solve_in_place_unvalidated(&self, y: &mut [T]) {
        debug_assert!(
            y.len() >= self.n,
            "work buffer too small: got {}, need at least {}",
            y.len(),
            self.n
        );
        self.forward(y);
        self.backward(y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, RngExt, SeedableRng};
    use std::collections::BTreeSet;

    fn random_sequence(rng: &mut impl Rng, n: usize, n_steps: usize) -> EliminationSequence<f64> {
        let mut vertices = Vec::with_capacity(n_steps);
        let mut offsets = Vec::with_capacity(n_steps + 1);
        let mut neighbor_indices = Vec::new();
        let mut elimination_fractions = Vec::new();
        let mut inv_diagonal = Vec::with_capacity(n_steps);
        offsets.push(0);

        for _ in 0..n_steps {
            let v = rng.random_range(0..n);
            let max_deg = n.saturating_sub(1).min(4);
            let deg = if max_deg == 0 {
                0
            } else {
                rng.random_range(0..=max_deg)
            };

            let mut seen = BTreeSet::new();
            while seen.len() < deg {
                let u = rng.random_range(0..n);
                if u != v {
                    seen.insert(u);
                }
            }

            for (idx, u) in seen.into_iter().enumerate() {
                neighbor_indices.push(u as u32);
                let frac = if idx + 1 == deg {
                    1.0
                } else {
                    rng.random_range(0.0..1.0)
                };
                elimination_fractions.push(frac);
            }

            vertices.push(v as u32);
            inv_diagonal.push(rng.random_range(-2.0..2.0));
            offsets.push(neighbor_indices.len() as u32);
        }

        EliminationSequence {
            vertices,
            offsets,
            neighbor_indices,
            elimination_fractions,
            inv_diagonal,
        }
    }

    fn reference_forward(seq: &EliminationSequence<f64>, y: &mut [f64]) {
        for i in 0..seq.n_steps() {
            let step = seq.step(i);
            let n = step.neighbor_indices.len();
            let one = 1.0;
            let mut yi = y[step.vertex];

            if n == 0 {
                let inv = seq.inv_diagonal[i];
                if inv != 0.0 {
                    y[step.vertex] = yi * inv;
                }
                continue;
            }

            for (&j, &f) in step.neighbor_indices[..n - 1]
                .iter()
                .zip(step.elimination_fractions.iter())
            {
                y[j as usize] += f * yi;
                yi *= one - f;
            }

            let j_last = step.neighbor_indices[n - 1] as usize;
            y[j_last] += yi;
            y[step.vertex] = if seq.inv_diagonal[i] != 0.0 {
                yi * seq.inv_diagonal[i]
            } else {
                yi
            };
        }
    }

    fn reference_backward(seq: &EliminationSequence<f64>, y: &mut [f64]) {
        for i in (0..seq.n_steps()).rev() {
            let step = seq.step(i);
            let n = step.neighbor_indices.len();
            if n == 0 {
                continue;
            }

            let j_last = step.neighbor_indices[n - 1] as usize;
            let mut yi = y[step.vertex] + y[j_last];
            for (&j, &f) in step.neighbor_indices[..n - 1]
                .iter()
                .zip(step.elimination_fractions.iter())
                .rev()
            {
                yi = (1.0 - f) * yi + f * y[j as usize];
            }
            y[step.vertex] = yi;
        }
    }

    fn reference_project_zero_mean(y: &mut [f64], n: usize) {
        if n == 0 {
            return;
        }
        let mean = y[..n].iter().sum::<f64>() / n as f64;
        for yi in &mut y[..n] {
            *yi -= mean;
        }
    }

    fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (i, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= tol,
                "mismatch at {i}: lhs={a:.16e}, rhs={b:.16e}, diff={diff:.3e}"
            );
        }
    }

    #[test]
    fn unsafe_solve_kernel_matches_checked_reference_randomized() {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x5EED_BAAD_F00D);

        for _ in 0..300 {
            let n = rng.random_range(1..=16);
            let n_steps = rng.random_range(1..=n);
            let sequence = random_sequence(&mut rng, n, n_steps);
            let factor = Factor {
                n,
                original_n: n,
                sequence,
            };

            let rhs_len = rng.random_range(0..=n);
            let mut rhs = vec![0.0; rhs_len];
            for v in &mut rhs {
                *v = rng.random_range(-5.0..5.0);
            }
            let project = rng.random_bool(0.5);

            let mut unsafe_work = vec![0.0; n];
            factor.solve_into_kernel(&rhs, &mut unsafe_work, project);

            let mut checked_work = vec![0.0; n];
            checked_work[..rhs_len].copy_from_slice(&rhs);
            checked_work[rhs_len..].fill(0.0);
            reference_forward(&factor.sequence, &mut checked_work);
            reference_backward(&factor.sequence, &mut checked_work);
            if project {
                reference_project_zero_mean(&mut checked_work, n);
            }

            assert_close(&unsafe_work, &checked_work, 1e-12);
        }
    }
}
