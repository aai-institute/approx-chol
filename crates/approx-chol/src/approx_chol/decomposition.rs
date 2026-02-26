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
    /// Forward elimination: scatter pivot weight to neighbors, then scale by D^{-1}.
    #[inline(always)]
    pub(crate) fn apply_forward(&self, y: &mut [T], inv_diag: T) {
        let n = self.neighbor_indices.len();
        let zero = T::zero();
        let one = T::one();
        if n == 0 {
            if inv_diag != zero {
                unsafe {
                    *y.get_unchecked_mut(self.vertex) = *y.get_unchecked(self.vertex) * inv_diag
                };
            }
            return;
        }

        let mut yi = unsafe { *y.get_unchecked(self.vertex) };

        for (&j, &f) in self.neighbor_indices[..n - 1]
            .iter()
            .zip(self.elimination_fractions.iter())
        {
            unsafe {
                *y.get_unchecked_mut(j as usize) = *y.get_unchecked(j as usize) + f * yi;
            };
            yi = yi * (one - f);
        }

        let j_last = unsafe { *self.neighbor_indices.get_unchecked(n - 1) } as usize;
        unsafe { *y.get_unchecked_mut(j_last) = *y.get_unchecked(j_last) + yi };
        let val = if inv_diag != zero { yi * inv_diag } else { yi };
        unsafe { *y.get_unchecked_mut(self.vertex) = val };
    }

    /// Backward substitution: gather neighbor contributions back to pivot.
    #[inline(always)]
    pub(crate) fn apply_backward(&self, y: &mut [T]) {
        let n = self.neighbor_indices.len();
        let one = T::one();
        if n == 0 {
            return;
        }

        let j_last = unsafe { *self.neighbor_indices.get_unchecked(n - 1) } as usize;
        let mut yi = unsafe { *y.get_unchecked(self.vertex) + *y.get_unchecked(j_last) };

        for (&j, &f) in self.neighbor_indices[..n - 1]
            .iter()
            .zip(self.elimination_fractions.iter())
            .rev()
        {
            yi = (one - f) * yi + f * unsafe { *y.get_unchecked(j as usize) };
        }

        unsafe { *y.get_unchecked_mut(self.vertex) = yi };
    }
}

/// Contiguous memory owner for a sequence of elimination steps.
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
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    pub(crate) n: usize,
    pub(crate) sequence: EliminationSequence<T>,
}

impl<T> Factor<T>
where
    T: num_traits::Float + Send + Sync + 'static,
{
    #[inline]
    fn assert_rhs_and_work(&self, b: &[T], work: &[T]) {
        assert!(
            b.len() <= self.n,
            "rhs length {} exceeds factor dimension {}",
            b.len(),
            self.n
        );
        assert!(
            work.len() >= self.n,
            "work buffer too small: got {}, need at least {}",
            work.len(),
            self.n
        );
    }

    #[inline]
    fn assert_in_place_work(&self, y: &[T]) {
        assert!(
            y.len() >= self.n,
            "work buffer too small: got {}, need at least {}",
            y.len(),
            self.n
        );
    }

    /// Dimension of the factor (may be larger than the original matrix if
    /// Gremban augmentation was applied).
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
        for i in 0..seq.n_steps() {
            let step = seq.step(i);
            let inv_diag = unsafe { *seq.inv_diagonal.get_unchecked(i) };
            step.apply_forward(y, inv_diag);
        }
    }

    fn backward(&self, y: &mut [T]) {
        let seq = &self.sequence;
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
        let mean = y[..n].iter().fold(T::zero(), |a, &b| a + b)
            / <T as num_traits::NumCast>::from(n).expect("n to scalar");
        for yi in &mut y[..n] {
            *yi = *yi - mean;
        }
    }

    /// Solve LDL^T x = b, returning a newly allocated solution vector.
    #[must_use]
    pub fn solve(&self, b: &[T]) -> Vec<T> {
        let mut work = vec![T::zero(); self.n];
        self.solve_into(b, &mut work);
        work
    }

    /// Solve L D L^T x = b in-place, writing the result into `work`.
    ///
    /// # Panics
    ///
    /// Panics if `b.len() > self.n()` or `work.len() < self.n()`.
    pub fn solve_into(&self, b: &[T], work: &mut [T]) {
        self.solve_into_with_projection(b, work, true);
    }

    /// Solve L D L^T x = b in-place with configurable zero-mean projection.
    ///
    /// # Panics
    ///
    /// Panics if `b.len() > self.n()` or `work.len() < self.n()`.
    pub fn solve_into_with_projection(&self, b: &[T], work: &mut [T], project_zero_mean: bool) {
        self.assert_rhs_and_work(b, work);
        work[..b.len()].copy_from_slice(b);
        work[b.len()..self.n].fill(T::zero());
        self.forward(work);
        self.backward(work);
        if project_zero_mean {
            self.project_zero_mean(work);
        }
    }

    /// Solve L D L^T x = b in-place, assuming `y` already contains the RHS.
    ///
    /// # Panics
    ///
    /// Panics if `y.len() < self.n()`.
    pub fn solve_in_place(&self, y: &mut [T]) {
        self.assert_in_place_work(y);
        // SAFETY: `assert_in_place_work` guarantees `y.len() >= self.n()`.
        unsafe { self.solve_in_place_unchecked(y) };
    }

    /// Solve L D L^T x = b in-place without checking `y` length.
    ///
    /// # Safety
    ///
    /// Caller must ensure `y.len() >= self.n()`.
    pub unsafe fn solve_in_place_unchecked(&self, y: &mut [T]) {
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
