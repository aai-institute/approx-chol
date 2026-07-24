//! The [`Factor`] LDLᵀ decomposition and its solve API.

use super::sequence::EliminationSequence;
use super::FactorError;
use core::fmt;

#[cfg(test)]
mod tests;

/// Approximate Cholesky decomposition L D L^T of an SDDM matrix.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            serialize = "T: serde::Serialize",
            deserialize = "T: serde::de::DeserializeOwned"
        ),
        try_from = "FactorData<T>"
    )
)]
#[derive(Clone, Debug)]
pub struct Factor<T = f64> {
    /// Dimension of the internal factorization (may include Gremban augmentation vertex).
    pub(crate) n: usize,
    /// Original input matrix dimension (before possible Gremban augmentation).
    pub(crate) original_n: usize,
    pub(crate) sequence: EliminationSequence<T>,
}

/// Deserialization shadow for [`Factor`]: the raw persisted fields, validated
/// on conversion so an invalid factor can never be constructed via serde.
///
/// `Factor` is the only publicly-reachable deserialize entry (`EliminationSequence`
/// derives `Deserialize` but is not re-exported), so validating here is
/// sufficient to cover every persisted-factor path.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "T: serde::de::DeserializeOwned"))]
struct FactorData<T> {
    n: usize,
    original_n: usize,
    sequence: EliminationSequence<T>,
}

#[cfg(feature = "serde")]
impl<T> TryFrom<FactorData<T>> for Factor<T> {
    type Error = FactorError;

    fn try_from(data: FactorData<T>) -> Result<Self, Self::Error> {
        let factor = Self {
            n: data.n,
            original_n: data.original_n,
            sequence: data.sequence,
        };
        factor.validate_structure()?;
        Ok(factor)
    }
}

// Structural validation (no numeric `T` bound; shared by the solve-path
// `debug_assert` and the serde deserialize boundary).
impl<T> Factor<T> {
    /// Check the invariants the solve path relies on: `original_n <= n` and a
    /// [`EliminationSequence::validate_for_dim`]-valid sequence for dimension `n`.
    fn validate_structure(&self) -> Result<(), FactorError> {
        if self.original_n > self.n {
            return Err(FactorError::OriginalDimExceedsInternal {
                original_n: self.original_n,
                n: self.n,
            });
        }
        self.sequence.validate_for_dim(self.n)
    }

    #[inline]
    fn debug_assert_valid_structure(&self) {
        debug_assert_eq!(self.validate_structure(), Ok(()));
    }
}

/// Errors returned by fallible [`Factor`] solve methods.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveError {
    /// Right-hand side length exceeds the solvable (original) dimension.
    RhsLengthExceedsFactor {
        /// Provided RHS length.
        rhs_len: usize,
        /// Maximum accepted RHS length (`Factor::original_n()`).
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
                "rhs length {} exceeds original matrix dimension {}",
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
        if b.len() > self.original_n {
            return Err(SolveError::RhsLengthExceedsFactor {
                rhs_len: b.len(),
                factor_dim: self.original_n,
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
        self.debug_assert_valid_structure();
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
        self.debug_assert_valid_structure();
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

    /// Index of the Gremban auxiliary "ground" vertex (appended after the
    /// original vertices), or `None` for a non-augmented factor.
    #[inline]
    fn aux_vertex(&self) -> Option<usize> {
        (self.n > self.original_n).then_some(self.original_n)
    }

    /// Recover the original solution by grounding against the aux vertex:
    /// `x_i = y_i - y_aux` (see [`Self::solve_into_kernel`] for why).
    #[inline]
    fn ground_by_aux(&self, y: &mut [T], aux: usize) {
        let y_aux = y[aux];
        for yi in &mut y[..aux] {
            *yi = *yi - y_aux;
        }
    }

    #[inline]
    fn solve_into_kernel(&self, b: &[T], work: &mut [T]) {
        work[..b.len()].copy_from_slice(b);
        work[b.len()..self.n].fill(T::zero());
        if let Some(aux) = self.aux_vertex() {
            // Gremban-augmented SDDM: the augmented Laplacian is singular with
            // M·1 = surplus ≠ 0, so a global zero-mean projection would break
            // M x = b. Instead put -Σb on the ground vertex (so the padded RHS
            // lies in range) and recover x_i = y_i - y_aux.
            let surplus = work[..aux].iter().fold(T::zero(), |acc, &x| acc + x);
            work[aux] = -surplus;
            self.forward(work);
            self.backward(work);
            self.ground_by_aux(work, aux);
        } else {
            // Pure Laplacian: singular with the constant null space; pick the
            // canonical zero-mean representative.
            self.forward(work);
            self.backward(work);
            self.project_zero_mean(work);
        }
    }

    /// Solve `M x = b`, returning a newly allocated solution of the original
    /// dimension (zero-mean for a pure Laplacian, Gremban-grounded for SDDM).
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.original_n()`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, SolveError> {
        let mut work = vec![T::zero(); self.n];
        self.solve_into(b, &mut work)?;
        work.truncate(self.original_n);
        Ok(work)
    }

    /// Solve `M x = b` in-place, writing the recovered solution into `work`.
    ///
    /// [`Self::solve_in_place`] performs only the raw triangular solve, without
    /// this recovery step.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthExceedsFactor`] if `b.len() > self.original_n()`.
    /// Returns [`SolveError::WorkBufferTooSmall`] if `work.len() < self.n()`.
    pub fn solve_into(&self, b: &[T], work: &mut [T]) -> Result<(), SolveError> {
        self.validate_rhs_and_work(b, work)?;
        self.solve_into_kernel(b, work);
        Ok(())
    }

    /// Apply the raw `L D L^T` triangular solve in place, assuming `y` already
    /// contains the RHS — no zero-mean projection and no Gremban grounding.
    ///
    /// For pure-Laplacian preconditioning, where the iterative solver absorbs
    /// the constant null space. For an augmented SDDM factor the raw result is
    /// *not* the solution of `M x = b`; use [`Self::solve`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::WorkBufferTooSmall`] if `y.len() < self.n()`.
    pub fn solve_in_place(&self, y: &mut [T]) -> Result<(), SolveError> {
        self.validate_in_place_work(y)?;
        self.forward(y);
        self.backward(y);
        Ok(())
    }
}
