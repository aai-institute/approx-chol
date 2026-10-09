//! Reduced-system factor for Schur-complement local solves.
//!
//! [`ReducedFactor`] wraps an `approx-chol` factor, either directly or behind a
//! Gremban cover. [`factor_sparse`] bridges to the `approx-chol` builder.

use crate::low_level::Builder;
use crate::{CsrRef, Error, Factor, SolveError};

use super::csr_matrix::CsrMatrix;
use crate::sgdd::factor_pairs::Grounding;

/// Reduced-system factor for Schur-complement local solves.
#[derive(Clone, Debug)]
pub(crate) enum ReducedFactor {
    /// Factor of the reduced Schur CSR itself, carrying the gauge applied around the solve.
    Direct {
        /// Factor of the reduced Schur.
        factor: Factor,
        /// Gauge of the reduced system.
        grounding: Grounding,
    },
    /// Gremban cover of a signed reduced Schur, kept here so the operator stays single-sized.
    Cover {
        /// Factor of the doubled cover; the tail past `2*m` is the grounding augmentation.
        inner: Factor,
        /// Single signed reduced dimension exposed to the caller.
        m: usize,
    },
}

impl ReducedFactor {
    /// The gauge the operator-level solve applies; `None` for a self-grounding cover.
    pub(crate) fn grounding(&self) -> Option<Grounding> {
        match self {
            Self::Direct { grounding, .. } => Some(*grounding),
            Self::Cover { .. } => None,
        }
    }

    /// Explicit ground vertex appended past the kept block; `None` for the kept block alone.
    pub(crate) fn explicit_ground_index(&self, n_kept: usize) -> Option<usize> {
        (self.grounding() == Some(Grounding::Grounded) && self.input_dimension() == n_kept + 1)
            .then_some(n_kept)
    }

    /// Whether the factor was built over a kept block of this size, with or without ground.
    pub(crate) fn spans_kept_block(&self, n_kept: usize) -> bool {
        self.input_dimension() == n_kept || self.explicit_ground_index(n_kept).is_some()
    }

    pub(crate) fn fallbacks(&self) -> &[crate::Fallback] {
        match self {
            Self::Direct { factor, .. } => factor.fallbacks(),
            Self::Cover { inner, .. } => inner.fallbacks(),
        }
    }

    /// Dimension handed to the backend, which adds grounding vertices of its own.
    pub(crate) fn input_dimension(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.n(),
            // The cover is hidden behind the single signed interface.
            Self::Cover { m, .. } => *m,
        }
    }

    pub(crate) fn solve_dimension(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.n(),
            Self::Cover { m, .. } => *m,
        }
    }

    /// Extra scratch beyond the reduced buffers; only [`Self::Cover`] embeds into a larger system.
    pub(crate) fn scratch_len(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.scratch_len(),
            Self::Cover { inner, .. } => inner.n() + inner.scratch_len(),
        }
    }

    /// `x` spans [`Self::solve_dimension`]; `scratch` is at least [`Self::scratch_len`] long.
    pub(crate) fn solve_in_place(
        &self,
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SolveError> {
        match self {
            Self::Direct { factor, .. } => {
                debug_assert_eq!(factor.n(), x.len());
                solve_approx(factor, x, scratch)
            }
            Self::Cover { inner, m } => {
                debug_assert_eq!(*m, x.len());
                let cover_n = inner.n();
                let (buf, inner_scratch) = scratch.split_at_mut(cover_n);
                // Grounding vertices past the `2m` cover nodes have zero RHS, so clear the tail.
                buf[..*m].copy_from_slice(x);
                for (out, &v) in buf[*m..2 * *m].iter_mut().zip(x.iter()) {
                    *out = -v;
                }
                for slot in buf[2 * *m..].iter_mut() {
                    *slot = 0.0;
                }
                solve_approx(inner, buf, inner_scratch)?;
                // Read back the antisymmetric solution: x = (x⁺ - x⁻) / 2.
                for (i, out) in x.iter_mut().enumerate() {
                    *out = 0.5 * (buf[i] - buf[*m + i]);
                }
                Ok(())
            }
        }
    }
}

fn solve_approx(f: &Factor, x: &mut [f64], scratch: &mut [f64]) -> Result<(), SolveError> {
    f.solve_in_place(x, &mut scratch[..f.scratch_len()])
}

/// Returns the `approx-chol` error unmapped, so the caller can spot an unusable exact pivot.
pub(crate) fn factor_sparse(matrix: &CsrMatrix, config: crate::Config) -> Result<Factor, Error> {
    let csr = CsrRef::new(
        matrix.indptr(),
        matrix.indices(),
        matrix.data(),
        u32::try_from(matrix.n()).expect("Schur complement dimension exceeds u32::MAX"),
    )?;
    Builder::new(config).build(csr)
}
