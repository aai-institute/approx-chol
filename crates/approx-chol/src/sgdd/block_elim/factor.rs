//! Reduced-system factor for Schur-complement local solves.
//!
//! [`ReducedFactor`] wraps an `approx-chol` factor, either directly or behind a
//! Gremban cover. [`ReducedSystem`] bridges to the `approx-chol` builder.

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
        /// Factor of the doubled cover.
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

    pub(crate) fn fallbacks(&self) -> &[crate::Fallback] {
        match self {
            Self::Direct { factor, .. } => factor.fallbacks(),
            Self::Cover { inner, .. } => inner.fallbacks(),
        }
    }

    /// The kept block's size; the backend's ground vertex and the cover's sheets stay inside.
    pub(crate) fn dimension(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.n(),
            Self::Cover { m, .. } => *m,
        }
    }

    /// The reduced buffer's length: the kept block, then the backend's ground slot when it has one.
    pub(crate) fn slots(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.slots(),
            Self::Cover { m, .. } => *m,
        }
    }

    /// Extra scratch beyond the reduced buffers; only [`Self::Cover`] embeds into a larger system.
    pub(crate) fn scratch_len(&self) -> usize {
        match self {
            Self::Direct { factor, .. } => factor.slots_scratch_len(),
            Self::Cover { inner, .. } => inner.slots() + inner.slots_scratch_len(),
        }
    }

    /// `x` spans [`Self::slots`], the kept block first; `scratch` is at least [`Self::scratch_len`] long.
    pub(crate) fn solve_in_place(
        &self,
        x: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), SolveError> {
        match self {
            Self::Direct { factor, .. } => {
                factor.solve_in_slots(x, &mut scratch[..factor.slots_scratch_len()])
            }
            Self::Cover { inner, m } => {
                debug_assert_eq!(*m, x.len());
                debug_assert_eq!(inner.n(), 2 * *m);
                let (buf, inner_scratch) = scratch.split_at_mut(inner.slots());
                buf[..*m].copy_from_slice(x);
                for (out, &v) in buf[*m..2 * *m].iter_mut().zip(x.iter()) {
                    *out = -v;
                }
                inner.solve_in_slots(buf, &mut inner_scratch[..inner.slots_scratch_len()])?;
                // Read back the antisymmetric solution: x = (x⁺ - x⁻) / 2.
                for (i, out) in x.iter_mut().enumerate() {
                    *out = 0.5 * (buf[i] - buf[*m + i]);
                }
                Ok(())
            }
        }
    }
}

/// Reduced SDDM as Laplacian off-diagonals plus each row's surplus, exact rather than re-derived.
#[derive(Clone, Debug)]
pub(crate) struct ReducedSystem {
    laplacian: CsrMatrix,
    surplus: Vec<f64>,
}

impl ReducedSystem {
    /// `laplacian` stores no diagonal: the backend sums each row's adjacency and adds its surplus.
    pub(crate) fn new(laplacian: CsrMatrix, surplus: Vec<f64>) -> Self {
        debug_assert_eq!(laplacian.n(), surplus.len());
        Self { laplacian, surplus }
    }

    /// Returns the `approx-chol` error unmapped, so the caller can spot an unusable exact pivot.
    pub(crate) fn factor(&self, config: crate::Config) -> Result<Factor, Error> {
        let csr = CsrRef::new(
            self.laplacian.indptr(),
            self.laplacian.indices(),
            self.laplacian.data(),
            u32::try_from(self.laplacian.n()).expect("Schur complement dimension exceeds u32::MAX"),
        )?;
        Builder::new(config).build_with_surpluses(csr, &self.surplus)
    }
}
