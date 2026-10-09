use crate::clique_tree::Streams;
use crate::config::Multiplicity;
use crate::dense::{DenseCholesky, Fallback};
use crate::min_degree::{self, SampledCholesky};
use crate::sddm::{Grounding, Sddm, Sheets};
use crate::Real;

/// The factor of the SDDM the kernel sees; it builds its own Cholesky, so its facts come from that matrix.
#[derive(Clone, Debug)]
pub(super) struct SddmFactor<T> {
    sheets: Sheets,
    floating: bool,
    cholesky: Cholesky<T>,
}

impl<T: Real> SddmFactor<T> {
    /// `vertices` names each base vertex in `A`, so a declined pivot is reported there.
    pub(super) fn dense(sddm: &Sddm<'_, T>, vertices: &[u32]) -> Result<Self, Fallback> {
        let dense = DenseCholesky::factor(sddm, vertices)?;
        Ok(Self::of(sddm, Cholesky::Dense(dense)))
    }

    pub(super) fn sampled(
        sddm: &Sddm<'_, T>,
        multiplicity: Multiplicity,
        streams: Streams,
    ) -> Self {
        let sampled = min_degree::eliminate(sddm, multiplicity, streams);
        Self::of(sddm, Cholesky::Sampled(sampled))
    }

    fn of(sddm: &Sddm<'_, T>, cholesky: Cholesky<T>) -> Self {
        Self {
            sheets: sddm.sheets(),
            floating: matches!(sddm.grounding(), Grounding::Floating),
            cholesky,
        }
    }

    /// Only one floating sheet is singular on the solve's subspace: a cover's null vector is off it.
    pub(super) fn singular(&self) -> bool {
        self.sheets == Sheets::One && self.floating
    }

    /// Lifts to the sheets, appends a grounded matrix's ground slot, substitutes, and descends.
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T]) {
        todo!()
    }
}

/// Exact or sampled; both leave one slot uneliminated.
#[derive(Clone, Debug)]
enum Cholesky<T> {
    Dense(DenseCholesky<T>),
    Sampled(SampledCholesky<T>),
}
