use crate::config::Multiplicity;
use crate::dense::{Fallback, LowerTriangular};
use crate::laplacian::{Sddm, Sheets};
use crate::min_degree::{self, Sequence};
use crate::sampling::Streams;
use crate::Real;

/// The kept vertices' factor; it builds its own Cholesky, so the facts it reads come from that matrix.
#[derive(Clone, Debug)]
pub(super) struct Kept<T> {
    sheets: Sheets,
    grounded: bool,
    cholesky: Cholesky<T>,
}

impl<T: Real> Kept<T> {
    /// `vertices` names the rows a declined pivot reports.
    pub(super) fn dense(sddm: &Sddm<T>, vertices: &[u32]) -> Result<Self, Fallback> {
        let lower = LowerTriangular::factor(sddm, vertices)?;
        Ok(Self::of(sddm, Cholesky::Dense(lower)))
    }

    pub(super) fn sampled(sddm: &Sddm<T>, multiplicity: Multiplicity, streams: Streams) -> Self {
        let sequence = min_degree::eliminate(sddm, multiplicity, streams);
        Self::of(sddm, Cholesky::Sequence(sequence))
    }

    fn of(sddm: &Sddm<T>, cholesky: Cholesky<T>) -> Self {
        Self {
            sheets: sddm.sheets(),
            grounded: sddm.ground().surplus().is_some(),
            cholesky,
        }
    }

    /// Only one floating sheet leaves `A_c` singular: a cover's null vector is off its range.
    pub(super) fn singular(&self) -> bool {
        self.sheets == Sheets::One && !self.grounded
    }

    /// Lifts to the sheets, pins the ground or projects the mean, substitutes, and folds back.
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T]) {
        todo!()
    }
}

/// Exact or sampled; both leave one slot uneliminated.
#[derive(Clone, Debug)]
enum Cholesky<T> {
    Dense(LowerTriangular<T>),
    Sequence(Sequence<T>),
}
