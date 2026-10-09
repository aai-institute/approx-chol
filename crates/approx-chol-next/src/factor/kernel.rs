use crate::dense::LowerTriangular;
use crate::laplacian::{Sddm, Sheets};
use crate::min_degree::Sequence;
use crate::Real;

/// The kept vertices' factor, and the two facts its solve reads off the matrix it was built from.
#[derive(Clone, Debug)]
pub(super) struct Kept<T> {
    sheets: Sheets,
    grounded: bool,
    cholesky: Cholesky<T>,
}

impl<T: Real> Kept<T> {
    pub(super) fn new(sddm: &Sddm<T>, cholesky: Cholesky<T>) -> Self {
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
pub(super) enum Cholesky<T> {
    Dense(LowerTriangular<T>),
    Sequence(Sequence<T>),
}
