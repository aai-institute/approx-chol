use crate::dense::LowerTriangular;
use crate::min_degree::Sequence;
use crate::sgdd::{Ground, Sheets};
use crate::Real;

/// What factors the kept vertices; on two sheets it solves `[b; −b]` and returns half their difference.
#[derive(Clone, Debug)]
pub(super) struct Kept<T> {
    pub(super) sheets: Sheets,
    pub(super) kernel: Kernel<T>,
}

impl<T: Real> Kept<T> {
    pub(super) fn solve(&self, x: &mut [T], scratch: &mut [T]) {
        todo!()
    }
}

/// A Cholesky with one slot left free; the variant is how that slot is fixed.
#[derive(Clone, Debug)]
pub(super) enum Kernel<T> {
    /// `L + diag(s)` as `L` plus a ground in the last slot, pinned to `0`.
    Grounded(Cholesky<T>),
    /// `L`, whose solution is the zero-mean one.
    Floating(Cholesky<T>),
}

impl<T: Real> Kernel<T> {
    pub(super) fn new(ground: &Ground<T>, cholesky: Cholesky<T>) -> Self {
        match ground {
            Ground::Grounded(_) => Self::Grounded(cholesky),
            Ground::Floating => Self::Floating(cholesky),
        }
    }

    /// Input rows, then a grounded kernel's ground slot.
    pub(super) fn slots(&self) -> usize {
        todo!()
    }

    /// `slots` holds the right-hand side, then the solution.
    pub(super) fn solve(&self, slots: &mut [T]) {
        todo!()
    }
}

/// Exact or sampled; both leave one slot uneliminated.
#[derive(Clone, Debug)]
pub(super) enum Cholesky<T> {
    Dense(LowerTriangular<T>),
    Sequence(Sequence<T>),
}
