use super::{Congruence, OffDiagonal};
use crate::Real;

/// The signs `S` folding `A_c` into a Z-matrix; `None` when a cycle with an odd number of positive entries forbids it.
pub(super) fn signature<T: Real>(off_diagonal: &OffDiagonal<T>) -> Option<Congruence<T>> {
    todo!()
}
