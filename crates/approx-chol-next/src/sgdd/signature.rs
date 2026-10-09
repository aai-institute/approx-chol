use super::{Congruence, OffDiagonal};
use crate::Real;

/// The signs `S` folding `A_c` into a Z-matrix, identity without a traversal when it is one; `None` when frustrated.
pub(super) fn signature<T: Real>(off_diagonal: &OffDiagonal<T>) -> Option<Congruence<T>> {
    todo!()
}
