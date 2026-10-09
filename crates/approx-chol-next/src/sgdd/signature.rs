use super::{Congruence, OffDiagonal};
use crate::Real;

/// The signs that fold `A` into a Z-matrix, or `None`: a negative cycle admits no signature.
pub(super) fn signature<T: Real>(off_diagonal: &OffDiagonal<T>) -> Option<Congruence<T>> {
    todo!()
}
