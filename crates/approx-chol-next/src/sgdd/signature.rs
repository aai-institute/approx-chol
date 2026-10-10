use super::{Congruence, OffDiagonal};
use crate::Real;

/// The signs `S` folding `A_c` into a Z-matrix; `None` when frustrated.
pub(super) fn signature<T: Real>(off_diagonal: &OffDiagonal<T>) -> Option<Congruence<T>> {
    match off_diagonal {
        OffDiagonal::Bipartite(cross) if cross.positive() => {
            Some(Congruence::negate_from(cross.n_eliminated()))
        }
        _ => todo!("a mixed-sign or general pattern searches for signs"),
    }
}
