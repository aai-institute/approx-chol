use super::OffDiagonal;
use crate::Real;

/// The signs `S` that fold `A_c` into a Z-matrix.
pub(super) enum Signature {
    /// Already a Z-matrix.
    Identity,
    /// A positive bipartite block: every vertex from this one on is negated.
    NegateFrom(usize),
    /// `true` where the vertex is negated.
    Negated(Box<[bool]>),
}

/// `None` for a frustrated component: a negative cycle admits no signature.
pub(super) fn signature<T: Real>(off_diagonal: &OffDiagonal<T>) -> Option<Signature> {
    todo!()
}
