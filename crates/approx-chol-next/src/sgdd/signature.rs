use super::Edges;
use crate::Real;

/// The signs that fold `A`, or none: a negative cycle admits no signature.
pub(super) enum Signature {
    /// `true` where the vertex is negated.
    Balanced(Box<[bool]>),
    Frustrated,
}

pub(super) fn signature<T: Real>(edges: &Edges<T>) -> Signature {
    todo!()
}
