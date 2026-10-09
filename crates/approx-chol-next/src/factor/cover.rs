use crate::sgdd::{Ground, Sddm};
use crate::storage::UpperAdjacency;
use crate::Real;

/// The Gremban cover of a signed general matrix: a Laplacian on two sheets that acts as it on `[z; −z]`.
pub(super) fn cover<T: Real>(signed: &UpperAdjacency<T>, ground: &Ground<T>) -> Sddm<'static, T> {
    todo!()
}
