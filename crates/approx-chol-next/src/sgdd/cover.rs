use super::SignedGeneral;
use crate::laplacian::Sddm;
use crate::Real;

impl<T: Real> SignedGeneral<T> {
    /// The Gremban cover: a Laplacian on two sheets that acts as this matrix on `[z; −z]`.
    pub(crate) fn cover(&self) -> Sddm<T> {
        todo!()
    }
}
