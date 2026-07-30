//! Both arms leave one variable free, so a block's solve never asks which one it got —
//! the exact arm the pinned last vertex, the approximate one whichever min-degree spared.

use super::approximate::EliminationSequence;
#[cfg(any(feature = "serde", test))]
use super::block::BlockDim;
use super::exact::LowerTriangular;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::types::Real;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) enum Cholesky<T> {
    /// Algorithm 8's sampled elimination sequence.
    Approximate(EliminationSequence<T>),
    /// Exact dense factor over the variables the block solves for.
    Exact(LowerTriangular<T>),
}

impl<T: Real> Cholesky<T> {
    pub(super) fn apply(&self, values: &mut [T]) {
        match self {
            Self::Approximate(sequence) => sequence.substitute(values),
            Self::Exact(lower) => lower.substitute(values),
        }
    }
}

#[cfg(any(feature = "serde", test))]
impl<T: num_traits::Float> Cholesky<T> {
    pub(super) fn validate_for_dim(&self, dim: BlockDim) -> Result<(), FactorError> {
        match self {
            // An elimination sequence indexes the whole block, the pinned variable
            // included; a dense factor covers only the variables solved for.
            Self::Approximate(sequence) => sequence.validate_for_dim(dim.total()),
            Self::Exact(lower) => lower.validate_for_dim(dim),
        }
    }
}
