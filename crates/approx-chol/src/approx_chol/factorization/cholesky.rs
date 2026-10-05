//! Both arms leave one slot of their block free, so a block's solve never asks which one
//! it got — the exact arm the last slot, the approximate one whichever min-degree spared.

use super::approximate::EliminationSequence;
use super::exact::LowerTriangular;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::types::Real;

#[cfg(test)]
mod tests;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"
    ))
)]
#[derive(Clone, Debug)]
pub(crate) enum Cholesky<T> {
    /// Algorithm 8's sampled elimination sequence.
    Approximate(EliminationSequence<T>),
    /// Exact dense factor over every slot but the last.
    Exact(LowerTriangular<T>),
}

impl<T> Cholesky<T> {
    pub(super) fn eliminated(&self) -> usize {
        match self {
            Self::Approximate(sequence) => sequence.n_steps(),
            Self::Exact(lower) => lower.rows(),
        }
    }
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
    pub(super) fn validate(&self) -> Result<(), FactorError> {
        match self {
            Self::Approximate(sequence) => sequence.validate_values(),
            Self::Exact(lower) => lower.validate_values(),
        }
    }
}
