use super::anchor::Anchor;
use super::cholesky::Cholesky;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::types::Real;
use core::num::NonZeroUsize;

#[cfg(test)]
mod tests;

/// A block's dimension in both forms its consumers ask for, so none of them spells
/// the pinned variable's subtraction itself.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockDim(NonZeroUsize);

impl BlockDim {
    /// `None` for a block of no variables: every dimension derived from it would
    /// underflow.
    pub(crate) fn of(total: usize) -> Option<Self> {
        NonZeroUsize::new(total).map(Self)
    }

    pub(crate) fn total(self) -> usize {
        self.0.get()
    }

    /// Variables the block solves for — all but the pinned last one.
    pub(crate) fn solved(self) -> usize {
        self.0.get() - 1
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            serialize = "T: serde::Serialize",
            deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"
        ),
        try_from = "BlockData<T>"
    )
)]
#[derive(Clone, Debug)]
pub(crate) struct Block<T> {
    pub(super) dim: BlockDim,
    pub(super) anchor: Anchor,
    pub(super) cholesky: Cholesky<T>,
}

impl<T> Block<T> {
    pub(crate) fn new(dim: BlockDim, anchor: Anchor, cholesky: Cholesky<T>) -> Self {
        Self {
            dim,
            anchor,
            cholesky,
        }
    }
}

/// A block as a payload carries it: a `dim` nothing has yet held its `cholesky` to. Same
/// fields in the same order as [`Block`], which is what writes these bytes.
#[cfg(any(feature = "serde", test))]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "T: serde::de::DeserializeOwned"))
)]
struct BlockData<T> {
    dim: BlockDim,
    anchor: Anchor,
    cholesky: Cholesky<T>,
}

/// Pinning the dim to the payload behind it here is what lets every consumer sum block
/// dims without a check of its own.
#[cfg(any(feature = "serde", test))]
impl<T: num_traits::Float> TryFrom<BlockData<T>> for Block<T> {
    type Error = FactorError;

    fn try_from(data: BlockData<T>) -> Result<Self, Self::Error> {
        data.cholesky.validate_for_dim(data.dim)?;
        Ok(Self::new(data.dim, data.anchor, data.cholesky))
    }
}

impl<T: Real> Block<T> {
    fn solve(&self, values: &mut [T], canonical: bool) {
        self.anchor.prepare(values);
        self.cholesky.apply(values);
        self.anchor.recover(values, canonical);
    }

    pub(super) fn solve_anchored(&self, values: &mut [T]) {
        self.solve(values, false);
    }

    pub(super) fn solve_canonical(&self, values: &mut [T]) {
        self.solve(values, true);
    }
}
