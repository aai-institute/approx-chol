use super::anchor::Anchor;
use super::cholesky::Cholesky;
#[cfg(any(feature = "serde", test))]
use super::FactorError;
use crate::types::Real;
use core::num::NonZeroUsize;

#[cfg(test)]
mod tests;

/// A block's dimension in both forms its consumers ask for, and derivable from either,
/// so none of them spells the pinned variable's offset itself.
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

    #[cfg(any(feature = "serde", test))]
    pub(crate) fn pinning(solved: usize) -> Self {
        Self(NonZeroUsize::MIN.saturating_add(solved))
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
    dim: BlockDim,
    anchor: Anchor,
    cholesky: Cholesky<T>,
}

impl<T> Block<T> {
    pub(super) fn dim(&self) -> BlockDim {
        self.dim
    }

    pub(super) fn is_ground(&self) -> bool {
        self.anchor == Anchor::Ground
    }
}

impl<T: num_traits::Float> Block<T> {
    pub(crate) fn new(dim: BlockDim, anchor: Anchor, cholesky: Cholesky<T>) -> Self {
        // What the wire has to be told, a builder can get wrong too; every consumer sums
        // these dims trusting that neither did.
        #[cfg(any(feature = "serde", test))]
        debug_assert_eq!(cholesky.validate_for_dim(dim), Ok(()));
        Self {
            dim,
            anchor,
            cholesky,
        }
    }
}

/// A block as a payload carries it — a `dim` nothing has held its `cholesky` to — in
/// [`Block`]'s own field order, which is what a positional reader needs.
#[cfg(any(feature = "serde", test))]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(deserialize = "T: serde::de::DeserializeOwned + num_traits::Float"))
)]
struct BlockData<T> {
    dim: BlockDim,
    anchor: Anchor,
    cholesky: Cholesky<T>,
}

/// Pinning the dim to the payload behind it is what lets every consumer sum block dims
/// without a check of its own.
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
