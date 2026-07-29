//! One block of the factorization: an [`Anchor`] paired with a [`Cholesky`].
//!
//! The two are orthogonal and independently chosen — the anchor by augmentation, the
//! Cholesky by policy — so all four combinations occur.

use super::anchor::Anchor;
use super::cholesky::Cholesky;
use crate::types::Real;
use core::num::NonZeroUsize;

/// A block's dimension, in the two forms its consumers actually ask for. The
/// pinned variable is the block's last, so the routing bound, the dense factor and
/// the step count all want [`solved`](Self::solved) rather than the total — and
/// none of them has to spell the subtraction to get it.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockDim(NonZeroUsize);

impl BlockDim {
    /// `None` for a block of no variables: it has nothing to pin, and every
    /// dimension derived from it would underflow.
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
    serde(bound(
        serialize = "T: serde::Serialize",
        deserialize = "T: serde::de::DeserializeOwned"
    ))
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

    pub(super) fn n_steps(&self) -> usize {
        self.cholesky.n_steps()
    }
}

impl<T: Real> Block<T> {
    /// `canonical` picks the zero-mean representative for a floating block; without
    /// it the block is left anchored at zero in its pinned variable.
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
