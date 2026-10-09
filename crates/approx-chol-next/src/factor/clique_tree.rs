//! One pivot's star eliminated by a sampled clique tree (GKS 2023, Algorithm 5): its column and fill.

mod cdf;
mod copies;
mod star;

use crate::Real;
use cdf::CdfSampler;
use cdf::Stream;
pub(super) use cdf::Streams;
pub(super) use copies::{Copies, Multi, Single};
use star::Star;

/// Fill edges `(lo, hi, weight)`, `lo < hi`, as an elimination emits them.
pub(super) type Fill<T> = Vec<(u32, u32, T)>;

/// A pivot's elimination, reused across pivots; AC and AC2 differ only in `C`.
pub(super) struct CliqueSampler<T, C: Copies> {
    star: Star<T, C>,
    draws: CdfSampler<T>,
    column: Column<T>,
    split: C::Split,
}

impl<T: Real, C: Copies> CliqueSampler<T, C> {
    /// `split` caps every merge, so one sampler cannot mix two.
    pub(super) fn new(stream: Stream, split: C::Split) -> Self {
        todo!()
    }

    /// Draws from `stream` next, so a star's tree does not depend on which task sampled it.
    pub(super) fn restart(&mut self, stream: Stream) {
        todo!()
    }

    /// Merges `neighbors`, samples their clique tree into `fill` as `w_i (S_{>i} / d)`, and returns the pivot.
    pub(super) fn eliminate(
        &mut self,
        neighbors: impl Iterator<Item = (u32, C, T)>,
        fill: &mut Fill<T>,
    ) -> T {
        todo!()
    }

    /// Copies each neighbor of the last pivot lost.
    pub(super) fn degree_losses(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.star.degree_losses()
    }

    pub(super) fn column(&self) -> &Column<T> {
        &self.column
    }
}

/// One column of the factor: shares of the original pivot, the remainder named, not weighted.
pub(super) struct Column<T> {
    shares: Vec<(u32, T)>,
    /// `None` keeps the whole pivot: an empty star hands nothing on.
    remainder: Option<u32>,
}

impl<T: Real> Column<T> {
    /// The shares and the neighbor that absorbs what they leave; nothing for an empty star.
    pub(super) fn shares(&self) -> Option<(&[(u32, T)], u32)> {
        Some((&self.shares, self.remainder?))
    }
}
