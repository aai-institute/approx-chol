//! One pivot's star eliminated by a sampled clique tree (GKS 2023, Algorithm 5): its column and fill.

mod cdf;
mod copies;
mod star;
mod streams;

use crate::Real;
use cdf::CdfSampler;
pub(super) use copies::{Copies, Multi, Single};
use star::Star;
pub(super) use streams::{SequentialDraws, StarDraws, StarStreams, Streams};

/// Fill edges `(lo, hi, weight)`, `lo < hi`, as an elimination emits them.
pub(super) type Fill<T> = Vec<(u32, u32, T)>;

/// A pivot's elimination, reused across pivots; AC and AC2 differ only in `C`.
pub(super) struct CliqueSampler<T, C: Copies> {
    star: Star<T, C>,
    cdf: CdfSampler<T>,
    column: PivotColumn<T>,
    split: C::Split,
}

impl<T: Real, C: Copies> CliqueSampler<T, C> {
    /// `split` caps every merge, so one sampler cannot mix two.
    pub(super) fn new(split: C::Split) -> Self {
        todo!()
    }

    /// Merges `neighbors`, samples their clique tree into `fill` as `w_i (S_{>i} / d)` from the run's `draws`, and returns the pivot.
    pub(super) fn eliminate(
        &mut self,
        draws: &mut SequentialDraws,
        neighbors: impl Iterator<Item = (u32, C, T)>,
        fill: &mut Fill<T>,
    ) -> T {
        todo!()
    }

    /// A neighborhood with no duplicate, as a row of `C` is, consuming its star's own `draws` so no task boundary moves one; every neighbor splits alike, so raw weights sort once.
    pub(super) fn eliminate_unique(
        &mut self,
        draws: StarDraws,
        neighbors: impl Iterator<Item = (u32, T)>,
        fill: &mut Fill<T>,
    ) -> T {
        todo!()
    }

    /// Copies each neighbor of the last pivot lost.
    pub(super) fn degree_losses(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.star.degree_losses()
    }

    pub(super) fn column(&self) -> &PivotColumn<T> {
        &self.column
    }
}

/// The factor column a pivot leaves: shares of the original pivot, the remainder named, not weighted.
pub(super) struct PivotColumn<T> {
    shares: Vec<(u32, T)>,
    /// `None` keeps the whole pivot: an empty star hands nothing on.
    remainder: Option<u32>,
}

impl<T: Real> PivotColumn<T> {
    /// The shares and the neighbor that absorbs what they leave; nothing for an empty star.
    pub(super) fn shares(&self) -> Option<(&[(u32, T)], u32)> {
        Some((&self.shares, self.remainder?))
    }
}
