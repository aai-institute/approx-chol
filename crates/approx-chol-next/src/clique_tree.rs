//! One pivot's star eliminated by a sampled clique tree (GKS 2023, Algorithm 5): its column and fill.

mod cdf;
mod copies;
mod star;

pub(crate) use cdf::{Stream, Streams};
pub(crate) use copies::{Copies, Multi, Single};
pub(crate) use star::Star;

use crate::storage::Fill;
use crate::Real;
use cdf::CdfSampler;

/// A pivot's elimination, reused across pivots; AC and AC2 differ only in `C`.
pub(crate) struct CliqueSampler<T, C: Copies> {
    star: Star<T, C>,
    draws: CdfSampler<T>,
    column: Column<T>,
    factor: C::Factor,
}

impl<T: Real, C: Copies> CliqueSampler<T, C> {
    /// `factor` caps every merge, so one sampler cannot mix two.
    pub(crate) fn new(stream: Stream, factor: C::Factor) -> Self {
        todo!()
    }

    /// Draws from `stream` next, so a star's tree does not depend on which task sampled it.
    pub(crate) fn restart(&mut self, stream: Stream) {
        todo!()
    }

    /// Merges `neighbors`, samples their clique tree into `fill`, and returns the pivot.
    pub(crate) fn eliminate(
        &mut self,
        neighbors: impl Iterator<Item = (u32, C, T)>,
        fill: &mut Fill<T>,
    ) -> T {
        todo!()
    }

    /// The last pivot's star.
    pub(crate) fn star(&self) -> &Star<T, C> {
        &self.star
    }

    pub(crate) fn column(&self) -> &Column<T> {
        &self.column
    }
}

/// One column of the factor: shares of the original pivot, the remainder named, not weighted.
pub(crate) struct Column<T> {
    shares: Vec<(u32, T)>,
    /// `None` keeps the whole pivot: an empty star hands nothing on.
    remainder: Option<u32>,
}

impl<T: Real> Column<T> {
    /// The shares and the neighbor that absorbs what they leave; nothing for an empty star.
    pub(crate) fn shares(&self) -> Option<(&[(u32, T)], u32)> {
        Some((&self.shares, self.remainder?))
    }
}
