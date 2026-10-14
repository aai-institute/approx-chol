//! One pivot's star eliminated by a sampled clique tree (GKS 2023, Algorithm 5): its column and fill.

mod cdf;
mod copies;
mod star;
mod streams;

use rand::rngs::SmallRng;

use crate::Real;
use cdf::CdfSampler;
pub(super) use copies::{Copies, Multi, Single};
use star::Star;
pub(super) use streams::Streams;

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
        Self {
            star: Star::new(),
            cdf: CdfSampler::new(),
            column: PivotColumn {
                shares: Vec::new(),
                remainder: None,
            },
            split,
        }
    }

    /// Merges `neighbors`, samples their clique tree into `fill` as `w_i (S_{>i} / d)` from the sequential `draws`, and returns the pivot.
    pub(super) fn eliminate(
        &mut self,
        draws: &mut SmallRng,
        neighbors: impl Iterator<Item = (u32, C, T)>,
        fill: &mut Fill<T>,
    ) -> T {
        self.star.refill(neighbors, self.split);
        self.sample(draws, fill);
        let column = &mut self.column;
        column.shares.clear();
        let Some((last, rest)) = self.star.entries().split_last() else {
            column.remainder = None;
            return T::zero();
        };
        let d = self.cdf.suffix_past(None);
        let shares = rest.iter().map(|entry| (entry.neighbor, entry.weight / d));
        column.shares.extend(shares);
        column.remainder = Some(last.neighbor);
        last.weight * (last.weight / d)
    }

    /// A neighborhood with no duplicate, as a row of `C` is, consuming its star's own `draws` so no task boundary moves one; every neighbor splits alike, so raw weights sort once; fill only, as the complement holds its pivots.
    pub(super) fn eliminate_unique(
        &mut self,
        mut rng: SmallRng,
        neighbors: impl Iterator<Item = (u32, T)>,
        fill: &mut Fill<T>,
    ) {
        self.star.refill_unique(neighbors, C::initial(self.split));
        self.sample(&mut rng, fill);
    }

    /// Prepares the star's CDF, then fills `per_copy(w_i (S_{>i} / d))` per copy of each entry but the heaviest, each copy drawing past it.
    fn sample(&mut self, rng: &mut SmallRng, fill: &mut Fill<T>) {
        let entries = self.star.entries();
        let Some((_, rest)) = entries.split_last() else {
            return;
        };
        self.cdf.prepare(entries.iter().map(|entry| entry.weight));
        let d = self.cdf.suffix_past(None);
        debug_assert!(
            d.is_finite() && d > T::zero(),
            "positive weights under the diagonal bound"
        );
        for (i, entry) in rest.iter().enumerate() {
            let weight =
                (entry.copies).per_copy(entry.weight * (self.cdf.suffix_past(Some(i)) / d));
            // An underflowed share couples nothing, so it spends no draw.
            if weight == T::zero() {
                continue;
            }
            for _ in 0..entry.copies.get() {
                let other = entries[self.cdf.draw_after(rng, i)].neighbor;
                fill.push((entry.neighbor.min(other), entry.neighbor.max(other), weight));
            }
        }
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
