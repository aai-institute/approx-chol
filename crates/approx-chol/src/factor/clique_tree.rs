//! One pivot's star eliminated by a sampled clique tree (GKS 2023, Algorithm 5): its column and fill.

mod cdf;
mod copies;
mod star;

use rand::rngs::Xoshiro256PlusPlus;

use crate::{Index, Real};
use cdf::{draw_after, suffix_past};
pub(super) use copies::{Copies, Multi, Single};
use star::Star;

/// Fill edges `(lo, hi, weight)`, `lo < hi`, as an elimination emits them.
pub(super) type Fill<T, I> = Vec<(I, I, T)>;

/// A pivot's column: its heaviest weight, its other neighbors' shares, and the heaviest neighbor, which absorbs what they leave.
pub(super) type PivotColumn<'a, T, I> = (T, &'a [(I, T)], I);

/// A pivot's elimination, reused across pivots; AC and AC2 differ only in `C`.
pub(super) struct CliqueSampler<T, I, C: Copies> {
    star: Star<T, I, C>,
    /// The star's weights' prefix sums, heaviest last, so each suffix's draw range is one subtraction.
    cumsum: Vec<T>,
    shares: Vec<(I, T)>,
    split: C::Split,
}

impl<T: Real, I: Index, C: Copies> CliqueSampler<T, I, C> {
    /// `split` caps every merge, so one sampler cannot mix two.
    pub(super) fn new(split: C::Split) -> Self {
        Self {
            star: Star::new(),
            cumsum: Vec::new(),
            shares: Vec::new(),
            split,
        }
    }

    /// Merges `neighbors`, samples their clique tree into `fill` as `w_i (S_{>i} / d)` from the sequential `draws`, and returns the pivot's column.
    pub(super) fn eliminate(
        &mut self,
        draws: &mut Xoshiro256PlusPlus,
        neighbors: impl Iterator<Item = (I, C, T)>,
        fill: &mut Fill<T, I>,
    ) -> PivotColumn<'_, T, I> {
        self.star.refill(neighbors, self.split);
        self.sample(draws, |edge| fill.push(edge));
        self.shares.clear();
        let (last, rest) = (self.star.entries().split_last())
            .expect("a pivot of a connected graph has a neighbor");
        let d = suffix_past(&self.cumsum, None);
        let shares = rest.iter().map(|entry| (entry.neighbor, entry.weight / d));
        self.shares.extend(shares);
        (last.weight, &self.shares, last.neighbor)
    }

    /// A neighborhood with no duplicate, as a row of `C` is, consuming its star's own `draws` so no task boundary moves one; every neighbor splits alike, so raw weights sort once; fill only, as the complement holds its pivots.
    pub(super) fn eliminate_unique(
        &mut self,
        mut rng: Xoshiro256PlusPlus,
        neighbors: impl Iterator<Item = (I, T)>,
        fill: impl FnMut((I, I, T)),
    ) {
        self.star.refill_unique(neighbors, C::initial(self.split));
        self.sample(&mut rng, fill);
    }

    /// Prepares the star's CDF, then fills `per_copy(w_i (S_{>i} / d))` per copy of each entry but the heaviest, each copy drawing past it.
    fn sample(&mut self, rng: &mut Xoshiro256PlusPlus, mut fill: impl FnMut((I, I, T))) {
        let entries = self.star.entries();
        let Some((_, rest)) = entries.split_last() else {
            return;
        };
        self.cumsum.clear();
        let mut acc = T::zero();
        for entry in entries {
            acc = acc + entry.weight;
            self.cumsum.push(acc);
        }
        let d = suffix_past(&self.cumsum, None);
        debug_assert!(
            d.is_finite() && d > T::zero(),
            "positive weights under the diagonal bound"
        );
        for (i, entry) in rest.iter().enumerate() {
            let weight =
                (entry.copies).per_copy(entry.weight * (suffix_past(&self.cumsum, Some(i)) / d));
            // An underflowed share couples nothing, so it spends no draw.
            if weight == T::zero() {
                continue;
            }
            for _ in 0..entry.copies.get() {
                let other = entries[draw_after(&self.cumsum, rng, i)].neighbor;
                fill((entry.neighbor.min(other), entry.neighbor.max(other), weight));
            }
        }
    }

    /// Copies each neighbor of the last pivot lost.
    pub(super) fn degree_losses(&self) -> impl Iterator<Item = (I, u32)> + '_ {
        self.star.degree_losses()
    }
}
