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

/// A pivot's elimination, reused across pivots; AC and AC2 differ only in `C`.
pub(super) struct CliqueSampler<T, I, C: Copies> {
    star: Star<T, I, C>,
    /// The star's weights' prefix sums, heaviest last, so each suffix's draw range is one subtraction.
    cumsum: Vec<T>,
    shares: Vec<(I, T)>,
    /// `None` keeps the whole pivot: an empty star hands nothing on.
    remainder: Option<I>,
    split: C::Split,
}

impl<T: Real, I: Index, C: Copies> CliqueSampler<T, I, C> {
    /// `split` caps every merge, so one sampler cannot mix two.
    pub(super) fn new(split: C::Split) -> Self {
        Self {
            star: Star::new(),
            cumsum: Vec::new(),
            shares: Vec::new(),
            remainder: None,
            split,
        }
    }

    /// Merges `neighbors`, samples their clique tree into `fill` as `w_i (S_{>i} / d)` from the sequential `draws`, and returns the pivot.
    pub(super) fn eliminate(
        &mut self,
        draws: &mut Xoshiro256PlusPlus,
        neighbors: impl Iterator<Item = (I, C, T)>,
        fill: &mut Fill<T, I>,
    ) -> T {
        self.star.refill(neighbors, self.split);
        self.sample(draws, fill);
        self.shares.clear();
        let Some((last, rest)) = self.star.entries().split_last() else {
            self.remainder = None;
            return T::zero();
        };
        let d = suffix_past(&self.cumsum, None);
        let shares = rest.iter().map(|entry| (entry.neighbor, entry.weight / d));
        self.shares.extend(shares);
        self.remainder = Some(last.neighbor);
        last.weight * (last.weight / d)
    }

    /// A neighborhood with no duplicate, as a row of `C` is, consuming its star's own `draws` so no task boundary moves one; every neighbor splits alike, so raw weights sort once; fill only, as the complement holds its pivots.
    pub(super) fn eliminate_unique(
        &mut self,
        mut rng: Xoshiro256PlusPlus,
        neighbors: impl Iterator<Item = (I, T)>,
        fill: &mut Fill<T, I>,
    ) {
        self.star.refill_unique(neighbors, C::initial(self.split));
        self.sample(&mut rng, fill);
    }

    /// Prepares the star's CDF, then fills `per_copy(w_i (S_{>i} / d))` per copy of each entry but the heaviest, each copy drawing past it.
    fn sample(&mut self, rng: &mut Xoshiro256PlusPlus, fill: &mut Fill<T, I>) {
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
                fill.push((entry.neighbor.min(other), entry.neighbor.max(other), weight));
            }
        }
    }

    /// Copies each neighbor of the last pivot lost.
    pub(super) fn degree_losses(&self) -> impl Iterator<Item = (I, u32)> + '_ {
        self.star.degree_losses()
    }

    /// The factor column the last pivot left: shares of the original pivot and the neighbor absorbing what they leave; nothing for an empty star.
    pub(super) fn column(&self) -> Option<(&[(I, T)], I)> {
        Some((&self.shares, self.remainder?))
    }
}
