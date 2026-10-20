//! One pivot's star eliminated by a sampled clique tree (GKS 2023, Algorithm 6; AC is its `k = l = 1`): its column and fill.

mod cdf;
mod copies;
mod star;

use rand::rngs::Xoshiro256PlusPlus;

use crate::{Index, Real};
use cdf::{draw_after, suffix_past};
pub(super) use copies::{Copies, Multi, Single};
use star::StarEntry;

/// Fill edges `(lo, hi, copies, weight)`, `lo < hi`, as an elimination emits them: one per entry and partner, its weight summed over its copies.
pub(super) type Fill<T, I, C> = Vec<(I, I, C, T)>;

/// A pivot's elimination, reused across pivots; AC and AC2 differ only in `C`.
pub(super) struct CliqueSampler<T, I, C: Copies> {
    /// A pivot's deduplicated neighborhood in clique-tree order.
    entries: Vec<StarEntry<T, I, C>>,
    /// What merging or capping collapsed, per neighbor.
    removed_copies: Vec<(I, u32)>,
    /// Grown to `n` on first need: above 32 entries, dedup scatters instead of sorting 16-byte entries twice.
    scatter: Vec<I>,
    /// Each merged entry's summed copies before the cap: `C` cannot add.
    counts: Vec<u32>,
    /// The star's weights' prefix sums, heaviest last, so each suffix's draw range is one subtraction.
    cumsum: Vec<T>,
    split: C,
    /// Scratch [`Copies::draw`] reuses across entries; one copy never touches it.
    tally: Vec<(usize, u32)>,
}

impl<T: Real, I: Index, C: Copies> CliqueSampler<T, I, C> {
    /// `split` caps every merge, so one sampler cannot mix two.
    pub(super) fn new(split: C) -> Self {
        Self {
            entries: Vec::new(),
            removed_copies: Vec::new(),
            scatter: Vec::new(),
            counts: Vec::new(),
            cumsum: Vec::new(),
            split,
            tally: Vec::new(),
        }
    }

    /// Samples the merged star's clique tree into `fill` as `w_i (S_{>i} / d)`; the last neighbor absorbs what the shares leave.
    pub(super) fn eliminate(
        &mut self,
        draws: &mut Xoshiro256PlusPlus,
        neighbors: impl Iterator<Item = (I, C, T)>,
        fill: &mut Fill<T, I, C>,
    ) -> (T, impl Iterator<Item = (I, T)> + '_, I) {
        self.refill(neighbors);
        self.sample(draws, |edge| fill.push(edge));
        let (last, rest) =
            (self.entries.split_last()).expect("a pivot of a connected graph has a neighbor");
        let d = suffix_past(&self.cumsum, None);
        let shares = rest
            .iter()
            .map(move |entry| (entry.neighbor, entry.weight / d));
        (last.weight, shares, last.neighbor)
    }

    /// A neighborhood with no duplicate, as a row of `C` is, consuming its star's own `draws` so no task boundary moves one; every neighbor splits alike, so raw weights sort once; fill only, as the complement holds its pivots.
    pub(super) fn eliminate_unique(
        &mut self,
        mut rng: Xoshiro256PlusPlus,
        neighbors: impl Iterator<Item = (I, T)>,
        fill: impl FnMut((I, I, C, T)),
    ) {
        self.refill_unique(neighbors);
        self.sample(&mut rng, fill);
    }

    /// Prepares the star's CDF, then fills `per_copy(w_i (S_{>i} / d))` per copy of each entry but the heaviest, each copy drawing past it.
    fn sample(&mut self, rng: &mut Xoshiro256PlusPlus, mut fill: impl FnMut((I, I, C, T))) {
        let entries = &self.entries;
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
            // Clamped, never dropped: the clique tree keeps the star's neighbors connected, and every weight's inverse finite.
            let weight = (entry.copies)
                .per_copy(entry.weight * (suffix_past(&self.cumsum, Some(i)) / d))
                .max(T::min_positive_value());
            let draw = || draw_after(&self.cumsum, rng, i);
            for (at, copies, weight) in (entry.copies).draw(&mut self.tally, weight, draw) {
                let other = entries[at].neighbor;
                fill((
                    entry.neighbor.min(other),
                    entry.neighbor.max(other),
                    copies,
                    weight,
                ));
            }
        }
    }
}
