//! One pivot's star eliminated by a sampled clique tree (GKS 2023, Algorithm 6; AC is its `k = l = 1`): its column and fill.

mod cdf;
mod copies;
mod star;

use rand::rngs::Xoshiro256PlusPlus;

use super::sign::Sign;
use crate::{narrow, widen, Index, Real};
use cdf::{draw_after, suffix_past};
pub(super) use copies::{Copies, Degrees, Multi, Single};
use star::StarEntry;

/// Fill edges `(lo, hi, copies, weight)`, `lo < hi`, as an elimination emits them: one per entry and partner, its weight summed over its copies.
pub(super) type Fill<T, I, C> = Vec<(I, I, C, T)>;

/// A pivot's elimination, reused across pivots; AC and AC2 differ only in `C`.
pub(super) struct CliqueTreeSampler<T, I, C: Copies> {
    /// A pivot's deduplicated neighborhood in clique-tree order.
    entries: Vec<StarEntry<T, I, C>>,
    /// Grown on first need to the highest neighbor a star has scattered.
    scatter: Vec<I>,
    /// The star's weights' prefix sums, heaviest last, so each suffix's draw range is one subtraction; `f64`, as `f32` loses short suffixes (#275).
    cumsum: Vec<f64>,
    split: C,
    /// Scratch [`Copies::draw`] reuses across entries; one copy never touches it.
    tally: Vec<(usize, u32)>,
}

impl<T: Real, I: Index, C: Copies> CliqueTreeSampler<T, I, C> {
    /// `split` caps every merge, so one sampler cannot mix two.
    pub(super) fn new(split: C) -> Self {
        Self {
            entries: Vec::new(),
            scatter: Vec::new(),
            cumsum: Vec::new(),
            split,
            tally: Vec::new(),
        }
    }

    /// Samples the merged star's clique tree into `fill` as `w_i (S_{>i} / d)`, after what its merges cancelled; returns `d` and each neighbor's share `w/d`.
    pub(super) fn eliminate<S: Sign<T, I>>(
        &mut self,
        draws: &mut Xoshiro256PlusPlus,
        neighbors: impl Iterator<Item = (I, C, T)>,
        fill: &mut Fill<T, I, C>,
        sign: S,
    ) -> (f64, impl Iterator<Item = (I, f64)> + '_) {
        self.refill(neighbors, sign, |edge| fill.push(edge));
        let d = (self.sample(draws, |edge| fill.push(edge), sign))
            .expect("a pivot of a connected graph has a neighbor");
        let shares =
            (self.entries.iter()).map(move |entry| (entry.neighbor, widen(entry.weight) / d));
        (d, shares)
    }

    /// A neighborhood with no duplicate, as a row of `C` is, consuming its star's own `draws` so no task boundary moves one; every neighbor splits alike, so raw weights sort once; fill only, as the complement holds its pivots.
    pub(super) fn eliminate_unique<S: Sign<T, I>>(
        &mut self,
        mut rng: Xoshiro256PlusPlus,
        neighbors: impl Iterator<Item = (I, T)>,
        fill: impl FnMut((I, I, C, T)),
        sign: S,
    ) {
        self.refill_unique(neighbors, sign);
        self.sample(&mut rng, fill, sign);
    }

    /// Prepares the star's CDF, then fills `per_copy(w_i (S_{>i} / d))` per copy of each entry but the heaviest, each copy drawing past it; `d`, unless the star is empty.
    fn sample<S: Sign<T, I>>(
        &mut self,
        rng: &mut Xoshiro256PlusPlus,
        mut fill: impl FnMut((I, I, C, T)),
        sign: S,
    ) -> Option<f64> {
        let entries = &self.entries;
        let (_, rest) = entries.split_last()?;
        self.cumsum.clear();
        let mut acc = 0.0;
        for entry in entries {
            acc += widen(sign.magnitude(entry.weight));
            self.cumsum.push(acc);
        }
        let d = suffix_past(&self.cumsum, None);
        debug_assert!(
            d.is_finite() && d > 0.0,
            "positive weights under the diagonal bound"
        );
        for (i, entry) in rest.iter().enumerate() {
            let magnitude = widen(sign.magnitude(entry.weight));
            let total: T = narrow(magnitude * (suffix_past(&self.cumsum, Some(i)) / d));
            // Clamped, never dropped: the clique tree keeps the star's neighbors connected, and every weight's inverse finite.
            let weight = (entry.copies).per_copy(total).max(T::min_positive_value());
            let draw = || draw_after(&self.cumsum, rng, i);
            for (at, copies, weight) in (entry.copies).draw(&mut self.tally, weight, draw) {
                let other = entries[at];
                let (lo, hi) = (
                    entry.neighbor.min(other.neighbor),
                    entry.neighbor.max(other.neighbor),
                );
                fill((
                    lo,
                    hi,
                    copies,
                    sign.fill(entry.weight, other.weight, weight, hi),
                ));
            }
        }
        Some(d)
    }
}
