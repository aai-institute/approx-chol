use core::cmp::Ordering;

use super::copies::Copies;
use super::CliqueTreeSampler;
use crate::factor::sign::Sign;
use crate::{Index, Real};

/// Copies stored as the graph stores them, so a single-copy star pays no count or division.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct StarEntry<T, I, C> {
    pub(super) neighbor: I,
    pub(super) copies: C,
    pub(super) weight: T,
}

/// Total on a deduped star, which keeps the clique tree off `sort_unstable`'s width heuristics.
fn by_weight_then_neighbor<T: Real, I: Index, C, S: Sign<T, I>>(
    sign: S,
    a: &StarEntry<T, I, C>,
    b: &StarEntry<T, I, C>,
) -> Ordering {
    (sign.magnitude(a.weight))
        .total_cmp(&sign.magnitude(b.weight))
        .then_with(|| a.neighbor.cmp(&b.neighbor))
}

/// A function of each entry, so still a total order; `Single`'s identity makes it the raw sort.
fn by_per_copy_then_neighbor<T: Real, I: Index, C: Copies, S: Sign<T, I>>(
    sign: S,
    a: &StarEntry<T, I, C>,
    b: &StarEntry<T, I, C>,
) -> Ordering {
    let (x, y) = (
        a.copies.per_copy(sign.magnitude(a.weight)),
        b.copies.per_copy(sign.magnitude(b.weight)),
    );
    x.total_cmp(&y).then_with(|| a.neighbor.cmp(&b.neighbor))
}

/// Searching the merged prefix beats the scatter path's random access up to here.
const SEARCH_LIMIT: usize = 8;

/// Rests in every slot of [`CliqueTreeSampler::scatter`] between merges.
fn unseen<I: Index>() -> I {
    I::max_value()
}

impl<T: Real, I: Index, C: Copies> CliqueTreeSampler<T, I, C> {
    /// Merges duplicate neighbors under the sampler's `split` cap, grounding what each merge cancels at both its ends, the neighbor's into `fill` and the pivot's into its star, and drops a neighbor cancelled to zero, which then couples nothing but still moved; then sorts by per-copy magnitude.
    pub(super) fn refill<S: Sign<T, I>>(
        &mut self,
        neighbors: impl Iterator<Item = (I, C, T)>,
        sign: S,
        mut fill: impl FnMut((I, I, C, T)),
    ) {
        self.load(neighbors);
        let (split, mut cancelled) = (self.split, T::zero());
        let mut grounded = |neighbor: I, gain: T| {
            if let Some(ground) = sign.ground().filter(|_| gain != T::zero()) {
                cancelled = cancelled + gain;
                fill((neighbor, ground, split, gain));
            }
        };
        let merged = match self.entries.len() <= SEARCH_LIMIT {
            true => self.merge_by_search(sign, &mut grounded),
            false => self.merge_by_scatter(sign, &mut grounded),
        };
        self.entries.truncate(merged);
        // Under any sign: a task's next block may be unsigned, and its order holds none of these.
        self.dropped.clear();
        if let Some(ground) = sign.ground().filter(|_| cancelled != T::zero()) {
            self.ground(ground, cancelled);
            let dropped = &mut self.dropped;
            (self.entries).retain(|entry| {
                let coupled = entry.weight != T::zero();
                if !coupled {
                    dropped.push(entry.neighbor);
                }
                coupled
            });
        }
        (self.entries).sort_unstable_by(|a, b| by_per_copy_then_neighbor(sign, a, b));
    }

    fn load(&mut self, neighbors: impl Iterator<Item = (I, C, T)>) {
        self.entries.clear();
        let entries = neighbors.map(|(neighbor, copies, weight)| StarEntry {
            neighbor,
            copies,
            weight,
        });
        self.entries.extend(entries);
    }

    /// First-seen order, each entry searched in the merged prefix, `cancelled` told each merge's neighbor and gain; returns the merged count.
    fn merge_by_search<S: Sign<T, I>>(
        &mut self,
        sign: S,
        mut cancelled: impl FnMut(I, T),
    ) -> usize {
        let mut write = 0;
        for read in 0..self.entries.len() {
            let entry = self.entries[read];
            let seen =
                (self.entries[..write].iter()).position(|merged| merged.neighbor == entry.neighbor);
            match seen {
                Some(position) => cancelled(entry.neighbor, self.absorb(sign, position, entry)),
                None => {
                    self.entries[write] = entry;
                    write += 1;
                }
            }
        }
        write
    }

    /// First-seen order, each slot back at [`unseen`] on return, `cancelled` told each merge's neighbor and gain; returns the merged count.
    fn merge_by_scatter<S: Sign<T, I>>(
        &mut self,
        sign: S,
        mut cancelled: impl FnMut(I, T),
    ) -> usize {
        let mut write = 0;
        for read in 0..self.entries.len() {
            let entry = self.entries[read];
            let slot = entry.neighbor.index();
            if slot >= self.scatter.len() {
                self.scatter.resize(slot + 1, unseen());
            }
            match self.scatter[slot] {
                position if position == unseen() => {
                    self.scatter[slot] = I::from_index(write);
                    self.entries[write] = entry;
                    write += 1;
                }
                position => {
                    cancelled(entry.neighbor, self.absorb(sign, position.index(), entry));
                }
            }
        }
        for entry in &self.entries[..write] {
            self.scatter[entry.neighbor.index()] = unseen();
        }
        write
    }

    /// Capped as the sum grows, which keeps the cap of the whole sum: no input exceeds the split; returns the magnitude the sum cancelled.
    fn absorb<S: Sign<T, I>>(
        &mut self,
        sign: S,
        position: usize,
        duplicate: StarEntry<T, I, C>,
    ) -> T {
        let merged = &mut self.entries[position];
        let (weight, gain) = sign.merge(merged.weight, duplicate.weight);
        merged.weight = weight;
        let copies = merged.copies.get().saturating_add(duplicate.copies.get());
        merged.copies = self.split.cap(copies);
        gain
    }

    /// `cancelled` onto the star's `ground` entry, added where the pivot has none.
    fn ground(&mut self, ground: I, cancelled: T) {
        match self
            .entries
            .iter_mut()
            .find(|entry| entry.neighbor == ground)
        {
            Some(entry) => entry.weight = entry.weight + cancelled,
            None => self.entries.push(StarEntry {
                neighbor: ground,
                copies: self.split,
                weight: cancelled,
            }),
        }
    }

    /// Neighbors already unique, each at the split's copies, which keeps per-copy order the raw order: one sort, no scratch.
    pub(super) fn refill_unique<S: Sign<T, I>>(
        &mut self,
        neighbors: impl Iterator<Item = (I, T)>,
        sign: S,
    ) {
        let copies = self.split;
        self.entries.clear();
        self.dropped.clear();
        let entries = neighbors.map(|(neighbor, weight)| StarEntry {
            neighbor,
            copies,
            weight,
        });
        self.entries.extend(entries);
        (self.entries).sort_unstable_by(|a, b| by_weight_then_neighbor(sign, a, b));
    }

    /// The last star's neighbors in clique-tree order; with those it [`dropped`](Self::dropped), every vertex its pivot's elimination moved.
    pub(in crate::factor) fn neighbors(&self) -> impl Iterator<Item = I> + '_ {
        self.entries.iter().map(|entry| entry.neighbor)
    }

    /// The last star's neighbors its merges cancelled to zero, empty unless signed.
    pub(in crate::factor) fn dropped(&self) -> impl Iterator<Item = I> + '_ {
        self.dropped.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::super::{CliqueTreeSampler, Copies, Multi, Single};
    use crate::factor::sign::{Signed, Unsigned};

    /// Merged with `merge` under a split of `k`, as `(neighbor, copies, weight bits)`.
    fn merged(
        k: u32,
        star: &[(u32, u32, f32)],
        merge: fn(&mut CliqueTreeSampler<f32, u32, Multi>) -> usize,
    ) -> Vec<(u32, u64, u32)> {
        let split = Multi::split(k).expect("AC2");
        let mut sampler = CliqueTreeSampler::new(split);
        sampler.load(
            (star.iter()).map(|&(neighbor, copies, weight)| (neighbor, split.cap(copies), weight)),
        );
        let merged = merge(&mut sampler);
        (sampler.entries[..merged].iter())
            .map(|e| (e.neighbor, u64::from(e.copies.get()), e.weight.to_bits()))
            .collect()
    }

    /// First-seen order, weights summed in input order, copies summed whole in `u64`, then capped at `k`.
    fn model(k: u32, star: &[(u32, u32, f32)]) -> Vec<(u32, u64, u32)> {
        let mut merged: Vec<(u32, u64, f32)> = Vec::new();
        for &(neighbor, copies, weight) in star {
            match merged.iter_mut().find(|seen| seen.0 == neighbor) {
                Some(seen) => (seen.1, seen.2) = (seen.1 + u64::from(copies), seen.2 + weight),
                None => merged.push((neighbor, u64::from(copies), weight)),
            }
        }
        (merged.into_iter())
            .map(|(neighbor, copies, weight)| {
                (neighbor, copies.min(u64::from(k)), weight.to_bits())
            })
            .collect()
    }

    proptest! {
        /// f32, where three duplicates' sum can depend on their order; copies within the split, as the graph keeps them, summing past it and, at 2^31, past `u32`.
        #[test]
        fn merges_follow_the_model(
            k in prop_oneof![Just(4u32), Just(1 << 31)],
            star in vec((0u32..6, prop_oneof![1u32..6, (1u32 << 31) - 2..=1 << 31], 0.1f32..10.0), 0..40),
        ) {
            let star: Vec<_> = (star.into_iter()).map(|(neighbor, copies, weight)| (neighbor, copies.min(k), weight)).collect();
            prop_assert_eq!(merged(k, &star, |s| s.merge_by_search(Unsigned, |_, _| {})), model(k, &star));
            prop_assert_eq!(merged(k, &star, |s| s.merge_by_scatter(Unsigned, |_, _| {})), model(k, &star));
        }
    }

    #[test]
    fn a_cancelling_merge_grounds_its_gain_at_both_ends() {
        let star = [(1, 2.0), (2, -1.0), (1, -0.5)];
        for (ground, grounded) in [(Some(0.25), 1.25), (None, 1.0)] {
            let mut sampler = CliqueTreeSampler::<f64, u32, Single>::new(Single);
            let mut fill = Vec::new();
            let neighbors = star.into_iter().chain(ground.map(|g| (9, g)));
            let neighbors = neighbors.map(|(j, w)| (j, Single, w));
            sampler.refill(neighbors, Signed { ground: 9 }, |e| fill.push(e));
            let mut entries: Vec<_> = (sampler.entries.iter())
                .map(|e| (e.neighbor, e.weight))
                .collect();
            entries.sort_by_key(|&(j, _)| j);
            assert_eq!(entries, [(1, 1.5), (2, -1.0), (9, grounded)]);
            let fill: Vec<_> = fill.iter().map(|&(lo, hi, _, w)| (lo, hi, w)).collect();
            assert_eq!(fill, [(1, 9, 1.0)]);
        }
    }

    /// Opposite duplicates summing to zero leave the star, grounded, and still reach the order's rekey, until the next star.
    #[test]
    fn an_exact_cancellation_drops_the_neighbor_but_rekeys_it() {
        let sign = Signed { ground: 9 };
        let mut sampler = CliqueTreeSampler::<f64, u32, Single>::new(Single);
        let mut fill = Vec::new();
        let star = [(1, 1.0), (2, -1.5), (1, -1.0)].map(|(j, w)| (j, Single, w));
        sampler.refill(star.into_iter(), sign, |edge| fill.push(edge));
        let mut entries: Vec<_> = (sampler.entries.iter())
            .map(|e| (e.neighbor, e.weight))
            .collect();
        entries.sort_by_key(|&(j, _)| j);
        assert_eq!(entries, [(2, -1.5), (9, 2.0)]);
        let fill: Vec<_> = fill.iter().map(|&(lo, hi, _, w)| (lo, hi, w)).collect();
        assert_eq!(fill, [(1, 9, 2.0)]);
        assert_eq!(sampler.dropped().collect::<Vec<_>>(), [1]);
        sampler.refill([(3, Single, 1.0)].into_iter(), Unsigned, |_| {});
        assert_eq!(sampler.dropped().count(), 0, "the next star's alone");
    }
}
