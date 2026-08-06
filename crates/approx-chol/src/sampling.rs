use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::types::Real;
use num_traits::NumCast;

/// At or below this range size a linear CDF scan beats binary search.
const LINEAR_THRESHOLD: usize = 32;

/// The canonical 53-bit draw, in `[0, 1)` by construction: `bits >> 11` is at most
/// `2^53 - 1` and the scale is a power of two, so the product is exact and no rounding
/// lands on 1.0. Taking bits rather than an rng keeps the mapping testable at the
/// extremes the generator reaches only by chance.
#[inline]
fn draw_from(bits: u64) -> f64 {
    ((bits >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
}

/// A star's neighbors as a weighted distribution, drawn from by suffix. Owning
/// `neighbors` rather than borrowing leaves one length in play, so a draw answers
/// with a neighbor rather than an offset the caller re-indexes.
pub(crate) struct CdfSampler<T = f64> {
    neighbors: Vec<u32>,
    cumsum: Vec<T>,
    seed: u64,
    rng: SmallRng,
}

impl<T> CdfSampler<T> {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            neighbors: Vec::new(),
            cumsum: Vec::new(),
            seed,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// A stream per block off the one seed, keeping the scratch. `seed_from_u64`
    /// decorrelates neighboring values, so `block` needs no hashing.
    pub(crate) fn restart(&mut self, block: u64) {
        self.rng = SmallRng::seed_from_u64(self.seed.wrapping_add(block));
    }

    pub(crate) fn seed(&self) -> u64 {
        self.seed
    }

    /// `next_u64` for rand 0.9/0.10 compatibility.
    #[inline]
    fn draw(&mut self) -> f64 {
        draw_from(self.rng.next_u64())
    }
}

impl<T: Real> CdfSampler<T> {
    /// Takes the pairs rather than whatever holds them, so the sampler stays
    /// independent of how a caller stores a weighted neighborhood.
    #[inline]
    pub(crate) fn prepare(&mut self, entries: impl IntoIterator<Item = (u32, T)>) {
        self.neighbors.clear();
        self.cumsum.clear();
        let mut acc = T::zero();
        for (neighbor, w) in entries {
            acc = acc + w;
            self.neighbors.push(neighbor);
            self.cumsum.push(acc);
        }
    }

    /// The end is the prepared distribution's own length, so no caller can name a
    /// stale one.
    #[inline]
    pub(crate) fn sample_after(&mut self, start: usize) -> Option<u32> {
        let index = self.sample_suffix(start)?;
        Some(self.neighbors[index])
    }

    #[inline]
    fn sample_suffix(&mut self, start: usize) -> Option<usize> {
        let u = self.draw();
        self.index_for_draw(start, u)
    }

    #[inline]
    fn index_for_draw(&self, start: usize, u: f64) -> Option<usize> {
        let end = self.cumsum.len();
        if start >= end {
            return None;
        }

        let base = if start > 0 {
            self.cumsum[start - 1]
        } else {
            T::zero()
        };
        let remaining = self.cumsum[end - 1] - base;
        // A positive interval is all a draw needs; a floor above zero would refuse to
        // sample a suffix whose mass is small only because the input's scale is.
        if remaining <= T::zero() {
            return None;
        }

        let r = <T as NumCast>::from(u)? * remaining + base;

        let k = if end - start <= LINEAR_THRESHOLD {
            let mut k = start;
            while k < end && self.cumsum[k] < r {
                let previous = k;
                k += 1;
                // A broken advance leaves this condition re-checking the same index
                // forever instead of failing, so a bad increment hangs rather than
                // panics.
                debug_assert!(k > previous, "sample_suffix linear scan failed to advance");
            }
            k
        } else {
            self.cumsum[start..end].partition_point(|&c| c < r) + start
        };
        // `u < 1.0` does not survive narrowing to `f32`, which rounds every draw above
        // `1 - 2^-25` to exactly 1.0 and leaves `fl(remaining + base) > cumsum[end - 1]`
        // reachable. At `f64` the draw's own bound already rules that out.
        Some(k.min(end - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 42;

    /// Every fixture names entry `i` neighbor `i`, so a draw indexes its own tally.
    fn sample_counts(entries: &[(u32, f64)], start: usize, n_samples: usize) -> Vec<u32> {
        let mut sampler = CdfSampler::new(SEED);
        sampler.prepare(entries.iter().copied());
        let mut counts = vec![0u32; entries.len()];
        for _ in 0..n_samples {
            if let Some(neighbor) = sampler.sample_after(start) {
                counts[neighbor as usize] += 1;
            }
        }
        counts
    }

    /// Three entries stay under [`LINEAR_THRESHOLD`], fifty do not. Critical values
    /// are chi-squared at `p = 0.001` for `len - 1` degrees of freedom.
    #[test]
    fn draws_follow_the_weights_on_both_search_arms() {
        let skewed: Vec<(u32, f64)> = vec![(0, 1.0), (1, 2.0), (2, 7.0)];
        let uniform: Vec<(u32, f64)> = (0..50).map(|i| (i, 1.0)).collect();

        for (label, entries, critical) in [("linear", skewed, 13.82), ("binary", uniform, 85.35)] {
            let n_samples = 50_000;
            let counts = sample_counts(&entries, 0, n_samples);
            let total: f64 = entries.iter().map(|&(_, w)| w).sum();

            let chi2: f64 = counts
                .iter()
                .zip(&entries)
                .map(|(&observed, &(_, weight))| {
                    let expected = weight / total * n_samples as f64;
                    (observed as f64 - expected).powi(2) / expected
                })
                .sum();
            assert!(
                chi2 < critical,
                "{label}: chi-squared {chi2:.2} exceeds {critical}; counts = {counts:?}"
            );
        }
    }

    /// A suffix draw is uniform over `[base, total)`, so the prefix's mass must leave
    /// the interval's width untouched. `monotonic_suffix` sweeps every start but only
    /// judges range and reachability, both of which a mis-sized interval still
    /// satisfies — it just aims the draws at the wrong end of the suffix.
    #[test]
    fn a_heavy_prefix_does_not_reshape_the_suffix() {
        let entries: Vec<(u32, f64)> = vec![(0, 100.0), (1, 1.0), (2, 2.0), (3, 7.0)];
        let n_samples = 50_000;
        let counts = sample_counts(&entries, 1, n_samples);

        let suffix = &entries[1..];
        let total: f64 = suffix.iter().map(|&(_, w)| w).sum();
        let chi2: f64 = suffix
            .iter()
            .map(|&(neighbor, weight)| {
                let expected = weight / total * n_samples as f64;
                (counts[neighbor as usize] as f64 - expected).powi(2) / expected
            })
            .sum();

        assert_eq!(counts[0], 0, "the prefix is not in the suffix");
        assert!(
            chi2 < 13.82,
            "chi-squared {chi2:.2} exceeds 13.82; counts = {counts:?}"
        );
    }

    /// Scaling every weight scales the CDF, not the distribution, so a suffix stays as
    /// samplable at `1e-300` as at unit magnitude.
    #[test]
    fn a_uniformly_scaled_distribution_draws_alike() {
        let reference = sample_counts(&[(0, 1.0), (1, 2.0), (2, 7.0)], 0, 1_000);
        for scale in [1e-6f64, 1e-20, 1e-300, 1e6, 1e300] {
            let entries: Vec<(u32, f64)> = [(0, 1.0), (1, 2.0), (2, 7.0)]
                .map(|(neighbor, weight): (u32, f64)| (neighbor, weight * scale))
                .to_vec();
            assert_eq!(
                sample_counts(&entries, 0, 1_000),
                reference,
                "scale {scale:e} changed the draws"
            );
        }
    }

    /// Sweeps every start, so both arms of the suffix search run.
    #[test]
    fn monotonic_suffix() {
        let mut sampler = CdfSampler::new(SEED);
        let n = 64;
        let entries: Vec<(u32, f64)> = (0..n).map(|i| (i as u32, (i + 1) as f64)).collect();
        sampler.prepare(entries.iter().copied());

        let mut sampled_any = vec![false; n];

        for start in 1..n {
            for _ in 0..20 {
                if let Some(neighbor) = sampler.sample_after(start) {
                    let idx = neighbor as usize;
                    assert!(
                        idx >= start && idx < n,
                        "index {idx} out of range [{start}, {n})"
                    );
                    sampled_any[idx] = true;
                }
            }
        }

        let heavy_half = n / 2..n;
        for i in heavy_half {
            assert!(sampled_any[i], "heavy index {i} was never sampled");
        }
    }

    /// The old `/(u64::MAX as f64 + 1.0)` mapping rounded the top of `next_u64`'s range to
    /// exactly `1.0`. Stepping by `1 << 11` moves the retained bits every iteration — a
    /// stride of 1 only walks the 11 discarded bits and re-tests one draw.
    #[test]
    fn every_draw_is_below_one() {
        let top = (0..4096u64).map(|d| u64::MAX - d * (1 << 11));
        let spread = (0..4096u64).map(|d| d.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for bits in top
            .chain(spread)
            .chain([0, 1, (1 << 11) - 1, 1 << 53, 1 << 63])
        {
            let u = draw_from(bits);
            assert!((0.0..1.0).contains(&u), "u = {u:.20} for bits = {bits:#x}");
        }
    }

    /// Reverting #109 moves `u` by at most `2^-53`, which changes the chosen bucket only on
    /// a boundary straddle — so the mapping has to be pinned on the draw's own bits, where
    /// truncation and round-to-nearest disagree on every `bits` with a nonzero low 11.
    #[test]
    fn the_samplers_own_draw_is_the_documented_mapping() {
        let mut sampler = CdfSampler::<f64>::new(SEED);
        let mut stream = SmallRng::seed_from_u64(SEED);

        for i in 0..4_096 {
            let expected = draw_from(stream.next_u64());
            assert_eq!(
                sampler.draw().to_bits(),
                expected.to_bits(),
                "draw {i}: the sampler left the documented mapping"
            );
        }
    }

    /// The draw's value is [`the_samplers_own_draw_is_the_documented_mapping`]'s; this pins
    /// what surrounds it — exactly one draw per call, `start` reaching the search unchanged.
    #[test]
    fn sample_suffix_maps_one_raw_draw_per_call() {
        let entries: Vec<(u32, f64)> = (0..64).map(|i| (i, (i + 1) as f64)).collect();
        let mut sampler = CdfSampler::new(SEED);
        sampler.prepare(entries.iter().copied());
        let mut stream = SmallRng::seed_from_u64(SEED);

        for start in (0..entries.len()).cycle().take(4_096) {
            let expected = sampler.index_for_draw(start, draw_from(stream.next_u64()));
            assert_eq!(
                sampler.sample_suffix(start),
                expected,
                "start {start}: the sampler's own draw left the documented mapping"
            );
        }
    }

    /// Narrowing to `f32` rounds every draw above `1 - 2^-25` to exactly 1.0, so
    /// `fl(remaining + base)` can still exceed the last cumulative sum and the clamp stays
    /// load-bearing — it is not dead code left over from the `[0, 1]` draw. These weights
    /// were found by searching random distributions for a start whose `remaining` rounds up;
    /// at `f64` that search returns nothing, which is why only the `f32` case is pinned here.
    #[test]
    fn the_top_draw_narrowed_to_f32_still_needs_the_clamp() {
        let top = draw_from(u64::MAX);
        assert_eq!(
            <f32 as NumCast>::from(top).expect("the draw narrows to f32"),
            1.0,
            "the top draw must narrow to 1.0f32 or this tests nothing"
        );

        let weights: [f32; 5] = [18852.719, 69.055_58, 113_884.05, 70647.53, 956.364_56];
        let padding: Vec<f32> = vec![1.0; LINEAR_THRESHOLD];
        // The overrunning start is at index 2, reached through each search arm in turn.
        for extra in [0, padding.len()] {
            let entries: Vec<(u32, f32)> = weights
                .iter()
                .chain(padding.iter().take(extra))
                .enumerate()
                .map(|(i, &w)| (i as u32, w))
                .collect();
            let mut sampler = CdfSampler::<f32>::new(SEED);
            sampler.prepare(entries.iter().copied());
            for start in 0..entries.len() {
                let index = sampler
                    .index_for_draw(start, top)
                    .expect("a positive suffix is samplable");
                assert!(
                    index < entries.len(),
                    "len {}, start {start}: index {index} is past the CDF",
                    entries.len()
                );
            }
        }
    }

    /// There is no distribution left to sample, so every draw must answer alike.
    #[test]
    fn a_degenerate_suffix_answers_deterministically() {
        let cases = [
            ("weights that sum to zero", vec![(0u32, 0.0f64); 3], None),
            ("empty", vec![], None),
            ("one entry", vec![(0, 5.0)], Some(0u32)),
        ];
        for (label, entries, expected) in cases {
            let mut sampler = CdfSampler::new(SEED);
            sampler.prepare(entries.iter().copied());
            for _ in 0..100 {
                assert_eq!(sampler.sample_after(0), expected, "{label}");
            }
        }
    }
}
