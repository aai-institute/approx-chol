use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::types::{near_zero, Real};
use num_traits::NumCast;

/// Crossover point: for ranges of this size or smaller, a linear CDF scan is faster
/// than binary search due to branch-prediction and cache effects.
const LINEAR_THRESHOLD: usize = 32;

/// A star's neighbors as a weighted distribution, drawn from by suffix, and the
/// stream those draws come from.
///
/// Holding `neighbors` rather than borrowing the entries is what leaves one length
/// in play: both arrays are refilled by the same [`prepare`](Self::prepare) loop, so
/// a draw answers with a neighbor without any caller re-indexing an offset it was
/// handed.
///
/// The stream outlives any one distribution on purpose. A fixed seed reproduces a
/// factor only because *one* sampler is threaded through every block, so owning the
/// rng here is what makes a second stream unconstructible.
pub(crate) struct CdfSampler<T = f64> {
    neighbors: Vec<u32>,
    cumsum: Vec<T>,
    rng: SmallRng,
}

impl<T> CdfSampler<T> {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            neighbors: Vec::new(),
            cumsum: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl<T: Real> CdfSampler<T> {
    /// Take `entries` as the distribution to draw from, replacing the previous one.
    /// Cumulative sum by naive summation (assumes well-conditioned weights).
    ///
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

    /// Draw one neighbor at or after `start` proportional to weight, or `None` when
    /// that suffix is empty or carries negligible weight. The end is the prepared
    /// distribution's own length, so no caller can name a stale one.
    #[inline]
    pub(crate) fn sample_after(&mut self, start: usize) -> Option<u32> {
        let index = self.sample_suffix(start)?;
        Some(self.neighbors[index])
    }

    #[inline]
    fn sample_suffix(&mut self, start: usize) -> Option<usize> {
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
        if remaining <= near_zero::<T>() {
            return None;
        }

        // Draw a uniform in [0, 1) via next_u64 for rand 0.9/0.10 compatibility.
        let u = (self.rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0);
        let r = <T as NumCast>::from(u)? * remaining + base;

        // The clamp guards floating-point rounding that puts `r` past the last
        // cumulative sum.
        let k = if end - start <= LINEAR_THRESHOLD {
            let mut k = start;
            while k < end && self.cumsum[k] < r {
                k += 1;
            }
            k
        } else {
            self.cumsum[start..end].partition_point(|&c| c < r) + start
        };
        Some(k.min(end - 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 42;

    /// Every fixture below names entry `i` neighbor `i`, so a draw indexes the
    /// tally it belongs to and the suffix bound reads off the neighbor directly.
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

    /// Draws land in proportion to weight on both arms of the suffix search: three
    /// entries stay under [`LINEAR_THRESHOLD`], fifty do not. Critical values are
    /// chi-squared at `p = 0.001` for `len - 1` degrees of freedom.
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

    /// Sweeps every start, so both the linear and `partition_point` arms of the
    /// suffix search run, and no draw may fall before the start it was given.
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

    /// A suffix with only one place to land, or nowhere worth landing, answers the
    /// same way every draw — there is no distribution left to sample.
    #[test]
    fn a_degenerate_suffix_answers_deterministically() {
        let cases = [
            (
                "weights below the near-zero floor",
                vec![(0u32, 1e-20f64); 3],
                None,
            ),
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
