use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::types::Real;
use num_traits::NumCast;

/// At or below this range size a linear CDF scan beats binary search.
const LINEAR_THRESHOLD: usize = 32;

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

        // Draw a uniform in [0, 1) via next_u64 for rand 0.9/0.10 compatibility.
        let u = (self.rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0);
        let r = <T as NumCast>::from(u)? * remaining + base;

        // The clamp guards floating-point rounding that puts `r` past the last
        // cumulative sum.
        let k = if end - start <= LINEAR_THRESHOLD {
            let mut k = start;
            while k < end && self.cumsum[k] < r {
                let advanced = k;
                k += 1;
                // A broken advance leaves this condition re-checking the same index
                // forever instead of failing, so a bad increment hangs rather than
                // panics.
                debug_assert!(k > advanced, "sample_suffix linear scan failed to advance");
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
