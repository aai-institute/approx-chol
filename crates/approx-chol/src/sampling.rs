use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::types::Real;
use num_traits::NumCast;

/// Weight threshold below which remaining weight is considered negligible.
#[inline]
pub(crate) fn near_zero<T: Real>() -> T {
    T::near_zero()
}

/// Crossover point: for ranges of this size or smaller, a linear CDF scan is faster
/// than binary search due to branch-prediction and cache effects.
pub(crate) const LINEAR_THRESHOLD: usize = 32;

/// Sample one index from `cumsum[start..end]` proportional to weight.
///
/// Uses a linear scan for small ranges (≤ [`LINEAR_THRESHOLD`]) and binary
/// search via `partition_point` for larger ones. Returns `None` if the
/// remaining weight in the range is negligible.
///
/// The `.min(end - 1)` clamp guards against floating-point rounding where
/// the random value slightly exceeds the cumulative sum range.
pub(crate) fn sample_from_cumsum<T: Real>(
    cumsum: &[T],
    rng: &mut SmallRng,
    start: usize,
    end: usize,
) -> Option<usize> {
    debug_assert!(start < end && end <= cumsum.len());

    let base = if start > 0 {
        cumsum[start - 1]
    } else {
        T::zero()
    };
    let remaining = cumsum[end - 1] - base;

    if remaining <= near_zero::<T>() {
        return None;
    }

    // Draw a uniform in [0, 1) via next_u64 for rand 0.9/0.10 compatibility.
    let u = (rng.next_u64() as f64) / ((u64::MAX as f64) + 1.0);
    let u_t = <T as NumCast>::from(u)?;
    let r = u_t * remaining + base;

    let range_size = end - start;
    let k = if range_size <= LINEAR_THRESHOLD {
        let mut k = start;
        while k < end && cumsum[k] < r {
            k += 1;
        }
        k.min(end - 1)
    } else {
        (cumsum[start..end].partition_point(|&c| c < r) + start).min(end - 1)
    };

    Some(k)
}

/// Strategy for weighted index sampling during clique-tree factorization.
pub(crate) trait WeightedSampler<T: Real> {
    /// Build internal state from the full set of `(vertex_index, weight)` entries.
    fn prepare(&mut self, entries: &[(u32, T)]);

    /// Sample one index from `entries[start..end]` proportional to weight.
    fn sample_from_range(&mut self, start: usize, end: usize) -> Option<usize>;
}

/// Inverse-CDF sampler with hybrid linear/binary search.
pub struct CdfSampler<T = f64> {
    cumsum: Vec<T>,
    rng: SmallRng,
}

impl<T> CdfSampler<T> {
    /// Create a new sampler with the given PRNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            cumsum: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl<T> Default for CdfSampler<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T: Real> WeightedSampler<T> for CdfSampler<T> {
    /// Build cumulative sum from weights using naive summation (assumes well-conditioned weights).
    #[inline]
    fn prepare(&mut self, entries: &[(u32, T)]) {
        self.cumsum.clear();
        let mut acc = T::zero();
        for &(_, w) in entries {
            acc = acc + w;
            self.cumsum.push(acc);
        }
    }

    #[inline]
    fn sample_from_range(&mut self, start: usize, end: usize) -> Option<usize> {
        sample_from_cumsum(&self.cumsum, &mut self.rng, start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 42;

    fn sample_counts(
        sampler: &mut impl WeightedSampler<f64>,
        entries: &[(u32, f64)],
        start: usize,
        end: usize,
        n_samples: usize,
    ) -> Vec<u32> {
        sampler.prepare(entries);
        let mut counts = vec![0u32; entries.len()];
        for _ in 0..n_samples {
            if let Some(idx) = sampler.sample_from_range(start, end) {
                counts[idx] += 1;
            }
        }
        counts
    }

    #[test]
    fn distribution_accuracy() {
        let mut sampler = CdfSampler::new(SEED);
        let entries: Vec<(u32, f64)> = vec![(0, 1.0), (1, 2.0), (2, 7.0)];
        let n_samples = 50_000;
        let counts = sample_counts(&mut sampler, &entries, 0, 3, n_samples);

        let total_w = 10.0;
        let expected = [1.0 / total_w, 2.0 / total_w, 7.0 / total_w];

        let mut chi2 = 0.0;
        for i in 0..3 {
            let obs = counts[i] as f64;
            let exp = expected[i] * n_samples as f64;
            chi2 += (obs - exp).powi(2) / exp;
        }
        assert!(
            chi2 < 13.82,
            "chi-squared {chi2:.2} exceeds critical value; counts = {counts:?}"
        );
    }

    #[test]
    fn suffix_range() {
        let mut sampler = CdfSampler::new(SEED);
        let entries: Vec<(u32, f64)> = (0..5).map(|i| (i as u32, (i + 1) as f64)).collect();
        let n_samples = 10_000;
        let counts = sample_counts(&mut sampler, &entries, 2, 5, n_samples);

        assert_eq!(counts[0], 0, "index 0 sampled from range [2,5)");
        assert_eq!(counts[1], 0, "index 1 sampled from range [2,5)");
        for (i, &count) in counts.iter().enumerate().skip(2).take(3) {
            assert!(count > 0, "index {i} never sampled from range [2,5)");
        }
    }

    #[test]
    fn monotonic_suffix() {
        let mut sampler = CdfSampler::new(SEED);
        let n = 64;
        let entries: Vec<(u32, f64)> = (0..n).map(|i| (i as u32, (i + 1) as f64)).collect();
        sampler.prepare(&entries);

        let mut sampled_any = vec![false; n];

        for start in 1..n {
            for _ in 0..20 {
                if let Some(idx) = sampler.sample_from_range(start, n) {
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

    #[test]
    fn near_zero_weights() {
        let mut sampler = CdfSampler::new(SEED);
        let entries: Vec<(u32, f64)> = vec![(0, 1e-20), (1, 1e-20), (2, 1e-20)];
        sampler.prepare(&entries);

        for _ in 0..100 {
            assert!(
                sampler.sample_from_range(0, 3).is_none(),
                "expected None for near-zero weights"
            );
        }
    }

    #[test]
    fn single_entry() {
        let mut sampler = CdfSampler::new(SEED);
        let entries = vec![(0u32, 5.0)];
        sampler.prepare(&entries);
        for _ in 0..100 {
            assert_eq!(sampler.sample_from_range(0, 1), Some(0));
        }
    }

    #[test]
    fn two_entries() {
        let mut sampler = CdfSampler::new(SEED);
        let entries = vec![(0u32, 1.0), (1, 3.0)];
        let n_samples = 20_000;
        let counts = sample_counts(&mut sampler, &entries, 0, 2, n_samples);

        let ratio = counts[1] as f64 / counts[0].max(1) as f64;
        assert!(
            (2.0..=4.5).contains(&ratio),
            "expected ratio ~3, got {ratio:.2} (counts: {counts:?})"
        );
    }

    #[test]
    fn equal_weights() {
        let mut sampler = CdfSampler::new(SEED);
        let n = 50;
        let entries: Vec<(u32, f64)> = (0..n).map(|i| (i as u32, 1.0)).collect();
        let n_samples = 100_000;
        let counts = sample_counts(&mut sampler, &entries, 0, n, n_samples);

        let expected = n_samples as f64 / n as f64;
        for (i, &c) in counts.iter().enumerate() {
            let deviation = (c as f64 - expected).abs() / expected;
            assert!(
                deviation < 0.15,
                "index {i}: count {c}, expected ~{expected:.0}, deviation {deviation:.2}"
            );
        }
    }
}
