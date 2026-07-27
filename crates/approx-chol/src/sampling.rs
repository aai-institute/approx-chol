use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::types::Real;
use num_traits::NumCast;

/// Crossover point: for ranges of this size or smaller, a linear CDF scan is faster
/// than binary search due to branch-prediction and cache effects.
const LINEAR_THRESHOLD: usize = 32;

/// Sample one index from `cumsum[start..]` proportional to weight, or `None` when
/// the suffix is empty or its remaining weight is negligible.
///
/// The `.min(end - 1)` clamp guards against floating-point rounding where the
/// random value slightly exceeds the cumulative sum range.
fn sample_from_cumsum<T: Real>(cumsum: &[T], rng: &mut SmallRng, start: usize) -> Option<usize> {
    let end = cumsum.len();
    if start >= end {
        return None;
    }

    let base = if start > 0 {
        cumsum[start - 1]
    } else {
        T::zero()
    };
    let remaining = cumsum[end - 1] - base;

    if remaining <= T::near_zero() {
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

/// Inverse-CDF sampler with hybrid linear/binary search.
pub(crate) struct CdfSampler<T = f64> {
    cumsum: Vec<T>,
    rng: SmallRng,
}

impl<T> CdfSampler<T> {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            cumsum: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl<T: Real> CdfSampler<T> {
    /// Cumulative sum by naive summation (assumes well-conditioned weights).
    #[inline]
    pub(crate) fn prepare(&mut self, entries: &[(u32, T)]) {
        self.cumsum.clear();
        let mut acc = T::zero();
        for &(_, w) in entries {
            acc = acc + w;
            self.cumsum.push(acc);
        }
    }

    /// Sample one index from the prepared entries at or after `start`,
    /// proportional to weight. The end is the prepared set's own length, so no
    /// caller can name a stale one.
    #[inline]
    pub(crate) fn sample_suffix(&mut self, start: usize) -> Option<usize> {
        sample_from_cumsum(&self.cumsum, &mut self.rng, start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 42;

    fn sample_counts(
        sampler: &mut CdfSampler<f64>,
        entries: &[(u32, f64)],
        start: usize,
        n_samples: usize,
    ) -> Vec<u32> {
        sampler.prepare(entries);
        let mut counts = vec![0u32; entries.len()];
        for _ in 0..n_samples {
            if let Some(idx) = sampler.sample_suffix(start) {
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
        let counts = sample_counts(&mut sampler, &entries, 0, n_samples);

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

    /// Sweeps every start, so both the linear and `partition_point` arms of the
    /// suffix search run, and no draw may fall before the start it was given.
    #[test]
    fn monotonic_suffix() {
        let mut sampler = CdfSampler::new(SEED);
        let n = 64;
        let entries: Vec<(u32, f64)> = (0..n).map(|i| (i as u32, (i + 1) as f64)).collect();
        sampler.prepare(&entries);

        let mut sampled_any = vec![false; n];

        for start in 1..n {
            for _ in 0..20 {
                if let Some(idx) = sampler.sample_suffix(start) {
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
                sampler.sample_suffix(0).is_none(),
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
            assert_eq!(sampler.sample_suffix(0), Some(0));
        }
    }

    /// `n` above [`LINEAR_THRESHOLD`], so this is the binary-search arm's
    /// distribution; [`distribution_accuracy`] covers the linear one.
    #[test]
    fn equal_weights() {
        let mut sampler = CdfSampler::new(SEED);
        let n = 50;
        let entries: Vec<(u32, f64)> = (0..n).map(|i| (i as u32, 1.0)).collect();
        let n_samples = 100_000;
        let counts = sample_counts(&mut sampler, &entries, 0, n_samples);

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
