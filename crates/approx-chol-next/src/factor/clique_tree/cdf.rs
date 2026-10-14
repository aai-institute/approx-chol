use rand::rngs::Xoshiro256PlusPlus;
use rand::Rng;

use crate::Real;

const LINEAR_THRESHOLD: usize = 32;

/// Inverse-CDF draws over a star's suffixes, by position: the star owns its neighbors, the caller its draws.
pub(super) struct CdfSampler<T> {
    cumsum: Vec<T>,
}

impl<T: Real> CdfSampler<T> {
    pub(super) fn new() -> Self {
        Self { cumsum: Vec::new() }
    }

    /// Loads a star's weights, heaviest last, so each suffix's draw range is one subtraction.
    pub(super) fn prepare(&mut self, weights: impl Iterator<Item = T>) {
        self.cumsum.clear();
        let mut acc = T::zero();
        for w in weights {
            acc = acc + w;
            self.cumsum.push(acc);
        }
    }

    /// The weight past `position`, all of it for `None`: one subtraction, as each draw's range.
    pub(super) fn suffix_past(&self, position: Option<usize>) -> T {
        let total = *self.cumsum.last().expect("a prepared star");
        position.map_or(total, |p| total - self.cumsum[p])
    }

    /// A position in the suffix past `position`, clamped to its last: an `f32` draw can round up to the total; a linear scan up to 32 entries (256 measured +4-5%).
    pub(super) fn draw_after(&self, rng: &mut Xoshiro256PlusPlus, position: usize) -> usize {
        let (start, end) = (position + 1, self.cumsum.len());
        // 53 bits times a power of two is exact, so never 1.0.
        let u = ((rng.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
        let u = T::from(u).expect("the draw is representable in T");
        let r = u * self.suffix_past(Some(position)) + self.cumsum[position];
        let k = if end - start <= LINEAR_THRESHOLD {
            let mut k = start;
            while k < end && self.cumsum[k] < r {
                k += 1;
            }
            k
        } else {
            self.cumsum[start..end].partition_point(|&c| c < r) + start
        };
        k.min(end - 1)
    }
}
