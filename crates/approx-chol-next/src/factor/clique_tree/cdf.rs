use rand::rngs::SmallRng;

use crate::Real;

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
        todo!()
    }

    /// A position in the suffix past `position`, clamped to its last: an `f32` draw can round up to the total; a linear scan up to 32 entries (256 measured +4-5%).
    pub(super) fn draw_after(&self, rng: &mut SmallRng, position: usize) -> usize {
        todo!()
    }
}
