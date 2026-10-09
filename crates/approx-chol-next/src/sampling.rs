//! Seeded inverse-CDF draws, one stream per part and star so no draw depends on scheduling.

use rand::rngs::SmallRng;

use crate::Real;

/// Inverse-CDF draws over a star's suffixes.
pub(crate) struct CdfSampler<T> {
    neighbors: Vec<u32>,
    cumsum: Vec<T>,
    rng: SmallRng,
}

impl<T: Real> CdfSampler<T> {
    /// Loads a star, heaviest last, so each suffix's draw range is one subtraction.
    pub(crate) fn prepare(&mut self, star: impl Iterator<Item = (u32, T)>) {
        todo!()
    }

    /// A neighbor from the suffix past `position`, weighted by its share.
    pub(crate) fn draw_after(&mut self, position: usize) -> u32 {
        todo!()
    }
}

/// One part's streams, keyed so no draw depends on thread scheduling or on how other parts routed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Streams {
    seed: u64,
    component: u32,
}

impl Streams {
    pub(crate) fn new(seed: u64, component: u32) -> Self {
        Self { seed, component }
    }

    pub(crate) fn kernel<T: Real>(self) -> CdfSampler<T> {
        todo!()
    }

    pub(crate) fn star<T: Real>(self, star: usize) -> CdfSampler<T> {
        todo!()
    }
}
