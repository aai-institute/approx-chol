use rand::rngs::SmallRng;

use crate::Real;

/// Inverse-CDF draws over a star's suffixes.
pub(super) struct CdfSampler<T> {
    neighbors: Vec<u32>,
    cumsum: Vec<T>,
    rng: SmallRng,
}

impl<T: Real> CdfSampler<T> {
    pub(super) fn new(stream: Stream) -> Self {
        todo!()
    }

    pub(super) fn restart(&mut self, stream: Stream) {
        todo!()
    }

    /// Loads a star, heaviest last, so each suffix's draw range is one subtraction.
    pub(super) fn prepare(&mut self, star: impl Iterator<Item = (u32, T)>) {
        todo!()
    }

    /// A neighbor from the suffix past `position`, clamped to its last: an `f32` draw can round up to the total; a linear scan up to 32 entries (256 measured +4-5%).
    pub(super) fn draw_after(&mut self, position: usize) -> u32 {
        todo!()
    }
}

/// One part's streams, keyed so no draw depends on thread scheduling or on how other parts routed.
#[derive(Clone, Copy, Debug)]
pub(in crate::factor) struct Streams {
    seed: u64,
    part: u32,
}

impl Streams {
    pub(in crate::factor) fn new(seed: u64, part: u32) -> Self {
        Self { seed, part }
    }

    pub(in crate::factor) fn remainder(self) -> Stream {
        todo!()
    }

    pub(in crate::factor) fn star(self, eliminated: usize) -> Stream {
        todo!()
    }
}

/// One seeded draw sequence.
#[derive(Clone, Copy, Debug)]
pub(in crate::factor) struct Stream(u64);
