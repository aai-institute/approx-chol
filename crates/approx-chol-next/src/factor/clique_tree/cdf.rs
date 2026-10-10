use rand::rngs::SmallRng;

use crate::Real;

/// Inverse-CDF draws over a star's suffixes; the draws themselves belong to the caller's stream.
pub(super) struct CdfSampler<T> {
    neighbors: Vec<u32>,
    cumsum: Vec<T>,
}

impl<T: Real> CdfSampler<T> {
    pub(super) fn new() -> Self {
        Self {
            neighbors: Vec::new(),
            cumsum: Vec::new(),
        }
    }

    /// Loads a star, heaviest last, so each suffix's draw range is one subtraction.
    pub(super) fn prepare(&mut self, star: impl Iterator<Item = (u32, T)>) {
        todo!()
    }

    /// A neighbor from the suffix past `position`, clamped to its last: an `f32` draw can round up to the total; a linear scan up to 32 entries (256 measured +4-5%).
    pub(super) fn draw_after(&self, draws: &mut Draws, position: usize) -> u32 {
        todo!()
    }
}

/// One component's streams, keyed so no draw depends on thread scheduling or on how other components routed.
#[derive(Clone, Copy, Debug)]
pub(in crate::factor) struct Streams {
    seed: u64,
    component: u32,
}

impl Streams {
    pub(in crate::factor) fn new(seed: u64, component: u32) -> Self {
        Self { seed, component }
    }

    pub(in crate::factor) fn cholesky(self) -> Stream {
        self.stream(Stage::Cholesky)
    }

    pub(in crate::factor) fn star(self, eliminated: usize) -> Stream {
        self.stream(Stage::Star(eliminated))
    }

    fn stream(self, stage: Stage) -> Stream {
        Stream {
            seed: self.seed,
            component: self.component,
            stage,
        }
    }
}

/// One seeded draw sequence, keyed whole and mixed once, so no two stages' sequences coincide.
#[derive(Clone, Copy, Debug)]
pub(in crate::factor) struct Stream {
    seed: u64,
    component: u32,
    stage: Stage,
}

impl Stream {
    /// The whole key mixed once, the only place a stream becomes draws.
    pub(in crate::factor) fn draws(self) -> Draws {
        todo!()
    }
}

/// One stream's draws, owned by what consumes that stream: a whole elimination, or one Schur star.
#[derive(Debug)]
pub(in crate::factor) struct Draws(SmallRng);

/// Which draws of a component a stream feeds.
#[derive(Clone, Copy, Debug)]
enum Stage {
    Cholesky,
    Star(usize),
}
