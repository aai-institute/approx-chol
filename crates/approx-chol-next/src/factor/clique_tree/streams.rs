use rand::rngs::SmallRng;

/// One component's keys, so no draw depends on thread scheduling or on how other components routed.
#[derive(Clone, Copy, Debug)]
pub(in crate::factor) struct Streams {
    seed: u64,
    component: u32,
}

impl Streams {
    pub(in crate::factor) fn new(seed: u64, component: u32) -> Self {
        Self { seed, component }
    }

    /// The approximate Cholesky's one sequence, threaded through every pivot.
    pub(in crate::factor) fn cholesky(self) -> RunDraws {
        RunDraws(rng(self, Stage::Cholesky))
    }

    /// The Schur stars' keys alone, so a complement can never reach the Cholesky's sequence.
    pub(in crate::factor) fn stars(self) -> StarStreams {
        StarStreams(self)
    }
}

/// Keys for the eliminated stars of one component's Schur complement.
#[derive(Clone, Copy, Debug)]
pub(in crate::factor) struct StarStreams(Streams);

impl StarStreams {
    /// Star `eliminated`'s own sequence, by its index in the component, never by task.
    pub(in crate::factor) fn star(self, eliminated: usize) -> StarDraws {
        StarDraws(rng(self.0, Stage::Star(eliminated)))
    }
}

/// The Cholesky's draws, borrowed by each pivot in turn.
#[derive(Debug)]
pub(in crate::factor) struct RunDraws(pub(super) SmallRng);

/// One Schur star's draws, consumed by its elimination, so no two stars share one.
#[derive(Debug)]
pub(in crate::factor) struct StarDraws(pub(super) SmallRng);

/// Which draws of a component a key feeds.
#[derive(Clone, Copy, Debug)]
enum Stage {
    Cholesky,
    Star(usize),
}

/// The whole key mixed once, the only place a key becomes draws.
fn rng(streams: Streams, stage: Stage) -> SmallRng {
    todo!()
}
