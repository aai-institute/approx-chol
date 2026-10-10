use rand::rngs::SmallRng;
use rand::SeedableRng;

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

    /// The approximate Cholesky's one sequence: pivot `t`'s sample is conditioned on every earlier one.
    pub(in crate::factor) fn cholesky(self) -> SequentialDraws {
        SequentialDraws(rng(self, Stage::Cholesky))
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

/// One sequence for a sequential elimination, borrowed by each pivot in turn.
#[derive(Debug)]
pub(in crate::factor) struct SequentialDraws(pub(super) SmallRng);

/// One star's independent sample of the Schur complement, consumed by its elimination, so no two stars share one.
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
    let stage = match stage {
        Stage::Cholesky => 0,
        Stage::Star(eliminated) => eliminated as u64 + 1,
    };
    let key = [u64::from(streams.component), stage]
        .into_iter()
        .fold(splitmix(streams.seed), |key, part| splitmix(key ^ part));
    SmallRng::seed_from_u64(key)
}

/// SplitMix64's finalizer: each part changes every bit of the key, so no two keys collide by addition.
fn splitmix(x: u64) -> u64 {
    let x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
