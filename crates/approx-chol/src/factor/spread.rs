//! A solve's budget spread over the coupled blocks by size: a block past the budget's share of the work on the whole budget, one after another, the rest at once when their work pays for waking the pool, one thread each.

use inflight::parallel::{Disjoint, Partition, Regions};

use super::{ApproxInverse, Block, Indexed};
use crate::order::{Order, Sets};
use crate::{Batch, Index, MemStack, Real, StackReq, Threads};

/// Steps of the blocks within the budget's share below which they stay on the calling thread: waking the pool cost ~20 µs per apply, and at 3,840–6,336 steps four threads at best broke even.
const SPREAD_STEPS: usize = 8192;

/// Claims per worker: a few dozen let unequal blocks even out, while 100,000 blocks of 9 vertices still claim dozens at once.
const CLAIMS_PER_WORKER: usize = 16;

/// `count` slots of `V` on a stack; none asks no alignment either, so any stack serves a solve that takes none.
fn slots<V>(count: usize) -> StackReq {
    match count {
        0 => StackReq::EMPTY,
        count => StackReq::new::<V>(count),
    }
}

/// What a solve at a budget takes from the stack, from which each path's need follows.
pub(super) struct Room {
    /// A column of the operator's rows.
    pub(super) column: StackReq,
    /// Per lane, the most a block past the budget's share takes alone.
    alone: StackReq,
    /// Per lane, the most one of the others takes, on each thread solving them at once.
    region: StackReq,
    /// Threads solving the others at once.
    regions: usize,
}

impl Room {
    pub(super) fn of<T: Real>(inverse: &ApproxInverse<T>, threads: Threads) -> Self {
        Spread::of(inverse, threads).room(inverse)
    }

    /// Rows of `lanes` right-hand sides each, solved where they are.
    pub(super) fn in_place(&self, lanes: usize) -> StackReq {
        let regions = self.region.array(lanes).array(self.regions);
        self.alone.array(lanes).or(regions)
    }
}

/// Every coupled block summed, and the most steps one takes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Coupled {
    pub(super) all: Small,
    most_steps: usize,
}

impl Coupled {
    pub(super) fn of<T: Real, I: Index>(blocks: &[Block<T, I>], order: &Order) -> Self {
        let (mut all, mut most_steps) = (Small::default(), 0);
        for (k, block) in blocks.iter().enumerate() {
            let steps = block.n_steps();
            all.add(steps, block.global_scratch_slots(order.slots(k)));
            most_steps = most_steps.max(steps);
        }
        Self { all, most_steps }
    }
}

/// How a solve at a budget runs the coupled blocks.
struct Spread {
    /// Threads at a time: the budget's, at most the pool's and one per block.
    tasks: usize,
    within: Within,
}

/// Which blocks run within the budget's share, the rest alone.
enum Within {
    /// Every block, summed at build.
    All(Small),
    /// Those of at most these steps.
    Share(usize),
}

/// Some blocks summed: their steps and the most slots one takes.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Small {
    pub(super) steps: usize,
    /// The most slots one of them takes.
    slots: usize,
}

impl Small {
    fn add(&mut self, steps: usize, slots: usize) {
        self.steps += steps;
        self.slots = self.slots.max(slots);
    }
}

impl Spread {
    fn of<T: Real>(inverse: &ApproxInverse<T>, threads: Threads) -> Self {
        let blocks = on!(&inverse.blocks, blocks => blocks.len());
        let tasks = threads.tasks().min(blocks).max(1);
        let Coupled { all, most_steps } = inverse.coupled;
        let share = all.steps / tasks;
        let within = match most_steps > share {
            true => Within::Share(share),
            false => Within::All(all),
        };
        Self { tasks, within }
    }

    /// The solve's room on the stack: the most a block running alone takes, and regions of the most another one takes, one per thread when they run at once.
    fn room<T: Real>(&self, inverse: &ApproxInverse<T>) -> Room {
        let mut alone = 0;
        let small = on!(&inverse.blocks, blocks => self.split(blocks, &inverse.order, |k, block| {
            alone = alone.max(block.global_scratch_slots(inverse.order.slots(k)));
        }));
        let regions = match self.spreads(&small) {
            true => self.tasks,
            false => 1,
        };
        Room {
            column: slots::<T>(inverse.n()),
            alone: slots::<T>(alone),
            region: slots::<T>(small.slots),
            regions,
        }
    }

    /// Hands each block past the share to `alone` with its index, in order, and sums the others.
    fn split<T: Real, I: Index>(
        &self,
        blocks: &[Block<T, I>],
        order: &Order,
        mut alone: impl FnMut(usize, &Block<T, I>),
    ) -> Small {
        let share = match self.within {
            Within::All(small) => return small,
            Within::Share(share) => share,
        };
        let mut small = Small::default();
        for (k, block) in blocks.iter().enumerate() {
            match block.n_steps() > share {
                true => alone(k, block),
                false => small.add(block.n_steps(), block.global_scratch_slots(order.slots(k))),
            }
        }
        small
    }

    /// Whether a block runs within the budget's share.
    fn within<T: Real, I: Index>(&self, block: &Block<T, I>) -> bool {
        match self.within {
            Within::All(_) => true,
            Within::Share(share) => block.n_steps() <= share,
        }
    }

    fn spreads(&self, small: &Small) -> bool {
        self.tasks > 1 && small.steps >= SPREAD_STEPS
    }
}

/// Solves every coupled block of `x`, of `n` rows, through `stack`, which holds one region of its [`room`](Spread::room) or more: as many blocks within the share at once as it holds regions, at most the budget's threads.
pub(super) fn solve_blocks<T: Real, V: Batch<T>>(
    inverse: &ApproxInverse<T>,
    x: &mut [V],
    threads: Threads,
    stack: &mut MemStack,
) {
    let spread = Spread::of(inverse, threads);
    on!(&inverse.blocks, blocks => spread.solve(blocks, &inverse.order, x, threads, stack));
}

impl Spread {
    fn solve<T: Real, I: Index, V: Batch<T>>(
        &self,
        blocks: &[Block<T, I>],
        order: &Order,
        x: &mut [V],
        threads: Threads,
        stack: &mut MemStack,
    ) {
        let small = self.split(blocks, order, |k, block| {
            block.solve_global(order, k, x, threads, stack);
        });
        let region = slots::<V>(small.slots);
        let workers = match self.spreads(&small) {
            true => self.tasks,
            false => 1,
        };
        let regions = Regions::fitting(stack, region, workers);
        if regions.count() < 2 {
            for (k, block) in blocks.iter().enumerate() {
                if self.within(block) {
                    block.solve_global(order, k, x, threads, stack);
                }
            }
            return;
        }
        let claim = blocks.len().div_ceil(regions.count() * CLAIMS_PER_WORKER);
        match &order.sets {
            Sets::U32(sets) => self.at_once(blocks, sets, x, claim, regions, stack),
            Sets::U64(sets) => self.at_once(blocks, sets, x, claim, regions, stack),
        }
    }

    /// The blocks within the share, `claim` at a time on each of `regions`, each on its set of `x`.
    fn at_once<T: Real, I: Index, V: Batch<T>, P: inflight::parallel::ListedIndex + Sync>(
        &self,
        blocks: &[Block<T, I>],
        sets: &Partition<P>,
        x: &mut [V],
        claim: usize,
        regions: Regions,
        stack: &mut MemStack,
    ) {
        let x = Disjoint::new(x, sets);
        let work = |stack: &mut MemStack| {
            for (k, rows) in x.claims(claim) {
                // Past the blocks, the isolated vertices' set, which no block solves.
                match blocks.get(k) {
                    Some(block) if self.within(block) => block.solve_claimed(rows, stack),
                    _ => {}
                }
            }
        };
        regions.run(stack, work);
    }
}
