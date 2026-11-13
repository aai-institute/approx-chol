//! A solve's budget spread over the coupled blocks by size: a block past the budget's share of the work on the whole budget, one after another, the rest at once when their work pays for waking the pool, one thread each.
#![allow(unsafe_code)]

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::apply::Room;
use super::{slots, ApproxInverse, Block, Indexed};
use crate::order::{Order, Rows};
use crate::threads::ONE;
use crate::{Batch, Index, MemStack, Real, StackReq, Threads};

/// Steps of the blocks within the budget's share below which they stay on the calling thread: waking the pool cost ~20 µs per apply, and at 3,840–6,336 steps four threads at best broke even.
const SPREAD_STEPS: usize = 8192;

/// Claims per worker: a few dozen let unequal blocks even out, while 100,000 blocks of 9 vertices still claim dozens at once.
const CLAIMS_PER_WORKER: usize = 16;

/// Every coupled block summed, and the most steps one takes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Coupled {
    pub(super) all: Small,
    most_steps: usize,
}

impl Coupled {
    pub(super) fn of<T: Real, I: Index>(blocks: &[Block<T, I>]) -> Self {
        let (mut all, mut most_steps) = (Small::default(), 0);
        for block in blocks {
            let steps = block.n_steps();
            all.add(steps, block.global_scratch_slots());
            most_steps = most_steps.max(steps);
        }
        Self { all, most_steps }
    }
}

/// How a solve at a budget runs the coupled blocks.
pub(super) struct Spread {
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
    pub(super) fn of<T: Real>(inverse: &ApproxInverse<T>, threads: Threads) -> Self {
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
    pub(super) fn room<T: Real>(&self, inverse: &ApproxInverse<T>) -> Room {
        let mut alone = 0;
        let small = on!(&inverse.blocks, blocks => self.split(blocks, |block| {
            alone = alone.max(block.global_scratch_slots());
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

    /// Hands each block past the share to `alone`, in order, and sums the others.
    fn split<T: Real, I: Index>(
        &self,
        blocks: &[Block<T, I>],
        mut alone: impl FnMut(&Block<T, I>),
    ) -> Small {
        let share = match self.within {
            Within::All(small) => return small,
            Within::Share(share) => share,
        };
        let mut small = Small::default();
        for block in blocks {
            match block.n_steps() > share {
                true => alone(block),
                false => small.add(block.n_steps(), block.global_scratch_slots()),
            }
        }
        small
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
    // Sound for an ApproxInverse's own blocks alone: `pack` and decoding leave their segments disjoint.
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
        let small = self.split(blocks, |block| {
            block.solve_global(order, x, threads, stack);
        });
        let within = |block: &&Block<T, I>| match self.within {
            Within::All(_) => true,
            Within::Share(share) => block.n_steps() <= share,
        };
        let region = slots::<V>(small.slots);
        let workers = match self.spreads(&small) {
            true => regions(stack, region, self.tasks),
            false => 1,
        };
        if workers < 2 {
            for block in blocks.iter().filter(within) {
                block.solve_global(order, x, threads, stack);
            }
            return;
        }
        let claim = blocks.len().div_ceil(workers * CLAIMS_PER_WORKER);
        let next = AtomicUsize::new(0);
        let x = Shared::new(x);
        let work = |stack: &mut MemStack| loop {
            let start = next.fetch_add(claim, Ordering::Relaxed);
            let Some(claimed) = blocks.get(start..(start + claim).min(blocks.len())) else {
                break;
            };
            for block in claimed.iter().filter(within) {
                // SAFETY: each block is claimed once, and an ApproxInverse's segments share no vertex.
                let mut rows = unsafe { x.claim() };
                order
                    .slots(block.segment)
                    .on_local(&mut rows, stack, |local, rest| {
                        block.solve_unchecked(local, rest, ONE)
                    });
            }
        };
        split(stack, region, workers, &work);
    }
}

/// How many regions of `region` `stack` holds, at most `tasks`; at least one, as the apply checked.
fn regions(stack: &MemStack, region: StackReq, tasks: usize) -> usize {
    let fit = match region.size_bytes() {
        0 => tasks,
        size => (stack.len_bytes() / size).min(tasks),
    };
    (1..=fit)
        .rev()
        .find(|&count| stack.can_hold(region.array(count)))
        .unwrap_or(1)
}

/// Runs `work` on `workers` regions of `region` carved from `stack`, each on a task of its own.
fn split(
    stack: &mut MemStack,
    region: StackReq,
    workers: usize,
    work: &(impl Fn(&mut MemStack) + Sync),
) {
    if workers == 1 {
        return work(stack);
    }
    let left = workers / 2;
    let carved = region.array(left);
    let (mine, rest) = stack.make_aligned_uninit::<u8>(carved.size_bytes(), carved.align_bytes());
    rayon::join(
        || split(MemStack::new(mine), region, left, work),
        || split(rest, region, workers - left, work),
    );
}

/// `x` solved by blocks at once, each through a [`Claim`] of its own.
struct Shared<'x, V> {
    ptr: *mut V,
    len: usize,
    _x: PhantomData<&'x mut [V]>,
}

// SAFETY: a `&mut [V]` moves between threads when `V` does, and claims touch disjoint rows.
unsafe impl<V: Send> Sync for Shared<'_, V> {}

impl<'x, V> Shared<'x, V> {
    fn new(x: &'x mut [V]) -> Self {
        Self {
            ptr: x.as_mut_ptr(),
            len: x.len(),
            _x: PhantomData,
        }
    }

    /// Rows of `x` for one block.
    ///
    /// # Safety
    /// While the claim lives, no other touches a row it touches.
    unsafe fn claim(&self) -> Claim<'_, V> {
        Claim {
            ptr: self.ptr,
            len: self.len,
            _shared: PhantomData,
        }
    }
}

/// One block's rows of a [`Shared`] `x`; every row bounds-checked.
struct Claim<'s, V> {
    ptr: *mut V,
    len: usize,
    _shared: PhantomData<&'s ()>,
}

impl<V: Copy> Rows<V> for Claim<'_, V> {
    fn run(&mut self, start: usize, len: usize) -> &mut [V] {
        let inside = start.checked_add(len).is_some_and(|end| end <= self.len);
        assert!(inside, "a run inside x");
        // SAFETY: inside `x`, and no other claim touches these rows.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), len) }
    }

    fn get(&self, row: usize) -> V {
        assert!(row < self.len, "a row inside x");
        // SAFETY: inside `x`, and no other claim writes it.
        unsafe { self.ptr.add(row).read() }
    }

    fn set(&mut self, row: usize, value: V) {
        assert!(row < self.len, "a row inside x");
        // SAFETY: inside `x`, and no other claim reads or writes it.
        unsafe { self.ptr.add(row).write(value) }
    }
}
