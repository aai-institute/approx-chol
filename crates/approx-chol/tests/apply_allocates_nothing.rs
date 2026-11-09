//! Its own binary: the process must start without rayon's global pool, and the allocator counts every allocation.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use approx_chol::{
    factorize_with, ApproxInverse, ColumnMajor, Config, MatrixMut, MemBuffer, MemStack,
    ScalingPolicy, Threads,
};

#[path = "integration/common/bipartite.rs"]
mod bipartite;

use bipartite::Bipartite;

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator, only counting.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations on any thread while `f` runs.
fn allocations(f: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    COUNTING.store(true, Ordering::SeqCst);
    f();
    COUNTING.store(false, Ordering::SeqCst);
    ALLOCATIONS.load(Ordering::SeqCst)
}

/// Factored in a pool of its own, its workers joined on return so none allocates later, and rayon's global pool unbuilt.
fn factored() -> ApproxInverse<f64> {
    let gram = Bipartite::three_groups();
    let mut workers = Vec::new();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .spawn_handler(|thread| {
            workers.push(std::thread::spawn(|| thread.run()));
            Ok(())
        })
        .build()
        .unwrap();
    let factor = pool
        .install(|| factorize_with(gram.view(), Config::default(), ScalingPolicy::default()))
        .unwrap();
    drop(pool);
    // A joined thread has run its thread-local destructors, which allocate.
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(factor.components().len(), 3);
    factor
}

/// Columns stored apart, so the apply copies them through its scratch.
struct Apart([Vec<f64>; 2]);

impl MatrixMut<f64> for Apart {
    fn rows(&self) -> usize {
        self.0[0].len()
    }

    fn columns(&self) -> usize {
        self.0.len()
    }

    fn read_column(&self, column: usize, into: &mut [f64]) {
        into.copy_from_slice(&self.0[column]);
    }

    fn write_column(&mut self, column: usize, from: &[f64]) {
        self.0[column].copy_from_slice(from);
    }
}

#[test]
fn an_apply_on_one_thread_allocates_nothing_and_builds_no_pool() {
    let factor = factored();
    let n = factor.n();
    let one = Threads::Count(NonZeroUsize::MIN);
    let b = |i: usize| ((i * 7919) % 101) as f64 / 50.0 - 1.0;
    let mut rows: Vec<[f64; 2]> = (0..n).map(|i| [b(i), 1.0]).collect();
    let mut column: Vec<f64> = (0..n).map(b).collect();
    let mut data: Vec<f64> = (0..2 * n).map(b).collect();
    let mut major = ColumnMajor::new(&mut data, n).unwrap();
    let mut apart = Apart([column.clone(), column.clone()]);
    let mut scratch = MemBuffer::new(factor.scratch(2, one));
    let stack = MemStack::new(&mut scratch);

    let counted = allocations(|| {
        for block in [
            &mut rows as &mut dyn MatrixMut<f64>,
            &mut column,
            &mut major,
            &mut apart,
        ] {
            factor.apply_with(block, one, stack).unwrap();
        }
    });

    assert_eq!(counted, 0, "allocations of apply_with at Count(1)");
    assert!(
        rayon::ThreadPoolBuilder::new().build_global().is_ok(),
        "apply_with at Count(1) built rayon's global pool"
    );
}
