//! Its own binary: the process must start without rayon's global pool, and the allocator counts every allocation.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use approx_chol::{
    factorize_with, ApproxInverse, ColumnMajor, Config, MatrixMut, MemBuffer, MemStack,
    ScalingPolicy, Threads,
};

#[path = "integration/common/apart.rs"]
mod apart;
#[path = "integration/common/bipartite.rs"]
mod bipartite;

use apart::Apart;
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

#[test]
fn an_apply_on_one_thread_allocates_nothing_and_builds_no_pool() {
    let factor = factored();
    let n = factor.n();
    let one = Threads::Count(NonZeroUsize::MIN);
    let b = |i: usize| ((i * 7919) % 101) as f64 / 50.0 - 1.0;
    let mut rows: Vec<[f64; 2]> = (0..n).map(|i| [b(i), 1.0]).collect();
    let mut column: Vec<f64> = (0..n).map(b).collect();
    // Nine columns run a whole pack and a column alone.
    let mut data: Vec<f64> = (0..9 * n).map(b).collect();
    let mut major = ColumnMajor::new(&mut data, n).unwrap();
    let mut apart = Apart(vec![column.clone(); 9]);
    // A standard-order array is staged, a Fortran-order one solved where its columns are.
    #[cfg(feature = "ndarray")]
    let mut arrays = {
        use ndarray::ShapeBuilder;
        let value = |(i, j): (usize, usize)| b(i + j * n);
        [(n, 9).set_f(false), (n, 9).f()].map(|shape| ndarray::Array2::from_shape_fn(shape, value))
    };
    let mut scratch = MemBuffer::new(factor.scratch(9, one));
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
        #[cfg(feature = "ndarray")]
        for array in &mut arrays {
            factor.apply_with(&mut **array, one, stack).unwrap();
        }
    });

    assert_eq!(counted, 0, "allocations of apply_with at Count(1)");
    assert!(
        rayon::ThreadPoolBuilder::new().build_global().is_ok(),
        "apply_with at Count(1) built rayon's global pool"
    );
}
