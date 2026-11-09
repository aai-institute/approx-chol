use std::num::NonZeroUsize;

use approx_chol::Threads;

/// The calling thread alone.
pub const ONE: Threads = Threads::Count(NonZeroUsize::MIN);
/// The pool's threads, the calling one, and a cap below thread_invariance's largest pool.
pub const BUDGETS: [Threads; 3] = [
    Threads::All,
    ONE,
    Threads::Count(NonZeroUsize::new(3).unwrap()),
];

/// `run` under pools of one, two and seven threads: each result equals one thread's.
pub fn across_pools<R: PartialEq + Send>(run: impl Fn() -> R + Sync) {
    let at = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(&run)
    };
    let serial = at(1);
    for threads in [2, 7] {
        assert!(at(threads) == serial, "{threads} threads differ from 1");
    }
}
