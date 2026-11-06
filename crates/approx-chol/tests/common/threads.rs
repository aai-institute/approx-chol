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
