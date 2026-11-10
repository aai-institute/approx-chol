//! How many threads a solve may use.

use std::num::NonZeroUsize;

/// The threads a solve may use; the solution's bits do not depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    /// Every thread of the caller's rayon pool.
    All,
    /// At most `n` threads at a time, whatever the pool's size; `Count(1)` runs on the calling thread.
    Count(NonZeroUsize),
}

/// The calling thread alone.
pub(crate) const ONE: Threads = Threads::Count(NonZeroUsize::MIN);

impl Threads {
    /// The most threads it runs at a time: no more than the current pool has; the calling thread alone asks no pool, as outside one rayon would build its global pool.
    pub(crate) fn tasks(self) -> usize {
        match self {
            ONE => 1,
            Threads::All => rayon::current_num_threads(),
            Threads::Count(n) => n.get().min(rayon::current_num_threads()),
        }
    }

    /// Units per task, at least `grain`, in at most [`tasks`](Self::tasks) tasks under a count; `None` when one task takes all `len` or one thread runs, which leaves them on the calling thread.
    pub(crate) fn per_task(self, len: usize, grain: usize) -> Option<usize> {
        let tasks = self.tasks();
        let per = match self {
            Threads::All => grain,
            Threads::Count(_) => grain.max(len.div_ceil(tasks)),
        }
        .max(1);
        (per < len && tasks > 1).then_some(per)
    }

    /// The budget of work nested in each of `tasks` tasks of a split loop: an equal share of a count, at least one.
    pub(crate) fn within_task(self, tasks: usize) -> Self {
        match self {
            Threads::All => Threads::All,
            Threads::Count(n) => NonZeroUsize::new(n.get() / tasks).map_or(ONE, Threads::Count),
        }
    }
}

/// Runs one item of a split build loop; under test, `probe` counts the threads inside one at once.
#[cfg(not(test))]
#[inline(always)]
pub(crate) fn task<R>(run: impl FnOnce() -> R) -> R {
    run()
}

#[cfg(test)]
pub(crate) use probe::task;

#[cfg(test)]
pub(crate) mod probe {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

    /// The name prefix of the threads counted: other tests build concurrently on others.
    pub(crate) const POOL: &str = "budget-probe";

    static ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static DEPTH: Cell<usize> = const { Cell::new(0) };
    }

    pub(crate) fn task<R>(run: impl FnOnce() -> R) -> R {
        let counted = (std::thread::current().name()).is_some_and(|name| name.starts_with(POOL));
        if !counted {
            return run();
        }
        // A thread waiting on its own loop runs other items: it counts once.
        let entered = DEPTH.with(|depth| depth.replace(depth.get() + 1)) == 0;
        if entered {
            PEAK.fetch_max(ACTIVE.fetch_add(1, SeqCst) + 1, SeqCst);
        }
        let result = run();
        DEPTH.with(|depth| depth.set(depth.get() - 1));
        if entered {
            ACTIVE.fetch_sub(1, SeqCst);
        }
        result
    }

    /// The most counted threads inside an item at once while `run` runs.
    pub(crate) fn peak(run: impl FnOnce()) -> usize {
        PEAK.store(0, SeqCst);
        run();
        PEAK.load(SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    #[test]
    fn all_splits_by_the_grain() {
        pool(4).install(|| {
            for len in [0, 1, 2, 7, 100, 4097] {
                for grain in [0, 1, 625] {
                    let per = grain.max(1);
                    let split = Threads::All.per_task(len, grain);
                    assert_eq!(split, (per < len).then_some(per));
                }
            }
        });
    }

    #[test]
    fn a_count_splits_into_at_most_its_tasks() {
        pool(9).install(|| {
            for n in 1..=9 {
                let threads = Threads::Count(NonZeroUsize::new(n).unwrap());
                for len in [0, 1, 2, 7, 8, 9, 100, 4097, 100_003] {
                    for grain in [0, 1, 625] {
                        let Some(per) = threads.per_task(len, grain) else {
                            assert!(
                                n == 1 || len <= grain.max(1),
                                "{len} rows unsplit under {n}"
                            );
                            continue;
                        };
                        let tasks = len.div_ceil(per);
                        assert!(
                            per >= grain && (2..=n).contains(&tasks),
                            "{len} in {tasks} under {n}"
                        );
                        if grain <= 1 && len >= n * n {
                            assert_eq!(tasks, n, "{len} rows under {n}");
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn a_count_splits_into_at_most_the_pools_threads() {
        let eight = Threads::Count(NonZeroUsize::new(8).unwrap());
        pool(3).install(|| {
            assert_eq!(eight.per_task(100, 1), Some(34));
        });
        pool(1).install(|| {
            for threads in [Threads::All, eight] {
                assert_eq!(threads.per_task(100, 1), None, "{threads:?}");
            }
        });
    }

    #[test]
    fn nested_work_shares_a_count() {
        let count = |n| Threads::Count(NonZeroUsize::new(n).unwrap());
        assert_eq!(Threads::All.within_task(5), Threads::All);
        assert_eq!(count(7).within_task(2), count(3));
        assert_eq!(count(3).within_task(3), ONE);
        assert_eq!(count(2).within_task(3), ONE);
    }
}
