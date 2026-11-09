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

/// faer's `Par::Seq` is `Count(1)`, `Par::Rayon(n)` is `Count(n)`.
#[cfg(feature = "faer")]
impl From<faer::Par> for Threads {
    fn from(par: faer::Par) -> Self {
        NonZeroUsize::new(par.degree()).map_or(ONE, Threads::Count)
    }
}

impl Threads {
    /// The most threads it runs at a time: no more than the current pool has; the calling thread alone asks no pool, as outside one rayon would build its global pool.
    pub(crate) fn tasks(self) -> usize {
        match self {
            ONE => 1,
            Threads::All => rayon::current_num_threads(),
            Threads::Count(n) => n.get().min(rayon::current_num_threads()),
        }
    }

    /// Rows per task, at least `grain`, in at most `n` tasks under `Count(n)`; `None` when one task takes all `len`, run on the calling thread.
    pub(crate) fn chunk(self, len: usize, grain: usize) -> Option<usize> {
        let per = match self {
            Threads::All => grain,
            Threads::Count(n) => grain.max(len.div_ceil(n.get())),
        }
        .max(1);
        (per < len).then_some(per)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_splits_by_the_grain() {
        for len in [0, 1, 2, 7, 100, 4097] {
            for grain in [0, 1, 625] {
                let per = grain.max(1);
                assert_eq!(Threads::All.chunk(len, grain), (per < len).then_some(per));
            }
        }
    }

    #[test]
    fn a_count_splits_into_at_most_its_tasks() {
        for n in 1..=9 {
            let threads = Threads::Count(NonZeroUsize::new(n).unwrap());
            for len in [0, 1, 2, 7, 8, 9, 100, 4097, 100_003] {
                for grain in [0, 1, 625] {
                    let Some(per) = threads.chunk(len, grain) else {
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
    }
}
