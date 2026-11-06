//! A bipartite block's eliminated side as `L`'s first block column, and the solve `M⁻¹x` around `S`'s.

use rayon::prelude::*;

use super::{within_one, MULTIPLY_ADD_GRAIN_BITS};
use crate::storage::Rows;
use crate::{Batch, Index, Real, Threads};

/// `M`'s cross block scaled by its inverse pivots, `D_e⁻¹ C`, and `D_e⁻¹`, applied as two products because the block's steps commute.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self"),
    serde(bound(
        serialize = "T: serde::Serialize, I: crate::Index + serde::Serialize",
        deserialize = "T: crate::Real + serde::Deserialize<'de>, I: crate::Index + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub(super) struct BlockElimination<T, I> {
    /// `D_e⁻¹ C`, signs kept and shared with the block: a frustrated block's cover never doubles it; each row's magnitudes sum to at most one, so no product with a solution overflows.
    cross: Rows<T, I>,
    inv_pivot: Box<[T]>,
    /// `(D_e⁻¹ C)ᵀ` past one grain of the forward's `nnz` multiply-adds, so a solve may split by kept row as [`Threads::chunk`] says.
    #[cfg_attr(feature = "serde", serde(skip))]
    ct: Option<Rows<T, I>>,
}

impl<T: Real, I: Index> BlockElimination<T, I> {
    /// `C`'s rows scaled by the pivots `S` was formed with.
    pub(super) fn new(mut cross: Rows<T, I>, inv_pivot: Box<[T]>) -> Self {
        cross.scale_rows(&inv_pivot);
        Self::scaled(cross, inv_pivot)
    }

    fn scaled(cross: Rows<T, I>, inv_pivot: Box<[T]>) -> Self {
        debug_assert!(rows_within_one(&cross), "a D_e⁻¹C row sums past one");
        let ct = (cross.nnz() > 1 << MULTIPLY_ADD_GRAIN_BITS).then(|| cross.transpose());
        Self {
            cross,
            inv_pivot,
            ct,
        }
    }

    pub(super) fn n_eliminated(&self) -> usize {
        self.inv_pivot.len()
    }

    /// `2·nnz(C) + n_eliminated + n_kept`, within's estimate, when it splits, else `0`.
    pub(super) fn splittable_work(&self) -> usize {
        match self.ct {
            Some(_) => 2 * self.cross.nnz() + self.cross.n_rows() + self.cross.n_cols(),
            None => 0,
        }
    }

    /// `x ← M⁻¹x`, with `reduced` solving `S` in place on the kept half between the two products.
    pub(super) fn solve<V: Batch<T>>(
        &self,
        x: &mut [V],
        threads: Threads,
        reduced: impl FnOnce(&mut [V]),
    ) {
        let (eliminated, kept) = x.split_at_mut(self.n_eliminated());
        self.forward(eliminated, kept, threads);
        reduced(kept);
        self.backward(eliminated, kept, threads);
    }

    /// Each `b_k` subtracts `(D_e⁻¹ C)_ek b_e` term by term in ascending `e`: scattered through `D_e⁻¹ C`, or gathered through `(D_e⁻¹ C)ᵀ` where it is kept and [`Threads::chunk`] splits the rows, the same bits either way.
    fn forward<V: Batch<T>>(&self, b: &[V], kept: &mut [V], threads: Threads) {
        if let (Some(ct), Some(per)) = (&self.ct, threads.chunk(kept.len(), 1)) {
            let gather = |k: usize, b_k: &mut V| {
                for (e, &c) in ct.row(k) {
                    *b_k = b_k.sub(b[e].mul(c));
                }
            };
            // Rayon's adaptive split of single rows beats par_chunks_mut(1).
            if per == 1 {
                kept.par_iter_mut()
                    .enumerate()
                    .for_each(|(k, b_k)| gather(k, b_k));
            } else {
                (kept.par_chunks_mut(per).enumerate()).for_each(|(task, b_ks)| {
                    (task * per..).zip(b_ks).for_each(|(k, b_k)| gather(k, b_k));
                });
            }
        } else {
            let c = &self.cross;
            for (e, &b_e) in b.iter().enumerate() {
                for (k, &c) in c.row(e) {
                    kept[k] = kept[k].sub(b_e.mul(c));
                }
            }
        }
    }

    /// `x_e = D_e⁻¹ b_e − D_e⁻¹ C x_k` by eliminated row, split as [`Threads::chunk`] says where `(D_e⁻¹ C)ᵀ` is kept.
    fn backward<V: Batch<T>>(&self, eliminated: &mut [V], kept: &[V], threads: Threads) {
        let c = &self.cross;
        // Scaled here: a serial pass after the parallel gather rewrites lines its workers hold.
        let rows = |first: usize, ys: &mut [V]| {
            let inv_pivots = &self.inv_pivot[first..][..ys.len()];
            for (e, (y, &inv_pivot)) in (first..).zip(ys.iter_mut().zip(inv_pivots)) {
                let mut cx = V::zero();
                for (k, &c) in c.row(e) {
                    cx = cx.add(kept[k].mul(c));
                }
                *y = y.mul(inv_pivot).sub(cx);
            }
        };
        const ROWS_PER_TASK: usize = 4096;
        let len = eliminated.len();
        // A fixed grain, as TBB's simple partitioner: concurrent blocks share the pool; a short block still splits 16 ways.
        let grain = ROWS_PER_TASK.min(len.div_ceil(16));
        if let (Some(_), Some(per)) = (&self.ct, threads.chunk(len, grain)) {
            (eliminated.par_chunks_mut(per).enumerate())
                .for_each(|(task, ys)| rows(task * per, ys));
        } else {
            rows(0, eliminated);
        }
    }
}

/// Each `|D_e⁻¹C|` row sums to at most one: the backward sum cannot overflow where the solution fits.
fn rows_within_one<T: Real, I: Index>(cross: &Rows<T, I>) -> bool {
    (0..cross.n_rows()).all(|e| within_one(cross.row(e).map(|(_, &c)| c)))
}

#[cfg(feature = "serde")]
checked_serde!(BlockElimination);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> BlockElimination<T, I> {
    pub(super) fn n_kept(&self) -> usize {
        self.cross.n_cols()
    }

    /// `(D_e⁻¹ C)ᵀ` is derived, so it is rebuilt rather than read: a stored one could disagree with `D_e⁻¹ C`.
    fn checked(self) -> Option<Self> {
        let pivots = self.inv_pivot.len() == self.cross.n_rows()
            && self
                .inv_pivot
                .iter()
                .all(|p| p.is_finite() && *p > T::zero());
        let (n_rows, n_cols) = (self.cross.n_rows(), self.cross.n_cols());
        // The larger side is eliminated, which bounds `n_cols` by decoded rows before `n` or `(D_e⁻¹ C)ᵀ` exists.
        let sides = n_cols <= n_rows;
        (pivots && sides && rows_within_one(&self.cross))
            .then(|| Self::scaled(self.cross, self.inv_pivot))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    use super::*;

    static WORKERS: Mutex<BTreeSet<Option<usize>>> = Mutex::new(BTreeSet::new());
    static EXPECTED: AtomicUsize = AtomicUsize::new(1);

    #[derive(Clone, Copy)]
    struct Traced(f64);

    impl Batch<f64> for Traced {
        type Wide = f64;

        fn zero() -> Self {
            Traced(0.0)
        }

        fn add(self, other: Self) -> Self {
            Traced(self.0 + other.0)
        }

        fn sub(self, other: Self) -> Self {
            Traced(self.0 - other.0)
        }

        fn neg(self) -> Self {
            Traced(-self.0)
        }

        fn mul(self, by: f64) -> Self {
            let first = WORKERS
                .lock()
                .unwrap()
                .insert(rayon::current_thread_index());
            // Held until the pass's other workers arrive, so a pass of fewer tasks shows fewer workers.
            let deadline = Instant::now() + Duration::from_secs(2);
            while first
                && WORKERS.lock().unwrap().len() < EXPECTED.load(Ordering::Relaxed)
                && Instant::now() < deadline
            {
                std::hint::spin_loop();
            }
            Traced(self.0 * by)
        }

        fn div(self, by: f64) -> Self {
            Traced(self.0 / by)
        }

        fn widen(self) -> f64 {
            self.0
        }

        fn narrow(wide: f64) -> Self {
            Traced(wide)
        }
    }

    fn workers(expected: usize, run: impl FnOnce()) -> BTreeSet<Option<usize>> {
        EXPECTED.store(expected, Ordering::Relaxed);
        WORKERS.lock().unwrap().clear();
        run();
        std::mem::take(&mut *WORKERS.lock().unwrap())
    }

    #[test]
    fn each_pass_splits_across_exactly_its_threads() {
        // Odd lengths, which a halving splitter leaves in fewer than `n` tasks.
        let (n_eliminated, n_kept, quarter) = (40_001, 10_001, 2_500);
        let ptrs = (0..=n_eliminated).map(|e| 4 * e).collect();
        let indices = (0..n_eliminated)
            .flat_map(|e| (0..4).map(move |j| (e % quarter + j * quarter) as u32))
            .collect();
        let cross = Rows::new(ptrs, indices, vec![0.25; 4 * n_eliminated], n_kept);
        let elimination = BlockElimination::new(cross, vec![1.0; n_eliminated].into());
        assert!(elimination.ct.is_some(), "the solve may split");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mut x = vec![Traced(1.0); n_eliminated + n_kept];
        let (eliminated, kept) = x.split_at_mut(n_eliminated);
        for n in 1..=8 {
            let threads = Threads::Count(NonZeroUsize::new(n).unwrap());
            let forward =
                pool.install(|| workers(n, || elimination.forward(eliminated, kept, threads)));
            let backward =
                pool.install(|| workers(n, || elimination.backward(eliminated, kept, threads)));
            for used in [forward, backward] {
                assert_eq!(used.len(), n, "{used:?} under {n}");
            }
        }
        let one = Threads::Count(NonZeroUsize::MIN);
        let alone = workers(1, || elimination.solve(&mut x, one, |_| ()));
        assert_eq!(
            alone,
            BTreeSet::from([None]),
            "Count(1) stays on the caller"
        );
    }
}
