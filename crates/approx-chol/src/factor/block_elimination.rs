//! A bipartite block's eliminated side as `L`'s first block column, and the solve `M⁻¹x` around `S`'s.

use std::ops::Range;

use rayon::prelude::*;

use super::{within_one, MULTIPLY_ADD_GRAIN_BITS};
use crate::storage::Rows;
use crate::{Batch, Index, MemStack, Real, Threads};

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
    /// The forward's leaves of eliminated rows, fixed by the shape so its sums are too; [`Threads::per_task`] only groups them.
    #[cfg_attr(feature = "serde", serde(skip))]
    leaves: Box<[usize]>,
}

impl<T: Real, I: Index> BlockElimination<T, I> {
    /// `C`'s rows scaled by the pivots `S` was formed with.
    pub(super) fn new(mut cross: Rows<T, I>, inv_pivot: Box<[T]>) -> Self {
        cross.scale_rows(&inv_pivot);
        Self::scaled(cross, inv_pivot)
    }

    fn scaled(cross: Rows<T, I>, inv_pivot: Box<[T]>) -> Self {
        debug_assert!(rows_within_one(&cross), "a D_e⁻¹C row sums past one");
        let leaves = leaves(&cross);
        Self {
            cross,
            inv_pivot,
            leaves,
        }
    }

    pub(super) fn n_eliminated(&self) -> usize {
        self.inv_pivot.len()
    }

    /// Slots the forward takes from a solve's stack, freed before `reduced` runs: one sum per leaf and kept row.
    pub(super) fn scratch_slots(&self) -> usize {
        match self.leaves.len() - 1 {
            0 | 1 => 0,
            n_leaves => n_leaves * self.n_kept(),
        }
    }

    pub(super) fn n_kept(&self) -> usize {
        self.cross.n_cols()
    }

    /// `x ← M⁻¹x` through a stack of at least [`scratch_slots`](Self::scratch_slots), with `reduced` solving `S` in place on the kept half between the two products.
    pub(super) fn solve<V: Batch<T>>(
        &self,
        x: &mut [V],
        threads: Threads,
        stack: &mut MemStack,
        reduced: impl FnOnce(&mut [V], &mut MemStack),
    ) {
        let (eliminated, kept) = x.split_at_mut(self.n_eliminated());
        self.forward(eliminated, kept, threads, stack);
        reduced(kept, stack);
        self.backward(eliminated, kept, threads);
    }

    /// Each leaf sums `−(D_e⁻¹ C)_ek b_e` in ascending `e` from zero, and each `b_k` adds its leaves' sums in leaf order: the same bits under any threads.
    fn forward<V: Batch<T>>(
        &self,
        b: &[V],
        kept: &mut [V],
        threads: Threads,
        stack: &mut MemStack,
    ) {
        let c = &self.cross;
        let scatter = |rows: Range<usize>, sums: &mut [V]| {
            for e in rows {
                for (k, &c) in c.row(e) {
                    sums[k] = sums[k].sub(b[e].mul(c));
                }
            }
        };
        let (bounds, n_kept) = (&self.leaves, kept.len());
        let n_leaves = bounds.len() - 1;
        if n_leaves < 2 {
            scatter(0..b.len(), kept);
            return;
        }
        let (mut partial, _) = stack.make_with(n_leaves * n_kept, |_| V::zero());
        let leaf = |l: usize, sums: &mut [V]| scatter(bounds[l]..bounds[l + 1], sums);
        match threads.per_task(n_leaves, 1) {
            Some(per) => {
                (partial.par_chunks_mut(per * n_kept).enumerate()).for_each(|(task, sums)| {
                    (task * per..)
                        .zip(sums.chunks_mut(n_kept))
                        .for_each(|(l, s)| leaf(l, s));
                })
            }
            None => (partial.chunks_mut(n_kept).enumerate()).for_each(|(l, s)| leaf(l, s)),
        }
        let add = |first: usize, b_ks: &mut [V]| {
            for sums in partial.chunks(n_kept) {
                for (b_k, &s) in b_ks.iter_mut().zip(&sums[first..]) {
                    *b_k = b_k.add(s);
                }
            }
        };
        match threads.per_task(n_kept, (1 << LEAF_BITS) / n_leaves + 1) {
            Some(per) => (kept.par_chunks_mut(per).enumerate())
                .for_each(|(task, b_ks)| add(task * per, b_ks)),
            None => add(0, kept),
        }
    }

    /// `x_e = D_e⁻¹ b_e − D_e⁻¹ C x_k` by eliminated row, split as [`Threads::per_task`] says past one grain of multiply-adds.
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
        let splits = self.cross.nnz() > 1 << MULTIPLY_ADD_GRAIN_BITS;
        if let (true, Some(per)) = (splits, threads.per_task(len, grain)) {
            (eliminated.par_chunks_mut(per).enumerate())
                .for_each(|(task, ys)| rows(task * per, ys));
        } else {
            rows(0, eliminated);
        }
    }
}

/// Multiply-adds a forward leaf reaches at least.
const LEAF_BITS: u32 = 14;

/// Row bounds of the forward's leaves: one past one grain of multiply-adds, else each reaching `max(16·n_kept, 2^LEAF_BITS)`, so adding the leaves' sums costs a sixteenth.
fn leaves<T: Real, I: Index>(cross: &Rows<T, I>) -> Box<[usize]> {
    let n = cross.n_rows();
    if cross.nnz() <= 1 << MULTIPLY_ADD_GRAIN_BITS {
        return [0, n].into();
    }
    let target = (16 * cross.n_cols()).max(1 << LEAF_BITS);
    let mut bounds = vec![0];
    let mut filled = 0;
    for e in 0..n {
        filled += cross.row_len(e);
        if filled >= target {
            bounds.push(e + 1);
            filled = 0;
        }
    }
    if bounds.last() != Some(&n) {
        bounds.push(n);
    }
    bounds.into()
}

/// Each `|D_e⁻¹C|` row sums to at most one: the backward sum cannot overflow where the solution fits.
fn rows_within_one<T: Real, I: Index>(cross: &Rows<T, I>) -> bool {
    (0..cross.n_rows()).all(|e| within_one(cross.row(e).map(|(_, &c)| c)))
}

#[cfg(feature = "serde")]
checked_serde!(BlockElimination);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> BlockElimination<T, I> {
    /// The leaves are derived, so they are rebuilt rather than read: stored ones could disagree with `D_e⁻¹ C`.
    fn checked(self) -> Option<Self> {
        let pivots = self.inv_pivot.len() == self.cross.n_rows()
            && self
                .inv_pivot
                .iter()
                .all(|p| p.is_finite() && *p > T::zero());
        let (n_rows, n_cols) = (self.cross.n_rows(), self.cross.n_cols());
        // The larger side is eliminated, which bounds `n_cols` by decoded rows before `n` or the leaves exist.
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

        const LANES: usize = 1;

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
        // Odd lengths, which a halving splitter leaves in fewer than `n` tasks; ~61 leaves so every `n ≤ 8` groups them into `n`.
        let (n_eliminated, n_kept, quarter) = (250_001, 101, 25);
        let ptrs = (0..=n_eliminated).map(|e| 4 * e).collect();
        let indices = (0..n_eliminated)
            .flat_map(|e| (0..4).map(move |j| (e % quarter + j * quarter) as u32))
            .collect();
        let cross = Rows::new(ptrs, indices, vec![0.25; 4 * n_eliminated], n_kept);
        let elimination = BlockElimination::new(cross, vec![1.0; n_eliminated].into());
        assert!(elimination.leaves.len() > 60, "the forward may split");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let mut scratch =
            crate::MemBuffer::new(crate::StackReq::new::<Traced>(elimination.scratch_slots()));
        let stack = MemStack::new(&mut scratch);
        let mut x = vec![Traced(1.0); n_eliminated + n_kept];
        let (eliminated, kept) = x.split_at_mut(n_eliminated);
        for n in 1..=8 {
            let threads = Threads::Count(NonZeroUsize::new(n).unwrap());
            let forward = pool
                .install(|| workers(n, || elimination.forward(eliminated, kept, threads, stack)));
            let backward =
                pool.install(|| workers(n, || elimination.backward(eliminated, kept, threads)));
            for used in [forward, backward] {
                assert_eq!(used.len(), n, "{used:?} under {n}");
            }
        }
        let alone = workers(1, || {
            elimination.solve(&mut x, crate::threads::ONE, stack, |_, _| ())
        });
        assert_eq!(
            alone,
            BTreeSet::from([None]),
            "Count(1) stays on the caller"
        );
    }
}
