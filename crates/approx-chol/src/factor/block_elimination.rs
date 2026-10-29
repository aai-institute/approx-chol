//! A bipartite block's eliminated side as `L`'s first block column, and the solve `M⁻¹x` around `S`'s.

use rayon::prelude::*;

use super::config::InnerParallelism;
use super::{within_one, MULTIPLY_ADD_GRAIN_BITS};
use crate::storage::Rows;
use crate::{Index, Real};

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
    /// `(D_e⁻¹ C)ᵀ` past one grain of the forward's `nnz` multiply-adds, so `Allow` may split the solve by kept row.
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
    pub(super) fn solve(
        &self,
        x: &mut [T],
        inner: InnerParallelism,
        reduced: impl FnOnce(&mut [T]),
    ) {
        let (eliminated, kept) = x.split_at_mut(self.n_eliminated());
        self.forward(eliminated, kept, inner);
        reduced(kept);
        self.backward(eliminated, kept, inner);
    }

    /// Each `b_k` subtracts `(D_e⁻¹ C)_ek b_e` term by term in ascending `e`: scattered through `D_e⁻¹ C`, or gathered through `(D_e⁻¹ C)ᵀ` under `Allow` where it is kept, the same bits either way.
    fn forward(&self, b: &[T], kept: &mut [T], inner: InnerParallelism) {
        if let (InnerParallelism::Allow, Some(ct)) = (inner, &self.ct) {
            kept.par_iter_mut().enumerate().for_each(|(k, b_k)| {
                for (e, &c) in ct.row(k) {
                    *b_k = *b_k - c * b[e];
                }
            });
        } else {
            let c = &self.cross;
            for (e, &b_e) in b.iter().enumerate() {
                for (k, &c) in c.row(e) {
                    kept[k] = kept[k] - c * b_e;
                }
            }
        }
    }

    /// `x_e = D_e⁻¹ b_e − D_e⁻¹ C x_k` by eliminated row, split under `Allow` where `(D_e⁻¹ C)ᵀ` is kept.
    fn backward(&self, eliminated: &mut [T], kept: &[T], inner: InnerParallelism) {
        const ROWS_PER_TASK: usize = 4096;
        let c = &self.cross;
        // Scaled here: a serial pass after the parallel gather rewrites lines its workers hold.
        let rows = |first: usize, ys: &mut [T]| {
            let inv_pivots = &self.inv_pivot[first..][..ys.len()];
            for (e, (y, &inv_pivot)) in (first..).zip(ys.iter_mut().zip(inv_pivots)) {
                let mut cx = T::zero();
                for (k, &c) in c.row(e) {
                    cx = cx + c * kept[k];
                }
                *y = *y * inv_pivot - cx;
            }
        };
        if inner == InnerParallelism::Allow && self.ct.is_some() {
            // A fixed grain, as TBB's simple partitioner: concurrent blocks share the pool; a short block still splits 16 ways.
            let per = ROWS_PER_TASK.min(c.n_rows().div_ceil(16));
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
