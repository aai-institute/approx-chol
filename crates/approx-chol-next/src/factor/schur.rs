//! Block elimination of a bipartite component's larger, independent side, and the Schur complement it leaves.

use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;

use super::cholesky::CholeskyInput;
use super::clique_tree::{CliqueSampler, Copies, Fill, Multi, Single, StarStreams};
use super::config::{CliqueRule, InnerParallelism, Multiplicity};
use super::sddm::{from_fill, RowSums, Sddm, Sheets};
use crate::sgdd::{Balance, Grounding};
use crate::storage::{narrow, widen, CrossBlock, UpperAdjacency};
use crate::Real;

/// `M`'s cross block and the pivots `d_e`, applied as two products because the block's steps commute.
#[derive(Clone, Debug)]
pub(super) struct BlockElimination<T> {
    /// `M`'s own `C`, signs kept and shared with the component: a frustrated component's cover never doubles it.
    cross: Arc<CrossBlock<T>>,
    inv_pivot: Box<[T]>,
}

impl<T: Real> BlockElimination<T> {
    pub(super) fn n_eliminated(&self) -> usize {
        self.inv_pivot.len()
    }

    /// `2·nnz(C) + n_eliminated + n_kept`, within's estimate, when it [`splits`](Self::splits), else `0`.
    pub(super) fn splittable_work(&self) -> usize {
        match self.splits() {
            true => 2 * self.cross.c().nnz() + self.cross.n(),
            false => 0,
        }
    }

    /// The larger side past 10 000 rows (0.6's threshold): the one test `Allow` and the work estimate share.
    fn splits(&self) -> bool {
        self.cross.n_eliminated() > 10_000
    }

    /// `y_e = D_e⁻¹ b_e`, then each `b_k` subtracts `c_ek y_e` term by term in ascending `e`: scattered through `C`, or gathered through `Cᵀ` when `Allow` splits it, the same bits either way, so only a split builds `Cᵀ`.
    pub(super) fn forward(&self, x: &mut [T], inner: InnerParallelism) {
        let (eliminated, kept) = x.split_at_mut(self.n_eliminated());
        for (y, &inv_pivot) in eliminated.iter_mut().zip(&self.inv_pivot) {
            *y = *y * inv_pivot;
        }
        let eliminated = &*eliminated;
        if inner == InnerParallelism::Allow && self.splits() {
            let ct = self.cross.ct();
            kept.par_iter_mut().enumerate().for_each(|(k, b)| {
                for (e, &c) in ct.row(k) {
                    *b = *b - c * eliminated[e];
                }
            });
        } else {
            let c = self.cross.c();
            for (e, &y) in eliminated.iter().enumerate() {
                for (k, &c) in c.row(e) {
                    kept[k] = kept[k] - c * y;
                }
            }
        }
    }

    /// `x_e = y_e − D_e⁻¹ C x_k`, by eliminated row, split under `Allow` when it [`splits`](Self::splits).
    pub(super) fn backward(&self, eliminated: &mut [T], kept: &[T], inner: InnerParallelism) {
        let c = self.cross.c();
        let row = |e: usize, y: &mut T| {
            let mut cx = T::zero();
            for (k, &c) in c.row(e) {
                cx = cx + c * kept[k];
            }
            *y = *y - self.inv_pivot[e] * cx;
        };
        if inner == InnerParallelism::Allow && self.splits() {
            eliminated
                .par_iter_mut()
                .enumerate()
                .for_each(|(e, y)| row(e, y));
        } else {
            eliminated
                .iter_mut()
                .enumerate()
                .for_each(|(e, y)| row(e, y));
        }
    }
}

/// `S = D_k − Cᵀ D_e⁻¹ C` before the elimination's stored form is taken, so one pivot vector serves every complement and the solve.
pub(super) struct SchurComplement<'a, T> {
    elimination: BlockElimination<T>,
    balance: &'a Balance<T>,
    /// How the approximate arm's `S` is built; a dense arm always takes it exact.
    rule: CliqueRule,
    stars: StarStreams,
}

impl<'a, T: Real> SchurComplement<'a, T> {
    /// Pivots are `Σ|c_ek| + s_e`, the same on both sheets of a cover.
    pub(super) fn new(
        cross: &Arc<CrossBlock<T>>,
        balance: &'a Balance<T>,
        rule: CliqueRule,
        stars: StarStreams,
    ) -> Self {
        let surplus = balance.grounding().surplus();
        let c = cross.c();
        let inv_pivot = (0..cross.n_eliminated())
            .map(|e| {
                let s_e = surplus.map_or(0.0, |s| widen(s[e]));
                let d_e = c.row(e).fold(s_e, |d, (_, &c)| d + widen(c.abs()));
                T::one() / narrow::<T>(d_e)
            })
            .collect();
        Self {
            elimination: BlockElimination {
                cross: Arc::clone(cross),
                inv_pivot,
            },
            balance,
            rule,
            stars,
        }
    }

    /// `S` on the component's sheets: exact by kept row, sampled by star in parallel.
    fn build(&self, rule: CliqueRule) -> Sddm<'static, T> {
        let n_kept = self.elimination.cross.n_kept();
        let (adjacency, ground) = match rule {
            CliqueRule::Exact => exact_rows(self),
            CliqueRule::Sampled(Multiplicity::Single) => {
                from_fill(n_kept, sampled_fill::<T, Single>(self, ()))
            }
            CliqueRule::Sampled(Multiplicity::Split(k)) => {
                from_fill(n_kept, sampled_fill::<T, Multi>(self, k))
            }
        };
        let kept = (self.balance.grounding().surplus()).map(|s| &s[s.len() - n_kept..]);
        let surplus = match kept {
            Some(kept) => (kept.iter().zip(ground)).map(|(&s, g)| s + g).collect(),
            None => ground,
        };
        let grounding = Grounding::of(surplus);
        Sddm::new(
            Cow::Owned(adjacency),
            Cow::Owned(grounding),
            Sheets::of(self.balance),
        )
    }

    pub(super) fn finish(self) -> BlockElimination<T> {
        self.elimination
    }
}

/// `S` built once, after its arm is chosen.
impl<'a, T: Real> CholeskyInput<'a, T> for &SchurComplement<'_, T> {
    fn dimension(&self) -> usize {
        Sheets::of(self.balance).count() * self.elimination.cross.n_kept()
    }

    fn for_dense(self) -> Sddm<'a, T> {
        self.build(CliqueRule::Exact)
    }

    fn for_approximate(self) -> Sddm<'a, T> {
        self.build(self.rule)
    }
}

/// Each eliminated star's clique tree under its own stream, the ground a final neighbor unless its share is zero; a row of `C` repeats no column, so stars skip the merge; chunks in star order, so merged duplicates sum alike at any thread count.
fn sampled_fill<T: Real, C: Copies>(
    schur: &SchurComplement<'_, T>,
    split: C::Split,
) -> Vec<Fill<T>> {
    if let Balance::Frustrated(_) = schur.balance {
        todo!("a cover samples each star on both sheets");
    }
    let cross = &schur.elimination.cross;
    let (c, ground) = (cross.c(), cross.n_kept() as u32);
    let surplus = schur.balance.grounding().surplus();
    (0..cross.n_eliminated())
        .into_par_iter()
        .fold(
            || (CliqueSampler::<T, C>::new(split), Fill::new()),
            |(mut sampler, mut fill), e| {
                let s_e = surplus.map(|s| s[e]).filter(|&s| s > T::zero());
                let neighbors = c.row(e).map(|(k, &c)| (k as u32, c.abs()));
                let neighbors = neighbors.chain(s_e.map(|s| (ground, s)));
                sampler.eliminate_unique(schur.stars.star(e), neighbors, &mut fill);
                (sampler, fill)
            },
        )
        .map(|(_, fill)| fill)
        .collect()
}

/// Rows of the cover's complement: `|c_ke||c_je|/d_e` to `j`'s same sheet when signs agree, else its other, as the smaller times the larger's share, the ground's `|c_ke| s_e/d_e` as column `n_kept`; a product underflowing to zero is dropped, as a stored zero couples nothing.
fn exact_rows<T: Real>(schur: &SchurComplement<'_, T>) -> (UpperAdjacency<T>, Vec<T>) {
    if let Balance::Frustrated(_) = schur.balance {
        todo!("a cover routes each product by its signs");
    }
    let BlockElimination { cross, inv_pivot } = &schur.elimination;
    let (c, ct, n_kept) = (cross.c(), cross.ct(), cross.n_kept());
    let surplus = schur.balance.grounding().surplus();
    let mut rows = RowSums::new(n_kept);
    for k in 0..n_kept {
        for (e, c_ek) in ct.row(k) {
            let c_ek = c_ek.abs();
            for (j, c_ej) in c.row(e).skip_while(|&(j, _)| j <= k) {
                rows.add(j, share(c_ek, c_ej.abs(), inv_pivot[e]));
            }
            if let Some(surplus) = surplus {
                rows.add(n_kept, share(c_ek, surplus[e], inv_pivot[e]));
            }
        }
        rows.end_row();
    }
    rows.finish()
}

/// `x·y/d` as the smaller times the larger's share `max·d⁻¹ ≤ 1`, so no product overflows before the division.
fn share<T: Real>(x: T, y: T, inv_pivot: T) -> T {
    x.min(y) * (x.max(y) * inv_pivot)
}
