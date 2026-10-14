//! Block elimination of a bipartite block's larger, independent side, and the Schur complement it leaves.

use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;

use super::cholesky::CholeskyInput;
use rand::rngs::Xoshiro256PlusPlus;

use super::clique_tree::{CliqueSampler, Copies, Fill, Multi, Single};
use super::config::{CliqueRule, InnerParallelism, Multiplicity};
use super::sddm::{cover_vertex, crosses, Sddm, SddmRows, Sheets};
use super::Streams;
use crate::sgdd::Grounding;
use crate::storage::{Rows, UpperAdjacency};
use crate::{narrow, widen, Index, Real};

/// `M`'s cross block and the pivots `d_e`, applied as two products because the block's steps commute.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(remote = "Self"),
    serde(bound(
        deserialize = "T: crate::Real + serde::Deserialize<'de>, I: crate::Index + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub(super) struct BlockElimination<T, I> {
    /// `M`'s own `C`, signs kept and shared with the block: a frustrated block's cover never doubles it.
    cross: Arc<Rows<T, I>>,
    inv_pivot: Box<[T]>,
    /// `Cᵀ` where the larger side passes 10 000 rows (0.6's threshold), so `Allow` may split the solve by kept row.
    #[cfg_attr(feature = "serde", serde(skip))]
    ct: Option<Rows<T, I>>,
}

impl<T: Real, I: Index> BlockElimination<T, I> {
    /// Pivots are `Σ|c_ek| + s_e`, the same on both sheets of a cover.
    pub(super) fn new(cross: &Arc<Rows<T, I>>, grounding: &Grounding<T>) -> Self {
        let surplus = grounding.surplus();
        let c = &**cross;
        let inv_pivot = (0..c.n_rows())
            .map(|e| {
                let s_e = surplus.map_or(0.0, |s| widen(s[e]));
                let d_e = c.row(e).fold(s_e, |d, (_, &c)| d + widen(c.abs()));
                T::one() / narrow::<T>(d_e)
            })
            .collect();
        Self::with_pivots(Arc::clone(cross), inv_pivot)
    }

    fn with_pivots(cross: Arc<Rows<T, I>>, inv_pivot: Box<[T]>) -> Self {
        let ct = (cross.n_rows() > 10_000).then(|| cross.transpose());
        Self {
            cross,
            inv_pivot,
            ct,
        }
    }

    /// `S` on `sheets`, built only once its Cholesky arm is chosen.
    pub(super) fn complement<'a>(
        &'a self,
        sheets: Sheets,
        grounding: &'a Grounding<T>,
        rule: CliqueRule,
        stars: Streams,
    ) -> SchurComplement<'a, T, I> {
        SchurComplement {
            elimination: self,
            sheets,
            grounding,
            rule,
            stars,
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

    /// `Cᵀ`, borrowed where it is kept.
    fn transpose(&self) -> Cow<'_, Rows<T, I>> {
        (self.ct.as_ref()).map_or_else(|| Cow::Owned(self.cross.transpose()), Cow::Borrowed)
    }

    /// `y_e = D_e⁻¹ b_e`, then each `b_k` subtracts `c_ek y_e` term by term in ascending `e`: scattered through `C`, or gathered through `Cᵀ` under `Allow` where it is kept, the same bits either way.
    pub(super) fn forward(&self, x: &mut [T], inner: InnerParallelism) {
        let (eliminated, kept) = x.split_at_mut(self.n_eliminated());
        for (y, &inv_pivot) in eliminated.iter_mut().zip(&self.inv_pivot) {
            *y = *y * inv_pivot;
        }
        let eliminated = &*eliminated;
        if let (InnerParallelism::Allow, Some(ct)) = (inner, &self.ct) {
            kept.par_iter_mut().enumerate().for_each(|(k, b)| {
                for (e, &c) in ct.row(k) {
                    *b = *b - c * eliminated[e];
                }
            });
        } else {
            let c = &self.cross;
            for (e, &y) in eliminated.iter().enumerate() {
                for (k, &c) in c.row(e) {
                    kept[k] = kept[k] - c * y;
                }
            }
        }
    }

    /// `x_e = y_e − D_e⁻¹ C x_k`, by eliminated row, split under `Allow` where `Cᵀ` is kept.
    pub(super) fn backward(&self, eliminated: &mut [T], kept: &[T], inner: InnerParallelism) {
        let c = &self.cross;
        let row = |e: usize, y: &mut T| {
            let mut cx = T::zero();
            for (k, &c) in c.row(e) {
                cx = cx + c * kept[k];
            }
            *y = *y - self.inv_pivot[e] * cx;
        };
        if inner == InnerParallelism::Allow && self.ct.is_some() {
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

/// `S = D_k − Cᵀ D_e⁻¹ C` of a borrowed elimination, so one pivot vector serves every complement and the solve.
pub(super) struct SchurComplement<'a, T, I> {
    elimination: &'a BlockElimination<T, I>,
    sheets: Sheets,
    grounding: &'a Grounding<T>,
    /// How the approximate arm's `S` is built; a dense arm always takes it exact.
    rule: CliqueRule,
    stars: Streams,
}

impl<T: Real, I: Index> SchurComplement<'_, T, I> {
    /// `S` on the block's sheets: exact by kept row, sampled by star in parallel.
    fn build(&self, rule: CliqueRule) -> Sddm<'static, T, I> {
        let (n_kept, sheets) = (self.elimination.cross.n_cols(), self.sheets);
        let n = sheets.count() * n_kept;
        let (adjacency, ground) = match (rule, sheets) {
            (CliqueRule::Exact, Sheets::One) => exact_rows(self),
            (CliqueRule::Exact, Sheets::Two) => cover_rows(self),
            (CliqueRule::Sampled(Multiplicity::Single), _) => {
                from_fill(n, sampled_fill::<T, I, Single>(self, ()))
            }
            (CliqueRule::Sampled(Multiplicity::Split(k)), _) => {
                from_fill(n, sampled_fill::<T, I, Multi>(self, k))
            }
        };
        let kept = (self.grounding.surplus()).map(|s| &s[s.len() - n_kept..]);
        let surplus = match kept {
            Some(kept) => {
                let copies = kept
                    .iter()
                    .flat_map(|&s| std::iter::repeat_n(s, sheets.count()));
                copies.zip(ground).map(|(s, g)| s + g).collect()
            }
            None => ground,
        };
        let grounding = Grounding::of(surplus);
        Sddm::new(Cow::Owned(adjacency), Cow::Owned(grounding), sheets)
    }
}

/// `S` built once, after its arm is chosen.
impl<'a, T: Real, I: Index> CholeskyInput<'a, T, I> for SchurComplement<'_, T, I> {
    fn dimension(&self) -> usize {
        self.sheets.count() * self.elimination.cross.n_cols()
    }

    fn for_dense(self) -> Sddm<'a, T, I> {
        self.build(CliqueRule::Exact)
    }

    fn for_approximate(self) -> Sddm<'a, T, I> {
        self.build(self.rule)
    }
}

/// Counting-sorts chunks unflattened, merges each row's duplicates, stores `−w`; an edge to `n` is `lo`'s ground weight.
fn from_fill<T: Real, I: Index>(
    n: usize,
    chunks: Vec<Fill<T, I>>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let mut starts = vec![0; n + 1];
    // Nested loops, not `flatten()`: the flat iterator cost ~10% on both passes at 1M+ edges.
    for chunk in &chunks {
        for &(lo, _, _) in chunk {
            starts[lo.index() + 1] += 1;
        }
    }
    for i in 0..n {
        starts[i + 1] += starts[i];
    }
    let mut next = starts[..n].to_vec();
    let (mut columns, mut weights) = (vec![I::zero(); starts[n]], vec![T::zero(); starts[n]]);
    for chunk in chunks {
        for (lo, hi, w) in chunk {
            let at = next[lo.index()];
            columns[at] = hi;
            weights[at] = w;
            next[lo.index()] += 1;
        }
    }
    drop(next);
    let mut rows = SddmRows::new(n);
    for span in starts.windows(2) {
        for (&hi, &w) in columns[span[0]..span[1]]
            .iter()
            .zip(&weights[span[0]..span[1]])
        {
            rows.add(hi.index(), w);
        }
        rows.end_row();
    }
    rows.finish()
}

/// Each eliminated star's clique tree on each sheet under its own stream, the ground a final neighbor unless its share is zero; one sheet maps no column, as the balanced loop pays for any map.
fn sampled_fill<T: Real, I: Index, C: Copies>(
    schur: &SchurComplement<'_, T, I>,
    split: C::Split,
) -> Vec<Fill<T, I>> {
    let c = &schur.elimination.cross;
    let (n_eliminated, n_kept) = (c.n_rows(), c.n_cols());
    let sheets = schur.sheets;
    let ground = I::from_index(sheets.count() * n_kept);
    let surplus = schur.grounding.surplus();
    let ground_share = move |e: usize| {
        let s_e = surplus.map(|s| s[e]).filter(|&s| s > T::zero());
        s_e.map(|s| (ground, s))
    };
    match sheets {
        Sheets::One => fill_stars::<T, I, C, _>(split, n_eliminated, |e| {
            let neighbors = c.row(e).map(|(k, &c)| (I::from_index(k), c.abs()));
            (schur.stars.star(e), neighbors.chain(ground_share(e)))
        }),
        Sheets::Two => fill_stars::<T, I, C, _>(split, 2 * n_eliminated, |star| {
            let (sheet, e) = (star / n_eliminated, star % n_eliminated);
            let neighbors = c.row(e).map(move |(k, &c)| {
                let on = sheet ^ usize::from(crosses(c));
                (I::from_index(cover_vertex(k, on)), c.abs())
            });
            (schur.stars.star(star), neighbors.chain(ground_share(e)))
        }),
    }
}

/// A row of `C` repeats no column, so stars skip the merge; chunks in star order, so merged duplicates sum alike at any thread count.
fn fill_stars<T: Real, I: Index, C: Copies, N: Iterator<Item = (I, T)>>(
    split: C::Split,
    n_stars: usize,
    star: impl Fn(usize) -> (Xoshiro256PlusPlus, N) + Sync,
) -> Vec<Fill<T, I>> {
    (0..n_stars)
        .into_par_iter()
        .fold(
            || (CliqueSampler::<T, I, C>::new(split), Fill::new()),
            |(mut sampler, mut fill), i| {
                let (draws, neighbors) = star(i);
                sampler.eliminate_unique(draws, neighbors, &mut fill);
                (sampler, fill)
            },
        )
        .map(|(_, mut fill)| {
            fill.shrink_to_fit();
            fill
        })
        .collect()
}

/// Rows of the cover's complement: `|c_ke||c_je|/d_e` to `j`'s same sheet when signs agree, else its other; `k⁻` mirrors `k⁺`, so each sums only `j > k`; the ground's share as column `2·n_kept`.
fn cover_rows<T: Real, I: Index>(
    schur: &SchurComplement<'_, T, I>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let elimination = schur.elimination;
    let BlockElimination {
        cross, inv_pivot, ..
    } = elimination;
    let (c, ct, n_kept) = (cross, elimination.transpose(), cross.n_cols());
    let surplus = schur.grounding.surplus();
    SddmRows::split(Sheets::Two, n_kept, rows_per_task(c), |mut rows, k| {
        for (e, &c_ek) in ct.row(k) {
            for (j, &c_ej) in c.row(e).skip_while(|&(j, _)| j <= k) {
                let sheet = usize::from(crosses(c_ek) != crosses(c_ej));
                rows.add(
                    cover_vertex(j, sheet),
                    share(c_ek.abs(), c_ej.abs(), inv_pivot[e]),
                );
            }
            if let Some(surplus) = surplus {
                rows.add(2 * n_kept, share(c_ek.abs(), surplus[e], inv_pivot[e]));
            }
        }
    })
}

/// Rows of a folded block's complement, `|c_ke||c_je|/d_e` as the smaller times the larger's share, the ground's `|c_ke| s_e/d_e` as column `n_kept`; a product underflowing to zero is dropped, as a stored zero couples nothing.
fn exact_rows<T: Real, I: Index>(
    schur: &SchurComplement<'_, T, I>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let elimination = schur.elimination;
    let BlockElimination {
        cross, inv_pivot, ..
    } = elimination;
    let (c, ct, n_kept) = (cross, elimination.transpose(), cross.n_cols());
    let surplus = schur.grounding.surplus();
    SddmRows::split(Sheets::One, n_kept, rows_per_task(c), |mut rows, k| {
        for (e, c_ek) in ct.row(k) {
            let c_ek = c_ek.abs();
            for (j, c_ej) in c.row(e).skip_while(|&(j, _)| j <= k) {
                rows.add(j, share(c_ek, c_ej.abs(), inv_pivot[e]));
            }
            if let Some(surplus) = surplus {
                rows.add(n_kept, share(c_ek, surplus[e], inv_pivot[e]));
            }
        }
    })
}

/// Kept rows per task so each carries `2^17` of the complement's `nnz·(nnz/n_e)/2` multiply-adds, below which a split costs more than it saves.
fn rows_per_task<T: Real, I: Index>(c: &Rows<T, I>) -> usize {
    let work = c.nnz() * (c.nnz() / c.n_rows().max(1)) / 2;
    (c.n_cols() << 17).div_ceil(work.max(1))
}

/// `x·y/d` as the smaller times the larger's share `max·d⁻¹ ≤ 1`, so no product overflows before the division.
fn share<T: Real>(x: T, y: T, inv_pivot: T) -> T {
    x.min(y) * (x.max(y) * inv_pivot)
}

#[cfg(feature = "serde")]
checked_serde!(BlockElimination);

#[cfg(feature = "serde")]
impl<T: Real, I: Index> BlockElimination<T, I> {
    pub(super) fn n_kept(&self) -> usize {
        self.cross.n_cols()
    }

    /// `Cᵀ` is derived, so it is rebuilt rather than read: a stored one could disagree with `C`.
    fn checked(self) -> Option<Self> {
        let pivots = self.inv_pivot.len() == self.cross.n_rows()
            && self
                .inv_pivot
                .iter()
                .all(|p| p.is_finite() && *p > T::zero());
        pivots.then(|| Self::with_pivots(self.cross, self.inv_pivot))
    }
}
