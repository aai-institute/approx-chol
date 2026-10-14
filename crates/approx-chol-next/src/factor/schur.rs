//! Block elimination of a bipartite component's larger, independent side, and the Schur complement it leaves.

use std::borrow::Cow;
use std::sync::Arc;

use rayon::prelude::*;

use super::cholesky::CholeskyInput;
use rand::rngs::SmallRng;

use super::clique_tree::{CliqueSampler, Copies, Fill, Multi, Single, Streams};
use super::config::{CliqueRule, InnerParallelism, Multiplicity};
use super::sddm::{crosses, Sddm, SddmRows, Sheets};
use crate::sgdd::Grounding;
use crate::storage::{CrossBlock, Rows, UpperAdjacency};
use crate::{narrow, widen, Real};

/// `M`'s cross block and the pivots `d_e`, applied as two products because the block's steps commute.
#[derive(Clone, Debug)]
pub(super) struct BlockElimination<T> {
    /// `M`'s own `C`, signs kept and shared with the component: a frustrated component's cover never doubles it.
    cross: Arc<CrossBlock<T>>,
    inv_pivot: Box<[T]>,
    /// `Cᵀ` where the larger side passes 10 000 rows (0.6's threshold), so `Allow` may split the solve by kept row.
    ct: Option<Rows<T>>,
}

impl<T: Real> BlockElimination<T> {
    /// Pivots are `Σ|c_ek| + s_e`, the same on both sheets of a cover.
    pub(super) fn new(cross: &Arc<CrossBlock<T>>, grounding: &Grounding<T>) -> Self {
        let surplus = grounding.surplus();
        let c = cross.c();
        let inv_pivot = (0..cross.n_eliminated())
            .map(|e| {
                let s_e = surplus.map_or(0.0, |s| widen(s[e]));
                let d_e = c.row(e).fold(s_e, |d, (_, &c)| d + widen(c.abs()));
                T::one() / narrow::<T>(d_e)
            })
            .collect();
        let ct = (cross.n_eliminated() > 10_000).then(|| c.transpose());
        Self {
            cross: Arc::clone(cross),
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
    ) -> SchurComplement<'a, T> {
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
            Some(_) => 2 * self.cross.c().nnz() + self.cross.n(),
            None => 0,
        }
    }

    /// `Cᵀ`, borrowed where it is kept.
    fn transpose(&self) -> Cow<'_, Rows<T>> {
        (self.ct.as_ref()).map_or_else(|| Cow::Owned(self.cross.c().transpose()), Cow::Borrowed)
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
            let c = self.cross.c();
            for (e, &y) in eliminated.iter().enumerate() {
                for (k, &c) in c.row(e) {
                    kept[k] = kept[k] - c * y;
                }
            }
        }
    }

    /// `x_e = y_e − D_e⁻¹ C x_k`, by eliminated row, split under `Allow` where `Cᵀ` is kept.
    pub(super) fn backward(&self, eliminated: &mut [T], kept: &[T], inner: InnerParallelism) {
        let c = self.cross.c();
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
pub(super) struct SchurComplement<'a, T> {
    elimination: &'a BlockElimination<T>,
    sheets: Sheets,
    grounding: &'a Grounding<T>,
    /// How the approximate arm's `S` is built; a dense arm always takes it exact.
    rule: CliqueRule,
    stars: Streams,
}

impl<T: Real> SchurComplement<'_, T> {
    /// `S` on the component's sheets: exact by kept row, sampled by star in parallel.
    fn build(&self, rule: CliqueRule) -> Sddm<'static, T> {
        let (n_kept, sheets) = (self.elimination.cross.n_kept(), self.sheets);
        let n = sheets.count() * n_kept;
        let (adjacency, ground) = match (rule, sheets) {
            (CliqueRule::Exact, Sheets::One) => exact_rows(self),
            (CliqueRule::Exact, Sheets::Two) => cover_rows(self),
            (CliqueRule::Sampled(Multiplicity::Single), _) => {
                from_fill(n, sampled_fill::<T, Single>(self, ()))
            }
            (CliqueRule::Sampled(Multiplicity::Split(k)), _) => {
                from_fill(n, sampled_fill::<T, Multi>(self, k))
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
impl<'a, T: Real> CholeskyInput<'a, T> for SchurComplement<'_, T> {
    fn dimension(&self) -> usize {
        self.sheets.count() * self.elimination.cross.n_kept()
    }

    fn for_dense(self) -> Sddm<'a, T> {
        self.build(CliqueRule::Exact)
    }

    fn for_approximate(self) -> Sddm<'a, T> {
        self.build(self.rule)
    }
}

/// Counting-sorts chunks unflattened, merges each row's duplicates, stores `−w`; an edge to `n` is `lo`'s ground weight.
fn from_fill<T: Real>(n: usize, chunks: Vec<Fill<T>>) -> (UpperAdjacency<T>, Vec<T>) {
    let mut starts = vec![0; n + 1];
    // Nested loops, not `flatten()`: the flat iterator cost ~10% on both passes at 1M+ edges.
    for chunk in &chunks {
        for &(lo, _, _) in chunk {
            starts[lo as usize + 1] += 1;
        }
    }
    for i in 0..n {
        starts[i + 1] += starts[i];
    }
    let mut next = starts[..n].to_vec();
    let mut by_row = vec![(0, T::zero()); starts[n]];
    for chunk in chunks {
        for (lo, hi, w) in chunk {
            by_row[next[lo as usize]] = (hi as usize, w);
            next[lo as usize] += 1;
        }
    }
    let mut rows = SddmRows::new(n);
    for span in starts.windows(2) {
        for &(hi, w) in &by_row[span[0]..span[1]] {
            rows.add(hi, w);
        }
        rows.end_row();
    }
    rows.finish()
}

/// Each eliminated star's clique tree on each sheet under its own stream, the ground a final neighbor unless its share is zero; one sheet maps no column, as the balanced loop pays for any map.
fn sampled_fill<T: Real, C: Copies>(
    schur: &SchurComplement<'_, T>,
    split: C::Split,
) -> Vec<Fill<T>> {
    let cross = &schur.elimination.cross;
    let (c, n_eliminated, n_kept) = (cross.c(), cross.n_eliminated(), cross.n_kept());
    let sheets = schur.sheets;
    let ground = (sheets.count() * n_kept) as u32;
    let surplus = schur.grounding.surplus();
    let ground_share = move |e: usize| {
        let s_e = surplus.map(|s| s[e]).filter(|&s| s > T::zero());
        s_e.map(|s| (ground, s))
    };
    match sheets {
        Sheets::One => fill_stars::<T, C, _>(split, n_eliminated, |e| {
            let neighbors = c.row(e).map(|(k, &c)| (k as u32, c.abs()));
            (schur.stars.star(e), neighbors.chain(ground_share(e)))
        }),
        Sheets::Two => fill_stars::<T, C, _>(split, 2 * n_eliminated, |star| {
            let (sheet, e) = (star / n_eliminated, star % n_eliminated);
            let neighbors = c.row(e).map(move |(k, &c)| {
                let on = sheet ^ usize::from(crosses(c));
                (sheets.vertex(k, on) as u32, c.abs())
            });
            (schur.stars.star(star), neighbors.chain(ground_share(e)))
        }),
    }
}

/// A row of `C` repeats no column, so stars skip the merge; chunks in star order, so merged duplicates sum alike at any thread count.
fn fill_stars<T: Real, C: Copies, N: Iterator<Item = (u32, T)>>(
    split: C::Split,
    n_stars: usize,
    star: impl Fn(usize) -> (SmallRng, N) + Sync,
) -> Vec<Fill<T>> {
    (0..n_stars)
        .into_par_iter()
        .fold(
            || (CliqueSampler::<T, C>::new(split), Fill::new()),
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
fn cover_rows<T: Real>(schur: &SchurComplement<'_, T>) -> (UpperAdjacency<T>, Vec<T>) {
    let elimination = schur.elimination;
    let BlockElimination {
        cross, inv_pivot, ..
    } = elimination;
    let (c, ct, n_kept) = (cross.c(), elimination.transpose(), cross.n_kept());
    let surplus = schur.grounding.surplus();
    let mut rows = SddmRows::new(2 * n_kept);
    for k in 0..n_kept {
        for (e, &c_ek) in ct.row(k) {
            for (j, &c_ej) in c.row(e).skip_while(|&(j, _)| j <= k) {
                let sheet = usize::from(crosses(c_ek) != crosses(c_ej));
                rows.add(
                    Sheets::Two.vertex(j, sheet),
                    share(c_ek.abs(), c_ej.abs(), inv_pivot[e]),
                );
            }
            if let Some(surplus) = surplus {
                rows.add(2 * n_kept, share(c_ek.abs(), surplus[e], inv_pivot[e]));
            }
        }
        rows.end_row();
        rows.mirror_row();
    }
    rows.finish()
}

/// Rows of a folded component's complement, `|c_ke||c_je|/d_e` as the smaller times the larger's share, the ground's `|c_ke| s_e/d_e` as column `n_kept`; a product underflowing to zero is dropped, as a stored zero couples nothing.
fn exact_rows<T: Real>(schur: &SchurComplement<'_, T>) -> (UpperAdjacency<T>, Vec<T>) {
    let elimination = schur.elimination;
    let BlockElimination {
        cross, inv_pivot, ..
    } = elimination;
    let (c, ct, n_kept) = (cross.c(), elimination.transpose(), cross.n_kept());
    let surplus = schur.grounding.surplus();
    let mut rows = SddmRows::new(n_kept);
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
