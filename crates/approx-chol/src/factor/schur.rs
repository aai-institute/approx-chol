//! Block elimination of a bipartite block's larger, independent side, and the Schur complement it leaves.

use std::borrow::Cow;

use rayon::prelude::*;

use rand::rngs::Xoshiro256PlusPlus;

use super::clique_tree::{CliqueSampler, Copies, Fill, Multi, Single};
use super::config::{InnerParallelism, Multiplicity};
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
    cross: Rows<T, I>,
    inv_pivot: Box<[T]>,
    /// `Cᵀ` where [`Self::transposes`], so `Allow` may split the solve by kept row.
    #[cfg_attr(feature = "serde", serde(skip))]
    ct: Option<Rows<T, I>>,
}

impl<T: Real, I: Index> BlockElimination<T, I> {
    pub(super) fn new(cross: Rows<T, I>, grounding: &Grounding<T>) -> Self {
        let inv_pivot = Self::pivots(&cross, grounding);
        Self::with_pivots(cross, inv_pivot)
    }

    /// Inverse pivots `1/(Σ|c_ek| + s_e)`, the same on both sheets of a cover.
    pub(super) fn pivots(c: &Rows<T, I>, grounding: &Grounding<T>) -> Box<[T]> {
        let surplus = grounding.surplus();
        (0..c.n_rows())
            .map(|e| {
                let s_e = surplus.map_or(0.0, |s| widen(s[e]));
                let d_e = c.row(e).fold(s_e, |d, (_, &c)| d + widen(c.abs()));
                T::one() / narrow::<T>(d_e)
            })
            .collect()
    }

    /// Whether an elimination of `c` keeps `Cᵀ`: past `2^17` entries, the forward's work, as `rows_per_task` splits it.
    pub(super) fn transposes(c: &Rows<T, I>) -> bool {
        c.nnz() > 1 << 17
    }

    pub(super) fn with_pivots(cross: Rows<T, I>, inv_pivot: Box<[T]>) -> Self {
        let ct = Self::transposes(&cross).then(|| cross.transpose());
        Self {
            cross,
            inv_pivot,
            ct,
        }
    }

    /// `C`, for `S` sampled after the pivots.
    pub(super) fn cross(&self) -> &Rows<T, I> {
        &self.cross
    }

    /// `S = D_k − Cᵀ D_e⁻¹ C` on `sheets`, exact by kept row.
    pub(super) fn complement(&self, sheets: Sheets, grounding: Grounding<T>) -> Sddm<T, I> {
        let surplus = grounding.surplus();
        let rows = match sheets {
            Sheets::One => exact_rows(self, surplus),
            Sheets::Two => cover_rows(self, surplus),
        };
        schur(rows, self.cross.n_cols(), sheets, surplus)
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

/// `S` sampled by star in parallel; clique trees need no pivots, so no elimination exists yet. σ is read last here, so it is freed before `S` is factored.
pub(super) fn sampled<T: Real, I: Index>(
    c: &Rows<T, I>,
    copies: Multiplicity,
    sheets: Sheets,
    grounding: Grounding<T>,
    stars: Streams,
) -> Sddm<T, I> {
    let (n_kept, surplus) = (c.n_cols(), grounding.surplus());
    let n = sheets.count() * n_kept;
    let ground = I::from_index(n);
    let ground_share = move |e: usize| {
        let s_e = surplus.map(|s| s[e]).filter(|&s| s > T::zero());
        s_e.map(|s| (ground, s))
    };
    let fill = match sheets {
        // One sheet maps no column, as the balanced loop pays for any map.
        Sheets::One => fill_stars(copies, c.n_rows(), n, |e| {
            let neighbors = c.row(e).map(|(k, &c)| (I::from_index(k), c.abs()));
            (stars.star(e), neighbors.chain(ground_share(e)))
        }),
        Sheets::Two => fill_stars(copies, 2 * c.n_rows(), n, |star| {
            let (sheet, e) = (star / c.n_rows(), star % c.n_rows());
            let neighbors = c.row(e).map(move |(k, &c)| {
                let on = sheet ^ usize::from(crosses(c));
                (I::from_index(cover_vertex(k, on)), c.abs())
            });
            (stars.star(star), neighbors.chain(ground_share(e)))
        }),
    };
    schur(from_fill(n, fill), n_kept, sheets, surplus)
}

/// Rows per fill bucket, as a shift: at most 16 buckets, so one bucket's sorted copy is all the fill ever copied at once.
fn bucket_shift(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).saturating_sub(4)
}

/// One thread's fill over `n` rows, split by `lo` into the ranges of [`bucket_shift`].
struct Buckets<T, I> {
    shift: u32,
    fill: Vec<Fill<T, I>>,
}

impl<T, I: Index> Buckets<T, I> {
    fn new(n: usize) -> Self {
        let shift = bucket_shift(n);
        let fill = (0..n.div_ceil(1 << shift)).map(|_| Fill::new()).collect();
        Self { shift, fill }
    }

    fn push(&mut self, edge: (I, I, T)) {
        self.fill[edge.0.index() >> self.shift].push(edge);
    }
}

/// The kept rows with each kept vertex's surplus added to its ground share.
fn schur<T: Real, I: Index>(
    (adjacency, ground): (UpperAdjacency<T, I>, Vec<T>),
    n_kept: usize,
    sheets: Sheets,
    surplus: Option<&[T]>,
) -> Sddm<T, I> {
    let kept = surplus.map(|s| &s[s.len() - n_kept..]);
    let surplus = match kept {
        Some(kept) => {
            let copies = kept
                .iter()
                .flat_map(|&s| std::iter::repeat_n(s, sheets.count()));
            copies.zip(ground).map(|(s, g)| s + g).collect()
        }
        None => ground,
    };
    Sddm::new(adjacency, Grounding::of(surplus), sheets)
}

/// Bucket by bucket, counting-sorts the chunks' rows unflattened, merges each row's duplicates, stores `−w`; an edge to `n` is `lo`'s ground weight.
fn from_fill<T: Real, I: Index>(
    n: usize,
    mut pieces: Vec<Buckets<T, I>>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let stride = 1 << bucket_shift(n);
    let count = n.div_ceil(stride);
    let bucket = |b: usize| pieces.iter().map(|p| p.fill[b].len()).sum();
    let most = (0..count).map(bucket).max().unwrap_or(0);
    let (mut columns, mut weights) = (vec![I::zero(); most], vec![T::zero(); most]);
    let (mut starts, mut next) = (Vec::new(), Vec::new());
    let mut rows = SddmRows::new(n);
    for b in 0..count {
        let chunks: Vec<Fill<T, I>> = (pieces.iter_mut())
            .map(|p| std::mem::take(&mut p.fill[b]))
            .collect();
        let first = b * stride;
        let len = (n - first).min(stride);
        starts.clear();
        starts.resize(len + 1, 0);
        // Nested loops, not `flatten()`: the flat iterator cost ~10% on both passes at 1M+ edges.
        for chunk in &chunks {
            for &(lo, _, _) in chunk {
                starts[lo.index() - first + 1] += 1;
            }
        }
        for i in 0..len {
            starts[i + 1] += starts[i];
        }
        next.clear();
        next.extend_from_slice(&starts[..len]);
        for chunk in chunks {
            for (lo, hi, w) in chunk {
                let at = next[lo.index() - first];
                columns[at] = hi;
                weights[at] = w;
                next[lo.index() - first] += 1;
            }
        }
        for span in starts.windows(2) {
            for (&hi, &w) in columns[span[0]..span[1]]
                .iter()
                .zip(&weights[span[0]..span[1]])
            {
                rows.add(hi.index(), w);
            }
            rows.end_row();
        }
    }
    rows.finish()
}

/// A row of `C` repeats no column, so stars skip the merge; chunks in star order, so merged duplicates sum alike at any thread count.
fn fill_stars<T: Real, I: Index, N: Iterator<Item = (I, T)>>(
    copies: Multiplicity,
    n_stars: usize,
    n: usize,
    star: impl Fn(usize) -> (Xoshiro256PlusPlus, N) + Sync,
) -> Vec<Buckets<T, I>> {
    match copies {
        Multiplicity::Single => sample_stars::<T, I, Single, N>((), n_stars, n, star),
        Multiplicity::Split(k) => sample_stars::<T, I, Multi, N>(k, n_stars, n, star),
    }
}

/// Each eliminated star's clique tree on each sheet under its own stream, the ground a final neighbor unless its share is zero; fill bucketed over `n` rows.
fn sample_stars<T: Real, I: Index, C: Copies, N: Iterator<Item = (I, T)>>(
    split: C::Split,
    n_stars: usize,
    n: usize,
    star: impl Fn(usize) -> (Xoshiro256PlusPlus, N) + Sync,
) -> Vec<Buckets<T, I>> {
    (0..n_stars)
        .into_par_iter()
        .fold(
            || (CliqueSampler::<T, I, C>::new(split), Buckets::new(n)),
            |(mut sampler, mut buckets), i| {
                let (draws, neighbors) = star(i);
                sampler.eliminate_unique(draws, neighbors, |edge| buckets.push(edge));
                (sampler, buckets)
            },
        )
        .map(|(_, mut buckets)| {
            buckets.fill.iter_mut().for_each(Vec::shrink_to_fit);
            buckets
        })
        .collect()
}

/// Rows of the cover's complement: `|c_ke||c_je|/d_e` to `j`'s same sheet when signs agree, else its other; `k⁻` mirrors `k⁺`, so each sums only `j > k`; the ground's share as column `2·n_kept`.
fn cover_rows<T: Real, I: Index>(
    elimination: &BlockElimination<T, I>,
    surplus: Option<&[T]>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let BlockElimination {
        cross, inv_pivot, ..
    } = elimination;
    let (c, ct, n_kept) = (cross, elimination.transpose(), cross.n_cols());
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
    elimination: &BlockElimination<T, I>,
    surplus: Option<&[T]>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let BlockElimination {
        cross, inv_pivot, ..
    } = elimination;
    let (c, ct, n_kept) = (cross, elimination.transpose(), cross.n_cols());
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
        let (n_rows, n_cols) = (self.cross.n_rows(), self.cross.n_cols());
        // The larger side is eliminated, which bounds `n_cols` by decoded rows before `n` or `Cᵀ` exists.
        let sides = n_cols <= n_rows;
        (pivots && sides).then(|| Self::with_pivots(self.cross, self.inv_pivot))
    }
}
