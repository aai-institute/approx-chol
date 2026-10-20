//! Block elimination of a bipartite block's larger, independent side, and the Schur complement it leaves.

use rayon::prelude::*;

use rand::rngs::Xoshiro256PlusPlus;

use super::clique_tree::{CliqueTreeSampler, Copies, Multi, Single};
use super::config::InnerParallelism;
use super::sddm::{cover_vertex, crosses, Sddm, SddmRows, Sheets};
use super::{within_one, Streams};
use crate::sgdd::Grounding;
use crate::storage::{Rows, UpperAdjacency};
use crate::{narrow, per_task, widen, Index, Real};

/// `M`'s cross block scaled by its inverse pivots, `D_e⁻¹ C`, and `D_e⁻¹`, applied as two products because the block's steps commute.
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
    /// `D_e⁻¹ C`, signs kept and shared with the block: a frustrated block's cover never doubles it; each row's magnitudes sum to at most one, so no product with a solution overflows.
    cross: Rows<T, I>,
    inv_pivot: Box<[T]>,
    /// `Cᵀ` where [`Self::transposes`], so `Allow` may split the solve by kept row.
    #[cfg_attr(feature = "serde", serde(skip))]
    ct: Option<Rows<T, I>>,
}

impl<T: Real, I: Index> BlockElimination<T, I> {
    /// Inverse pivots `1/(Σ|c_ek| + s_e)`, the same on both sheets of a cover.
    pub(super) fn inverse_pivots(c: &Rows<T, I>, grounding: &Grounding<T>) -> Box<[T]> {
        (0..c.n_rows())
            .map(|e| {
                let s_e = grounding.edge(e).map_or(0.0, widen);
                let d_e = c.row(e).fold(s_e, |d, (_, &c)| d + widen(c.abs()));
                T::one() / narrow::<T>(d_e)
            })
            .collect()
    }

    /// Whether an elimination of `c` keeps `Cᵀ`: past `2^17` entries, the forward's work, as `rows_per_task` splits it.
    fn transposes(c: &Rows<T, I>) -> bool {
        c.nnz() > 1 << 17
    }

    pub(super) fn with_inverse_pivots(mut cross: Rows<T, I>, inv_pivot: Box<[T]>) -> Self {
        cross.scale_rows(&inv_pivot);
        Self::scaled(cross, inv_pivot)
    }

    fn scaled(cross: Rows<T, I>, inv_pivot: Box<[T]>) -> Self {
        debug_assert!(rows_within_one(&cross), "a D_e⁻¹C row sums past one");
        let ct = Self::transposes(&cross).then(|| cross.transpose());
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

    /// Each `b_k` subtracts `(D_e⁻¹ C)_ek b_e` term by term in ascending `e`: scattered through `C`, or gathered through `Cᵀ` under `Allow` where it is kept, the same bits either way.
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

    /// `x_e = D_e⁻¹ b_e − D_e⁻¹ C x_k` by eliminated row, split under `Allow` where `Cᵀ` is kept.
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

/// `S = D_k − Cᵀ D_e⁻¹ C` on `sheets`, exact by kept row through a `Cᵀ` freed before `S` is factored.
pub(super) fn exact<T: Real, I: Index>(
    c: &Rows<T, I>,
    inv_pivot: &[T],
    sheets: Sheets,
    grounding: Grounding<T>,
) -> Sddm<T, I> {
    let ct = c.transpose();
    let rows = match sheets {
        Sheets::One => exact_rows(c, &ct, inv_pivot, &grounding),
        Sheets::Two => cover_rows(c, &ct, inv_pivot, &grounding),
    };
    schur(rows, c.n_cols(), sheets, &grounding)
}

/// `S` sampled by star in parallel; clique trees need no pivots, so no elimination exists yet. σ is read last here, so it is freed before `S` is factored.
pub(super) fn sampled<T: Real, I: Index>(
    c: &Rows<T, I>,
    split: Option<Multi>,
    sheets: Sheets,
    grounding: Grounding<T>,
    stars: Streams,
) -> Sddm<T, I> {
    let n_kept = c.n_cols();
    let n = sheets.count() * n_kept;
    let ground = I::from_index(n);
    let ground_share = |e: usize| grounding.edge(e).map(|s| (ground, s));
    let stars_per_task = per_task(c.n_rows(), c.nnz() as u64, 14);
    let fill = match sheets {
        // One sheet maps no column, as the balanced loop pays for any map.
        Sheets::One => fill_stars(split, c.n_rows(), stars_per_task, n, |e| {
            let neighbors = c.row(e).map(|(k, &c)| (I::from_index(k), c.abs()));
            (stars.star(e), neighbors.chain(ground_share(e)))
        }),
        Sheets::Two => fill_stars(split, 2 * c.n_rows(), stars_per_task, n, |star| {
            let (sheet, e) = (star / c.n_rows(), star % c.n_rows());
            let neighbors = c.row(e).map(move |(k, &c)| {
                let on = sheet ^ usize::from(crosses(c));
                (I::from_index(cover_vertex(k, on)), c.abs())
            });
            (stars.star(star), neighbors.chain(ground_share(e)))
        }),
    };
    schur(from_fill(n, fill), n_kept, sheets, &grounding)
}

/// Rows per fill bucket, as a shift: at most 16 buckets, so one bucket's sorted copy is all the fill ever copied at once.
fn bucket_shift(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).saturating_sub(4)
}

/// One thread's fill edges `(lo, hi, weight)` over `n` rows, split by `lo` into the ranges of [`bucket_shift`].
type Buckets<T, I> = Vec<Vec<(I, I, T)>>;

/// The kept rows with each kept vertex's surplus added to its ground share.
fn schur<T: Real, I: Index>(
    (adjacency, mut ground): (UpperAdjacency<T, I>, Vec<T>),
    n_kept: usize,
    sheets: Sheets,
    grounding: &Grounding<T>,
) -> Sddm<T, I> {
    if let Some(surplus) = grounding.surplus() {
        let kept = surplus[surplus.len() - n_kept..].iter();
        let copies = kept.flat_map(|&s| std::iter::repeat_n(s, sheets.count()));
        (ground.iter_mut().zip(copies)).for_each(|(g, s)| *g = s + *g);
    }
    Sddm::new(adjacency, Grounding::of(ground), sheets)
}

/// Bucket by bucket, counting-sorts the chunks' rows unflattened, merges each row's duplicates, stores `−w`; an edge to `n` is `lo`'s ground weight.
fn from_fill<T: Real, I: Index>(
    n: usize,
    mut pieces: Vec<Buckets<T, I>>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let stride = 1 << bucket_shift(n);
    let count = n.div_ceil(stride);
    let bucket = |b: usize| pieces.iter().map(|p| p[b].len()).sum();
    let most = (0..count).map(bucket).max().unwrap_or(0);
    let (mut columns, mut weights) = (vec![I::zero(); most], vec![T::zero(); most]);
    let (mut starts, mut next) = (Vec::new(), Vec::new());
    let mut rows = SddmRows::new(n);
    for b in 0..count {
        let first = b * stride;
        let len = (n - first).min(stride);
        starts.clear();
        starts.resize(len + 1, 0);
        // Nested loops, not `flatten()`: the flat iterator cost ~10% on both passes at 1M+ edges.
        for piece in &pieces {
            for &(lo, _, _) in &piece[b] {
                starts[lo.index() - first + 1] += 1;
            }
        }
        for i in 0..len {
            starts[i + 1] += starts[i];
        }
        next.clear();
        next.extend_from_slice(&starts[..len]);
        for piece in &mut pieces {
            for (lo, hi, w) in std::mem::take(&mut piece[b]) {
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
    split: Option<Multi>,
    n_stars: usize,
    per_task: usize,
    n: usize,
    star: impl Fn(usize) -> (Xoshiro256PlusPlus, N) + Sync,
) -> Vec<Buckets<T, I>> {
    match split {
        None => sample_stars(Single, n_stars, per_task, n, star),
        Some(split) => sample_stars(split, n_stars, per_task, n, star),
    }
}

/// Each eliminated star's clique tree on each sheet under its own stream, the ground a final neighbor unless its share is zero; fill bucketed over `n` rows.
fn sample_stars<T: Real, I: Index, C: Copies, N: Iterator<Item = (I, T)>>(
    split: C,
    n_stars: usize,
    per_task: usize,
    n: usize,
    star: impl Fn(usize) -> (Xoshiro256PlusPlus, N) + Sync,
) -> Vec<Buckets<T, I>> {
    let shift = bucket_shift(n);
    let count = n.div_ceil(1 << shift);
    let empty = || (CliqueTreeSampler::new(split), vec![Vec::new(); count]);
    (0..n_stars)
        .into_par_iter()
        .fold_chunks(per_task, empty, |(mut sampler, mut buckets), i| {
            let (draws, neighbors) = star(i);
            let push =
                |(lo, hi, _, w): (I, I, C, T)| buckets[lo.index() >> shift].push((lo, hi, w));
            sampler.eliminate_unique(draws, neighbors, push);
            (sampler, buckets)
        })
        .map(|(_, mut buckets)| {
            buckets.iter_mut().for_each(Vec::shrink_to_fit);
            buckets
        })
        .collect()
}

/// Rows of the cover's complement: `|c_ke||c_je|/d_e` to `j`'s same sheet when signs agree, else its other; `k⁻` mirrors `k⁺`, so each sums only `j > k`; the ground's share as column `2·n_kept`.
fn cover_rows<T: Real, I: Index>(
    c: &Rows<T, I>,
    ct: &Rows<T, I>,
    inv_pivot: &[T],
    grounding: &Grounding<T>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let n_kept = c.n_cols();
    SddmRows::split(Sheets::Two, n_kept, rows_per_task(c), |mut rows, k| {
        for (e, &c_ek) in ct.row(k) {
            for (j, &c_ej) in c.row(e).skip_while(|&(j, _)| j <= k) {
                let sheet = usize::from(crosses(c_ek) != crosses(c_ej));
                rows.add(
                    cover_vertex(j, sheet),
                    share(c_ek.abs(), c_ej.abs(), inv_pivot[e]),
                );
            }
            if let Some(s_e) = grounding.edge(e) {
                rows.add(2 * n_kept, share(c_ek.abs(), s_e, inv_pivot[e]));
            }
        }
    })
}

/// Rows of a folded block's complement, `|c_ke||c_je|/d_e` as a [`share`], the ground's `|c_ke| s_e/d_e` as column `n_kept`.
fn exact_rows<T: Real, I: Index>(
    c: &Rows<T, I>,
    ct: &Rows<T, I>,
    inv_pivot: &[T],
    grounding: &Grounding<T>,
) -> (UpperAdjacency<T, I>, Vec<T>) {
    let n_kept = c.n_cols();
    SddmRows::split(Sheets::One, n_kept, rows_per_task(c), |mut rows, k| {
        for (e, c_ek) in ct.row(k) {
            let c_ek = c_ek.abs();
            for (j, c_ej) in c.row(e).skip_while(|&(j, _)| j <= k) {
                rows.add(j, share(c_ek, c_ej.abs(), inv_pivot[e]));
            }
            if let Some(s_e) = grounding.edge(e) {
                rows.add(n_kept, share(c_ek, s_e, inv_pivot[e]));
            }
        }
    })
}

/// Kept rows per task so each carries `2^17` of the complement's `nnz·(nnz/n_e)/2` multiply-adds, below which a split costs more than it saves.
fn rows_per_task<T: Real, I: Index>(c: &Rows<T, I>) -> usize {
    let work = c.nnz() as u64 * (c.nnz() / c.n_rows().max(1)) as u64 / 2;
    per_task(c.n_cols(), work, 17)
}

/// `x·y/d` as the smaller times the larger's share `max·d⁻¹ ≤ 1`, so no product overflows before the division; at least `MIN_POSITIVE`, so no edge of `S` drops and every pivot's inverse is finite.
fn share<T: Real>(x: T, y: T, inv_pivot: T) -> T {
    (x.min(y) * (x.max(y) * inv_pivot)).max(T::min_positive_value())
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
        (pivots && sides && rows_within_one(&self.cross))
            .then(|| Self::scaled(self.cross, self.inv_pivot))
    }
}
