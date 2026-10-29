//! Block elimination of a bipartite block's independent side, and the Schur complement it leaves.

use std::borrow::Borrow;

use rayon::prelude::*;

use super::block_elimination::BlockElimination;
use super::clique_tree::{CliqueTreeSampler, Copies, Multi, Single};
use super::config::{Config, SchurMode};
use super::sddm::{cover_vertex, crosses, Sddm, SddmRows, Sheets};
use super::Streams;
use crate::sgdd::{Balance, Grounding};
use crate::storage::{Rows, UpperAdjacency};
use crate::{narrow, per_task, widen, Index, Real};

impl<T: Real, I: Index> BlockElimination<T, I> {
    /// `L` and `factor(S)` over one pivot vector; `Cᵀ` is built once `factor` has consumed `S`, so it never shares that peak.
    pub(super) fn eliminate<R>(
        mut cross: Rows<T, I>,
        balance: Balance<T>,
        config: &Config,
        streams: Streams,
        factor: impl FnOnce(Sddm<T, I>) -> R,
    ) -> (Self, R) {
        let (sheets, grounding) = sheets(balance);
        let inv_pivot = Self::inverse_pivots(&cross, &grounding);
        let s = schur_complement(&cross, Some(&inv_pivot), sheets, grounding, config, streams);
        let reduced = factor(s);
        cross.scale_rows(&inv_pivot);
        (Self::scaled(cross, inv_pivot), reduced)
    }

    /// Inverse pivots `1/(Σ|c_ek| + s_e)`, the same on both sheets of a cover.
    fn inverse_pivots(c: &Rows<T, I>, grounding: &Grounding<T>) -> Box<[T]> {
        (0..c.n_rows())
            .map(|e| {
                let s_e = grounding.edge(e).map_or(0.0, widen);
                let d_e = c.row(e).fold(s_e, |d, (_, &c)| d + widen(c.abs()));
                T::one() / narrow::<T>(d_e)
            })
            .collect()
    }
}

/// `S` alone, a complement's summand: only exact computes pivots, and `C` is freed before the sampled merge.
pub(super) fn complement<T: Real, I: Index>(
    cross: Rows<T, I>,
    balance: Balance<T>,
    config: &Config,
    streams: Streams,
) -> Sddm<T, I> {
    let (sheets, grounding) = sheets(balance);
    schur_complement(cross, None, sheets, grounding, config, streams)
}

/// A balanced block folds onto one sheet; a frustrated one needs the cover's two.
fn sheets<T>(balance: Balance<T>) -> (Sheets, Grounding<T>) {
    match balance {
        Balance::Balanced(grounding) => (Sheets::One, grounding),
        Balance::Frustrated(grounding) => (Sheets::Two, grounding),
    }
}

/// `S`, `cross`'s eliminated rows' complement onto its columns: exact when dense or asked, else sampled; only exact needs the pivots.
fn schur_complement<T: Real, I: Index>(
    cross: impl Borrow<Rows<T, I>>,
    inv_pivot: Option<&[T]>,
    sheets: Sheets,
    grounding: Grounding<T>,
    config: &Config,
    streams: Streams,
) -> Sddm<T, I> {
    let c = cross.borrow();
    let dense = config.dense(sheets.count() * c.n_cols());
    match (dense, config.complement) {
        (true, _) | (false, SchurMode::Exact) => match inv_pivot {
            Some(inv_pivot) => exact(c, inv_pivot, sheets, grounding),
            None => {
                let inv_pivot = BlockElimination::inverse_pivots(c, &grounding);
                exact(c, &inv_pivot, sheets, grounding)
            }
        },
        (false, sampled) => {
            let split = match sampled {
                SchurMode::Approximate { split_merge } => split_merge,
                _ => config.split_merge,
            }
            .and_then(Multi::split);
            match split {
                None => self::sampled(cross, Single, sheets, grounding, streams),
                Some(split) => self::sampled(cross, split, sheets, grounding, streams),
            }
        }
    }
}

/// `S = D_k − Cᵀ D_e⁻¹ C` on `sheets`, exact by kept row through a `Cᵀ` freed before `S` is factored.
fn exact<T: Real, I: Index>(
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
fn sampled<T: Real, I: Index, C: Copies>(
    cross: impl Borrow<Rows<T, I>>,
    split: C,
    sheets: Sheets,
    grounding: Grounding<T>,
    streams: Streams,
) -> Sddm<T, I> {
    let c = cross.borrow();
    let n_kept = c.n_cols();
    let n = sheets.count() * n_kept;
    let stars = Stars {
        c,
        sheets,
        grounding: &grounding,
        streams,
        ground: I::from_index(n),
    };
    let fill = sample_stars(split, &stars, per_task(c.n_rows(), c.nnz() as u64, 14), n);
    // An owned `C` is freed here: the merge never reads it.
    drop(cross);
    schur(from_fill(n, fill), n_kept, sheets, &grounding)
}

/// The eliminated side's stars: row `e` of `C` on each sheet, its ground share last, each under its own stream.
struct Stars<'a, T, I> {
    c: &'a Rows<T, I>,
    sheets: Sheets,
    grounding: &'a Grounding<T>,
    streams: Streams,
    ground: I,
}

impl<T: Real, I: Index> Stars<'_, T, I> {
    fn count(&self) -> usize {
        self.sheets.count() * self.c.n_rows()
    }

    /// Star `s` sampled under its own stream; a row of `C` repeats no column, so it skips the merge.
    fn eliminate<C: Copies>(
        &self,
        s: usize,
        sampler: &mut CliqueTreeSampler<T, I, C>,
        fill: impl FnMut((I, I, C, T)),
    ) {
        let c = self.c;
        let ground = |e: usize| self.grounding.edge(e).map(|s_e| (self.ground, s_e));
        let draws = self.streams.star(s);
        match self.sheets {
            // One sheet maps no column, as the balanced loop pays for any map.
            Sheets::One => {
                let neighbors = c.row(s).map(|(k, &c)| (I::from_index(k), c.abs()));
                sampler.eliminate_unique(draws, neighbors.chain(ground(s)), fill);
            }
            Sheets::Two => {
                let (sheet, e) = (s / c.n_rows(), s % c.n_rows());
                let neighbors = c.row(e).map(move |(k, &c)| {
                    let on = sheet ^ usize::from(crosses(c));
                    (I::from_index(cover_vertex(k, on)), c.abs())
                });
                sampler.eliminate_unique(draws, neighbors.chain(ground(e)), fill);
            }
        }
    }
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
    Sddm::of_rows((adjacency, ground), sheets)
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

/// Each star's clique tree, chunked in star order so merged duplicates sum alike at any thread count; fill bucketed over `n` rows.
fn sample_stars<T: Real, I: Index, C: Copies>(
    split: C,
    stars: &Stars<'_, T, I>,
    per_task: usize,
    n: usize,
) -> Vec<Buckets<T, I>> {
    let shift = bucket_shift(n);
    let count = n.div_ceil(1 << shift);
    let empty = || (CliqueTreeSampler::new(split), vec![Vec::new(); count]);
    (0..stars.count())
        .into_par_iter()
        .fold_chunks(per_task, empty, |(mut sampler, mut buckets), s| {
            let push =
                |(lo, hi, _, w): (I, I, C, T)| buckets[lo.index() >> shift].push((lo, hi, w));
            stars.eliminate(s, &mut sampler, push);
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
