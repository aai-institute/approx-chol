//! The Schur complement `S` a bipartite block leaves on its kept side, formed exact or sampled.

use rayon::prelude::*;

use super::block_elimination::BlockElimination;
use super::clique_tree::{CliqueTreeSampler, Copies, Multi, Single};
use super::config::{Config, SchurMode};
use super::sddm::{cover_vertex, crosses, Sddm, SddmRows, Sheets};
use super::{Streams, MULTIPLY_ADD_GRAIN_BITS};
use crate::sgdd::{Balance, Grounding};
use crate::storage::Rows;
use crate::{narrow, per_task, widen, Index, Real};

/// How one block's Schur complement is formed, resolved from [`Config`] and its dimension.
enum Formation {
    /// Each star's whole clique.
    Exact,
    /// Clique trees sampled with this split.
    Sampled(Option<Multi>),
}

/// How a complement of `dimension` is formed: exact when dense or asked, else sampled with its own split or under `Same` the factor's.
fn formation(config: &Config, dimension: usize) -> Formation {
    match config.complement {
        _ if config.dense(dimension) => Formation::Exact,
        SchurMode::Exact => Formation::Exact,
        SchurMode::Approximate { split_merge } => {
            Formation::Sampled(split_merge.and_then(Multi::split))
        }
        SchurMode::Same => Formation::Sampled(config.split()),
    }
}

/// The bipartite block `M = [D_e C; Cᵀ D_k]` on its sheets: `C` by eliminated row, and every row's surplus, eliminated rows first.
pub(super) struct BipartiteBlock<T, I> {
    cross: Rows<T, I>,
    sheets: Sheets,
    grounding: Grounding<T>,
}

impl<T: Real, I: Index> BipartiteBlock<T, I> {
    /// A balanced block folds onto one sheet; a frustrated one needs the cover's two.
    pub(super) fn of(cross: Rows<T, I>, balance: Balance<T>) -> Self {
        let (sheets, grounding) = match balance {
            Balance::Balanced(grounding) => (Sheets::One, grounding),
            Balance::Frustrated(grounding) => (Sheets::Two, grounding),
        };
        Self {
            cross,
            sheets,
            grounding,
        }
    }

    /// `L` and `factor(S)` over one pivot vector; σ is freed before `S` is factored, and `(D_e⁻¹ C)ᵀ` built after.
    pub(super) fn eliminate<R>(
        self,
        config: &Config,
        streams: Streams,
        factor: impl FnOnce(Sddm<T, I>) -> R,
    ) -> (BlockElimination<T, I>, R) {
        let inv_pivot = self.inverse_pivots();
        let (s, cross) = self.complement_and_cross(config, streams, &inv_pivot);
        let reduced = factor(s);
        (BlockElimination::new(cross, inv_pivot), reduced)
    }

    /// `S`, and `C` handed back for `L`: σ dies here, before `S` is factored.
    fn complement_and_cross(
        self,
        config: &Config,
        streams: Streams,
        inv_pivot: &[T],
    ) -> (Sddm<T, I>, Rows<T, I>) {
        let rows = match formation(config, self.complement_dimension()) {
            Formation::Exact => self.exact(inv_pivot),
            Formation::Sampled(split) => self.sample(split, streams).merge(),
        };
        (
            Sddm::of_rows(rows, self.sheets, &self.grounding),
            self.cross,
        )
    }

    /// `S` alone, a complement's summand: only exact computes pivots, and `C` is freed before the sampled merge.
    pub(super) fn complement(self, config: &Config, streams: Streams) -> Sddm<T, I> {
        let rows = match formation(config, self.complement_dimension()) {
            Formation::Exact => self.exact(&self.inverse_pivots()),
            Formation::Sampled(split) => {
                let fill = self.sample(split, streams);
                // The merge never reads `C`.
                drop(self.cross);
                fill.merge()
            }
        };
        Sddm::of_rows(rows, self.sheets, &self.grounding)
    }

    /// `S`'s dimension, a cover's two sheets counted.
    fn complement_dimension(&self) -> usize {
        self.sheets.count() * self.cross.n_cols()
    }

    /// Inverse pivots `1/(Σ|c_ek| + s_e)`, the same on both sheets of a cover.
    fn inverse_pivots(&self) -> Box<[T]> {
        let (c, grounding) = (&self.cross, &self.grounding);
        (0..c.n_rows())
            .map(|e| {
                let s_e = grounding.edge(e).map_or(0.0, widen);
                let d_e = c.row(e).fold(s_e, |d, (_, &c)| d + widen(c.abs()));
                T::one() / narrow::<T>(d_e)
            })
            .collect()
    }

    /// `S − diag(s_k)` exact by kept row, `S = D_k − Cᵀ D_e⁻¹ C`, through a `Cᵀ` freed before `of_rows` adds `s_k`.
    fn exact(&self, inv_pivot: &[T]) -> SddmRows<T, I> {
        match self.sheets {
            Sheets::One => self.exact_rows(inv_pivot, |j, _, _| j),
            Sheets::Two => self.exact_rows(inv_pivot, |j, c_ek, c_ej| {
                cover_vertex(j, usize::from(crosses(c_ek) != crosses(c_ej)))
            }),
        }
    }

    /// `|c_ke||c_je|/d_e` to `j`'s `vertex`, each row summing only `j > k` (a cover's `k⁻` mirrors `k⁺`), the ground's share last.
    fn exact_rows(
        &self,
        inv_pivot: &[T],
        vertex: impl Fn(usize, T, T) -> usize + Sync,
    ) -> SddmRows<T, I> {
        let (c, grounding) = (&self.cross, &self.grounding);
        let ct = c.transpose();
        let ground = self.complement_dimension();
        // Each task carries one grain of the complement's `nnz·(nnz/n_e)/2` multiply-adds.
        let work = c.nnz() as u64 * (c.nnz() / c.n_rows().max(1)) as u64 / 2;
        SddmRows::split(
            self.sheets,
            c.n_cols(),
            per_task(c.n_cols(), work, MULTIPLY_ADD_GRAIN_BITS),
            |mut sums, k| {
                for (e, &c_ek) in ct.row(k) {
                    for (j, &c_ej) in c.row(e).skip_while(|&(j, _)| j <= k) {
                        let w = share(c_ek.abs(), c_ej.abs(), inv_pivot[e]);
                        sums.add(vertex(j, c_ek, c_ej), w);
                    }
                    if let Some(s_e) = grounding.edge(e) {
                        sums.add(ground, share(c_ek.abs(), s_e, inv_pivot[e]));
                    }
                }
            },
        )
    }

    /// Clique trees need no pivots, so sampling reads only `C`, σ and the streams.
    fn sample(&self, split: Option<Multi>, streams: Streams) -> Fill<T, I> {
        let stars = Stars {
            block: self,
            streams,
            ground: I::from_index(self.complement_dimension()),
        };
        match split {
            None => stars.sample_with(Single),
            Some(split) => stars.sample_with(split),
        }
    }
}

/// The eliminated side's stars: row `e` of `C` on each sheet, its ground share last, each under its own stream.
struct Stars<'a, T, I> {
    block: &'a BipartiteBlock<T, I>,
    streams: Streams,
    ground: I,
}

impl<T: Real, I: Index> Stars<'_, T, I> {
    /// Star `s` sampled under its own stream; a row of `C` repeats no column, so it skips the merge.
    fn eliminate<C: Copies>(
        &self,
        s: usize,
        sampler: &mut CliqueTreeSampler<T, I, C>,
        fill: impl FnMut((I, I, C, T)),
    ) {
        let c = &self.block.cross;
        let ground = |e: usize| self.block.grounding.edge(e).map(|s_e| (self.ground, s_e));
        let draws = self.streams.star(s);
        match self.block.sheets {
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

    /// Each star's clique tree, chunked in star order so merged duplicates sum alike at any thread count.
    fn sample_with<C: Copies>(&self, split: C) -> Fill<T, I> {
        let n = self.block.complement_dimension();
        let c = &self.block.cross;
        let shift = bucket_shift(n);
        let count = n.div_ceil(1 << shift);
        let per_task = per_task(c.n_rows(), c.nnz() as u64, 14);
        let empty = || (CliqueTreeSampler::new(split), vec![Vec::new(); count]);
        let stars = self.block.sheets.count() * c.n_rows();
        let pieces = (0..stars)
            .into_par_iter()
            .fold_chunks(per_task, empty, |(mut sampler, mut buckets), s| {
                let push =
                    |(lo, hi, _, w): (I, I, C, T)| buckets[lo.index() >> shift].push((lo, hi, w));
                self.eliminate(s, &mut sampler, push);
                (sampler, buckets)
            })
            .map(|(_, mut buckets)| {
                buckets.iter_mut().for_each(Vec::shrink_to_fit);
                buckets
            })
            .collect();
        Fill { n, pieces }
    }
}

/// Rows per fill bucket, as a shift: at most 16 buckets, so one bucket's sorted copy is all the fill ever copied at once.
fn bucket_shift(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).saturating_sub(4)
}

/// One thread's fill edges `(lo, hi, weight)`, split by `lo` into the ranges of [`bucket_shift`].
type Buckets<T, I> = Vec<Vec<(I, I, T)>>;

/// Every task's sampled fill over the `n` rows it was bucketed by; an edge to `n` is `lo`'s ground weight.
struct Fill<T, I> {
    n: usize,
    pieces: Vec<Buckets<T, I>>,
}

impl<T: Real, I: Index> Fill<T, I> {
    /// Bucket by bucket, counting-sorts the chunks' rows unflattened, merges each row's duplicates, stores `−w`.
    fn merge(self) -> SddmRows<T, I> {
        let Self { n, mut pieces } = self;
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
                let (columns, weights) = (&columns[span[0]..span[1]], &weights[span[0]..span[1]]);
                rows.row(|mut sums| {
                    (columns.iter().zip(weights)).for_each(|(&hi, &w)| sums.add(hi.index(), w));
                });
            }
        }
        rows
    }
}

/// `x·y/d` as the smaller times the larger's share `max·d⁻¹ ≤ 1`, so no product overflows before the division; at least `MIN_POSITIVE`, so no edge of `S` drops and every pivot's inverse is finite.
fn share<T: Real>(x: T, y: T, inv_pivot: T) -> T {
    (x.min(y) * (x.max(y) * inv_pivot)).max(T::min_positive_value())
}
