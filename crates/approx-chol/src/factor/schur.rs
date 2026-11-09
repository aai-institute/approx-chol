//! The Schur complement `S` a bipartite block leaves on its kept side, formed exact or sampled.

use rayon::prelude::*;

use super::block_elimination::BlockElimination;
use super::cholesky::{Arm, Signed};
use super::clique_tree::{CliqueTreeSampler, Copies, Multi, Single};
use super::config::{Config, SchurMode};
use super::sddm::{cover_vertex, crosses, Layout, Sddm, SddmRows};
use super::sign::Unsigned;
use super::{Streams, MULTIPLY_ADD_GRAIN_BITS};
use crate::sgdd::{Balance, Grounding};
use crate::storage::Rows;
use crate::{narrow, per_task, widen, Index, Real};

/// How one block's Schur complement is formed, resolved from [`Config`] and its kept side.
enum Formation {
    /// Exact and signed, for a dense Cholesky of `S` itself.
    Dense,
    /// Each star's whole clique.
    Exact,
    /// Clique trees sampled with these copies.
    Sampled(Option<Multi>),
}

/// How a complement on `n_kept` vertices is formed: signed when dense, exact when asked, else sampled with its own sampling or under `Same` the factor's.
fn formation(config: &Config, n_kept: usize) -> Formation {
    match config.complement {
        _ if config.dense(n_kept) => Formation::Dense,
        SchurMode::Exact => Formation::Exact,
        SchurMode::Approximate { sampling } => Formation::Sampled(sampling.copies()),
        SchurMode::Same => Formation::Sampled(config.sampling.copies()),
    }
}

/// The bipartite block `M = [D_e C; Cᵀ D_k]`: `C` by eliminated row, and every row's surplus, eliminated rows first.
pub(super) struct BipartiteBlock<T, I> {
    cross: Rows<T, I>,
    balance: Balance,
    grounding: Grounding<T>,
}

impl<T: Real, I: Index> BipartiteBlock<T, I> {
    pub(super) fn of(cross: Rows<T, I>, balance: Balance, grounding: Grounding<T>) -> Self {
        Self {
            cross,
            balance,
            grounding,
        }
    }

    /// `L` and `factor(S)` over one pivot vector; σ is freed before `S` is factored, and `(D_e⁻¹ C)ᵀ` built after.
    pub(super) fn eliminate<R>(
        self,
        config: &Config,
        streams: Streams,
        factor: impl FnOnce(Arm<T, I>) -> R,
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
    ) -> (Arm<T, I>, Rows<T, I>) {
        let rows = match formation(config, self.cross.n_cols()) {
            Formation::Dense => return (Arm::Exact(self.signed(inv_pivot)), self.cross),
            Formation::Exact => self.exact(inv_pivot, Layout::sddm(self.balance)),
            Formation::Sampled(copies) => self.sample(copies, streams).merge(),
        };
        let s = Sddm::of_rows(rows, &self.grounding, config.frustrated);
        (Arm::Approximate(s), self.cross)
    }

    /// `S` alone, a complement's summand: only exact computes pivots, and `C` is freed before the sampled merge.
    pub(super) fn complement(self, config: &Config, streams: Streams) -> Arm<T, I> {
        let rows = match formation(config, self.cross.n_cols()) {
            Formation::Dense => return Arm::Exact(self.signed(&self.inverse_pivots())),
            Formation::Exact => self.exact(&self.inverse_pivots(), Layout::sddm(self.balance)),
            Formation::Sampled(copies) => {
                let fill = self.sample(copies, streams);
                // The merge never reads `C`.
                drop(self.cross);
                fill.merge()
            }
        };
        let route = config.frustrated;
        Arm::Approximate(Sddm::of_rows(rows, &self.grounding, route))
    }

    /// `S` itself: a balanced block's exact rows, a frustrated one's signed pairs folded.
    fn signed(&self, inv_pivot: &[T]) -> Signed<T, I> {
        let rows = self.exact(inv_pivot, Layout::as_is(self.balance));
        let (adjacency, grounding) = rows.close(&self.grounding);
        Signed::new(adjacency, self.balance, grounding)
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

    /// `S − diag(s_k)` exact by kept row in `layout`, `S = D_k − Cᵀ D_e⁻¹ C`, through a `Cᵀ` freed before closing adds `s_k`.
    fn exact(&self, inv_pivot: &[T], layout: Layout) -> SddmRows<T, I> {
        match layout {
            Layout::Sheet => self.exact_rows(inv_pivot, layout, |j, _, _| j),
            Layout::Cover | Layout::SignedPairs => {
                self.exact_rows(inv_pivot, layout, |j, c_ek, c_ej| {
                    cover_vertex(j, usize::from(crosses(c_ek) != crosses(c_ej)))
                })
            }
        }
    }

    /// `|c_ke||c_je|/d_e` to `j`'s `vertex`, each row summing only `j > k` (a cover's `k⁻` mirrors `k⁺`), the ground's share last.
    fn exact_rows(
        &self,
        inv_pivot: &[T],
        layout: Layout,
        vertex: impl Fn(usize, T, T) -> usize + Sync,
    ) -> SddmRows<T, I> {
        let (c, grounding) = (&self.cross, &self.grounding);
        let ct = c.transpose();
        let ground = layout.columns(c.n_cols());
        // Each task carries one grain of the complement's `nnz·(nnz/n_e)/2` multiply-adds.
        let work = c.nnz() as u64 * (c.nnz() / c.n_rows().max(1)) as u64 / 2;
        SddmRows::split(
            layout,
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
    fn sample(&self, copies: Option<Multi>, streams: Streams) -> Fill<T, I> {
        let stars = Stars {
            block: self,
            streams,
            ground: I::from_index(Layout::sddm(self.balance).columns(self.cross.n_cols())),
        };
        match copies {
            None => stars.sample_with(Single),
            Some(copies) => stars.sample_with(copies),
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
        match self.block.balance {
            // One sheet maps no column, as the balanced loop pays for any map.
            Balance::Balanced => {
                let neighbors = c.row(s).map(|(k, &c)| (I::from_index(k), c.abs()));
                sampler.eliminate_unique(draws, neighbors.chain(ground(s)), fill, Unsigned);
            }
            Balance::Frustrated => {
                let (sheet, e) = (s / c.n_rows(), s % c.n_rows());
                let neighbors = c.row(e).map(move |(k, &c)| {
                    let on = sheet ^ usize::from(crosses(c));
                    (I::from_index(cover_vertex(k, on)), c.abs())
                });
                sampler.eliminate_unique(draws, neighbors.chain(ground(e)), fill, Unsigned);
            }
        }
    }

    /// Each star's clique tree, chunked in star order so merged duplicates sum alike at any thread count.
    fn sample_with<C: Copies>(&self, copies: C) -> Fill<T, I> {
        let c = &self.block.cross;
        let layout = Layout::sddm(self.block.balance);
        let n = layout.columns(c.n_cols());
        let shift = bucket_shift(n);
        let count = n.div_ceil(1 << shift);
        let per_task = per_task(c.n_rows(), c.nnz() as u64, 14);
        let empty = || (CliqueTreeSampler::new(copies), vec![Vec::new(); count]);
        let stars = layout.columns(c.n_rows());
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
        Fill {
            layout,
            kept: c.n_cols(),
            pieces,
        }
    }
}

/// Rows per fill bucket, as a shift: at most 16 buckets, so one bucket's sorted copy is all the fill ever copied at once.
fn bucket_shift(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).saturating_sub(4)
}

/// One thread's fill edges `(lo, hi, weight)`, split by `lo` into the ranges of [`bucket_shift`].
type Buckets<T, I> = Vec<Vec<(I, I, T)>>;

/// Every task's sampled fill over the rows of `kept` vertices in `layout`, bucketed by row; an edge to the last column is `lo`'s ground weight.
struct Fill<T, I> {
    layout: Layout,
    kept: usize,
    pieces: Vec<Buckets<T, I>>,
}

impl<T: Real, I: Index> Fill<T, I> {
    /// Bucket by bucket, counting-sorts the chunks' rows unflattened, merges each row's duplicates, stores `−w`.
    fn merge(self) -> SddmRows<T, I> {
        let Self {
            layout,
            kept,
            mut pieces,
        } = self;
        let n = layout.columns(kept);
        let stride = 1 << bucket_shift(n);
        let count = n.div_ceil(stride);
        let bucket = |b: usize| pieces.iter().map(|p| p[b].len()).sum();
        let most = (0..count).map(bucket).max().unwrap_or(0);
        let (mut columns, mut weights) = (vec![I::zero(); most], vec![T::zero(); most]);
        let (mut starts, mut next) = (Vec::new(), Vec::new());
        let mut rows = SddmRows::new(layout, kept);
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
