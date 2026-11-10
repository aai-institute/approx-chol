//! The Schur complement `S` a bipartite block leaves on its kept side, formed exact or sampled.

use rayon::prelude::*;

use super::block_elimination::BlockElimination;
use super::cholesky::Arm;
use super::clique_tree::{CliqueTreeSampler, Copies, Multi, Single};
use super::config::{Config, CoverSurplus, FrustratedRoute, SchurMode};
use super::sddm::{cover_vertex, crosses, Factored, Layout, SddmRows};
use super::sign::Unsigned;
use super::{Streams, MULTIPLY_ADD_GRAIN_BITS};
use crate::sgdd::{Balance, Grounding};
use crate::storage::Rows;
use crate::threads::task;
use crate::{narrow, per_task, widen, Index, Real, Threads};

/// How one block's Schur complement is formed, resolved from [`Config`] and what it is factored as.
enum Formation {
    /// Each star's whole clique.
    Exact,
    /// Clique trees sampled with these copies.
    Sampled(Option<Multi>),
}

/// Exact when factored as it is, which spends nothing on sparsity, or when asked, else sampled with its own sampling or under `Same` the factor's.
fn formation(config: &Config, factored: Factored) -> Formation {
    match config.complement {
        _ if matches!(factored, Factored::AsIs(_)) => Formation::Exact,
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
        threads: Threads,
        factor: impl FnOnce(Arm<T, I>) -> R,
    ) -> (BlockElimination<T, I>, R) {
        let inv_pivot = self.inverse_pivots();
        let (s, cross) = self.complement_and_cross(config, streams, threads, &inv_pivot);
        let reduced = factor(s);
        (BlockElimination::new(cross, inv_pivot), reduced)
    }

    /// `S`, and `C` handed back for `L`: σ dies here, before `S` is factored.
    fn complement_and_cross(
        self,
        config: &Config,
        streams: Streams,
        threads: Threads,
        inv_pivot: &[T],
    ) -> (Arm<T, I>, Rows<T, I>) {
        let factored = self.factored(config);
        let rows = match formation(config, factored) {
            Formation::Exact => self.exact(inv_pivot, factored, threads),
            Formation::Sampled(copies) => {
                let kept = Self::kept_surplus(&self.grounding, self.cross.n_rows());
                self.sample(copies, factored, streams, threads).merge(kept)
            }
        };
        (rows.arm(), self.cross)
    }

    /// `S` alone, a complement's summand: only exact computes pivots, and `C` is freed before the sampled merge.
    pub(super) fn complement(
        self,
        config: &Config,
        streams: Streams,
        threads: Threads,
    ) -> Arm<T, I> {
        let factored = self.factored(config);
        let rows = match formation(config, factored) {
            Formation::Exact => self.exact(&self.inverse_pivots(), factored, threads),
            Formation::Sampled(copies) => {
                let fill = self.sample(copies, factored, streams, threads);
                let kept = Self::kept_surplus(&self.grounding, self.cross.n_rows());
                // The merge never reads `C`.
                drop(self.cross);
                fill.merge(kept)
            }
        };
        rows.arm()
    }

    /// What `S` is factored as, by its kept side; under `Signed` it is still covered, as `Cover(Auto)` carries it.
    fn factored(&self, config: &Config) -> Factored {
        let frustrated = match config.frustrated {
            FrustratedRoute::Signed => FrustratedRoute::Cover(CoverSurplus::Auto),
            route => route,
        };
        let config = Config {
            frustrated,
            ..*config
        };
        Factored::of(&config, self.balance, self.cross.n_cols())
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

    /// The kept rows' surplus `s_k`, after the `n_e` eliminated rows'; borrowed from σ alone, so `C` can be freed while it is held.
    fn kept_surplus(grounding: &Grounding<T>, n_e: usize) -> Option<&[T]> {
        grounding.surplus().map(|s| &s[n_e..])
    }

    /// `S` exact by kept row in its layout, `S = D_k − Cᵀ D_e⁻¹ C`, through a `Cᵀ` freed before the rows close.
    fn exact(&self, inv_pivot: &[T], factored: Factored, threads: Threads) -> SddmRows<'_, T, I> {
        match factored.layout() {
            Layout::Sheet => self.exact_rows(inv_pivot, factored, threads, |j, _, _| j),
            Layout::Cover | Layout::SignedPairs => {
                self.exact_rows(inv_pivot, factored, threads, |j, c_ek, c_ej| {
                    cover_vertex(j, usize::from(crosses(c_ek) != crosses(c_ej)))
                })
            }
        }
    }

    /// `|c_ke||c_je|/d_e` to `j`'s `vertex`, each row summing only `j > k` (a cover's `k⁻` mirrors `k⁺`), the ground's share last.
    fn exact_rows(
        &self,
        inv_pivot: &[T],
        factored: Factored,
        threads: Threads,
        vertex: impl Fn(usize, T, T) -> usize + Sync,
    ) -> SddmRows<'_, T, I> {
        let (c, grounding) = (&self.cross, &self.grounding);
        let ct = c.transpose();
        let ground = factored.layout().columns(c.n_cols());
        // Each task carries one grain of the complement's `nnz·(nnz/n_e)/2` multiply-adds.
        let work = c.nnz() as u64 * (c.nnz() / c.n_rows().max(1)) as u64 / 2;
        SddmRows::split(
            factored,
            c.n_cols(),
            per_task(c.n_cols(), work, MULTIPLY_ADD_GRAIN_BITS),
            threads,
            Self::kept_surplus(grounding, c.n_rows()),
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
    fn sample(
        &self,
        copies: Option<Multi>,
        factored: Factored,
        streams: Streams,
        threads: Threads,
    ) -> Fill<T, I> {
        let layout = factored.layout();
        let stars = Stars {
            block: self,
            streams,
            layout,
            ground: I::from_index(layout.rows(self.cross.n_cols())),
        };
        let pieces = match copies {
            None => stars.sample_with(Single, threads),
            Some(copies) => stars.sample_with(copies, threads),
        };
        Fill {
            factored,
            kept: self.cross.n_cols(),
            pieces,
        }
    }
}

/// The eliminated side's stars: row `e` of `C` on each of `layout`'s sheets, its ground share last at the vertex past the rows, each under its own stream.
struct Stars<'a, T, I> {
    block: &'a BipartiteBlock<T, I>,
    streams: Streams,
    layout: Layout,
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
    fn sample_with<C: Copies>(&self, copies: C, threads: Threads) -> Vec<Buckets<T, I>> {
        let c = &self.block.cross;
        let n = self.layout.rows(c.n_cols());
        let shift = bucket_shift(n);
        let count = n.div_ceil(1 << shift);
        let per_task = per_task(c.n_rows(), c.nnz() as u64, 14);
        let stars = self.layout.rows(c.n_rows());
        let piece = |task: usize| {
            let (mut sampler, mut buckets) =
                (CliqueTreeSampler::new(copies), vec![Vec::new(); count]);
            for s in task * per_task..stars.min((task + 1) * per_task) {
                let push =
                    |(lo, hi, _, w): (I, I, C, T)| buckets[lo.index() >> shift].push((lo, hi, w));
                self.eliminate(s, &mut sampler, push);
            }
            buckets.iter_mut().for_each(Vec::shrink_to_fit);
            buckets
        };
        let tasks = stars.div_ceil(per_task);
        match threads.per_task(tasks, 1) {
            Some(per) => (0..tasks)
                .into_par_iter()
                .with_min_len(per)
                .map(|k| task(|| piece(k)))
                .collect(),
            None => (0..tasks).map(piece).collect(),
        }
    }
}

/// Rows per fill bucket, as a shift: at most 16 buckets, so one bucket's sorted copy is all the fill ever copied at once.
fn bucket_shift(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).saturating_sub(4)
}

/// One thread's fill edges `(lo, hi, weight)`, split by `lo` into the ranges of [`bucket_shift`].
type Buckets<T, I> = Vec<Vec<(I, I, T)>>;

/// Every task's sampled fill over the rows of `kept` vertices for `factored`, bucketed by row; an edge to the last column is `lo`'s ground weight.
struct Fill<T, I> {
    factored: Factored,
    kept: usize,
    pieces: Vec<Buckets<T, I>>,
}

impl<T: Real, I: Index> Fill<T, I> {
    /// Bucket by bucket, counting-sorts the chunks' rows unflattened, merges each row's duplicates, stores `−w`; each row's ground gains its vertex's `surplus`.
    fn merge(self, surplus: Option<&[T]>) -> SddmRows<'_, T, I> {
        let Self {
            factored,
            kept,
            mut pieces,
        } = self;
        let n = factored.layout().rows(kept);
        let stride = 1 << bucket_shift(n);
        let count = n.div_ceil(stride);
        let bucket = |b: usize| pieces.iter().map(|p| p[b].len()).sum();
        let most = (0..count).map(bucket).max().unwrap_or(0);
        let (mut columns, mut weights) = (vec![I::zero(); most], vec![T::zero(); most]);
        let (mut starts, mut next) = (Vec::new(), Vec::new());
        let mut rows = SddmRows::new(factored, kept, surplus);
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
