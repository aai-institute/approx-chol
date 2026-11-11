//! The Schur complement `S` a bipartite block leaves on its kept side, formed exact or sampled.

use rayon::prelude::*;

use super::block_elimination::BlockElimination;
use super::cholesky::Arm;
use super::clique_tree::{CliqueTreeSampler, Copies, Multi, Single};
use super::config::{Config, SchurMode};
use super::sddm::{cover_vertex, crosses, Factored, Layout, SddmRows};
use super::sign::{Signed, Unsigned};
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
        let kept = self.kept_ground(inv_pivot);
        let factored = self.factored(config, kept.as_deref(), inv_pivot);
        let rows = match formation(config, factored) {
            Formation::Exact => self.exact(inv_pivot, factored, threads, kept),
            Formation::Sampled(copies) => {
                (self.sample(copies, factored, streams, threads)).merge(kept)
            }
        };
        (rows.arm(), self.cross)
    }

    /// `S` alone, a complement's summand: `C` is freed before the sampled merge.
    pub(super) fn complement(
        self,
        config: &Config,
        streams: Streams,
        threads: Threads,
    ) -> Arm<T, I> {
        let inv_pivot = self.inverse_pivots();
        let kept = self.kept_ground(&inv_pivot);
        let factored = self.factored(config, kept.as_deref(), &inv_pivot);
        let rows = match formation(config, factored) {
            Formation::Exact => self.exact(&inv_pivot, factored, threads, kept),
            Formation::Sampled(copies) => {
                drop(inv_pivot);
                let fill = self.sample(copies, factored, streams, threads);
                // The merge never reads `C`.
                drop(self.cross);
                fill.merge(kept)
            }
        };
        rows.arm()
    }

    /// What `S` is factored as, by its kept side, its `ground` and its off-diagonal as formed exactly: a cover twists or grounds before any clique tree is drawn.
    fn factored(&self, config: &Config, ground: Option<&[T]>, inv_pivot: &[T]) -> Factored {
        let off_diagonal = |scale| self.scaled_off_diagonal(inv_pivot, scale);
        Factored::of(
            config,
            self.balance,
            self.cross.n_cols(),
            ground,
            off_diagonal,
        )
    }

    /// `scale · Σ_{j≠k} Σ_e |c_ej||c_ek| / d_e`, `S`'s off-diagonal before opposite shares cancel, as a cover holds it on each sheet.
    fn scaled_off_diagonal(&self, inv_pivot: &[T], scale: f64) -> f64 {
        let c = &self.cross;
        let mut total = 0.0;
        for (e, &inv_pivot) in inv_pivot.iter().enumerate() {
            let mut before = 0.0;
            for (_, &c) in c.row(e) {
                let w = widen(c.abs());
                total += scale * w * (before * widen(inv_pivot));
                before += w;
            }
        }
        2.0 * total
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

    /// `S`'s ground on each kept vertex, its surplus `s_k` plus every star's share `|c_ek| s_e / d_e` in ascending `e`, exact under either formation, what a clique tree draws onto the ground dropped: a twisted cover's draws none, so its kept side stays connected (#437).
    fn kept_ground(&self, inv_pivot: &[T]) -> Option<Vec<T>> {
        let (c, grounding) = (&self.cross, &self.grounding);
        let s = &grounding.surplus()?[c.n_rows()..];
        let mut ground = vec![T::zero(); s.len()];
        for (e, &inv_pivot) in inv_pivot.iter().enumerate() {
            if let Some(s_e) = grounding.edge(e) {
                for (k, &c) in c.row(e) {
                    ground[k] = ground[k] + share(c.abs(), s_e, inv_pivot);
                }
            }
        }
        (ground.iter_mut().zip(s)).for_each(|(g, &s)| *g = s + *g);
        Some(ground)
    }

    /// `S` exact by kept row in its layout, `S = D_k − Cᵀ D_e⁻¹ C`, through a `Cᵀ` freed before the rows close, each vertex's rows grounded by `ground`.
    fn exact(
        &self,
        inv_pivot: &[T],
        factored: Factored,
        threads: Threads,
        ground: Option<Vec<T>>,
    ) -> SddmRows<T, I> {
        match factored.layout() {
            Layout::Sheet => self.exact_rows(inv_pivot, factored, threads, ground, |j, _, _| j),
            Layout::Cover | Layout::SignedPairs => {
                self.exact_rows(inv_pivot, factored, threads, ground, |j, c_ek, c_ej| {
                    cover_vertex(j, usize::from(crosses(c_ek) != crosses(c_ej)))
                })
            }
        }
    }

    /// `|c_ke||c_je|/d_e` to `j`'s `vertex`, each row summing only `j > k` (a cover's `k⁻` mirrors `k⁺`).
    fn exact_rows(
        &self,
        inv_pivot: &[T],
        factored: Factored,
        threads: Threads,
        ground: Option<Vec<T>>,
        vertex: impl Fn(usize, T, T) -> usize + Sync,
    ) -> SddmRows<T, I> {
        let c = &self.cross;
        let ct = c.transpose();
        // Each task carries one grain of the complement's `nnz·(nnz/n_e)/2` multiply-adds.
        let work = c.nnz() as u64 * (c.nnz() / c.n_rows().max(1)) as u64 / 2;
        SddmRows::split(
            factored,
            c.n_cols(),
            per_task(c.n_cols(), work, MULTIPLY_ADD_GRAIN_BITS),
            threads,
            ground,
            |mut sums, k| {
                for (e, &c_ek) in ct.row(k) {
                    for (j, &c_ej) in c.row(e).skip_while(|&(j, _)| j <= k) {
                        let w = share(c_ek.abs(), c_ej.abs(), inv_pivot[e]);
                        sums.add(vertex(j, c_ek, c_ej), w);
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
        let ground = match factored {
            Factored::TwistedCover => None,
            _ => Some(I::from_index(layout.rows(self.cross.n_cols()))),
        };
        let stars = Stars {
            block: self,
            streams,
            layout,
            ground,
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

/// The eliminated side's stars: row `e` of `C` on each of `layout`'s sheets, each under its own stream, its ground an entry at `ground` whose draws drop, or, where the cover twists, in the degree alone.
struct Stars<'a, T, I> {
    block: &'a BipartiteBlock<T, I>,
    streams: Streams,
    layout: Layout,
    ground: Option<I>,
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
        let ground = |e: usize| match (self.ground, self.block.grounding.edge(e)) {
            (Some(at), s_e) => (s_e.map(|s_e| (at, s_e)), 0.0),
            (None, s_e) => (None, s_e.map_or(0.0, widen)),
        };
        let draws = self.streams.star(s);
        match self.layout {
            // One sheet maps no column, as the balanced loop pays for any map.
            Layout::Sheet => {
                let (drawn, degree) = ground(s);
                let neighbors = c.row(s).map(|(k, &c)| (I::from_index(k), c.abs()));
                let fill = self.off_ground(fill);
                sampler.eliminate_unique(draws, neighbors.chain(drawn), degree, fill, Unsigned);
            }
            Layout::Cover => {
                let (sheet, e) = (s / c.n_rows(), s % c.n_rows());
                let (drawn, degree) = ground(e);
                let neighbors = c.row(e).map(move |(k, &c)| {
                    let on = sheet ^ usize::from(crosses(c));
                    (I::from_index(cover_vertex(k, on)), c.abs())
                });
                let fill = self.off_ground(fill);
                sampler.eliminate_unique(draws, neighbors.chain(drawn), degree, fill, Unsigned);
            }
            Layout::SignedPairs => {
                let (drawn, degree) = ground(s);
                let neighbors = c.row(s).map(|(k, &c)| (I::from_index(k), -c));
                let sign = Signed {
                    ground: I::from_index(c.n_cols()),
                };
                let mut fill = fill;
                let paired = |(lo, hi, copies, w): (I, I, C, T)| {
                    let at = cover_vertex(hi.index(), usize::from(w < T::zero()));
                    fill((lo, I::from_index(at), copies, w.abs()));
                };
                let fill = self.off_ground(paired);
                sampler.eliminate_unique(draws, neighbors.chain(drawn), degree, fill, sign);
            }
        }
    }

    /// `fill` without a clique tree's draws onto the ground: `S`'s ground is exact, so they add nothing.
    fn off_ground<C: Copies>(
        &self,
        mut fill: impl FnMut((I, I, C, T)),
    ) -> impl FnMut((I, I, C, T)) {
        let ground = self.ground;
        move |edge| {
            if Some(edge.1) != ground {
                fill(edge);
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

/// Every task's sampled fill over the rows of `kept` vertices for `factored`, bucketed by row.
struct Fill<T, I> {
    factored: Factored,
    kept: usize,
    pieces: Vec<Buckets<T, I>>,
}

impl<T: Real, I: Index> Fill<T, I> {
    /// Bucket by bucket, counting-sorts the chunks' rows unflattened, merges each row's duplicates, stores `−w`; each row's ground is its vertex's `ground`.
    fn merge(self, ground: Option<Vec<T>>) -> SddmRows<T, I> {
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
        let mut rows = SddmRows::new(factored, kept, ground);
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
#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::factor::cholesky::Eliminated;
    use crate::factor::config::{CoverSurplus, FrustratedRoute, Sampling};
    use crate::threads::ONE;

    /// Kept `0` and `1` joined through eliminated `e₀` by `c = (−a₀, −b₀)` and `e₁` by `(a₁, −b₁)`, kept surplus `s`: two degree-two stars of opposite sign onto one pair.
    fn opposed([a0, b0, a1, b1]: [f64; 4], s: [f64; 2]) -> (Rows<f64, u32>, Grounding<f64>) {
        let cross = Rows::new(vec![0, 2, 4], vec![0, 1, 0, 1], vec![-a0, -b0, a1, -b1], 2);
        (cross, Grounding::of(vec![0.0, 0.0, s[0], s[1]]))
    }

    /// `S = D_k − Cᵀ D_e⁻¹ C` as `(−S₀₁, S₀₀ − |S₀₁|, S₁₁ − |S₀₁|)`: the signed elimination's edge and the surplus left on each end.
    fn schur([a0, b0, a1, b1]: [f64; 4], s: [f64; 2]) -> [f64; 3] {
        let (d0, d1) = (a0 + b0, a1 + b1);
        let s01 = -(a0 * b0 / d0 - a1 * b1 / d1);
        let s00 = a0 + a1 + s[0] - a0 * a0 / d0 - a1 * a1 / d1;
        let s11 = b0 + b1 + s[1] - b0 * b0 / d0 - b1 * b1 / d1;
        [-s01, s00 - s01.abs(), s11 - s01.abs()]
    }

    /// Unequal opposite shares onto one pair keep their signed remainder and ground what cancels on both ends, under exact and sampled formation, either sign dominant.
    #[test]
    fn unequal_opposite_shares_keep_their_remainder_and_ground_the_cancelled() {
        let sampled = |copies| SchurMode::Approximate {
            sampling: Sampling::CliqueTree {
                copies: NonZeroU32::new(copies).expect("a copy"),
            },
        };
        let s = [0.25, 0.0];
        let mut defects = Vec::new();
        for (alike, weights) in [(true, [2.0, 2.0, 1.0, 1.0]), (false, [1.0, 1.0, 3.0, 1.0])] {
            let want = schur(weights, s);
            assert_eq!(want[0] > 0.0, alike, "the alike share dominates");
            for (formed, complement) in [
                ("exact", SchurMode::Exact),
                ("AC", sampled(1)),
                ("AC2", sampled(2)),
            ] {
                let config = Config {
                    exact_max_dim: 0,
                    complement,
                    frustrated: FrustratedRoute::Signed,
                    ..Config::default()
                };
                let (cross, grounding) = opposed(weights, s);
                let block = BipartiteBlock::of(cross, Balance::Frustrated, grounding);
                let Arm::Signed(m) = block.complement(&config, Streams::of(0, &config), ONE) else {
                    panic!("a frustrated complement is signed under Signed")
                };
                let mut got = [0.0; 3];
                m.grounded_edges(|i, j, w| match (i, j) {
                    (0, 1) => got[0] += w,
                    (k, 2) => got[1 + k as usize] += w,
                    other => panic!("no edge {other:?}"),
                });
                let off = (got.iter().zip(&want)).fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
                if off > 1e-15 {
                    defects.push(format!("alike {alike}, {formed}: {got:?}, want {want:?}"));
                }
            }
        }
        assert!(defects.is_empty(), "{defects:#?}");
    }

    /// Kept `0, 1, 2` through eliminated `e₀, e₁, e₂`, frustrated unless `balanced`; `e₀` grounded by a subnormal, `e₁` and `e₂` by normal surpluses.
    fn grounded_stars<T: Real>(balanced: bool) -> BipartiteBlock<T, u32> {
        let t = |x: f64| narrow::<T>(x);
        let flip = if balanced { -1.0 } else { 1.0 };
        let values = [-1.0, -0.5, 0.75 * flip, -1.0, -2.0, -0.25, -0.125].map(t);
        let cross = Rows::new(
            vec![0, 2, 4, 7],
            vec![0, 1, 0, 1, 0, 1, 2],
            values.to_vec(),
            3,
        );
        let subnormal = T::min_positive_value() / t(4.0);
        assert!(subnormal > T::zero() && !subnormal.is_normal());
        let surplus = vec![subnormal, t(0.25), t(1.5), t(0.1), T::zero(), t(0.5)];
        let balance = if balanced {
            Balance::Balanced
        } else {
            Balance::Frustrated
        };
        BipartiteBlock::of(cross, balance, Grounding::of(surplus))
    }

    /// Each row's ground edge's bits, by row.
    fn ground_bits<T: Real>(arm: Arm<T, u32>) -> Vec<(u32, u64)> {
        let Arm::Sddm(m) = arm else {
            panic!("an approximately factored complement is an SDDM")
        };
        let n = m.n() as u32;
        let mut ground = Vec::new();
        m.grounded_edges(|i, j, w| {
            if j == n {
                ground.push((i, w.to_f64().expect("a float").to_bits()));
            }
        });
        ground
    }

    fn assert_grounds_are_exact<T: Real>() {
        let sampled = |copies| SchurMode::Approximate {
            sampling: Sampling::CliqueTree {
                copies: NonZeroU32::new(copies).expect("a copy"),
            },
        };
        for balanced in [true, false] {
            let config = |complement| Config {
                exact_max_dim: 0,
                complement,
                frustrated: FrustratedRoute::Cover(CoverSurplus::GroundVertex),
                ..Config::default()
            };
            let grounds = |complement| {
                let config = config(complement);
                let block = grounded_stars::<T>(balanced);
                ground_bits(block.complement(&config, Streams::of(0, &config), ONE))
            };
            let exact = grounds(SchurMode::Exact);
            assert_eq!(exact.len(), if balanced { 3 } else { 6 }, "{exact:?}");
            for copies in [1, 2, 3] {
                assert_eq!(
                    grounds(sampled(copies)),
                    exact,
                    "balanced {balanced}, copies {copies}"
                );
            }
            if !balanced {
                for pair in exact.chunks_exact(2) {
                    assert_eq!(
                        (pair[0].0 / 2, pair[0].1),
                        (pair[1].0 / 2, pair[1].1),
                        "{exact:?}"
                    );
                }
            }
        }
    }

    /// Under `Auto`, a sampled complement is bit for bit the one under the surplus it resolves to, on either side of the twist band: the cover is decided before any clique tree is drawn.
    #[test]
    fn auto_samples_the_complement_of_the_surplus_it_resolves_to() {
        let edges = |surplus, scale: f64| {
            let config = Config {
                exact_max_dim: 0,
                complement: SchurMode::Approximate {
                    sampling: Sampling::CliqueTree {
                        copies: NonZeroU32::MIN,
                    },
                },
                frustrated: FrustratedRoute::Cover(surplus),
                ..Config::default()
            };
            let stars = grounded_stars::<f64>(false);
            let ground = stars.grounding.surplus().expect("a ground");
            let grounding = Grounding::of(ground.iter().map(|s| s * scale).collect());
            let block = BipartiteBlock::of(stars.cross, stars.balance, grounding);
            let Arm::Sddm(m) = block.complement(&config, Streams::of(0, &config), ONE) else {
                panic!("a cover is an SDDM")
            };
            let mut edges = Vec::new();
            m.grounded_edges(|i, j, w: f64| edges.push((i, j, w.to_bits())));
            edges
        };
        let (twisted, grounded) = (CoverSurplus::SurplusEdges, CoverSurplus::GroundVertex);
        for (scale, resolved, other) in [(1e-20, twisted, grounded), (1.0, grounded, twisted)] {
            let auto = edges(CoverSurplus::Auto, scale);
            assert_eq!(auto, edges(resolved, scale), "scale {scale}");
            assert_ne!(auto, edges(other, scale), "scale {scale}");
        }
    }

    /// A sampled complement grounds each kept vertex as the exact one does, bit for bit, under AC and AC2, a subnormal star's ground included, and a cover grounds both copies alike (#437).
    #[test]
    fn a_sampled_complement_grounds_bit_for_bit_as_the_exact_one() {
        assert_grounds_are_exact::<f64>();
        assert_grounds_are_exact::<f32>();
    }
}
