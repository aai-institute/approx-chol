//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (GKS 2023, Algorithm 8, stars by Algorithm 6).

mod graph;
mod ordering;

use rand::rngs::Xoshiro256PlusPlus;

use crate::factor::clique_tree::{absorb, CliqueTreeSampler, Copies, Fill, Multi, Single};
use crate::factor::sdd::Sdd;
use crate::factor::sign::Sign;
#[cfg(feature = "serde")]
use crate::factor::within_one;
use crate::sgdd::FactorTail;
use crate::{narrow, Batch, Index, Real};
use graph::EliminationGraph;
use ordering::MinDegreeOrder;

/// A task's AC or AC2 elimination; small blocks reuse its buffers (malloc was 17% of 20-vertex covers).
pub(crate) struct CliqueTreeElimination<T, I>(ByCopies<T, I>);

enum ByCopies<T, I> {
    Ac(Elimination<T, I, Single>),
    Ac2(Elimination<T, I, Multi>),
}

impl<T: Real, I: Index> CliqueTreeElimination<T, I> {
    pub(in crate::factor) fn new(copies: Option<Multi>) -> Self {
        Self(match copies {
            None => ByCopies::Ac(Elimination::new(Single)),
            Some(copies) => ByCopies::Ac2(Elimination::new(copies)),
        })
    }

    /// Every vertex of `M` in minimum-degree order, each star replaced by a sampled clique tree, but a floating `M`'s last: returned as the slot the solve zeroes.
    pub(super) fn eliminate<S: Sign<T, I>>(
        &mut self,
        m: Sdd<T, I>,
        sign: S,
        draws: Xoshiro256PlusPlus,
        tail: Option<FactorTail<'_, T>>,
    ) -> (ApproximateCholesky<T, I>, Option<usize>) {
        match &mut self.0 {
            ByCopies::Ac(elimination) => elimination.eliminate(m, sign, draws, tail),
            ByCopies::Ac2(elimination) => elimination.eliminate(m, sign, draws, tail),
        }
    }
}

/// A block this small keeps its buffers for the task's next, its graph's slab never compacted; a larger one frees them.
const REUSED_UP_TO: usize = 1024;

/// One copy count's elimination: `C` reaches every edge, so it is fixed per task.
struct Elimination<T, I, C: Copies> {
    copies: C,
    graph: EliminationGraph<T, I, C>,
    clique: CliqueTreeSampler<T, I, C>,
    fill: Fill<T, I, C>,
    order: MinDegreeOrder<I>,
}

impl<T: Real, I: Index, C: Copies> Elimination<T, I, C> {
    fn new(copies: C) -> Self {
        Self {
            copies,
            graph: EliminationGraph::new(),
            clique: CliqueTreeSampler::new(copies),
            fill: Fill::new(),
            order: MinDegreeOrder::new(),
        }
    }

    fn eliminate<S: Sign<T, I>>(
        &mut self,
        m: Sdd<T, I>,
        sign: S,
        mut draws: Xoshiro256PlusPlus,
        tail: Option<FactorTail<'_, T>>,
    ) -> (ApproximateCholesky<T, I>, Option<usize>) {
        let (slots, ground, gauge) = (m.n(), m.ground(), m.gauge(tail));
        let never = ground.or(gauge);
        let room = EliminationGraph::<T, I, C>::room(&m);
        let n = room.len();
        let first = (n > PHASED_FROM)
            .then(|| LocalMinima::of(&m, &room, never))
            .flatten();
        let (mut steps, queued) = match &first {
            None => {
                self.graph.load(&m, &room, self.copies);
                drop((m, room));
                (ApproximateCholesky::with_capacity(n, 0), n)
            }
            Some(first) => {
                let steps = self.pivot_first((m, room), first, sign, &mut draws, slots);
                (steps, first.left)
            }
        };
        let edge_ends = self.graph.edge_ends();
        let gone = |v| first.as_ref().is_some_and(|first| first.gone(v));
        let queue = |v| Some(v) != never && !gone(v);
        self.order.load(&self.graph, self.copies, queue, edge_ends);
        drop(first);
        steps.neighbors.reserve_exact(edge_ends);
        steps.coefficients.reserve_exact(edge_ends);
        // Every vertex left but one: the never-pivoted one, or the order's last.
        self.pivot_all(&mut draws, &mut steps, slots, sign, queued - 1);
        // Doubling past the estimate otherwise retains up to 2x the entries (+30% on grids).
        steps.neighbors.shrink_to_fit();
        steps.coefficients.shrink_to_fit();
        let pinned = ground
            .is_none()
            .then(|| gauge.unwrap_or_else(|| self.order.next_vertex()));
        if n > REUSED_UP_TO {
            *self = Self::new(self.copies);
        }
        (steps, pinned)
    }

    #[inline(never)] // Inlined, its codegen moves with the buffer handling: grids +1-3%.
    fn pivot_all<S: Sign<T, I>>(
        &mut self,
        draws: &mut Xoshiro256PlusPlus,
        steps: &mut ApproximateCholesky<T, I>,
        slots: usize,
        sign: S,
        pivots: usize,
    ) {
        let Self {
            copies: split,
            graph,
            clique,
            fill,
            order,
        } = self;
        for _ in 0..pivots {
            let v = order.next_vertex();
            pivot(
                clique,
                fill,
                draws,
                (steps, slots),
                v,
                graph.neighbors(v),
                sign,
            );
            graph.eliminate_vertex(v);
            for &edge in fill.iter() {
                graph.push_fill(edge, *split, sign);
            }
            order.update(graph, clique.neighbors());
            // Its own pass: chained onto the neighbors, it cost unsigned 16-vertex paths +5-8%.
            order.update(graph, clique.dropped());
        }
    }

    /// Pivots `first`'s vertices, each star its row of `m`, their fill staged with `m`'s other edges and merged as it grows; loads the graph with the merged pairs.
    fn pivot_first<S: Sign<T, I>>(
        &mut self,
        (m, room): (Sdd<T, I>, Vec<u32>),
        first: &LocalMinima,
        sign: S,
        draws: &mut Xoshiro256PlusPlus,
        slots: usize,
    ) -> ApproximateCholesky<T, I> {
        let (n, split) = (room.len(), self.copies);
        let mut start = vec![0usize; n + 1];
        for v in 0..n {
            let row = if first.gone(v) { room[v] - 1 } else { 0 };
            start[v + 1] = start[v] + row as usize;
        }
        drop(room);
        let mut rows = vec![(I::from_index(0), T::zero()); start[n]];
        let mut at = start.clone();
        let mut staged = Vec::new();
        m.grounded_edges(|k, j, w| {
            let (row, to) = match (first.gone(k.index()), first.gone(j.index())) {
                (true, _) => (k, j),
                (_, true) => (j, k),
                _ => return staged.push((k, j, split, w, split.get())),
            };
            rows[at[row.index()]] = (to, w);
            at[row.index()] += 1;
        });
        drop((at, m));
        let mut steps = ApproximateCholesky::with_capacity(n, start[n]);
        let mut merged = staged.len();
        let Self { clique, fill, .. } = self;
        for v in (0..n).filter(|&v| first.gone(v)) {
            let row = rows[start[v]..start[v + 1]].iter();
            let row = row.map(|&(u, w)| (u, split, w));
            pivot(clique, fill, draws, (&mut steps, slots), v, row, sign);
            staged.extend(
                fill.iter()
                    .map(|&(u, w, c, weight)| (u, w, c, weight, c.get())),
            );
            // Merged at each doubling, so the staging stays within twice its merged size.
            if staged.len() >= (2 * merged).max(1 << 16) {
                merge_pairs(&mut staged, first, split, sign);
                merged = staged.len();
            }
        }
        drop((rows, start));
        merge_pairs(&mut staged, first, split, sign);
        self.graph.load_merged(&staged, |v| first.gone(v), n, split);
        steps
    }
}

/// `v` pivoted on `neighbors`: its star sampled into `fill`, its column appended to `steps`; the one step both phases take.
#[inline(always)] // pivot_all's codegen is measured; this keeps its loop body as it was.
fn pivot<T: Real, I: Index, C: Copies, S: Sign<T, I>>(
    clique: &mut CliqueTreeSampler<T, I, C>,
    fill: &mut Fill<T, I, C>,
    draws: &mut Xoshiro256PlusPlus,
    (steps, slots): (&mut ApproximateCholesky<T, I>, usize),
    v: usize,
    neighbors: impl Iterator<Item = (I, C, T)>,
    sign: S,
) {
    fill.clear();
    let column = clique.eliminate(draws, neighbors, fill, sign);
    steps.push(v, column, slots);
}

/// Above it, the local minima may pivot first; a smaller block has too few to pay the staging.
const PHASED_FROM: usize = 1024;

/// Entries a merged pair stands as at most, so the copies its cap dropped still count in the degree: without, fixest's person × firm band took 49-54 PCG iterations instead of 17-19; 4 measured as the full count.
const STOOD_UP_TO: u32 = 4;

/// A staged pair `(lo, hi, copies, weight, merged)`: `merged` counts its copies before the cap.
type Staged<T, I, C> = (I, I, C, T, u32);

/// The vertices that pivot first, each `(degree, id)` below every neighbor's but the never-pivoted one's: an independent set (Liu's multiple minimum degree), so each star is its input row.
struct LocalMinima {
    /// Each other vertex's id among those minimum degree eliminates, [`u32::MAX`] for one pivoted first.
    ids: Vec<u32>,
    /// The vertices minimum degree eliminates, the never-pivoted one counted.
    left: usize,
}

impl LocalMinima {
    /// The set, if its edges are a quarter of all and the vertices fit `u32`; each edge marks its larger end unless its smaller is `never`, which an edge to the ground always is.
    fn of<T: Real, I: Index>(m: &Sdd<T, I>, room: &[u32], never: Option<usize>) -> Option<Self> {
        u32::try_from(room.len()).ok()?;
        let key = |v: usize| (u64::from(room[v]) << 32) | v as u64;
        let candidate = |v: usize| Some(v) != never && room[v] > 1;
        let edges = |v: usize| u64::from(room[v] - 1);
        let total: u64 = (0..room.len()).map(edges).sum();
        // The unmarked candidates' edges: what the set may still cover.
        let mut open: u64 = (0..room.len()).filter(|&v| candidate(v)).map(edges).sum();
        let mut above = vec![false; room.len()];
        for i in coarse_first(m.n()) {
            for j in m.neighbors_past(i) {
                let (lo, hi) = if key(i) < key(j) { (i, j) } else { (j, i) };
                if Some(lo) != never && !above[hi] {
                    above[hi] = true;
                    if candidate(hi) {
                        open -= edges(hi);
                    }
                }
            }
            // Below a quarter, grids would pay the staging for few pivots; a sparse sweep rejects one after a fifth to a third of its rows.
            if 4 * open < total {
                return None;
            }
        }
        let first = |v: usize| candidate(v) && !above[v];
        let mut left = 0;
        let ids = (0..room.len())
            .map(|v| match first(v) {
                true => u32::MAX,
                false => {
                    left += 1;
                    left as u32 - 1
                }
            })
            .collect();
        Some(Self { ids, left })
    }

    fn gone(&self, v: usize) -> bool {
        self.ids[v] == u32::MAX
    }

    fn id<I: Index>(&self, v: I) -> usize {
        self.ids[v.index()] as usize
    }
}

/// `0..n` by strides 64, 32, …, 1, each once: marks don't depend on the order, and a sparse sweep first marks most of a near-regular graph early.
fn coarse_first(n: usize) -> impl Iterator<Item = usize> {
    let rest = (0..6)
        .rev()
        .flat_map(move |s| ((1 << s)..n).step_by(2 << s));
    (0..n).step_by(64).chain(rest)
}

/// Each pair once, its duplicates absorbed into it as the star absorbs them; what a merge cancels grounds both its ends, merged in turn, and a pair cancelled to zero leaves.
fn merge_pairs<T: Real, I: Index, C: Copies, S: Sign<T, I>>(
    pairs: &mut Vec<Staged<T, I, C>>,
    first: &LocalMinima,
    split: C,
    sign: S,
) {
    loop {
        let grounded = merge_once(pairs, first, split, sign);
        if grounded.is_empty() {
            return;
        }
        // An edge to the ground is a magnitude, so merging these cancels nothing more.
        pairs.extend(grounded);
    }
}

/// One merge, bucketed by `lo`'s id, then scattered on `hi`'s; returns the ground edges its cancellations owe.
fn merge_once<T: Real, I: Index, C: Copies, S: Sign<T, I>>(
    pairs: &mut Vec<Staged<T, I, C>>,
    first: &LocalMinima,
    split: C,
    sign: S,
) -> Vec<Staged<T, I, C>> {
    let left = first.left;
    let mut start = vec![0usize; left + 1];
    for pair in pairs.iter() {
        start[first.id(pair.0) + 1] += 1;
    }
    for l in 0..left {
        start[l + 1] += start[l];
    }
    let mut bucketed = pairs.clone();
    let mut at = start.clone();
    for &pair in pairs.iter() {
        let l = first.id(pair.0);
        bucketed[at[l]] = pair;
        at[l] += 1;
    }
    drop(at);
    let (mut seen, mut out, mut grounded) = (vec![usize::MAX; left], 0, Vec::new());
    for l in 0..left {
        let from = out;
        for &pair in &bucketed[start[l]..start[l + 1]] {
            match seen[first.id(pair.1)] {
                usize::MAX => {
                    seen[first.id(pair.1)] = out;
                    pairs[out] = pair;
                    out += 1;
                }
                held => {
                    let held = &mut pairs[held];
                    let duplicate = (pair.2, pair.3);
                    let cancelled = absorb(sign, split, (&mut held.2, &mut held.3), duplicate);
                    held.4 = held.4.saturating_add(pair.4);
                    if let Some(ground) = sign.ground().filter(|_| cancelled != T::zero()) {
                        let owed = |end| (end, ground, split, cancelled, split.get());
                        grounded.extend([owed(pair.0), owed(pair.1)]);
                    }
                }
            }
        }
        for pair in &pairs[from..out] {
            seen[first.id(pair.1)] = usize::MAX;
        }
    }
    pairs.truncate(out);
    pairs.retain(|pair| pair.3 != T::zero());
    grounded
}

/// A step's neighbors follow the previous step's, so the solve's running `usize` offset finds them at any size.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
struct StepHeader<T, I> {
    vertex: I,
    /// Below `n`, so `I` always holds it and a `u32` header stays 16 bytes.
    len: I,
    inverse_degree: T,
}

/// `LDLᵀ` with sampled columns, appended in pivot order: a serial recurrence, so shares are composed against the original pivot.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct ApproximateCholesky<T, I> {
    headers: Vec<StepHeader<T, I>>,
    neighbors: Vec<I>,
    coefficients: Vec<T>,
}

impl<T: Real, I: Index> ApproximateCholesky<T, I> {
    /// Growing to `entries` by doubling measured +1.5-2.5% on grids.
    fn with_capacity(n: usize, entries: usize) -> Self {
        Self {
            headers: Vec::with_capacity(n),
            neighbors: Vec::with_capacity(entries),
            coefficients: Vec::with_capacity(entries),
        }
    }

    /// Each share `w/d` and `D⁻¹` rounded once from `f64`: no share is derived from the others, so none cancels (#275, #302); only `M`'s `slots` take one, not the ground past them.
    fn push(
        &mut self,
        vertex: usize,
        (d, shares): (f64, impl Iterator<Item = (I, f64)>),
        slots: usize,
    ) {
        let start = self.neighbors.len();
        for (neighbor, share) in shares.filter(|&(j, _)| j.index() < slots) {
            self.neighbors.push(neighbor);
            self.coefficients.push(narrow(share));
        }
        // Finite, as fill and `S` never weigh below `MIN_POSITIVE`.
        self.headers.push(StepHeader {
            vertex: I::from_index(vertex),
            len: I::from_index(self.neighbors.len() - start),
            inverse_degree: narrow(1.0 / d),
        });
    }

    /// Forward, zero a floating factor's `pinned` slot, backward; the gather dispatches `[]`, `[_]`, lanes by row length.
    pub(super) fn substitute<V: Batch<T>>(&self, y: &mut [V], pinned: Option<usize>) {
        let Self {
            headers,
            neighbors,
            coefficients,
        } = self;
        let mut end = 0;
        for header in headers {
            let span = end..end + header.len.index();
            end = span.end;
            let pivot = y[header.vertex.index()];
            match coefficients[span.clone()] {
                // A lone neighbor takes the whole pivot unless the ground shared it; a multiply on this serial chain cost paths +24%.
                [c] if c == T::one() => {
                    let j = neighbors[span.start].index();
                    y[j] = y[j].add(pivot);
                }
                _ => {
                    for (&j, &c) in neighbors[span.clone()].iter().zip(&coefficients[span]) {
                        y[j.index()] = y[j.index()].add(pivot.mul(c));
                    }
                }
            }
            y[header.vertex.index()] = pivot.mul(header.inverse_degree);
        }
        if let Some(pinned) = pinned {
            y[pinned] = V::zero();
        }
        for header in headers.iter().rev() {
            let span = end - header.len.index()..end;
            end = span.start;
            let v = header.vertex.index();
            match coefficients[span.clone()] {
                [] => {}
                [c] if c == T::one() => y[v] = y[v].add(y[neighbors[span.start].index()]),
                [c] => y[v] = y[v].add(y[neighbors[span.start].index()].mul(c)),
                _ => {
                    let mut lanes = [V::zero(); SPLIT];
                    lanes[0] = y[v];
                    let mut js = neighbors[span.clone()].chunks_exact(SPLIT);
                    let mut cs = coefficients[span].chunks_exact(SPLIT);
                    for (js, cs) in js.by_ref().zip(cs.by_ref()) {
                        for lane in 0..SPLIT {
                            lanes[lane] = lanes[lane].add(y[js[lane].index()].mul(cs[lane]));
                        }
                    }
                    let mut total = lanes.iter().fold(V::zero(), |sum, &lane| sum.add(lane));
                    for (&j, &c) in js.remainder().iter().zip(cs.remainder()) {
                        total = total.add(y[j.index()].mul(c));
                    }
                    y[v] = total;
                }
            }
        }
    }
}

/// The split reassociates the row's sum, so this is part of a factor's solve output.
const SPLIT: usize = 4;

#[cfg(feature = "serde")]
impl<T: Real, I: Index> ApproximateCholesky<T, I> {
    /// Each slot but `pin` pivots once, giving only to slots not yet pivoted, shares `S` admits of at most its whole.
    pub(super) fn fits<S: Sign<T, I>>(&self, slots: usize, pin: Option<usize>) -> bool {
        let Self {
            headers,
            neighbors,
            coefficients,
        } = self;
        if headers.len() + usize::from(pin.is_some()) != slots
            || neighbors.len() != coefficients.len()
        {
            return false;
        }
        let mut pivoted = vec![false; slots];
        let mut end = 0usize;
        for header in headers {
            let (v, start) = (header.vertex.index(), end);
            end = end.saturating_add(header.len.index());
            if Some(v) == pin
                || v >= slots
                || std::mem::replace(&mut pivoted[v], true)
                || end > neighbors.len()
            {
                return false;
            }
            let later = neighbors[start..end]
                .iter()
                .all(|&j| j.index() < slots && !pivoted[j.index()]);
            let shares = &coefficients[start..end];
            let pivot = header.inverse_degree.is_finite() && header.inverse_degree > T::zero();
            let admitted = shares.iter().all(|&c| S::admits(c));
            if !later || !pivot || !admitted || !within_one(shares.iter().copied()) {
                return false;
            }
        }
        end == neighbors.len()
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, RngExt, SeedableRng};

    use super::*;
    use crate::factor::clique_tree::{Multi, Single};
    use crate::factor::config::{Config, FrustratedRoute, Sampling, SchurMode};
    use crate::factor::schur::BipartiteBlock;
    use crate::factor::sdd::Layout;
    use crate::factor::sign::{Signed, Unsigned};
    use crate::factor::{within_one, Streams};
    use crate::sgdd::{Balance, Grounding};
    use crate::storage::{Rows, UpperAdjacency};
    use crate::threads::ONE;
    use crate::widen;

    /// Every vertex left, numbered by itself.
    fn none_first(n: u32) -> LocalMinima {
        LocalMinima {
            ids: (0..n).collect(),
            left: n as usize,
        }
    }

    /// Every row once, whatever `n`.
    #[test]
    fn coarse_first_visits_each_row_once() {
        for n in [0, 1, 2, 63, 64, 65, 127, 1000] {
            let mut rows: Vec<_> = coarse_first(n).collect();
            rows.sort_unstable();
            assert_eq!(rows, (0..n).collect::<Vec<_>>(), "{n}");
        }
    }

    /// Each pair once, its weights and uncapped copies summed, its copies capped.
    #[test]
    fn merging_sums_each_pairs_parallel_edges() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let (n, split) = (12u32, Multi::split(4).expect("AC2"));
        let mut pairs: Vec<Staged<f64, u32, Multi>> = (0..300)
            .map(|_| {
                let (a, b) = (rng.random_range(0..n), rng.random_range(0..n - 1));
                let b = b + u32::from(b >= a);
                let (w, copies) = (f64::from(rng.random_range(1..5u8)), rng.random_range(1..3));
                (a.min(b), a.max(b), split.cap(copies), w, copies)
            })
            .collect();
        let mut want = std::collections::BTreeMap::<(u32, u32), (f64, u32)>::new();
        for &(a, b, _, w, copies) in &pairs {
            let held = want.entry((a, b)).or_default();
            *held = (held.0 + w, held.1 + copies);
        }
        merge_pairs(&mut pairs, &none_first(n), split, Unsigned);
        let have: std::collections::BTreeMap<_, _> = (pairs.iter())
            .map(|&(a, b, copies, w, merged)| {
                assert_eq!(copies.get(), merged.min(4));
                ((a, b), (w, merged))
            })
            .collect();
        assert_eq!(have.len(), pairs.len(), "a pair held twice");
        assert_eq!(have, want);
    }

    /// Opposite duplicates cancel onto the ground at both ends, merged with the ground edges held, and a pair cancelled to zero leaves.
    #[test]
    fn merging_grounds_what_a_signed_pair_cancels() {
        let sign = Signed { ground: 3u32 };
        let mut pairs: Vec<Staged<f64, u32, Single>> = vec![
            (0, 1, Single, 2.0, 1),
            (0, 3, Single, 1.0, 1),
            (0, 1, Single, -0.5, 1),
            (1, 2, Single, 1.0, 1),
            (1, 2, Single, -1.0, 1),
        ];
        merge_pairs(&mut pairs, &none_first(4), Single, sign);
        pairs.sort_by_key(|p| (p.0, p.1));
        let have: Vec<_> = pairs.iter().map(|p| (p.0, p.1, p.3, p.4)).collect();
        assert_eq!(
            have,
            [
                (0, 1, 1.5, 2),
                (0, 3, 2.0, 2),
                (1, 3, 3.0, 2),
                (2, 3, 2.0, 1)
            ]
        );
    }

    fn stars(m: usize) -> [(&'static str, Vec<f32>); 3] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let log_uniform = |_| ((rng.next_u64() >> 40) as f32 / (1 << 21) as f32 - 4.0).exp2();
        [
            ("0.1", vec![0.1; m]),
            ("0.7", vec![0.7; m]),
            ("log-uniform", (0..m).map(log_uniform).collect()),
        ]
    }

    /// Worst relative error of each entry's summed fill against `w_i S_{>i} / d`, and `d`, both in `f64`.
    fn fill_error<C: Copies>(w: &[f32], fill: &Fill<f32, u32, C>) -> (f64, f64) {
        let mut order: Vec<u32> = (1..=w.len() as u32).collect();
        order.sort_by(|&a, &b| {
            w[a as usize - 1]
                .total_cmp(&w[b as usize - 1])
                .then(a.cmp(&b))
        });
        let mut position = vec![0; w.len() + 1];
        order
            .iter()
            .enumerate()
            .for_each(|(p, &j)| position[j as usize] = p);
        let mut sums = vec![0.0; w.len()];
        for &(lo, hi, _, weight) in fill {
            sums[position[lo as usize].min(position[hi as usize])] += f64::from(weight);
        }
        let sorted: Vec<f64> = order
            .iter()
            .map(|&j| f64::from(w[j as usize - 1]))
            .collect();
        let d: f64 = sorted.iter().sum();
        let mut suffix = d;
        let worst = (sorted.iter().zip(&sums).take(w.len() - 1))
            .map(|(&wi, &sum)| {
                suffix -= wi;
                (sum / (wi * suffix / d) - 1.0).abs()
            })
            .fold(0.0, f64::max);
        (worst, d)
    }

    /// Worst relative error of every stored share, the heaviest's included, against `w_j / d`; `w_j` is neighbor `j`'s weight, `j ≥ 1`.
    fn share_error<T: Real>(steps: &ApproximateCholesky<T, u32>, w: &[f64], d: f64) -> f64 {
        assert_eq!(steps.coefficients.len(), w.len(), "every neighbor's share");
        (steps.neighbors.iter().zip(&steps.coefficients))
            .map(|(&j, &c)| (crate::widen(c) / (w[j as usize - 1] / d) - 1.0).abs())
            .fold(0.0, f64::max)
    }

    fn pivot_defects<C: Copies>(copies: C, label: &str, w: &[f32]) -> Option<String> {
        let mut clique = CliqueTreeSampler::<f32, u32, C>::new(copies);
        let mut fill = Fill::new();
        let mut steps = ApproximateCholesky::<f32, u32>::with_capacity(w.len() + 1, w.len());
        let draws = &mut Xoshiro256PlusPlus::seed_from_u64(0);
        let star = (1..).zip(w).map(|(j, &w)| (j, copies, w));
        steps.push(
            0,
            clique.eliminate(draws, star, &mut fill, Unsigned),
            w.len() + 1,
        );
        let (fill_off, d) = fill_error(w, &fill);
        let off = (f64::from(steps.headers[0].inverse_degree) * d - 1.0).abs();
        let w: Vec<f64> = w.iter().copied().map(f64::from).collect();
        let shares_off = share_error(&steps, &w, d);
        let bounded = within_one(steps.coefficients.iter().copied());
        (off >= 1e-6 || shares_off >= 1e-6 || fill_off >= 1e-5 || !bounded).then(|| {
            format!("{label}: D⁻¹ off {off:e}, shares off {shares_off:e}, fill off {fill_off:e}, Σ|c| ≤ 1 {bounded}")
        })
    }

    fn unique_fill_defects(label: &str, w: &[f32]) -> Option<String> {
        let mut clique = CliqueTreeSampler::<f32, u32, Single>::new(Single);
        let mut fill = Fill::new();
        let star = (1..).zip(w.iter().copied());
        clique.eliminate_unique(
            Xoshiro256PlusPlus::seed_from_u64(0),
            star,
            0.0,
            |e| fill.push(e),
            Unsigned,
        );
        let (fill_off, _) = fill_error(w, &fill);
        (fill_off >= 1e-5).then(|| format!("{label}: fill off {fill_off:e}"))
    }

    #[test]
    fn high_degree_f32_star_keeps_its_inverse_degree_and_fill() {
        let ac2 = Multi::split(2).expect("k = 2 is AC2");
        let mut defects = Vec::new();
        for m in [6000, 20000, 50000] {
            for (kind, w) in stars(m) {
                let label = |arm| format!("{arm} m {m} w {kind}");
                defects.extend(pivot_defects(Single, &label("AC"), &w));
                defects.extend(pivot_defects(ac2, &label("AC2"), &w));
                defects.extend(unique_fill_defects(&label("unique"), &w));
            }
        }
        assert!(defects.is_empty(), "{defects:#?}");
    }

    /// At degree 10⁵ the heaviest share as `1 − Σ` erred 1.9e-7 relative in `f64` (#306); as `w/d` it rounds once.
    #[test]
    fn a_high_degree_f64_star_rounds_each_share_once() {
        let w = vec![0.1; 100_000];
        let mut clique = CliqueTreeSampler::<f64, u32, Single>::new(Single);
        let mut steps = ApproximateCholesky::<f64, u32>::with_capacity(w.len() + 1, w.len());
        let star = (1..).zip(&w).map(|(j, &w)| (j, Single, w));
        let draws = &mut Xoshiro256PlusPlus::seed_from_u64(0);
        steps.push(
            0,
            clique.eliminate(draws, star, &mut Fill::new(), Unsigned),
            w.len() + 1,
        );
        let d = w.iter().sum();
        let off = share_error(&steps, &w, d);
        assert!(off <= f64::EPSILON, "a share off by {off:e}");
    }

    fn log_uniform<T: Real>(rng: &mut Xoshiro256PlusPlus) -> T {
        narrow(((rng.next_u64() % 1600) as f64 / 100.0 - 8.0).exp2())
    }

    /// A grounded Z-matrix on `n` vertices, a spanning path and random edges, log-uniform; its `D`-switched copy by random `σ`; and `σ`.
    fn switched<T: Real>(n: usize, seed: u64) -> (Sdd<T, u32>, Sdd<T, u32>, Vec<bool>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut pairs: Vec<(usize, usize)> = (1..n).map(|v| (v - 1, v)).collect();
        for _ in 0..2 * n {
            let (a, b) = (rng.next_u64() as usize % n, rng.next_u64() as usize % n);
            pairs.extend((a != b).then_some((a.min(b), a.max(b))));
        }
        pairs.sort_unstable();
        pairs.dedup();
        let flip: Vec<bool> = (0..n).map(|_| rng.next_u64() % 2 == 1).collect();
        let (mut ptrs, mut indices, mut z, mut signed) = (vec![0], vec![], vec![], vec![]);
        for i in 0..n {
            for &(_, j) in pairs.iter().filter(|&&(a, _)| a == i) {
                let w: T = log_uniform(&mut rng);
                indices.push(j as u32);
                z.push(-w);
                signed.push(if flip[i] != flip[j] { w } else { -w });
            }
            ptrs.push(indices.len());
        }
        let surplus: Vec<T> = (0..n)
            .map(|v| narrow(if v % 5 == 0 { 0.5 } else { 0.0 }))
            .collect();
        let adjacency =
            |values| UpperAdjacency::new(Rows::new(ptrs.clone(), indices.clone(), values, n));
        (
            Sdd::new(adjacency(z), Grounding::of(surplus.clone()), Layout::Like),
            Sdd::new(adjacency(signed), Grounding::of(surplus), Layout::Signed),
            flip,
        )
    }

    /// A grounded bipartite Z-block, `n_e` eliminated vertices each joined to two to five of `n_k` kept ones, log-uniform; its complement under `complement` and the complement of its `D`-switched copy, eliminated signed; and `σ` on the kept side.
    fn switched_complements<T: Real>(
        (n_e, n_k): (usize, usize),
        seed: u64,
        complement: SchurMode,
    ) -> (Sdd<T, u32>, Sdd<T, u32>, Vec<bool>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let flip: Vec<bool> = (0..n_e + n_k).map(|_| rng.next_u64() % 2 == 1).collect();
        let (mut ptrs, mut indices, mut z, mut signed) = (vec![0], vec![], vec![], vec![]);
        for e in 0..n_e {
            let extra = (0..rng.next_u64() % 4).map(|_| rng.next_u64() as usize % n_k);
            let mut row: Vec<usize> = [e % n_k, (e + 1) % n_k].into_iter().chain(extra).collect();
            row.sort_unstable();
            row.dedup();
            for k in row {
                let w: T = log_uniform(&mut rng);
                indices.push(k as u32);
                z.push(-w);
                signed.push(if flip[e] != flip[n_e + k] { w } else { -w });
            }
            ptrs.push(indices.len());
        }
        let surplus: Vec<T> = (0..n_e + n_k)
            .map(|v| narrow(if v % 7 == 0 { 0.5 } else { 0.0 }))
            .collect();
        let config = Config {
            exact_max_dim: 0,
            complement,
            frustrated: FrustratedRoute::Signed,
            ..Config::default()
        };
        let streams = Streams::of(0, &config);
        let complement = |values, balance| {
            let cross = Rows::new(ptrs.clone(), indices.clone(), values, n_k);
            let grounding = Grounding::of(surplus.clone());
            let block = BipartiteBlock::of(cross, balance, grounding);
            block.complement(&config, streams, ONE)
        };
        let sddm = complement(z, Balance::Balanced);
        let signed = complement(signed, Balance::Frustrated);
        let folds = sddm.layout() == Layout::Like && signed.layout() == Layout::Signed;
        assert!(
            folds,
            "a balanced complement folds, a frustrated one is signed"
        );
        (sddm, signed, flip[n_e..].to_vec())
    }

    /// Magnitudes, draws and order agree, so only each share's sign moves, by `σ_v σ_j`.
    fn eliminate_alike<T: Real>(
        (sddm, signed, flip): (Sdd<T, u32>, Sdd<T, u32>, Vec<bool>),
        seed: u64,
        copies: Option<Multi>,
    ) -> bool {
        let draws = Xoshiro256PlusPlus::seed_from_u64(seed);
        let (ground, new) = (signed.n() as u32, || CliqueTreeElimination::new(copies));
        let (z, z_pin) = new().eliminate(sddm, Unsigned, draws.clone(), None);
        let (s, s_pin) = new().eliminate(signed, Signed { ground }, draws, None);
        let bits = |c: T| widen(c).to_bits();
        let headers = |steps: &ApproximateCholesky<T, u32>| -> Vec<_> {
            (steps.headers.iter())
                .map(|h| (h.vertex, h.len, bits(h.inverse_degree)))
                .collect()
        };
        let mut end = 0;
        let mut flipped = Vec::new();
        for h in &z.headers {
            let v = h.vertex as usize;
            for k in end..end + h.len as usize {
                let c = z.coefficients[k];
                flipped.push(bits(if flip[v] != flip[z.neighbors[k] as usize] {
                    -c
                } else {
                    c
                }));
            }
            end += h.len as usize;
        }
        z_pin.is_none()
            && s_pin.is_none()
            && headers(&z) == headers(&s)
            && z.neighbors == s.neighbors
            && flipped == s.coefficients.iter().map(|&c| bits(c)).collect::<Vec<_>>()
    }

    fn switched_defects<T: Real>(seed: u64, copies: Option<Multi>) -> Option<String> {
        let alike = eliminate_alike(switched::<T>(80, seed), seed, copies);
        let copies = copies.map_or(1, |c| c.get());
        (!alike).then(|| {
            format!(
                "seed {seed}, {copies} copies, {}",
                std::any::type_name::<T>()
            )
        })
    }

    fn complement_defects<T: Real>(
        seed: u64,
        (formed, complement): (&str, SchurMode),
        copies: Option<Multi>,
    ) -> Option<String> {
        let switched = switched_complements::<T>((90, 40), seed, complement);
        let alike = eliminate_alike(switched, seed, copies);
        let copies = copies.map_or(1, |c| c.get());
        let label = format!("seed {seed}, {formed} complement, {copies} copies");
        (!alike).then(|| format!("{label}, {}", std::any::type_name::<T>()))
    }

    #[test]
    fn a_switched_z_matrix_eliminates_signed_as_its_seed_does_unsigned() {
        let ac2 = Multi::split(2).expect("AC2");
        let mut defects = Vec::new();
        for seed in 0..16 {
            for copies in [None, Some(ac2)] {
                defects.extend(switched_defects::<f64>(seed, copies));
                defects.extend(switched_defects::<f32>(seed, copies));
            }
        }
        assert!(defects.is_empty(), "{defects:#?}");
    }

    /// A bipartite Z-block's complement, exact or sampled, eliminates as its switched copy's signed complement does, each fill signed by its ends and nothing cancelled.
    #[test]
    fn a_switched_bipartite_z_block_forms_and_eliminates_its_complement_signed() {
        let sampled = |copies| SchurMode::Approximate {
            sampling: Sampling::CliqueTree {
                copies: std::num::NonZeroU32::new(copies).expect("a copy"),
            },
        };
        let ac2 = Multi::split(2).expect("AC2");
        let mut defects = Vec::new();
        for seed in 0..16 {
            for formed in [
                ("exact", SchurMode::Exact),
                ("AC", sampled(1)),
                ("AC2", sampled(2)),
            ] {
                for copies in [None, Some(ac2)] {
                    defects.extend(complement_defects::<f64>(seed, formed, copies));
                    defects.extend(complement_defects::<f32>(seed, formed, copies));
                }
            }
        }
        assert!(defects.is_empty(), "{defects:#?}");
    }
}
