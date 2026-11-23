//! Approximate Cholesky of a whole SDDM: clique-tree elimination in minimum-degree order (GKS 2023, Algorithm 8, stars by Algorithm 6); a block above [`PHASED_FROM`] vertices may first pivot its local minima, in index order.

mod graph;
mod ordering;

use rand::rngs::Xoshiro256PlusPlus;

use crate::factor::clique_tree::{absorb, CliqueTreeSampler, Copies, Fill, Multi, Pair, Single};
use crate::factor::sdd::Sdd;
use crate::factor::sign::Sign;
#[cfg(feature = "serde")]
use crate::factor::within_one;
use crate::sgdd::FactorTail;
use crate::{narrow, Batch, Index, Real};
use graph::{EliminationGraph, FillSearch};
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

    /// Every vertex of `M`, each star replaced by a sampled clique tree, but a floating `M`'s last: returned as the slot the solve zeroes. Above [`PHASED_FROM`] vertices, [`LocalMinima`] may pivot first in index order; the rest go in minimum-degree order.
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
        // Every vertex left but one: the never-pivoted one, or the order's last.
        let pivots = first.as_ref().map_or(n, |first| first.left) - 1;
        let (mut steps, last) = match first {
            None => {
                self.graph.load(&m, &room, self.copies);
                drop((m, room));
                let steps = ApproximateCholesky::with_capacity(n, 0);
                self.pivot_rest(steps, (never, n), sign, &mut draws, slots, pivots)
            }
            Some(first) => {
                let steps = self.pivot_first((m, room), &first, sign, &mut draws, slots);
                // The graph left numbers the vertices left by `first`'s ids, which keep their order, so the ground stays past the slots.
                let from = steps.ends();
                let vertices: Vec<I> = (0..n)
                    .filter(|&v| !first.gone(v))
                    .map(I::from_index)
                    .collect();
                let id = |v: usize| first.id(I::from_index(v));
                let (never, sign) = (
                    never.map(id),
                    sign.renumbered(|g| I::from_index(id(g.index()))),
                );
                let slots = slots - (n - first.left);
                drop(first);
                let (mut steps, last) =
                    self.pivot_rest(steps, (never, n), sign, &mut draws, slots, pivots);
                steps.renumber(from, |v| vertices[v.index()]);
                (steps, last.map(|v| vertices[v].index()))
            }
        };
        // Doubling past the estimate otherwise retains up to 2x the entries (+30% on grids).
        steps.neighbors.shrink_to_fit();
        steps.coefficients.shrink_to_fit();
        let pinned =
            (ground.is_none()).then(|| gauge.or(last).expect("a floating block leaves one vertex"));
        if n > REUSED_UP_TO {
            *self = Self::new(self.copies);
        }
        (steps, pinned)
    }

    /// The graph's `pivots` in minimum-degree order but `never`, keys scaled for `scale_n` vertices; returns the steps and, unless `never` stays, the vertex left.
    fn pivot_rest<S: Sign<T, I>>(
        &mut self,
        mut steps: ApproximateCholesky<T, I>,
        (never, scale_n): (Option<usize>, usize),
        sign: S,
        draws: &mut Xoshiro256PlusPlus,
        slots: usize,
        pivots: usize,
    ) -> (ApproximateCholesky<T, I>, Option<usize>) {
        let edge_ends = self.graph.edge_ends();
        let queue = |v| Some(v) != never;
        self.order
            .load(&self.graph, self.copies, queue, edge_ends, scale_n);
        steps.neighbors.reserve_exact(edge_ends);
        steps.coefficients.reserve_exact(edge_ends);
        self.pivot_all(draws, &mut steps, slots, sign, pivots);
        let last = never.is_none().then(|| self.order.next_vertex());
        (steps, last)
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
        for remaining in (0..pivots).rev() {
            let v = order.next_vertex();
            fill.clear();
            let row = graph.neighbors(v);
            let entries = row.len() as u32;
            let column = clique.eliminate(draws, row, fill, sign);
            steps.push(v, column, slots);
            graph.eliminate_vertex(v);
            let mut search = FillSearch::new(entries, clique.neighbors().len(), remaining);
            for &edge in fill.iter() {
                graph.push_fill(edge, &mut search, *split, sign);
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
        let row = |v: usize| (room[v] as usize - 1) * usize::from(first.gone(v));
        // Row `v` is written from `start[v + 1]`, which then ends it where `start[v]` begins it.
        let mut start = vec![0usize; n + 1];
        for v in 1..n {
            start[v + 1] = start[v] + row(v - 1);
        }
        let entries = start[n] + row(n - 1);
        let (mut to, mut weights) = (vec![I::from_index(0); entries], vec![T::zero(); entries]);
        drop(room);
        let mut staged = Vec::new();
        m.grounded_edges(|k, j, w| {
            let (row, other) = match (first.gone(k.index()), first.gone(j.index())) {
                (true, _) => (k, j),
                (_, true) => (j, k),
                _ => return staged.push((k, j, split, w)),
            };
            let at = &mut start[row.index() + 1];
            (to[*at], weights[*at]) = (other, w);
            *at += 1;
        });
        drop(m);
        // The rows become the steps: each column is written over its own row, which it never outgrows.
        let mut steps = ApproximateCholesky::over_rows(n, to, weights);
        let (mut written, mut merged) = (0, staged.len());
        let Self { clique, fill, .. } = self;
        for v in (0..n).filter(|&v| first.gone(v)) {
            let row = (start[v]..start[v + 1])
                .map(|k| (steps.neighbors[k], split, steps.coefficients[k]));
            fill.clear();
            let d = clique.pivot(draws, row, fill, sign);
            steps.write(v, (d, clique.shares(d)), slots, |steps, neighbor, share| {
                (steps.neighbors[written], steps.coefficients[written]) = (neighbor, share);
                written += 1;
            });
            // Later minima still read the rows past it: `refill` never returns more entries than the row it reads.
            debug_assert!(written <= start[v + 1], "column {v} outgrows its row");
            staged.extend_from_slice(fill);
            // So past 2^16 pairs the staging stays within twice its merged size.
            if staged.len() >= (2 * merged).max(1 << 16) {
                merge_pairs(&mut staged, first, split, sign);
                merged = staged.len();
            }
        }
        steps.neighbors.truncate(written);
        steps.coefficients.truncate(written);
        drop(start);
        merge_pairs(&mut staged, first, split, sign);
        for pair in &mut staged {
            (pair.0, pair.1) = (
                I::from_index(first.id(pair.0)),
                I::from_index(first.id(pair.1)),
            );
        }
        self.graph.load_merged(&staged, first.left, split);
        steps
    }
}

/// Above it, the local minima may pivot first; a smaller block has too few to pay the staging.
const PHASED_FROM: usize = 1024;

/// The vertices that pivot first, each `(degree, id)` below every neighbor's but the never-pivoted one's: an independent set (Liu's multiple minimum degree), so each star is its input row.
struct LocalMinima {
    /// Each other vertex's id among those minimum degree eliminates, [`u32::MAX`] for one pivoted first.
    ids: Vec<u32>,
    /// The vertices minimum degree eliminates, the never-pivoted one counted.
    left: usize,
}

impl LocalMinima {
    /// The set, if its edges are a quarter of all and the vertices fit `u32`; each edge between slots marks its larger end unless its smaller is `never`; edges to the ground are not walked.
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

/// Each pair once, its duplicates absorbed into it as the star absorbs them; what a merge cancels grounds both its ends, absorbed into their ground pairs, and a pair cancelled to zero leaves.
fn merge_pairs<T: Real, I: Index, C: Copies, S: Sign<T, I>>(
    pairs: &mut Vec<Pair<T, I, C>>,
    first: &LocalMinima,
    split: C,
    sign: S,
) {
    let grounded = merge_once(pairs, first, split, sign);
    if !grounded.is_empty() {
        ground_into(pairs, &grounded, first, split, sign);
    }
}

/// The ground edges a merge owes absorbed into the merged `pairs` as merging them all again would: each into its end's ground pair in the order owed, or where an end has none, into a new pair closing its bucket. Every edge to the ground weighs above zero, a surplus or a magnitude, so nothing cancels.
fn ground_into<T: Real, I: Index, C: Copies, S: Sign<T, I>>(
    pairs: &mut Vec<Pair<T, I, C>>,
    grounded: &[Pair<T, I, C>],
    first: &LocalMinima,
    split: C,
    sign: S,
) {
    let (ground, before) = (grounded[0].1, pairs.len());
    let mut host = vec![usize::MAX; first.left];
    for (at, pair) in pairs.iter().enumerate().filter(|(_, p)| p.1 == ground) {
        host[first.id(pair.0)] = at;
    }
    for &edge in grounded {
        match host[first.id(edge.0)] {
            usize::MAX => {
                host[first.id(edge.0)] = pairs.len();
                pairs.push(edge);
            }
            at => {
                let held = &mut pairs[at];
                let cancelled = absorb(sign, split, (&mut held.2, &mut held.3), (edge.2, edge.3));
                // A second merge would ground what this cancels and drop a pair it zeroes.
                debug_assert!(cancelled == T::zero() && held.3 > T::zero());
            }
        }
    }
    // A merge leaves `pairs` in their `lo` buckets in id order, so a stable sort closes each bucket with its new pair.
    if pairs.len() > before {
        pairs.sort_by_key(|pair| first.id(pair.0));
    }
}

/// One merge, bucketed by `lo`'s id, then scattered on `hi`'s; returns the ground edges its cancellations owe.
fn merge_once<T: Real, I: Index, C: Copies, S: Sign<T, I>>(
    pairs: &mut Vec<Pair<T, I, C>>,
    first: &LocalMinima,
    split: C,
    sign: S,
) -> Vec<Pair<T, I, C>> {
    let left = first.left;
    // Bucket `l` is written from `end[l]`, which then ends it.
    let mut end = vec![0usize; left + 1];
    for pair in pairs.iter() {
        end[first.id(pair.0) + 1] += 1;
    }
    for l in 0..left {
        end[l + 1] += end[l];
    }
    let mut bucketed = pairs.clone();
    for &pair in pairs.iter() {
        let l = first.id(pair.0);
        bucketed[end[l]] = pair;
        end[l] += 1;
    }
    // One past each `hi`'s pair in the output; at most `from`, it is an earlier bucket's.
    let (mut seen, mut out, mut begin, mut grounded) = (vec![0usize; left], 0, 0, Vec::new());
    for &stop in &end[..left] {
        let from = out;
        for &pair in &bucketed[std::mem::replace(&mut begin, stop)..stop] {
            let hi = first.id(pair.1);
            if seen[hi] <= from {
                seen[hi] = out + 1;
                pairs[out] = pair;
                out += 1;
                continue;
            }
            let held = &mut pairs[seen[hi] - 1];
            let cancelled = absorb(sign, split, (&mut held.2, &mut held.3), (pair.2, pair.3));
            if let Some(ground) = sign.ground().filter(|_| cancelled != T::zero()) {
                grounded.extend([pair.0, pair.1].map(|end| (end, ground, split, cancelled)));
            }
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

    /// Headers for `n` steps over `neighbors` and `coefficients` as rows that each step reads before it writes.
    fn over_rows(n: usize, neighbors: Vec<I>, coefficients: Vec<T>) -> Self {
        Self {
            headers: Vec::with_capacity(n),
            neighbors,
            coefficients,
        }
    }

    /// Where the steps end: their headers, and their entries.
    fn ends(&self) -> (usize, usize) {
        (self.headers.len(), self.neighbors.len())
    }

    /// The steps past `from` renumbered by `vertex`, pivots and neighbors.
    fn renumber(&mut self, (headers, entries): (usize, usize), vertex: impl Fn(I) -> I) {
        for header in &mut self.headers[headers..] {
            header.vertex = vertex(header.vertex);
        }
        for neighbor in &mut self.neighbors[entries..] {
            *neighbor = vertex(*neighbor);
        }
    }

    /// The column's step, its entries appended.
    fn push(&mut self, vertex: usize, column: (f64, impl Iterator<Item = (I, f64)>), slots: usize) {
        self.write(vertex, column, slots, |steps, neighbor, share| {
            steps.neighbors.push(neighbor);
            steps.coefficients.push(share);
        });
    }

    /// The column's step, each entry stored by `put`: each share `w/d` and `D⁻¹` rounded once from `f64`, so no share is derived from the others and none cancels (#275, #302); only `M`'s `slots` take one, not the ground past them.
    fn write(
        &mut self,
        vertex: usize,
        (d, shares): (f64, impl Iterator<Item = (I, f64)>),
        slots: usize,
        mut put: impl FnMut(&mut Self, I, T),
    ) {
        let mut len = 0;
        for (neighbor, share) in shares.filter(|&(j, _)| j.index() < slots) {
            put(self, neighbor, narrow(share));
            len += 1;
        }
        // Finite, as fill and `S` never weigh below `MIN_POSITIVE`.
        self.headers.push(StepHeader {
            vertex: I::from_index(vertex),
            len: I::from_index(len),
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

    /// `people` joined to two or three of `firms`, plus random person–person edges, so ties and neighboring minima occur, weighed in `[1/2, 2)`, signed at random under [`Layout::Signed`]; grounded at every tenth person when `grounded`.
    fn panel(people: usize, firms: usize, grounded: bool, layout: Layout) -> Sdd<f64, u32> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let n = people + firms;
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for person in 0..people {
            let spells = 2 + rng.next_u64() as usize % 2;
            pairs.extend((0..spells).map(|_| (person, people + rng.next_u64() as usize % firms)));
        }
        for _ in 0..people / 4 {
            let (a, b) = (
                rng.next_u64() as usize % people,
                rng.next_u64() as usize % people,
            );
            pairs.extend((a != b).then_some((a.min(b), a.max(b))));
        }
        pairs.sort_unstable();
        pairs.dedup();
        let (mut ptrs, mut indices) = (vec![0], Vec::new());
        for i in 0..n {
            indices.extend((pairs.iter()).filter(|p| p.0 == i).map(|p| p.1 as u32));
            ptrs.push(indices.len());
        }
        let values = (indices.iter())
            .map(|_| {
                let w = 0.5 + rng.random::<f64>() * 1.5;
                match layout == Layout::Signed && rng.random() {
                    true => w,
                    false => -w,
                }
            })
            .collect();
        let surplus = (0..n)
            .map(|v| {
                if grounded && v < people && v % 10 == 0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let adjacency = UpperAdjacency::new(Rows::new(ptrs, indices, values, n));
        Sdd::new(adjacency, Grounding::of(surplus), layout)
    }

    /// The set holds no edge, each member's `(degree, id)` below its neighbors' but `never`'s, every vertex so placed is in it, and `never` is not.
    #[test]
    fn local_minima_are_an_independent_set_without_never() {
        for grounded in [false, true] {
            let m = panel(1500, 60, grounded, Layout::Like);
            let room = EliminationGraph::<f64, u32, Single>::room(&m);
            let mut adjacent = vec![Vec::new(); m.n()];
            for i in 0..m.n() {
                for j in m.neighbors_past(i) {
                    adjacent[i].push(j);
                    adjacent[j].push(i);
                }
            }
            let key = |v: usize| (room[v], v);
            let below_all = |v: usize, but: usize| {
                room[v] > 1 && (adjacent[v].iter()).all(|&j| j == but || key(v) < key(j))
            };
            // A local minimum whose neighbor sits below all but it: as `never`, it leaves that neighbor in the set.
            let blocking = (0..m.n())
                .find(|&v| below_all(v, v) && adjacent[v].iter().any(|&j| below_all(j, v)))
                .expect("a local minimum blocking a neighbor");
            for never in [m.ground(), Some(blocking)] {
                let first = LocalMinima::of(&m, &room, never).expect("people hold most edges");
                let label = format!("grounded {grounded}, never {never:?}");
                assert!(
                    never.is_none_or(|v| v >= m.n() || !first.gone(v)),
                    "{label}"
                );
                for v in 0..m.n() {
                    let below = (adjacent[v].iter()).all(|&j| Some(j) == never || key(v) < key(j));
                    let member = Some(v) != never && room[v] > 1 && below;
                    assert_eq!(first.gone(v), member, "{label}: vertex {v}");
                    if first.gone(v) {
                        assert!(adjacent[v].iter().all(|&j| !first.gone(j)), "{label}: {v}");
                    }
                }
                let left = (0..room.len()).filter(|&v| !first.gone(v)).count();
                assert_eq!(first.left, left, "{label}");
            }
        }
    }

    /// A phased block's steps and pin, hashed bit for bit.
    fn phased_bits<C: Copies>(m: Sdd<f64, u32>, copies: Option<Multi>) -> u64 {
        let room = EliminationGraph::<f64, u32, C>::room(&m);
        assert!(
            LocalMinima::of(&m, &room, m.ground()).is_some(),
            "the phase fires"
        );
        let (mut elimination, draws) = (
            CliqueTreeElimination::new(copies),
            Xoshiro256PlusPlus::seed_from_u64(7),
        );
        let ground = m.n() as u32;
        let (steps, pinned) = match m.layout() {
            Layout::Signed => elimination.eliminate(m, Signed { ground }, draws, None),
            _ => elimination.eliminate(m, Unsigned, draws, None),
        };
        let headers = (steps.headers.iter()).flat_map(|h| {
            [
                u64::from(h.vertex),
                u64::from(h.len),
                h.inverse_degree.to_bits(),
            ]
        });
        let words = (headers.chain(steps.neighbors.iter().map(|&j| u64::from(j))))
            .chain(steps.coefficients.iter().map(|c| c.to_bits()))
            .chain(pinned.map(|p| p as u64));
        words.fold(0xcbf2_9ce4_8422_2325, |h, w| {
            (h ^ w).wrapping_mul(0x100_0000_01b3)
        })
    }

    /// A block whose local minima pivot first keeps every bit of its factor, AC and AC2, floating, grounded and signed, as built before the phase's rows became its columns in place and the graph past it numbered only the vertices left (a8ec5cbb).
    #[test]
    fn a_phased_block_factors_as_it_did() {
        let mut have = Vec::new();
        for (grounded, layout) in [
            (false, Layout::Like),
            (true, Layout::Like),
            (true, Layout::Signed),
        ] {
            let m = || panel(1500, 60, grounded, layout);
            have.push(phased_bits::<Single>(m(), None));
            // At 64 copies the order's key budget binds, so its buckets scale with the vertices counted.
            for copies in [2, 64] {
                have.push(phased_bits::<Multi>(m(), Multi::split(copies)));
            }
        }
        let want: [u64; 9] = [
            0x387e_0f96_6cf8_4f25,
            0x8f70_43e5_a5cc_f19b,
            0x4a17_5e81_b969_5a12,
            0x5660_9c9d_92c3_16c1,
            0x6bb1_77de_9f92_2205,
            0x6852_3368_cb7a_4b97,
            0x77e5_2100_21e6_803e,
            0xd155_9957_b8ac_2617,
            0x7364_72c2_674b_0d16,
        ];
        assert_eq!(have, want, "{have:#x?}");
    }

    /// A tree on `n` vertices, `edges` weighed in `[1/2, 3/2)`, its surplus `1` at `grounded`; and `x ↦ M x`.
    fn tree(
        n: usize,
        edges: &[(usize, usize)],
        grounded: &[usize],
    ) -> (Sdd<f64, u32>, impl Fn(&[f64]) -> Vec<f64>) {
        let weight = |a: usize, b: usize| 0.5 + ((31 * a + 17 * b) % 100) as f64 / 100.0;
        let mut upper: Vec<_> = edges.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        upper.sort_unstable();
        let (mut ptrs, mut indices, mut values) = (vec![0], vec![], vec![]);
        for i in 0..n {
            for &(_, j) in upper.iter().filter(|e| e.0 == i) {
                indices.push(j as u32);
                values.push(-weight(i, j));
            }
            ptrs.push(indices.len());
        }
        let surplus: Vec<f64> = (0..n)
            .map(|v| f64::from(u8::from(grounded.contains(&v))))
            .collect();
        let adjacency = UpperAdjacency::new(Rows::new(ptrs, indices, values, n));
        let m = Sdd::new(adjacency, Grounding::of(surplus.clone()), Layout::Like);
        let apply = move |x: &[f64]| {
            let mut y: Vec<f64> = (0..n).map(|v| surplus[v] * x[v]).collect();
            for &(a, b) in &upper {
                let flow = weight(a, b) * (x[a] - x[b]);
                y[a] += flow;
                y[b] -= flow;
            }
            y
        };
        (m, apply)
    }

    /// A tree factors exactly, as no pivot meets three neighbors: a caterpillar's legs, all of a star but its hub, or none of a path pivot first, each later column mapped back to its vertex, with fill or without, so `M x` solves to `x`, less a floating tree's pinned value.
    #[test]
    fn late_columns_map_back_to_their_vertices() {
        let (spine, legs) = (150, 9);
        // Spine vertex `s` is `10 s`, its legs the nine after it, so the vertices left skip nine of ten.
        let caterpillar: Vec<_> = (0..spine)
            .flat_map(|s| {
                let at = s * (legs + 1);
                let next = (s + 1 < spine).then_some((at, at + legs + 1));
                (1..=legs).map(move |l| (at, at + l)).chain(next)
            })
            .collect();
        let hub = 700;
        let star: Vec<_> = (0..1500).filter(|&v| v != hub).map(|v| (v, hub)).collect();
        let path: Vec<_> = (1..1500).map(|v| (v - 1, v)).collect();
        // Each tree, where it is grounded, and how many vertices its phase leaves, the ground not counted.
        for (label, edges, ground_at, left) in [
            ("caterpillar", &caterpillar, [1, 2], Some(spine)),
            ("star", &star, [3, 1400], Some(1)),
            ("path", &path, [0, 0], None),
        ] {
            for grounded in [false, true] {
                let (n, label) = (edges.len() + 1, format!("{label}, grounded {grounded}"));
                let (m, apply) = tree(n, edges, if grounded { &ground_at } else { &[] });
                let room = EliminationGraph::<f64, u32, Single>::room(&m);
                let first = LocalMinima::of(&m, &room, m.ground()).map(|first| first.left);
                assert_eq!(first, left.map(|l| l + usize::from(grounded)), "{label}");
                let x: Vec<f64> = (0..n)
                    .map(|v| ((v * 7919) % 101) as f64 / 50.0 - 1.0)
                    .collect();
                let mut y = apply(&x);
                let draws = Xoshiro256PlusPlus::seed_from_u64(5);
                let (steps, pinned) =
                    CliqueTreeElimination::new(None).eliminate(m, Unsigned, draws, None);
                assert_eq!(pinned.is_none(), grounded, "{label}");
                steps.substitute(&mut y, pinned);
                let shift = pinned.map_or(0.0, |p| x[p]);
                let off = (0..n)
                    .map(|v| (y[v] - (x[v] - shift)).abs())
                    .fold(0.0, f64::max);
                assert!(off < 1e-9, "{label}: off by {off:e}");
            }
        }
    }

    /// Each pair once, its weights summed and its copies summed and capped.
    #[test]
    fn merging_sums_each_pairs_parallel_edges() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let (n, split) = (12u32, Multi::split(4).expect("AC2"));
        let mut pairs: Vec<Pair<f64, u32, Multi>> = (0..300)
            .map(|_| {
                let (a, b) = (rng.random_range(0..n), rng.random_range(0..n - 1));
                let b = b + u32::from(b >= a);
                let (w, copies) = (f64::from(rng.random_range(1..5u8)), rng.random_range(1..3));
                (a.min(b), a.max(b), split.cap(copies), w)
            })
            .collect();
        let mut want = std::collections::BTreeMap::<(u32, u32), (f64, u32)>::new();
        for &(a, b, copies, w) in &pairs {
            let held = want.entry((a, b)).or_default();
            *held = (held.0 + w, (held.1 + copies.get()).min(4));
        }
        merge_pairs(&mut pairs, &none_first(n), split, Unsigned);
        let have: std::collections::BTreeMap<_, _> = (pairs.iter())
            .map(|&(a, b, copies, w)| ((a, b), (w, copies.get())))
            .collect();
        assert_eq!(have.len(), pairs.len(), "a pair held twice");
        assert_eq!(have, want);
    }

    /// Opposite duplicates cancel onto the ground at both ends, merged with the ground edges held, and a pair cancelled to zero leaves.
    #[test]
    fn merging_grounds_what_a_signed_pair_cancels() {
        let sign = Signed { ground: 3u32 };
        let split = Multi::split(4).expect("AC2");
        let one = split.cap(1);
        let mut pairs: Vec<Pair<f64, u32, Multi>> = vec![
            (0, 1, one, 2.0),
            (0, 3, one, 1.0),
            (0, 1, one, -0.5),
            (1, 2, one, 1.0),
            (1, 2, one, -1.0),
        ];
        merge_pairs(&mut pairs, &none_first(4), split, sign);
        pairs.sort_by_key(|p| (p.0, p.1));
        let pairs: Vec<_> = pairs
            .iter()
            .map(|&(a, b, c, w)| (a, b, c.get(), w))
            .collect();
        // Owed ground edges carry the split, summed into a held ground pair and capped.
        assert_eq!(
            pairs,
            [
                (0, 1, 2, 1.5),
                (0, 3, 4, 2.0),
                (1, 3, 4, 3.0),
                (2, 3, 4, 2.0)
            ]
        );
    }

    /// Absorbing the owed ground edges leaves the pairs, their order and bits as merging every pair again did, with or without ground pairs held, under either copy count and with vertices pivoted first.
    #[test]
    fn owed_ground_edges_merge_as_a_second_merge_would() {
        let (n, ground) = (40u32, 40u32);
        // Every fifth vertex pivoted first, so ids skip it.
        let mut left = 0;
        let ids = (0..=n)
            .map(|v| match v % 5 == 2 {
                true => u32::MAX,
                false => {
                    left += 1;
                    left - 1
                }
            })
            .collect();
        let sparse = LocalMinima {
            ids,
            left: left as usize,
        };
        let dense = none_first(n + 1);
        for seed in 0..32 {
            let first = [&dense, &sparse][seed as usize % 2];
            let split = Multi::split(2).expect("AC2");
            again_alike(seed, first, split, (n, ground));
            again_alike(seed, first, Single, (n, ground));
        }
    }

    fn again_alike<C: Copies>(seed: u64, first: &LocalMinima, split: C, (n, ground): (u32, u32)) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let kept: Vec<u32> = (0..n).filter(|&v| !first.gone(v as usize)).collect();
        let mut pairs: Vec<Pair<f64, u32, C>> = (0..400)
            .map(|_| {
                let a = rng.random_range(0..kept.len());
                let b = rng.random_range(0..kept.len() - 1);
                let (a, b) = (kept[a], kept[b + usize::from(b >= a)]);
                // Tenths, so sums round and a merge grouped otherwise would move bits.
                let w = f64::from(rng.random_range(1..50u8))
                    * 0.1
                    * [-1.0, 1.0][rng.random_range(0..2)];
                (a.min(b), a.max(b), split, w)
            })
            .collect();
        let held = kept.iter().filter(|_| rng.random_range(0..3) == 0);
        pairs.extend(held.map(|&k| (k, ground, split, 0.3)));
        let sign = Signed { ground };
        let mut again = pairs.clone();
        loop {
            let grounded = merge_once(&mut again, first, split, sign);
            if grounded.is_empty() {
                break;
            }
            again.extend(grounded);
        }
        merge_pairs(&mut pairs, first, split, sign);
        let bits = |pairs: &[Pair<f64, u32, C>]| -> Vec<_> {
            (pairs.iter())
                .map(|&(a, b, copies, w)| (a, b, w.to_bits(), copies.get()))
                .collect()
        };
        assert_eq!(
            bits(&pairs),
            bits(&again),
            "seed {seed}, {} copies",
            split.get()
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
