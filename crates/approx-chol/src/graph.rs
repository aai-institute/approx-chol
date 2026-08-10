//! Elimination graph. Building one from CSR input lives in [`ingest`].

mod ingest;

pub(crate) use ingest::{BlockVertices, Ingestion};

use crate::types::{count_as_scalar, Real};

/// Carries the edge's multiplicity storage, so the AC path has no field to fill in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neighbor<T, C> {
    pub to: u32,
    /// Accumulated weight the neighbor's copies carry between them.
    pub fill_weight: T,
    pub count: C,
}

struct BitVec {
    words: Vec<u64>,
}

impl BitVec {
    fn new(n: usize) -> Self {
        Self {
            words: vec![0u64; n.div_ceil(64)],
        }
    }

    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline]
    fn get(&self, i: usize) -> bool {
        self.words[i >> 6] & (1u64 << (i & 63)) != 0
    }
}

/// AC is AC2 at one copy per edge, so the whole difference between them is this
/// trait. `Single` is a ZST, so an AC edge carries no count at all.
pub(crate) trait EdgeCount: Clone + Copy {
    /// `Single` cannot name a `k`, which makes an AC factorization over split
    /// multi-edges a type error rather than a mistake to avoid.
    type Split: Copy;

    /// Known statically, which lets the single-copy path sort on weights instead of
    /// quotients that are all division by one.
    const SINGLE_COPY: bool;

    fn one() -> Self;
    fn get(&self) -> u32;

    /// The identity for `Single`, so the AC path performs no division rather than
    /// dividing by one.
    fn per_copy<T: Real>(&self, total: T) -> T;

    /// `Single` keeps one, so its discard count is the duplicates the merge
    /// collapsed. The cap is the split itself, which is why `Single` has none to name.
    fn cap(copies: u32, limit: Self::Split) -> (Self, u32);

    /// The degree-bucket scale and the per-neighbor sample count are one number because
    /// they are one return value; the merge cap is [`Self::Split`] instead.
    fn split_edges<T: Real>(graph: &mut AdjListGraph<Self, T>, split: Self::Split) -> u32;
}

/// AC: every edge is a single edge, so there is nothing to store.
#[derive(Clone, Copy)]
pub(crate) struct Single;

/// This edge's virtual copy count, only ever lowered from the [`SplitFactor`] by the
/// merge cap.
#[derive(Clone, Copy)]
pub(crate) struct Multi(u32);

impl Multi {
    /// Only a test needs this: elimination makes a `Multi` from the validated split
    /// or from [`EdgeCount::cap`], never from a bare number.
    #[cfg(test)]
    pub(crate) fn new(count: u32) -> Self {
        Self(count)
    }
}

impl From<SplitFactor> for Multi {
    #[inline]
    fn from(split: SplitFactor) -> Self {
        Self(split.get())
    }
}

/// Distinct from the per-edge count [`Multi`] carries. Only the factors AC2 is
/// defined for exist, so `1/k` is never infinite and no split is a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SplitFactor(u32);

impl SplitFactor {
    /// `None` when `k` selects standard AC instead.
    pub(crate) fn new(k: u32) -> Option<Self> {
        (k >= 2).then_some(Self(k))
    }

    pub(crate) fn get(self) -> u32 {
        self.0
    }
}

impl EdgeCount for Single {
    type Split = ();
    const SINGLE_COPY: bool = true;

    #[inline]
    fn one() -> Self {
        Self
    }
    #[inline]
    fn get(&self) -> u32 {
        1
    }
    #[inline]
    fn per_copy<T: Real>(&self, total: T) -> T {
        total
    }
    #[inline]
    fn cap(copies: u32, _limit: ()) -> (Self, u32) {
        (Self, copies - 1)
    }
    /// A slim edge has nowhere to put a multiplicity.
    #[inline]
    fn split_edges<T: Real>(_graph: &mut AdjListGraph<Self, T>, _split: ()) -> u32 {
        1
    }
}

impl EdgeCount for Multi {
    type Split = SplitFactor;
    const SINGLE_COPY: bool = false;

    #[inline]
    fn one() -> Self {
        Self(1)
    }
    #[inline]
    fn get(&self) -> u32 {
        self.0
    }
    #[inline]
    fn per_copy<T: Real>(&self, total: T) -> T {
        total / count_as_scalar::<T, _>(self.0)
    }
    #[inline]
    fn cap(copies: u32, limit: SplitFactor) -> (Self, u32) {
        let kept = copies.min(limit.get());
        (Self(kept), copies - kept)
    }
    #[inline]
    fn split_edges<T: Real>(graph: &mut AdjListGraph<Self, T>, split: SplitFactor) -> u32 {
        graph.mark_split_edges(split);
        split.get()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Edge<T: Real, C> {
    weight: T,
    to: u32,
    /// Index of this edge's mirror in `adj[to]`; whatever moves an edge preserves it.
    rev: u32,
    count: C,
}

impl<T: Real, C: EdgeCount> Edge<T, C> {
    #[inline]
    fn new(weight: T, to: u32, rev: u32) -> Self {
        Self {
            weight,
            to,
            rev,
            count: C::one(),
        }
    }

    /// Splitting sets the count and leaves the weight alone.
    #[inline]
    fn fill_weight(&self) -> T {
        self.weight
    }
}

/// Adjacency-list elimination graph, generic over edge multiplicity storage.
pub(crate) struct AdjListGraph<C, T: Real> {
    /// Per-vertex adjacency list.
    adj: Vec<Vec<Edge<T, C>>>,
    /// `eliminated[v]` is `true` after `eliminate_vertex(v)` has been called.
    eliminated: BitVec,
}

/// AC2 path: edges with virtual multi-edge counts.
pub(crate) type MultiEdgeGraph<T> = AdjListGraph<Multi, T>;

/// Tiny lists keep their capacity; larger ones are released rather than retained
/// across eliminations.
const RETAIN_ADJ_CAPACITY_MAX: usize = 64;

impl<C: EdgeCount, T: Real> AdjListGraph<C, T> {
    fn from_adjacency(adj: Vec<Vec<Edge<T, C>>>) -> Self {
        Self {
            eliminated: BitVec::new(adj.len()),
            adj,
        }
    }

    /// Number of vertices (fixed at construction time).
    pub(crate) fn n(&self) -> usize {
        self.adj.len()
    }

    /// Current degree of vertex `v` (sum of multi-edge counts; includes stale entries).
    pub(crate) fn degree(&self, v: usize) -> usize {
        self.adj[v].iter().map(|e| e.count.get() as usize).sum()
    }

    /// Collect live (non-eliminated, positive-weight) neighbors of `v` into `scratch`.
    pub(crate) fn live_neighbors(&self, v: usize, scratch: &mut Vec<Neighbor<T, C>>) {
        scratch.clear();
        scratch.extend(self.adj[v].iter().filter_map(|e| {
            // Positive predicate: a NaN weight is dead (`!(w > 0)` differs from
            // `w <= 0` at NaN). if/else (not `bool::then`) keeps `fill_weight()`
            // lazy for dead edges and avoids `clippy::filter_map_bool_then`.
            if e.weight > T::zero() && !self.eliminated.get(e.to as usize) {
                Some(Neighbor {
                    to: e.to,
                    fill_weight: e.fill_weight(),
                    count: e.count,
                })
            } else {
                None
            }
        }));
    }

    /// Mark `v` as eliminated and release its adjacency storage.
    pub(crate) fn eliminate_vertex(&mut self, v: usize) {
        self.eliminated.set(v);
        while let Some(edge) = self.adj[v].pop() {
            let u = edge.to as usize;
            if self.eliminated.get(u) {
                continue;
            }
            debug_assert!(
                (edge.rev as usize) < self.adj[u].len(),
                "reverse pointer out of bounds: rev={} but adj[{}].len()={}",
                edge.rev,
                u,
                self.adj[u].len()
            );
            remove_edge_at(&mut self.adj, u, edge.rev as usize);
        }
        if self.adj[v].capacity() > RETAIN_ADJ_CAPACITY_MAX {
            self.adj[v] = Vec::new();
        }
    }

    /// Insert a symmetric fill edge between `u` and `v` with the given weight.
    pub(crate) fn add_fill_edge(&mut self, u: u32, v: u32, weight: T) {
        if u == v {
            return;
        }
        add_edge_pair(&mut self.adj, u as usize, v as usize, weight);
    }
}

impl<T: Real> MultiEdgeGraph<T> {
    /// The weight stays the total across the copies, so this cannot underflow one
    /// away; [`EdgeCount::per_copy`] divides where a single copy is wanted.
    pub(crate) fn mark_split_edges(&mut self, k: SplitFactor) {
        for adj_list in &mut self.adj {
            for edge in adj_list.iter_mut() {
                edge.count = k.into();
            }
        }
    }
}

#[inline]
fn add_edge_pair<T: Real, C: EdgeCount>(
    adj: &mut [Vec<Edge<T, C>>],
    u: usize,
    v: usize,
    weight: T,
) {
    // u32 reverse pointers; overflow is unreachable for tractable inputs,
    // so assert (release too) rather than truncate and corrupt removal.
    assert!(
        adj[u].len() < u32::MAX as usize && adj[v].len() < u32::MAX as usize,
        "adjacency list exceeds u32 edge capacity"
    );
    let rev_u = adj[v].len() as u32;
    let rev_v = adj[u].len() as u32;
    adj[u].push(Edge::new(weight, v as u32, rev_u));
    adj[v].push(Edge::new(weight, u as u32, rev_v));
}

/// Swap-remove, repairing the moved edge's reverse pointer.
fn remove_edge_at<T: Real, C: EdgeCount>(adj: &mut [Vec<Edge<T, C>>], u: usize, idx: usize) {
    let last_idx = adj[u].len() - 1;
    adj[u].swap_remove(idx);
    if idx < last_idx {
        let moved = adj[u][idx];
        let w = moved.to as usize;
        let rev = moved.rev as usize;
        adj[w][rev].rev = idx as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AC edge must not pay for the multiplicity it does not store: `Single`
    /// is a ZST, so both layouts are what the two hand-written structs were.
    #[test]
    fn edge_layout_is_unchanged_by_the_shared_definition() {
        assert_eq!(
            size_of::<Edge<f64, Single>>(),
            size_of::<f64>() + 2 * size_of::<u32>()
        );
        assert_eq!(
            size_of::<Edge<f64, Multi>>(),
            size_of::<Edge<f64, Single>>() + size_of::<f64>(),
            "one u32 plus its alignment padding"
        );
        assert_eq!(size_of::<Single>(), 0);
    }

    /// The predicate is positive on purpose: a zero weight carries no coupling and a
    /// NaN one is not evidence of any, so both are dead even though the neighbor is
    /// live. Reading either as a live neighbor puts a phantom edge in the star.
    #[test]
    fn only_positively_weighted_edges_are_live() {
        let graph = MultiEdgeGraph::<f64>::from_adjacency(vec![
            vec![
                Edge::new(2.0, 1, 0),
                Edge::new(0.0, 2, 0),
                Edge::new(f64::NAN, 3, 0),
            ],
            vec![Edge::new(2.0, 0, 0)],
            vec![Edge::new(0.0, 0, 1)],
            vec![Edge::new(f64::NAN, 0, 2)],
        ]);

        let mut neighbors = Vec::new();
        graph.live_neighbors(0, &mut neighbors);

        let live: Vec<u32> = neighbors.iter().map(|n| n.to).collect();
        assert_eq!(
            live,
            vec![1],
            "only the positively weighted neighbor is live"
        );
    }

    /// A cap that drifted from [`MultiEdgeGraph::mark_split_edges`] would bound every
    /// star at a multiplicity the graph does not carry, unnoticed elsewhere.
    #[test]
    fn the_reported_cap_is_the_count_written_on_the_edges() {
        let k = SplitFactor::new(3).expect("3 splits");
        let mut graph = MultiEdgeGraph::<f64>::from_adjacency(vec![
            vec![Edge::new(1.0, 1, 0)],
            vec![Edge::new(1.0, 0, 0)],
        ]);

        let cap = Multi::split_edges(&mut graph, k);

        let mut neighbors = Vec::new();
        graph.live_neighbors(0, &mut neighbors);
        assert_eq!(cap, k.get(), "the cap is the configured split");
        assert_eq!(
            neighbors[0].count.get(),
            cap,
            "the cap is the count the edges carry"
        );
    }
}
